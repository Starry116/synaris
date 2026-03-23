"""
@File       : milvus_client.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: Milvus 向量数据库异步客户端封装，提供连接管理、Collection 生命周期与索引操作。
@Features:
  - pymilvus 连接管理：懒初始化单例，连接参数从 settings 读取
  - ensure_collection_exists()：Collection 不存在时自动按 Schema 创建
  - Schema 字段：
      id           AUTO_ID INT64  主键，Milvus 自动生成
      content      VARCHAR(2048)  文本块原始内容
      embedding    FLOAT_VECTOR(1536)  OpenAI text-embedding-3-small 向量维度
      metadata     JSON           扩展元数据（页码/标题/文件类型等）
      source       VARCHAR(512)   文档来源标识（文件名/URL）
      created_at   INT64          Unix 时间戳（毫秒）
  - HNSW 索引（M=16, efConstruction=256）：
      M              — 每个节点在图中的最大双向连接数，越大召回率越高、内存占用越多
      efConstruction — 构建阶段候选集大小，越大索引质量越高、构建越慢
      内积度量（IP）  — 配合 L2 归一化向量使用，等价余弦相似度，查询速度更快
  - create_index() / drop_index() / get_index_info()
  - 健康检查 ping() → bool（不抛异常，适合 /health/detailed 调用）
  - get_milvus_client() 工厂函数，兼容 FastAPI Depends 依赖注入

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
    2026-03-23  Starry  Remove unused `time` import
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

from app.config.settings import get_settings
from app.core.exceptions import ErrorCode, VectorDBError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# 常量：Collection Schema 配置
# ---------------------------------------------------------------------------

# 向量维度（对应 OpenAI text-embedding-3-small 输出维度）
EMBEDDING_DIM: int = 1536

# HNSW 索引参数
# M:              每个节点在 HNSW 图中保留的最大双向链接数
#                 • 范围建议：4~64；越大图更稠密，召回率更高，但内存与构建时间增加
#                 • 16 是工程实践中质量与资源的均衡点
# efConstruction: 构建索引时每个节点的动态候选集大小
#                 • 范围建议：8~512；越大索引质量越高（更准确），但构建越慢
#                 • 256 在大规模文档库（百万级）下仍能保证良好召回率
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 256

# 搜索时的 ef 参数（搜索候选集大小，越大召回率越高但延迟增加）
# 此处作为索引参数默认值，查询时可按需覆盖
HNSW_EF_SEARCH: int = 64

# 度量类型：IP（内积）配合 L2 归一化向量 = 余弦相似度，比 L2 查询速度更快
METRIC_TYPE: str = "IP"

# Collection 名称
DEFAULT_COLLECTION_NAME: str = "documents"

# 连接别名（pymilvus 多连接管理标识符）
CONNECTION_ALIAS: str = "synaris_default"

# 线程池：pymilvus 为同步 SDK，所有 IO 操作通过线程池转为异步
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="milvus_")


# ---------------------------------------------------------------------------
# Schema 定义
# ---------------------------------------------------------------------------


def _build_collection_schema(description: str = "") -> CollectionSchema:
    """
    构建 Synaris 文档向量 Collection 的标准 Schema。

    字段说明：
      id          — INT64，主键，auto_id=True 由 Milvus 自动生成雪花 ID
                    不使用 UUID 字符串是因为 Milvus 主键类型对 INT64 有性能优化
      content     — VARCHAR(2048)，文本块原始内容，用于 RAG 回答时的引用溯源
      embedding   — FLOAT_VECTOR(1536)，OpenAI 嵌入向量，ANN 检索的核心字段
      metadata    — JSON，扩展元数据（如页码、标题、章节、文件类型），
                    Milvus 2.3+ 支持 JSON 字段过滤查询
      source      — VARCHAR(512)，文档来源标识（文件名或 URL），
                    支持按 source 删除整个文档的所有向量
      created_at  — INT64，入库时间戳（毫秒级 Unix 时间），
                    支持按时间范围过滤（标量过滤 + 向量搜索混合）
    """
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="主键，Milvus 自动生成雪花 ID",
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="文本块原始内容（chunk_size ≤ 512 字符，余量用于特殊字符）",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description=f"文本向量（OpenAI text-embedding-3-small，dim={EMBEDDING_DIM}）",
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="扩展元数据：{page, title, section, file_type, chunk_index, ...}",
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="文档来源标识，用于按文档删除全部向量",
        ),
        FieldSchema(
            name="created_at",
            dtype=DataType.INT64,
            description="入库时间戳（毫秒级 Unix 时间）",
        ),
    ]

    return CollectionSchema(
        fields=fields,
        description=description or "Synaris 文档向量存储",
        enable_dynamic_field=True,  # 允许插入 Schema 未定义的额外字段（向前兼容）
    )


# ---------------------------------------------------------------------------
# 内部工具：同步操作转异步
# ---------------------------------------------------------------------------


async def _run_sync(func, *args, **kwargs) -> Any:
    """
    将 pymilvus 同步阻塞调用通过线程池转为协程，避免阻塞 asyncio 事件循环。

    pymilvus 当前版本（2.x）为同步 SDK，官方异步版本（AsyncMilvusClient）
    在 2.4+ 开始逐步支持，但 API 尚不稳定。
    此封装保证在异步 FastAPI 环境中安全使用 pymilvus，同时为未来迁移预留接口。
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        partial(func, *args, **kwargs),
    )


# ---------------------------------------------------------------------------
# MilvusClient
# ---------------------------------------------------------------------------


class MilvusClient:
    """
    Synaris Milvus 客户端。

    生命周期：
      - 通过 get_milvus_client() 获取模块级单例
      - lifespan 启动时调用 connect() 建立连接
      - lifespan 关闭时调用 close() 释放连接

    线程安全：pymilvus connections 模块内部有连接管理锁，
    多协程并发调用 Collection 操作时是安全的。
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._alias: str = CONNECTION_ALIAS

    # ── 连接管理 ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        建立 Milvus 连接。
        若已连接则跳过（幂等）。

        连接参数从 settings 读取：
          MILVUS_HOST / MILVUS_PORT / MILVUS_USER / MILVUS_PASSWORD
        """
        if self._connected:
            return

        try:
            await _run_sync(
                connections.connect,
                alias=self._alias,
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=getattr(settings, "milvus_user", ""),
                password=getattr(settings, "milvus_password", ""),
                timeout=10,
            )
            self._connected = True
            logger.info(
                "Milvus 连接已建立",
                extra={
                    "host": settings.milvus_host,
                    "port": settings.milvus_port,
                    "alias": self._alias,
                },
            )
        except MilvusException as exc:
            self._connected = False
            logger.error(
                "Milvus 连接失败",
                extra={
                    "host": settings.milvus_host,
                    "port": settings.milvus_port,
                    "error": str(exc),
                },
            )
            raise VectorDBError(
                message=f"Milvus 连接失败（{settings.milvus_host}:{settings.milvus_port}）",
                error_code=ErrorCode.VECTOR_DB_CONNECTION,
                detail={"host": settings.milvus_host, "error": str(exc)},
            ) from exc

    async def close(self) -> None:
        """断开 Milvus 连接，释放资源。lifespan shutdown 阶段调用。"""
        if not self._connected:
            return
        try:
            await _run_sync(connections.disconnect, alias=self._alias)
            self._connected = False
            logger.info("Milvus 连接已释放", extra={"alias": self._alias})
        except MilvusException as exc:
            logger.warning("Milvus 断开连接异常", extra={"error": str(exc)})

    def _ensure_connected(self) -> None:
        """操作前检查连接状态，未连接时抛出 VectorDBError。"""
        if not self._connected:
            raise VectorDBError(
                message="Milvus 未连接，请先调用 connect()",
                error_code=ErrorCode.VECTOR_DB_CONNECTION,
            )

    # ── Collection 管理 ───────────────────────────────────────────────

    async def ensure_collection_exists(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
        description: str = "",
    ) -> Collection:
        """
        确保 Collection 存在，不存在则自动创建（含索引）。

        幂等操作：多次调用安全，不会重复创建。

        Args:
            name:        Collection 名称
            description: Collection 描述（仅创建时使用）

        Returns:
            pymilvus Collection 实例（已加载到内存）

        Raises:
            VectorDBError: 创建或加载失败
        """
        self._ensure_connected()

        try:
            exists = await _run_sync(utility.has_collection, name, using=self._alias)

            if exists:
                logger.debug("Collection 已存在，跳过创建", extra={"collection": name})
                collection = await _run_sync(Collection, name, using=self._alias)
            else:
                logger.info("Collection 不存在，开始创建", extra={"collection": name})
                schema = _build_collection_schema(description)
                collection = await _run_sync(
                    Collection,
                    name=name,
                    schema=schema,
                    using=self._alias,
                )
                logger.info("Collection 创建成功", extra={"collection": name})

                # 新建 Collection 立即创建索引
                await self.create_index(name)

            # 加载 Collection 到内存（查询前必须 load）
            await self._load_collection(collection)
            return collection

        except VectorDBError:
            raise
        except MilvusException as exc:
            logger.error(
                "Collection 创建/加载失败",
                extra={"collection": name, "error": str(exc)},
            )
            raise VectorDBError(
                message=f"Collection 操作失败（{name}）",
                error_code=ErrorCode.VECTOR_DB_SCHEMA_ERROR,
                detail={"collection": name, "error": str(exc)},
            ) from exc

    async def get_collection(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
    ) -> Collection:
        """
        获取已存在的 Collection 实例。

        与 ensure_collection_exists 的区别：
          - 此方法不会自动创建，Collection 不存在时抛出 VectorDBError
          - 适合读取操作（search / query），避免意外创建空 Collection

        Raises:
            VectorDBError: Collection 不存在
        """
        self._ensure_connected()

        try:
            exists = await _run_sync(utility.has_collection, name, using=self._alias)
            if not exists:
                raise VectorDBError(
                    message=f"Collection 不存在：{name}",
                    error_code=ErrorCode.VECTOR_DB_QUERY_FAILED,
                    detail={"collection": name},
                )
            collection = await _run_sync(Collection, name, using=self._alias)
            await self._load_collection(collection)
            return collection

        except VectorDBError:
            raise
        except MilvusException as exc:
            raise VectorDBError(
                message=f"获取 Collection 失败（{name}）",
                error_code=ErrorCode.VECTOR_DB_QUERY_FAILED,
                detail={"collection": name, "error": str(exc)},
            ) from exc

    async def drop_collection(self, name: str) -> None:
        """
        删除整个 Collection（含所有向量和索引）。
        危险操作，需明确传入 Collection 名称（无默认值保护）。

        Raises:
            VectorDBError: 删除失败
        """
        self._ensure_connected()
        try:
            exists = await _run_sync(utility.has_collection, name, using=self._alias)
            if not exists:
                logger.warning(
                    "Collection 不存在，跳过删除", extra={"collection": name}
                )
                return

            await _run_sync(utility.drop_collection, name, using=self._alias)
            logger.info("Collection 已删除", extra={"collection": name})

        except VectorDBError:
            raise
        except MilvusException as exc:
            raise VectorDBError(
                message=f"删除 Collection 失败（{name}）",
                error_code=ErrorCode.VECTOR_DB_SCHEMA_ERROR,
                detail={"collection": name, "error": str(exc)},
            ) from exc

    async def list_collections(self) -> List[str]:
        """返回当前 Milvus 实例中所有 Collection 名称列表。"""
        self._ensure_connected()
        try:
            return await _run_sync(utility.list_collections, using=self._alias)
        except MilvusException as exc:
            raise VectorDBError(
                message="获取 Collection 列表失败",
                error_code=ErrorCode.VECTOR_DB_QUERY_FAILED,
                detail={"error": str(exc)},
            ) from exc

    async def get_collection_stats(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
    ) -> Dict[str, Any]:
        """
        获取 Collection 统计信息：向量总数、内存占用等。
        用于监控面板和健康检查。
        """
        self._ensure_connected()
        try:
            collection = await _run_sync(Collection, name, using=self._alias)
            # get_stats 返回 {"row_count": N} 等字段
            stats = await _run_sync(collection.get_stats)
            return {
                "collection": name,
                "row_count": int(stats.get("row_count", 0)),
            }
        except MilvusException as exc:
            raise VectorDBError(
                message=f"获取 Collection 统计失败（{name}）",
                error_code=ErrorCode.VECTOR_DB_QUERY_FAILED,
                detail={"collection": name, "error": str(exc)},
            ) from exc

    # ── 索引管理 ──────────────────────────────────────────────────────

    async def create_index(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        field_name: str = "embedding",
        *,
        m: int = HNSW_M,
        ef_construction: int = HNSW_EF_CONSTRUCTION,
    ) -> None:
        """
        在指定字段上创建 HNSW 向量索引。

        HNSW（Hierarchical Navigable Small World）参数详解：
        ┌─────────────────┬──────────────────────────────────────────────────┐
        │ 参数            │ 说明                                              │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ M               │ 每个节点在多层图中的最大双向连接数（邻居数）         │
        │                 │ • 增大 → 召回率↑，内存↑，构建时间↑                 │
        │                 │ • 推荐范围：4~64；16 适合通用文本检索场景           │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ efConstruction  │ 构建阶段的动态候选集大小（贪心搜索的探索深度）       │
        │                 │ • 增大 → 索引质量↑，构建时间↑（查询速度不受影响）   │
        │                 │ • 推荐范围：8~512；256 在百万级文档库下质量良好     │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ metric_type     │ 向量相似度度量方式                                 │
        │                 │ • IP（内积）：配合 L2 归一化向量 = 余弦相似度       │
        │                 │   查询速度比 L2 快约 15-20%（无需计算平方和）       │
        │                 │ • L2（欧氏距离）：适合未归一化向量                  │
        └─────────────────┴──────────────────────────────────────────────────┘

        Args:
            collection_name: 目标 Collection 名称
            field_name:      向量字段名（默认 "embedding"）
            m:               HNSW M 参数
            ef_construction: HNSW efConstruction 参数

        Raises:
            VectorDBError: 索引创建失败
        """
        self._ensure_connected()

        index_params = {
            "index_type": "HNSW",
            "metric_type": METRIC_TYPE,
            "params": {
                "M": m,
                "efConstruction": ef_construction,
            },
        }

        try:
            collection = await _run_sync(Collection, collection_name, using=self._alias)
            await _run_sync(
                collection.create_index,
                field_name=field_name,
                index_params=index_params,
                index_name=f"idx_{field_name}_hnsw",
            )
            logger.info(
                "HNSW 索引创建成功",
                extra={
                    "collection": collection_name,
                    "field": field_name,
                    "M": m,
                    "efConstruction": ef_construction,
                    "metric_type": METRIC_TYPE,
                },
            )
        except MilvusException as exc:
            logger.error(
                "HNSW 索引创建失败",
                extra={"collection": collection_name, "error": str(exc)},
            )
            raise VectorDBError(
                message=f"创建 HNSW 索引失败（{collection_name}.{field_name}）",
                error_code=ErrorCode.VECTOR_DB_SCHEMA_ERROR,
                detail={
                    "collection": collection_name,
                    "field": field_name,
                    "error": str(exc),
                },
            ) from exc

    async def drop_index(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        field_name: str = "embedding",
    ) -> None:
        """
        删除指定字段的向量索引。

        注意：删除索引后 Collection 仍可暴力扫描查询，但性能会大幅下降。
        通常在需要重建索引（修改参数）时先 drop 再 create。

        Raises:
            VectorDBError: 索引删除失败
        """
        self._ensure_connected()
        try:
            collection = await _run_sync(Collection, collection_name, using=self._alias)
            await _run_sync(
                collection.drop_index,
                index_name=f"idx_{field_name}_hnsw",
            )
            logger.info(
                "索引已删除",
                extra={"collection": collection_name, "field": field_name},
            )
        except MilvusException as exc:
            raise VectorDBError(
                message=f"删除索引失败（{collection_name}.{field_name}）",
                error_code=ErrorCode.VECTOR_DB_SCHEMA_ERROR,
                detail={"collection": collection_name, "error": str(exc)},
            ) from exc

    async def get_index_info(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        field_name: str = "embedding",
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定字段的索引信息。
        索引不存在时返回 None（不抛异常）。
        """
        self._ensure_connected()
        try:
            collection = await _run_sync(Collection, collection_name, using=self._alias)
            indexes = await _run_sync(collection.indexes)
            for idx in indexes:
                if idx.field_name == field_name:
                    return {
                        "index_name": idx.index_name,
                        "field_name": idx.field_name,
                        "index_type": idx.params.get("index_type"),
                        "metric_type": idx.params.get("metric_type"),
                        "params": idx.params.get("params", {}),
                    }
            return None
        except MilvusException as exc:
            logger.warning(
                "获取索引信息失败",
                extra={"collection": collection_name, "error": str(exc)},
            )
            return None

    async def rebuild_index(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        field_name: str = "embedding",
        *,
        m: int = HNSW_M,
        ef_construction: int = HNSW_EF_CONSTRUCTION,
    ) -> None:
        """
        重建向量索引（先 drop 再 create）。
        用于修改索引参数（如调整 M / efConstruction）。
        """
        logger.info(
            "开始重建索引",
            extra={"collection": collection_name, "field": field_name},
        )
        await self.drop_index(collection_name, field_name)
        await self.create_index(
            collection_name, field_name, m=m, ef_construction=ef_construction
        )
        logger.info("索引重建完成", extra={"collection": collection_name})

    # ── 内部辅助 ──────────────────────────────────────────────────────

    async def _load_collection(self, collection: Collection) -> None:
        """
        将 Collection 加载到 Milvus 内存，未加载时查询会报错。
        若已加载则跳过（幂等）。

        load_state 说明：
          Loaded   → 已在内存，可直接查询
          Loading  → 加载中，等待完成
          NotLoad  → 未加载，需调用 load()
        """
        try:
            load_state = await _run_sync(
                utility.load_state,
                collection.name,
                using=self._alias,
            )
            # LoadState.Loaded 枚举值在 pymilvus 2.x 中为字符串 "Loaded"
            state_str = str(load_state)
            if "Loaded" in state_str:
                logger.debug(
                    "Collection 已在内存，跳过加载",
                    extra={"collection": collection.name},
                )
                return

            logger.info(
                "加载 Collection 到内存",
                extra={"collection": collection.name},
            )
            await _run_sync(collection.load)

        except MilvusException as exc:
            # 加载失败不直接抛出，降级为警告（某些操作如 insert 不需要预加载）
            logger.warning(
                "Collection 加载到内存失败",
                extra={"collection": collection.name, "error": str(exc)},
            )

    # ── 健康检查 ──────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """
        检查 Milvus 连接是否正常。

        探测方式：调用 utility.list_collections()，
        成功返回（即使列表为空）即认为连接健康。

        Returns:
            True  — 连接正常
            False — 连接异常（不抛异常，适合 /health/detailed 调用）
        """
        if not self._connected:
            # 尝试自动重连一次
            try:
                await self.connect()
            except VectorDBError:
                return False

        try:
            await _run_sync(utility.list_collections, using=self._alias)
            return True
        except Exception as exc:
            logger.warning("Milvus PING 失败", extra={"error": str(exc)})
            self._connected = False  # 标记连接已断开，下次操作触发重连
            return False


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """
    获取模块级 MilvusClient 单例。

    FastAPI Depends 兼容，也可直接调用。

    生命周期：
      - 首次调用时创建实例（不建立连接，仅初始化对象）
      - 连接在 lifespan 的 connect() 中建立
      - 连接在 lifespan 的 close() 中释放

    用法（FastAPI Depends）：
        @router.get("/example")
        async def example(
            milvus: MilvusClient = Depends(get_milvus_client),
        ):
            collection = await milvus.get_collection()

    用法（直接调用）：
        client = get_milvus_client()
        await client.ensure_collection_exists()
    """
    global _client
    if _client is None:
        _client = MilvusClient()
        logger.debug("MilvusClient 实例已创建")
    return _client


"""

## 设计说明

### Schema 字段选型
```
字段          类型                  设计理由
─────────────────────────────────────────────────────────────────
id            INT64 auto_id         Milvus 对 INT64 主键有雪花 ID 优化，比 VARCHAR UUID 检索更快
content       VARCHAR(2048)         chunk_size=512 字符，2048 给多字节字符和边界情况留出 4× 余量
embedding     FLOAT_VECTOR(1536)    对应 text-embedding-3-small 输出维度，升级至 text-embedding-3-large(3072)时需重建 Schema
metadata      JSON                  Milvus 2.3+ JSON 字段支持键值过滤，存储页码/标题/章节等动态属性
source        VARCHAR(512)          按文档删除所有向量时使用expr='source == "report.pdf"'
created_at    INT64(毫秒时间戳)      支持时间范围过滤混合检索expr='created_at > 1700000000000'
```

### HNSW 参数选型对比
```
场景              M     efConstruction   召回率    内存    构建时间
────────────────────────────────────────────────────────────────
轻量/低成本       8      100             ~0.90    低      快
推荐（当前）     16      256             ~0.97    中      中
高精度           32      512             ~0.99    高      慢
极限精度         64      512             ~0.999   极高    极慢
```

### 同步转异步的必要性

pymilvus 2.x 为同步 SDK，直接在 `async def` 中调用会阻塞整个 asyncio 事件循环。`_run_sync()` 通过 `loop.run_in_executor(ThreadPoolExecutor)` 将所有 pymilvus 调用推到独立线程池，保证主事件循环不被阻塞：
```
asyncio 事件循环（主线程）
    │
    └─ await _run_sync(collection.search, ...)
              │
              └─ ThreadPoolExecutor（4线程）
                    └─ pymilvus 同步网络 IO → Milvus gRPC

"""
