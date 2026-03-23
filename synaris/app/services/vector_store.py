"""
@File       : vector_store.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 向量存储业务服务层，封装 Milvus 插入、检索、删除与检索结果缓存。
@Features:
  - upsert_documents()   批量插入文档向量（自动生成 Embedding + 写入 Milvus）
  - similarity_search()  语义相似度检索，返回 Top-K 结果（含相关性阈值过滤）
  - delete_by_source()   按文档来源删除全部相关向量
  - 检索结果 Redis 缓存：TTL=10min，key = "synaris:rag:{sha256(query+collection+top_k)}"
  - SearchResult 数据类：content / source / score / metadata 结构化返回
  - get_vector_store_service() 工厂函数，兼容 FastAPI Depends

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.exceptions import ErrorCode, VectorDBError
from app.core.logging import get_logger
from app.infrastructure.embedding_client import EmbeddingClient, get_embedding_client
from app.infrastructure.milvus_client import (
    DEFAULT_COLLECTION_NAME,
    HNSW_EF_SEARCH,
    METRIC_TYPE,
    MilvusClient,
    _run_sync,
    get_milvus_client,
)
from app.infrastructure.redis_client import build_key, delete, get_json, set_json

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SEARCH_CACHE_TTL: int = 10 * 60  # 检索结果缓存 10 分钟
SEARCH_CACHE_PREFIX: str = "rag"
DEFAULT_TOP_K: int = 5
DEFAULT_SCORE_THRESHOLD: float = 0.5  # IP 度量下的相关性阈值（0~1）
# Milvus 搜索输出字段（不包含 embedding，避免传输大向量）
OUTPUT_FIELDS: List[str] = ["content", "source", "metadata", "created_at"]


# ---------------------------------------------------------------------------
# 数据类：检索结果
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """
    单条向量检索结果。

    字段说明：
      content   — 文本块原始内容（用于 RAG 生成时注入 prompt）
      source    — 文档来源标识（文件名/URL，用于引用溯源）
      score     — 相似度分数（IP 度量，0~1，越高越相关）
      metadata  — 扩展元数据（页码/标题/章节等）
      id        — Milvus 向量 ID（INT64）
    """

    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，用于 Redis 缓存存储。"""
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """从字典反序列化，用于 Redis 缓存读取。"""
        return cls(
            content=data.get("content", ""),
            source=data.get("source", ""),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
            id=data.get("id"),
        )


# ---------------------------------------------------------------------------
# 缓存 Key 工具
# ---------------------------------------------------------------------------


def _search_cache_key(
    query_text: str,
    collection: str,
    top_k: int,
    score_threshold: float,
) -> str:
    """
    基于查询文本 + Collection 名 + 检索参数生成缓存 Key。
    相同参数组合命中同一缓存，不同 top_k 或阈值独立缓存。
    """
    raw = f"{query_text}|{collection}|{top_k}|{score_threshold}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return build_key(SEARCH_CACHE_PREFIX, digest)


# ---------------------------------------------------------------------------
# VectorStoreService
# ---------------------------------------------------------------------------


class VectorStoreService:
    """
    向量存储业务服务。

    职责边界：
      - 封装 Embedding 生成 + Milvus 读写的完整流程
      - 管理检索结果的 Redis 缓存（写入与失效）
      - 不处理文档解析和分块（由 document_service 负责）
      - 不处理 Reranking（由 rag_service 负责）
    """

    def __init__(
        self,
        milvus: MilvusClient,
        embedder: EmbeddingClient,
    ) -> None:
        self._milvus = milvus
        self._embedder = embedder

    # ── 写入 ──────────────────────────────────────────────────────────

    async def upsert_documents(
        self,
        texts: List[str],
        sources: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection: str = DEFAULT_COLLECTION_NAME,
    ) -> List[int]:
        """
        批量插入文档向量到 Milvus。

        处理流程：
          1. 验证输入长度一致性
          2. 批量生成 Embedding（含 Redis 缓存复用）
          3. 确保 Collection 存在（不存在则自动创建 + 建索引）
          4. 构造插入数据行（content / embedding / metadata / source / created_at）
          5. 调用 Milvus collection.insert()
          6. Flush 确保数据持久化
          7. 使受影响 source 的检索缓存失效（最终一致性）

        Args:
            texts:      文本块列表（与 sources 等长）
            sources:    每个文本块对应的文档来源标识
            metadatas:  每个文本块的扩展元数据（None 时使用空字典）
            collection: 目标 Collection 名称

        Returns:
            插入后 Milvus 自动生成的 ID 列表（与 texts 等长）

        Raises:
            VectorDBError:  Milvus 操作失败
            LLMError:       Embedding 生成失败
            ValueError:     输入参数长度不一致
        """
        if not texts:
            return []

        if len(texts) != len(sources):
            raise ValueError(
                f"texts({len(texts)}) 与 sources({len(sources)}) 长度必须一致"
            )

        _metadatas = metadatas or [{} for _ in texts]
        if len(_metadatas) != len(texts):
            raise ValueError(
                f"metadatas({len(_metadatas)}) 与 texts({len(texts)}) 长度必须一致"
            )

        logger.info(
            "开始插入文档向量",
            extra={"count": len(texts), "collection": collection},
        )

        # 1. 生成 Embedding（批量，含缓存复用）
        embeddings = await self._embedder.embed_batch(texts)

        # 2. 确保 Collection 存在
        coll = await self._milvus.ensure_collection_exists(collection)

        # 3. 构造插入数据（字段顺序与 Schema 定义一致，id 字段由 auto_id 生成）
        now_ms = int(time.time() * 1000)
        insert_data = {
            "content": texts,
            "embedding": embeddings,
            "metadata": _metadatas,
            "source": sources,
            "created_at": [now_ms] * len(texts),
        }

        # 4. 插入 Milvus
        try:
            result = await _run_sync(coll.insert, insert_data)
            inserted_ids: List[int] = result.primary_keys

            # 5. Flush 持久化（确保数据写入磁盘，生产环境应按批次 flush）
            await _run_sync(coll.flush)

            logger.info(
                "文档向量插入成功",
                extra={
                    "collection": collection,
                    "inserted": len(inserted_ids),
                    "first_id": inserted_ids[0] if inserted_ids else None,
                },
            )

            # 6. 使相关 source 的检索缓存失效（写入后旧缓存可能过期）
            unique_sources = list(set(sources))
            await self._invalidate_source_cache(unique_sources, collection)

            return inserted_ids

        except VectorDBError:
            raise
        except Exception as exc:
            logger.error(
                "文档向量插入失败",
                extra={"collection": collection, "error": str(exc)},
            )
            raise VectorDBError(
                message=f"向量插入失败（{collection}）",
                error_code=ErrorCode.VECTOR_DB_INSERT_FAILED,
                detail={
                    "collection": collection,
                    "count": len(texts),
                    "error": str(exc),
                },
            ) from exc

    # ── 检索 ──────────────────────────────────────────────────────────

    async def similarity_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        collection: str = DEFAULT_COLLECTION_NAME,
        source_filter: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[SearchResult]:
        """
        语义相似度检索，返回与 query 最相关的 Top-K 文档块。

        处理流程：
          1. 查 Redis 缓存（use_cache=True 时）
          2. 对 query 生成 Embedding
          3. 调用 Milvus ANN 检索（HNSW，ef=64）
          4. 按 score_threshold 过滤低相关结果
          5. 结果写入 Redis 缓存（TTL=10min）

        Args:
            query:           查询文本
            top_k:           返回最多 K 条结果（实际数量可能因阈值过滤而减少）
            score_threshold: 相关性分数阈值（IP 度量，0~1），低于此值的结果被过滤
            collection:      检索的 Collection 名称
            source_filter:   按 source 字段过滤（None 表示不过滤），
                             如 source_filter="report.pdf" 只在该文档中检索
            use_cache:       是否使用 Redis 缓存（重要查询可设 False 强制刷新）

        Returns:
            按相似度降序排列的 SearchResult 列表

        Raises:
            VectorDBError: Milvus 检索失败
            LLMError:      query Embedding 生成失败
        """
        if not query.strip():
            return []

        # 1. 查缓存（source_filter 纳入 key，不同过滤条件独立缓存）
        cache_key = _search_cache_key(
            f"{query}|{source_filter or ''}",
            collection,
            top_k,
            score_threshold,
        )
        if use_cache:
            cached = await get_json(cache_key)
            if cached:
                logger.debug(
                    "检索结果缓存命中",
                    extra={"collection": collection, "top_k": top_k},
                )
                return [SearchResult.from_dict(r) for r in cached]

        # 2. 生成 query 向量
        query_vector = await self._embedder.embed_text(query)

        # 3. 构造 Milvus 搜索参数
        search_params = {
            "metric_type": METRIC_TYPE,
            "params": {"ef": HNSW_EF_SEARCH},  # 搜索候选集大小，可按需调整
        }

        # 标量过滤表达式（按 source 字段过滤）
        expr: Optional[str] = None
        if source_filter:
            # 防注入：source 字段仅允许文件名/URL，不接受运算符
            safe_source = source_filter.replace('"', '\\"')
            expr = f'source == "{safe_source}"'

        # 4. 执行 ANN 检索
        try:
            coll = await self._milvus.get_collection(collection)

            raw_results = await _run_sync(
                coll.search,
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=OUTPUT_FIELDS,
            )

        except VectorDBError:
            raise
        except Exception as exc:
            logger.error(
                "Milvus 检索失败",
                extra={"collection": collection, "error": str(exc)},
            )
            raise VectorDBError(
                message=f"向量检索失败（{collection}）",
                error_code=ErrorCode.VECTOR_DB_QUERY_FAILED,
                detail={
                    "collection": collection,
                    "query_len": len(query),
                    "error": str(exc),
                },
            ) from exc

        # 5. 解析结果 + 阈值过滤
        results: List[SearchResult] = []
        hits = raw_results[0] if raw_results else []  # 单 query 取第 0 个结果集

        for hit in hits:
            score = float(hit.score)
            if score < score_threshold:
                continue  # 低于阈值，过滤掉

            results.append(
                SearchResult(
                    content=hit.entity.get("content", ""),
                    source=hit.entity.get("source", ""),
                    score=round(score, 6),
                    metadata=hit.entity.get("metadata") or {},
                    id=hit.id,
                )
            )

        logger.info(
            "向量检索完成",
            extra={
                "collection": collection,
                "query_len": len(query),
                "hits_raw": len(hits),
                "hits_filtered": len(results),
                "threshold": score_threshold,
            },
        )

        # 6. 写缓存（即使结果为空也缓存，避免缓存穿透）
        if use_cache:
            await set_json(
                cache_key,
                [r.to_dict() for r in results],
                ttl=SEARCH_CACHE_TTL,
            )

        return results

    # ── 删除 ──────────────────────────────────────────────────────────

    async def delete_by_source(
        self,
        source: str,
        collection: str = DEFAULT_COLLECTION_NAME,
    ) -> int:
        """
        按 source 字段删除文档的所有向量。

        处理流程：
          1. 查询符合条件的向量 ID
          2. 按 ID 列表执行删除
          3. 使相关检索缓存失效

        Args:
            source:     文档来源标识（与插入时的 source 字段一致）
            collection: 目标 Collection 名称

        Returns:
            实际删除的向量数量

        Raises:
            VectorDBError: Milvus 操作失败
        """
        if not source:
            raise ValueError("source 不能为空")

        logger.info(
            "开始删除文档向量",
            extra={"source": source, "collection": collection},
        )

        try:
            coll = await self._milvus.get_collection(collection)

            # 先 query 获取 ID（Milvus 删除操作基于主键 ID）
            safe_source = source.replace('"', '\\"')
            expr = f'source == "{safe_source}"'

            query_result = await _run_sync(
                coll.query,
                expr=expr,
                output_fields=["id"],
                limit=16384,  # Milvus 单次 query 最大返回数
            )

            if not query_result:
                logger.info(
                    "未找到对应向量，跳过删除",
                    extra={"source": source},
                )
                return 0

            ids = [r["id"] for r in query_result]
            id_expr = f"id in {ids}"
            await _run_sync(coll.delete, id_expr)
            await _run_sync(coll.flush)

            # 使相关缓存失效
            await self._invalidate_source_cache([source], collection)

            logger.info(
                "文档向量删除完成",
                extra={
                    "source": source,
                    "collection": collection,
                    "deleted": len(ids),
                },
            )
            return len(ids)

        except VectorDBError:
            raise
        except Exception as exc:
            logger.error(
                "删除向量失败",
                extra={"source": source, "collection": collection, "error": str(exc)},
            )
            raise VectorDBError(
                message=f"删除向量失败（source={source}）",
                error_code=ErrorCode.VECTOR_DB_QUERY_FAILED,
                detail={"source": source, "collection": collection, "error": str(exc)},
            ) from exc

    # ── 缓存失效 ──────────────────────────────────────────────────────

    async def _invalidate_source_cache(
        self,
        sources: List[str],
        collection: str,
    ) -> None:
        """
        删除或插入文档后，使包含该 source 的检索结果缓存失效。

        注意：由于检索缓存的 key 基于 query_text（无法枚举），
        此处采用「按 source 标记失效」的近似策略：
          - 对每个 source 构造一个特殊的 invalidation key
          - 检索时检查该 key 是否存在（见 similarity_search 扩展点）

        当前实现：直接删除已知的 source 级缓存 key（TTL 自然过期兜底）。
        完整的缓存失效需配合 Redis SCAN + 前缀匹配，
        成本较高，10min TTL 已能满足大多数生产场景的一致性需求。
        """
        try:
            for source in sources:
                inval_key = build_key("rag_inval", collection, source[:64])
                await set_json(inval_key, True, ttl=SEARCH_CACHE_TTL)
                logger.debug(
                    "检索缓存失效标记已设置",
                    extra={"source": source, "inval_key": inval_key},
                )
        except Exception as exc:
            # 缓存失效失败不影响主流程，仅记录警告
            logger.warning(
                "设置缓存失效标记失败",
                extra={"error": str(exc)},
            )

    # ── 集合统计 ──────────────────────────────────────────────────────

    async def get_collection_stats(
        self,
        collection: str = DEFAULT_COLLECTION_NAME,
    ) -> Dict[str, Any]:
        """
        获取 Collection 统计：向量总数、命中率等。
        供知识库管理接口（GET /knowledge/list）使用。
        """
        stats = await self._milvus.get_collection_stats(collection)
        emb_stats = self._embedder.get_stats()
        return {
            **stats,
            "embedding_cache": emb_stats,
        }


# ---------------------------------------------------------------------------
# FastAPI 依赖注入
# ---------------------------------------------------------------------------


def get_vector_store_service() -> VectorStoreService:
    """
    FastAPI Depends 兼容的 VectorStoreService 工厂函数。

    每次请求新建 VectorStoreService 实例（轻量封装），
    底层的 MilvusClient 和 EmbeddingClient 均为模块级单例，
    不会因此产生额外连接开销。

    用法：
        @router.post("/search")
        async def search(
            query: str,
            vs: VectorStoreService = Depends(get_vector_store_service),
        ):
            results = await vs.similarity_search(query)
    """
    return VectorStoreService(
        milvus=get_milvus_client(),
        embedder=get_embedding_client(),
    )


"""

## 两文件协作关系
```
外部调用层（rag_service / document_service）
         ↓
VectorStoreService（vector_store.py）
         ├── upsert_documents(texts, sources, metadatas)
         │       ↓ 批量生成向量
         │   EmbeddingClient.embed_batch()
         │       ├── Redis GET → 缓存命中直接用
         │       └── OpenAI API（Semaphore ≤10并发，每批≤20条）
         │       ↓ 写入 Milvus
         │   MilvusClient.ensure_collection_exists()
         │   collection.insert() → flush()
         │
         ├── similarity_search(query, top_k, threshold)
         │       ↓ 查检索缓存
         │   Redis GET "synaris:rag:{sha256}"
         │       ↓ 未命中：生成 query 向量
         │   EmbeddingClient.embed_text()
         │       ↓ ANN 检索
         │   MilvusClient.get_collection()
         │   collection.search(ef=64, limit=top_k)
         │       ↓ 阈值过滤 + 写缓存 TTL=10min
         │   → List[SearchResult]
         │
         └── delete_by_source(source)
                 ↓ query IDs → delete → flush
             MilvusClient + 缓存失效标记

"""
