"""
@File       : dependencies.py
@Author     : Starry Hung
@Created    : 2026-04-22
@Version    : 1.0.0
@Description: 全局服务实例注入容器（Dependency Injection Container）。
@Features:
  - 集中管理所有有状态服务的单例实例，避免分散初始化导致的状态不一致
  - 懒初始化（Lazy Init）：首次访问时创建，避免启动时因依赖服务未就绪而崩溃
  - 提供 LLMClientWrapper：将 LangChain ChatOpenAI 包装为 rag_service.py 期望的接口
  - 提供 initialize_dependencies() 在 lifespan 启动阶段主动预热所有实例
  - 对外暴露三个模块级变量供 api/ 层导入：
      document_service_instance   → DocumentService
      vector_store_instance       → VectorStoreService
      rag_service_instance        → RAGService

  领域建模类比（「工厂备料室」模式）：
    工厂开工前，备料室（dependencies.py）预先备好所有原材料（服务实例），
    各条产线（API 路由）按需取用，不需要自己去仓库（底层基础设施）搬运。

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-22  Starry  Initial creation
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLMClientWrapper
#
# 问题背景：
#   rag_service.py 期望注入的 llm_client 提供两个方法：
#     - get_llm(strategy, streaming) → LangChain ChatOpenAI 实例（用于 LCEL Chain）
#     - get_last_token_usage()       → int（最近一次调用的 Token 数）
#
#   但 llm_client.py 中的 LLMClientInterface 是「调用-等待结果」模式，
#   不直接暴露底层 ChatOpenAI 对象，两者接口不兼容。
#
# 解决方案：
#   在此处定义 LLMClientWrapper，封装 LangChain ChatOpenAI 的构造逻辑，
#   提供 rag_service.py 所需的两个接口，并通过 LLMClientPool 复用实例。
#
# 类比：「翻译适配器」——rag_service 说的语言与 llm_client 说的语言不同，
#       LLMClientWrapper 在中间做实时翻译，让双方都能正常工作。
# ─────────────────────────────────────────────────────────────────────────────

class LLMClientWrapper:
    """
    LangChain ChatOpenAI 的轻量级包装器，对齐 rag_service.py 的调用接口。

    接口契约：
      get_llm(strategy, streaming=False) → ChatOpenAI
      get_last_token_usage()             → int
    """

    # 策略 → 模型名称映射（与 llm_router.py 的 TASK_ROUTING_TABLE 保持一致）
    _STRATEGY_MODEL_MAP: dict[str, str] = {
        "quality":  "gpt-4o",
        "balanced": "gpt-4o-mini",
        "economy":  "gpt-3.5-turbo",
    }

    def __init__(self) -> None:
        # 模型实例缓存：key = "model_name:streaming"，避免重复构造
        self._cache: dict[str, object] = {}
        # 记录最近一次调用的 Token 用量（由 CostService 回写，当前以 0 占位）
        self._last_token_usage: int = 0

    def get_llm(self, strategy: str = "balanced", streaming: bool = False) -> object:
        """
        根据路由策略返回对应的 ChatOpenAI 实例。

        Args:
            strategy:  路由策略（quality / balanced / economy）
            streaming: 是否启用流式输出

        Returns:
            LangChain ChatOpenAI 实例（可直接组入 LCEL Chain）

        实现细节：
          同一 (model, streaming) 组合复用同一实例（连接池语义）。
          temperature 在 RAG 场景下固定为 0.3（忠实度优先，低创造性）。
        """
        model_name = self._STRATEGY_MODEL_MAP.get(strategy, "gpt-4o-mini")
        cache_key = f"{model_name}:{streaming}"

        if cache_key not in self._cache:
            try:
                from langchain_openai import ChatOpenAI  # type: ignore[import]
                from app.config.settings import get_settings

                settings = get_settings()
                api_key = settings.openai.get_api_key()
                base_url = settings.openai.OPENAI_BASE_URL or None

                self._cache[cache_key] = ChatOpenAI(
                    model=model_name,
                    temperature=0.3,        # RAG 生成偏低温，保证内容忠实
                    streaming=streaming,
                    openai_api_key=api_key,
                    openai_api_base=base_url,
                    max_retries=0,          # 重试由 llm_router 统一管理
                )
                logger.debug(
                    "LLMClientWrapper: ChatOpenAI 实例已创建",
                    extra={"model": model_name, "streaming": streaming},
                )
            except ImportError as exc:
                raise RuntimeError(
                    f"langchain-openai 未安装，请执行: pip install langchain-openai\n原始错误: {exc}"
                ) from exc

        return self._cache[cache_key]

    def get_last_token_usage(self) -> int:
        """
        返回最近一次 LLM 调用的 Token 总用量。

        当前阶段（Step 24 CostService 完成前）返回 0 作为占位。
        Step 24 完成后，CostService 的 record_usage() 会回调写入此值。
        """
        return self._last_token_usage

    def update_token_usage(self, tokens: int) -> None:
        """由 CostService 在记录用量后回调更新，对业务层透明。"""
        self._last_token_usage = tokens


# ─────────────────────────────────────────────────────────────────────────────
# 2. 私有单例持有变量
#
# 使用「可选类型 + 懒初始化」模式：
#   - None 表示「尚未初始化」
#   - 首次访问（通过 getter）时创建实例
#   - 后续访问直接返回已有实例（单例语义）
#
# 为什么不直接写 module-level 赋值？
#   因为 VectorStoreService 依赖 MilvusClient，而 MilvusClient 需要在
#   lifespan 的 connect() 之后才能正常工作。在模块导入时立即构造会触发
#   「连接尚未建立」的错误，懒初始化可以将构造推迟到真正使用时。
# ─────────────────────────────────────────────────────────────────────────────

_vector_store_instance: Optional["VectorStoreService"] = None       # type: ignore[name-defined]
_document_service_instance: Optional["DocumentService"] = None      # type: ignore[name-defined]
_rag_service_instance: Optional["RAGService"] = None                # type: ignore[name-defined]
_llm_wrapper_instance: Optional[LLMClientWrapper] = None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Getter 函数（内部使用，实现懒初始化）
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm_wrapper() -> LLMClientWrapper:
    """LLMClientWrapper 懒初始化 getter。"""
    global _llm_wrapper_instance
    if _llm_wrapper_instance is None:
        _llm_wrapper_instance = LLMClientWrapper()
        logger.debug("LLMClientWrapper 单例已创建")
    return _llm_wrapper_instance


def _get_vector_store() -> "VectorStoreService":  # type: ignore[name-defined]
    """
    VectorStoreService 懒初始化 getter。

    依赖链：
        get_milvus_client()    → MilvusClient（基础设施层单例）
        get_embedding_client() → EmbeddingClient（基础设施层单例）
        VectorStoreService(milvus, embedder) → 业务层单例
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        try:
            from app.infrastructure.milvus_client import get_milvus_client
            from app.infrastructure.embedding_client import get_embedding_client
            from app.services.vector_store import VectorStoreService

            _vector_store_instance = VectorStoreService(
                milvus=get_milvus_client(),
                embedder=get_embedding_client(),
            )
            logger.info("VectorStoreService 单例已创建")
        except Exception as exc:
            logger.error("VectorStoreService 初始化失败: %s", exc)
            raise
    return _vector_store_instance


def _get_document_service() -> "DocumentService":  # type: ignore[name-defined]
    """
    DocumentService 懒初始化 getter。

    依赖链：
        _get_vector_store() → VectorStoreService
        DocumentService(vector_store, chunk_size, chunk_overlap) → 业务层单例
    """
    global _document_service_instance
    if _document_service_instance is None:
        try:
            from app.services.document_service import DocumentService
            from app.config.settings import get_settings

            settings = get_settings()
            _document_service_instance = DocumentService(
                vector_store=_get_vector_store(),
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            logger.info("DocumentService 单例已创建")
        except Exception as exc:
            logger.error("DocumentService 初始化失败: %s", exc)
            raise
    return _document_service_instance


def _get_rag_service() -> "RAGService":  # type: ignore[name-defined]
    """
    RAGService 懒初始化 getter。

    依赖链：
        _get_vector_store()    → VectorStoreService
        get_embedding_client() → EmbeddingClient
        _get_llm_wrapper()     → LLMClientWrapper
        RAGService(vector_store, embedding_client, llm_client) → 业务层单例
    """
    global _rag_service_instance
    if _rag_service_instance is None:
        try:
            from app.infrastructure.embedding_client import get_embedding_client
            from app.services.rag_service import RAGService
            from app.config.settings import get_settings

            settings = get_settings()
            _rag_service_instance = RAGService(
                vector_store=_get_vector_store(),
                embedding_client=get_embedding_client(),
                llm_client=_get_llm_wrapper(),
                top_k=settings.milvus.MILVUS_DEFAULT_TOP_K,
                top_rerank=settings.RAG_RERANK_TOP_K,
                score_threshold=settings.milvus.MILVUS_SIMILARITY_THRESHOLD,
            )
            logger.info("RAGService 单例已创建")
        except Exception as exc:
            logger.error("RAGService 初始化失败: %s", exc)
            raise
    return _rag_service_instance


# ─────────────────────────────────────────────────────────────────────────────
# 4. 对外暴露的模块级变量
#
# api/knowledge.py 和 api/rag.py 通过以下方式导入：
#   from dependencies import document_service_instance
#   from dependencies import vector_store_instance
#   from dependencies import rag_service_instance
#
# 使用「属性访问代理」模式：
#   定义一个 _ServiceProxy 类，让模块级变量在被访问时触发懒初始化，
#   而不是在模块导入时立即构造（避免循环导入和启动时序问题）。
#
# 这类似于 Java 的 ApplicationContext.getBean()，
# 或 Python 中常见的 LazyProxy 模式。
# ─────────────────────────────────────────────────────────────────────────────

class _ServiceProxy:
    """
    服务实例的懒加载代理。

    当外部代码访问 `document_service_instance.some_method()` 时，
    代理对象拦截所有属性访问，首次访问时触发真实实例的构造。

    类比：超市的「收银台叫号系统」——
      你拿到号码牌（代理对象）时，柜台还没准备好；
      真正到你时（属性访问），柜台（服务实例）才开始服务你。
    """

    def __init__(self, getter):
        # 使用 object.__setattr__ 绕过自定义的 __getattr__，避免递归
        object.__setattr__(self, "_getter", getter)
        object.__setattr__(self, "_instance", None)

    def _get_instance(self):
        instance = object.__getattribute__(self, "_instance")
        if instance is None:
            getter = object.__getattribute__(self, "_getter")
            instance = getter()
            object.__setattr__(self, "_instance", instance)
        return instance

    def __getattr__(self, name: str):
        return getattr(self._get_instance(), name)

    def __repr__(self) -> str:
        try:
            return repr(self._get_instance())
        except Exception:
            return f"<_ServiceProxy getter={object.__getattribute__(self, '_getter').__name__}>"


#: 文档处理服务单例（供 api/knowledge.py 导入）
document_service_instance = _ServiceProxy(_get_document_service)

#: 向量存储服务单例（供 api/knowledge.py 导入）
vector_store_instance = _ServiceProxy(_get_vector_store)

#: RAG 问答服务单例（供 api/rag.py 导入）
rag_service_instance = _ServiceProxy(_get_rag_service)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 主动预热函数（在 lifespan 启动阶段调用）
#
# 虽然懒初始化能保证功能正确性，但首次请求时会承担初始化延迟。
# 通过在 lifespan 阶段主动预热，将初始化开销前移，确保服务就绪后
# 所有实例都已可用，用户请求无需等待。
#
# 类比：餐厅开门前的「备餐」——大厨提前备好食材，
#       顾客点单时可以立即开始烹饪，而不是临时去买菜。
# ─────────────────────────────────────────────────────────────────────────────

async def initialize_dependencies() -> None:
    """
    主动预热所有服务单例，建议在 FastAPI lifespan 的启动阶段调用。

    调用方式（main.py lifespan）：
        from dependencies import initialize_dependencies

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await initialize_dependencies()   # ← 加在 Redis/Milvus 连接之后
            yield
            ...

    执行顺序（严格按依赖层次）：
        1. LLMClientWrapper（无外部依赖，最轻量）
        2. VectorStoreService（依赖 Milvus + EmbeddingClient）
        3. DocumentService（依赖 VectorStoreService）
        4. RAGService（依赖 VectorStoreService + EmbeddingClient + LLMWrapper）
    """
    logger.info("开始预热服务实例...")

    errors: list[str] = []

    # 1. LLMClientWrapper（无外部 IO，必然成功）
    try:
        _get_llm_wrapper()
        logger.info("  ✓ LLMClientWrapper 预热完成")
    except Exception as exc:
        msg = f"  ✗ LLMClientWrapper 预热失败: {exc}"
        logger.error(msg)
        errors.append(msg)

    # 2. VectorStoreService（依赖 Milvus，若未就绪会失败）
    try:
        _get_vector_store()
        logger.info("  ✓ VectorStoreService 预热完成")
    except Exception as exc:
        msg = f"  ✗ VectorStoreService 预热失败: {exc}"
        logger.warning(msg)  # 降级为 warning，不阻断启动（Milvus 可能还在冷启动）
        errors.append(msg)

    # 3. DocumentService（依赖 VectorStoreService）
    try:
        _get_document_service()
        logger.info("  ✓ DocumentService 预热完成")
    except Exception as exc:
        msg = f"  ✗ DocumentService 预热失败: {exc}"
        logger.warning(msg)
        errors.append(msg)

    # 4. RAGService（依赖最多，放最后）
    try:
        _get_rag_service()
        logger.info("  ✓ RAGService 预热完成")
    except Exception as exc:
        msg = f"  ✗ RAGService 预热失败: {exc}"
        logger.warning(msg)
        errors.append(msg)

    if errors:
        logger.warning(
            "服务预热部分失败（共 %d 项），相关功能首次访问时将重试初始化",
            len(errors),
        )
    else:
        logger.info("所有服务实例预热完成 ✓")


def reset_dependencies() -> None:
    """
    重置所有单例（仅用于测试，生产环境禁止调用）。

    测试用法：
        from dependencies import reset_dependencies
        reset_dependencies()  # 在每个测试用例的 teardown 中调用
    """
    global _vector_store_instance, _document_service_instance
    global _rag_service_instance, _llm_wrapper_instance

    _vector_store_instance = None
    _document_service_instance = None
    _rag_service_instance = None
    _llm_wrapper_instance = None

    logger.warning("所有依赖实例已重置（测试模式）")
