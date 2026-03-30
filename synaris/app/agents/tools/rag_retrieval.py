"""
@File       : rag_retrieval.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Agent 内部知识库检索工具。
@Features:
  - @tool 装饰器：供 LangGraph ToolSelector 节点直接调用
  - 将 services/vector_store.py 的 similarity_search() 包装为 Agent 可调用工具
  - 输入：query(str) + collection(str) + top_k(int) + score_threshold(float)
  - 输出：格式化的检索结果文本，包含序号 / 来源 / 相关度分数 / 内容片段
  - 相关度过滤：score_threshold（0~1），低于阈值的结果自动剔除
  - 异步兼容：提供同步包装（LangChain @tool 要求同步函数），
    内部使用 asyncio.run_coroutine_threadsafe 或 nest_asyncio 处理
  - 错误处理：Milvus 连接失败 / Collection 不存在 / 空结果均返回结构化提示

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────
_DEFAULT_COLLECTION      = "default"   # 默认知识库 Collection 名称
_DEFAULT_TOP_K           = 3           # 默认返回结果数量
_DEFAULT_SCORE_THRESHOLD = 0.5         # 默认相关度阈值（0~1，越高越严格）
_MAX_TOP_K               = 10          # 硬上限，防止 Token 爆炸
_CONTENT_PREVIEW_LEN     = 300         # 每条内容预览的最大字符数


# ─────────────────────────────────────────────
# 1. 入参模型
# ─────────────────────────────────────────────

class RAGRetrievalInput(BaseModel):
    """rag_retrieval 工具的入参模型（供 ToolSelector 生成合法 JSON 入参）。"""

    query: str = Field(
        description="检索查询语句，建议使用清晰的自然语言描述，如「公司差旅报销流程」"
    )
    collection: str = Field(
        default=_DEFAULT_COLLECTION,
        description=(
            "目标知识库 Collection 名称。"
            "默认为 'default'（公共知识库）。"
            "如需检索特定领域知识库，传入对应名称，如 'hr_policies'、'product_docs'。"
        ),
    )
    top_k: int = Field(
        default=_DEFAULT_TOP_K,
        ge=1,
        le=_MAX_TOP_K,
        description=f"返回最相关的文档片段数量（1-{_MAX_TOP_K}，默认 {_DEFAULT_TOP_K}）",
    )
    score_threshold: float = Field(
        default=_DEFAULT_SCORE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "相关度阈值（0~1）。"
            "低于此分数的结果将被过滤。"
            "值越高，结果越精准但可能返回更少结果；"
            "值越低，召回率更高但可能包含不相关内容。"
            "建议范围：精准检索 0.7+，宽泛检索 0.3~0.5。"
        ),
    )


# ─────────────────────────────────────────────
# 2. 异步 → 同步适配器
# ─────────────────────────────────────────────

def _run_async(coro) -> any:
    """
    在同步上下文中运行异步协程的通用适配器。

    处理三种运行时场景：
    1. 没有事件循环（普通 Python 脚本）：直接 asyncio.run()
    2. 有事件循环但未运行（测试 fixture 中）：loop.run_until_complete()
    3. 有正在运行的事件循环（FastAPI / Celery）：使用线程池提交，避免死锁

    类比：
    - 这相当于在「已经有一位售货员在忙」的收银台旁边，
      开一个临时窗口处理一笔同步请求，不堵塞原来的队伍。
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 没有事件循环，直接创建并运行
        return asyncio.run(coro)

    if loop.is_running():
        # 在已有运行中的事件循环里（FastAPI / Jupyter / Celery）
        # 使用 concurrent.futures 在新线程中运行，避免 "cannot run nested event loop" 错误
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=30)
    else:
        return loop.run_until_complete(coro)


# ─────────────────────────────────────────────
# 3. 核心检索逻辑
# ─────────────────────────────────────────────

async def _retrieve_async(
    query:           str,
    collection:      str,
    top_k:           int,
    score_threshold: float,
) -> list[dict]:
    """
    异步调用 vector_store.similarity_search()，返回原始结果列表。

    返回的每条结果字典格式（由 vector_store.py 定义）：
        {
            "content":    str,   # 文档片段内容
            "source":     str,   # 来源文件名/标识
            "score":      float, # 相似度分数（0~1）
            "metadata":   dict,  # 额外元数据（chunk_index / doc_id 等）
        }
    """
    # 延迟导入，避免循环依赖（tools 不应在模块加载时触发 Milvus 连接）
    from services.vector_store import VectorStore  # type: ignore[import]

    vector_store = VectorStore()
    results = await vector_store.similarity_search(
        query=query,
        collection_name=collection,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return results


# ─────────────────────────────────────────────
# 4. 结果格式化
# ─────────────────────────────────────────────

def _format_retrieval_results(
    results: list[dict],
    query:   str,
    collection: str,
) -> str:
    """
    将原始检索结果格式化为 LLM 易读的文本块。

    格式示例：
        知识库检索结果 | 关键词：XX | Collection：default | 共 3 条

        【1】来源：产品手册_v2.pdf  |  相关度：0.87
            内容：用户可通过「设置 → 账户」页面修改密码…

        【2】来源：FAQ.docx  |  相关度：0.74
            内容：密码忘记时，点击登录页的「忘记密码」…
    """
    if not results:
        return (
            f"知识库「{collection}」中未找到与「{query}」相关的内容。\n"
            f"建议：\n"
            f"  1. 尝试更换关键词（如使用同义词或更宽泛的描述）\n"
            f"  2. 降低 score_threshold 参数值（当前过滤阈值可能过高）\n"
            f"  3. 确认目标文档已上传至知识库"
        )

    lines = [
        f"知识库检索结果 | 关键词：{query} | "
        f"Collection：{collection} | 共 {len(results)} 条\n"
    ]

    for i, item in enumerate(results, start=1):
        content  = item.get("content", "")
        source   = item.get("source", "未知来源")
        score    = item.get("score", 0.0)
        metadata = item.get("metadata", {})

        # 内容预览截断
        preview = (
            content[:_CONTENT_PREVIEW_LEN] + "…"
            if len(content) > _CONTENT_PREVIEW_LEN
            else content
        )

        # 可选：显示 chunk 位置信息（若 metadata 中包含）
        chunk_hint = ""
        if "chunk_index" in metadata:
            chunk_hint = f"  |  第 {metadata['chunk_index']} 段"

        lines.append(
            f"【{i}】来源：{source}  |  相关度：{score:.2f}{chunk_hint}\n"
            f"    {preview}\n"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 5. @tool 入口
# ─────────────────────────────────────────────

@tool(args_schema=RAGRetrievalInput)
def rag_retrieval(
    query:           str,
    collection:      str  = _DEFAULT_COLLECTION,
    top_k:           int  = _DEFAULT_TOP_K,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
) -> str:
    """
    检索企业内部知识库，获取与问题最相关的文档片段。

    当需要以下情况时调用此工具：
    - 回答与企业内部文档、规章制度、产品手册相关的问题
    - 在生成内容前先获取准确的内部知识作为依据
    - 验证某个说法是否有知识库文档支撑

    注意事项：
    - 此工具只检索企业已上传的文档，不能访问互联网
    - 如需查询实时信息（如最新新闻、股价），请使用 web_search 工具
    - 如果返回结果相关度低，可尝试降低 score_threshold

    Args:
        query:           检索查询语句（自然语言，无需特殊格式）
        collection:      目标知识库名称（默认 "default"）
        top_k:           返回结果数量（1-10，默认 3）
        score_threshold: 相关度过滤阈值（0~1，默认 0.5）

    Returns:
        格式化的检索结果文本，包含来源、相关度分数和内容预览。
    """
    query = query.strip()

    if not query:
        return "错误：检索查询不能为空。"

    # 参数规范化
    top_k           = min(max(1, top_k), _MAX_TOP_K)
    score_threshold = max(0.0, min(1.0, score_threshold))

    start_time = time.monotonic()
    logger.info(
        "rag_retrieval 开始 | collection=%s | top_k=%d | threshold=%.2f | query=%s",
        collection, top_k, score_threshold, query,
    )

    try:
        results = _run_async(
            _retrieve_async(query, collection, top_k, score_threshold)
        )
        elapsed = time.monotonic() - start_time

        logger.info(
            "rag_retrieval 完成 | collection=%s | results=%d | elapsed=%.2fs",
            collection, len(results), elapsed,
        )

        return _format_retrieval_results(results, query, collection)

    except ModuleNotFoundError as exc:
        # vector_store 模块尚未实现（Step 9 之前）
        logger.error("rag_retrieval: vector_store 模块未就绪: %s", exc)
        return (
            "知识库服务暂不可用：vector_store 模块未初始化。\n"
            "此工具需要在 Step 9（services/vector_store.py）完成后才能正常使用。"
        )

    except ConnectionError as exc:
        logger.error("rag_retrieval: Milvus 连接失败: %s", exc)
        return (
            f"知识库连接失败：无法连接到向量数据库（Milvus）。\n"
            f"错误详情：{exc}\n"
            f"建议：请检查 Milvus 服务是否正常运行（docker-compose ps milvus）。"
        )

    except ValueError as exc:
        # Collection 不存在等业务异常
        logger.warning("rag_retrieval: 业务错误: %s", exc)
        return f"检索失败：{exc}"

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "rag_retrieval 异常 | collection=%s | elapsed=%.2fs | error=%s",
            collection, elapsed, exc,
        )
        return (
            f"检索失败：{type(exc).__name__}: {exc}\n"
            f"如果问题持续，请联系管理员检查 Milvus 服务状态。"
        )