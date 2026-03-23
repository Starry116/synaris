"""
@File       : rag_service.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 完整 RAG Chain 服务（LCEL 语法）。
@Features:
  - 完整流水线：query → embed → milvus_search → rerank → prompt_inject → llm_generate
  - Reranking：按 score 降序，取 top_rerank（默认 3）注入 Prompt
  - 引用溯源：响应附带 SourceDoc 列表（source / chunk_index / snippet / score）
  - async query()        → RAGResponse（完整回答 + 引用）
  - async query_stream() → AsyncGenerator[str]（SSE 逐 token 输出）
@Project    : Synaris
@License    : Apache License 2.0
@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import time
from typing import AsyncGenerator, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.exceptions import LLMError, VectorDBError
from core.logging import get_logger, log_execution_time
from infrastructure.embedding_client import EmbeddingClient
from infrastructure.llm_client import LLMClient
from schemas.rag import RAGResponse, SourceDoc
from services.vector_store import VectorStoreService

logger = get_logger(__name__)

# ─── RAG Prompt ────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """你是 Synaris 企业知识库助手。请严格基于以下参考文档回答用户问题。

参考文档：
{context}

回答要求：
1. 只基于参考文档中的信息作答，不得编造
2. 如果文档中没有相关信息，明确告知用户
3. 回答简洁、专业，使用中文

用户问题：{question}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
)


# ─── 服务主体 ───────────────────────────────────────────────────────────────────


class RAGService:
    """
    RAG Chain 服务。

    流水线（类比「图书馆馆员」工作流）：
      1. 理解问题 → embed（语义编码）
      2. 查找书架 → milvus_search（向量检索）
      3. 筛选最相关 → rerank（按分数取 top_rerank）
      4. 摆上桌面 → prompt_inject（构建上下文）
      5. 撰写答案 → llm_generate（LLM 生成）
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        top_k: int = 5,
        top_rerank: int = 3,
        score_threshold: float = 0.5,
    ) -> None:
        self._vector_store = vector_store
        self._embedding = embedding_client
        self._llm = llm_client
        self._top_k = top_k
        self._top_rerank = top_rerank
        self._score_threshold = score_threshold

    # ─── 内部工具方法 ───────────────────────────────────────────────────────────

    async def _retrieve_and_rerank(
        self,
        question: str,
        collection_name: str,
        top_k: int,
        score_threshold: float,
    ) -> tuple[List[Document], List[Document]]:
        """
        检索 + Reranking。

        Returns:
            (all_retrieved, reranked_top)
              all_retrieved : 原始检索结果
              reranked_top  : Reranking 后取 top_rerank 的子集
        """
        try:
            results: List[Document] = await self._vector_store.similarity_search(
                query=question,
                collection_name=collection_name,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise VectorDBError(f"Milvus 检索失败：{exc}") from exc

        if not results:
            logger.info("未检索到相关文档", extra={"question": question[:50]})
            return [], []

        # Reranking：按 score 降序，取 top_rerank
        ranked = sorted(
            results,
            key=lambda d: d.metadata.get("score", 0.0),
            reverse=True,
        )[: self._top_rerank]

        logger.debug(
            "Reranking 完成",
            extra={"retrieved": len(results), "after_rerank": len(ranked)},
        )
        return results, ranked

    @staticmethod
    def _build_context(docs: List[Document]) -> str:
        """将文档列表拼接为 Prompt 上下文字符串。"""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            parts.append(f"[文档{i}]（来源：{source}）\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_sources(docs: List[Document]) -> List[SourceDoc]:
        """从检索结果构建引用溯源列表。"""
        sources = []
        for doc in docs:
            meta = doc.metadata
            sources.append(
                SourceDoc(
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    page_num=meta.get("page_num", 0),
                    snippet=doc.page_content[:50].strip(),
                    score=round(float(meta.get("score", 0.0)), 4),
                )
            )
        return sources

    # ─── 公开接口：完整回答 ─────────────────────────────────────────────────────

    @log_execution_time
    async def query(
        self,
        question: str,
        collection_name: str = "default",
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """
        完整 RAG 查询，返回答案 + 引用溯源。

        LCEL Chain：
          context_builder | prompt | llm | str_parser
        """
        top_k = top_k or self._top_k
        score_threshold = score_threshold or self._score_threshold
        start = time.monotonic()

        # ① 检索 + Reranking
        _, reranked = await self._retrieve_and_rerank(
            question, collection_name, top_k, score_threshold
        )

        # ② 无文档时的降级处理
        if not reranked:
            return RAGResponse(
                answer="抱歉，知识库中未找到与该问题相关的内容，请尝试换一种提问方式或上传相关文档。",
                sources=[],
                tokens_used=0,
                collection_name=collection_name,
                reranked_count=0,
            )

        context = self._build_context(reranked)
        sources = self._build_sources(reranked)

        # ③ LCEL Chain 组装
        llm = self._llm.get_llm(strategy="balanced")
        chain = (
            RunnablePassthrough.assign(context=lambda _: context)
            | _PROMPT
            | llm
            | StrOutputParser()
        )

        try:
            answer: str = await chain.ainvoke(
                {"question": question, "context": context}
            )
        except Exception as exc:
            raise LLMError(f"LLM 生成失败：{exc}") from exc

        # ④ Token 统计（从 LLM 客户端获取最近调用记录）
        tokens_used = self._llm.get_last_token_usage()

        logger.info(
            "RAG 查询完成",
            extra={
                "question": question[:50],
                "sources_count": len(sources),
                "tokens_used": tokens_used,
                "duration_ms": round((time.monotonic() - start) * 1000, 1),
            },
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            tokens_used=tokens_used,
            collection_name=collection_name,
            reranked_count=len(reranked),
        )

    # ─── 公开接口：流式回答 ─────────────────────────────────────────────────────

    async def query_stream(
        self,
        question: str,
        collection_name: str = "default",
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式 RAG 查询，逐 token yield。

        SSE 格式由 api/rag.py 层封装，此处只 yield 原始文本片段。
        """
        top_k = top_k or self._top_k
        score_threshold = score_threshold or self._score_threshold

        _, reranked = await self._retrieve_and_rerank(
            question, collection_name, top_k, score_threshold
        )

        # 无文档时直接 yield 提示语
        if not reranked:
            yield "抱歉，知识库中未找到与该问题相关的内容。"
            return

        context = self._build_context(reranked)
        sources = self._build_sources(reranked)

        llm = self._llm.get_llm(strategy="balanced", streaming=True)
        chain = (
            RunnablePassthrough.assign(context=lambda _: context)
            | _PROMPT
            | llm
            | StrOutputParser()
        )

        try:
            async for chunk in chain.astream(
                {"question": question, "context": context}
            ):
                yield chunk
        except Exception as exc:
            logger.error("流式 RAG 生成失败", extra={"error": str(exc)})
            yield f"\n\n[生成中断：{exc}]"
            return

        # 流结束后以 JSON 附加引用溯源（SSE 最后一帧）
        import json

        sources_payload = [s.model_dump() for s in sources]
        yield f"\n\n[SOURCES]{json.dumps(sources_payload, ensure_ascii=False)}[/SOURCES]"
