"""
@File       : rag.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: RAG 查询 API（普通回答 + SSE 流式回答）。
@Features:
  - POST /rag/query        → 完整 RAG 回答 + 引用溯源，ApiResponse[RAGResponse]
  - POST /rag/query/stream → SSE 流式输出（text/event-stream）
  - SSE 帧格式：data: <token>\n\n，末帧附带 [SOURCES] JSON
  - 流式结束帧：data: [DONE]\n\n
@Project    : Synaris
@License    : Apache License 2.0
@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from core.exceptions import LLMError, VectorDBError
from core.logging import get_logger
from schemas.base import ApiResponse
from schemas.rag import RAGQueryRequest, RAGResponse
from services.rag_service import RAGService

logger = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


# ─── 依赖注入 ────────────────────────────────────────────────────────────────────


def get_rag_service() -> RAGService:
    from dependencies import rag_service_instance  # noqa: WPS433

    return rag_service_instance


# ─── POST /rag/query ─────────────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=ApiResponse[RAGResponse],
    summary="RAG 知识库问答",
    description="基于上传文档进行语义检索，经 Reranking 后由 LLM 生成带引用溯源的回答。",
)
async def rag_query(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ApiResponse[RAGResponse]:
    """
    完整 RAG 流水线：
      embed(question) → milvus_search → rerank(top3) → prompt → llm → answer + sources
    """
    logger.info(
        "收到 RAG 查询",
        extra={
            "question": request.question[:50],
            "collection": request.collection_name,
        },
    )

    try:
        response: RAGResponse = await rag_service.query(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except VectorDBError as exc:
        logger.error("向量检索失败", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except LLMError as exc:
        logger.error("LLM 生成失败", extra={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except Exception as exc:
        logger.error("RAG 查询异常", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RAG 查询失败"
        )

    return ApiResponse(success=True, message="查询成功", data=response)


# ─── POST /rag/query/stream ───────────────────────────────────────────────────────


@router.post(
    "/query/stream",
    summary="RAG 流式问答（SSE）",
    description="与 /rag/query 相同的 RAG 流水线，以 SSE 格式逐 token 流式输出。末帧附带引用溯源 JSON。",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE 流式文本",
        }
    },
)
async def rag_query_stream(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """
    SSE 帧格式：
      data: <token_chunk>\n\n          ← 正文逐 token
      data: [SOURCES]{...}[/SOURCES]\n\n  ← 末帧：引用溯源 JSON
      data: [DONE]\n\n                  ← 结束标志
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in rag_service.query_stream(
                question=request.question,
                collection_name=request.collection_name,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            ):
                # [SOURCES] 帧已由 rag_service 内部附加，直接透传
                yield f"data: {chunk}\n\n"

        except VectorDBError as exc:
            error_frame = json.dumps(
                {"error": "向量检索失败", "detail": str(exc)}, ensure_ascii=False
            )
            yield f"data: [ERROR]{error_frame}[/ERROR]\n\n"
        except LLMError as exc:
            error_frame = json.dumps(
                {"error": "LLM 生成失败", "detail": str(exc)}, ensure_ascii=False
            )
            yield f"data: [ERROR]{error_frame}[/ERROR]\n\n"
        except Exception as exc:
            logger.error("流式 RAG 异常", exc_info=True)
            error_frame = json.dumps(
                {"error": "内部错误", "detail": str(exc)}, ensure_ascii=False
            )
            yield f"data: [ERROR]{error_frame}[/ERROR]\n\n"
        finally:
            yield "data: [DONE]\n\n"

    logger.info(
        "收到 RAG 流式查询",
        extra={
            "question": request.question[:50],
            "collection": request.collection_name,
        },
    )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，保证实时推送
            "Connection": "keep-alive",
        },
    )
