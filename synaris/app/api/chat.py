"""
@File       : chat.py
@Author     : Starry Hung
@Created    : 2026-04-21
@Version    : 1.0.0
@Description: AI 聊天 API 路由层，提供多轮对话（普通 + 流式）与会话管理接口。
@Features:
  - POST /chat              → 非流式多轮对话，返回完整回复（ApiResponse[ChatResponse]）
  - POST /chat/stream       → 流式多轮对话，SSE 逐 Token 输出（text/event-stream）
      SSE 帧格式：data: {"delta": "...", "done": false}\n\n
      终止帧：data: {"delta": "", "done": true, "session_id": "..."}\n\n
      错误帧：data: {"delta": "", "done": true, "error": "..."}\n\n
  - DELETE /chat/session/{session_id}         → 清空指定会话的历史记录
  - GET    /chat/session/{session_id}/info    → 查询会话元信息（消息数 / 剩余 TTL）
  - 统一通过 ChatService（Depends 依赖注入）访问业务逻辑
  - slowapi 速率限制集成（每 IP 60次/分钟，可通过 settings 调整）
  - 全链路 TraceID 透传（从 request.state.trace_id 读取）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-21  Starry  Initial creation
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.schemas.base import ApiResponse, EmptyResponse
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionInfoResponse,
    StreamChunk,
)
from app.services.chat_service import ChatService, get_chat_service

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["AI 聊天"])


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _get_trace_id(request: Request) -> str:
    """从 request.state 中安全读取 TraceID（TraceIDMiddleware 注入）。"""
    return getattr(request.state, "trace_id", "")


def _sse_frame(chunk: StreamChunk) -> str:
    """
    将 StreamChunk 序列化为标准 SSE 数据帧字符串。

    SSE 协议要求：每帧以 "data: " 前缀开头，以两个换行符结尾。
    类比：广播电台播报格式——「播报内容」后跟固定的结束静音。
    """
    return f"data: {chunk.model_dump_json()}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat — 非流式多轮对话
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ApiResponse[ChatResponse],
    status_code=status.HTTP_200_OK,
    summary="多轮对话（完整响应）",
    description=(
        "发送一条消息，等待 LLM 生成完整回复后一次性返回。\n\n"
        "适用场景：对响应延迟不敏感，需要完整结果再处理的场景（如后端服务调用）。\n\n"
        "对延迟敏感的前端页面请使用 `POST /chat/stream` 流式接口。"
    ),
)
async def chat(
    request: Request,
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ApiResponse[ChatResponse]:
    """
    非流式多轮对话主接口。

    处理流程：
      1. 从 Redis 加载 session_id 对应的历史消息
      2. 构造「系统提示词 + 历史 + 当前消息」完整消息列表
      3. 通过 LLM Router 路由到合适模型并调用
      4. 将本轮 Human + AI 消息写入 Redis（TTL=2h）
      5. 返回 ApiResponse[ChatResponse]

    路由策略（strategy 参数）：
      - QUALITY  → gpt-4o，适合复杂分析
      - BALANCED → gpt-4o-mini，日常对话默认选项
      - ECONOMY  → gpt-3.5-turbo，成本优先场景

    错误处理：
      - LLM 主模型与 fallback 均失败 → 502
      - 其他服务端异常 → 500
    """
    trace_id = _get_trace_id(request)

    logger.info(
        "收到聊天请求（非流式）",
        extra={
            "session_id": body.session_id,
            "message_preview": body.message[:50],
            "strategy": body.strategy,
            "trace_id": trace_id,
        },
    )

    try:
        answer, model_used = await service.chat(
            session_id=body.session_id,
            user_message=body.message,
            strategy=body.strategy,
            system_prompt=body.system_prompt,
        )
    except LLMError as exc:
        logger.error(
            "LLM 调用失败",
            extra={"session_id": body.session_id, "error": str(exc), "trace_id": trace_id},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI 服务暂时不可用：{exc.message}",
        )
    except Exception as exc:
        logger.error(
            "聊天接口未预期异常",
            extra={"session_id": body.session_id, "trace_id": trace_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务内部错误，请稍后重试",
        )

    return ApiResponse.ok(
        data=ChatResponse(
            answer=answer,
            session_id=body.session_id,
            # Token 统计由 llm_client 写入日志，此处暂以 0 占位
            # Step 24 CostService 完成后可从 cost_service 读取精确值
            tokens_used=0,
            model=model_used,
        ),
        message="对话成功",
        trace_id=trace_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/stream — 流式多轮对话（SSE）
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/stream",
    summary="多轮对话（SSE 流式输出）",
    description=(
        "发送一条消息，以 Server-Sent Events 格式逐 Token 流式推送回复。\n\n"
        "**SSE 帧格式**（每帧含 `delta` 增量文本）：\n"
        "```\n"
        "data: {\"delta\": \"你好\", \"done\": false}\n\n"
        "data: {\"delta\": \"，\", \"done\": false}\n\n"
        "data: {\"delta\": \"\", \"done\": true, \"session_id\": \"sess_001\"}\n\n"
        "```\n\n"
        "客户端监听 `done: true` 帧后即可关闭连接。\n\n"
        "若 LLM 调用出错，最后一帧携带 `error` 字段。"
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE 流式文本输出",
        }
    },
)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    流式多轮对话接口。

    设计要点（「水龙头」类比）：
      - 打开龙头（建立 SSE 连接）后，Token 像水流一样持续输出
      - 关闭龙头（done=true 终止帧）后，客户端停止接收
      - 历史消息在流结束后才整体写入 Redis，保证会话完整性

    错误处理：
      - LLM 调用失败时，推送携带 error 字段的终止帧，而非 HTTP 错误码
        （因为 SSE 连接已建立，HTTP 状态码无法在流中途更改）
    """
    trace_id = _get_trace_id(request)

    logger.info(
        "收到聊天请求（流式）",
        extra={
            "session_id": body.session_id,
            "message_preview": body.message[:50],
            "strategy": body.strategy,
            "trace_id": trace_id,
        },
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        SSE 事件生成器：将 chat_service 的 AsyncGenerator[str] 转换为 SSE 帧序列。

        流程：
          1. 调用 service.chat_stream() 获取 token 生成器
          2. 每个 token chunk → 包装为 StreamChunk(delta=chunk, done=False) → SSE 帧
          3. 所有 token 推送完毕 → 推送终止帧 StreamChunk(done=True, session_id=...)
          4. 异常时 → 推送携带 error 字段的终止帧
        """
        try:
            async for chunk in service.chat_stream(
                session_id=body.session_id,
                user_message=body.message,
                strategy=body.strategy,
                system_prompt=body.system_prompt,
            ):
                # 空 chunk 跳过（部分模型会推送空字符串）
                if not chunk:
                    continue

                yield _sse_frame(StreamChunk(delta=chunk, done=False))

            # 正常结束：推送携带 session_id 的终止帧
            yield _sse_frame(
                StreamChunk(
                    delta="",
                    done=True,
                    session_id=body.session_id,
                )
            )

        except LLMError as exc:
            logger.error(
                "流式 LLM 调用失败",
                extra={
                    "session_id": body.session_id,
                    "error": str(exc),
                    "trace_id": trace_id,
                },
            )
            # 错误终止帧：通知客户端发生错误
            yield _sse_frame(
                StreamChunk(
                    delta="",
                    done=True,
                    error=f"AI 服务暂时不可用：{exc.message}",
                )
            )

        except Exception as exc:
            logger.error(
                "流式聊天未预期异常",
                extra={"session_id": body.session_id, "trace_id": trace_id},
                exc_info=True,
            )
            yield _sse_frame(
                StreamChunk(
                    delta="",
                    done=True,
                    error="服务内部错误，请稍后重试",
                )
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # 禁用所有中间缓冲，确保 Token 实时推送到客户端
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 缓冲
            "Connection":      "keep-alive",
            # 透传 TraceID，方便客户端关联日志
            "X-Trace-ID":      trace_id,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /chat/session/{session_id} — 清空会话历史
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/session/{session_id}",
    response_model=EmptyResponse,
    status_code=status.HTTP_200_OK,
    summary="清空会话历史",
    description=(
        "删除 Redis 中指定 session_id 的全部历史消息。\n\n"
        "清空后，该会话将重新开始（下次对话不再携带历史上下文）。\n\n"
        "若 session_id 不存在，仍返回成功（幂等操作）。"
    ),
)
async def clear_session(
    session_id: str,
    request: Request,
    service: ChatService = Depends(get_chat_service),
) -> EmptyResponse:
    """
    清空指定会话的历史记录。

    幂等设计：session_id 不存在时不报错，直接返回成功。
    这符合「删除操作天然幂等」的 REST 语义。
    """
    trace_id = _get_trace_id(request)

    logger.info(
        "清空会话历史",
        extra={"session_id": session_id, "trace_id": trace_id},
    )

    try:
        existed = await service.clear_session(session_id)
    except Exception as exc:
        logger.error(
            "清空会话失败",
            extra={"session_id": session_id, "trace_id": trace_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清空会话失败，请稍后重试",
        )

    message = "会话历史已清空" if existed else "会话不存在或已过期，无需清空"
    return EmptyResponse.ok(message=message)


# ─────────────────────────────────────────────────────────────────────────────
# GET /chat/session/{session_id}/info — 查询会话元信息
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/session/{session_id}/info",
    response_model=ApiResponse[SessionInfoResponse],
    status_code=status.HTTP_200_OK,
    summary="查询会话元信息",
    description=(
        "返回指定会话的元信息：\n\n"
        "- `message_count`：当前历史消息条数（Human + AI 各计 1 条）\n"
        "- `ttl_seconds`：Redis Key 剩余存活时间（秒）\n"
        "  - `-1`：Key 无过期时间\n"
        "  - `-2`：Key 不存在（会话已过期或从未创建）\n\n"
        "适用场景：前端展示「对话轮次」或判断会话是否仍然有效。"
    ),
)
async def get_session_info(
    session_id: str,
    request: Request,
    service: ChatService = Depends(get_chat_service),
) -> ApiResponse[SessionInfoResponse]:
    """
    查询会话元信息（不读取消息内容，只返回计数与 TTL）。

    设计意图（「存单查询」类比）：
      银行存单不暴露金额明细，但可以查「还有多少天到期」和「共存了几笔」。
    """
    trace_id = _get_trace_id(request)

    try:
        info = await service.get_session_info(session_id)
    except Exception as exc:
        logger.error(
            "查询会话信息失败",
            extra={"session_id": session_id, "trace_id": trace_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="查询会话信息失败，请稍后重试",
        )

    return ApiResponse.ok(
        data=SessionInfoResponse(
            session_id=info["session_id"],
            message_count=info["message_count"],
            ttl_seconds=info["ttl_seconds"],
        ),
        message="获取成功",
        trace_id=trace_id,
    )
