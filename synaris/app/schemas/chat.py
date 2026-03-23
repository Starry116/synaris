"""
@File       : chat.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 聊天模块 Pydantic 请求/响应模型。
@Features:
  - ChatRequest       POST /chat 与 POST /chat/stream 的统一入参
  - ChatResponse      非流式响应的业务数据载荷
  - StreamChunk       SSE 流式推送的单帧数据结构
  - SessionClearRequest  清空会话历史的入参
  - SessionInfoResponse  会话元信息响应（消息数量 / 创建时间）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import Field, field_validator

from app.core.llm_router import RoutingStrategy
from app.schemas.base import SynarisBaseModel


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatRequest(SynarisBaseModel):
    """
    POST /chat 与 POST /chat/stream 共用的请求体。

    字段说明：
      session_id  - 会话唯一标识符，不传时自动生成 UUID，
                    同一 session_id 的请求共享历史记录（Redis TTL=2h）
      message     - 用户当前输入，1~4000 字符
      strategy    - 路由策略：QUALITY / BALANCED / ECONOMY，默认 BALANCED
      system_prompt - 可选的会话级系统提示词覆盖，None 时使用 prompts.py 默认值
    """

    session_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="会话 ID，不传则自动生成",
        examples=["a1b2c3d4e5f6"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户消息内容",
        examples=["你好，请介绍一下 RAG 技术"],
    )
    strategy: RoutingStrategy = Field(
        default=RoutingStrategy.BALANCED,
        description="路由策略：quality / balanced / economy",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="会话级系统提示词（覆盖默认值），None 时使用系统默认",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("消息内容不能为空白字符串")
        return v.strip()


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------


class ChatResponse(SynarisBaseModel):
    """
    POST /chat 非流式响应的业务数据载荷（嵌套在 ApiResponse.data 中）。

    字段说明：
      answer       - LLM 生成的完整回复文本
      session_id   - 当前会话 ID（方便客户端存储，用于后续请求）
      tokens_used  - 本次调用消耗的 Token 总量（0 表示统计不可用）
      model        - 实际使用的模型名称（可能为 fallback 模型）
    """

    answer: str = Field(..., description="AI 回复内容")
    session_id: str = Field(..., description="当前会话 ID")
    tokens_used: int = Field(0, ge=0, description="本次消耗 Token 总量")
    model: str = Field(..., description="实际使用的模型名称")


class StreamChunk(SynarisBaseModel):
    """
    SSE 流式推送的单帧 JSON 数据结构。

    每帧通过 text/event-stream 格式推送：
        data: {"delta": "...", "done": false}\n\n

    字段说明：
      delta    - 本帧文本增量片段（done=true 时为空字符串）
      done     - 是否为终止帧（true 表示流已结束）
      session_id - 仅在 done=true 的终止帧中携带，方便客户端关联
      error    - 若流中途出错，done=true 且 error 携带错误描述
    """

    delta: str = Field("", description="文本增量片段")
    done: bool = Field(False, description="是否为终止帧")
    session_id: Optional[str] = Field(None, description="终止帧中携带的会话 ID")
    error: Optional[str] = Field(None, description="错误描述（仅错误终止帧使用）")


# ---------------------------------------------------------------------------
# 会话管理辅助模型
# ---------------------------------------------------------------------------


class SessionClearRequest(SynarisBaseModel):
    """POST /chat/session/{session_id}/clear 的请求体（当前无额外字段，预留扩展）。"""

    pass


class SessionInfoResponse(SynarisBaseModel):
    """
    GET /chat/session/{session_id}/info 的响应载荷。

    字段说明：
      session_id      - 会话 ID
      message_count   - 当前历史消息条数（含 Human + AI 消息）
      ttl_seconds     - Redis 中剩余 TTL（秒），-1 表示无过期，-2 表示 Key 不存在
    """

    session_id: str
    message_count: int = Field(0, ge=0, description="历史消息条数")
    ttl_seconds: int = Field(-2, description="Redis TTL（秒）")
