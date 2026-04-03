"""
@File       : session.py  (models/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: 会话记录 ORM 模型。
@Features:
  - ChatSession 模型：记录每个用户的对话会话元数据
      · session_type 枚举：chat / rag / agent / multi_agent
      · metadata JSONB：灵活存储会话相关扩展信息
      · 关联 User（多对一）
  - 设计目标：
      · 会话 ID（session_id）在 Redis 中存储实际历史消息
      · PostgreSQL 只存会话元数据（创建时间/类型/归属用户）
      · 支持后续的会话列表查询和管理

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base


# ─────────────────────────────────────────────
# 1. 会话类型枚举
# ─────────────────────────────────────────────

class SessionType(str, Enum):
    """
    会话类型枚举，标识本次会话的业务场景。

    对应三大核心功能模块：
      CHAT        → AI 聊天（api/chat.py）
      RAG         → 知识库问答（api/rag.py）
      AGENT       → 单 Agent 任务（api/agent.py，AgentMode.SINGLE）
      MULTI_AGENT → 多 Agent 协作（api/agent.py，AgentMode.MULTI）
    """
    CHAT        = "chat"
    RAG         = "rag"
    AGENT       = "agent"
    MULTI_AGENT = "multi_agent"


# ─────────────────────────────────────────────
# 2. ChatSession 模型
# ─────────────────────────────────────────────

class ChatSession(Base):
    """
    会话元数据表（chat_sessions）。

    类比「档案封面」：
      - 封面记录「这个档案属于谁、什么类型、什么时候建立的」
      - 档案内容（消息历史）存在 Redis（热数据）中，不在此表

    与 Redis 的分工：
      Redis key: chat:{session_id}  → 存储最近 20 条消息（TTL=2h）
      PostgreSQL chat_sessions 表   → 存储会话元数据（永久，软删除）

    典型查询场景：
      - 查询用户的历史会话列表（GET /chat/sessions）
      - 按类型过滤（只看 Agent 任务的会话）
      - 统计用户活跃度（按 created_at 聚合）
    """

    __tablename__ = "chat_sessions"

    # ── 关联用户 ───────────────────────────────────────────────────────────
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,    # 允许匿名会话（未登录用户也可使用聊天功能）
        index=True,
        comment="所属用户 ID（NULL=匿名会话）",
    )

    user: Mapped[Optional["models.user.User"]] = relationship(  # type: ignore[name-defined]
        "User",
        foreign_keys=[user_id],
        lazy="select",
    )

    # ── 会话标识 ──────────────────────────────────────────────────────────
    # session_id 与 Redis key 中的 session_id 一一对应
    # 使用 String 而非 UUID，因为 Redis 中用字符串格式（sess-xxxxxxxx）
    session_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
        index=True,
        comment="会话唯一标识符（与 Redis 中的 key 对应）",
    )

    # ── 会话元数据 ────────────────────────────────────────────────────────
    session_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=SessionType.CHAT.value,
        comment="会话类型：chat / rag / agent / multi_agent",
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="会话标题（由第一条消息自动生成，或用户手动命名）",
    )

    # metadata 存储会话的扩展信息，使用 Text（JSON 字符串）保持跨 DB 兼容
    # 典型内容：{"model": "gpt-4o", "collection": "hr_policies", "task_id": "..."}
    session_metadata: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="会话扩展元数据（JSON 字符串：model/collection/task_id 等）",
    )

    # ── 统计 ─────────────────────────────────────────────────────────────
    message_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="本次会话的消息总数（每次追加消息时 +1）",
    )

    total_tokens: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="本次会话累计消耗的 Token 数",
    )

    @property
    def session_type_enum(self) -> SessionType:
        return SessionType(self.session_type)
