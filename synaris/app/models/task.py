"""
@File       : task.py  (models/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: 异步任务记录 ORM 模型。
@Features:
  - AgentTask 模型：持久化记录所有 Celery / asyncio 异步任务的完整生命周期
      · 与 Redis 的分工：Redis 存实时状态（TTL=3天），PostgreSQL 存永久归档
      · input_data / result：JSONB 字段，存储任意结构的入参和结果
      · progress：0-100 整数，供前端进度条使用
      · started_at / finished_at：精确记录实际执行时间（区别于 created_at 提交时间）
  - TaskType 枚举：文档处理 / Agent 单任务 / 多 Agent / RAG 批量
  - 辅助方法：duration_seconds（计算执行耗时）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base


# ─────────────────────────────────────────────
# 1. 任务类型枚举
# ─────────────────────────────────────────────

class TaskType(str, Enum):
    """
    异步任务类型枚举。

    类比「快递单的寄件类型」，决定由哪个 Worker 处理：
      DOCUMENT_PROCESS → document_worker.py（文档解析/Embedding/入库）
      AGENT_SINGLE     → agent_worker.py（LangGraph 单 Agent 工作流）
      AGENT_MULTI      → agent_worker.py（Supervisor + Workers 多 Agent）
      RAG_BATCH        → document_worker.py（批量 RAG 评估）
    """
    DOCUMENT_PROCESS = "document_process"
    AGENT_SINGLE     = "agent_single"
    AGENT_MULTI      = "agent_multi"
    RAG_BATCH        = "rag_batch"


class TaskStatusEnum(str, Enum):
    """
    任务状态枚举（与 agents/state.py 的 TaskStatus 对应，但独立定义避免循环依赖）。
    PostgreSQL 层不直接依赖 Agent 层的类型。
    """
    PENDING       = "pending"
    RUNNING       = "running"
    WAITING_HUMAN = "waiting_human"
    COMPLETED     = "completed"
    FAILED        = "failed"
    CANCELLED     = "cancelled"


# ─────────────────────────────────────────────
# 2. AgentTask 模型
# ─────────────────────────────────────────────

class AgentTask(Base):
    """
    异步任务记录表（agent_tasks）。

    数据流设计（类比「快递追踪系统」）：
    ┌─────────────────────────────────────────────────────────────┐
    │  POST /agent/run                                            │
    │    → 创建 AgentTask 记录（status=pending）                  │
    │    → 写入 Redis agent:status:{task_id}（实时状态）          │
    │    → 启动 Celery 任务                                       │
    │                                                             │
    │  Celery Worker 执行中                                       │
    │    → 更新 Redis progress（0→100）                           │
    │    → Pub/Sub 推送 AgentStepEvent 给 WebSocket              │
    │                                                             │
    │  任务完成/失败                                               │
    │    → 写入 PostgreSQL result（永久归档）                      │
    │    → Redis TTL 到期后自动清理（3天）                         │
    └─────────────────────────────────────────────────────────────┘

    JSONB 字段的使用：
      input_data  → 存储任务入参（task描述/mode/config/session_id等）
      result      → 存储最终产出（final_answer/final_output/tool_results等）
      step_log    → 存储执行轨迹摘要（每个节点的 StepSummary 列表）
    """

    __tablename__ = "agent_tasks"

    # ── 关联用户 ───────────────────────────────────────────────────────────
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="提交任务的用户 ID",
    )

    # ── 任务标识 ──────────────────────────────────────────────────────────
    # task_id 与 Redis key agent:status:{task_id} 一一对应
    task_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
        index=True,
        comment="任务唯一标识符（格式：task-xxxxxxxx，与 Redis key 对应）",
    )

    session_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="关联的会话 ID（用于 Human-in-the-Loop 断点续跑）",
    )

    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Celery 任务 ID（用于 revoke 取消、inspect 状态查询）",
    )

    # ── 任务分类 ──────────────────────────────────────────────────────────
    task_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="任务类型：document_process / agent_single / agent_multi / rag_batch",
    )

    mode: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        comment="Agent 运行模式：single / multi / rag_only",
    )

    # ── 状态与进度 ────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=TaskStatusEnum.PENDING.value,
        index=True,
        comment="任务状态：pending / running / waiting_human / completed / failed / cancelled",
    )

    progress: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="任务进度（0-100 整数，文档处理场景使用）",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="失败时的错误信息（status=failed 时填写）",
    )

    # ── 时间记录 ──────────────────────────────────────────────────────────
    # created_at 继承自 Base（提交时间）
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Worker 实际开始执行的时间（区别于提交时间）",
    )

    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="任务完成/失败的时间",
    )

    # ── 数据载荷（JSONB）────────────────────────────────────────────────────
    input_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="任务入参（task/mode/config/session_id 等）",
    )

    result: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="任务结果（final_answer/tool_results/worker_results 等）",
    )

    step_log: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        comment="执行轨迹摘要（List[StepSummary]，供历史回溯使用）",
    )

    # ── 资源消耗 ──────────────────────────────────────────────────────────
    tokens_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="本次任务消耗的总 Token 数（用于成本核算）",
    )

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    @property
    def duration_seconds(self) -> Optional[float]:
        """计算任务实际执行耗时（秒）。started_at 或 finished_at 为 None 时返回 None。"""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def is_terminal(self) -> bool:
        """判断任务是否已处于终态（不会再变更状态）。"""
        return self.status in (
            TaskStatusEnum.COMPLETED.value,
            TaskStatusEnum.FAILED.value,
            TaskStatusEnum.CANCELLED.value,
        )

    def mark_started(self) -> None:
        """标记任务开始执行（由 Celery Worker 在任务开始时调用）。"""
        self.status     = TaskStatusEnum.RUNNING.value
        self.started_at = datetime.now(timezone.utc)
        self.progress   = 0

    def mark_completed(self, result: dict, tokens_used: int = 0) -> None:
        """标记任务成功完成（由 Celery Worker 在任务结束时调用）。"""
        self.status      = TaskStatusEnum.COMPLETED.value
        self.finished_at = datetime.now(timezone.utc)
        self.progress    = 100
        self.result      = result
        self.tokens_used = tokens_used

    def mark_failed(self, error: str) -> None:
        """标记任务失败（由 Celery Worker 的 on_failure 回调调用）。"""
        self.status        = TaskStatusEnum.FAILED.value
        self.finished_at   = datetime.now(timezone.utc)
        self.error_message = error[:2000]   # 防止超长错误信息
