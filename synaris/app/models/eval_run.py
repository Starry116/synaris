"""
@File       : eval_run.py  (models/)
@Author     : Codex
@Created    : 2026-04-16
@Version    : 1.0.0
@Description: LLM 评估运行记录 ORM 模型。
@Features:
  - EvaluationRun：保存一次完整评估批次的汇总指标与逐条结果
  - 支持按 prompt_name / prompt_version_id 回溯历史趋势
  - results(JSONB) 保存逐 case 评分明细，summary(JSONB) 保存扩展统计
  - 与 PromptVersion 建立可选关联，便于版本效果对比
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base


class EvaluationRun(Base):
    """
    评估运行记录表（evaluation_runs）。

    一条记录代表一次完整的回归评估批次，而不是单条样本。
    这样可以直接用于看趋势：
      - 某个 Prompt 最近 10 次评估是否持续上升
      - 某个版本切流前后 faithfulness 是否改善
      - 某个数据集在不同 judge model 下的结果差异
    """

    __tablename__ = "evaluation_runs"

    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="评估运行名称，如 rag_regression / prompt_ab_eval",
    )

    eval_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="regression",
        comment="评估类型：regression / online_sample / ab_test 等",
    )

    prompt_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="关联的 Prompt 名称（如 rag_generate）",
    )

    prompt_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="关联的 PromptVersion 主键，便于精确追踪版本效果",
    )

    judge_model: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="执行 LLM-as-judge 的模型名称，如 gpt-4o-mini",
    )

    dataset_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="本次评估的数据集样本数",
    )

    completed_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="成功完成评分的样本数",
    )

    failed_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="评分失败的样本数（如模型异常、JSON 解析失败）",
    )

    accuracy: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="达标率（通过样本数 / 总样本数）",
    )

    average_relevance: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="平均相关性得分（0-1）",
    )

    average_faithfulness: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="平均忠实度得分（0-1）",
    )

    average_quality: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="平均综合质量得分（0-1）",
    )

    average_hallucination_rate: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="平均幻觉率（1-faithfulness）",
    )

    p95_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="逐样本评估耗时的 P95（毫秒）",
    )

    results: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        comment="逐样本评估结果列表（case_name / scores / latency / error 等）",
    )

    summary: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="扩展汇总信息（如阈值、额外指标、维度分解）",
    )

    extra_payload: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="附加元数据（数据集标签、调用来源、实验分组等）",
    )

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="备注信息，用于记录本次评估背景或异常说明",
    )

    prompt_version: Mapped[Optional["models.prompt_version.PromptVersion"]] = relationship(  # type: ignore[name-defined]
        "PromptVersion",
        lazy="select",
    )
