"""
@File       : prompt_version.py  (models/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Prompt 版本管理 ORM 模型。
@Features:
  - PromptVersion 模型：管理所有提示词的版本历史
      · version 字段：语义版本（Semver，如 1.0.0 / 1.1.0-beta）
      · variables JSONB：声明该 Prompt 需要的变量列表及描述
      · is_active：当前生效版本标记（同一 name 只能有一个 is_active=True）
      · ab_test_weight：A/B 测试权重（0-100，Step 25 prompt_version_service 使用）
      · tags：标签列表（如 ["rag", "production", "v3"]）
  - 唯一约束：(name, version) 联合唯一，防止同名同版本重复创建
  - 辅助方法：parse_variables()（将 JSONB 字段反序列化为 Python dict）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Boolean, Float, ForeignKey, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base


class PromptVersion(Base):
    """
    Prompt 版本管理表（prompt_versions）。

    类比「软件版本发布系统」：
    ┌──────────────────────────────────────────────────────────┐
    │  name="rag_generate"   → Prompt 的「包名」（不变）        │
    │  version="1.0.0"       → 版本号（每次修改递增）           │
    │  content               → Prompt 正文（可含 {variable}）   │
    │  is_active=True        → 当前生效版本（同名只能有一个）    │
    │  ab_test_weight=50     → A/B 测试时各版本的流量分配比例   │
    └──────────────────────────────────────────────────────────┘

    A/B 测试工作原理（Step 25 prompt_version_service 实现）：
      假设 name="rag_generate" 有两个 is_active=True 的版本：
        v1.0.0: ab_test_weight=70
        v1.1.0: ab_test_weight=30
      → 70% 的请求使用 v1.0.0，30% 使用 v1.1.0
      → 通过评估指标（相关性/忠实度）决定是否推广 v1.1.0

    ⚠️ 约束说明：
      (name, version) 联合唯一约束在数据库层保证同名同版本不重复。
      但「is_active 只有一个」是业务约束，由 Service 层保证（不设数据库约束，
      因为切换时需要先启用新版、再禁用旧版的中间态）。
    """

    __tablename__ = "prompt_versions"

    __table_args__ = (
        # 同一 name 下，version 必须唯一
        UniqueConstraint("name", "version", name="uq_prompt_name_version"),
        {"comment": "Prompt 版本管理表，支持版本历史、A/B 测试和一键回滚"},
    )

    # ── 标识字段 ──────────────────────────────────────────────────────────
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Prompt 名称（如 rag_generate / agent_planner），对应 PromptKey 枚举",
    )

    version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="语义版本号（如 1.0.0 / 2.1.0-beta），遵循 Semver 规范",
    )

    description: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="版本变更说明（改了什么、为什么改）",
    )

    # ── Prompt 内容 ───────────────────────────────────────────────────────
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Prompt 模板正文（可含 {variable_name} 占位符）",
    )

    variables: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "变量声明（JSONB）。格式：{variable_name: {type, description, required}}。"
            "示例：{\"context\": {\"type\": \"str\", \"description\": \"检索到的文档片段\"}}"
        ),
    )

    # ── 版本状态 ──────────────────────────────────────────────────────────
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
        index=True,
        comment="是否为当前生效版本（Service 层保证同 name 最多一个 True）",
    )

    # ── A/B 测试 ──────────────────────────────────────────────────────────
    ab_test_weight: Mapped[float] = mapped_column(
        Float,
        default=100.0,
        server_default=text("100.0"),
        nullable=False,
        comment=(
            "A/B 测试权重（0-100）。"
            "单个生效版本设为 100；多版本并行时按权重比例分流。"
            "示例：v1=70, v2=30 → 70% 流量走 v1"
        ),
    )

    # ── 审计字段 ──────────────────────────────────────────────────────────
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="创建此版本的用户 ID",
    )

    creator: Mapped[Optional["models.user.User"]] = relationship(  # type: ignore[name-defined]
        "User",
        foreign_keys=[created_by],
        lazy="select",
    )

    # ── 标签 ──────────────────────────────────────────────────────────────
    # PostgreSQL 原生 ARRAY 类型，支持 GIN 索引（可按标签快速过滤）
    tags: Mapped[Optional[list]] = mapped_column(
        ARRAY(String(64)),
        nullable=True,
        comment="标签列表（如 ['rag', 'production', 'v3']），用于分类和搜索",
    )

    # ── 评估指标（Step 25 eval_service 回填）─────────────────────────────
    eval_relevance_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="最新一次评估的相关性得分（0-1，由 eval_service 写入）",
    )

    eval_faithfulness_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="最新一次评估的忠实度得分（0-1，幻觉检测）",
    )

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    def get_variables(self) -> dict[str, dict]:
        """
        返回变量声明字典（若 variables 为 None 则返回空字典）。

        返回格式：
            {
              "context": {"type": "str", "description": "检索到的文档片段", "required": True},
              "question": {"type": "str", "description": "用户问题", "required": True},
            }
        """
        return self.variables or {}

    def required_variable_names(self) -> list[str]:
        """返回所有 required=True 的变量名列表（供 render 时做入参校验）。"""
        return [
            name
            for name, meta in self.get_variables().items()
            if meta.get("required", True)
        ]

    def render(self, **kwargs: str) -> str:
        """
        用传入的变量渲染 Prompt 模板（简单字符串替换）。

        对于复杂的 LangChain PromptTemplate 渲染，使用 prompt_version_service.py
        中的 render() 方法，此处仅作简单版本供快速使用。

        示例：
            pv.content = "请回答：{question}\n参考资料：{context}"
            pv.render(question="公司假期政策", context="...") 
            → "请回答：公司假期政策\n参考资料：..."
        """
        try:
            return self.content.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Prompt 变量缺失：{exc}。"
                f"需要变量：{self.required_variable_names()}"
            ) from exc
