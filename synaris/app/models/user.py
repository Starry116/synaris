"""
@File       : user.py  (models/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: 用户账号与 API Key ORM 模型。
@Features:
  - User 模型：账号体系核心表
      · 角色枚举：admin / member / viewer（三级权限）
      · 密码：bcrypt 哈希存储，明文永不入库
      · 关联：一用户 → 多个 APIKey（cascade 删除）
  - APIKey 模型：API Key 管理表
      · 存储 bcrypt 哈希值，明文仅在创建时返回一次
      · 支持过期时间（expires_at=None 表示永不过期）
      · 启用/禁用状态（is_active）
      · 最后使用时间（last_used_at，用于审计）
  - UserRole 枚举：统一的角色常量，供 auth.py 的 @require_role 装饰器使用

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, MappedColumn, mapped_column, relationship

from models.base import Base

if TYPE_CHECKING:
    pass    # 防止循环导入，类型注解仅在类型检查时引入


# ─────────────────────────────────────────────
# 1. 角色枚举
# ─────────────────────────────────────────────

class UserRole(str, Enum):
    """
    用户角色枚举。

    权限层级（从高到低）：
      admin  → 系统管理员：管理用户/API Key/Prompt 版本，访问所有接口
      member → 普通成员：使用全部 AI 功能（聊天/RAG/Agent），无管理权限
      viewer → 只读访问：仅可查询（RAG），不可提交 Agent 任务或上传文档
    """
    ADMIN  = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


# ─────────────────────────────────────────────
# 2. User 模型
# ─────────────────────────────────────────────

class User(Base):
    """
    用户账号表（users）。

    字段设计原则：
      - email 作为登录凭证（唯一索引）
      - username 用于显示（唯一索引，可修改）
      - hashed_password 只存哈希（bcrypt），明文不入库
      - preferences（JSONB）：扩展字段，存储用户偏好，Step 23 Memory Service 使用

    关联：
      api_keys → 一对多，cascade="all, delete-orphan"
                 用户删除时自动删除其所有 API Key
    """

    __tablename__ = "users"

    # ── 基础账号字段 ─────────────────────────────────────────────────────────
    username: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="用户名（唯一，可修改，用于展示）",
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="邮箱（唯一，作为登录凭证）",
    )

    hashed_password: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="bcrypt 哈希密码（明文不入库）",
    )

    role: Mapped[str] = mapped_column(
        String(20),
        default=UserRole.MEMBER.value,
        server_default=text(f"'{UserRole.MEMBER.value}'"),
        nullable=False,
        comment="用户角色：admin / member / viewer",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
        comment="账号是否启用（false=封禁状态）",
    )

    # ── 扩展字段（Step 23 Memory Service 回填）──────────────────────────────
    # preferences 存储用户偏好（语言/领域标签/沟通风格等）
    # 使用 Text 而非 JSONB 以保持跨数据库兼容性（Alembic 迁移更简单）
    # Step 23 完成后可用 JSONB 重建迁移
    preferences: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="用户偏好 JSON 字符串（Step 23 User Profile Memory 使用）",
    )

    # ── 关联 ─────────────────────────────────────────────────────────────────
    api_keys: Mapped[list["APIKey"]] = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",   # 用户删除时级联删除所有 API Key
        lazy="select",                  # 默认懒加载，避免不必要的 JOIN
    )

    @property
    def role_enum(self) -> UserRole:
        """将 role 字段字符串转为 UserRole 枚举（便于权限判断）。"""
        return UserRole(self.role)

    def has_role(self, required: UserRole) -> bool:
        """
        检查用户是否具备所需角色权限（包含高于所需角色的情况）。

        权限继承：admin > member > viewer
        示例：has_role(UserRole.MEMBER) 对 admin 用户也返回 True
        """
        role_levels = {
            UserRole.ADMIN:  3,
            UserRole.MEMBER: 2,
            UserRole.VIEWER: 1,
        }
        return role_levels.get(self.role_enum, 0) >= role_levels.get(required, 999)


# ─────────────────────────────────────────────
# 3. APIKey 模型
# ─────────────────────────────────────────────

class APIKey(Base):
    """
    API Key 管理表（api_keys）。

    安全设计（类比「门禁卡」）：
      - key_hash：存储 bcrypt 哈希值，原始 Key 只在创建时返回一次
      - key_prefix：存储 Key 前 8 位明文（如 sk-syn_ab12），用于列表展示时识别
      - 验证时：对请求携带的 Key 做 bcrypt.checkpw(raw, key_hash)
      - 即使数据库泄露，攻击者也无法从哈希值还原原始 Key

    生命周期：
      created_at → 创建时间（继承自 Base）
      expires_at → 过期时间（None=永不过期）
      last_used_at → 最后使用时间（每次成功认证时更新）
      is_active → 主动禁用（不等过期也可立即撤销）
    """

    __tablename__ = "api_keys"

    # ── 关联 ─────────────────────────────────────────────────────────────────
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属用户 ID",
    )

    user: Mapped["User"] = relationship("User", back_populates="api_keys")

    # ── Key 本体 ──────────────────────────────────────────────────────────────
    name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Key 名称（用途标注，如「生产环境」）",
    )

    key_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="API Key 的 bcrypt 哈希值（不存明文）",
    )

    key_prefix: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Key 前缀（如 sk-syn_ab12），用于列表展示识别，不可还原完整 Key",
    )

    # ── 有效期与状态 ──────────────────────────────────────────────────────────
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
        comment="是否启用（false=已撤销，即使未过期也不可用）",
    )

    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="过期时间（NULL=永不过期）",
    )

    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后一次成功认证的时间（用于审计与清理）",
    )

    # ── 使用统计 ──────────────────────────────────────────────────────────────
    request_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
        comment="累计请求次数（粗粒度统计，精细统计见 observability）",
    )

    def is_valid(self) -> bool:
        """
        检查 API Key 当前是否有效（启用 + 未过期）。

        在认证中间件中调用，避免散落的条件判断。
        """
        if not self.is_active:
            return False
        if self.expires_at is None:
            return True
        from datetime import timezone as tz
        return datetime.now(tz.utc) < self.expires_at
