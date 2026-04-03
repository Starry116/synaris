"""
@File       : base.py  (models/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: SQLAlchemy ORM 公共基类，所有数据模型的「地基」。
@Features:
  - DeclarativeBase：SQLAlchemy 2.0 声明式基类（替换旧的 declarative_base()）
  - TimestampMixin：created_at / updated_at 自动时间戳（server_default + onupdate）
  - SoftDeleteMixin：is_deleted 软删除标记（不做物理删除，保留审计轨迹）
  - Base：组合以上 Mixin 的最终基类，所有模型直接继承 Base

  「地基」类比：
    TimestampMixin  → 自动记录「谁、什么时候盖的楼」
    SoftDeleteMixin → 拆迁时打「废弃」标记，档案仍在，不销毁记录
    Base            → 整栋楼的地基图纸，每个房间（模型）按此规格建造

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedColumn, mapped_column


# ─────────────────────────────────────────────
# 1. SQLAlchemy 2.0 声明式基类
# ─────────────────────────────────────────────

class DeclarativeBase_(DeclarativeBase):
    """
    SQLAlchemy 2.0 的声明式基类。

    与 SQLAlchemy 1.x 的 declarative_base() 等价，
    但支持 PEP 681 的 Mapped[T] 类型注解语法。
    """
    pass


# ─────────────────────────────────────────────
# 2. Mixin：自动时间戳
# ─────────────────────────────────────────────

class TimestampMixin:
    """
    自动管理创建时间和更新时间的 Mixin。

    设计细节：
      created_at → server_default=func.now()：由 PostgreSQL 服务器在 INSERT 时填入，
                   避免客户端时区不一致问题。
      updated_at → onupdate=func.now()：每次 UPDATE 时由 ORM 自动更新。
                   注意：批量 UPDATE（session.execute(update(...))）不会触发此 hook，
                   需要手动在 WHERE 子句中 SET updated_at=now()。
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="记录创建时间（UTC，由数据库服务器填入）",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="记录最后更新时间（UTC，ORM 自动维护）",
    )


# ─────────────────────────────────────────────
# 3. Mixin：软删除
# ─────────────────────────────────────────────

class SoftDeleteMixin:
    """
    软删除 Mixin。

    所有继承此 Mixin 的模型，「删除」操作只将 is_deleted 标记为 True，
    而不做物理行删除，保证数据可追溯、可恢复。

    使用规范：
      - 查询时应在 WHERE 子句中过滤 is_deleted=False
      - 推荐在 Service 层统一处理，不要在每个查询处手写
      - 真正需要物理删除时（如 GDPR 数据擦除），使用 session.delete() 明确标注

    删除示例：
        user.is_deleted = True
        user.updated_at = datetime.now(timezone.utc)
        await session.commit()
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
        index=True,    # 几乎所有查询都需要过滤此字段，加索引提升性能
        comment="软删除标记（True=已删除，查询时需过滤）",
    )


# ─────────────────────────────────────────────
# 4. 最终组合基类
# ─────────────────────────────────────────────

class Base(DeclarativeBase_, TimestampMixin, SoftDeleteMixin):
    """
    Synaris 所有 ORM 模型的统一基类。

    包含：
      - SQLAlchemy 2.0 声明式基类能力
      - 自动时间戳（created_at / updated_at）
      - 软删除（is_deleted）
      - id 字段使用 PostgreSQL 原生 UUID 类型（由数据库生成，性能优于应用层生成）

    继承方式：
        class User(Base):
            __tablename__ = "users"
            id: Mapped[uuid.UUID] = mapped_column(
                UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
            )
            ...
    """

    __abstract__ = True    # 告知 SQLAlchemy 此类本身不对应任何表

    # ── 主键：PostgreSQL 原生 UUID ──────────────────────────────────────────
    # server_default=gen_random_uuid()：由 PostgreSQL 生成，无需应用层传入
    # as_uuid=True：Python 侧自动转为 uuid.UUID 对象
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        comment="主键（UUID v4，由 PostgreSQL gen_random_uuid() 生成）",
    )

    def to_dict(self) -> dict:
        """
        将 ORM 对象转为字典（仅包含已加载的列，不触发懒加载）。
        主要用于日志记录和调试，不建议在 API 响应中直接使用。
        """
        return {
            col.key: getattr(self, col.key)
            for col in self.__table__.columns
            if hasattr(self, col.key)
        }

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        pk = getattr(self, "id", "?")
        return f"<{cls_name} id={pk}>"
