"""
@File       : postgres_client.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: 异步 PostgreSQL 数据库客户端。
@Features:
  - asyncpg 驱动 + SQLAlchemy 2.0 异步引擎（async_sessionmaker）
  - 连接池配置：pool_size=10, max_overflow=20, pool_pre_ping=True
  - 三种使用方式：
      1. get_db_session()   — FastAPI Depends 依赖注入（自动提交/回滚）
      2. db_session()       — 异步上下文管理器（手动使用）
      3. get_engine()       — 直接获取 Engine（Alembic 迁移专用）
  - 连接健康检查：ping() → bool
  - 启动/关闭生命周期：init_db() / close_db()（在 main.py lifespan 中调用）
  - 全局异常转化：SQLAlchemy 异常 → AppException 体系

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# ── 模块级单例（在 init_db() 时初始化）────────────────────────────────────────
_engine:         AsyncEngine | None        = None
_session_factory: async_sessionmaker | None = None


# ─────────────────────────────────────────────
# 1. 引擎构建与生命周期
# ─────────────────────────────────────────────

def _build_dsn() -> str:
    """
    从 settings 构造 asyncpg DSN。

    SQLAlchemy 异步引擎需要 postgresql+asyncpg:// 前缀，
    与标准 postgresql:// 的区别在于驱动标识。
    """
    try:
        from config.settings import get_settings  # type: ignore[import]
        s = get_settings()
        # settings 中存储的是标准 DSN，这里替换 scheme
        dsn = getattr(s, "postgres_dsn", None) or (
            f"postgresql+asyncpg://{s.postgres_user}:{s.postgres_password}"
            f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"
        )
        # 确保使用 asyncpg 驱动前缀
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    except ImportError:
        # settings 未就绪时使用环境变量兜底
        import os
        host     = os.getenv("POSTGRES_HOST",     "localhost")
        port     = os.getenv("POSTGRES_PORT",     "5432")
        db       = os.getenv("POSTGRES_DB",       "synaris")
        user     = os.getenv("POSTGRES_USER",     "synaris")
        password = os.getenv("POSTGRES_PASSWORD", "synaris_dev")
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


async def init_db() -> None:
    """
    初始化数据库引擎和 Session 工厂。
    在 FastAPI lifespan 的「启动」阶段调用。

    连接池参数说明（类比「收银台窗口」）：
      pool_size=10       → 常驻 10 个「收银员」（始终保持的连接数）
      max_overflow=20    → 高峰期最多再开 20 个临时窗口（超出 pool_size 的扩容上限）
      pool_timeout=30    → 等不到空闲连接时，最多等 30 秒再报错
      pool_recycle=1800  → 连接使用超 30 分钟后强制回收，防止 PostgreSQL 超时断开
      pool_pre_ping=True → 每次取连接时先 ping 一下，自动剔除已断开的「僵尸连接」
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.warning("init_db: 引擎已初始化，跳过重复初始化")
        return

    dsn = _build_dsn()
    logger.info("init_db: 连接 PostgreSQL（DSN 已脱敏）")

    _engine = create_async_engine(
        dsn,
        echo=False,               # 生产环境关闭 SQL 回显；调试时改为 True
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        # asyncpg 连接参数
        connect_args={
            "server_settings": {
                "application_name": "synaris",
                "timezone": "UTC",
            }
        },
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,   # 提交后对象字段不失效，避免 lazy load 问题
        autocommit=False,
        autoflush=False,
    )

    # 验证连接可用
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("init_db: PostgreSQL 连接成功")
    except OperationalError as exc:
        logger.error("init_db: PostgreSQL 连接失败: %s", exc)
        raise


async def close_db() -> None:
    """
    关闭数据库引擎，释放所有连接池资源。
    在 FastAPI lifespan 的「关闭」阶段调用。
    """
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine         = None
        _session_factory = None
        logger.info("close_db: PostgreSQL 连接池已关闭")


def get_engine() -> AsyncEngine:
    """
    获取全局 AsyncEngine 实例。
    主要供 Alembic 异步迁移（migrations/env.py）使用。
    """
    if _engine is None:
        raise RuntimeError(
            "数据库引擎尚未初始化，请先调用 await init_db()。"
            "（在 FastAPI lifespan 或测试 fixture 中调用）"
        )
    return _engine


# ─────────────────────────────────────────────
# 2. Session 获取方式
# ─────────────────────────────────────────────

@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    异步上下文管理器（手动使用场景）。

    自动处理提交/回滚，确保异常时事务不残留。

    使用方式：
        async with db_session() as session:
            result = await session.execute(select(User).where(User.id == uid))
            user = result.scalar_one_or_none()
    """
    if _session_factory is None:
        raise RuntimeError("数据库未初始化，请先调用 await init_db()")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.error("db_session: 事务回滚 | error=%s", exc)
            raise
        except Exception:
            await session.rollback()
            raise


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Depends 依赖注入版本。

    使用方式：
        @router.get("/users")
        async def list_users(session: AsyncSession = Depends(get_db_session)):
            ...

    工作原理：
        FastAPI 会在每个请求开始时调用此生成器，yield 提供 session，
        请求结束后继续执行 yield 之后的 finally 块，自动关闭 session。
    """
    if _session_factory is None:
        raise RuntimeError("数据库未初始化，请先调用 await init_db()")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.error("get_db_session: 请求事务回滚 | error=%s", exc)
            # 转化为业务异常，避免把 SQLAlchemy 内部细节暴露给 API 调用方
            try:
                from core.exceptions import AppException, ErrorCode  # type: ignore[import]
                raise AppException(
                    code=ErrorCode.DB_ERROR,
                    message="数据库操作失败",
                    detail=str(exc),
                ) from exc
            except ImportError:
                raise
        except Exception:
            await session.rollback()
            raise


# ─────────────────────────────────────────────
# 3. 健康检查
# ─────────────────────────────────────────────

async def ping() -> bool:
    """
    PostgreSQL 连接健康检查。
    供 GET /health/detailed 接口使用。

    Returns:
        True  → 数据库连接正常
        False → 连接失败（会记录错误日志）
    """
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("postgres ping 失败: %s", exc)
        return False
