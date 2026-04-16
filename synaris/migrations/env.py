"""
@File       : env.py  (migrations/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Alembic 异步迁移环境配置。
@Features:
  - 同时支持「离线模式」（仅生成 SQL 脚本）和「在线模式」（直接执行迁移）
  - 在线模式使用 asyncpg 异步引擎（与应用层共用同一套连接配置）
  - 自动导入所有 ORM 模型（确保 autogenerate 能检测到全部表变更）
  - target_metadata 指向 Base.metadata，驱动 autogenerate 对比

  常用 Alembic 命令：
    alembic revision --autogenerate -m "add user table"  # 自动生成迁移脚本
    alembic upgrade head      # 升级到最新版本
    alembic downgrade -1      # 回滚一个版本
    alembic history           # 查看迁移历史
    alembic current           # 查看当前版本

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── 导入所有 ORM 模型（autogenerate 必须能看到所有 Table）────────────────────
# 顺序：先导入 Base，再导入所有依赖 Base 的模型
# 若漏导入某个模型，autogenerate 就不会检测到对应表的变更
from models.base import Base          # noqa: F401 — 必须导入，包含 DeclarativeBase

# 逐一导入所有模型，确保 Base.metadata 已注册全部表
from models.user           import User, APIKey           # noqa: F401
from models.session        import ChatSession            # noqa: F401
from models.task           import AgentTask              # noqa: F401
from models.prompt_version import PromptVersion         # noqa: F401
from models.eval_run       import EvaluationRun         # noqa: F401

# ── Alembic Config ─────────────────────────────────────────────────────────────
config = context.config

# 从 alembic.ini 读取日志配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 告知 Alembic 哪个 metadata 代表「目标状态」（数据库应该长什么样）
target_metadata = Base.metadata


# ─────────────────────────────────────────────
# 1. DSN 解析（从环境变量读取，与应用层一致）
# ─────────────────────────────────────────────

def _get_db_url() -> str:
    """
    优先从 alembic.ini 的 sqlalchemy.url 读取；
    若未设置则从环境变量拼装（与 postgres_client.py 逻辑保持一致）。

    Alembic 迁移使用同步引擎（psycopg2），
    但在线模式通过 async_engine_from_config 使用异步引擎（asyncpg）。

    注意：.ini 文件中的 %(DB_URL)s 插值写法会读取同名环境变量。
    """
    # 优先读 alembic.ini 中的配置
    url = config.get_main_option("sqlalchemy.url")
    if url and url != "driver://user:pass@localhost/dbname":
        # 确保使用 asyncpg 驱动前缀（离线模式例外）
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # 从环境变量拼装（CI / Docker 环境）
    host     = os.getenv("POSTGRES_HOST",     "localhost")
    port     = os.getenv("POSTGRES_PORT",     "5432")
    db       = os.getenv("POSTGRES_DB",       "synaris")
    user     = os.getenv("POSTGRES_USER",     "synaris")
    password = os.getenv("POSTGRES_PASSWORD", "synaris_dev")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


# ─────────────────────────────────────────────
# 2. 离线模式（生成 SQL 脚本，不连接数据库）
# ─────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    离线迁移模式：只生成 SQL 语句，不连接数据库。

    使用场景：
      - 生成用于 DBA 审查的 SQL 文件（alembic upgrade head --sql > migration.sql）
      - 在无数据库连接的 CI 环境中验证迁移脚本语法

    命令：
        alembic upgrade head --sql
    """
    # 离线模式使用同步 DSN（不需要 asyncpg）
    url = _get_db_url().replace("postgresql+asyncpg://", "postgresql://", 1)

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # 比较选项：精确检测列类型、服务器默认值、索引变更
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ─────────────────────────────────────────────
# 3. 在线模式（直接执行迁移）
# ─────────────────────────────────────────────

def do_run_migrations(connection: Connection) -> None:
    """在已建立的同步连接上运行迁移（由 run_migrations_online 调用）。"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # 事务性 DDL：PostgreSQL 支持将 DDL 包在事务中，迁移失败可整体回滚
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    异步迁移：使用 asyncpg 驱动连接 PostgreSQL，在事件循环中执行迁移。

    为什么用异步引擎？
    - 与应用层共用 asyncpg 驱动，保持技术栈统一
    - 避免同时安装 psycopg2（同步）和 asyncpg（异步）两个驱动

    NullPool 的原因：
    - Alembic 迁移是短生命周期操作，不需要连接池
    - 迁移完成后立即释放连接，不留残留连接
    """
    db_url = _get_db_url()

    # 将 DSN 注入 Alembic config（覆盖 .ini 文件中可能存在的占位符）
    config.set_main_option("sqlalchemy.url", db_url)

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,    # 短连接，用完即销毁
    )

    async with connectable.connect() as connection:
        # run_sync 让异步连接以同步方式执行 do_run_migrations
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    在线迁移入口：启动事件循环并执行异步迁移。

    Alembic 的 env.py 本身是同步的，但我们需要异步引擎，
    通过 asyncio.run() 桥接两个世界。
    """
    asyncio.run(run_async_migrations())


# ─────────────────────────────────────────────
# 4. 入口分发（Alembic 自动调用）
# ─────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
