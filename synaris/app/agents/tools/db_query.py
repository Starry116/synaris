"""
@File       : db_query.py
@Author     : Starry Hung
@Created    : 2026-04-11
@Version    : 1.0.0
@Description: Agent 结构化数据库查询工具（只读安全版）。

@Features:
  - @tool 装饰器：供 LangGraph ToolSelector 节点直接调用
  - 三层安全防线（从快到慢，逐层拦截）：
      1. 词法快检  : 关键字黑名单正则扫描（O(n)，微秒级，快速拒绝明显威胁）
      2. AST 语法校验 : sqlglot 解析 SQL 语法树，只允许 SELECT / WITH（CTE），
                        拒绝 DDL（CREATE/DROP/ALTER）、DML（INSERT/UPDATE/DELETE），
                        以及 CTE 内部隐藏写操作（WITH evil AS (INSERT ...) SELECT ...）
      3. 执行超时  : asyncio.wait_for 5 秒硬超时，防止慢查询阻塞 Agent
  - 查询结果限制：最多返回 100 行，超出时附加截断提示（blockquote 格式）
  - 输出格式：自动生成列宽自适应的 Markdown 表格，便于 LLM 阅读与引用
  - 多数据库路由：database 参数映射到 settings 中的连接别名
  - 降级策略：sqlglot 未安装时退化为增强关键字检查，功能可用但安全级别略低
  - 错误处理：校验拒绝 / 超时 / 运行时异常均返回中文结构化提示

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-11  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 执行限制常量 ───────────────────────────────────────────────────────────────
_MAX_ROWS    = 100    # 最多返回行数
_TIMEOUT_SEC = 5      # 查询超时（秒）
_MAX_SQL_LEN = 2000   # SQL 最大字符长度（防 DoS）
_MAX_CELL_LEN = 200   # 单元格内容最大字符数（防表格爆炸）

# ── SQL 关键字黑名单（第一层：词法快检） ─────────────────────────────────────
# 正则在模块加载时编译一次，后续复用，性能最优
_FORBIDDEN_KW = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|REPLACE|MERGE|UPSERT"
    r"|CREATE|DROP|ALTER|TRUNCATE|RENAME"
    r"|GRANT|REVOKE|DENY"
    r"|EXECUTE|EXEC|CALL"
    r"|LOAD\s+DATA|INTO\s+OUTFILE|INTO\s+DUMPFILE"
    r"|COPY\s+TO|COPY\s+FROM"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. 三层安全校验
# ---------------------------------------------------------------------------

class _SQLSafetyError(ValueError):
    """SQL 安全校验失败的专用异常，与其他 ValueError 精确区分。"""


def _layer1_keyword_check(sql: str) -> None:
    """
    第一层：关键字黑名单快检。

    类比：机场入口的金属探测门——有金属直接报警，
    不需要等到 X 光机才发现问题，速度最快。
    """
    match = _FORBIDDEN_KW.search(sql)
    if match:
        raise _SQLSafetyError(
            f"检测到禁止关键字「{match.group()}」。\n"
            f"db_query 只允许 SELECT 查询，禁止任何写操作或 DDL 语句。"
        )


def _layer2_ast_check(sql: str) -> None:
    """
    第二层：sqlglot AST 语法树精确校验。

    将 SQL 解析为完整语法树后遍历每个节点，确保：
      (a) 顶层语句只能是 Select 或 With（CTE + SELECT 组合）
      (b) 整棵树中不含任何写操作节点（防 CTE 注入攻击）

    类比：X 光机精检——即使外层是「SELECT」，
    如果 CTE 内部藏着「INSERT」也会被发现。

    降级：sqlglot 未安装时，退化为多语句分号检查 + 关键字重检。
    """
    try:
        import sqlglot                        # type: ignore[import]
        import sqlglot.expressions as exp     # type: ignore[import]
    except ImportError:
        logger.warning(
            "sqlglot 未安装，db_query 安全校验退化为关键字模式",
            extra={"hint": "pip install sqlglot"},
        )
        _layer2_fallback(sql)
        return

    try:
        statements = sqlglot.parse(sql)
    except Exception as exc:
        raise _SQLSafetyError(f"SQL 语法解析失败：{exc}") from exc

    if not statements or all(s is None for s in statements):
        raise _SQLSafetyError("无法解析 SQL 语句，请检查语法是否正确。")

    # 允许的顶层节点类型（白名单）
    _ALLOWED_TOP = (exp.Select, exp.With)
    # 禁止出现在树任何位置的写操作节点（黑名单）
    _WRITE_NODES = (
        exp.Insert, exp.Update, exp.Delete,
        exp.Create, exp.Drop, exp.AlterTable,
        exp.TruncateTable, exp.Grant, exp.Revoke,
    )

    for stmt in statements:
        if stmt is None:
            continue

        # (a) 顶层类型检查
        if not isinstance(stmt, _ALLOWED_TOP):
            raise _SQLSafetyError(
                f"不允许的语句类型「{type(stmt).__name__}」。\n"
                f"db_query 只允许 SELECT / WITH（CTE）查询语句。"
            )

        # (b) 深度遍历，查找隐藏写操作节点
        for node in stmt.walk():
            if isinstance(node, _WRITE_NODES):
                raise _SQLSafetyError(
                    f"检测到隐藏的写操作节点「{type(node).__name__}」"
                    f"（可能位于 CTE 或子查询中）。\n"
                    f"db_query 禁止任何形式的写操作，包括 CTE 内的写操作。"
                )


def _layer2_fallback(sql: str) -> None:
    """
    sqlglot 不可用时的 fallback：检查多语句注入 + 重复关键字检查。
    安全性略低于 AST 解析，但仍能拦截大多数攻击向量。
    """
    # 禁止分号分隔的多语句（防止 SELECT 1; DROP TABLE users）
    clean = sql.strip().rstrip(";")
    parts = [p.strip() for p in clean.split(";") if p.strip()]
    if len(parts) > 1:
        raise _SQLSafetyError(
            "检测到多条 SQL 语句（分号分隔）。\n"
            "db_query 每次只允许执行单条 SELECT 语句。"
        )


def validate_sql(sql: str) -> None:
    """
    统一入口：按序执行两层校验，任一层失败抛出 _SQLSafetyError。

    调用方只需 try/except _SQLSafetyError，无需感知内部分层细节。
    """
    _layer1_keyword_check(sql)
    _layer2_ast_check(sql)


# ---------------------------------------------------------------------------
# 2. 数据库连接路由
# ---------------------------------------------------------------------------

_SUPPORTED_DATABASES: frozenset[str] = frozenset(
    {"default", "postgres", "postgresql"}
)


async def _get_engine(database: str):
    """
    根据 database 别名返回对应的 SQLAlchemy AsyncEngine。

    扩展方式：在 settings 中增加 extra_databases 映射，
    然后在此函数中添加 elif database in extra_map 分支。
    """
    if database.lower() not in _SUPPORTED_DATABASES:
        raise ValueError(
            f"未知数据库别名「{database}」。\n"
            f"当前支持：{', '.join(sorted(_SUPPORTED_DATABASES))}。\n"
            f"如需接入新数据库，请在 settings.extra_databases 中添加配置。"
        )
    try:
        from app.infrastructure.postgres_client import get_engine  # type: ignore[import]
        return get_engine()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            f"数据库引擎未就绪：{exc}\n"
            f"请确认 Step 21（postgres_client.py）已完成，且数据库服务正常运行。"
        ) from exc


# ---------------------------------------------------------------------------
# 3. 查询执行（带超时）
# ---------------------------------------------------------------------------

async def _execute_query(sql: str, database: str) -> list[dict[str, Any]]:
    """
    在数据库中执行 SELECT 查询，返回结果行列表（最多 _MAX_ROWS+1 行）。

    多取一行（_MAX_ROWS + 1）的原因：
    用于判断结果是否被截断，避免额外的 COUNT(*) 查询浪费资源。
    若实际返回 101 行，说明真实结果超过 100 行，需要提示用户。
    """
    from sqlalchemy import text  # type: ignore[import]

    engine = await _get_engine(database)

    async def _run() -> list[dict[str, Any]]:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchmany(_MAX_ROWS + 1)
            return [dict(zip(columns, row)) for row in rows]

    # asyncio.wait_for 提供精确的超时控制
    try:
        return await asyncio.wait_for(_run(), timeout=_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"查询超时（超过 {_TIMEOUT_SEC} 秒）。\n"
            f"优化建议：\n"
            f"  - 添加 WHERE 条件缩小扫描范围\n"
            f"  - 为过滤列添加索引\n"
            f"  - 使用 LIMIT 限制返回行数"
        )


def _run_async_query(sql: str, database: str) -> list[dict[str, Any]]:
    """
    在同步上下文（@tool 是同步接口）中安全执行异步查询。

    处理三种运行时场景：
      1. 无事件循环（普通脚本）      → asyncio.run()
      2. 有循环但未运行（测试 fixture）→ loop.run_until_complete()
      3. 有正在运行的循环（FastAPI/Celery）→ 线程池提交新循环
    """
    coro = _execute_query(sql, database)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在已有运行中的事件循环里，不能直接 run_until_complete
            # 开一个独立线程运行新的事件循环，避免死锁
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=_TIMEOUT_SEC + 2)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 4. Markdown 表格格式化
# ---------------------------------------------------------------------------

def _cell_str(value: Any) -> str:
    """
    将单元格值转为安全的 Markdown 字符串：
      - None → 空字符串（不显示「None」）
      - 超长内容截断并附加「…」
      - 竖线「|」转义（防止破坏表格结构）
    """
    if value is None:
        return ""
    s = str(value)
    if len(s) > _MAX_CELL_LEN:
        s = s[:_MAX_CELL_LEN] + "…"
    return s.replace("|", "\\|")


def _to_markdown_table(rows: list[dict[str, Any]], truncated: bool) -> str:
    """
    将查询结果生成列宽自适应的 Markdown 表格。

    输出示例：
        | id | username | email              |
        |----|----------|--------------------|
        | 1  | alice    | alice@example.com  |
        | 2  | bob      | bob@example.com    |

        > ⚠️ 结果已截断，仅显示前 100 行。

    设计细节：
      - 列宽 = max(列名长度, 所有行该列最大值长度, 3)，保证分隔行可读
      - 截断提示用 blockquote（> ），与表格内容视觉区分
    """
    if not rows:
        return "_查询结果为空（0 行）。_"

    columns = list(rows[0].keys())

    # 预处理所有单元格，避免重复计算
    processed: list[list[str]] = []
    for row in rows:
        processed.append([_cell_str(row.get(col)) for col in columns])

    # 计算每列最大显示宽度
    col_widths: list[int] = []
    for i, col in enumerate(columns):
        max_w = max(
            len(col),
            max((len(r[i]) for r in processed), default=0),
            3,
        )
        col_widths.append(max_w)

    def _pad(text: str, width: int) -> str:
        return text.ljust(width)

    # 组装表格行
    header_cells = [_pad(col, col_widths[i]) for i, col in enumerate(columns)]
    sep_cells    = ["-" * col_widths[i] for i in range(len(columns))]

    lines: list[str] = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells)    + " |",
    ]
    for row_cells in processed:
        padded = [_pad(row_cells[i], col_widths[i]) for i in range(len(columns))]
        lines.append("| " + " | ".join(padded) + " |")

    table = "\n".join(lines)

    if truncated:
        table += (
            f"\n\n> ⚠️ 结果已截断，仅显示前 **{_MAX_ROWS}** 行。"
            f"如需查看全部结果，请添加更精确的 `WHERE` 条件。"
        )
    return table


# ---------------------------------------------------------------------------
# 5. 入参模型
# ---------------------------------------------------------------------------

class DBQueryInput(BaseModel):
    """db_query_tool 的入参模型，供 ToolSelector 生成合法 JSON 入参。"""

    sql: str = Field(
        description=(
            "要执行的 SELECT SQL 语句。\n"
            "安全限制：\n"
            "  · 只允许 SELECT / WITH（CTE）查询\n"
            "  · 禁止 INSERT / UPDATE / DELETE / CREATE / DROP 等写操作\n"
            "  · 禁止多条语句（分号分隔）\n"
            "  · 查询结果最多返回 100 行，超时限制 5 秒\n\n"
            "示例：\n"
            "  SELECT id, username, email FROM users WHERE is_active = true LIMIT 10\n"
            "  SELECT task_type, COUNT(*) AS cnt FROM agent_tasks GROUP BY task_type"
        )
    )
    database: str = Field(
        default="default",
        description=(
            "目标数据库别名。\n"
            "当前支持：default（主 PostgreSQL）。\n"
            "多数据库场景下，传入对应别名路由到指定实例。"
        ),
    )


# ---------------------------------------------------------------------------
# 6. @tool 入口
# ---------------------------------------------------------------------------

@tool(args_schema=DBQueryInput)
def db_query_tool(sql: str, database: str = "default") -> str:
    """
    在企业数据库中执行只读 SELECT 查询，返回 Markdown 格式结果表格。

    当需要以下情况时调用此工具：
    - 查询业务数据（用户、订单、任务、文档等记录）
    - 统计汇总（COUNT / SUM / AVG / GROUP BY）
    - 数据核查（验证某条记录是否存在、字段值是否正确）
    - 辅助 RAG（补充结构化数据，与知识库向量检索配合使用）

    安全保证（三层防线）：
    - ✅ 允许：SELECT、WITH（CTE）、JOIN、子查询（只读）、聚合函数
    - ❌ 禁止：INSERT / UPDATE / DELETE / CREATE / DROP / GRANT 等全部写操作
    - ❌ 禁止：CTE 内隐藏的写操作（WITH evil AS (INSERT ...) SELECT ...）
    - ⏱️ 超时：5 秒后自动终止，防止慢查询阻塞 Agent 工作流

    使用建议：
    - 先用 SELECT * FROM information_schema.columns WHERE table_name='表名'
      查询表结构，再构造精确的 SELECT 语句
    - 对大表务必加 WHERE 条件或 LIMIT，避免全表扫描

    Args:
        sql:      SELECT SQL 语句（字符串）
        database: 数据库别名，默认 "default"（主 PostgreSQL）

    Returns:
        Markdown 格式的查询结果表格，或结构化的错误说明。
    """
    sql = sql.strip()

    # ── 基础入参校验 ──────────────────────────────────────────────────────
    if not sql:
        return "错误：SQL 语句不能为空。"

    if len(sql) > _MAX_SQL_LEN:
        return (
            f"错误：SQL 语句过长（{len(sql)} 字符，上限 {_MAX_SQL_LEN} 字符）。\n"
            f"请精简查询，避免在 SQL 中嵌入大量字面量数据。"
        )

    start_time = time.monotonic()
    logger.info(
        "db_query 开始执行",
        extra={"database": database, "sql_preview": sql[:120]},
    )

    # ── 第一、二层：SQL 安全校验（同步，纳秒到毫秒级）───────────────────────
    try:
        validate_sql(sql)
    except _SQLSafetyError as exc:
        logger.info(
            "db_query 安全校验拒绝",
            extra={"database": database, "reason": str(exc)[:200]},
        )
        return f"🚫 SQL 安全校验失败\n\n{exc}"

    # ── 第三层：执行查询（带超时）──────────────────────────────────────────
    try:
        rows = _run_async_query(sql, database)
    except TimeoutError as exc:
        logger.warning(
            "db_query 查询超时",
            extra={"database": database, "sql_preview": sql[:120]},
        )
        return f"⏱️ {exc}"
    except Exception as exc:
        logger.error(
            "db_query 执行异常",
            extra={"database": database, "error": str(exc)},
            exc_info=True,
        )
        return (
            f"❌ 查询执行失败（{type(exc).__name__}）\n\n"
            f"错误详情：{exc}\n\n"
            f"常见原因：\n"
            f"  · 表名或列名拼写错误\n"
            f"  · 数据库连接未就绪（确认 PostgreSQL 服务正常）\n"
            f"  · SQL 语法与目标数据库方言不兼容"
        )

    # ── 截断判断 ─────────────────────────────────────────────────────────
    truncated    = len(rows) > _MAX_ROWS
    display_rows = rows[:_MAX_ROWS] if truncated else rows

    elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)
    row_count  = len(display_rows)

    logger.info(
        "db_query 执行成功",
        extra={
            "database":     database,
            "rows_returned": row_count,
            "truncated":    truncated,
            "elapsed_ms":   elapsed_ms,
        },
    )

    # ── 生成 Markdown 表格 ────────────────────────────────────────────────
    summary = (
        f"**数据库查询结果**｜"
        f"数据库：`{database}`｜"
        f"返回 **{row_count}** 行｜"
        f"耗时 {elapsed_ms} ms\n\n"
    )
    table = _to_markdown_table(display_rows, truncated)
    return summary + table