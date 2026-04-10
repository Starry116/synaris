"""
@File       : tool_registry.py
@Author     : Starry Hung
@Created    : 2026-04-05
@Version    : 2.0.0
@Description: Agent 工具动态注册中心（单例）。
@Features:
  - _ToolRegistry 单例类（模块级单例），全项目共享同一注册表
  - register(tool, roles, agent_types): 注册工具，绑定角色权限与适用 Agent 类型
  - get_tools_for_agent(agent_type, user_role): 按 Agent 类型 + 用户角色双维度过滤
  - get_tool_schema(): 返回 OpenAPI 格式的工具 Schema 列表，供前端/文档使用
  - invoke_with_audit(): 带自动审计的工具调用包装器，记录 latency_ms / input / output
  - 用户角色枚举：UserRole.ADMIN | USER | READONLY（支持 IDE 智能提示）
  - Agent 类型枚举：AgentType.RESEARCH | WRITER | REVIEWER | EXECUTOR | GENERAL | SUPERVISOR
  - 预定义工具组常量: RESEARCH_TOOL_NAMES / EXECUTION_TOOL_NAMES / ALL_TOOL_NAMES

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-10  Starry: 升级为 Enum 枚举系统、模块级单例、None 语义权限管理
    2026-03-24  Starry：Initial creation
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.tools import BaseTool

from app.core.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 枚举：用户角色 & Agent 类型
# ─────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    """
    用户角色枚举。
    类比：机场旅客的票务舱位级别——不同级别可以进入不同登机口。
    admin > user > readonly，权限依次递减。
    """
    ADMIN    = "admin"    # 超级管理员，可使用所有工具
    USER     = "user"     # 普通用户，可使用大多数工具
    READONLY = "readonly" # 只读用户，仅限查询类工具


class AgentType(str, Enum):
    """
    Agent 类型枚举。
    类比：机场的不同功能区（国内、国际、货运），每个区域开放的服务不同。
    """
    RESEARCH  = "research"   # ResearchAgent：擅长信息检索与收集
    WRITER    = "writer"     # WriterAgent：擅长内容生成
    REVIEWER  = "reviewer"   # ReviewerAgent：擅长质量审核
    EXECUTOR  = "executor"   # ExecutorAgent：擅长代码/计算执行
    GENERAL   = "general"    # 通用单 Agent，访问全量工具
    SUPERVISOR = "supervisor" # Supervisor：工具权限最高


# ─────────────────────────────────────────────────────────────────────
# 角色常量（与 core/auth.py 保持同步）
# ─────────────────────────────────────────────────────────────────────
class Role:
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


# ─────────────────────────────────────────────
# Agent 类型常量（与 agents/workers.py 保持同步）
# ─────────────────────────────────────────────
class AgentType:
    RESEARCH = "research"
    WRITER = "writer"
    REVIEWER = "reviewer"
    SUPERVISOR = "supervisor"
    GENERAL = "general"  # 单 Agent 通用


# ─────────────────────────────────────────────────────────────────────
# 工具元信息容器
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ToolEntry:
    """
    工具注册条目：将一个 LangChain Tool 与它的权限配置绑在一起。

    类比：机场地勤系统里的「航班信息卡」，记录航班号（工具）、
    可登机的旅客舱位（roles）以及适用航站楼（agent_types）。
    """
    tool: BaseTool
    # None 表示所有角色均可访问
    allowed_roles: Optional[set[UserRole]] = None
    # None 表示所有 Agent 类型均可访问
    allowed_agent_types: Optional[set[AgentType]] = None
    # 注册时间戳（秒），便于审计与排查
    registered_at: float = field(default_factory=time.time)

    @property
    def name(self) -> str:
        return self.tool.name

    def is_accessible(self, agent_type: AgentType, user_role: UserRole) -> bool:
        """
        双维度权限校验：Agent 类型 AND 用户角色均须满足。

        逻辑：
            角色检查：若 allowed_roles 为 None → 不限角色；否则判断是否在白名单中。
            类型检查：若 allowed_agent_types 为 None → 不限类型；否则判断是否在白名单中。
        """
        role_ok = (
            self.allowed_roles is None
            or user_role in self.allowed_roles
        )
        type_ok = (
            self.allowed_agent_types is None
            or agent_type in self.allowed_agent_types
        )
        return role_ok and type_ok


# ─────────────────────────────────────────────────────────────────────
# ToolRegistry 单例
# ─────────────────────────────────────────────────────────────────────

class _ToolRegistry:
    """
    工具动态注册中心（内部类，外部通过 tool_registry 单例访问）。

    单例实现方式：模块级实例（module-level singleton）。
    相比 __new__ 重写，这种方式更 Pythonic，且线程安全（GIL 保证模块导入原子性）。

    线程安全：注册操作使用 threading.Lock 保护，避免并发注册时的 dict 竞争。

    类比：
        整个机场只有一个「航班调度中心」——无论哪个航站楼的柜台调用，
        查到的都是同一份航班数据库。
    """

    def __init__(self) -> None:
        # 以工具名为 key 的注册表，值为 ToolEntry
        self._registry: dict[str, ToolEntry] = {}
        self._lock = threading.Lock()
        logger.info("ToolRegistry 初始化完成")

    # ──────────────────────────────────────────────────────────────────
    # 注册
    # ──────────────────────────────────────────────────────────────────

    def register(
        self,
        tool: BaseTool,
        roles: Optional[list[UserRole]] = None,
        agent_types: Optional[list[AgentType]] = None,
    ) -> "_ToolRegistry":
        """
        将工具注册到注册中心。

        参数：
            tool        : LangChain BaseTool 实例
            roles       : 允许使用此工具的角色列表；None 表示不限角色
            agent_types : 允许使用此工具的 Agent 类型列表；None 表示不限类型

        返回：
            self，支持链式调用：
            registry.register(web_search, roles=[UserRole.USER]).register(...)

        幂等性：重复注册同名工具会覆盖旧条目，并打印警告日志。
        """
        entry = ToolEntry(
            tool=tool,
            allowed_roles=set(roles) if roles else None,
            allowed_agent_types=set(agent_types) if agent_types else None,
        )
        with self._lock:
            if tool.name in self._registry:
                logger.warning(
                    "工具重复注册，将覆盖旧条目",
                    extra={"tool_name": tool.name},
                )
            self._registry[tool.name] = entry

        logger.info(
            "工具注册成功",
            extra={
                "tool_name": tool.name,
                "allowed_roles": (
                    [r.value for r in entry.allowed_roles]
                    if entry.allowed_roles else "ALL"
                ),
                "allowed_agent_types": (
                    [t.value for t in entry.allowed_agent_types]
                    if entry.allowed_agent_types else "ALL"
                ),
            },
        )
        return self  # 支持链式注册

    def register_group(
        self,
        tools: list[BaseTool],
        roles: Optional[list[UserRole]] = None,
        agent_types: Optional[list[AgentType]] = None,
    ) -> "_ToolRegistry":
        """批量注册一组工具，共享同一套权限配置"""
        for tool in tools:
            self.register(tool, roles=roles, agent_types=agent_types)
        return self  # 支持链式调用

    # ──────────────────────────────────────────────────────────────────
    # 查询：按权限过滤
    # ──────────────────────────────────────────────────────────────────

    def get_tools_for_agent(
        self,
        agent_type: AgentType,
        user_role: UserRole,
    ) -> list[BaseTool]:
        """
        根据 Agent 类型 + 用户角色，返回该 Agent 有权使用的工具列表。

        过滤逻辑（双维度 AND）：
            1. agent_type 是否在工具的 allowed_agent_types 中（或无限制）
            2. user_role  是否在工具的 allowed_roles 中（或无限制）

        类比：安检系统同时检查「登机牌目的地」和「护照签证级别」，
        两项都通过才能放行。
        """
        accessible: list[BaseTool] = []
        denied: list[str] = []

        with self._lock:
            entries = list(self._registry.values())

        for entry in entries:
            if entry.is_accessible(agent_type, user_role):
                accessible.append(entry.tool)
            else:
                denied.append(entry.name)

        logger.debug(
            "工具权限过滤完成",
            extra={
                "agent_type": agent_type.value,
                "user_role": user_role.value,
                "accessible_tools": [t.name for t in accessible],
                "denied_tools": denied,
            },
        )
        return accessible

    # ──────────────────────────────────────────────────────────────────
    # 查询：OpenAPI Schema
    # ──────────────────────────────────────────────────────────────────

    def get_tool_schema(self) -> list[dict[str, Any]]:
        """
        返回所有已注册工具的 OpenAPI 格式 Schema 列表。

        输出结构（每个工具）：
        {
            "name": "web_search",
            "description": "搜索互联网...",
            "parameters": { ... },   # JSON Schema
            "allowed_roles": ["admin", "user"],
            "allowed_agent_types": ["research", "general"]
        }

        用途：前端工具面板渲染、API 文档生成、调试。
        """
        schemas: list[dict[str, Any]] = []
        with self._lock:
            entries = list(self._registry.values())

        for entry in entries:
            tool = entry.tool
            # LangChain BaseTool 通过 args_schema 暴露 Pydantic 模型
            parameters: dict[str, Any] = {}
            if hasattr(tool, "args_schema") and tool.args_schema is not None:
                try:
                    parameters = tool.args_schema.model_json_schema()
                except Exception:
                    parameters = {}

            schemas.append({
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
                "allowed_roles": (
                    [r.value for r in entry.allowed_roles]
                    if entry.allowed_roles else ["ALL"]
                ),
                "allowed_agent_types": (
                    [t.value for t in entry.allowed_agent_types]
                    if entry.allowed_agent_types else ["ALL"]
                ),
                "registered_at": entry.registered_at,
            })

        return schemas

    # ──────────────────────────────────────────────────────────────────
    # 工具调用审计
    # ──────────────────────────────────────────────────────────────────

    def audit_call(
        self,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float,
        agent_type: str = "",
        user_role: str = "",
        error: Optional[str] = None,
    ) -> None:
        """
        记录工具调用审计日志。

        类比：工具间的「访客记录本」——每次取用工具都留档，
        供安全审计、成本分析、异常追踪使用。

        通常不直接调用此方法，而是使用 invoke_with_audit() 包装器，
        它会自动计时并在 finally 块中调用 audit_call()。
        """
        log_payload: dict[str, Any] = {
            "audit_type": "tool_call",
            "tool_name": tool_name,
            "agent_type": agent_type,
            "user_role": user_role,
            "latency_ms": round(latency_ms, 2),
            # 截断过长内容，避免日志爆炸
            "input_preview": _truncate(str(input_data), 300),
            "output_preview": _truncate(str(output_data), 500),
            "success": error is None,
        }
        if error:
            log_payload["error"] = error
            logger.warning("工具调用失败", extra=log_payload)
        else:
            logger.info("工具调用审计", extra=log_payload)

    # ──────────────────────────────────────────────────────────────────
    # 便捷方法
    # ──────────────────────────────────────────────────────────────────

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """按名称获取单个工具（不做权限过滤，仅内部使用）。"""
        with self._lock:
            entry = self._registry.get(name)
        return entry.tool if entry else None

    def list_tools(self) -> list[str]:
        """返回所有已注册工具的名称列表。"""
        with self._lock:
            return list(self._registry.keys())

    def unregister(self, tool_name: str) -> bool:
        """
        从注册表中移除工具（动态卸载场景）。
        返回 True 表示成功移除，False 表示工具不存在。
        """
        with self._lock:
            if tool_name in self._registry:
                del self._registry[tool_name]
                logger.info("工具已从注册表移除", extra={"tool_name": tool_name})
                return True
        logger.warning("尝试移除不存在的工具", extra={"tool_name": tool_name})
        return False

    def clear(self) -> None:
        """清空所有注册工具（通常用于测试，生产环境慎用）"""
        with self._lock:
            self._registry.clear()
        logger.warning("ToolRegistry 已清空所有工具")

    def __repr__(self) -> str:
        with self._lock:
            names = list(self._registry.keys())
        return f"<ToolRegistry tools={names}>"


# ─────────────────────────────────────────────────────────────────────
# 模块级单例（对外暴露的唯一实例）
# ─────────────────────────────────────────────────────────────────────

#: 全项目唯一的工具注册中心实例，其他模块直接 import 此对象使用。
tool_registry = _ToolRegistry()

# 类型别名，方便类型注解
ToolRegistry = _ToolRegistry


# ─────────────────────────────────────────────────────────────────────
# 带审计包装的工具调用辅助函数
# ─────────────────────────────────────────────────────────────────────

def invoke_with_audit(
    tool: BaseTool,
    input_data: Any,
    agent_type: AgentType | str = "",
    user_role: UserRole | str = "",
) -> Any:
    """
    带自动审计的工具调用包装器。

    在 LangGraph tool_executor 节点中统一使用此函数，
    确保每次调用都记录审计日志，无需手动调用 audit_call()。

    参数：
        tool: LangChain BaseTool 实例
        input_data: 工具的输入数据
        agent_type: Agent 类型（AgentType 枚举或字符串）
        user_role: 用户角色（UserRole 枚举或字符串）

    返回：
        工具的执行结果

    Example（在 agents/workflow.py 的 tool_executor 节点）:
        result = invoke_with_audit(
            tool,
            tool_input,
            agent_type=AgentType.RESEARCH,
            user_role=UserRole.USER,
        )
    """
    # 支持枚举和字符串两种输入
    agent_type_str = agent_type.value if isinstance(agent_type, AgentType) else str(agent_type)
    user_role_str = user_role.value if isinstance(user_role, UserRole) else str(user_role)

    start = time.perf_counter()
    error_msg: Optional[str] = None
    result: Any = None

    try:
        result = tool.invoke(input_data)
        return result
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        tool_registry.audit_call(
            tool_name=tool.name,
            input_data=input_data,
            output_data=result,
            latency_ms=latency_ms,
            agent_type=agent_type_str,
            user_role=user_role_str,
            error=error_msg,
        )


# ─────────────────────────────────────────────────────────────────────
# 预定义工具组（懒加载，避免循环导入）
# ─────────────────────────────────────────────────────────────────────

def _load_tool_groups() -> tuple[list[BaseTool], list[BaseTool], list[BaseTool]]:
    """
    懒加载工具实例，避免模块导入时产生循环依赖。

    三组工具说明：
      RESEARCH_TOOLS   = [web_search, rag_retrieval, db_query]
      EXECUTION_TOOLS  = [calculator, code_executor, external_api]
      ALL_TOOLS        = RESEARCH_TOOLS + EXECUTION_TOOLS

    注意：db_query / external_api 在 Step 22 后半段完成后取消注释
    """
    from app.agents.tools.web_search import web_search_tool        # noqa: PLC0415
    from app.agents.tools.rag_retrieval import rag_retrieval_tool  # noqa: PLC0415
    from app.agents.tools.calculator import calculator_tool        # noqa: PLC0415
    from app.agents.tools.code_executor import code_executor_tool  # noqa: PLC0415
    # Step 22 后半段生成后取消注释：
    # from app.agents.tools.db_query import db_query_tool
    # from app.agents.tools.external_api import external_api_tool

    research_tools: list[BaseTool] = [
        web_search_tool,
        rag_retrieval_tool,
        # db_query_tool,       ← Step 22b 后加入
    ]

    execution_tools: list[BaseTool] = [
        calculator_tool,
        code_executor_tool,
        # external_api_tool,   ← Step 22b 后加入
    ]

    all_tools: list[BaseTool] = research_tools + execution_tools

    return research_tools, execution_tools, all_tools


def setup_default_registry() -> ToolRegistry:
    """
    使用默认权限策略初始化全局注册中心。

    权限矩阵（对应 PRD §5.3 工具调用 + §8.1 安全合规）:

    ┌──────────────────┬──────────────────────────┬───────────────────────────────┐
    │ 工具             │ 允许角色                  │ 允许 Agent 类型                │
    ├──────────────────┼──────────────────────────┼───────────────────────────────┤
    │ web_search       │ admin, user              │ research, general             │
    │ rag_retrieval    │ admin, user, readonly    │ research, general, writer     │
    │ db_query *       │ admin                    │ research, general             │
    │ calculator       │ admin, user              │ general, writer               │
    │ code_executor    │ admin                    │ general                       │
    │ external_api *   │ admin                    │ general                       │
    └──────────────────┴──────────────────────────┴───────────────────────────────┘
    * 待 Step 22b 完成后启用

    通常在 main.py lifespan startup 阶段调用一次：
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            setup_default_registry()
            yield
    """
    research_tools, execution_tools, _ = _load_tool_groups()

    web_search_tool    = research_tools[0]
    rag_retrieval_tool = research_tools[1]
    calculator_tool    = execution_tools[0]
    code_executor_tool = execution_tools[1]

    # ── RESEARCH_TOOLS ──────────────────────────────────────────────────
    # 定位：信息检索类工具，readonly 角色也能使用
    # Agent 类型：RESEARCH / GENERAL 均可调用
    _RESEARCH_AGENT_TYPES = [
        AgentType.RESEARCH,
        AgentType.GENERAL,
    ]
    _RESEARCH_ROLES = [UserRole.ADMIN, UserRole.USER, UserRole.READONLY]

    tool_registry.register(
        web_search_tool,
        roles=[UserRole.ADMIN, UserRole.USER],
        agent_types=_RESEARCH_AGENT_TYPES,
    )
    tool_registry.register(
        rag_retrieval_tool,
        roles=_RESEARCH_ROLES,
        agent_types=[AgentType.RESEARCH, AgentType.GENERAL, AgentType.WRITER],
    )
    # db_query：Step 22b 后取消注释
    # tool_registry.register(
    #     db_query_tool,
    #     roles=[UserRole.ADMIN],
    #     agent_types=_RESEARCH_AGENT_TYPES,
    # )

    # ── EXECUTION_TOOLS ────────────────────────────────────────────────
    # 定位：执行类工具，readonly 角色无权使用（防止只读账号触发副作用）
    _EXEC_AGENT_TYPES = [
        AgentType.GENERAL,
    ]
    _EXEC_ROLES = [UserRole.ADMIN, UserRole.USER]  # 不含 READONLY

    tool_registry.register(
        calculator_tool,
        roles=_EXEC_ROLES,
        agent_types=[AgentType.GENERAL, AgentType.WRITER],
    )
    tool_registry.register(
        code_executor_tool,
        roles=[UserRole.ADMIN],
        agent_types=[AgentType.GENERAL],
    )
    # external_api：Step 22b 后取消注释
    # tool_registry.register(
    #     external_api_tool,
    #     roles=[UserRole.ADMIN],
    #     agent_types=[AgentType.GENERAL],
    # )

    logger.info(
        "默认工具组注册完成",
        extra={"registered_tools": tool_registry.list_tools()},
    )
    return tool_registry


# ─────────────────────────────────────────────────────────────────────
# 预定义工具组常量
# ─────────────────────────────────────────────────────────────────────

#: 研究类工具名称集合
RESEARCH_TOOL_NAMES: frozenset[str] = frozenset(
    {"web_search", "rag_retrieval", "db_query"}
)

#: 执行类工具名称集合
EXECUTION_TOOL_NAMES: frozenset[str] = frozenset(
    {"calculator", "code_executor", "external_api"}
)

#: 全量工具名称集合
ALL_TOOL_NAMES: frozenset[str] = RESEARCH_TOOL_NAMES | EXECUTION_TOOL_NAMES


# ─────────────────────────────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    """截断字符串，防止审计日志单条过大"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"…（已截断，原长 {len(text)} 字符）"


# ─────────────────────────────────────────────────────────────────────
# 公开 API 汇总
# ─────────────────────────────────────────────────────────────────────

__all__ = [
    # 单例
    "tool_registry",
    "ToolRegistry",
    # 枚举
    "UserRole",
    "AgentType",
    # 兼容性（deprecated）
    "Role",
    # 数据类
    "ToolEntry",
    # 函数
    "invoke_with_audit",
    "setup_default_registry",
    # 常量
    "RESEARCH_TOOL_NAMES",
    "EXECUTION_TOOL_NAMES",
    "ALL_TOOL_NAMES",
]