"""
@File       : state.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: LangGraph Agent 状态定义模块。
@Features:
  - AgentState(TypedDict): LangGraph 状态机的核心状态容器，
    承载单次 Agent 任务从启动到结束的全部上下文
  - TaskStatus 枚举: 标准化任务生命周期状态
    (PENDING → RUNNING → WAITING_HUMAN → COMPLETED / FAILED)
  - AgentMessage: Agent 间通信的标准消息格式，
    支持 system / user / assistant / tool 四种角色
  - InterruptPayload: Human-in-the-Loop 中断请求载体
  - AgentConfig: 单次任务的运行时配置（模式、超时、最大迭代）
  - 辅助工厂函数: make_user_message / make_tool_message /
    make_assistant_message / make_system_message — 减少调用方样板代码
  - initial_state(): 构造干净的初始状态，作为 graph.invoke() 入参

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import uuid
import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# 1. TaskStatus — 任务生命周期枚举
# ─────────────────────────────────────────────


class TaskStatus(str, Enum):
    """
    Agent 任务的标准生命周期状态。

    状态流转示意：

        PENDING
            ↓  (Celery worker 拾取任务)
        RUNNING
            ├──→ WAITING_HUMAN  (遇到需人工确认的节点)
            │         ↓  (人工响应后恢复执行)
            │       RUNNING
            ├──→ COMPLETED      (正常完成)
            └──→ FAILED         (执行异常)
    """

    PENDING = "pending"  # 已提交，等待 Worker 拾取
    RUNNING = "running"  # 执行中
    WAITING_HUMAN = "waiting_human"  # Human-in-the-Loop 挂起，等待人工响应
    COMPLETED = "completed"  # 成功完成，final_answer 已填入
    FAILED = "failed"  # 执行失败，error 字段含异常信息


# ─────────────────────────────────────────────
# 2. MessageRole / AgentMessage — 标准消息格式
# ─────────────────────────────────────────────


class MessageRole(str, Enum):
    """
    消息角色，对齐 OpenAI Chat Completion 规范，并扩展 tool 角色。

    类比通话录音中的说话人标签：
      - system    → 系统提示（导演指令，不对用户展示）
      - user      → 用户输入（或上游 Agent 的任务委托）
      - assistant → AI 回复（当前 Agent 的推理输出）
      - tool      → 工具返回结果（工具箱里的反馈）
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentMessage(BaseModel):
    """
    Agent 间通信的标准消息单元。

    类比邮件信封：
      - role         → 发件人身份标签
      - content      → 信件正文
      - tool_name    → 若来自工具调用，记录工具名称（相当于"部门章"）
      - tool_call_id → 与上游 LLM tool_calls[i].id 配对，形成 Request-Response 链路
      - agent_id     → 多 Agent 场景下标识消息来源（哪个 Worker 产生了这条消息）
      - message_id   → 全局唯一 ID，防止消息重复处理
      - timestamp    → 消息产生时间（UTC），用于追溯与排序
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    tool_name: Optional[str] = None  # role=TOOL 时填写调用的工具名称
    tool_call_id: Optional[str] = None  # 与 LLM tool_calls 响应中的 id 对应
    agent_id: Optional[str] = None  # 多 Agent 场景：发送方 Agent 标识
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"use_enum_values": True}

    def to_langchain_dict(self) -> dict[str, Any]:
        """
        转换为 LangChain / OpenAI messages 列表所需的最小字典格式。
        tool 角色额外附带 tool_call_id，满足 OpenAI API 规范。
        """
        base: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.role == MessageRole.TOOL and self.tool_call_id:
            base["tool_call_id"] = self.tool_call_id
        return base

    def __repr__(self) -> str:
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"AgentMessage(role={self.role}, content='{preview}')"


# ─────────────────────────────────────────────
# 3. InterruptPayload — Human-in-the-Loop 中断载体
# ─────────────────────────────────────────────


class InterruptPayload(BaseModel):
    """
    当 Agent 遇到需要人工决策的节点时，
    通过此模型将「问题」结构化推送给前端（via WebSocket），
    并在人工回复后携带「答案」回填，以恢复执行。

    类比审批单：
      - interrupt_reason  → 为何需要审批（事由）
      - question          → 向审批人提出的具体问题
      - options           → 预设选项（下拉菜单，可为空表示自由文本）
      - human_response    → 审批人填写的答案（回填后触发恢复）
    """

    interrupt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interrupt_reason: str  # Agent 为何需要人工介入（内部原因说明）
    question: str  # 向人类提出的具体问题（展示给用户）
    options: list[str] = Field(default_factory=list)  # 可选答案列表
    human_response: Optional[str] = None  # 人工回填后写入此字段
    responded_at: Optional[datetime] = None  # 人工响应时间戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_responded(self) -> bool:
        """判断人工是否已响应此中断。"""
        return self.human_response is not None


# ─────────────────────────────────────────────
# 4. AgentMode / AgentConfig — 任务运行时配置
# ─────────────────────────────────────────────


class AgentMode(str, Enum):
    """
    Agent 运行模式。

    - SINGLE    : 单 Agent 工作流（planner → tool_selector → executor → observer）
    - MULTI     : 多 Agent 协作模式（Supervisor 分解并分配给 Worker Agents）
    - RAG_ONLY  : 纯 RAG 检索模式，不调用外部工具，仅做向量检索 + LLM 生成
    """

    SINGLE = "single"
    MULTI = "multi"
    RAG_ONLY = "rag_only"


class AgentConfig(BaseModel):
    """
    任务级运行参数，序列化后存入 AgentState.metadata["config"]，
    供 LangGraph 各节点按需读取。

    不放入 AgentState 顶层字段是为了保持状态结构的简洁性，
    同时允许运行时动态修改而不影响 LangGraph 的类型检查。
    """

    mode: AgentMode = AgentMode.SINGLE
    max_iterations: int = Field(
        default=10, ge=1, le=50, description="最大迭代轮次，防止死循环"
    )
    timeout_seconds: int = Field(
        default=300, ge=30, le=3600, description="任务超时时间（秒）"
    )
    enable_human_loop: bool = False  # 是否启用 Human-in-the-Loop 节点
    allowed_tools: list[str] = Field(
        default_factory=list, description="空列表表示允许全部工具"
    )
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────
# 5. AgentState — LangGraph 核心状态容器（TypedDict）
# ─────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """
    LangGraph StateGraph 的核心状态字典。

    设计原则 —— 类比一部舞台剧的执行档案：
    ┌─────────────────────────────────────────────────────┐
    │  messages        → 演员台词本（对话历史，只增不减）    │
    │  task            → 本场演出主题（用户原始任务）        │
    │  plan            → 导演分镜表（Planner 生成的步骤）    │
    │  current_step    → 正在演第几幕（当前步骤索引）        │
    │  tool_results    → 道具间反馈（工具调用结果列表）      │
    │  final_answer    → 谢幕词（最终交付给用户的回答）      │
    │  status          → 场务状态板（任务当前状态）          │
    │  error           → NG 记录单（异常信息）               │
    │  iteration_count → 已循环轮次（防死循环计数器）        │
    │  interrupt       → 中场暂停单（Human-in-the-Loop）     │
    │  memory_context  → 记忆注入区（Step 23 由 Memory 填充）│
    │  metadata        → 演出手册（trace_id / config 等）    │
    └─────────────────────────────────────────────────────┘

    ⚠️  messages 字段使用 Annotated[list, operator.add] 声明 reducer：
        这是 LangGraph 并发安全写入的关键机制。
        多个节点并发向 messages 追加时，框架会自动合并而不是覆盖。
        其余字段默认采用「最后写入覆盖（last-wins）」语义。
    """

    # ── 对话历史（append-only，由 LangGraph reducer 保证并发安全）─
    messages: Annotated[list[AgentMessage], operator.add]

    # ── 任务描述与执行计划 ──────────────────────────────────────
    task: str  # 用户原始任务描述文本
    plan: list[str]  # Planner 节点输出的分步骤列表
    current_step: int  # 当前执行步骤索引（0-based）

    # ── 工具调用结果列表 ────────────────────────────────────────
    # 每条记录格式：
    #   {
    #     "tool":       str,         # 工具名称
    #     "input":      Any,         # 工具调用入参
    #     "output":     Any,         # 工具返回结果
    #     "error":      str | None,  # 若调用失败，记录异常信息
    #     "elapsed_ms": float,       # 工具执行耗时（毫秒）
    #     "called_at":  str,         # ISO8601 调用时间戳
    #   }
    tool_results: list[dict[str, Any]]

    # ── 最终输出与任务状态 ──────────────────────────────────────
    final_answer: str  # 最终回答（COMPLETED 时填入）
    status: TaskStatus  # 当前任务状态
    error: Optional[str]  # 异常信息（FAILED 时填入）
    iteration_count: int  # 已执行迭代次数（上限由 AgentConfig 控制）

    # ── Human-in-the-Loop ───────────────────────────────────────
    interrupt: Optional[InterruptPayload]  # 非 None 表示任务当前处于中断等待

    # ── 记忆上下文（Step 23 MemoryService 完成后回填）──────────
    # 当前声明为 Optional[Any]，Step 23 完成后替换为 Optional[MemoryContext] 强类型
    memory_context: Optional[Any]

    # ── 扩展元数据 ──────────────────────────────────────────────
    # 建议存储：trace_id / config(AgentConfig.model_dump()) / timing 等
    metadata: dict[str, Any]


# ─────────────────────────────────────────────
# 6. 工厂函数 — 减少调用方样板代码
# ─────────────────────────────────────────────


def make_user_message(
    content: str,
    agent_id: Optional[str] = None,
) -> AgentMessage:
    """创建用户消息（或上游 Agent 发出的任务委托消息）。"""
    return AgentMessage(
        role=MessageRole.USER,
        content=content,
        agent_id=agent_id,
    )


def make_assistant_message(
    content: str,
    agent_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AgentMessage:
    """创建 AI 助手消息（Agent 推理输出）。"""
    return AgentMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        agent_id=agent_id,
        metadata=metadata or {},
    )


def make_tool_message(
    content: str,
    tool_name: str,
    tool_call_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> AgentMessage:
    """
    创建工具结果消息。

    tool_call_id 必须与 LLM 返回的 tool_calls[i].id 一致，
    否则部分模型（如 GPT-4o）会拒绝该消息。
    """
    return AgentMessage(
        role=MessageRole.TOOL,
        content=content,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        agent_id=agent_id,
    )


def make_system_message(content: str) -> AgentMessage:
    """创建系统消息（通常由各 Agent 的系统提示词模板渲染后调用）。"""
    return AgentMessage(role=MessageRole.SYSTEM, content=content)


# ─────────────────────────────────────────────
# 7. initial_state() — 构造初始状态
# ─────────────────────────────────────────────


def initial_state(
    task: str,
    config: Optional[AgentConfig] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> AgentState:
    """
    构造一个干净的初始 AgentState，作为 LangGraph graph.invoke() / graph.ainvoke() 的入参。

    使用示例：
        from agents.state import initial_state, AgentConfig, AgentMode

        state = initial_state(
            task="请分析最新季度财报并生成三段式摘要",
            config=AgentConfig(
                mode=AgentMode.MULTI,
                enable_human_loop=True,
                max_iterations=15,
            ),
            session_id="sess-abc123",
            user_id="user-xyz789",
        )
        result = await graph.ainvoke(state, config={"recursion_limit": 50})
    """
    cfg = config or AgentConfig()

    # 将 session_id / user_id 写入 config，保证各节点可通过 metadata 读取
    update: dict[str, Any] = {}
    if session_id:
        update["session_id"] = session_id
    if user_id:
        update["user_id"] = user_id
    if update:
        cfg = cfg.model_copy(update=update)

    return AgentState(
        messages=[],
        task=task,
        plan=[],
        current_step=0,
        tool_results=[],
        final_answer="",
        status=TaskStatus.PENDING,
        error=None,
        iteration_count=0,
        interrupt=None,
        memory_context=None,
        metadata={
            "trace_id": trace_id or str(uuid.uuid4()),
            "config": cfg.model_dump(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
