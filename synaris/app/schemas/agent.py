"""
@File       : agent.py  (schemas/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Agent API 层的全部 Pydantic 请求/响应/事件模型。
@Features:
  - AgentRunRequest   : POST /agent/run 的请求体
  - AgentRunResponse  : POST /agent/run 的立即响应（含 task_id）
  - AgentStatusResponse: GET /agent/status/{task_id} 的完整状态响应
  - StepSummary       : 单节点执行摘要（嵌入 AgentStatusResponse.steps）
  - AgentStepEvent    : WebSocket 实时推送的事件单元
      · step_type 枚举: node_start / node_end / tool_call /
                        stream_chunk / done / error / interrupt
  - AgentCancelResponse: POST /agent/cancel/{task_id} 的响应
  - HumanResumeRequest : POST /agent/resume/{task_id} 的请求体（Human-in-the-Loop）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from agents.state import AgentMode, TaskStatus


# ─────────────────────────────────────────────
# 1. 枚举：步骤事件类型
# ─────────────────────────────────────────────

class StepEventType(str, Enum):
    """
    WebSocket 推送事件的类型标签。

    类比「施工现场对讲机」的播报内容：
      node_start    → 「XX 节点开始施工」
      node_end      → 「XX 节点完工，耗时 N 毫秒」
      tool_call     → 「调用了工具 XX，结果是 ...」
      stream_chunk  → 「LLM 流式输出的一小块文字」（保留扩展）
      interrupt     → 「需要人工确认，暂停施工」
      done          → 「整体任务完工，最终成果如下」
      error         → 「出现故障，任务终止」
    """
    NODE_START    = "node_start"
    NODE_END      = "node_end"
    TOOL_CALL     = "tool_call"
    STREAM_CHUNK  = "stream_chunk"
    INTERRUPT     = "interrupt"
    DONE          = "done"
    ERROR         = "error"


# ─────────────────────────────────────────────
# 2. 请求模型
# ─────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    """
    POST /agent/run 请求体。

    Fields:
        task:       用户任务描述（自然语言，必填）
        mode:       运行模式（single / multi / rag_only，默认 single）
        session_id: 会话 ID（选填，用于 Human-in-the-Loop 断点续跑；
                    不传则由服务器自动生成 UUID）
        config:     透传给 AgentConfig 的扩展参数（选填）
                    支持字段：max_iterations / enable_human_loop /
                              allowed_tools / timeout_seconds
        user_context: 用户背景信息（多 Agent 模式下注入 Supervisor 提示词）

    示例：
        {
          "task": "分析最新季度财报并生成管理层摘要",
          "mode": "multi",
          "session_id": "sess-abc123",
          "config": {
            "max_iterations": 15,
            "enable_human_loop": true,
            "allowed_tools": ["rag_retrieval", "web_search"]
          },
          "user_context": "用户是销售总监，偏好简洁的执行摘要风格"
        }
    """

    task: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="用户任务描述（自然语言）",
        examples=["请帮我分析公司 Q3 销售数据并生成报告"],
    )
    mode: AgentMode = Field(
        default=AgentMode.SINGLE,
        description="Agent 运行模式：single（单 Agent）/ multi（多 Agent）/ rag_only",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="会话 ID（不传则服务器自动生成）",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="AgentConfig 扩展参数，支持 max_iterations / enable_human_loop 等",
    )
    user_context: str = Field(
        default="",
        max_length=2000,
        description="用户背景信息（多 Agent 模式下注入 Supervisor）",
    )

    @field_validator("task")
    @classmethod
    def task_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task 不能为空白字符串")
        return v.strip()

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, v: Any) -> Any:
        """允许传入小写字符串，如 'single' / 'multi'。"""
        if isinstance(v, str):
            return v.lower()
        return v


class HumanResumeRequest(BaseModel):
    """
    POST /agent/resume/{task_id} 请求体。
    用于 Human-in-the-Loop 场景：用户填写确认内容后，携带此模型恢复工作流。
    """
    response: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="人工确认/回答内容",
    )


# ─────────────────────────────────────────────
# 3. 响应模型
# ─────────────────────────────────────────────

class AgentRunResponse(BaseModel):
    """
    POST /agent/run 的立即响应（异步启动，不等待任务完成）。

    设计原则：「先收据，后取货」——
    客户端收到 task_id 后，通过 WebSocket 或轮询 /status 获取进展。
    """
    task_id:    str        = Field(description="全局唯一任务 ID（UUID）")
    session_id: str        = Field(description="会话 ID（用于断点续跑）")
    status:     TaskStatus = Field(description="初始状态，始终为 pending")
    mode:       str        = Field(description="实际使用的运行模式")
    message:    str        = Field(description="提示信息")
    created_at: str        = Field(description="任务创建时间（ISO 8601）")
    stream_url: str        = Field(description="WebSocket 推送地址（相对路径）")

    model_config = {"use_enum_values": True}


class StepSummary(BaseModel):
    """
    单节点执行摘要，嵌入 AgentStatusResponse.steps 列表。
    供状态查询接口（GET /agent/status）返回执行轨迹。
    """
    node_name:   str              = Field(description="LangGraph 节点名称")
    status:      str              = Field(description="completed / failed / skipped")
    content:     str              = Field(description="节点输出摘要（最多 300 字符）")
    tool_name:   Optional[str]    = Field(default=None, description="若涉及工具调用，记录工具名称")
    elapsed_ms:  Optional[float]  = Field(default=None, description="节点执行耗时（毫秒）")
    timestamp:   str              = Field(description="节点完成时间（ISO 8601）")


class AgentStatusResponse(BaseModel):
    """
    GET /agent/status/{task_id} 的完整状态响应。

    包含从任务创建到当前时刻的完整执行快照：
      - 基础信息（task_id / session_id / status）
      - 执行结果（result = final_answer / final_output）
      - 执行轨迹（steps = 每个节点的摘要）
      - 资源消耗（tokens_used / duration_ms）
      - 错误信息（error，仅 status=failed 时填写）
      - 中断信息（interrupt_question，仅 status=waiting_human 时填写）
    """
    task_id:            str              = Field(description="任务 ID")
    session_id:         str              = Field(description="会话 ID")
    status:             TaskStatus       = Field(description="当前任务状态")
    mode:               str              = Field(description="运行模式")
    task:               str              = Field(description="原始任务描述")
    result:             Optional[str]    = Field(default=None, description="最终回答（完成时填写）")
    steps:              list[StepSummary] = Field(default_factory=list, description="节点执行轨迹")
    tokens_used:        int              = Field(default=0, description="总 Token 消耗")
    duration_ms:        float            = Field(default=0.0, description="任务总耗时（毫秒）")
    error:              Optional[str]    = Field(default=None, description="错误信息（failed 时）")
    interrupt_question: Optional[str]   = Field(default=None, description="人工确认问题（waiting_human 时）")
    created_at:         str             = Field(description="任务创建时间（ISO 8601）")
    finished_at:        Optional[str]   = Field(default=None, description="任务完成时间（ISO 8601）")

    model_config = {"use_enum_values": True}


class AgentCancelResponse(BaseModel):
    """POST /agent/cancel/{task_id} 的响应。"""
    task_id:   str  = Field(description="被取消的任务 ID")
    cancelled: bool = Field(description="是否成功取消（false 表示任务已结束，无需取消）")
    message:   str  = Field(description="操作说明")


# ─────────────────────────────────────────────
# 4. WebSocket 事件模型
# ─────────────────────────────────────────────

class AgentStepEvent(BaseModel):
    """
    WebSocket 实时推送的事件单元。

    每当 LangGraph 的一个节点开始/结束，或工具被调用时，
    服务端通过 WebSocket 推送一条此格式的 JSON 消息。

    字段说明：
      event_id   → 事件唯一 ID（客户端去重用）
      task_id    → 所属任务 ID
      step_type  → 事件类型（见 StepEventType 枚举）
      node_name  → 触发事件的 LangGraph 节点名称
      content    → 事件内容（文本摘要，300 字符以内）
      tool_name  → 若为 tool_call 事件，记录工具名称
      result     → 若为 done 事件，携带最终回答
      error      → 若为 error 事件，携带错误信息
      interrupt  → 若为 interrupt 事件，携带中断信息
      elapsed_ms → 节点执行耗时（仅 node_end 事件填写）
      metadata   → 扩展字段（iteration_count / step_index 等）
      timestamp  → 事件产生时间（UTC ISO 8601）

    WebSocket 消息体示例（node_end 事件）：
        {
          "event_id": "evt-abc123",
          "task_id":  "task-xyz789",
          "step_type": "node_end",
          "node_name": "tool_executor",
          "content": "工具 web_search 执行完成，返回 3 条搜索结果",
          "tool_name": "web_search",
          "elapsed_ms": 1234.5,
          "metadata": { "step_index": 1 },
          "timestamp": "2026-03-24T10:30:00.123Z"
        }

    WebSocket 消息体示例（done 事件）：
        {
          "event_id": "evt-done-001",
          "task_id":  "task-xyz789",
          "step_type": "done",
          "node_name": "observer",
          "content": "任务完成",
          "result": "根据检索到的资料，Q3 销售数据显示...",
          "timestamp": "2026-03-24T10:32:15.000Z"
        }
    """

    event_id:   str              = Field(
        default_factory=lambda: f"evt-{__import__('uuid').uuid4().hex[:12]}",
        description="事件唯一 ID",
    )
    task_id:    str              = Field(description="所属任务 ID")
    step_type:  StepEventType   = Field(description="事件类型")
    node_name:  str              = Field(description="触发事件的节点名称")
    content:    str              = Field(default="", description="事件内容摘要（≤300字）")
    tool_name:  Optional[str]   = Field(default=None, description="工具名称（tool_call 时）")
    result:     Optional[str]   = Field(default=None, description="最终结果（done 时）")
    error:      Optional[str]   = Field(default=None, description="错误信息（error 时）")
    interrupt:  Optional[dict]  = Field(default=None, description="中断信息（interrupt 时）")
    elapsed_ms: Optional[float] = Field(default=None, description="节点耗时（node_end 时）")
    metadata:   dict[str, Any]  = Field(default_factory=dict, description="扩展元数据")
    timestamp:  str             = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="事件产生时间（UTC ISO 8601）",
    )

    model_config = {"use_enum_values": True}

    def to_ws_bytes(self) -> bytes:
        """序列化为 WebSocket 发送的字节流（UTF-8 JSON）。"""
        return self.model_dump_json(exclude_none=True).encode("utf-8")

    @classmethod
    def make_node_start(cls, task_id: str, node_name: str, **meta: Any) -> "AgentStepEvent":
        return cls(
            task_id=task_id,
            step_type=StepEventType.NODE_START,
            node_name=node_name,
            content=f"节点 [{node_name}] 开始执行",
            metadata=meta,
        )

    @classmethod
    def make_node_end(
        cls, task_id: str, node_name: str,
        content: str = "", elapsed_ms: float = 0, **meta: Any,
    ) -> "AgentStepEvent":
        return cls(
            task_id=task_id,
            step_type=StepEventType.NODE_END,
            node_name=node_name,
            content=content[:300] if content else f"节点 [{node_name}] 执行完成",
            elapsed_ms=elapsed_ms,
            metadata=meta,
        )

    @classmethod
    def make_tool_call(
        cls, task_id: str, node_name: str,
        tool_name: str, result_preview: str, **meta: Any,
    ) -> "AgentStepEvent":
        return cls(
            task_id=task_id,
            step_type=StepEventType.TOOL_CALL,
            node_name=node_name,
            tool_name=tool_name,
            content=result_preview[:300],
            metadata=meta,
        )

    @classmethod
    def make_done(cls, task_id: str, result: str, node_name: str = "observer") -> "AgentStepEvent":
        return cls(
            task_id=task_id,
            step_type=StepEventType.DONE,
            node_name=node_name,
            content="任务完成",
            result=result,
        )

    @classmethod
    def make_error(cls, task_id: str, error: str, node_name: str = "system") -> "AgentStepEvent":
        return cls(
            task_id=task_id,
            step_type=StepEventType.ERROR,
            node_name=node_name,
            content=f"任务失败：{error[:200]}",
            error=error,
        )

    @classmethod
    def make_interrupt(
        cls, task_id: str, question: str, options: list[str], node_name: str = "human_interrupt",
    ) -> "AgentStepEvent":
        return cls(
            task_id=task_id,
            step_type=StepEventType.INTERRUPT,
            node_name=node_name,
            content=f"需要人工确认：{question[:200]}",
            interrupt={"question": question, "options": options},
        )
