"""
@File       : workflow.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: LangGraph 单 Agent 完整状态机实现。
@Features:
  - 5 个节点: planner → tool_selector → tool_executor → observer
              → human_interrupt（条件触发）
  - 条件路由: observer 根据 LLM 决策动态选择下一个节点
  - 最大迭代保护: 超出 max_iterations 时强制进入 complete 或 fail 分支
  - Human-in-the-Loop: interrupt 节点挂起等待人工响应，
    恢复后携带响应内容回到 observer 继续决策
  - 状态持久化: MemorySaver（内存），可替换为 RedisSaver（生产）
  - 统一错误处理: 每个节点均有 try/except，异常转化为 FAILED 状态而非崩溃
  - 工具注册表集成: 从 AgentConfig.allowed_tools 动态过滤可用工具
  - run_workflow() 异步入口: 对外暴露统一调用接口，隐藏 graph 内部细节

  ┌──────────────────────────────────────────────────────────────────┐
  │                      状态机完整流转图                              │
  │                                                                    │
  │   ┌─────────┐                                                      │
  │   │  START  │                                                      │
  │   └────┬────┘                                                      │
  │        │ initial_state()                                           │
  │        ▼                                                           │
  │   ┌──────────────┐   解析任务，生成 plan[]                          │
  │   │  planner     │   写入: plan / messages / status=RUNNING         │
  │   └──────┬───────┘                                                 │
  │          │                                                         │
  │          ▼                                                         │
  │   ┌──────────────┐   选择工具 + 入参（或 none）                     │
  │   │ tool_selector│   写入: metadata["pending_tool"]                 │
  │   └──────┬───────┘                                                 │
  │          │                                                         │
  │          ▼                                                         │
  │   ┌──────────────┐   调用工具，记录 tool_results[]                  │
  │   │ tool_executor│   写入: tool_results / messages                  │
  │   └──────┬───────┘                                                 │
  │          │                                                         │
  │          ▼                                                         │
  │   ┌──────────────┐   观察结果，做出决策                             │
  │   │   observer   │──────────────────────────────────────┐          │
  │   └──────┬───────┘                                      │          │
  │          │                                              │          │
  │     ┌────┴────────────────────────┐             decision=human     │
  │     │            │                │                    │           │
  │ decision=    decision=        decision=         ┌──────▼────────┐  │
  │ continue     retry            complete/fail     │human_interrupt│  │
  │ (next step)  (same step)           │            │   (挂起等待)  │  │
  │     │            │                │            └──────┬────────┘  │
  │     │            │            ┌───▼───┐               │           │
  │     │            │            │  END  │   人工响应后   │           │
  │     │            │            └───────┘               │           │
  │     └────────────┴──────────────────── → tool_selector │           │
  │          ↑ current_step 推进/不变                      │           │
  │          └────────────────────────────────────────────┘           │
  │                           恢复执行                                  │
  └──────────────────────────────────────────────────────────────────┘

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.state import (
    AgentConfig,
    AgentState,
    InterruptPayload,
    TaskStatus,
    initial_state,
    make_assistant_message,
    make_system_message,
    make_tool_message,
    make_user_message,
)
from core.prompts import PromptKey, get_prompt

logger = logging.getLogger(__name__)

# ── 节点名称常量（避免散落的魔法字符串）────────────────────────────────────────
NODE_PLANNER        = "planner"
NODE_TOOL_SELECTOR  = "tool_selector"
NODE_TOOL_EXECUTOR  = "tool_executor"
NODE_OBSERVER       = "observer"
NODE_HUMAN_INTERRUPT = "human_interrupt"

# Observer 决策值（与 prompts.py 中的 JSON 输出格式对应）
_DECISION_CONTINUE  = "continue"
_DECISION_COMPLETE  = "complete"
_DECISION_RETRY     = "retry"
_DECISION_HUMAN     = "human"
_DECISION_FAIL      = "fail"


# ─────────────────────────────────────────────
# 1. 辅助工具函数
# ─────────────────────────────────────────────

def _get_config(state: AgentState) -> AgentConfig:
    """从 state.metadata 中反序列化 AgentConfig。"""
    raw = state.get("metadata", {}).get("config", {})
    return AgentConfig(**raw) if raw else AgentConfig()


def _get_available_tools() -> dict[str, Any]:
    """
    获取所有可用工具的注册表映射。

    Step 22（tool_registry.py）完成后，将替换为：
        from agents.tool_registry import tool_registry
        return tool_registry.get_all()

    当前使用延迟导入，避免在 Step 13 之前因工具文件缺失而崩溃。
    """
    from agents.tools.web_search    import web_search
    from agents.tools.calculator    import calculator
    from agents.tools.rag_retrieval import rag_retrieval
    from agents.tools.code_executor import code_executor

    return {
        "web_search":    web_search,
        "calculator":    calculator,
        "rag_retrieval": rag_retrieval,
        "code_executor": code_executor,
    }


def _filter_tools(
    all_tools: dict[str, Any],
    allowed:   list[str],
) -> dict[str, Any]:
    """
    按 AgentConfig.allowed_tools 白名单过滤工具。
    空列表表示全部允许。
    """
    if not allowed:
        return all_tools
    return {k: v for k, v in all_tools.items() if k in allowed}


def _tools_detail_text(tools: dict[str, Any]) -> str:
    """
    生成供 ToolSelector 提示词使用的工具详情文本。
    尽可能复用工具的 docstring 作为说明。
    """
    lines = []
    for name, fn in tools.items():
        doc = (fn.__doc__ or "").strip().split("\n")[0]  # 取第一行摘要
        lines.append(f"  - {name}: {doc}")
    return "\n".join(lines) if lines else "  （当前无可用工具）"


def _tools_list_text(tools: dict[str, Any]) -> str:
    """生成工具名称列表文本（供 Planner 提示词使用）。"""
    return ", ".join(tools.keys()) if tools else "（无工具）"


def _parse_json_output(raw: str, node_name: str) -> dict:
    """
    安全解析 LLM 返回的 JSON 字符串。
    处理 LLM 经常附带的 ```json ... ``` 代码块包装。
    """
    text = raw.strip()

    # 剥离 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行 ```json 和末行 ```
        inner_lines = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "%s: JSON 解析失败，原始输出 = %r, error = %s",
            node_name, raw[:200], exc,
        )
        return {}


def _summarize_tool_results(tool_results: list[dict]) -> str:
    """将已有工具调用结果压缩为摘要文本，避免 Token 超限。"""
    if not tool_results:
        return "（暂无工具调用结果）"
    lines = []
    for i, r in enumerate(tool_results, start=1):
        output_preview = str(r.get("output", ""))[:300]
        if len(str(r.get("output", ""))) > 300:
            output_preview += "…"
        status = "✓" if not r.get("error") else "✗"
        lines.append(
            f"[{i}] {status} {r.get('tool', '?')}: {output_preview}"
        )
    return "\n".join(lines)


def _get_llm():
    """
    获取 LLM 实例。
    Step 6（llm_client.py）完成后，统一改为从 infrastructure.llm_client 导入。
    当前使用直接构造，保持 workflow.py 可独立运行。
    """
    try:
        from infrastructure.llm_client import get_llm_client  # type: ignore[import]
        return get_llm_client()
    except ImportError:
        from langchain_openai import ChatOpenAI
        import os
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            streaming=False,
        )


# ─────────────────────────────────────────────
# 2. 节点实现
# ─────────────────────────────────────────────

async def planner_node(state: AgentState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────┐
    │  Planner 节点                                │
    │  职责：分析用户任务，生成结构化执行计划          │
    │                                              │
    │  读取: task / memory_context                  │
    │  写入: plan / messages / status              │
    │  输出 JSON: { goal, steps[], estimated_tool_calls } │
    └─────────────────────────────────────────────┘
    """
    node_start = time.monotonic()
    task   = state.get("task", "")
    config = _get_config(state)

    logger.info("planner_node 开始 | task=%s", task[:80])

    try:
        # ── 准备工具列表文本 ────────────────────────────────────────────────
        all_tools     = _get_available_tools()
        active_tools  = _filter_tools(all_tools, config.allowed_tools)
        tools_text    = _tools_list_text(active_tools)

        # ── 记忆上下文（Step 23 前为"无"）──────────────────────────────────
        memory_ctx = state.get("memory_context") or "（暂无相关历史记忆）"

        # ── 调用 LLM 生成计划 ────────────────────────────────────────────
        llm      = _get_llm()
        prompt   = get_prompt(PromptKey.AGENT_PLANNER)
        chain    = prompt | llm | StrOutputParser()

        raw_output = await chain.ainvoke({
            "available_tools": tools_text,
            "task":            task,
            "memory_context":  str(memory_ctx),
        })

        parsed = _parse_json_output(raw_output, NODE_PLANNER)

        # ── 提取 steps，容错处理 ─────────────────────────────────────────
        steps: list[str] = parsed.get("steps", [])
        if not steps:
            # LLM 输出格式异常时，退化为单步计划
            logger.warning("planner_node: 未解析到步骤，使用退化计划")
            steps = [f"直接处理任务：{task}", "生成最终回答"]

        goal = parsed.get("goal", task)
        elapsed_ms = (time.monotonic() - node_start) * 1000

        logger.info(
            "planner_node 完成 | steps=%d | goal=%s | elapsed=%.0fms",
            len(steps), goal[:60], elapsed_ms,
        )

        # ── 写入状态 ─────────────────────────────────────────────────────
        return {
            "plan":    steps,
            "current_step": 0,
            "status":  TaskStatus.RUNNING,
            "messages": [
                make_system_message(
                    f"任务目标：{goal}\n执行计划共 {len(steps)} 步：\n"
                    + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
                ),
            ],
            "metadata": {
                **state.get("metadata", {}),
                "planner_goal": goal,
                "planner_elapsed_ms": elapsed_ms,
            },
        }

    except Exception as exc:
        logger.error("planner_node 异常: %s", exc, exc_info=True)
        return {
            "status": TaskStatus.FAILED,
            "error":  f"Planner 节点异常：{exc}",
        }


async def tool_selector_node(state: AgentState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────┐
    │  ToolSelector 节点                           │
    │  职责：为当前步骤选择最合适的工具及入参          │
    │                                              │
    │  读取: plan / current_step / tool_results    │
    │  写入: metadata["pending_tool"]              │
    │  输出 JSON: { tool_name, tool_input, reasoning } │
    └─────────────────────────────────────────────┘
    """
    node_start  = time.monotonic()
    plan        = state.get("plan", [])
    current_idx = state.get("current_step", 0)
    task        = state.get("task", "")
    config      = _get_config(state)

    # 边界检查：索引越界时回退到最后一步
    if current_idx >= len(plan):
        current_idx = max(0, len(plan) - 1)

    current_step_text = plan[current_idx] if plan else "执行任务"
    logger.info(
        "tool_selector_node 开始 | step=%d/%d | step_text=%s",
        current_idx + 1, len(plan), current_step_text[:60],
    )

    try:
        all_tools    = _get_available_tools()
        active_tools = _filter_tools(all_tools, config.allowed_tools)
        tools_detail = _tools_detail_text(active_tools)

        tool_summary = _summarize_tool_results(state.get("tool_results", []))

        llm   = _get_llm()
        prompt = get_prompt(PromptKey.AGENT_TOOL_SELECTOR)
        chain  = prompt | llm | StrOutputParser()

        raw_output = await chain.ainvoke({
            "tools_detail":       tools_detail,
            "current_step_index": current_idx + 1,
            "current_step":       current_step_text,
            "task":               task,
            "tool_results_summary": tool_summary,
        })

        parsed    = _parse_json_output(raw_output, NODE_TOOL_SELECTOR)
        tool_name = parsed.get("tool_name", "none").strip().lower()
        tool_input = parsed.get("tool_input", {})
        reasoning  = parsed.get("reasoning", "")

        elapsed_ms = (time.monotonic() - node_start) * 1000
        logger.info(
            "tool_selector_node 完成 | tool=%s | elapsed=%.0fms | reason=%s",
            tool_name, elapsed_ms, reasoning,
        )

        # 将选择结果暂存到 metadata，供 tool_executor 读取
        return {
            "metadata": {
                **state.get("metadata", {}),
                "pending_tool": {
                    "name":      tool_name,
                    "input":     tool_input,
                    "reasoning": reasoning,
                    "step_index": current_idx,
                },
            },
            "messages": [
                make_assistant_message(
                    f"[步骤 {current_idx + 1}] 选择工具：{tool_name}｜理由：{reasoning}"
                ),
            ],
        }

    except Exception as exc:
        logger.error("tool_selector_node 异常: %s", exc, exc_info=True)
        # 出错时设置为 none，让 executor 跳过工具调用
        return {
            "metadata": {
                **state.get("metadata", {}),
                "pending_tool": {
                    "name": "none", "input": {},
                    "reasoning": f"选择失败：{exc}", "step_index": current_idx,
                },
            },
        }


async def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────┐
    │  ToolExecutor 节点                           │
    │  职责：执行 tool_selector 选定的工具            │
    │                                              │
    │  读取: metadata["pending_tool"]              │
    │  写入: tool_results（追加）/ messages         │
    └─────────────────────────────────────────────┘
    """
    metadata     = state.get("metadata", {})
    pending_tool = metadata.get("pending_tool", {})
    tool_name    = pending_tool.get("name", "none")
    tool_input   = pending_tool.get("tool_input") or pending_tool.get("input", {})
    step_index   = pending_tool.get("step_index", state.get("current_step", 0))

    logger.info(
        "tool_executor_node 开始 | tool=%s | step=%d",
        tool_name, step_index + 1,
    )

    called_at  = datetime.now(timezone.utc).isoformat()
    exec_start = time.monotonic()

    # ── 无工具场景（纯推理/综合步骤）────────────────────────────────────────
    if tool_name == "none":
        elapsed_ms = (time.monotonic() - exec_start) * 1000
        result_record = {
            "tool":       "none",
            "input":      {},
            "output":     "（本步骤无需工具，为纯推理步骤）",
            "error":      None,
            "elapsed_ms": elapsed_ms,
            "called_at":  called_at,
            "step_index": step_index,
        }
        return {
            "tool_results": [result_record],
            "messages": [
                make_assistant_message(
                    f"[步骤 {step_index + 1}] 无需工具，进入推理综合阶段。"
                ),
            ],
        }

    # ── 工具调用 ────────────────────────────────────────────────────────────
    try:
        all_tools    = _get_available_tools()
        config       = _get_config(state)
        active_tools = _filter_tools(all_tools, config.allowed_tools)

        if tool_name not in active_tools:
            raise ValueError(
                f"工具「{tool_name}」不存在或不在允许列表中。"
                f"可用工具：{list(active_tools.keys())}"
            )

        tool_fn = active_tools[tool_name]

        # LangChain @tool 既支持同步也支持异步调用
        # 统一使用 ainvoke（若工具本身是同步的，LangChain 会自动 wrap）
        output = await tool_fn.ainvoke(tool_input)
        elapsed_ms = (time.monotonic() - exec_start) * 1000

        logger.info(
            "tool_executor_node 成功 | tool=%s | elapsed=%.0fms",
            tool_name, elapsed_ms,
        )

        result_record = {
            "tool":       tool_name,
            "input":      tool_input,
            "output":     output,
            "error":      None,
            "elapsed_ms": elapsed_ms,
            "called_at":  called_at,
            "step_index": step_index,
        }

        return {
            "tool_results": [result_record],
            "messages": [
                make_tool_message(
                    content=str(output)[:500],   # 消息历史中截断，完整结果在 tool_results
                    tool_name=tool_name,
                ),
            ],
        }

    except Exception as exc:
        elapsed_ms = (time.monotonic() - exec_start) * 1000
        error_msg  = f"{type(exc).__name__}: {exc}"

        logger.warning(
            "tool_executor_node 失败 | tool=%s | elapsed=%.0fms | error=%s",
            tool_name, elapsed_ms, error_msg,
        )

        result_record = {
            "tool":       tool_name,
            "input":      tool_input,
            "output":     None,
            "error":      error_msg,
            "elapsed_ms": elapsed_ms,
            "called_at":  called_at,
            "step_index": step_index,
        }

        return {
            "tool_results": [result_record],
            "messages": [
                make_tool_message(
                    content=f"工具调用失败：{error_msg}",
                    tool_name=tool_name,
                ),
            ],
        }


async def observer_node(state: AgentState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────┐
    │  Observer 节点                               │
    │  职责：观察执行结果，做出下一步决策              │
    │  这是状态机的「大脑」，所有路由逻辑从此发出      │
    │                                              │
    │  读取: plan / current_step / tool_results /  │
    │        iteration_count                       │
    │  写入: current_step / iteration_count /      │
    │        final_answer / status / error /       │
    │        interrupt                             │
    │  决策: continue / complete / retry /         │
    │        human / fail                          │
    └─────────────────────────────────────────────┘
    """
    node_start     = time.monotonic()
    task           = state.get("task", "")
    plan           = state.get("plan", [])
    current_idx    = state.get("current_step", 0)
    tool_results   = state.get("tool_results", [])
    iteration_count = state.get("iteration_count", 0) + 1  # 本次执行后 +1
    config         = _get_config(state)
    max_iterations = config.max_iterations

    logger.info(
        "observer_node 开始 | step=%d/%d | iteration=%d/%d",
        current_idx + 1, len(plan), iteration_count, max_iterations,
    )

    # ── 提取最新工具结果 ─────────────────────────────────────────────────────
    latest_result = "（无工具结果）"
    if tool_results:
        last = tool_results[-1]
        if last.get("error"):
            latest_result = f"工具 [{last['tool']}] 调用失败：{last['error']}"
        else:
            output_text = str(last.get("output", ""))
            preview = output_text[:800] + ("…" if len(output_text) > 800 else "")
            latest_result = f"工具 [{last['tool']}] 成功，输出：\n{preview}"

    # ── 迭代上限强制保护 ─────────────────────────────────────────────────────
    # 超出上限时不再询问 LLM，直接强制结束，避免死循环
    if iteration_count >= max_iterations:
        logger.warning(
            "observer_node: 达到最大迭代次数 %d，强制进入完成/失败分支",
            max_iterations,
        )
        # 用已有工具结果生成一个降级回答
        summary = _summarize_tool_results(tool_results)
        forced_answer = (
            f"（已达最大执行轮次 {max_iterations}，基于已收集信息生成回答）\n\n"
            f"任务：{task}\n\n"
            f"已收集信息摘要：\n{summary}"
        )
        return {
            "iteration_count": iteration_count,
            "final_answer":    forced_answer,
            "status":          TaskStatus.COMPLETED,
            "metadata": {
                **state.get("metadata", {}),
                "observer_decision":      "complete_forced",
                "observer_elapsed_ms":    (time.monotonic() - node_start) * 1000,
                "force_complete_reason":  "max_iterations_reached",
            },
        }

    # ── 调用 LLM 做决策 ──────────────────────────────────────────────────────
    try:
        plan_text = "\n".join(
            f"  {'✓' if i < current_idx else ('→' if i == current_idx else '○')} "
            f"步骤 {i+1}：{s}"
            for i, s in enumerate(plan)
        ) if plan else "（无执行计划）"

        current_step_text = plan[current_idx] if current_idx < len(plan) else "（步骤越界）"

        llm   = _get_llm()
        prompt = get_prompt(PromptKey.AGENT_OBSERVER)
        chain  = prompt | llm | StrOutputParser()

        raw_output = await chain.ainvoke({
            "task":               task,
            "plan":               plan_text,
            "current_step_index": current_idx + 1,
            "current_step":       current_step_text,
            "completed_steps":    current_idx,
            "total_steps":        len(plan),
            "iteration_count":    iteration_count,
            "max_iterations":     max_iterations,
            "latest_tool_result": latest_result,
        })

        parsed   = _parse_json_output(raw_output, NODE_OBSERVER)
        decision = parsed.get("decision", _DECISION_FAIL).lower().strip()
        reasoning     = parsed.get("reasoning", "")
        final_answer  = parsed.get("final_answer", "")
        question      = parsed.get("question", "")
        error_message = parsed.get("error_message", "")

    except Exception as exc:
        logger.error("observer_node: LLM 调用失败: %s", exc, exc_info=True)
        decision      = _DECISION_FAIL
        reasoning     = f"Observer LLM 调用异常：{exc}"
        final_answer  = ""
        question      = ""
        error_message = str(exc)

    elapsed_ms = (time.monotonic() - node_start) * 1000
    logger.info(
        "observer_node 决策: %s | step=%d | reason=%s | elapsed=%.0fms",
        decision, current_idx + 1, reasoning[:60], elapsed_ms,
    )

    # ── 根据决策构建状态更新 ─────────────────────────────────────────────────
    base_update: dict[str, Any] = {
        "iteration_count": iteration_count,
        "messages": [
            make_assistant_message(
                f"[Observer] 决策：{decision}｜{reasoning}"
            ),
        ],
        "metadata": {
            **state.get("metadata", {}),
            "observer_decision":   decision,
            "observer_reasoning":  reasoning,
            "observer_elapsed_ms": elapsed_ms,
        },
    }

    if decision == _DECISION_CONTINUE:
        # 推进到下一步（循环回 tool_selector）
        return {
            **base_update,
            "current_step": current_idx + 1,
        }

    elif decision == _DECISION_RETRY:
        # 保持 current_step 不变，循环回 tool_selector 重试同一步
        return {
            **base_update,
            "current_step": current_idx,   # 不推进
        }

    elif decision == _DECISION_COMPLETE:
        return {
            **base_update,
            "final_answer": final_answer or "（任务已完成，无额外输出）",
            "status":       TaskStatus.COMPLETED,
        }

    elif decision == _DECISION_HUMAN:
        interrupt = InterruptPayload(
            interrupt_reason=reasoning,
            question=question or "请确认是否继续执行？",
        )
        return {
            **base_update,
            "interrupt": interrupt,
            "status":    TaskStatus.WAITING_HUMAN,
        }

    else:  # fail 或未知决策
        return {
            **base_update,
            "error":  error_message or reasoning or f"任务失败（决策：{decision}）",
            "status": TaskStatus.FAILED,
        }


async def human_interrupt_node(state: AgentState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────┐
    │  HumanInterrupt 节点                         │
    │  职责：挂起等待人工响应，恢复后将响应内容         │
    │        注入 messages，以便 Observer 继续决策   │
    │                                              │
    │  读取: interrupt（InterruptPayload）          │
    │  写入: messages（注入人工响应）/ interrupt=None │
    │        status=RUNNING                        │
    │                                              │
    │  ⚠️ 实际挂起机制由 LangGraph interrupt()      │
    │     配合外部 WebSocket 通知实现（Step 16）     │
    │     此节点负责「恢复后的状态处理」              │
    └─────────────────────────────────────────────┘
    """
    interrupt = state.get("interrupt")

    if interrupt is None:
        # 正常不应到达此分支
        logger.warning("human_interrupt_node: interrupt 为 None，跳过")
        return {"status": TaskStatus.RUNNING}

    if not interrupt.is_responded():
        # 尚未收到人工响应，继续挂起
        # （LangGraph interrupt() 机制会阻止图继续推进，直到外部更新状态）
        logger.info(
            "human_interrupt_node: 等待人工响应 | question=%s",
            interrupt.question[:80],
        )
        return {}  # 返回空字典，状态不变，LangGraph 会在此节点暂停

    # ── 人工已响应，将响应注入消息历史 ─────────────────────────────────────
    human_response = interrupt.human_response
    logger.info(
        "human_interrupt_node: 收到人工响应 | response=%s",
        str(human_response)[:80],
    )

    return {
        "interrupt": None,            # 清除中断标记
        "status":    TaskStatus.RUNNING,
        "messages": [
            make_user_message(
                content=f"[人工确认] {human_response}",
            ),
        ],
    }


# ─────────────────────────────────────────────
# 3. 条件边路由函数
# ─────────────────────────────────────────────

def _route_after_planner(
    state: AgentState,
) -> Literal["tool_selector"]:
    """
    Planner 之后永远进入 ToolSelector。
    （若 Planner 失败，status=FAILED，但 LangGraph 仍按此路由，
      tool_selector 会因 plan 为空而快速跳过）
    """
    return NODE_TOOL_SELECTOR


def _route_after_observer(
    state: AgentState,
) -> Literal[
    "tool_selector",
    "human_interrupt",
    "__end__",
]:
    """
    Observer 节点之后的路由决策。

    路由规则（优先级从高到低）：
    1. FAILED / COMPLETED → END（终止图执行）
    2. WAITING_HUMAN      → human_interrupt
    3. current_step 越界  → END（所有步骤已完成）
    4. 其他（RUNNING）    → tool_selector（推进到下一步或重试）

    ⚠️ 注意：路由函数读取的是 observer_node 写入后的最新状态
    """
    status      = state.get("status")
    current_idx = state.get("current_step", 0)
    plan        = state.get("plan", [])

    # 终止条件
    if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        logger.info("_route_after_observer → END (status=%s)", status)
        return END

    # Human-in-the-Loop
    if status == TaskStatus.WAITING_HUMAN:
        logger.info("_route_after_observer → human_interrupt")
        return NODE_HUMAN_INTERRUPT

    # 计划步骤全部完成（current_step 已超出 plan 长度）
    if current_idx >= len(plan):
        logger.info(
            "_route_after_observer → END (all steps done, step=%d >= total=%d)",
            current_idx, len(plan),
        )
        return END

    # 继续执行（continue 或 retry）
    logger.info(
        "_route_after_observer → tool_selector (step=%d/%d)",
        current_idx + 1, len(plan),
    )
    return NODE_TOOL_SELECTOR


def _route_after_human_interrupt(
    state: AgentState,
) -> Literal["tool_selector", "human_interrupt"]:
    """
    HumanInterrupt 节点之后的路由：
    - 若人工已响应 → 回到 tool_selector 继续执行
    - 若人工未响应（interrupt 仍存在）→ 留在 human_interrupt（继续等待）
    """
    interrupt = state.get("interrupt")
    if interrupt is not None and not interrupt.is_responded():
        return NODE_HUMAN_INTERRUPT   # 继续等待
    return NODE_TOOL_SELECTOR         # 恢复执行


# ─────────────────────────────────────────────
# 4. 图构建与编译
# ─────────────────────────────────────────────

def _build_graph(checkpointer=None) -> Any:
    """
    构建并编译 LangGraph StateGraph。

    图结构（用「地铁线路图」类比）：
    - 节点 = 站台（每站有明确的职责）
    - 边   = 铁路（固定路线）
    - 条件边 = 分叉道（根据状态动态选择路线）

    MemorySaver 的作用：
    - 在每次节点执行后，将完整 AgentState 快照到内存
    - 支持断点恢复（如 human_interrupt 后 resume）
    - 生产环境可替换为 RedisSaver 实现跨进程持久化
    """
    graph = StateGraph(AgentState)

    # ── 注册所有节点 ────────────────────────────────────────────────────────
    graph.add_node(NODE_PLANNER,         planner_node)
    graph.add_node(NODE_TOOL_SELECTOR,   tool_selector_node)
    graph.add_node(NODE_TOOL_EXECUTOR,   tool_executor_node)
    graph.add_node(NODE_OBSERVER,        observer_node)
    graph.add_node(NODE_HUMAN_INTERRUPT, human_interrupt_node)

    # ── 设置入口节点 ─────────────────────────────────────────────────────────
    graph.set_entry_point(NODE_PLANNER)

    # ── 固定边 ───────────────────────────────────────────────────────────────
    # planner → tool_selector（始终）
    graph.add_edge(NODE_PLANNER, NODE_TOOL_SELECTOR)

    # tool_selector → tool_executor（始终）
    graph.add_edge(NODE_TOOL_SELECTOR, NODE_TOOL_EXECUTOR)

    # tool_executor → observer（始终）
    graph.add_edge(NODE_TOOL_EXECUTOR, NODE_OBSERVER)

    # ── 条件边（从 observer 分叉）────────────────────────────────────────────
    graph.add_conditional_edges(
        source=NODE_OBSERVER,
        path=_route_after_observer,
        path_map={
            NODE_TOOL_SELECTOR:   NODE_TOOL_SELECTOR,
            NODE_HUMAN_INTERRUPT: NODE_HUMAN_INTERRUPT,
            END:                  END,
        },
    )

    # ── 条件边（从 human_interrupt 恢复）──────────────────────────────────────
    graph.add_conditional_edges(
        source=NODE_HUMAN_INTERRUPT,
        path=_route_after_human_interrupt,
        path_map={
            NODE_TOOL_SELECTOR:   NODE_TOOL_SELECTOR,
            NODE_HUMAN_INTERRUPT: NODE_HUMAN_INTERRUPT,
        },
    )

    # ── 编译（注入 checkpointer 用于断点续跑）────────────────────────────────
    return graph.compile(checkpointer=checkpointer)


# 模块级单例：使用 MemorySaver（开发/测试）
# 生产环境通过 build_workflow_with_redis() 替换
_memory_saver    = MemorySaver()
_compiled_graph  = _build_graph(checkpointer=_memory_saver)


def build_workflow_with_redis(redis_url: str) -> Any:
    """
    使用 Redis 持久化替换 MemorySaver，返回新的编译图。

    适用于生产环境，支持跨进程断点恢复（Celery Worker 重启后可继续）。

    使用方式：
        from agents.workflow import build_workflow_with_redis
        graph = build_workflow_with_redis("redis://localhost:6379")
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": session_id}})

    ⚠️ 需要安装 langgraph-checkpoint-redis：
        pip install langgraph-checkpoint-redis
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore[import]
        redis_saver = RedisSaver.from_conn_string(redis_url)
        return _build_graph(checkpointer=redis_saver)
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-redis 未安装，回退到 MemorySaver。"
            "执行: pip install langgraph-checkpoint-redis"
        )
        return _compiled_graph


# ─────────────────────────────────────────────
# 5. 公开入口函数
# ─────────────────────────────────────────────

async def run_workflow(
    task:       str,
    session_id: Optional[str]  = None,
    user_id:    Optional[str]  = None,
    config:     Optional[AgentConfig] = None,
    trace_id:   Optional[str]  = None,
    graph:      Optional[Any]  = None,
) -> AgentState:
    """
    单 Agent 工作流的统一异步入口。

    对外隐藏 LangGraph 的 graph.ainvoke / config 细节，
    调用方只需传入任务描述和可选配置。

    Args:
        task:       用户任务描述（自然语言）
        session_id: 会话 ID（用于 LangGraph thread_id，实现同一会话的断点续跑）
        user_id:    用户 ID（记录到 metadata 供审计使用）
        config:     Agent 运行配置（模式/工具白名单/迭代上限等），None 时使用默认值
        trace_id:   链路追踪 ID（None 时自动生成 UUID）
        graph:      自定义图实例（None 时使用模块级单例，方便测试注入 Mock）

    Returns:
        执行完毕后的最终 AgentState（包含 final_answer / status / tool_results 等）

    使用示例：
        from agents.workflow import run_workflow, AgentConfig, AgentMode

        result = await run_workflow(
            task="查找并汇总公司最新的差旅报销政策",
            session_id="sess-abc123",
            config=AgentConfig(
                mode=AgentMode.SINGLE,
                enable_human_loop=True,
                allowed_tools=["rag_retrieval", "web_search"],
            ),
        )
        print(result["final_answer"])
        print(result["status"])           # TaskStatus.COMPLETED
        print(len(result["tool_results"])) # 工具调用次数
    """
    # ── 构造初始状态 ──────────────────────────────────────────────────────
    sid = session_id or str(uuid.uuid4())
    state = initial_state(
        task=task,
        config=config,
        session_id=sid,
        user_id=user_id,
        trace_id=trace_id,
    )

    # ── LangGraph 运行配置（thread_id 决定哪条「记忆线」）────────────────────
    cfg = _get_config(state)
    run_config: RunnableConfig = {
        "configurable": {
            "thread_id": sid,       # 同一 session_id 共享状态快照
        },
        "recursion_limit": cfg.max_iterations * 4 + 10,
        # 每次迭代最多经过 4 个节点（selector → executor → observer → selector…）
        # 额外 +10 作为安全余量
    }

    active_graph = graph or _compiled_graph

    logger.info(
        "run_workflow 启动 | session_id=%s | task=%s | max_iter=%d",
        sid, task[:80], cfg.max_iterations,
    )

    start_time = time.monotonic()

    try:
        final_state: AgentState = await active_graph.ainvoke(state, config=run_config)
        elapsed = time.monotonic() - start_time

        logger.info(
            "run_workflow 完成 | session_id=%s | status=%s | "
            "iterations=%d | elapsed=%.2fs",
            sid,
            final_state.get("status"),
            final_state.get("iteration_count", 0),
            elapsed,
        )

        return final_state

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "run_workflow 异常 | session_id=%s | elapsed=%.2fs | error=%s",
            sid, elapsed, exc,
            exc_info=True,
        )
        # 返回一个标记为 FAILED 的状态，而不是向上抛出异常
        # 保证调用方（api/agent.py / Celery Worker）总能拿到结构化结果
        return AgentState(
            messages=state.get("messages", []),
            task=task,
            plan=state.get("plan", []),
            current_step=state.get("current_step", 0),
            tool_results=state.get("tool_results", []),
            final_answer="",
            status=TaskStatus.FAILED,
            error=f"工作流执行异常：{type(exc).__name__}: {exc}",
            iteration_count=state.get("iteration_count", 0),
            interrupt=None,
            memory_context=None,
            metadata={
                **state.get("metadata", {}),
                "workflow_error":    str(exc),
                "workflow_elapsed_s": elapsed,
            },
        )


async def resume_workflow(
    session_id:     str,
    human_response: str,
    graph:          Optional[Any] = None,
) -> AgentState:
    """
    在 Human-in-the-Loop 中断后，携带人工响应恢复工作流执行。

    工作流程：
    1. 从 MemorySaver（或 RedisSaver）中加载 session_id 对应的最新快照
    2. 将 human_response 写入 interrupt.human_response
    3. 重新触发 graph.ainvoke，从 human_interrupt_node 继续执行

    Args:
        session_id:     原始任务的会话 ID（必须与 run_workflow 时一致）
        human_response: 人工输入的确认/回答内容
        graph:          自定义图实例（None 时使用模块级单例）

    Returns:
        恢复执行后的最终 AgentState
    """
    active_graph = graph or _compiled_graph

    # 从 checkpointer 加载最新状态快照
    run_config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 100,
    }

    try:
        # 获取当前挂起的状态
        current_snapshot = await active_graph.aget_state(run_config)
        if current_snapshot is None:
            raise ValueError(f"找不到 session_id={session_id} 的状态快照，无法恢复。")

        current_values: AgentState = current_snapshot.values
        interrupt = current_values.get("interrupt")

        if interrupt is None:
            raise ValueError(f"session_id={session_id} 当前没有待处理的人工中断。")

        # 填入人工响应
        responded_interrupt = InterruptPayload(
            interrupt_id=interrupt.interrupt_id,
            interrupt_reason=interrupt.interrupt_reason,
            question=interrupt.question,
            options=interrupt.options,
            human_response=human_response,
            responded_at=datetime.now(timezone.utc),
            created_at=interrupt.created_at,
        )

        # 更新状态并继续执行
        await active_graph.aupdate_state(
            run_config,
            {"interrupt": responded_interrupt, "status": TaskStatus.RUNNING},
        )

        # 从断点恢复执行（传入 None 触发从上次断点继续）
        final_state: AgentState = await active_graph.ainvoke(None, config=run_config)

        logger.info(
            "resume_workflow 完成 | session_id=%s | status=%s",
            session_id, final_state.get("status"),
        )
        return final_state

    except Exception as exc:
        logger.error(
            "resume_workflow 异常 | session_id=%s | error=%s",
            session_id, exc,
            exc_info=True,
        )
        raise
