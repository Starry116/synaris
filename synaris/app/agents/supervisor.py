"""
@File       : supervisor.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: LangGraph 多 Agent Supervisor 编排器。
@Features:
  - SupervisorState(TypedDict): 独立于 AgentState 的 Supervisor 专属状态容器，
    承载任务分解计划、Worker 执行结果、修订轮次等
  - AssignmentRecord: 单条 Worker 任务分配记录（含依赖关系 depends_on）
  - WorkerResult: 单个 Worker 的执行结果（含评审分数、verdict）
  - 3 个图节点:
      · decompose_node  — LLM 将主任务分解为结构化 assignments[]
      · dispatch_node   — 依赖感知调度器，串行/并行执行各 Worker
      · merge_node      — 整合所有 Worker 结果，生成最终交付内容
  - 两种调度模式（执行模式由 LLM 自动决定，也可 override）:
      · sequential — 严格按 depends_on 拓扑顺序逐个执行
      · parallel   — 同一依赖层级的 Worker 并发 asyncio.gather() 执行
  - 依赖感知调度算法: 拓扑排序 + 层级批次执行，同层并发，跨层串行
  - ReviewerAgent 触发重写机制:
      verdict=needs_revision → 自动追加新 WriterAgent 任务（最多 max_revisions 次）
      verdict=rejected       → 记录问题后继续（不阻塞整体流程）
  - 统一错误处理: Worker 失败不终止整体，降级为错误占位符继续 merge
  - run_supervisor() 异步入口: 对外统一调用接口

  ┌────────────────────────────────────────────────────────────────────────┐
  │                    Supervisor 状态机完整流转图                           │
  │                                                                         │
  │   ┌─────────┐                                                           │
  │   │  START  │                                                           │
  │   └────┬────┘                                                           │
  │        │ run_supervisor(task)                                           │
  │        ▼                                                                │
  │   ┌──────────────────┐                                                  │
  │   │  decompose_node  │  LLM 分析主任务                                  │
  │   │                  │  → assignments[] (agent/instruction/depends_on)  │
  │   │                  │  → execution_mode (sequential/parallel)          │
  │   └────────┬─────────┘                                                  │
  │            │ 固定边                                                      │
  │            ▼                                                            │
  │   ┌──────────────────┐                                                  │
  │   │  dispatch_node   │  依赖感知调度器                                   │
  │   │                  │  ┌─ parallel ─┐  同层任务 asyncio.gather()        │
  │   │                  │  │ Layer 0    │→ [ResearchAgent, ...]            │
  │   │                  │  │ Layer 1    │→ [WriterAgent] (依赖 Layer 0)    │
  │   │                  │  │ Layer 2    │→ [ReviewerAgent] (依赖 Layer 1)  │
  │   │                  │  └────────────┘                                  │
  │   └────────┬─────────┘                                                  │
  │            │                                                            │
  │     ┌──────┴───────────────────────────┐                               │
  │     │                                  │                               │
  │  verdict=needs_revision           all_done=True                        │
  │  revision_count < max             (或 revision 上限)                    │
  │     │                                  │                               │
  │     │  追加新 WriterAgent 任务           │ 固定边                        │
  │     └──→ dispatch_node (再次执行)       ▼                               │
  │                                  ┌──────────────┐                      │
  │                                  │  merge_node  │  LLM 整合所有结果     │
  │                                  │              │  → final_output       │
  │                                  └──────┬───────┘                      │
  │                                         │ 固定边                        │
  │                                         ▼                              │
  │                                      ┌─────┐                           │
  │                                      │ END │                           │
  │                                      └─────┘                           │
  └────────────────────────────────────────────────────────────────────────┘

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agents.state import (
    AgentMessage,
    AgentState,
    TaskStatus,
    make_assistant_message,
    make_system_message,
    make_user_message,
)
from core.prompts import PromptKey, get_prompt

logger = logging.getLogger(__name__)

# ── 常量 ───────────────────────────────────────────────────────────────────────
_MAX_REVISIONS   = 2       # ReviewerAgent 触发 WriterAgent 重写的最大次数
_MAX_ASSIGNMENTS = 10      # 单次 Supervisor 分配的最大子任务数（防止 LLM 滥分）
_WORKER_TIMEOUT  = 120     # 单个 Worker 执行超时（秒）

# 节点名称常量
NODE_DECOMPOSE = "decompose"
NODE_DISPATCH  = "dispatch"
NODE_MERGE     = "merge"

# Worker 名称常量（与 prompts.py 中的 agent 字段值对应）
AGENT_RESEARCH  = "ResearchAgent"
AGENT_WRITER    = "WriterAgent"
AGENT_REVIEWER  = "ReviewerAgent"

# ReviewerAgent verdict 值
VERDICT_APPROVED        = "approved"
VERDICT_NEEDS_REVISION  = "needs_revision"
VERDICT_REJECTED        = "rejected"


# ─────────────────────────────────────────────
# 1. 数据模型
# ─────────────────────────────────────────────

class AssignmentStatus(str):
    """Assignment 执行状态常量。"""
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class AssignmentRecord(BaseModel):
    """
    单条 Worker 任务分配记录。

    类比项目管理中的「子任务卡」：
      index       → 卡片编号（0-indexed）
      agent       → 负责人（哪个 Worker）
      instruction → 任务说明
      depends_on  → 前置任务编号（必须完成后才能开始此任务）
      priority    → 优先级（5最高，影响并行层内的执行顺序）
      status      → 当前执行状态
      result      → 执行产出
      error       → 若失败，记录错误信息
    """
    index:       int         = 0
    agent:       str                              # ResearchAgent | WriterAgent | ReviewerAgent
    instruction: str
    depends_on:  list[int]   = Field(default_factory=list)
    priority:    int         = Field(default=3, ge=1, le=5)
    status:      str         = AssignmentStatus.PENDING
    result:      Optional[str] = None
    error:       Optional[str] = None
    started_at:  Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_ms:  Optional[float] = None
    is_revision: bool        = False   # 是否为 Reviewer 触发的重写任务


class WorkerResult(BaseModel):
    """
    单个 Worker 执行结果的结构化容器。

    ReviewerAgent 的结果还附带评审分数与 verdict，
    供 dispatch_node 决定是否触发重写。
    """
    assignment_index: int
    agent:            str
    instruction:      str
    output:           str
    error:            Optional[str]   = None
    # ReviewerAgent 专属字段
    verdict:          Optional[str]   = None   # approved | needs_revision | rejected
    total_score:      Optional[int]   = None
    issues:           list[dict]      = Field(default_factory=list)
    is_revision:      bool            = False
    elapsed_ms:       float           = 0.0


class SupervisorState(TypedDict, total=False):
    """
    Supervisor 专属状态容器（独立于单 Agent 的 AgentState）。

    设计类比「项目看板」：
    ┌──────────────────────────────────────────────────────────┐
    │  task            → 项目名称（主任务描述）                  │
    │  user_context    → 客户背景（用户画像/偏好）               │
    │  memory_context  → 历史档案（跨会话记忆）                  │
    │  analysis        → 项目分析报告（Supervisor 的理解）       │
    │  execution_mode  → 工作方式（流水线/并行团队）             │
    │  assignments     → 任务卡列表（所有子任务分配计划）         │
    │  merge_strategy  → 整合方案（最终如何汇总各产出）           │
    │  worker_results  → 完工报告列表（Worker 的输出集合）       │
    │  revision_count  → 返修次数（Reviewer 触发重写的计数）     │
    │  final_output    → 最终交付物（merge 后的完整内容）         │
    │  status          → 项目状态（RUNNING/COMPLETED/FAILED）   │
    │  error           → 异常记录                               │
    │  messages        → 沟通记录（append-only，并发安全）       │
    │  metadata        → 项目元数据（trace_id/timing 等）        │
    └──────────────────────────────────────────────────────────┘
    """

    # ── 输入 ──────────────────────────────────────────────────
    task:           str
    user_context:   str
    memory_context: Optional[Any]

    # ── 分解结果（decompose_node 写入）────────────────────────
    analysis:        str
    execution_mode:  str               # "sequential" | "parallel"
    assignments:     list[AssignmentRecord]
    merge_strategy:  str

    # ── 执行结果（dispatch_node 写入）─────────────────────────
    worker_results:  list[WorkerResult]
    revision_count:  int

    # ── 最终输出（merge_node 写入）────────────────────────────
    final_output:    str
    status:          TaskStatus
    error:           Optional[str]

    # ── 通用字段 ───────────────────────────────────────────────
    messages:        Annotated[list[AgentMessage], operator.add]
    metadata:        dict[str, Any]


# ─────────────────────────────────────────────
# 2. 辅助函数
# ─────────────────────────────────────────────

def _parse_json(raw: str, node_name: str) -> dict:
    """安全解析 LLM 返回的 JSON（处理 ```json 代码块包装）。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("%s JSON 解析失败 | raw=%r | err=%s", node_name, raw[:200], exc)
        return {}


def _get_llm(temperature: float = 0.0):
    """
    获取 LLM 实例（延迟导入，Step 6 完成前可独立运行）。
    Supervisor 使用 gpt-4o（质量模式），对应 LLM Router 的 QUALITY 策略。
    """
    try:
        from infrastructure.llm_client import get_llm_client  # type: ignore[import]
        return get_llm_client(strategy="quality")
    except ImportError:
        from langchain_openai import ChatOpenAI
        import os
        return ChatOpenAI(
            model=os.getenv("SUPERVISOR_MODEL", "gpt-4o"),
            temperature=temperature,
        )


def _collect_dependency_results(
    assignment: AssignmentRecord,
    all_assignments: list[AssignmentRecord],
    worker_results: list[WorkerResult],
) -> str:
    """
    收集指定 assignment 的所有前置依赖的执行结果，
    拼接为可注入提示词的上下文字符串。

    类比「交接班记录」：当前 Worker 开工前，先读取所有前任的产出。
    """
    if not assignment.depends_on:
        return "（本任务无前置依赖）"

    parts = []
    result_map = {r.assignment_index: r for r in worker_results}

    for dep_idx in assignment.depends_on:
        dep_result = result_map.get(dep_idx)
        if dep_result:
            dep_assignment = next(
                (a for a in all_assignments if a.index == dep_idx), None
            )
            agent_label = dep_assignment.agent if dep_assignment else f"Task-{dep_idx}"
            output_preview = dep_result.output[:2000] + (
                "…" if len(dep_result.output) > 2000 else ""
            )
            parts.append(
                f"--- 来自 {agent_label}（任务 {dep_idx + 1}）的输出 ---\n"
                f"{output_preview}"
            )
        else:
            parts.append(f"--- 任务 {dep_idx + 1} 尚未完成（依赖缺失）---")

    return "\n\n".join(parts)


def _build_execution_layers(
    assignments: list[AssignmentRecord],
) -> list[list[AssignmentRecord]]:
    """
    拓扑排序 + 层级划分算法。

    将 assignments 按依赖关系划分为有序的「执行层」：
    - 同一层内的任务互不依赖，可以并行执行
    - 层与层之间严格串行（下一层必须等上一层全部完成）

    类比建筑施工：
    - 第 0 层：打地基（无前置）→ 可同时开多个工队并行
    - 第 1 层：砌墙（依赖地基）→ 等地基完成后才能开始
    - 第 2 层：装修（依赖砌墙）→ 以此类推

    算法：Kahn's Algorithm（BFS 拓扑排序变种）
    """
    if not assignments:
        return []

    # 计算每个 assignment 的入度（被依赖的次数）
    index_set    = {a.index for a in assignments}
    in_degree    = {a.index: 0 for a in assignments}
    adj          = {a.index: [] for a in assignments}   # index → 依赖它的任务

    for a in assignments:
        for dep in a.depends_on:
            if dep in index_set:
                in_degree[a.index] += 1
                adj[dep].append(a.index)

    index_map = {a.index: a for a in assignments}
    layers: list[list[AssignmentRecord]] = []
    ready = [idx for idx, deg in in_degree.items() if deg == 0]

    while ready:
        # 当前层：所有入度为 0 的任务，按 priority 降序排列（高优先级先）
        layer = sorted(
            [index_map[idx] for idx in ready],
            key=lambda a: -a.priority,
        )
        layers.append(layer)

        next_ready = []
        for a in layer:
            for neighbor in adj[a.index]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_ready.append(neighbor)
        ready = next_ready

    return layers


async def _call_worker(
    assignment:     AssignmentRecord,
    all_assignments: list[AssignmentRecord],
    worker_results:  list[WorkerResult],
    main_task:       str,
) -> WorkerResult:
    """
    调用对应的 Worker Agent 执行子任务。

    依赖关系注入策略：
    - 将所有前置任务的结果拼接为上下文
    - 注入到 Worker 的 instruction 或特定变量中

    Worker 模块延迟导入（workers.py 为 Step 15 第二部分）。
    """
    # 收集前置依赖输出
    dep_context = _collect_dependency_results(
        assignment, all_assignments, worker_results
    )

    start_time = time.monotonic()
    assignment.status     = AssignmentStatus.RUNNING
    assignment.started_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        "_call_worker 开始 | agent=%s | index=%d | deps=%s",
        assignment.agent, assignment.index, assignment.depends_on,
    )

    try:
        # 延迟导入 workers（Step 15 第二部分）
        from agents.workers import (  # type: ignore[import]
            run_research_agent,
            run_writer_agent,
            run_reviewer_agent,
        )

        if assignment.agent == AGENT_RESEARCH:
            output = await asyncio.wait_for(
                run_research_agent(
                    instruction=assignment.instruction,
                    main_task=main_task,
                ),
                timeout=_WORKER_TIMEOUT,
            )
            result = WorkerResult(
                assignment_index=assignment.index,
                agent=assignment.agent,
                instruction=assignment.instruction,
                output=output,
                is_revision=assignment.is_revision,
                elapsed_ms=(time.monotonic() - start_time) * 1000,
            )

        elif assignment.agent == AGENT_WRITER:
            # WriterAgent 需要研究结果作为输入
            research_context = dep_context
            output = await asyncio.wait_for(
                run_writer_agent(
                    instruction=assignment.instruction,
                    research_output=research_context,
                    main_task=main_task,
                ),
                timeout=_WORKER_TIMEOUT,
            )
            result = WorkerResult(
                assignment_index=assignment.index,
                agent=assignment.agent,
                instruction=assignment.instruction,
                output=output,
                is_revision=assignment.is_revision,
                elapsed_ms=(time.monotonic() - start_time) * 1000,
            )

        elif assignment.agent == AGENT_REVIEWER:
            # ReviewerAgent 需要：待审内容 + 研究参考
            content_to_review = dep_context   # 包含 WriterAgent 的产出
            # 同时查找 ResearchAgent 的原始产出作为参考
            research_ref = next(
                (r.output for r in worker_results if r.agent == AGENT_RESEARCH),
                "（无原始研究资料）",
            )
            raw_output = await asyncio.wait_for(
                run_reviewer_agent(
                    instruction=assignment.instruction,
                    content_to_review=content_to_review,
                    research_reference=research_ref,
                    main_task=main_task,
                ),
                timeout=_WORKER_TIMEOUT,
            )

            # 解析 ReviewerAgent 的 JSON 输出
            parsed = _parse_json(raw_output, AGENT_REVIEWER)
            verdict     = parsed.get("verdict", VERDICT_APPROVED)
            total_score = parsed.get("total_score", 0)
            issues      = parsed.get("issues", [])

            # 若 needs_revision，将修改建议附加到输出中供后续 WriterAgent 使用
            output = raw_output
            if verdict == VERDICT_NEEDS_REVISION:
                revised = parsed.get("revised_sections", "")
                issues_text = "\n".join(
                    f"  [{i['severity']}] {i['description']} → {i['suggestion']}"
                    for i in issues
                ) if issues else "无"
                output = (
                    f"评审结论：{verdict}（{total_score}分）\n\n"
                    f"主要问题：\n{issues_text}\n\n"
                    f"修改建议片段：\n{revised}"
                )

            result = WorkerResult(
                assignment_index=assignment.index,
                agent=assignment.agent,
                instruction=assignment.instruction,
                output=output,
                verdict=verdict,
                total_score=total_score,
                issues=issues,
                is_revision=assignment.is_revision,
                elapsed_ms=(time.monotonic() - start_time) * 1000,
            )

        else:
            raise ValueError(f"未知 Worker 类型：{assignment.agent}")

        elapsed_ms = (time.monotonic() - start_time) * 1000
        assignment.status      = AssignmentStatus.COMPLETED
        assignment.result      = result.output[:200]
        assignment.finished_at = datetime.now(timezone.utc).isoformat()
        assignment.elapsed_ms  = elapsed_ms

        logger.info(
            "_call_worker 完成 | agent=%s | index=%d | elapsed=%.0fms | verdict=%s",
            assignment.agent, assignment.index, elapsed_ms,
            result.verdict or "N/A",
        )
        return result

    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        error_msg  = f"Worker 执行超时（>{_WORKER_TIMEOUT}s）"
        assignment.status      = AssignmentStatus.FAILED
        assignment.error       = error_msg
        assignment.finished_at = datetime.now(timezone.utc).isoformat()
        assignment.elapsed_ms  = elapsed_ms

        logger.warning(
            "_call_worker 超时 | agent=%s | index=%d | elapsed=%.0fms",
            assignment.agent, assignment.index, elapsed_ms,
        )
        return WorkerResult(
            assignment_index=assignment.index,
            agent=assignment.agent,
            instruction=assignment.instruction,
            output=f"[{assignment.agent} 执行超时，无法提供输出]",
            error=error_msg,
            elapsed_ms=elapsed_ms,
        )

    except ModuleNotFoundError:
        # workers.py 尚未实现（Step 15 第二部分之前）
        elapsed_ms = (time.monotonic() - start_time) * 1000
        stub_output = (
            f"[{assignment.agent} 存根输出]\n"
            f"任务：{assignment.instruction}\n"
            f"（workers.py 尚未实现，Step 15 第二部分完成后此处将返回真实输出）"
        )
        assignment.status      = AssignmentStatus.COMPLETED
        assignment.result      = stub_output[:200]
        assignment.finished_at = datetime.now(timezone.utc).isoformat()
        assignment.elapsed_ms  = elapsed_ms

        logger.warning(
            "_call_worker: workers.py 未就绪，使用存根输出 | agent=%s",
            assignment.agent,
        )
        return WorkerResult(
            assignment_index=assignment.index,
            agent=assignment.agent,
            instruction=assignment.instruction,
            output=stub_output,
            elapsed_ms=elapsed_ms,
        )

    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        error_msg  = f"{type(exc).__name__}: {exc}"
        assignment.status      = AssignmentStatus.FAILED
        assignment.error       = error_msg
        assignment.finished_at = datetime.now(timezone.utc).isoformat()
        assignment.elapsed_ms  = elapsed_ms

        logger.error(
            "_call_worker 异常 | agent=%s | index=%d | error=%s",
            assignment.agent, assignment.index, error_msg,
            exc_info=True,
        )
        return WorkerResult(
            assignment_index=assignment.index,
            agent=assignment.agent,
            instruction=assignment.instruction,
            output=f"[{assignment.agent} 执行失败：{error_msg}]",
            error=error_msg,
            elapsed_ms=elapsed_ms,
        )


# ─────────────────────────────────────────────
# 3. 节点实现
# ─────────────────────────────────────────────

async def decompose_node(state: SupervisorState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  Decompose 节点（任务分解）                               │
    │  职责：调用 LLM（gpt-4o）将主任务分解为结构化 assignments │
    │                                                          │
    │  读取: task / user_context / memory_context              │
    │  写入: analysis / execution_mode / assignments /         │
    │        merge_strategy / messages / status                │
    │                                                          │
    │  输出 JSON:                                              │
    │    { analysis, execution_mode,                           │
    │      assignments[{agent, instruction,                    │
    │                   depends_on, priority}],                │
    │      merge_strategy }                                    │
    └─────────────────────────────────────────────────────────┘
    """
    node_start   = time.monotonic()
    task         = state.get("task", "")
    user_context = state.get("user_context") or "（未提供用户背景信息）"
    memory_ctx   = state.get("memory_context") or "（无相关历史记忆）"

    logger.info("decompose_node 开始 | task=%s", task[:80])

    try:
        llm   = _get_llm(temperature=0.0)
        prompt = get_prompt(PromptKey.MULTI_SUPERVISOR)
        chain  = prompt | llm | StrOutputParser()

        raw_output = await chain.ainvoke({
            "task":         task,
            "user_context": str(user_context),
            "memory_context": str(memory_ctx),
        })

        parsed = _parse_json(raw_output, NODE_DECOMPOSE)

        # ── 解析 assignments ─────────────────────────────────────────────
        raw_assignments = parsed.get("assignments", [])

        # 安全上限：LLM 有时会生成过多子任务
        raw_assignments = raw_assignments[:_MAX_ASSIGNMENTS]

        assignments: list[AssignmentRecord] = []
        for i, item in enumerate(raw_assignments):
            agent = item.get("agent", "")
            if agent not in (AGENT_RESEARCH, AGENT_WRITER, AGENT_REVIEWER):
                logger.warning(
                    "decompose_node: 忽略未知 Worker 类型 '%s'（index=%d）", agent, i
                )
                continue

            # depends_on 中的索引可能指向实际解析后的 index，需要映射
            depends_on = [d for d in item.get("depends_on", []) if isinstance(d, int)]

            assignments.append(AssignmentRecord(
                index=i,
                agent=agent,
                instruction=item.get("instruction", f"执行{agent}任务"),
                depends_on=depends_on,
                priority=max(1, min(5, item.get("priority", 3))),
            ))

        # 退化兜底：LLM 未能生成任何有效 assignment 时，生成最简单的 Research 任务
        if not assignments:
            logger.warning("decompose_node: 未解析到有效 assignments，使用退化计划")
            assignments = [
                AssignmentRecord(
                    index=0,
                    agent=AGENT_RESEARCH,
                    instruction=f"请收集与以下任务相关的信息并生成报告：{task}",
                    depends_on=[],
                    priority=5,
                )
            ]

        execution_mode = parsed.get("execution_mode", "sequential")
        if execution_mode not in ("sequential", "parallel"):
            execution_mode = "sequential"

        analysis       = parsed.get("analysis", "")
        merge_strategy = parsed.get("merge_strategy", "整合所有 Worker 输出，生成统一回答")
        elapsed_ms     = (time.monotonic() - node_start) * 1000

        # ── 构建执行计划摘要消息 ──────────────────────────────────────────
        plan_lines = [
            f"[Supervisor] 任务分解完成 | 模式：{execution_mode} | "
            f"子任务数：{len(assignments)}\n",
            f"分析：{analysis}\n",
        ]
        for a in assignments:
            dep_str = f"（依赖任务 {[d+1 for d in a.depends_on]}）" if a.depends_on else ""
            plan_lines.append(
                f"  [{a.index + 1}] {a.agent} {dep_str}：{a.instruction[:80]}…"
            )

        logger.info(
            "decompose_node 完成 | assignments=%d | mode=%s | elapsed=%.0fms",
            len(assignments), execution_mode, elapsed_ms,
        )

        return {
            "analysis":       analysis,
            "execution_mode": execution_mode,
            "assignments":    assignments,
            "merge_strategy": merge_strategy,
            "worker_results": [],      # 初始化为空列表
            "revision_count": 0,
            "status":         TaskStatus.RUNNING,
            "messages": [
                make_system_message("\n".join(plan_lines)),
            ],
            "metadata": {
                **state.get("metadata", {}),
                "decompose_elapsed_ms": elapsed_ms,
                "decompose_mode":       execution_mode,
                "total_assignments":    len(assignments),
            },
        }

    except Exception as exc:
        logger.error("decompose_node 异常: %s", exc, exc_info=True)
        return {
            "status": TaskStatus.FAILED,
            "error":  f"任务分解失败：{exc}",
        }


async def dispatch_node(state: SupervisorState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  Dispatch 节点（Worker 调度执行）                         │
    │  职责：按依赖关系和执行模式，调度所有待执行的 Worker        │
    │                                                          │
    │  读取: assignments / execution_mode / worker_results /   │
    │        revision_count / task                             │
    │  写入: assignments（更新 status）/ worker_results（追加） │
    │        revision_count（若触发重写）/ messages             │
    │                                                          │
    │  核心算法：                                               │
    │    1. 找出所有状态为 PENDING 的 assignments              │
    │    2. 用拓扑排序划分执行层                                │
    │    3. 逐层执行（层内并行 / 层间串行）                      │
    │    4. 若 ReviewerAgent verdict=needs_revision             │
    │       且未超 max_revisions → 追加新 WriterAgent 任务      │
    └─────────────────────────────────────────────────────────┘
    """
    node_start      = time.monotonic()
    task            = state.get("task", "")
    assignments     = list(state.get("assignments", []))
    execution_mode  = state.get("execution_mode", "sequential")
    worker_results  = list(state.get("worker_results", []))
    revision_count  = state.get("revision_count", 0)

    logger.info(
        "dispatch_node 开始 | mode=%s | total_assignments=%d | revision_count=%d",
        execution_mode, len(assignments), revision_count,
    )

    # ── 筛选出待执行的 assignments ──────────────────────────────────────────
    pending = [a for a in assignments if a.status == AssignmentStatus.PENDING]

    if not pending:
        logger.info("dispatch_node: 无待执行任务，直接跳过")
        return {"worker_results": worker_results}

    # ── 拓扑排序 → 分层 ────────────────────────────────────────────────────
    # 注意：只对 pending 的 assignments 做分层，已完成的作为「已满足的依赖」
    layers = _build_execution_layers(pending)

    new_results: list[WorkerResult] = []
    dispatch_messages: list[AgentMessage] = []

    # ── 逐层执行 ─────────────────────────────────────────────────────────
    for layer_idx, layer in enumerate(layers):
        logger.info(
            "dispatch_node: 执行第 %d 层 | assignments=%s | mode=%s",
            layer_idx,
            [f"{a.agent}[{a.index}]" for a in layer],
            execution_mode,
        )

        dispatch_messages.append(
            make_assistant_message(
                f"[Dispatch 层 {layer_idx}] 开始执行："
                + ", ".join(f"{a.agent}[{a.index+1}]" for a in layer)
            )
        )

        if execution_mode == "parallel":
            # ── 并行：同层所有 assignments 同时执行 ──────────────────────
            layer_results = await asyncio.gather(
                *[
                    _call_worker(a, assignments, worker_results + new_results, task)
                    for a in layer
                ],
                return_exceptions=False,   # 异常由 _call_worker 内部捕获并返回 WorkerResult
            )
            new_results.extend(layer_results)

        else:
            # ── 串行：同层按 priority 降序逐个执行 ───────────────────────
            for assignment in layer:
                result = await _call_worker(
                    assignment, assignments, worker_results + new_results, task
                )
                new_results.append(result)

    # ── 合并新旧结果 ──────────────────────────────────────────────────────
    all_results = worker_results + new_results

    # ── 检查 ReviewerAgent 是否触发重写 ───────────────────────────────────
    revision_assignments: list[AssignmentRecord] = []
    next_index = max((a.index for a in assignments), default=-1) + 1

    if revision_count < _MAX_REVISIONS:
        for result in new_results:
            if (
                result.agent == AGENT_REVIEWER
                and result.verdict == VERDICT_NEEDS_REVISION
                and not result.error
            ):
                # 找到对应的原始 WriterAgent 任务，生成修订版本
                original_writer = next(
                    (
                        a for a in assignments
                        if a.agent == AGENT_WRITER and a.index in
                        [dep for rev_a in assignments
                         if rev_a.agent == AGENT_REVIEWER and rev_a.index == result.assignment_index
                         for dep in rev_a.depends_on]
                    ),
                    None,
                )

                revision_instruction = (
                    f"基于以下评审意见修改内容：\n{result.output}\n\n"
                    f"原始写作任务：{original_writer.instruction if original_writer else '（见前置任务）'}"
                )

                # 新 WriterAgent 任务依赖当前 ReviewerAgent 的 index
                revision_assignment = AssignmentRecord(
                    index=next_index,
                    agent=AGENT_WRITER,
                    instruction=revision_instruction,
                    depends_on=[result.assignment_index],
                    priority=5,   # 修订任务优先级最高
                    is_revision=True,
                )
                revision_assignments.append(revision_assignment)

                logger.info(
                    "dispatch_node: ReviewerAgent 触发重写 | "
                    "reviewer_index=%d | new_writer_index=%d",
                    result.assignment_index, next_index,
                )

                dispatch_messages.append(
                    make_assistant_message(
                        f"[Dispatch] ReviewerAgent 评分 {result.total_score}，"
                        f"触发第 {revision_count + 1} 次修订。"
                    )
                )
                next_index += 1

    elapsed_ms = (time.monotonic() - node_start) * 1000
    logger.info(
        "dispatch_node 完成 | new_results=%d | revisions=%d | elapsed=%.0fms",
        len(new_results), len(revision_assignments), elapsed_ms,
    )

    updated_state: dict[str, Any] = {
        "assignments":    assignments + revision_assignments,
        "worker_results": all_results,
        "messages":       dispatch_messages,
        "metadata": {
            **state.get("metadata", {}),
            "dispatch_elapsed_ms": elapsed_ms,
            "dispatch_new_results": len(new_results),
        },
    }

    # 若有修订任务，增加 revision_count
    if revision_assignments:
        updated_state["revision_count"] = revision_count + 1

    return updated_state


async def merge_node(state: SupervisorState) -> dict[str, Any]:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  Merge 节点（结果整合）                                   │
    │  职责：调用 LLM 将所有 Worker 输出整合为最终交付内容        │
    │                                                          │
    │  读取: task / worker_results / merge_strategy /          │
    │        analysis / assignments                            │
    │  写入: final_output / status / messages                  │
    └─────────────────────────────────────────────────────────┘
    """
    node_start      = time.monotonic()
    task            = state.get("task", "")
    worker_results  = state.get("worker_results", [])
    merge_strategy  = state.get("merge_strategy", "整合所有输出")
    analysis        = state.get("analysis", "")

    logger.info("merge_node 开始 | worker_results=%d", len(worker_results))

    # ── 构建各 Worker 输出的汇总文本 ──────────────────────────────────────
    result_sections: list[str] = []
    for result in worker_results:
        revision_tag = "（修订版）" if result.is_revision else ""
        header = f"=== {result.agent}{revision_tag} 输出 ==="
        output_body = result.output if not result.error else f"[执行失败：{result.error}]"
        result_sections.append(f"{header}\n{output_body}")

    all_outputs_text = "\n\n".join(result_sections) if result_sections else "（无 Worker 输出）"

    # ── 整合提示词 ────────────────────────────────────────────────────────
    merge_prompt_text = (
        f"你是一个内容整合专家。请根据以下 Merge 策略，"
        f"将各 Worker 的输出整合为面向用户的最终回答。\n\n"
        f"主任务：{task}\n"
        f"任务分析：{analysis}\n"
        f"整合策略：{merge_strategy}\n\n"
        f"整合规则：\n"
        f"1. 以用户的原始任务目标为核心，删除重复或冗余内容。\n"
        f"2. 若有 ReviewerAgent 的评审意见，优先使用经过修订的内容。\n"
        f"3. 保持输出的专业性和可读性，适当添加过渡语句使内容连贯。\n"
        f"4. 若某个 Worker 执行失败，在对应位置注明「[信息缺失]」并继续。\n"
        f"5. 直接输出最终内容，不要添加「以下是整合结果」等元描述。"
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore[import]

        llm = _get_llm(temperature=0.2)   # merge 允许稍高的创造性
        messages_for_llm = [
            SystemMessage(content=merge_prompt_text),
            HumanMessage(content=f"各 Worker 输出：\n\n{all_outputs_text}"),
        ]

        response = await llm.ainvoke(messages_for_llm)
        final_output = response.content if hasattr(response, "content") else str(response)

    except Exception as exc:
        logger.error("merge_node: LLM 整合失败: %s", exc, exc_info=True)
        # 降级：直接拼接所有 Worker 输出
        final_output = (
            f"（整合 LLM 调用失败，以下为各 Worker 原始输出）\n\n"
            f"{all_outputs_text}"
        )

    elapsed_ms = (time.monotonic() - node_start) * 1000
    logger.info(
        "merge_node 完成 | output_len=%d | elapsed=%.0fms",
        len(final_output), elapsed_ms,
    )

    return {
        "final_output": final_output,
        "status":       TaskStatus.COMPLETED,
        "messages": [
            make_assistant_message(
                f"[Supervisor] 任务完成，最终输出长度：{len(final_output)} 字符。"
            ),
        ],
        "metadata": {
            **state.get("metadata", {}),
            "merge_elapsed_ms":    elapsed_ms,
            "merge_output_length": len(final_output),
        },
    }


# ─────────────────────────────────────────────
# 4. 条件边路由函数
# ─────────────────────────────────────────────

def _route_after_dispatch(
    state: SupervisorState,
) -> Literal["dispatch", "merge"]:
    """
    Dispatch 节点之后的路由决策。

    路由逻辑（类比「质检流程」）：
    - 若存在状态为 PENDING 的 assignments（即刚追加的修订任务）
      且未超 max_revisions → 回到 dispatch 继续执行
    - 否则 → 进入 merge 整合所有结果
    """
    assignments    = state.get("assignments", [])
    revision_count = state.get("revision_count", 0)
    status         = state.get("status")

    # 任务已失败，直接跳到 merge（输出错误信息）
    if status == TaskStatus.FAILED:
        logger.info("_route_after_dispatch → merge (status=FAILED)")
        return NODE_MERGE

    # 存在未完成的 PENDING 任务（修订任务追加后状态为 PENDING）
    pending_count = sum(1 for a in assignments if a.status == AssignmentStatus.PENDING)
    if pending_count > 0 and revision_count <= _MAX_REVISIONS:
        logger.info(
            "_route_after_dispatch → dispatch (pending=%d, revision=%d/%d)",
            pending_count, revision_count, _MAX_REVISIONS,
        )
        return NODE_DISPATCH

    logger.info("_route_after_dispatch → merge (all done)")
    return NODE_MERGE


# ─────────────────────────────────────────────
# 5. 图构建与编译
# ─────────────────────────────────────────────

def _build_supervisor_graph(checkpointer=None) -> Any:
    """
    构建并编译 Supervisor LangGraph StateGraph。

    图结构（3节点）：
      decompose → dispatch → [条件边] → dispatch（重写循环）
                                      → merge → END
    """
    graph = StateGraph(SupervisorState)

    graph.add_node(NODE_DECOMPOSE, decompose_node)
    graph.add_node(NODE_DISPATCH,  dispatch_node)
    graph.add_node(NODE_MERGE,     merge_node)

    graph.set_entry_point(NODE_DECOMPOSE)

    # decompose → dispatch（固定）
    graph.add_edge(NODE_DECOMPOSE, NODE_DISPATCH)

    # dispatch → [条件边] → dispatch 或 merge
    graph.add_conditional_edges(
        source=NODE_DISPATCH,
        path=_route_after_dispatch,
        path_map={
            NODE_DISPATCH: NODE_DISPATCH,
            NODE_MERGE:    NODE_MERGE,
        },
    )

    # merge → END（固定）
    graph.add_edge(NODE_MERGE, END)

    return graph.compile(checkpointer=checkpointer)


_supervisor_memory_saver = MemorySaver()
_supervisor_graph        = _build_supervisor_graph(checkpointer=_supervisor_memory_saver)


# ─────────────────────────────────────────────
# 6. 公开入口函数
# ─────────────────────────────────────────────

async def run_supervisor(
    task:           str,
    session_id:     Optional[str]   = None,
    user_context:   str             = "",
    memory_context: Optional[Any]   = None,
    execution_mode: Optional[str]   = None,   # None 表示由 LLM 自动决定
    parallel:       bool            = False,  # True → 强制并行模式
    graph:          Optional[Any]   = None,
    trace_id:       Optional[str]   = None,
) -> SupervisorState:
    """
    多 Agent Supervisor 工作流的统一异步入口。

    调用流程：
    1. 构造初始 SupervisorState
    2. 调用 graph.ainvoke()
    3. 返回最终状态（含 final_output 和所有 worker_results）

    Args:
        task:           主任务描述（自然语言）
        session_id:     会话 ID，用于 LangGraph thread_id（断点续跑）
        user_context:   用户背景信息（偏好/角色/部门等，注入 Supervisor 提示词）
        memory_context: 跨会话记忆上下文（Step 23 MemoryService 提供）
        execution_mode: 强制指定执行模式（"sequential"/"parallel"），None 让 LLM 决定
        parallel:       True 时强制 parallel 模式（与 execution_mode 二选一）
        graph:          自定义图实例（None 使用模块级单例，测试时可注入 Mock）
        trace_id:       链路追踪 ID

    Returns:
        最终 SupervisorState，包含：
          - final_output:   整合后的完整输出文本
          - worker_results: 所有 Worker 的详细执行记录
          - assignments:    所有子任务的执行状态
          - status:         COMPLETED / FAILED
          - messages:       完整的执行日志消息序列

    使用示例：
        from agents.supervisor import run_supervisor

        result = await run_supervisor(
            task="请分析公司 Q3 销售数据，撰写管理层报告，并进行质量审查",
            user_context="用户是销售总监，偏好简洁的执行摘要风格",
            parallel=False,
        )
        print(result["final_output"])
        print(f"共执行 {len(result['worker_results'])} 个子任务")
    """
    sid      = session_id or str(uuid.uuid4())
    tid      = trace_id   or str(uuid.uuid4())

    # 执行模式 override 逻辑
    # parallel=True → 强制 parallel（兼容 PRD 的 parallel 参数约定）
    forced_mode = None
    if parallel:
        forced_mode = "parallel"
    elif execution_mode in ("sequential", "parallel"):
        forced_mode = execution_mode

    # 构造初始状态
    init_state: SupervisorState = {
        "task":           task,
        "user_context":   user_context or "（未提供）",
        "memory_context": memory_context,
        "analysis":       "",
        "execution_mode": forced_mode or "",   # "" 表示让 decompose_node 的 LLM 决定
        "assignments":    [],
        "merge_strategy": "",
        "worker_results": [],
        "revision_count": 0,
        "final_output":   "",
        "status":         TaskStatus.PENDING,
        "error":          None,
        "messages":       [make_user_message(task)],
        "metadata": {
            "trace_id":         tid,
            "session_id":       sid,
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "forced_mode":      forced_mode,
            "max_revisions":    _MAX_REVISIONS,
        },
    }

    # 若有强制模式，在 decompose_node 之前就写入 execution_mode
    # decompose_node 会检查此值是否已设置，若已设置则不覆盖
    if forced_mode:
        init_state["execution_mode"] = forced_mode

    run_config: RunnableConfig = {
        "configurable": {"thread_id": sid},
        "recursion_limit": 30,   # decompose(1) + dispatch(最多10次) + merge(1)，足够
    }

    active_graph = graph or _supervisor_graph

    logger.info(
        "run_supervisor 启动 | session_id=%s | task=%s | parallel=%s",
        sid, task[:80], parallel,
    )

    start_time = time.monotonic()

    try:
        final_state: SupervisorState = await active_graph.ainvoke(
            init_state, config=run_config
        )
        elapsed = time.monotonic() - start_time

        logger.info(
            "run_supervisor 完成 | session_id=%s | status=%s | "
            "workers=%d | revisions=%d | elapsed=%.2fs",
            sid,
            final_state.get("status"),
            len(final_state.get("worker_results", [])),
            final_state.get("revision_count", 0),
            elapsed,
        )

        return final_state

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "run_supervisor 异常 | session_id=%s | elapsed=%.2fs | error=%s",
            sid, elapsed, exc,
            exc_info=True,
        )
        # 返回 FAILED 状态，保证调用方总能拿到结构化结果
        return SupervisorState(
            task=task,
            user_context=user_context,
            memory_context=memory_context,
            analysis="",
            execution_mode=forced_mode or "sequential",
            assignments=init_state.get("assignments", []),
            merge_strategy="",
            worker_results=init_state.get("worker_results", []),
            revision_count=0,
            final_output="",
            status=TaskStatus.FAILED,
            error=f"Supervisor 工作流异常：{type(exc).__name__}: {exc}",
            messages=init_state.get("messages", []),
            metadata={
                **init_state.get("metadata", {}),
                "supervisor_error":    str(exc),
                "supervisor_elapsed_s": elapsed,
            },
        )
