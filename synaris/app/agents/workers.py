"""
@File       : workers.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: 多 Agent 协作体系中的三个 Worker Agent 实现。
@Features:
  - ResearchAgent: 信息收集型 Worker
      · 工具：rag_retrieval（优先）+ web_search（补充）
      · 使用 LangGraph ReAct 子图，自主决策工具调用轮次
      · 产出：结构化 Markdown 研究报告
      · 最大工具调用轮次：5 次（防止过度检索）
  - WriterAgent: 内容创作型 Worker
      · 无工具调用（纯 LLM 生成），专注于写作质量
      · 输入：instruction + research_output（依赖注入）
      · 产出：执行摘要 → 正文 → 结论与建议
      · 支持多种输出格式（Markdown 报告 / 邮件 / 简报等）
  - ReviewerAgent: 质量评审型 Worker
      · 无工具调用（纯 LLM 评审），专注于批判性分析
      · 输入：content_to_review + research_reference（核实准确性）
      · 产出：JSON 格式评审报告（分数 + verdict + 修改建议）
      · verdict: approved（≥80）/ needs_revision（60-79）/ rejected（<60）
  - 三个公开入口函数（与 supervisor.py 的 _call_worker 接口完全对齐）:
      run_research_agent(instruction, main_task) → str
      run_writer_agent(instruction, research_output, main_task) → str
      run_reviewer_agent(instruction, content_to_review,
                         research_reference, main_task) → str
  - 统一的错误处理：Worker 失败返回降级字符串，不向上抛出异常

  ┌─────────────────────────────────────────────────────────────────────┐
  │                 三个 Worker 的执行模式对比                            │
  │                                                                      │
  │  ResearchAgent          WriterAgent           ReviewerAgent          │
  │  ─────────────          ───────────           ─────────────          │
  │  LangGraph ReAct        纯 LLM 调用           纯 LLM 调用            │
  │  子图（有工具）          （无工具）              （无工具）             │
  │       │                     │                      │                 │
  │  rag_retrieval          gpt-4o 写作            gpt-4o 评审           │
  │  web_search             专业语言风格           批判性视角             │
  │       │                     │                      │                 │
  │  Markdown 报告          结构化内容             JSON 评分报告          │
  │  （含来源标注）          （含信息缺口标注）      （含修改建议）         │
  └─────────────────────────────────────────────────────────────────────┘

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from core.prompts import PromptKey, get_prompt

logger = logging.getLogger(__name__)

# ── 常量 ───────────────────────────────────────────────────────────────────────
_RESEARCH_MAX_ITERATIONS = 5    # ResearchAgent 最大工具调用轮次
_WRITER_TEMPERATURE      = 0.3  # WriterAgent 允许适度创造性
_REVIEWER_TEMPERATURE    = 0.0  # ReviewerAgent 严格确定性输出
_OUTPUT_FORMAT_DEFAULT   = "Markdown 格式报告（含执行摘要、正文章节、结论）"

# ResearchAgent 专用工具列表（按优先级排列）
_RESEARCH_TOOLS = ["rag_retrieval", "web_search"]

# 节点名称常量（ResearchAgent 内部子图）
_NODE_RESEARCH_PLAN    = "research_plan"
_NODE_RESEARCH_EXECUTE = "research_execute"
_NODE_RESEARCH_COMPILE = "research_compile"


# ─────────────────────────────────────────────
# 1. 辅助函数（模块内共用）
# ─────────────────────────────────────────────

def _get_llm(strategy: str = "quality", temperature: float = 0.0):
    """
    获取 LLM 实例。

    策略映射（对齐 LLM Router 的分层路由）：
      quality  → gpt-4o（推理/写作任务，成本较高）
      balanced → gpt-4o-mini（一般任务，成本均衡）

    Step 6（llm_client.py）完成前使用直接构造作为降级路径。
    """
    try:
        from infrastructure.llm_client import get_llm_client  # type: ignore[import]
        return get_llm_client(strategy=strategy)
    except ImportError:
        from langchain_openai import ChatOpenAI
        import os
        model_map = {
            "quality":  os.getenv("QUALITY_MODEL",  "gpt-4o"),
            "balanced": os.getenv("BALANCED_MODEL", "gpt-4o-mini"),
        }
        return ChatOpenAI(
            model=model_map.get(strategy, "gpt-4o-mini"),
            temperature=temperature,
        )


def _parse_json(raw: str, caller: str) -> dict:
    """安全解析 LLM 的 JSON 输出（剥离 ```json 代码块）。"""
    text = raw.strip()
    if text.startswith("```"):
        inner = []
        for line in text.split("\n")[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("%s: JSON 解析失败 | raw=%r | err=%s", caller, raw[:200], exc)
        return {}


def _get_research_tools() -> dict[str, Any]:
    """
    获取 ResearchAgent 专用工具。
    延迟导入避免循环依赖。
    """
    tools: dict[str, Any] = {}
    try:
        from agents.tools.rag_retrieval import rag_retrieval
        tools["rag_retrieval"] = rag_retrieval
    except ImportError:
        logger.warning("workers: rag_retrieval 工具不可用")
    try:
        from agents.tools.web_search import web_search
        tools["web_search"] = web_search
    except ImportError:
        logger.warning("workers: web_search 工具不可用")
    return tools


def _tools_description(tools: dict[str, Any]) -> str:
    """生成工具描述文本，注入 ResearchAgent 系统提示词。"""
    if not tools:
        return "（当前无可用工具，请基于已有知识作答）"
    lines = []
    for name, fn in tools.items():
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  - {name}: {doc}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 2. ResearchAgent 内部子图（LangGraph ReAct 风格）
# ─────────────────────────────────────────────

class _ResearchState(TypedDict, total=False):
    """
    ResearchAgent 内部状态（不对外暴露）。

    ReAct 循环的「草稿本」：
      instruction    → 研究任务描述
      main_task      → 主任务背景（上下文）
      tool_calls     → 已执行的工具调用记录列表
      findings       → 累积的研究发现（文本拼接）
      iteration      → 当前迭代次数
      done           → 是否完成研究
      final_report   → 最终研究报告（compile 节点产出）
    """
    instruction:  str
    main_task:    str
    tool_calls:   list[dict]   # [{tool, input, output, source}]
    findings:     str          # 累积的研究发现文本
    iteration:    int
    done:         bool
    final_report: str


async def _research_plan_node(state: _ResearchState) -> dict[str, Any]:
    """
    ResearchAgent 规划节点：决定下一步调用哪个工具，或宣布研究完成。

    ReAct 模式类比「侦探推理」：
      - 每轮看一眼已有线索（findings）
      - 决定下一步去哪里找证据（tool_name + tool_input）
      - 或宣布「案子破了」（done=True）

    输出 JSON: { tool_name, tool_input, reasoning, done }
    """
    instruction = state.get("instruction", "")
    main_task   = state.get("main_task", "")
    findings    = state.get("findings", "（暂无发现）")
    iteration   = state.get("iteration", 0)
    tool_calls  = state.get("tool_calls", [])
    tools       = _get_research_tools()

    # 已用过的工具名称（避免重复调用相同工具+参数）
    used_queries = [
        f"{c['tool']}:{c.get('input', {}).get('query', '')}"
        for c in tool_calls
    ]

    plan_prompt = (
        f"你是一个研究规划助手。根据研究任务和已有发现，决定下一步行动。\n\n"
        f"研究任务：{instruction}\n"
        f"主任务背景：{main_task}\n\n"
        f"可用工具：\n{_tools_description(tools)}\n\n"
        f"已执行 {iteration} 轮研究（上限 {_RESEARCH_MAX_ITERATIONS} 轮）\n"
        f"已使用查询：{used_queries}\n\n"
        f"当前已有发现摘要：\n{findings[:1500]}\n\n"
        f"决策规则：\n"
        f"1. 优先使用 rag_retrieval 检索内部知识库，再用 web_search 补充。\n"
        f"2. 不要重复相同的工具+查询组合。\n"
        f"3. 若已有足够信息回答任务，或已达轮次上限，设 done=true。\n\n"
        f"严格按照以下 JSON 格式输出，不要输出其他内容：\n"
        f'{{\n'
        f'  "tool_name": "<rag_retrieval|web_search|none>",\n'
        f'  "tool_input": {{}},\n'
        f'  "reasoning": "<本轮决策理由（1句话）>",\n'
        f'  "done": <true|false>\n'
        f'}}'
    )

    try:
        from langchain_core.messages import HumanMessage
        llm = _get_llm(strategy="balanced", temperature=0.0)
        response = await llm.ainvoke([HumanMessage(content=plan_prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json(content, "_research_plan_node")

        tool_name  = parsed.get("tool_name", "none")
        tool_input = parsed.get("tool_input", {})
        done       = parsed.get("done", False) or (iteration >= _RESEARCH_MAX_ITERATIONS)

        return {
            "iteration": iteration + 1,
            "done":      done,
            "_pending_tool": {          # 临时存储，供 execute 节点读取
                "name":      tool_name,
                "input":     tool_input,
                "reasoning": parsed.get("reasoning", ""),
            },
        }

    except Exception as exc:
        logger.warning("_research_plan_node 异常: %s", exc)
        return {"done": True, "_pending_tool": {"name": "none", "input": {}}}


async def _research_execute_node(state: _ResearchState) -> dict[str, Any]:
    """
    ResearchAgent 执行节点：调用 plan 节点选定的工具，累积 findings。
    """
    pending     = state.get("_pending_tool", {})
    tool_name   = pending.get("name", "none")
    tool_input  = pending.get("input", {})
    reasoning   = pending.get("reasoning", "")
    findings    = state.get("findings", "")
    tool_calls  = list(state.get("tool_calls", []))

    if tool_name == "none" or state.get("done"):
        return {}   # 无需执行，直接进入下一步路由判断

    tools = _get_research_tools()

    if tool_name not in tools:
        logger.warning("_research_execute_node: 未知工具 %s，跳过", tool_name)
        return {}

    try:
        tool_fn = tools[tool_name]
        output  = await tool_fn.ainvoke(tool_input)
        output_str = str(output)

        # 追加到 findings
        source_tag = "知识库" if tool_name == "rag_retrieval" else "网络搜索"
        query_hint = tool_input.get("query", str(tool_input))[:60]
        new_finding = (
            f"\n\n--- [{source_tag}] 查询：{query_hint} ---\n"
            f"{output_str[:2000]}"  # 单次结果限 2000 字
        )

        tool_calls.append({
            "tool":      tool_name,
            "input":     tool_input,
            "output":    output_str[:500],
            "source":    source_tag,
            "reasoning": reasoning,
        })

        return {
            "findings":   findings + new_finding,
            "tool_calls": tool_calls,
        }

    except Exception as exc:
        logger.warning("_research_execute_node: 工具调用失败 %s: %s", tool_name, exc)
        tool_calls.append({
            "tool":   tool_name,
            "input":  tool_input,
            "output": f"调用失败：{exc}",
            "source": "error",
        })
        return {"tool_calls": tool_calls}


async def _research_compile_node(state: _ResearchState) -> dict[str, Any]:
    """
    ResearchAgent 整合节点：将所有 findings 整合为结构化研究报告。

    类比「情报分析员」最终整理档案：
    - 去除重复
    - 标注来源（知识库 vs 网络）
    - 识别信息缺口
    - 输出 Markdown 格式报告
    """
    instruction = state.get("instruction", "")
    main_task   = state.get("main_task", "")
    findings    = state.get("findings", "（无研究发现）")
    tool_calls  = state.get("tool_calls", [])

    # 统计工具使用情况
    rag_count  = sum(1 for c in tool_calls if c.get("source") == "知识库")
    web_count  = sum(1 for c in tool_calls if c.get("source") == "网络搜索")

    compile_system = (
        f"你是一个专业的信息研究员（ResearchAgent）。\n"
        f"将以下研究发现整合为结构化研究报告。\n\n"
        f"整合规则：\n"
        f"1. 报告格式：## 背景摘要 → ## 核心发现 → ## 数据与引用 → ## 信息缺口\n"
        f"2. 每条核心发现需标注来源：[知识库] 或 [网络来源]。\n"
        f"3. 相互矛盾的信息标注「⚠️ 存在分歧」并列出各方观点。\n"
        f"4. 保持客观，不加主观判断。\n"
        f"5. 使用 Markdown 格式，层次清晰。\n\n"
        f"本次研究统计：知识库检索 {rag_count} 次 / 网络搜索 {web_count} 次"
    )

    compile_human = (
        f"研究任务：{instruction}\n"
        f"主任务背景：{main_task}\n\n"
        f"原始研究发现：\n{findings}"
    )

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = _get_llm(strategy="quality", temperature=0.0)
        response = await llm.ainvoke([
            SystemMessage(content=compile_system),
            HumanMessage(content=compile_human),
        ])
        report = response.content if hasattr(response, "content") else str(response)

    except Exception as exc:
        logger.error("_research_compile_node: LLM 整合失败: %s", exc)
        # 降级：直接返回原始 findings
        report = (
            f"## 研究报告（降级输出）\n\n"
            f"**研究任务**：{instruction}\n\n"
            f"**原始发现**：\n{findings[:3000]}"
        )

    return {"final_report": report}


def _route_research_loop(
    state: _ResearchState,
) -> Literal["research_execute", "research_compile"]:
    """
    ReAct 循环路由：
    - done=True 或迭代达上限 → compile（退出循环）
    - done=False              → execute（继续研究）
    """
    if state.get("done") or state.get("iteration", 0) >= _RESEARCH_MAX_ITERATIONS:
        return _NODE_RESEARCH_COMPILE
    return _NODE_RESEARCH_EXECUTE


def _build_research_graph() -> Any:
    """
    构建 ResearchAgent 内部 LangGraph 子图（不使用 checkpointer，无需跨进程持久化）。

    子图结构（ReAct 循环）：
      plan → [条件边] → execute → plan（循环）
                      → compile → END
    """
    graph = StateGraph(_ResearchState)

    graph.add_node(_NODE_RESEARCH_PLAN,    _research_plan_node)
    graph.add_node(_NODE_RESEARCH_EXECUTE, _research_execute_node)
    graph.add_node(_NODE_RESEARCH_COMPILE, _research_compile_node)

    graph.set_entry_point(_NODE_RESEARCH_PLAN)

    # plan → [条件边]
    graph.add_conditional_edges(
        source=_NODE_RESEARCH_PLAN,
        path=_route_research_loop,
        path_map={
            _NODE_RESEARCH_EXECUTE: _NODE_RESEARCH_EXECUTE,
            _NODE_RESEARCH_COMPILE: _NODE_RESEARCH_COMPILE,
        },
    )

    # execute → plan（ReAct 循环回）
    graph.add_edge(_NODE_RESEARCH_EXECUTE, _NODE_RESEARCH_PLAN)

    # compile → END
    graph.add_edge(_NODE_RESEARCH_COMPILE, END)

    return graph.compile()


# 模块级单例（ResearchAgent 子图，无 checkpointer）
_research_graph = _build_research_graph()


# ─────────────────────────────────────────────
# 3. 三个公开 Worker 入口函数
# ─────────────────────────────────────────────

async def run_research_agent(
    instruction: str,
    main_task:   str,
) -> str:
    """
    ResearchAgent 入口：信息收集 + 研究报告生成。

    执行流程（ReAct 子图）：
        plan（决定工具）→ execute（调工具）→ plan（循环，最多5轮）
                                            → compile（整合报告）

    工具优先级：rag_retrieval（内部知识库）> web_search（外部补充）

    Args:
        instruction: 研究任务具体指令（来自 Supervisor 的 assignment.instruction）
        main_task:   主任务描述（提供更宏观的语境）

    Returns:
        Markdown 格式结构化研究报告字符串

    示例输出：
        ## 背景摘要
        本次研究聚焦于…

        ## 核心发现
        1. [知识库] …
        2. [网络来源] …

        ## 信息缺口
        - 暂未找到…的数据
    """
    start_time = time.monotonic()
    logger.info("run_research_agent 开始 | instruction=%s", instruction[:80])

    init_state: _ResearchState = {
        "instruction": instruction,
        "main_task":   main_task,
        "tool_calls":  [],
        "findings":    "",
        "iteration":   0,
        "done":        False,
        "final_report": "",
    }

    try:
        final: _ResearchState = await _research_graph.ainvoke(
            init_state,
            config=RunnableConfig(recursion_limit=_RESEARCH_MAX_ITERATIONS * 2 + 5),
        )
        report    = final.get("final_report") or "（研究未产生有效输出）"
        elapsed   = time.monotonic() - start_time
        tool_count = len(final.get("tool_calls", []))

        logger.info(
            "run_research_agent 完成 | tools_used=%d | report_len=%d | elapsed=%.2fs",
            tool_count, len(report), elapsed,
        )
        return report

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error("run_research_agent 异常: %s | elapsed=%.2fs", exc, elapsed, exc_info=True)
        return (
            f"## 研究报告（执行异常）\n\n"
            f"**任务**：{instruction}\n\n"
            f"**错误**：{type(exc).__name__}: {exc}\n\n"
            f"**降级处理**：无法完成信息收集，请人工补充相关资料。"
        )


async def run_writer_agent(
    instruction:     str,
    research_output: str,
    main_task:       str,
    output_format:   str = _OUTPUT_FORMAT_DEFAULT,
) -> str:
    """
    WriterAgent 入口：基于研究结果创作结构化内容（纯 LLM，无工具调用）。

    设计决策：WriterAgent 不调用任何工具，只依赖注入的研究结果。
    这样做的好处：
    1. 职责单一——写作 vs 检索 是两种完全不同的认知模式
    2. 可重复性——相同输入产出稳定的内容（temperature=0.3）
    3. 质量可控——ReviewerAgent 评审后若需修改，只需重跑 Writer 而不必重做 Research

    Args:
        instruction:     写作任务具体指令（来自 Supervisor 的 assignment.instruction）
        research_output: ResearchAgent 的研究报告（依赖注入）
        main_task:       主任务描述（宏观语境）
        output_format:   输出格式要求（默认 Markdown 报告）

    Returns:
        结构化内容字符串（Markdown 格式为主）

    内容结构（默认）：
        ## 执行摘要（3句话）
        ## 正文（分章节）
        ## 结论与建议
    """
    start_time = time.monotonic()
    logger.info("run_writer_agent 开始 | instruction=%s", instruction[:80])

    # 研究资料长度保护（过长的 research_output 会超 Token）
    research_preview = research_output
    if len(research_output) > 8000:
        research_preview = research_output[:8000] + "\n\n… [研究内容已截断，总长度超过 8000 字符]"

    try:
        llm    = _get_llm(strategy="quality", temperature=_WRITER_TEMPERATURE)
        prompt = get_prompt(PromptKey.MULTI_WRITER)
        chain  = prompt | llm | StrOutputParser()

        content = await chain.ainvoke({
            "instruction":     instruction,
            "research_output": research_preview,
            "main_task":       main_task,
            "output_format":   output_format,
        })

        elapsed = time.monotonic() - start_time
        logger.info(
            "run_writer_agent 完成 | content_len=%d | elapsed=%.2fs",
            len(content), elapsed,
        )
        return content

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error("run_writer_agent 异常: %s | elapsed=%.2fs", exc, elapsed, exc_info=True)
        return (
            f"## 内容草稿（执行异常）\n\n"
            f"**写作任务**：{instruction}\n\n"
            f"**错误**：{type(exc).__name__}: {exc}\n\n"
            f"**可用研究资料摘要**：\n{research_output[:500]}…\n\n"
            f"[待补充：WriterAgent 执行失败，请手动完成此章节]"
        )


async def run_reviewer_agent(
    instruction:        str,
    content_to_review:  str,
    research_reference: str,
    main_task:          str,
) -> str:
    """
    ReviewerAgent 入口：质量评审（纯 LLM，无工具调用）。

    评审框架（四维度，各 25 分，满分 100）：
      · 专业性：术语准确，逻辑严谨
      · 准确性：与原始研究资料一致，无事实错误
      · 完整性：覆盖主要议题，无明显遗漏
      · 可读性：结构清晰，语言流畅

    裁定标准：
      approved        ≥ 80 分  → 质量达标，可直接交付
      needs_revision  60-79 分 → 需修改，触发 WriterAgent 重写
      rejected        < 60 分  → 质量不合格，记录问题继续整体流程

    Args:
        instruction:        评审任务具体指令
        content_to_review:  待审内容（WriterAgent 的产出，或依赖上下文）
        research_reference: 原始研究资料（用于核实准确性）
        main_task:          主任务描述（宏观语境）

    Returns:
        JSON 格式评审报告字符串（supervisor.py 的 _call_worker 会解析此 JSON）

    JSON 结构：
        {
          "scores": { professionalism, accuracy, completeness, readability },
          "total_score": 85,
          "verdict": "approved",
          "strengths": ["..."],
          "issues": [{"severity": "high", "description": "...", "suggestion": "..."}],
          "revised_sections": "..."
        }
    """
    start_time = time.monotonic()
    logger.info("run_reviewer_agent 开始 | instruction=%s", instruction[:80])

    # 内容长度保护
    content_preview   = content_to_review
    reference_preview = research_reference

    if len(content_to_review) > 6000:
        content_preview = content_to_review[:6000] + "\n… [内容已截断]"
    if len(research_reference) > 4000:
        reference_preview = research_reference[:4000] + "\n… [参考资料已截断]"

    try:
        llm    = _get_llm(strategy="quality", temperature=_REVIEWER_TEMPERATURE)
        prompt = get_prompt(PromptKey.MULTI_REVIEWER)
        chain  = prompt | llm | StrOutputParser()

        raw_output = await chain.ainvoke({
            "instruction":        instruction,
            "content_to_review":  content_preview,
            "research_reference": reference_preview,
            "main_task":          main_task,
        })

        # 验证输出是否为合法 JSON（不强制，保持原始字符串返回给 _call_worker 解析）
        parsed = _parse_json(raw_output, "run_reviewer_agent")
        if not parsed:
            # JSON 解析失败时，构造一个最低分的 fallback 评审报告
            logger.warning("run_reviewer_agent: LLM 未输出合法 JSON，使用 fallback")
            raw_output = json.dumps({
                "scores": {
                    "professionalism": 15,
                    "accuracy":        15,
                    "completeness":    15,
                    "readability":     15,
                },
                "total_score": 60,
                "verdict":     "needs_revision",
                "strengths":   ["内容有一定信息量"],
                "issues": [{
                    "severity":    "high",
                    "description": "输出格式不符合要求",
                    "suggestion":  "请确保输出为规范的结构化格式",
                }],
                "revised_sections": "",
            }, ensure_ascii=False, indent=2)

        elapsed = time.monotonic() - start_time
        verdict = parsed.get("verdict", "unknown")
        score   = parsed.get("total_score", 0)

        logger.info(
            "run_reviewer_agent 完成 | verdict=%s | score=%d | elapsed=%.2fs",
            verdict, score, elapsed,
        )
        return raw_output

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error("run_reviewer_agent 异常: %s | elapsed=%.2fs", exc, elapsed, exc_info=True)

        # 返回一个合法的「失败评审」JSON，不向上抛出
        fallback = {
            "scores": {
                "professionalism": 0,
                "accuracy":        0,
                "completeness":    0,
                "readability":     0,
            },
            "total_score": 0,
            "verdict":     "needs_revision",
            "strengths":   [],
            "issues": [{
                "severity":    "high",
                "description": f"ReviewerAgent 执行异常：{type(exc).__name__}: {exc}",
                "suggestion":  "请检查 LLM 服务状态后重试。",
            }],
            "revised_sections": "",
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 4. 便捷同步包装（用于调试和单元测试）
# ─────────────────────────────────────────────

def run_research_agent_sync(instruction: str, main_task: str) -> str:
    """
    run_research_agent 的同步版本（仅供调试和单元测试使用）。

    生产环境应始终使用异步版本 run_research_agent()。
    """
    import asyncio
    return asyncio.run(run_research_agent(instruction, main_task))


def run_writer_agent_sync(
    instruction:     str,
    research_output: str,
    main_task:       str,
) -> str:
    """run_writer_agent 的同步版本（仅供调试和单元测试使用）。"""
    import asyncio
    return asyncio.run(run_writer_agent(instruction, research_output, main_task))


def run_reviewer_agent_sync(
    instruction:        str,
    content_to_review:  str,
    research_reference: str,
    main_task:          str,
) -> str:
    """run_reviewer_agent 的同步版本（仅供调试和单元测试使用）。"""
    import asyncio
    return asyncio.run(
        run_reviewer_agent(instruction, content_to_review, research_reference, main_task)
    )
