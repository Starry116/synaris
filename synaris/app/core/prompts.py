"""
@File       : prompts.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: 全项目提示词集中管理模块。
@Features:
  - PromptKey 枚举: 所有提示词的唯一标识符常量，避免散落的魔法字符串
  - Chat 提示词: 多轮对话系统提示 + 历史感知对话模板
  - RAG 提示词: 知识库问答生成模板（含引用格式约束）
  - 单 Agent 提示词集:
      · Planner       — 任务分解，输出结构化 JSON 步骤计划
      · ToolSelector  — 工具选择，输出工具名 + 入参 JSON
      · Observer      — 结果观察，决策下一步动作（continue/complete/retry/human/fail）
  - 多 Agent 提示词集:
      · Supervisor    — 任务分解并委派给 Worker，输出并行/串行调度 JSON
      · ResearchAgent — RAG + 搜索，产出结构化研究报告
      · WriterAgent   — 基于研究结果生成结构化内容
      · ReviewerAgent — 质量评审，输出 0-100 评分 + 修改建议 JSON
  - PromptRegistry: 单例注册表，统一管理全部模板，支持 get() / register() / list_keys()
  - get_prompt() / render_prompt(): 全局便捷访问函数

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ─────────────────────────────────────────────
# 1. PromptKey — 提示词唯一标识枚举
# ─────────────────────────────────────────────


class PromptKey(str, Enum):
    """
    全项目提示词的枚举常量。

    用途：杜绝散落在各模块的魔法字符串，
    类比数据库表名常量——统一在此声明，其他地方只引用枚举值。

    调用方式：
        from core.prompts import get_prompt, PromptKey
        chain = get_prompt(PromptKey.RAG_GENERATE) | llm | StrOutputParser()
    """

    # ── Chat ──────────────────────────────────
    CHAT_SYSTEM = "chat_system"
    CHAT_HISTORY_AWARE = "chat_history_aware"

    # ── RAG ───────────────────────────────────
    RAG_GENERATE = "rag_generate"
    RAG_CONDENSE_QUESTION = "rag_condense_question"

    # ── 单 Agent ──────────────────────────────
    AGENT_PLANNER = "agent_planner"
    AGENT_TOOL_SELECTOR = "agent_tool_selector"
    AGENT_OBSERVER = "agent_observer"

    # ── 多 Agent ──────────────────────────────
    MULTI_SUPERVISOR = "multi_supervisor"
    MULTI_RESEARCH = "multi_research"
    MULTI_WRITER = "multi_writer"
    MULTI_REVIEWER = "multi_reviewer"


# ─────────────────────────────────────────────
# 2. Chat 提示词
# ─────────────────────────────────────────────

# 2-A: 多轮对话系统提示（注入到每次对话的 system 角色）
_CHAT_SYSTEM = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 Synaris，一个专业、严谨、友好的企业级 AI 助手。\n"
            "你的职责是回答用户问题、辅助决策分析、协助内容创作。\n\n"
            "行为准则：\n"
            "1. 回答需准确、简洁，优先引用企业知识库中的内容。\n"
            "2. 不确定时主动说明，不捏造信息。\n"
            "3. 涉及敏感决策（如合同、财务、法律）时，提醒用户寻求专业人士确认。\n"
            "4. 用中文回复，技术术语可保留英文原词。\n\n"
            "当前时间：{current_time}\n"
            "用户 ID：{user_id}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# 2-B: 历史感知问题改写（多轮对话：将 follow-up 转为独立问题）
_CHAT_HISTORY_AWARE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个对话理解专家。\n"
            "根据下方聊天历史和最新的用户问题，"
            "将用户问题改写为一个完整的、不依赖上下文即可独立理解的问题。\n"
            "如果问题已经清晰完整，直接原样返回，不要添加任何解释。\n"
            "只输出改写后的问题，不要输出其他内容。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "原始问题：{question}"),
    ]
)


# ─────────────────────────────────────────────
# 3. RAG 提示词
# ─────────────────────────────────────────────

# 3-A: RAG 答案生成（基于检索到的上下文片段生成有引用的回答）
_RAG_GENERATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业知识库问答助手。请严格基于以下检索到的上下文片段回答问题。\n\n"
            "【上下文片段】\n"
            "{context}\n\n"
            "回答规则：\n"
            "1. 只使用上下文中明确存在的信息作答，不要推断或补充未提及的内容。\n"
            "2. 如果上下文不包含足够信息，回答「根据现有知识库，暂无相关资料，"
            "建议联系相关负责人确认」。\n"
            "3. 每个关键论点后用 [来源: {source_hint}] 标注引用，格式固定。\n"
            "4. 回答结构清晰，使用分点或分段，长度控制在 300 字以内（除非问题需要详述）。\n"
            "5. 用中文回复。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# 3-B: 问题压缩改写（多轮 RAG：将 follow-up 转为适合向量检索的独立 query）
_RAG_CONDENSE_QUESTION = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个信息检索专家。\n"
            "根据聊天历史，将用户最新的问题改写为一个适合向量数据库语义检索的独立查询语句。\n"
            "改写要求：\n"
            "- 保留核心语义，去除指代词（如「它」「这个」「上面提到的」）\n"
            "- 适当扩展关键词以提高召回率\n"
            "- 只输出改写后的查询语句，不要输出其他内容",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "原始问题：{question}"),
    ]
)


# ─────────────────────────────────────────────
# 4. 单 Agent 提示词（workflow.py 中各节点使用）
# ─────────────────────────────────────────────

# 4-A: Planner 节点 — 接收用户任务，输出结构化执行计划
#
# 输入变量: available_tools, task, memory_context
# 输出格式: JSON { goal, steps[], estimated_tool_calls }
_AGENT_PLANNER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个任务规划专家（Planner）。你的职责是将用户任务分解为清晰、可执行的步骤序列。\n\n"
            "可用工具列表：\n{available_tools}\n\n"
            "规划规则：\n"
            "1. 每个步骤必须是原子操作（单一职责），步骤数量控制在 3-7 步。\n"
            "2. 步骤之间若有依赖关系，在描述中注明（如「基于步骤 2 的结果」）。\n"
            "3. 优先使用工具获取信息，再进行分析综合。\n"
            "4. 最后一步必须是「整合所有结果，生成最终回答」。\n\n"
            "严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
            "{{\n"
            '  "goal": "<用一句话描述任务目标>",\n'
            '  "steps": [\n'
            '    "步骤 1：<具体操作描述>",\n'
            '    "步骤 2：<具体操作描述>"\n'
            "  ],\n"
            '  "estimated_tool_calls": <预计工具调用次数，整数>\n'
            "}}",
        ),
        ("human", "用户任务：{task}\n\n" "记忆上下文（若有）：\n{memory_context}"),
    ]
)

# 4-B: ToolSelector 节点 — 根据当前步骤选择最合适的工具及入参
#
# 输入变量: tools_detail, current_step_index, current_step,
#           task, tool_results_summary
# 输出格式: JSON { tool_name, tool_input, reasoning }
_AGENT_TOOL_SELECTOR = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个工具调用专家（ToolSelector）。"
            "根据当前执行步骤和可用工具，选择最合适的工具并确定调用参数。\n\n"
            "可用工具详情：\n{tools_detail}\n\n"
            "选择规则：\n"
            "1. 只选择一个最合适的工具，不要试图在一次调用中组合多个工具。\n"
            "2. 入参必须具体且完整，不能含有占位符或 TODO。\n"
            '3. 如果当前步骤无需工具（如纯推理/综合/总结），输出 tool_name 为 "none"。\n\n'
            "严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
            "{{\n"
            '  "tool_name": "<工具名称 或 none>",\n'
            '  "tool_input": {{<工具入参键值对>}},\n'
            '  "reasoning": "<选择该工具的简要理由（1句话）>"\n'
            "}}",
        ),
        (
            "human",
            "当前执行步骤（第 {current_step_index} 步）：{current_step}\n\n"
            "整体任务目标：{task}\n\n"
            "已有工具结果摘要：\n{tool_results_summary}",
        ),
    ]
)

# 4-C: Observer 节点 — 观察工具执行结果，决定下一步行动
#
# 输入变量: task, plan, current_step_index, current_step,
#           completed_steps, total_steps, iteration_count,
#           max_iterations, latest_tool_result
# 输出格式: JSON { decision, reasoning, final_answer, question, error_message }
_AGENT_OBSERVER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个任务执行观察者（Observer）。"
            "根据已完成步骤的结果，判断任务下一步行动。\n\n"
            "决策选项：\n"
            "  continue — 当前步骤已完成，继续执行计划中的下一步\n"
            "  complete — 所有步骤已完成，生成最终回答（必须填写 final_answer）\n"
            "  retry    — 当前步骤失败或结果不满意，需重试（每步最多重试 2 次）\n"
            "  human    — 遇到歧义或高风险操作，需人工确认（必须填写 question）\n"
            "  fail     — 任务无法完成（工具持续失败/信息不足），终止并报告原因\n\n"
            "决策优先级：若迭代次数已达上限，必须选择 complete 或 fail，不得选 continue/retry。\n\n"
            "严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
            "{{\n"
            '  "decision": "<continue|complete|retry|human|fail>",\n'
            '  "reasoning": "<决策理由（1-2句话）>",\n'
            '  "final_answer": "<面向用户的完整回答，仅 decision=complete 时填写，其余为空字符串>",\n'
            '  "question": "<需人工确认的具体问题，仅 decision=human 时填写，其余为空字符串>",\n'
            '  "error_message": "<失败原因说明，仅 decision=fail 时填写，其余为空字符串>"\n'
            "}}",
        ),
        (
            "human",
            "任务目标：{task}\n\n"
            "执行计划：\n{plan}\n\n"
            "当前步骤（第 {current_step_index} 步）：{current_step}\n"
            "已完成步骤数：{completed_steps} / {total_steps}\n"
            "已迭代次数：{iteration_count} / {max_iterations}\n\n"
            "最新工具执行结果：\n{latest_tool_result}",
        ),
    ]
)


# ─────────────────────────────────────────────
# 5. 多 Agent 提示词（supervisor.py + workers.py 使用）
# ─────────────────────────────────────────────

# 5-A: Supervisor — 任务分解 + Worker 调度编排
#
# 输入变量: task, user_context, memory_context
# 输出格式: JSON { analysis, execution_mode, assignments[], merge_strategy }
_MULTI_SUPERVISOR = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个多 Agent 任务调度主管（Supervisor）。\n"
            "你负责将复杂任务分解为子任务，并分配给以下专业 Worker Agents：\n\n"
            "  · ResearchAgent  — 信息收集：RAG 知识库检索 + 网络搜索\n"
            "  · WriterAgent    — 内容创作：基于研究结果生成结构化文档/报告\n"
            "  · ReviewerAgent  — 质量审查：评分、挑错、提出修改意见\n\n"
            "调度规则：\n"
            "1. 分析任务性质，决定需要哪些 Worker（可以不需要全部三个）。\n"
            "2. 明确每个 Worker 的输入依赖（如 Writer 必须依赖 Research 的输出）。\n"
            "3. 选择 sequential（串行，有依赖时）或 parallel（并行，无依赖时）调度。\n"
            "4. 每个子任务的 instruction 必须独立完整，Worker 仅能看到自己的指令。\n\n"
            "严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
            "{{\n"
            '  "analysis": "<对主任务的理解和分解思路（2-3句话）>",\n'
            '  "execution_mode": "<sequential 或 parallel>",\n'
            '  "assignments": [\n'
            "    {{\n"
            '      "agent": "<ResearchAgent|WriterAgent|ReviewerAgent>",\n'
            '      "instruction": "<给该 Agent 的完整任务指令>",\n'
            '      "depends_on": [<依赖前置 assignment 的序号列表，0-indexed，无依赖则为[]>],\n'
            '      "priority": <1-5，5最高>\n'
            "    }}\n"
            "  ],\n"
            '  "merge_strategy": "<最终如何整合各 Worker 结果（1句话）>"\n'
            "}}",
        ),
        (
            "human",
            "主任务：{task}\n\n"
            "用户背景信息：\n{user_context}\n\n"
            "相关记忆上下文：\n{memory_context}",
        ),
    ]
)

# 5-B: ResearchAgent — 信息收集 + 研究报告生成
#
# 输入变量: available_tools, instruction, main_task
# 输出格式: Markdown 结构化研究报告
_MULTI_RESEARCH = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的信息研究员（ResearchAgent）。\n"
            "你擅长从知识库和网络中收集、筛选、整合信息，产出结构化研究报告。\n\n"
            "可用工具：\n{available_tools}\n\n"
            "工作准则：\n"
            "1. 优先检索企业内部知识库（rag_retrieval），再补充外部搜索（web_search）。\n"
            "2. 信息需注明来源，区分「知识库来源」与「网络来源」。\n"
            "3. 对相互矛盾的信息，标注「存在分歧」并列出各方观点。\n"
            "4. 报告格式：背景摘要 → 核心发现（分点）→ 数据/引用 → 信息缺口（若有）。\n"
            "5. 保持客观，不要加入主观判断或行动建议。",
        ),
        (
            "human",
            "研究任务：{instruction}\n\n"
            "主任务背景：{main_task}\n\n"
            "请输出结构化研究报告（Markdown 格式）。",
        ),
    ]
)

# 5-C: WriterAgent — 基于研究结果创作结构化内容
#
# 输入变量: instruction, research_output, main_task, output_format
# 输出格式: 由 output_format 变量指定（Markdown / 报告 / 邮件等）
_MULTI_WRITER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的内容创作专家（WriterAgent）。\n"
            "你基于研究员提供的资料，创作结构清晰、语言流畅的企业级内容。\n\n"
            "写作准则：\n"
            "1. 严格基于提供的研究资料，不要编造数据或引用。\n"
            "2. 内容结构：执行摘要（3句话） → 正文（分章节） → 结论与建议。\n"
            "3. 语言风格：专业、简洁、易读，避免堆砌术语。\n"
            "4. 数字和关键结论须与研究资料完全一致，不要自行四舍五入或改写。\n"
            "5. 如研究资料存在信息缺口，在对应位置标注「[待补充：<缺失信息描述>]」。",
        ),
        (
            "human",
            "写作任务：{instruction}\n\n"
            "研究资料：\n{research_output}\n\n"
            "主任务背景：{main_task}\n\n"
            "输出格式要求：{output_format}",
        ),
    ]
)

# 5-D: ReviewerAgent — 质量评审 + 改进意见
#
# 输入变量: instruction, content_to_review, research_reference, main_task
# 输出格式: JSON { scores{}, total_score, verdict, strengths[], issues[], revised_sections }
_MULTI_REVIEWER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严格的内容质量审查专家（ReviewerAgent）。\n"
            "你从专业性、准确性、完整性、可读性四个维度对内容进行评审，各维度满分 25 分。\n\n"
            "评审维度：\n"
            "  · 专业性（25分）：术语使用是否准确，逻辑是否严谨\n"
            "  · 准确性（25分）：数据和引用是否与原始资料一致，是否存在事实错误\n"
            "  · 完整性（25分）：是否覆盖了主要议题，有无明显遗漏\n"
            "  · 可读性（25分）：结构是否清晰，语言是否流畅易懂\n\n"
            "裁定标准：\n"
            "  · approved        ≥ 80 分，内容质量达标，可直接交付\n"
            "  · needs_revision  60-79 分，需修改后再次确认\n"
            "  · rejected        < 60 分，质量不合格，需重新创作\n\n"
            "严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
            "{{\n"
            '  "scores": {{\n'
            '    "professionalism": <0-25>,\n'
            '    "accuracy": <0-25>,\n'
            '    "completeness": <0-25>,\n'
            '    "readability": <0-25>\n'
            "  }},\n"
            '  "total_score": <0-100>,\n'
            '  "verdict": "<approved|needs_revision|rejected>",\n'
            '  "strengths": ["<优点1>", "<优点2>"],\n'
            '  "issues": [\n'
            "    {{\n"
            '      "severity": "<high|medium|low>",\n'
            '      "description": "<问题描述>",\n'
            '      "suggestion": "<修改建议>"\n'
            "    }}\n"
            "  ],\n"
            '  "revised_sections": '
            '"<若 verdict=needs_revision，给出修改后的段落原文；否则为空字符串>"\n'
            "}}",
        ),
        (
            "human",
            "审查任务：{instruction}\n\n"
            "待审内容：\n{content_to_review}\n\n"
            "原始研究资料（用于核实准确性）：\n{research_reference}\n\n"
            "主任务背景：{main_task}",
        ),
    ]
)


# ─────────────────────────────────────────────
# 6. PromptRegistry — 单例注册表
# ─────────────────────────────────────────────


class PromptRegistry:
    """
    全项目提示词的单例注册表。

    类比药房药柜：每种药（提示词）有固定编号（PromptKey），
    取药时按编号查找，不需要知道药放在哪个抽屉。

    设计特点：
    - Eager Loading：所有模板在模块导入时初始化，避免运行时延迟。
    - 热更新支持：prompt_version_service.py 可通过 register() 在运行时
      用数据库中的 A/B 版本覆盖默认模板，无需重启服务。
    - 只对外暴露 PromptKey 枚举，不暴露内部字典键名。
    """

    def __init__(self) -> None:
        self._store: dict[str, ChatPromptTemplate] = {
            # Chat
            PromptKey.CHAT_SYSTEM: _CHAT_SYSTEM,
            PromptKey.CHAT_HISTORY_AWARE: _CHAT_HISTORY_AWARE,
            # RAG
            PromptKey.RAG_GENERATE: _RAG_GENERATE,
            PromptKey.RAG_CONDENSE_QUESTION: _RAG_CONDENSE_QUESTION,
            # 单 Agent
            PromptKey.AGENT_PLANNER: _AGENT_PLANNER,
            PromptKey.AGENT_TOOL_SELECTOR: _AGENT_TOOL_SELECTOR,
            PromptKey.AGENT_OBSERVER: _AGENT_OBSERVER,
            # 多 Agent
            PromptKey.MULTI_SUPERVISOR: _MULTI_SUPERVISOR,
            PromptKey.MULTI_RESEARCH: _MULTI_RESEARCH,
            PromptKey.MULTI_WRITER: _MULTI_WRITER,
            PromptKey.MULTI_REVIEWER: _MULTI_REVIEWER,
        }

    def get(self, key: PromptKey) -> ChatPromptTemplate:
        """
        获取指定 key 的提示词模板。若 key 未注册，抛出 KeyError。
        """
        template = self._store.get(key)
        if template is None:
            raise KeyError(
                f"PromptKey '{key}' 未注册。" f"可用 key：{self.list_keys()}"
            )
        return template

    def register(self, key: PromptKey, template: ChatPromptTemplate) -> None:
        """
        注册或覆盖一个提示词模板（支持运行时热更新）。

        主要用途：prompt_version_service.py 从数据库加载 A/B 测试版本时，
        动态覆盖内置默认模板，实现热更新而无需重启服务。

        示例：
            from core.prompts import prompt_registry, PromptKey
            new_tmpl = ChatPromptTemplate.from_messages([...])
            prompt_registry.register(PromptKey.RAG_GENERATE, new_tmpl)
        """
        self._store[key] = template

    def list_keys(self) -> list[str]:
        """返回所有已注册的 PromptKey 列表（调试与文档生成用）。"""
        return sorted(self._store.keys())

    def __repr__(self) -> str:
        return f"PromptRegistry(registered={len(self._store)} templates)"


# 模块级单例，全项目共享同一个注册表实例
prompt_registry = PromptRegistry()


# ─────────────────────────────────────────────
# 7. 便捷访问函数
# ─────────────────────────────────────────────


def get_prompt(key: PromptKey) -> ChatPromptTemplate:
    """
    全局便捷函数：获取提示词模板。

    推荐的调用方式（其他模块）：
        from core.prompts import get_prompt, PromptKey

        # 直接组装 LCEL Chain
        chain = get_prompt(PromptKey.RAG_GENERATE) | llm | StrOutputParser()

        # 或单独使用模板
        template = get_prompt(PromptKey.AGENT_PLANNER)
        messages = template.format_messages(
            available_tools="...", task="...", memory_context="无"
        )
    """
    return prompt_registry.get(key)


def render_prompt(key: PromptKey, **kwargs: Any) -> list[dict[str, str]]:
    """
    渲染提示词为 OpenAI messages 列表格式（list[dict]）。

    适用于绕过 LangChain Chain、直接向 OpenAI API 发送请求的场景
    （例如在 Celery Worker 中为减少依赖而直接调用 httpx）。

    示例：
        messages = render_prompt(
            PromptKey.AGENT_PLANNER,
            available_tools="web_search, calculator, rag_retrieval",
            task="分析今年 Q3 销售数据并生成摘要",
            memory_context="无",
        )
        response = await openai_client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
    """
    template = prompt_registry.get(key)
    formatted = template.format_messages(**kwargs)
    return [{"role": msg.type, "content": msg.content} for msg in formatted]
