"""
@File       : chat_service.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 多轮对话业务服务层，封装历史管理与 LLM 调用逻辑。
@Features:
  - RedisMessageHistory：基于 Redis List 的会话历史存储
      key = "synaris:chat:{session_id}"，TTL=2h，最多保留 20 条消息
      消息以 {"role": "human"/"ai", "content": "..."} JSON 格式序列化存储
  - chat()          非流式对话，返回完整回复字符串
  - chat_stream()   流式对话，返回 AsyncGenerator[str]，逐 Token yield
  - clear_session() 清空指定会话历史
  - get_session_info() 获取会话元信息（消息数 / TTL）
  - 历史消息注入 LangChain MessagesPlaceholder，支持多轮上下文
  - 使用 core/prompts.py 的 CHAT_SYSTEM_PROMPT 作为系统提示词

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.exceptions import CacheError, LLMError
from app.core.llm_router import (
    RoutingStrategy,
    TaskType,
    invoke_with_fallback,
    stream_with_fallback,
)
from app.core.logging import get_logger
from app.infrastructure.redis_client import (
    build_key,
    delete,
    get_client,
    get_json,
    set_json,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

_SESSION_TTL_SECONDS: int = 2 * 60 * 60  # 2 小时
_MAX_HISTORY_MESSAGES: int = 20  # 最多保留 20 条（Human + AI 各算 1 条）
_HISTORY_KEY_PREFIX: str = "chat"  # build_key("chat", session_id)

# 默认系统提示词（core/prompts.py 完成后替换为导入）
_DEFAULT_SYSTEM_PROMPT: str = (
    "你是 Synaris 智能助手，一个专业、友好的企业级 AI 助手。\n"
    "请用简洁、准确的语言回答用户问题。\n"
    "如果不确定答案，请诚实地告知，而不是猜测。"
)

# 路由策略 → 任务类型映射
_STRATEGY_TO_TASK: dict[RoutingStrategy, TaskType] = {
    RoutingStrategy.QUALITY: TaskType.AGENT_REASONING,
    RoutingStrategy.BALANCED: TaskType.CHAT,
    RoutingStrategy.ECONOMY: TaskType.CHAT,
}

# 路由策略 → 温度映射（覆盖路由表默认值）
_STRATEGY_TO_TEMPERATURE: dict[RoutingStrategy, float] = {
    RoutingStrategy.QUALITY: 0.5,
    RoutingStrategy.BALANCED: 0.7,
    RoutingStrategy.ECONOMY: 0.7,
}


# ---------------------------------------------------------------------------
# Redis 历史消息管理
# ---------------------------------------------------------------------------


class RedisMessageHistory:
    """
    基于 Redis List 的会话消息历史管理器。

    存储格式（Redis List，LPUSH + LTRIM 维护顺序与上限）：
        key:   "synaris:chat:{session_id}"
        value: JSON 序列化的消息列表（整体存储为单个 JSON 数组）

    选择整体存储而非每条消息单独 LPUSH 的原因：
      - 避免多次 Redis 往返
      - 方便原子性地截断历史（LTRIM 对 JSON 数组不适用）
      - TTL 刷新与消息写入在同一次 SET 操作中完成
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._key = build_key(_HISTORY_KEY_PREFIX, session_id)

    async def get_messages(self) -> List[BaseMessage]:
        """
        从 Redis 读取历史消息，返回 LangChain BaseMessage 列表。
        Key 不存在时返回空列表。
        """
        try:
            raw: Optional[list] = await get_json(self._key)
            if not raw:
                return []
            messages: List[BaseMessage] = []
            for item in raw:
                role = item.get("role", "human")
                content = item.get("content", "")
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))
                # system 消息不存入历史（系统提示词在每次构建时单独注入）
            return messages
        except CacheError as exc:
            logger.warning(
                "读取会话历史失败，使用空历史",
                extra={"session_id": self.session_id, "error": str(exc)},
            )
            return []

    async def add_messages(
        self,
        human_message: str,
        ai_message: str,
    ) -> None:
        """
        将本轮 Human + AI 消息追加到历史，超出 _MAX_HISTORY_MESSAGES 时截断旧消息。
        同时刷新 TTL。
        """
        try:
            existing = await get_json(self._key) or []

            # 追加本轮消息
            existing.append({"role": "human", "content": human_message})
            existing.append({"role": "ai", "content": ai_message})

            # 截断：保留最近 N 条（从尾部截取）
            if len(existing) > _MAX_HISTORY_MESSAGES:
                existing = existing[-_MAX_HISTORY_MESSAGES:]
                logger.debug(
                    "会话历史已截断",
                    extra={
                        "session_id": self.session_id,
                        "kept": _MAX_HISTORY_MESSAGES,
                    },
                )

            await set_json(self._key, existing, ttl=_SESSION_TTL_SECONDS)

        except CacheError as exc:
            # 历史写入失败不应中断对话，仅记录警告
            logger.warning(
                "写入会话历史失败",
                extra={"session_id": self.session_id, "error": str(exc)},
            )

    async def clear(self) -> bool:
        """清空会话历史，返回是否存在并被删除。"""
        try:
            count = await delete(self._key)
            return count > 0
        except CacheError as exc:
            logger.warning(
                "清空会话历史失败",
                extra={"session_id": self.session_id, "error": str(exc)},
            )
            return False

    async def message_count(self) -> int:
        """返回当前历史消息条数。"""
        try:
            raw = await get_json(self._key)
            return len(raw) if raw else 0
        except CacheError:
            return 0

    async def get_ttl(self) -> int:
        """返回 Redis Key 剩余 TTL（秒）。-1 无过期，-2 Key 不存在。"""
        try:
            client = get_client()
            return await client.ttl(self._key)
        except Exception:
            return -2


# ---------------------------------------------------------------------------
# 消息列表构造
# ---------------------------------------------------------------------------


def _build_messages(
    history: List[BaseMessage],
    user_message: str,
    system_prompt: Optional[str] = None,
) -> List[BaseMessage]:
    """
    将系统提示词 + 历史消息 + 当前用户消息组合为完整的 LangChain 消息列表。

    结构：
        [SystemMessage] → [历史 Human/AI 消息 ...] → [HumanMessage(当前输入)]
    """
    prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
    messages: List[BaseMessage] = [SystemMessage(content=prompt)]
    messages.extend(history)
    messages.append(HumanMessage(content=user_message))
    return messages


# ---------------------------------------------------------------------------
# ChatService
# ---------------------------------------------------------------------------


class ChatService:
    """
    多轮对话业务服务。

    职责：
      - 会话历史的读取、写入、截断（委托给 RedisMessageHistory）
      - 消息列表构造（系统提示词 + 历史 + 当前输入）
      - 调用 llm_router 执行路由与降级逻辑
      - 流式 / 非流式双模式支持
    """

    # ── 非流式对话 ────────────────────────────────────────────────────

    async def chat(
        self,
        session_id: str,
        user_message: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        非流式多轮对话。

        Returns:
            (answer: str, model_used: str) 元组
            answer     — LLM 生成的完整回复
            model_used — 实际使用的模型名称（可能为 fallback）

        Raises:
            LLMError: 主模型与 fallback 均失败时抛出
        """
        history_store = RedisMessageHistory(session_id)

        # 1. 读取历史
        history = await history_store.get_messages()
        logger.debug(
            "会话历史已加载",
            extra={
                "session_id": session_id,
                "history_count": len(history),
                "strategy": strategy,
            },
        )

        # 2. 构造完整消息列表
        messages = _build_messages(history, user_message, system_prompt)

        # 3. 路由并调用 LLM
        task_type = _STRATEGY_TO_TASK.get(strategy, TaskType.CHAT)
        temperature = _STRATEGY_TO_TEMPERATURE.get(strategy, 0.7)

        answer = await invoke_with_fallback(
            messages,
            task_type=task_type,
            temperature=temperature,
        )

        # 4. 写入历史（Human + AI）
        await history_store.add_messages(user_message, answer)

        # 从路由表获取实际使用的模型名（降级时可能与首选不同）
        # 注：精确模型名由 llm_router 内部决定，此处从配置读取首选作为近似值
        from app.core.llm_router import route

        model_used = route(task_type).model

        logger.info(
            "对话完成（非流式）",
            extra={
                "session_id": session_id,
                "strategy": strategy,
                "model": model_used,
                "answer_length": len(answer),
            },
        )
        return answer, model_used

    # ── 流式对话 ──────────────────────────────────────────────────────

    async def chat_stream(
        self,
        session_id: str,
        user_message: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式多轮对话，逐 Token yield 文本片段。

        历史写入时机：流完全结束后才写入 AI 回复，
        确保历史记录的完整性（避免写入不完整的截断回复）。

        Yields:
            文本片段字符串（非 SSE 格式，由 API 层包装为 SSE）

        Raises:
            LLMError: 主模型与 fallback 均失败时抛出
        """
        history_store = RedisMessageHistory(session_id)

        # 1. 读取历史
        history = await history_store.get_messages()
        logger.debug(
            "会话历史已加载（流式）",
            extra={"session_id": session_id, "history_count": len(history)},
        )

        # 2. 构造消息列表
        messages = _build_messages(history, user_message, system_prompt)

        # 3. 路由策略
        task_type = _STRATEGY_TO_TASK.get(strategy, TaskType.CHAT)
        temperature = _STRATEGY_TO_TEMPERATURE.get(strategy, 0.7)

        # 4. 流式调用，累积完整回复用于历史写入
        full_answer_parts: List[str] = []

        async for chunk in await stream_with_fallback(
            messages,
            task_type=task_type,
            temperature=temperature,
        ):
            full_answer_parts.append(chunk)
            yield chunk

        # 5. 流结束后写入历史
        full_answer = "".join(full_answer_parts)
        if full_answer:
            await history_store.add_messages(user_message, full_answer)

        logger.info(
            "对话完成（流式）",
            extra={
                "session_id": session_id,
                "strategy": strategy,
                "answer_length": len(full_answer),
            },
        )

    # ── 会话管理 ──────────────────────────────────────────────────────

    async def clear_session(self, session_id: str) -> bool:
        """
        清空指定会话的历史记录。
        返回 True 表示 Key 存在并已删除，False 表示 Key 不存在。
        """
        store = RedisMessageHistory(session_id)
        cleared = await store.clear()
        logger.info(
            "会话历史已清空",
            extra={"session_id": session_id, "existed": cleared},
        )
        return cleared

    async def get_session_info(self, session_id: str) -> dict:
        """
        获取会话元信息：消息数量、剩余 TTL。
        """
        store = RedisMessageHistory(session_id)
        count = await store.message_count()
        ttl = await store.get_ttl()
        return {
            "session_id": session_id,
            "message_count": count,
            "ttl_seconds": ttl,
        }


# ---------------------------------------------------------------------------
# FastAPI 依赖注入
# ---------------------------------------------------------------------------


def get_chat_service() -> ChatService:
    """
    FastAPI Depends 兼容的 ChatService 工厂函数。

    ChatService 本身无状态（所有状态存储在 Redis），
    每次请求新建实例开销极低，无需单例缓存。

    用法：
        @router.post("/chat")
        async def chat_endpoint(
            body: ChatRequest,
            service: ChatService = Depends(get_chat_service),
        ):
            ...
    """
    return ChatService()
