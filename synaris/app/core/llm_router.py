"""
@File       : llm_router.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: LLM 多模型路由层，根据任务类型选择最优模型并实现自动降级。
@Features:
  - RoutingStrategy 枚举：QUALITY / BALANCED / ECONOMY
  - TaskType 枚举：CHAT / RAG / AGENT_REASONING / CODE / EMBEDDING
  - TASK_ROUTING_TABLE：任务类型 → 首选模型 + 路由策略 的静态映射表
  - route(task_type) → 返回首选模型名称
  - invoke_with_fallback()：
      首选模型失败 → 自动切换 FALLBACK_MODEL → 仍失败才向上抛 LLMError
  - stream_with_fallback()：同上，流式版本
  - 每次路由决策、降级触发均写入结构化日志
  - ModelHealthTracker：轻量级熔断状态追踪，连续失败 N 次后标记为不健康，
    跳过首选直接走降级（可选功能，通过 settings.enable_model_health_tracking 控制）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, List, Optional, Union

from app.config.settings import get_settings
from app.core.exceptions import ErrorCode, LLMError
from app.core.logging import get_logger
from app.infrastructure.llm_client import (
    MessageInput,
    invoke as _llm_invoke,
    stream as _llm_stream,
)

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------


class RoutingStrategy(str, Enum):
    """
    路由策略枚举，控制模型选择的质量 / 成本权衡。

    QUALITY  → 优先使用高质量模型（gpt-4o），适合推理、分析类任务
    BALANCED → 质量与成本均衡（gpt-4o-mini），适合日常对话
    ECONOMY  → 成本优先（gpt-3.5-turbo），适合批量、低优先级任务
    """

    QUALITY = "quality"
    BALANCED = "balanced"
    ECONOMY = "economy"


class TaskType(str, Enum):
    """
    任务类型枚举，供服务层标注当前调用的业务场景。

    CHAT             → 多轮对话（对话质量优先）
    RAG              → 检索增强生成（需较强阅读理解能力）
    AGENT_REASONING  → Agent 推理 / 规划 / 工具选择（最高质量要求）
    CODE             → 代码生成 / 分析
    EMBEDDING        → 向量嵌入（不经过 ChatOpenAI，走 EmbeddingClient）
    SUMMARY          → 文本摘要（均衡策略，成本敏感）
    EVALUATION       → LLM 评估流水线（均衡策略）
    """

    CHAT = "chat"
    RAG = "rag"
    AGENT_REASONING = "agent_reasoning"
    CODE = "code"
    EMBEDDING = "embedding"
    SUMMARY = "summary"
    EVALUATION = "evaluation"


# ---------------------------------------------------------------------------
# 路由配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """
    单条路由规则：首选模型 + 路由策略 + 模型参数默认值。
    """

    model: str  # 首选模型名，如 "gpt-4o"
    strategy: RoutingStrategy  # 路由策略
    temperature: float = 0.7  # 生成温度
    max_tokens: Optional[int] = None  # None 表示模型默认值


# 任务类型 → 路由规则静态映射表
# 修改此表即可调整全局路由策略，无需改动业务代码
TASK_ROUTING_TABLE: dict[TaskType, ModelConfig] = {
    TaskType.CHAT: ModelConfig(
        model="gpt-4o-mini",
        strategy=RoutingStrategy.BALANCED,
        temperature=0.7,
    ),
    TaskType.RAG: ModelConfig(
        model="gpt-4o-mini",
        strategy=RoutingStrategy.BALANCED,
        temperature=0.3,  # RAG 生成需较低温度以保证忠实度
    ),
    TaskType.AGENT_REASONING: ModelConfig(
        model="gpt-4o",
        strategy=RoutingStrategy.QUALITY,
        temperature=0.1,  # 推理任务需要高确定性
        max_tokens=4096,
    ),
    TaskType.CODE: ModelConfig(
        model="gpt-4o",
        strategy=RoutingStrategy.QUALITY,
        temperature=0.2,
    ),
    TaskType.SUMMARY: ModelConfig(
        model="gpt-4o-mini",
        strategy=RoutingStrategy.BALANCED,
        temperature=0.5,
    ),
    TaskType.EVALUATION: ModelConfig(
        model="gpt-4o-mini",
        strategy=RoutingStrategy.BALANCED,
        temperature=0.0,  # 评估需完全确定性
    ),
    # EMBEDDING 不通过此路由表，由 EmbeddingClient 独立处理
    # 此处仅作占位，供 route() 查询时给出明确提示
    TaskType.EMBEDDING: ModelConfig(
        model="text-embedding-3-small",
        strategy=RoutingStrategy.ECONOMY,
    ),
}


# ---------------------------------------------------------------------------
# 模型健康追踪器（轻量级熔断）
# ---------------------------------------------------------------------------


@dataclass
class _ModelHealth:
    """单个模型的健康状态。"""

    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    is_healthy: bool = True


class ModelHealthTracker:
    """
    轻量级模型健康追踪器。

    连续失败 failure_threshold 次后将模型标记为不健康，
    等待 recovery_window_seconds 秒后自动恢复（允许下一次探测）。

    不健康的模型在 route_with_health_check() 中会被自动跳过，
    直接使用 FALLBACK_MODEL，减少等待超时的时间开销。
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_window_seconds: float = 60.0,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery_window = recovery_window_seconds
        self._state: dict[str, _ModelHealth] = defaultdict(lambda: _ModelHealth())

    def record_success(self, model: str) -> None:
        state = self._state[model]
        if not state.is_healthy or state.consecutive_failures > 0:
            logger.info(
                "模型健康状态恢复",
                extra={"model": model, "previous_failures": state.consecutive_failures},
            )
        state.consecutive_failures = 0
        state.is_healthy = True

    def record_failure(self, model: str) -> None:
        state = self._state[model]
        state.consecutive_failures += 1
        state.last_failure_time = time.time()
        if state.consecutive_failures >= self._threshold:
            state.is_healthy = False
            logger.warning(
                "模型已标记为不健康",
                extra={
                    "model": model,
                    "consecutive_failures": state.consecutive_failures,
                    "recovery_window_seconds": self._recovery_window,
                },
            )

    def is_healthy(self, model: str) -> bool:
        state = self._state[model]
        if state.is_healthy:
            return True
        # 检查是否超过恢复窗口（允许探测）
        if time.time() - state.last_failure_time >= self._recovery_window:
            logger.info(
                "模型恢复窗口到期，允许探测",
                extra={"model": model},
            )
            state.is_healthy = True  # 乐观恢复，下次调用失败再重新标记
            state.consecutive_failures = 0
            return True
        return False

    def reset(self, model: Optional[str] = None) -> None:
        """重置健康状态（测试用途）。"""
        if model:
            self._state.pop(model, None)
        else:
            self._state.clear()


# 模块级单例
_health_tracker = ModelHealthTracker(
    failure_threshold=3,
    recovery_window_seconds=60.0,
)


# ---------------------------------------------------------------------------
# 路由核心逻辑
# ---------------------------------------------------------------------------


def route(task_type: TaskType) -> ModelConfig:
    """
    根据任务类型返回路由配置（ModelConfig）。

    示例：
        config = route(TaskType.AGENT_REASONING)
        # → ModelConfig(model="gpt-4o", strategy=QUALITY, temperature=0.1)

    Args:
        task_type: 当前任务的业务类型

    Returns:
        ModelConfig：包含首选模型名及调用参数

    Raises:
        LLMError(EMBEDDING)：若 task_type=EMBEDDING，给出明确提示
    """
    if task_type == TaskType.EMBEDDING:
        raise LLMError(
            message="Embedding 任务应使用 EmbeddingClient，不经过 LLM Router",
            error_code=ErrorCode.LLM_INVALID_RESPONSE,
            detail={"task_type": task_type},
        )

    config = TASK_ROUTING_TABLE.get(task_type)
    if config is None:
        logger.warning(
            "未知任务类型，使用 BALANCED 默认策略",
            extra={"task_type": task_type},
        )
        config = TASK_ROUTING_TABLE[TaskType.CHAT]  # 兜底

    # 健康检查：首选模型不健康时直接切换 fallback
    primary_model = config.model
    fallback_model = settings.llm_fallback_model

    if (
        not _health_tracker.is_healthy(primary_model)
        and primary_model != fallback_model
    ):
        logger.info(
            "首选模型不健康，路由决策直接使用 Fallback",
            extra={
                "task_type": task_type,
                "primary_model": primary_model,
                "fallback_model": fallback_model,
            },
        )
        # 返回原 config 但替换模型名
        from dataclasses import replace

        config = replace(config, model=fallback_model)
    else:
        logger.debug(
            "路由决策",
            extra={
                "task_type": task_type,
                "model": primary_model,
                "strategy": config.strategy,
                "temperature": config.temperature,
            },
        )

    return config


# ---------------------------------------------------------------------------
# 带降级的调用接口
# ---------------------------------------------------------------------------


async def invoke_with_fallback(
    messages: MessageInput,
    *,
    task_type: TaskType,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    根据任务类型自动路由，调用失败时降级到 FALLBACK_MODEL。

    降级流程：
        route(task_type) → 首选模型
            ↓ 失败（LLMError）
        settings.llm_fallback_model
            ↓ 再失败
        向上抛出 LLMError（附带 primary + fallback 双重失败信息）

    Args:
        messages:    对话消息列表
        task_type:   任务类型（影响模型选择与温度）
        temperature: 覆盖路由表中的默认温度（None 表示使用路由表值）
        max_tokens:  覆盖最大生成 Token 数

    Returns:
        LLM 生成的完整文本字符串
    """
    config = route(task_type)
    primary_model = config.model
    fallback_model = settings.llm_fallback_model
    effective_temp = temperature if temperature is not None else config.temperature
    effective_max = max_tokens if max_tokens is not None else config.max_tokens

    # ── 尝试首选模型 ──────────────────────────────────────────────────
    try:
        result = await _llm_invoke(
            messages,
            model=primary_model,
            temperature=effective_temp,
            max_tokens=effective_max,
        )
        _health_tracker.record_success(primary_model)
        return result

    except LLMError as primary_exc:
        _health_tracker.record_failure(primary_model)

        # 首选模型与 fallback 相同时，无需重试，直接抛出
        if primary_model == fallback_model:
            logger.error(
                "LLM 调用失败（首选=Fallback，无法降级）",
                extra={
                    "task_type": task_type,
                    "model": primary_model,
                    "error_code": primary_exc.error_code.name,
                },
            )
            raise

        logger.warning(
            "首选模型失败，触发降级",
            extra={
                "task_type": task_type,
                "primary_model": primary_model,
                "fallback_model": fallback_model,
                "primary_error": primary_exc.error_code.name,
            },
        )

    # ── 尝试 Fallback 模型 ────────────────────────────────────────────
    try:
        result = await _llm_invoke(
            messages,
            model=fallback_model,
            temperature=effective_temp,
            max_tokens=effective_max,
        )
        _health_tracker.record_success(fallback_model)
        logger.info(
            "Fallback 模型调用成功",
            extra={"task_type": task_type, "fallback_model": fallback_model},
        )
        return result

    except LLMError as fallback_exc:
        _health_tracker.record_failure(fallback_model)
        logger.error(
            "首选模型与 Fallback 均失败，任务终止",
            extra={
                "task_type": task_type,
                "primary_model": primary_model,
                "fallback_model": fallback_model,
                "fallback_error": fallback_exc.error_code.name,
            },
        )
        raise LLMError(
            message="AI 服务暂时不可用，主模型与备用模型均无响应，请稍后重试",
            error_code=ErrorCode.LLM_UNAVAILABLE,
            detail={
                "task_type": task_type,
                "primary_model": primary_model,
                "fallback_model": fallback_model,
                "fallback_error": str(fallback_exc),
            },
        ) from fallback_exc


async def stream_with_fallback(
    messages: MessageInput,
    *,
    task_type: TaskType,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    流式版本的降级路由调用。

    流式接口的特殊性：流建立后才知道是否成功。
    因此降级策略为：
      1. 先尝试建立首选模型的流
      2. 若建立失败（连接阶段异常），切换 fallback 重试
      3. 若首选模型流中途中断，降级重新生成（从头开始，非断点续传）

    Yields:
        逐 Token 的文本片段

    Raises:
        LLMError: 首选和 fallback 均失败时抛出
    """
    config = route(task_type)
    primary_model = config.model
    fallback_model = settings.llm_fallback_model
    effective_temp = temperature if temperature is not None else config.temperature
    effective_max = max_tokens if max_tokens is not None else config.max_tokens

    # ── 尝试首选模型流 ────────────────────────────────────────────────
    primary_failed = False
    try:
        chunk_count = 0
        async for chunk in _llm_stream(
            messages,
            model=primary_model,
            temperature=effective_temp,
            max_tokens=effective_max,
        ):
            chunk_count += 1
            yield chunk

        _health_tracker.record_success(primary_model)
        return  # 流正常结束，直接返回

    except LLMError as primary_exc:
        _health_tracker.record_failure(primary_model)
        primary_failed = True

        if primary_model == fallback_model:
            raise

        logger.warning(
            "流式首选模型失败，触发降级",
            extra={
                "task_type": task_type,
                "primary_model": primary_model,
                "fallback_model": fallback_model,
                "error": primary_exc.error_code.name,
            },
        )

    if not primary_failed:
        return

    # ── 尝试 Fallback 模型流 ──────────────────────────────────────────
    try:
        async for chunk in _llm_stream(
            messages,
            model=fallback_model,
            temperature=effective_temp,
            max_tokens=effective_max,
        ):
            yield chunk

        _health_tracker.record_success(fallback_model)
        logger.info(
            "流式 Fallback 模型成功",
            extra={"task_type": task_type, "fallback_model": fallback_model},
        )

    except LLMError as fallback_exc:
        _health_tracker.record_failure(fallback_model)
        logger.error(
            "流式首选与 Fallback 均失败",
            extra={
                "task_type": task_type,
                "primary_model": primary_model,
                "fallback_model": fallback_model,
            },
        )
        raise LLMError(
            message="AI 服务暂时不可用，流式生成失败，请稍后重试",
            error_code=ErrorCode.LLM_UNAVAILABLE,
            detail={
                "task_type": task_type,
                "primary_model": primary_model,
                "fallback_model": fallback_model,
                "fallback_error": str(fallback_exc),
            },
        ) from fallback_exc


# ---------------------------------------------------------------------------
# 快捷函数：供服务层直接按任务类型调用（最常用入口）
# ---------------------------------------------------------------------------


async def chat(
    messages: MessageInput,
    *,
    streaming: bool = False,
    temperature: Optional[float] = None,
) -> Union[str, AsyncGenerator[str, None]]:
    """
    对话场景快捷入口。streaming=True 时返回 AsyncGenerator。
    """
    if streaming:
        return stream_with_fallback(
            messages, task_type=TaskType.CHAT, temperature=temperature
        )
    return await invoke_with_fallback(
        messages, task_type=TaskType.CHAT, temperature=temperature
    )


async def rag_generate(
    messages: MessageInput,
    *,
    streaming: bool = False,
) -> Union[str, AsyncGenerator[str, None]]:
    """RAG 生成场景快捷入口（低温度，高忠实度）。"""
    if streaming:
        return stream_with_fallback(messages, task_type=TaskType.RAG)
    return await invoke_with_fallback(messages, task_type=TaskType.RAG)


async def agent_reason(messages: MessageInput) -> str:
    """Agent 推理场景快捷入口（最高质量，非流式）。"""
    return await invoke_with_fallback(messages, task_type=TaskType.AGENT_REASONING)


"""

## 两文件关系与调用链

服务层（chat_service / rag_service / agents）
          ↓  调用
  llm_router.invoke_with_fallback(messages, task_type=...)
  llm_router.stream_with_fallback(messages, task_type=...)
          ↓  route() 决策 + 降级控制
  llm_router.route(task_type) → ModelConfig(model, temperature, ...)
          ↓  失败时切换 fallback_model
  llm_client.invoke(messages, model=...) / .stream(messages, model=...)
          ↓  LangChain ChatOpenAI.ainvoke / .astream
  OpenAI API
          ↑  异常时
  _map_openai_error() → LLMError
          ↑  被 router 捕获后
  _health_tracker.record_failure(model)
  → 连续失败 3 次 → is_healthy=False → 下次路由直接跳过

"""
