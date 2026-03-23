"""
@File       : llm_client.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: LangChain ChatOpenAI 异步封装，提供统一的 LLM 调用入口。
@Features:
  - 基于 LangChain ChatOpenAI，支持 Streaming / 非 Streaming 双模式
  - async invoke()    → str（非流式，等待完整响应）
  - async stream()    → AsyncGenerator[str, None]（流式，逐 Token yield）
  - 每次调用自动记录 Token 消耗到结构化日志
      prompt_tokens / completion_tokens / total_tokens / model / latency_ms
  - 模型不可用 / 超时 / 配额耗尽 → 抛出对应 LLMError 子类
  - get_llm_client() 工厂函数，兼容 FastAPI Depends 依赖注入
  - LLMClientPool 按模型名缓存客户端实例，避免重复初始化

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import time
from typing import AsyncGenerator, List, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.config.settings import get_settings
from app.core.exceptions import ErrorCode, LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# 消息格式转换工具
# ---------------------------------------------------------------------------

# 接受的消息输入类型：LangChain BaseMessage 列表，或原始 dict 列表
MessageInput = Union[List[BaseMessage], List[dict]]


def _normalize_messages(messages: MessageInput) -> List[BaseMessage]:
    """
    将 dict 格式的消息列表统一转换为 LangChain BaseMessage 列表。

    支持格式：
        {"role": "system",    "content": "..."}  → SystemMessage
        {"role": "user",      "content": "..."}  → HumanMessage
        {"role": "assistant", "content": "..."}  → AIMessage

    已经是 BaseMessage 实例的直接透传。
    """
    if not messages:
        return []

    # 已经是 BaseMessage，直接返回
    if isinstance(messages[0], BaseMessage):
        return messages  # type: ignore[return-value]

    normalized: List[BaseMessage] = []
    _role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "")
        cls = _role_map.get(role, HumanMessage)
        normalized.append(cls(content=content))
    return normalized


# ---------------------------------------------------------------------------
# Token 消耗记录
# ---------------------------------------------------------------------------


def _log_token_usage(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: float,
    streaming: bool,
) -> None:
    """
    将 Token 消耗写入结构化日志。
    日志级别 INFO，字段与 cost_service（Step 24）保持对齐，便于后续接入。
    """
    logger.info(
        "LLM 调用完成",
        extra={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "streaming": streaming,
        },
    )


# ---------------------------------------------------------------------------
# OpenAI 异常 → LLMError 映射
# ---------------------------------------------------------------------------


def _map_openai_error(exc: Exception, model: str) -> LLMError:
    """
    将 openai SDK 抛出的异常映射为 Synaris LLMError 子类，
    附带调试友好的 detail 字段。
    """
    detail = {"model": model, "error": str(exc)}

    if isinstance(exc, APITimeoutError):
        return LLMError(
            message=f"模型 {model} 响应超时，请稍后重试",
            error_code=ErrorCode.LLM_TIMEOUT,
            detail=detail,
        )
    if isinstance(exc, RateLimitError):
        return LLMError(
            message=f"OpenAI 配额耗尽或请求频率超限（{model}）",
            error_code=ErrorCode.LLM_QUOTA_EXCEEDED,
            detail=detail,
        )
    if isinstance(exc, AuthenticationError):
        return LLMError(
            message="OpenAI API Key 无效，请检查配置",
            error_code=ErrorCode.LLM_UNAVAILABLE,
            detail=detail,
        )
    if isinstance(exc, APIConnectionError):
        return LLMError(
            message=f"无法连接到 OpenAI 服务（{model}）",
            error_code=ErrorCode.LLM_UNAVAILABLE,
            detail=detail,
        )
    if isinstance(exc, APIStatusError):
        # 4xx / 5xx HTTP 状态码
        status = exc.status_code
        if status == 400 and "maximum context" in str(exc).lower():
            return LLMError(
                message="输入文本超出模型最大 Context 长度",
                error_code=ErrorCode.LLM_CONTEXT_TOO_LONG,
                detail={**detail, "status_code": status},
            )
        return LLMError(
            message=f"OpenAI API 返回错误（HTTP {status}）",
            error_code=ErrorCode.LLM_UNAVAILABLE,
            detail={**detail, "status_code": status},
        )

    # 未知异常兜底
    return LLMError(
        message=f"LLM 调用失败：{exc}",
        error_code=ErrorCode.LLM_UNAVAILABLE,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# LLMClientPool — 按模型名缓存 ChatOpenAI 实例
# ---------------------------------------------------------------------------


class LLMClientPool:
    """
    按模型名称维护 ChatOpenAI 实例缓存，避免重复初始化。

    线程安全说明：Python 的 dict 读写在 CPython GIL 下是原子的，
    异步场景下单进程内无竞争问题。多进程部署（Gunicorn）各进程独立缓存，
    不共享状态，也无问题。
    """

    def __init__(self) -> None:
        self._cache: dict[str, ChatOpenAI] = {}

    def get(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        request_timeout: int = 60,
    ) -> ChatOpenAI:
        """
        获取指定模型的 ChatOpenAI 实例。
        相同 model + streaming 组合复用缓存实例；
        不同 temperature / max_tokens 调用时应使用 with_config() 覆盖，
        而非重建实例（LangChain 支持运行时参数覆盖）。
        """
        cache_key = f"{model}:stream={streaming}"
        if cache_key not in self._cache:
            self._cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_api_base or None,
                request_timeout=request_timeout,
                max_retries=0,  # 重试由上层 llm_router 的降级逻辑统一管理
            )
            logger.debug(
                "ChatOpenAI 实例已创建",
                extra={"model": model, "streaming": streaming},
            )
        return self._cache[cache_key]

    def clear(self) -> None:
        """清空缓存（测试用途）。"""
        self._cache.clear()


# 模块级单例
_pool = LLMClientPool()


# ---------------------------------------------------------------------------
# 核心调用接口
# ---------------------------------------------------------------------------


async def invoke(
    messages: MessageInput,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    request_timeout: int = 60,
) -> str:
    """
    非流式 LLM 调用，返回完整响应字符串。

    Args:
        messages:        对话消息列表（dict 或 LangChain BaseMessage）
        model:           OpenAI 模型名称，如 "gpt-4o" / "gpt-4o-mini"
        temperature:     生成温度（0.0-2.0）
        max_tokens:      最大生成 Token 数，None 表示模型默认值
        request_timeout: 单次请求超时秒数

    Returns:
        LLM 生成的完整文本字符串

    Raises:
        LLMError: 所有 OpenAI 调用失败情况均转换为此异常
    """
    normalized = _normalize_messages(messages)
    client = _pool.get(
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=False,
        request_timeout=request_timeout,
    )

    start = time.perf_counter()
    try:
        logger.debug(
            "LLM 调用开始（非流式）",
            extra={"model": model, "message_count": len(normalized)},
        )
        response = await client.ainvoke(normalized)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 提取 Token 消耗（response_metadata 由 LangChain 填充）
        usage = response.response_metadata.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        _log_token_usage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=elapsed_ms,
            streaming=False,
        )

        content = response.content
        if not isinstance(content, str):
            # 多模态响应兜底（当前仅使用文本）
            content = str(content)
        return content

    except LLMError:
        raise
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "LLM 调用失败（非流式）",
            extra={
                "model": model,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": str(exc),
            },
        )
        raise _map_openai_error(exc, model) from exc


async def stream(
    messages: MessageInput,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    request_timeout: int = 60,
) -> AsyncGenerator[str, None]:
    """
    流式 LLM 调用，逐 Token yield 文本片段。

    Token 统计在流结束后一次性写入日志（累加 chunk 中的 usage_metadata）。

    典型用法（SSE 输出）：
        async for chunk in stream(messages, model="gpt-4o"):
            yield f"data: {chunk}\\n\\n"

    Raises:
        LLMError: 流建立失败或中途中断时抛出
    """
    normalized = _normalize_messages(messages)
    client = _pool.get(
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        request_timeout=request_timeout,
    )

    start = time.perf_counter()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    try:
        logger.debug(
            "LLM 调用开始（流式）",
            extra={"model": model, "message_count": len(normalized)},
        )

        async for chunk in client.astream(normalized):
            # 部分 chunk 携带 usage_metadata（尤其是最后一个 chunk）
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                meta = chunk.usage_metadata
                total_prompt_tokens = meta.get("input_tokens", total_prompt_tokens)
                total_completion_tokens = meta.get(
                    "output_tokens", total_completion_tokens
                )
                total_tokens = meta.get("total_tokens", total_tokens)

            content = chunk.content
            if content:
                yield content

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 若 usage_metadata 未提供（部分模型/版本不返回），使用 0 占位
        _log_token_usage(
            model=model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens
            or (total_prompt_tokens + total_completion_tokens),
            latency_ms=elapsed_ms,
            streaming=True,
        )

    except LLMError:
        raise
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "LLM 调用失败（流式）",
            extra={
                "model": model,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": str(exc),
            },
        )
        raise _map_openai_error(exc, model) from exc


# ---------------------------------------------------------------------------
# FastAPI 依赖注入
# ---------------------------------------------------------------------------


def get_llm_client() -> "LLMClientInterface":
    """
    FastAPI Depends 兼容的依赖注入工厂。
    返回 LLMClientInterface 包装实例，便于路由层注入与单测 Mock。

    用法：
        @router.post("/chat")
        async def chat(
            body: ChatRequest,
            llm: LLMClientInterface = Depends(get_llm_client),
        ):
            result = await llm.invoke(messages, model="gpt-4o")
    """
    return LLMClientInterface()


class LLMClientInterface:
    """
    对 invoke / stream 模块函数的薄封装，便于单测时替换为 Mock 实现。
    """

    async def invoke(
        self,
        messages: MessageInput,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        return await invoke(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def stream(
        self,
        messages: MessageInput,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        return stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
