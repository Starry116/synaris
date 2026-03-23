"""
@File       : logging.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 结构化日志。
@Features:
  - JSON 格式（生产/ELK）与彩色文本格式（开发）由 LOG_FORMAT 自动切换
  - 每条日志包含：timestamp / level / message / trace_id /
                  module / function / line /
                  service_name / environment / version（ELK 标识字段）
  - TraceID 通过 contextvars 注入，异步安全，跨协程传递
  - @log_execution_time 装饰器同时支持同步与异步函数
  - get_logger(name) 工厂函数供全项目统一使用

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from app.config.settings import get_settings

# ─────────────────────────────────────────────────────────────────────────────
# TraceID — 基于 contextvars 的异步安全请求追踪
# ─────────────────────────────────────────────────────────────────────────────

# ContextVar 是协程安全的：每个异步任务拥有独立的副本，互不干扰
# 默认值为空字符串，便于日志字段始终存在（不会出现 KeyError）
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """获取当前协程/线程的 TraceID。无则返回空字符串。"""
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前上下文的 TraceID。

    Args:
        trace_id: 指定值；为 None 时自动生成 UUID4 短串（前8位）。

    Returns:
        实际设置的 trace_id 字符串。
    """
    tid = trace_id or uuid.uuid4().hex[:16]
    _trace_id_var.set(tid)
    return tid


def clear_trace_id() -> None:
    """清除当前上下文的 TraceID（请求结束后调用）。"""
    _trace_id_var.set("")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 中间件 — TraceID 注入
# ─────────────────────────────────────────────────────────────────────────────

class TraceIDMiddleware:
    """为每个 HTTP 请求自动注入 TraceID 的 ASGI 中间件。

    优先级：
      1. 客户端在 X-Trace-Id Header 中传入（便于链路追踪）
      2. 自动生成 UUID4 短串

    响应 Header 中同时回写 X-Trace-Id，方便前端和网关关联日志。

    用法（在 main.py 中注册）：
        from app.core.logging import TraceIDMiddleware
        app.add_middleware(TraceIDMiddleware)
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # 从请求 Header 中读取已有的 TraceID，否则新生成一个
        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-trace-id", b"").decode()
        trace_id = set_trace_id(incoming or None)

        async def send_with_trace(message: Any) -> None:
            """在响应 Header 中回写 TraceID。"""
            if message["type"] == "http.response.start":
                # 将 trace_id 追加到响应头
                headers_list = list(message.get("headers", []))
                headers_list.append(
                    (b"x-trace-id", trace_id.encode())
                )
                message = {**message, "headers": headers_list}
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        finally:
            # 请求结束后清理，防止 ContextVar 泄漏到连接池中的下一个请求
            clear_trace_id()


# ─────────────────────────────────────────────────────────────────────────────
# 自定义 Formatter — JSON 格式（生产 / ELK）
# ─────────────────────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """将日志记录序列化为单行 JSON，兼容 ELK Filebeat 自动采集。

    输出字段：
      固定字段     → timestamp, level, message, trace_id
      定位字段     → module, function, line
      ELK标识字段  → service_name, environment, version
      异常字段     → exception（仅在有异常时出现）
      扩展字段     → extra 中的自定义 key-value
    """

    # ELK 标识字段在模块加载时从 settings 读取一次，避免每条日志重复调用
    _settings = None

    @classmethod
    def _get_settings(cls):
        if cls._settings is None:
            cls._settings = get_settings()
        return cls._settings

    def format(self, record: logging.LogRecord) -> str:
        s = self._get_settings()

        payload: dict[str, Any] = {
            # ── 时间（ISO-8601，带时区，Kibana 可直接解析）────────────────
            "timestamp": datetime.now(timezone.utc).isoformat(),

            # ── 基础字段 ──────────────────────────────────────────────────
            "level": record.levelname,
            "message": record.getMessage(),
            "trace_id": get_trace_id() or "-",

            # ── 代码定位字段 ──────────────────────────────────────────────
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,

            # ── ELK / Prometheus 服务标识字段 ─────────────────────────────
            "service_name": s.log.LOG_SERVICE_NAME,
            "environment": s.log.LOG_ENVIRONMENT,
            "version": s.app.APP_VERSION,
        }

        # 异常信息：有则附加，无则不输出该字段（保持 JSON 简洁）
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # 扩展字段：调用方通过 logger.info("msg", extra={"key": "val"}) 传入
        # 过滤掉 logging.LogRecord 的内置属性，只保留用户自定义字段
        _builtin_keys = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "message", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName",
            "exc_info", "exc_text", "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in _builtin_keys and not key.startswith("_"):
                payload[key] = val

        return json.dumps(payload, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# 自定义 Formatter — 彩色文本格式（本地开发）
# ─────────────────────────────────────────────────────────────────────────────

# ANSI 颜色码
_COLORS = {
    "DEBUG":    "\033[36m",   # 青色
    "INFO":     "\033[32m",   # 绿色
    "WARNING":  "\033[33m",   # 黄色
    "ERROR":    "\033[31m",   # 红色
    "CRITICAL": "\033[35m",   # 紫色
}
_RESET = "\033[0m"
_DIM   = "\033[2m"
_BOLD  = "\033[1m"


class _ColorTextFormatter(logging.Formatter):
    """本地开发用的彩色文本格式。

    示例输出：
      2026-03-23 14:05:01  INFO  [abc123de]  chat_service.send:42
          ↳ 用户消息处理完成  duration_ms=123.4
    """

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        trace = get_trace_id() or "-"

        # 时间（精确到毫秒，本地调试够用）
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # 组装定位信息
        location = f"{record.module}.{record.funcName}:{record.lineno}"

        # 主日志行
        line = (
            f"{_DIM}{ts}{_RESET}  "
            f"{color}{_BOLD}{record.levelname:<8}{_RESET}  "
            f"{_DIM}[{trace[:8]}]{_RESET}  "
            f"{_DIM}{location}{_RESET}\n"
            f"    {color}↳{_RESET} {record.getMessage()}"
        )

        # 扩展字段（灰色展示在消息后）
        _builtin_keys = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "message", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName",
            "exc_info", "exc_text", "taskName",
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _builtin_keys and not k.startswith("_")
        }
        if extras:
            kv = "  ".join(f"{k}={v}" for k, v in extras.items())
            line += f"  {_DIM}{kv}{_RESET}"

        # 异常堆栈
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


# ─────────────────────────────────────────────────────────────────────────────
# 日志系统初始化
# ─────────────────────────────────────────────────────────────────────────────

def _build_handler() -> logging.Handler:
    """根据 LOG_FORMAT 配置构建对应的 Handler。"""
    s = get_settings()
    handler = logging.StreamHandler(sys.stdout)

    if s.log.LOG_FORMAT == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(_ColorTextFormatter())

    return handler


def _configure_root_logger() -> None:
    """配置根 Logger，确保全局只执行一次。"""
    s = get_settings()
    root = logging.getLogger()

    # 避免重复添加 Handler（在 uvicorn reload 等场景下可能被多次调用）
    if root.handlers:
        root.handlers.clear()

    root.setLevel(s.log.LOG_LEVEL)
    root.addHandler(_build_handler())

    # 抑制第三方库的噪音日志，避免 uvicorn / httpx 等刷屏
    for noisy in ("uvicorn.access", "httpx", "httpcore", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# 模块加载时自动完成初始化，无需在 main.py 显式调用
_configure_root_logger()


# ─────────────────────────────────────────────────────────────────────────────
# 公共接口：get_logger
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 Logger，供各模块统一使用。

    推荐用法（在每个模块文件顶部声明）：
        from app.core.logging import get_logger
        logger = get_logger(__name__)

        # 基础用法
        logger.info("RAG 检索完成", extra={"top_k": 5, "duration_ms": 42.3})

        # 带异常
        try:
            ...
        except Exception as e:
            logger.error("Milvus 连接失败", exc_info=True, extra={"host": host})

    Args:
        name: 通常传入 __name__，日志中显示为模块路径。

    Returns:
        已配置好 Formatter 的 Logger 实例。
    """
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# 装饰器：@log_execution_time
# ─────────────────────────────────────────────────────────────────────────────

# TypeVar 用于保留被装饰函数的类型签名，确保 IDE 类型推导正常工作
F = TypeVar("F", bound=Callable[..., Any])

_deco_logger = get_logger("synaris.perf")


def log_execution_time(
    *,
    level: str = "DEBUG",
    include_args: bool = False,
) -> Callable[[F], F]:
    """记录函数执行耗时的装饰器，同时支持同步和异步函数。

    Args:
        level:        日志级别，默认 DEBUG（性能日志不应污染 INFO）
        include_args: 是否在日志中记录函数参数（敏感场景下关闭）

    用法：
        # 默认（DEBUG 级别，不记录参数）
        @log_execution_time()
        async def query_milvus(query: str) -> list:
            ...

        # 记录参数，用 INFO 级别
        @log_execution_time(level="INFO", include_args=True)
        def process_document(file_path: str) -> dict:
            ...
    """
    log_fn = getattr(_deco_logger, level.lower(), _deco_logger.debug)

    def decorator(func: F) -> F:
        qualname = f"{func.__module__}.{func.__qualname__}"
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                extra = _build_extra(func, qualname, args, kwargs, include_args)
                t0 = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    extra["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    extra["status"] = "ok"
                    log_fn("function executed", extra=extra)
                    return result
                except Exception as exc:
                    extra["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    extra["status"] = "error"
                    extra["error"] = type(exc).__name__
                    _deco_logger.error("function failed", exc_info=True, extra=extra)
                    raise
            return async_wrapper  # type: ignore[return-value]

        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                extra = _build_extra(func, qualname, args, kwargs, include_args)
                t0 = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    extra["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    extra["status"] = "ok"
                    log_fn("function executed", extra=extra)
                    return result
                except Exception as exc:
                    extra["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    extra["status"] = "error"
                    extra["error"] = type(exc).__name__
                    _deco_logger.error("function failed", exc_info=True, extra=extra)
                    raise
            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _build_extra(
    func: Callable,
    qualname: str,
    args: tuple,
    kwargs: dict,
    include_args: bool,
) -> dict[str, Any]:
    """构建 log_execution_time 的 extra 字段。"""
    extra: dict[str, Any] = {"func": qualname}
    if include_args:
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            # 过滤掉 self/cls，值截断为 100 字符防止日志膨胀
            extra["args"] = {
                k: str(v)[:100]
                for k, v in bound.arguments.items()
                if k not in ("self", "cls")
            }
        except Exception:
            pass  # 参数绑定失败时静默忽略，不影响业务逻辑
    return extra