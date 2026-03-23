"""
@File       : exceptions.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 全局自定义异常体系与业务错误码枚举。
@Features:
  - AppException 基类携带 http_status / error_code / message / detail 四要素
  - 业务子类：LLMError / CacheError / CacheConnectionError / VectorDBError /
              DocumentParseError / AgentError / AuthError / RateLimitError /
              ValidationError / NotFoundError / ToolExecutionError
  - ErrorCode 枚举：业务码 5位整数，前两位对应 HTTP 状态码语义
      - 4xxxx → 客户端错误
      - 5xxxx → 服务端 / 第三方依赖错误
  - FastAPI 全局异常处理器注册辅助函数 register_exception_handlers()

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# ErrorCode 枚举
# ---------------------------------------------------------------------------

class ErrorCode(IntEnum):
    """
    业务错误码规范
    ─────────────────────────────────────────────────────
    格式：5位整数
      • 前两位  → HTTP 状态码语义（40=4xx 客户端，50=5xx 服务端）
      • 后三位  → 具体错误序号

    示例：
      40001 → 通用参数校验失败（HTTP 400）
      40101 → 认证失败（HTTP 401）
      40301 → 权限不足（HTTP 403）
      40401 → 资源未找到（HTTP 404）
      42901 → 请求频率超限（HTTP 429）
      50001 → LLM 服务不可用（HTTP 502）
      50002 → LLM 响应超时（HTTP 504）
    ─────────────────────────────────────────────────────
    """

    # ── 通用客户端错误 (400) ──────────────────────────────
    VALIDATION_ERROR        = 40001   # 请求参数校验失败
    INVALID_REQUEST         = 40002   # 非法请求格式

    # ── 认证错误 (401) ───────────────────────────────────
    AUTH_FAILED             = 40101   # JWT / API Key 认证失败
    TOKEN_EXPIRED           = 40102   # JWT Token 已过期
    TOKEN_INVALID           = 40103   # JWT Token 格式非法
    API_KEY_INVALID         = 40104   # API Key 不存在或已禁用
    API_KEY_EXPIRED         = 40105   # API Key 已过期

    # ── 权限错误 (403) ───────────────────────────────────
    PERMISSION_DENIED       = 40301   # 角色权限不足
    TOOL_ACCESS_DENIED      = 40302   # 工具调用无权限

    # ── 资源未找到 (404) ─────────────────────────────────
    RESOURCE_NOT_FOUND      = 40401   # 通用资源未找到
    SESSION_NOT_FOUND       = 40402   # 会话 ID 不存在
    TASK_NOT_FOUND          = 40403   # 任务 ID 不存在
    DOCUMENT_NOT_FOUND      = 40404   # 文档不存在
    PROMPT_VERSION_NOT_FOUND= 40405   # Prompt 版本不存在

    # ── 请求频率超限 (429) ───────────────────────────────
    RATE_LIMIT_EXCEEDED     = 42901   # 请求频率超限

    # ── LLM 相关错误 (502 / 504) ─────────────────────────
    LLM_UNAVAILABLE         = 50001   # 主模型与降级模型均不可用
    LLM_TIMEOUT             = 50002   # LLM 响应超时
    LLM_QUOTA_EXCEEDED      = 50003   # OpenAI 配额耗尽
    LLM_INVALID_RESPONSE    = 50004   # LLM 返回格式无法解析
    LLM_CONTEXT_TOO_LONG    = 50005   # 输入超出模型最大 context

    # ── 缓存相关错误 (503) ───────────────────────────────
    CACHE_CONNECTION_ERROR  = 50301   # Redis 连接失败
    CACHE_OPERATION_ERROR   = 50302   # Redis 读写操作失败
    CACHE_SERIALIZATION_ERR = 50303   # JSON 序列化 / 反序列化失败

    # ── 向量数据库错误 (503) ─────────────────────────────
    VECTOR_DB_CONNECTION    = 50311   # Milvus 连接失败
    VECTOR_DB_INSERT_FAILED = 50312   # 向量写入失败
    VECTOR_DB_QUERY_FAILED  = 50313   # 向量检索失败
    VECTOR_DB_SCHEMA_ERROR  = 50314   # Collection Schema 异常

    # ── 文档处理错误 (422) ───────────────────────────────
    DOCUMENT_PARSE_FAILED   = 42201   # 文档解析失败（格式损坏等）
    DOCUMENT_TOO_LARGE      = 42202   # 文档超出大小限制
    DOCUMENT_FORMAT_UNSUP   = 42203   # 不支持的文档格式
    DOCUMENT_EMPTY          = 42204   # 文档内容为空

    # ── Agent 相关错误 (500) ─────────────────────────────
    AGENT_EXECUTION_FAILED  = 50021   # Agent 执行异常
    AGENT_MAX_ITER_EXCEEDED = 50022   # 达到最大迭代次数
    AGENT_TASK_CANCELLED    = 50023   # 任务被用户取消
    AGENT_STATE_CORRUPT     = 50024   # AgentState 状态异常
    AGENT_TOOL_EXEC_FAILED  = 50025   # 工具调用执行失败

    # ── 内部服务错误 (500) ───────────────────────────────
    INTERNAL_ERROR          = 50000   # 未分类内部错误
    DATABASE_ERROR          = 50041   # PostgreSQL 操作失败
    TASK_QUEUE_ERROR        = 50051   # Celery 任务队列异常


# ---------------------------------------------------------------------------
# HTTP 状态码映射表
# ---------------------------------------------------------------------------

_ERROR_CODE_TO_HTTP_STATUS: dict[ErrorCode, int] = {
    # 400
    ErrorCode.VALIDATION_ERROR:         400,
    ErrorCode.INVALID_REQUEST:          400,
    # 401
    ErrorCode.AUTH_FAILED:              401,
    ErrorCode.TOKEN_EXPIRED:            401,
    ErrorCode.TOKEN_INVALID:            401,
    ErrorCode.API_KEY_INVALID:          401,
    ErrorCode.API_KEY_EXPIRED:          401,
    # 403
    ErrorCode.PERMISSION_DENIED:        403,
    ErrorCode.TOOL_ACCESS_DENIED:       403,
    # 404
    ErrorCode.RESOURCE_NOT_FOUND:       404,
    ErrorCode.SESSION_NOT_FOUND:        404,
    ErrorCode.TASK_NOT_FOUND:           404,
    ErrorCode.DOCUMENT_NOT_FOUND:       404,
    ErrorCode.PROMPT_VERSION_NOT_FOUND: 404,
    # 422
    ErrorCode.DOCUMENT_PARSE_FAILED:    422,
    ErrorCode.DOCUMENT_TOO_LARGE:       422,
    ErrorCode.DOCUMENT_FORMAT_UNSUP:    422,
    ErrorCode.DOCUMENT_EMPTY:           422,
    # 429
    ErrorCode.RATE_LIMIT_EXCEEDED:      429,
    # 500
    ErrorCode.INTERNAL_ERROR:           500,
    ErrorCode.AGENT_EXECUTION_FAILED:   500,
    ErrorCode.AGENT_MAX_ITER_EXCEEDED:  500,
    ErrorCode.AGENT_TASK_CANCELLED:     500,
    ErrorCode.AGENT_STATE_CORRUPT:      500,
    ErrorCode.AGENT_TOOL_EXEC_FAILED:   500,
    ErrorCode.DATABASE_ERROR:           500,
    ErrorCode.TASK_QUEUE_ERROR:         500,
    # 502 / 504
    ErrorCode.LLM_UNAVAILABLE:          502,
    ErrorCode.LLM_TIMEOUT:              504,
    ErrorCode.LLM_QUOTA_EXCEEDED:       502,
    ErrorCode.LLM_INVALID_RESPONSE:     502,
    ErrorCode.LLM_CONTEXT_TOO_LONG:     400,
    # 503
    ErrorCode.CACHE_CONNECTION_ERROR:   503,
    ErrorCode.CACHE_OPERATION_ERROR:    503,
    ErrorCode.CACHE_SERIALIZATION_ERR:  500,
    ErrorCode.VECTOR_DB_CONNECTION:     503,
    ErrorCode.VECTOR_DB_INSERT_FAILED:  503,
    ErrorCode.VECTOR_DB_QUERY_FAILED:   503,
    ErrorCode.VECTOR_DB_SCHEMA_ERROR:   500,
}


def error_code_to_http_status(code: ErrorCode) -> int:
    """根据业务错误码返回对应 HTTP 状态码，未配置时兜底返回 500。"""
    return _ERROR_CODE_TO_HTTP_STATUS.get(code, 500)


# ---------------------------------------------------------------------------
# 异常基类
# ---------------------------------------------------------------------------

class AppException(Exception):
    """
    Synaris 业务异常基类。

    所有业务异常均继承此类，携带以下信息：
      - error_code   : ErrorCode 枚举值（业务语义）
      - message      : 面向用户的简短描述（可直接展示）
      - detail       : 调试细节，如原始异常信息、上下文数据（不对外暴露）
      - http_status  : 自动从 error_code 推导，也可手动指定覆盖

    示例：
        raise LLMError(
            error_code=ErrorCode.LLM_UNAVAILABLE,
            message="AI 服务暂时不可用，请稍后重试",
            detail={"model": "gpt-4o", "attempt": 3},
        )
    """

    def __init__(
        self,
        message: str = "服务异常，请稍后重试",
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        detail: Optional[Any] = None,
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.detail = detail
        self.http_status: int = (
            http_status
            if http_status is not None
            else error_code_to_http_status(error_code)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code.name}, "
            f"http_status={self.http_status}, "
            f"message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# 业务异常子类
# ---------------------------------------------------------------------------

class LLMError(AppException):
    """
    LLM 相关错误：模型不可用、响应超时、配额耗尽、返回格式异常等。

    示例：
        raise LLMError(
            message="GPT-4o 响应超时，已自动降级至 gpt-4o-mini",
            error_code=ErrorCode.LLM_TIMEOUT,
            detail={"model": "gpt-4o", "timeout_seconds": 30},
        )
    """
    def __init__(
        self,
        message: str = "LLM 服务异常",
        error_code: ErrorCode = ErrorCode.LLM_UNAVAILABLE,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, detail=detail)


class CacheError(AppException):
    """
    Redis 缓存操作错误：读写失败、序列化异常等。
    连接级别错误请使用更具体的 CacheConnectionError。

    示例：
        raise CacheError(
            message="缓存写入失败",
            error_code=ErrorCode.CACHE_OPERATION_ERROR,
            detail={"key": "synaris:session:abc"},
        )
    """
    def __init__(
        self,
        message: str = "缓存操作失败",
        error_code: ErrorCode = ErrorCode.CACHE_OPERATION_ERROR,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, detail=detail)


class CacheConnectionError(CacheError):
    """
    Redis 连接失败（网络不通、认证失败、连接池耗尽等）。
    继承 CacheError 以便统一捕获，同时支持单独区分连接级错误。
    """
    def __init__(
        self,
        message: str = "Redis 连接失败",
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.CACHE_CONNECTION_ERROR,
            detail=detail,
        )


class VectorDBError(AppException):
    """
    Milvus 向量数据库错误：连接失败、写入失败、检索异常、Schema 不一致等。

    示例：
        raise VectorDBError(
            message="向量检索失败，Collection 可能未就绪",
            error_code=ErrorCode.VECTOR_DB_QUERY_FAILED,
            detail={"collection": "documents", "top_k": 5},
        )
    """
    def __init__(
        self,
        message: str = "向量数据库操作失败",
        error_code: ErrorCode = ErrorCode.VECTOR_DB_QUERY_FAILED,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, detail=detail)


class DocumentParseError(AppException):
    """
    文档解析处理错误：格式不支持、内容损坏、文件过大、内容为空等。

    示例：
        raise DocumentParseError(
            message="PDF 文件已损坏，无法提取文本",
            error_code=ErrorCode.DOCUMENT_PARSE_FAILED,
            detail={"filename": "report.pdf", "size_mb": 12.5},
        )
    """
    def __init__(
        self,
        message: str = "文档解析失败",
        error_code: ErrorCode = ErrorCode.DOCUMENT_PARSE_FAILED,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, detail=detail)


class AgentError(AppException):
    """
    Agent 运行时错误：执行异常、超出最大迭代、状态损坏、工具调用失败等。

    示例：
        raise AgentError(
            message="Agent 已达最大迭代次数（10次），任务终止",
            error_code=ErrorCode.AGENT_MAX_ITER_EXCEEDED,
            detail={"task_id": "task-xyz", "iterations": 10},
        )
    """
    def __init__(
        self,
        message: str = "Agent 执行失败",
        error_code: ErrorCode = ErrorCode.AGENT_EXECUTION_FAILED,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, detail=detail)


class ToolExecutionError(AgentError):
    """
    工具调用执行失败（继承 AgentError，便于统一捕获 Agent 层所有错误）。
    """
    def __init__(
        self,
        message: str = "工具执行失败",
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_TOOL_EXEC_FAILED,
            detail=detail,
        )


class AuthError(AppException):
    """
    认证 / 授权错误：Token 失效、API Key 非法、角色权限不足等。
    """
    def __init__(
        self,
        message: str = "身份认证失败",
        error_code: ErrorCode = ErrorCode.AUTH_FAILED,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, detail=detail)


class RateLimitError(AppException):
    """
    请求频率超限（对应 HTTP 429）。
    """
    def __init__(
        self,
        message: str = "请求过于频繁，请稍后重试",
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            detail=detail,
        )


class NotFoundError(AppException):
    """
    资源未找到（对应 HTTP 404）。
    """
    def __init__(
        self,
        message: str = "请求的资源不存在",
        error_code: ErrorCode = ErrorCode.RESOURCE_NOT_FOUND,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, detail=detail)


class DatabaseError(AppException):
    """
    PostgreSQL 数据库操作失败。
    """
    def __init__(
        self,
        message: str = "数据库操作失败",
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# FastAPI 全局异常处理器
# ---------------------------------------------------------------------------

def _build_error_response(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    """
    将 AppException 转换为标准 ApiResponse JSON 响应。
    trace_id 从请求 state 读取（由 TraceID 中间件注入）。
    """
    import time
    from contextvars import copy_context

    trace_id: str = getattr(request.state, "trace_id", "unknown")

    return JSONResponse(
        status_code=exc.http_status,
        content={
            "success": False,
            "code": int(exc.error_code),
            "message": exc.message,
            "data": None,
            "trace_id": trace_id,
            "timestamp": time.time(),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    向 FastAPI 应用注册全局异常处理器。
    在 main.py 的 create_app() 中调用一次即可。

    覆盖范围：
      - AppException 及所有子类  →  标准 ApiResponse 错误格式
      - fastapi.RequestValidationError  →  400 参数校验失败
      - 未捕获 Exception  →  500 内部错误（隐藏 detail）
    """
    from fastapi.exceptions import RequestValidationError
    import time

    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request, exc: AppException
    ) -> JSONResponse:
        # 5xx 错误额外记录日志，方便排查
        if exc.http_status >= 500:
            from app.core.logging import get_logger as _get_logger
            _logger = _get_logger("synaris.exception")
            _logger.error(
                "业务异常",
                extra={
                    "error_code": exc.error_code.name,
                    "http_status": exc.http_status,
                    "message": exc.message,
                    "detail": exc.detail,
                    "path": request.url.path,
                },
            )
        return _build_error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", "unknown")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "code": int(ErrorCode.VALIDATION_ERROR),
                "message": "请求参数校验失败",
                "data": exc.errors(),   # 详细字段错误列表
                "trace_id": trace_id,
                "timestamp": time.time(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        from app.core.logging import get_logger as _get_logger
        _logger = _get_logger("synaris.exception")
        _logger.exception(
            "未捕获异常",
            extra={"path": request.url.path, "error": str(exc)},
        )
        trace_id = getattr(request.state, "trace_id", "unknown")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "code": int(ErrorCode.INTERNAL_ERROR),
                "message": "服务内部错误，请联系管理员",
                "data": None,
                "trace_id": trace_id,
                "timestamp": time.time(),
            },
        )