"""
@File       : main.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: FastAPI 应用入口，负责实例创建、中间件注册、路由挂载与生命周期管理。
@Features:
  - FastAPI 实例配置：标题 / 版本 / OpenAPI 路径 / lifespan
  - 中间件注册顺序（外层先执行）：
      CORS → TraceID 注入 → 请求日志 → slowapi 速率限制
  - lifespan：启动时初始化 Redis / Milvus 连接，关闭时优雅释放资源
  - 路由挂载：/health / /chat / /knowledge / /rag / /agent
  - slowapi 速率限制：默认 60次/分钟/IP（可按用户配置覆盖）
  - 全局异常处理器：AppException → ApiResponse / 422 / 500 兜底

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
    2026-04-15  Starry  完成TODO列表中的所有功能，解除占位钩子注释
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# slowapi 速率限制器（全局单例，路由层通过 Depends 引用）
# ---------------------------------------------------------------------------

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"],
    # 超限时返回标准 ApiResponse 格式（见 _rate_limit_handler）
    headers_enabled=True,         # 响应头携带 X-RateLimit-* 信息
    swallow_errors=False,
)


# ---------------------------------------------------------------------------
# 中间件：TraceID 注入
# ---------------------------------------------------------------------------

class TraceIDMiddleware(BaseHTTPMiddleware):
    """
    为每个请求生成唯一 TraceID，注入到：
      - request.state.trace_id        → 供路由层、服务层读取
      - contextvars（logging 模块）    → 结构化日志自动携带
      - 响应头 X-Trace-ID             → 供客户端关联日志

    TraceID 优先使用客户端传入的 X-Trace-ID 头（方便跨服务追踪），
    若未传则自动生成 UUID4。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id: str = (
            request.headers.get("X-Trace-ID") or uuid.uuid4().hex
        )
        request.state.trace_id = trace_id

        # 注入到 contextvars，使结构化日志的 trace_id 字段自动填充
        from app.core.logging import set_trace_id
        set_trace_id(trace_id)

        response: Response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response


# ---------------------------------------------------------------------------
# 中间件：请求日志
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    记录每个请求的进入与完成日志，包含：
      - method / path / client_ip
      - 响应状态码 / 耗时（毫秒）

    健康检查路径（/health）仅在 DEBUG 级别记录，避免日志噪音。
    """

    SILENT_PATHS = {"/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        client_ip = get_remote_address(request)
        trace_id = getattr(request.state, "trace_id", "-")

        is_silent = path in self.SILENT_PATHS

        if not is_silent:
            logger.info(
                "→ 请求进入",
                extra={
                    "method": method,
                    "path": path,
                    "client_ip": client_ip,
                    "trace_id": trace_id,
                },
            )

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        log_fn = logger.debug if is_silent else logger.info
        log_fn(
            "← 请求完成",
            extra={
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
                "trace_id": trace_id,
            },
        )
        return response


# ---------------------------------------------------------------------------
# lifespan：启动 / 关闭生命周期
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    生命周期管理。

    启动阶段（yield 之前）：
      1. 记录启动日志
      2. 初始化 Redis 连接池并执行 PING 健康检查
      3. 初始化 Milvus 连接并检查 Collection 状态
      4. （后续步骤）初始化 PostgreSQL 连接池

    关闭阶段（yield 之后）：
      1. 关闭 Redis 连接池
      2. 释放 Milvus 连接
    """
    # ── 启动 ────────────────────────────────────────────────────────────
    logger.info(
        "Synaris 启动ing...",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
        },
    )

    # Redis 初始化
    try:
        from app.infrastructure.redis_client import get_client, ping as redis_ping
        _client = get_client()  # 触发连接池初始化
        redis_ok = await redis_ping()
        if redis_ok:
            logger.info("Redis 连接OK")
        else:
            logger.warning("Redis PING 失败，服务降级运行")
    except Exception as exc:
        logger.error("Redis 初始化异常", extra={"error": str(exc)})

    # Milvus 初始化
    try:
        from app.infrastructure.milvus_client import get_milvus_client
        milvus_client = get_milvus_client()
        milvus_ok = await milvus_client.ping()
        if milvus_ok:
            logger.info("Milvus 连接OK")
        else:
            logger.warning("Milvus 连接失败，RAG 功能暂不可用")
    except Exception as exc:
        logger.error("Milvus 初始化异常", extra={"error": str(exc)})

    # 初始化 PostgreSQL 连接池
    from app.infrastructure.postgres_client import init_db
    await init_db()
    logger.info("PostgreSQL 连接OK")

    logger.info(
        "Synaris 启动OK，服务已就绪",
        extra={"api_prefix": settings.api_prefix},
    )

    yield  # ← 应用正常运行期间挂起于此（生命周期分割点：startup → runtime → shutdown）

    # ── 关闭 ────────────────────────────────────────────────────────────
    logger.info("Synaris 关闭ing，正在释放资源...")

    try:
        from app.infrastructure.redis_client import close_pool
        await close_pool()
        logger.info("Redis 连接池已关闭 ✓")
    except Exception as exc:
        logger.error("Redis 关闭异常", extra={"error": str(exc)})

    try:
        from app.infrastructure.milvus_client import get_milvus_client
        await get_milvus_client().close()
        logger.info("Milvus 连接已释放 ✓")
    except Exception as exc:
        logger.error("Milvus 关闭异常", extra={"error": str(exc)})

    logger.info("Synaris 已安全关闭")


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    FastAPI 应用工厂函数。
    """
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Enterprise RAG & Multi-Agent AI Platform\n\n"
            "基于 FastAPI + LangChain + LangGraph + Milvus 构建的企业级智能协作平台。"
        ),
        version=settings.app_version,
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        lifespan=lifespan,
    )

    # ── 1. 注册全局异常处理器（最先注册，确保覆盖所有后续中间件抛出的异常）
    register_exception_handlers(app)

    # ── 2. slowapi 速率限制（RateLimitExceeded → 标准 429 响应）
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        _build_rate_limit_handler(),
    )
    app.add_middleware(SlowAPIMiddleware)

    # ── 3. CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    # ── 4. TraceID 注入（必须在请求日志之前，日志才能读到 trace_id）
    app.add_middleware(TraceIDMiddleware)

    # ── 5. 请求日志
    app.add_middleware(RequestLoggingMiddleware)

    # 挂载 Auth 中间件
    from app.core.auth import AuthMiddleware
    app.add_middleware(AuthMiddleware)

    # ── 路由挂载 ──────────────────────────────────────────────────────
    _mount_routers(app)

    # 注册中间件
    from app.core.observability import MetricsMiddleware
    app.add_middleware(MetricsMiddleware)
    
    # 挂载 /metrics 端点
    from app.core.observability import metrics_router
    app.include_router(metrics_router)

    # 在 lifespan 启动阶段
    log_registered_metrics()
    
    return app


def _mount_routers(app: FastAPI) -> None:
    """
    集中挂载所有路由模块，每个模块独立 prefix + tags。
    路由模块在导入时才实例化，避免启动时因依赖缺失报错。
    """
    from app.api.health import router as health_router
    app.include_router(health_router)                  # /health（无 prefix）

    # AI聊天
    from app.api.chat import router as chat_router
    app.include_router(chat_router, prefix=settings.api_prefix)

    # RAG 相关：知识库管理 + 向量搜索
    from app.api.knowledge import router as knowledge_router
    from app.api.rag import router as rag_router
    app.include_router(knowledge_router, prefix=settings.api_prefix)
    app.include_router(rag_router, prefix=settings.api_prefix)

    # 智能体相关
    from app.api.agent import router as agent_router
    app.include_router(agent_router, prefix=settings.api_prefix)


def _build_rate_limit_handler():
    """
    将 slowapi 的 RateLimitExceeded 转换为 Synaris 标准 ApiResponse 格式。
    """
    import time as _time
    from fastapi.responses import JSONResponse
    from app.core.exceptions import ErrorCode

    async def handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", "")
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "code": int(ErrorCode.RATE_LIMIT_EXCEEDED),
                "message": f"请求过于频繁，限制：{exc.detail}，请稍后重试",
                "data": None,
                "trace_id": trace_id,
                "timestamp": _time.time(),
            },
            headers={
                "Retry-After": "60",
                "X-Trace-ID": trace_id,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# 应用实例（Uvicorn / Gunicorn 入口点）
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# 本地开发直接运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=False,   # 由 RequestLoggingMiddleware 统一记录，禁用 uvicorn 原生日志
    )