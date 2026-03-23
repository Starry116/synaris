"""
@File       : health.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 健康检查接口，供 Docker healthcheck、Kubernetes liveness/readiness probe 使用。
@Features:
  - GET /health
      轻量存活探针（Liveness Probe）
      仅返回服务进程状态，不检查外部依赖，响应 < 5ms
  - GET /health/detailed
      就绪探针（Readiness Probe）
      asyncio.gather 并发检查 Redis / Milvus / PostgreSQL 连通性
      各组件独立超时（3s），单组件失败不影响其他组件结果
      整体 status 由最差分量决定（degraded / unhealthy 优先级高于 ok）
  - 响应统一复用 ApiResponse[T] 结构
  - 不挂载在 API_PREFIX 下，保持 /health 路径简洁（运维习惯）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import time
from enum import str as StrEnum
from typing import Dict, Optional

from fastapi import APIRouter
from pydantic import Field

from app.core.logging import get_logger
from app.schemas.base import ApiResponse, SynarisBaseModel

logger = get_logger(__name__)

router = APIRouter(tags=["健康检查"])


# ---------------------------------------------------------------------------
# 响应 Schema
# ---------------------------------------------------------------------------


class ComponentStatus(StrEnum, str):
    """单个组件的健康状态枚举。"""

    OK = "ok"  # 连接正常，响应及时
    DEGRADED = "degraded"  # 连接成功但响应较慢（超过阈值）
    UNHEALTHY = "unhealthy"  # 连接失败或超时


class ComponentDetail(SynarisBaseModel):
    """单个依赖组件的检查结果。"""

    status: str = Field(..., description="ok / degraded / unhealthy")
    latency_ms: Optional[float] = Field(None, description="探测耗时（毫秒）")
    message: Optional[str] = Field(None, description="异常说明（正常时为 null）")


class HealthData(SynarisBaseModel):
    """GET /health 响应的 data 载荷。"""

    status: str = Field("ok", description="服务状态：ok")
    timestamp: float = Field(default_factory=time.time)
    uptime_seconds: float = Field(..., description="进程已运行秒数")
    version: str = Field(..., description="应用版本号")


class DetailedHealthData(SynarisBaseModel):
    """GET /health/detailed 响应的 data 载荷。"""

    status: str = Field(..., description="整体状态：ok / degraded / unhealthy")
    timestamp: float = Field(default_factory=time.time)
    version: str = Field(..., description="应用版本号")
    components: Dict[str, ComponentDetail] = Field(..., description="各组件探测结果")


# 记录进程启动时间（模块首次导入时）
_PROCESS_START_TIME: float = time.time()

# 单组件探测超时（秒）
_COMPONENT_TIMEOUT: float = 3.0

# 响应延迟阈值：超过此值视为 degraded（毫秒）
_DEGRADED_LATENCY_MS: float = 200.0


# ---------------------------------------------------------------------------
# GET /health  — 轻量存活探针
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=ApiResponse[HealthData],
    summary="存活探针（Liveness Probe）",
    description=(
        "轻量级健康检查，仅确认服务进程存活。\n\n"
        "适用于 Docker HEALTHCHECK 和 K8s liveness probe，响应通常 < 5ms。\n"
        "**不检查外部依赖**（Redis / Milvus / DB），保证即使依赖不可用此接口也能返回。"
    ),
)
async def health_check() -> ApiResponse[HealthData]:
    from app.config.settings import get_settings

    _settings = get_settings()

    uptime = time.time() - _PROCESS_START_TIME
    return ApiResponse.ok(
        data=HealthData(
            status="ok",
            uptime_seconds=round(uptime, 2),
            version=_settings.app_version,
        ),
        message="服务运行正常",
    )


# ---------------------------------------------------------------------------
# GET /health/detailed  — 就绪探针（含组件并发检查）
# ---------------------------------------------------------------------------


@router.get(
    "/health/detailed",
    response_model=ApiResponse[DetailedHealthData],
    summary="就绪探针（Readiness Probe）",
    description=(
        "并发检查所有外部依赖的连通性。\n\n"
        "- **Redis**：执行 PING 命令\n"
        "- **Milvus**：检查连接与 Collection 状态\n"
        "- **PostgreSQL**：执行 SELECT 1（Step 21 完成后启用）\n\n"
        "各组件独立超时 3s，单组件失败不影响其他检查结果。\n"
        "整体 `status` 由最差分量决定：`unhealthy > degraded > ok`。"
    ),
)
async def health_check_detailed() -> ApiResponse[DetailedHealthData]:
    from app.config.settings import get_settings

    _settings = get_settings()

    # 并发探测所有组件
    redis_result, milvus_result, postgres_result = await asyncio.gather(
        _check_redis(),
        _check_milvus(),
        _check_postgres(),
        return_exceptions=False,  # 每个探测函数内部已 catch 所有异常
    )

    components: Dict[str, ComponentDetail] = {
        "redis": redis_result,
        "milvus": milvus_result,
        "postgresql": postgres_result,
    }

    # 整体状态：取最差分量
    overall = _aggregate_status(components)

    logger.info(
        "健康检查完成",
        extra={
            "overall": overall,
            "redis": redis_result.status,
            "milvus": milvus_result.status,
            "postgresql": postgres_result.status,
        },
    )

    return ApiResponse.ok(
        data=DetailedHealthData(
            status=overall,
            version=_settings.app_version,
            components=components,
        ),
        message="健康检查完成" if overall == "ok" else f"服务部分异常（{overall}）",
    )


# ---------------------------------------------------------------------------
# 组件探测函数（各自独立，不向上抛异常）
# ---------------------------------------------------------------------------


async def _check_redis() -> ComponentDetail:
    """
    探测 Redis 连通性。
    通过 redis_client.ping() 发送 PING 命令，记录往返耗时。
    """
    start = time.perf_counter()
    try:
        from app.infrastructure.redis_client import ping as redis_ping

        ok = await asyncio.wait_for(redis_ping(), timeout=_COMPONENT_TIMEOUT)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not ok:
            return ComponentDetail(
                status=ComponentStatus.UNHEALTHY,
                latency_ms=round(elapsed_ms, 2),
                message="PING 返回 False，连接异常",
            )

        status = (
            ComponentStatus.DEGRADED
            if elapsed_ms > _DEGRADED_LATENCY_MS
            else ComponentStatus.OK
        )
        return ComponentDetail(
            status=status,
            latency_ms=round(elapsed_ms, 2),
            message="响应偏慢" if status == ComponentStatus.DEGRADED else None,
        )

    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("Redis 健康检查超时", extra={"elapsed_ms": elapsed_ms})
        return ComponentDetail(
            status=ComponentStatus.UNHEALTHY,
            latency_ms=round(_COMPONENT_TIMEOUT * 1000, 2),
            message=f"探测超时（>{_COMPONENT_TIMEOUT}s）",
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("Redis 健康检查异常", extra={"error": str(exc)})
        return ComponentDetail(
            status=ComponentStatus.UNHEALTHY,
            latency_ms=round(elapsed_ms, 2),
            message=str(exc),
        )


async def _check_milvus() -> ComponentDetail:
    """
    探测 Milvus 连通性。
    调用 milvus_client.ping()，检查连接与 Collection 加载状态。
    """
    start = time.perf_counter()
    try:
        from app.infrastructure.milvus_client import get_milvus_client

        client = get_milvus_client()
        ok = await asyncio.wait_for(client.ping(), timeout=_COMPONENT_TIMEOUT)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not ok:
            return ComponentDetail(
                status=ComponentStatus.UNHEALTHY,
                latency_ms=round(elapsed_ms, 2),
                message="Milvus 连接检查返回 False",
            )

        status = (
            ComponentStatus.DEGRADED
            if elapsed_ms > _DEGRADED_LATENCY_MS
            else ComponentStatus.OK
        )
        return ComponentDetail(
            status=status,
            latency_ms=round(elapsed_ms, 2),
            message="响应偏慢" if status == ComponentStatus.DEGRADED else None,
        )

    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("Milvus 健康检查超时", extra={"elapsed_ms": elapsed_ms})
        return ComponentDetail(
            status=ComponentStatus.UNHEALTHY,
            latency_ms=round(_COMPONENT_TIMEOUT * 1000, 2),
            message=f"探测超时（>{_COMPONENT_TIMEOUT}s）",
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("Milvus 健康检查异常", extra={"error": str(exc)})
        return ComponentDetail(
            status=ComponentStatus.UNHEALTHY,
            latency_ms=round(elapsed_ms, 2),
            message=str(exc),
        )


async def _check_postgres() -> ComponentDetail:
    """
    探测 PostgreSQL 连通性（Step 21 完成后启用实际检查）。

    当前阶段（Step 5）：PostgreSQL 客户端尚未实现，
    返回 degraded + 说明，不影响整体健康检查结构。
    Step 21 完成后，取消注释下方真实检查逻辑。
    """
    # TODO(Step 21): 启用真实 PostgreSQL 检查
    # start = time.perf_counter()
    # try:
    #     from app.infrastructure.postgres_client import check_db_health
    #     ok = await asyncio.wait_for(check_db_health(), timeout=_COMPONENT_TIMEOUT)
    #     elapsed_ms = (time.perf_counter() - start) * 1000
    #     status = ComponentStatus.DEGRADED if elapsed_ms > _DEGRADED_LATENCY_MS else ComponentStatus.OK
    #     return ComponentDetail(status=status, latency_ms=round(elapsed_ms, 2))
    # except Exception as exc:
    #     return ComponentDetail(status=ComponentStatus.UNHEALTHY, message=str(exc))

    return ComponentDetail(
        status=ComponentStatus.DEGRADED,
        latency_ms=None,
        message="PostgreSQL 客户端尚未初始化（Step 21 后启用）",
    )


# ---------------------------------------------------------------------------
# 工具函数：聚合整体状态
# ---------------------------------------------------------------------------

_STATUS_PRIORITY: Dict[str, int] = {
    ComponentStatus.UNHEALTHY: 2,
    ComponentStatus.DEGRADED: 1,
    ComponentStatus.OK: 0,
}


def _aggregate_status(components: Dict[str, ComponentDetail]) -> str:
    """
    从所有组件状态中取最差值作为整体 status。
    优先级：unhealthy > degraded > ok
    """
    worst_priority = 0
    worst_status = ComponentStatus.OK

    for detail in components.values():
        priority = _STATUS_PRIORITY.get(detail.status, 0)
        if priority > worst_priority:
            worst_priority = priority
            worst_status = detail.status

    return worst_status


"""

## 设计说明

### 中间件注册顺序

Starlette 的中间件是**洋葱模型**，`add_middleware` 越晚注册越靠近内层，请求经过顺序与注册顺序相反：

客户端请求
  ↓
SlowAPIMiddleware（速率限制，最外层拦截）
  ↓
CORSMiddleware（跨域预检，早于业务逻辑处理）
  ↓
TraceIDMiddleware（注入 TraceID，后续中间件可读取）
  ↓
RequestLoggingMiddleware（此时 trace_id 已就绪，日志完整）
  ↓
路由处理函数
  ↑
（响应沿反方向返回，各中间件可修改响应头）


### `/health/detailed` 并发检查设计

asyncio.gather(
  _check_redis(),      ← 最多等 3s
  _check_milvus(),     ← 最多等 3s  （并发，不串行）
  _check_postgres(),   ← 当前返回 degraded（Step 21 后激活）
)
整体耗时 ≈ max(各组件耗时) ≤ 3s

"""
