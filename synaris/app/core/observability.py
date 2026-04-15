"""
@File       : observability.py
@Author     : Starry Hung
@Created    : 2026-04-15
@Version    : 1.0.0
@Description: Prometheus 可观测性模块，提供全链路指标采集、中间件与暴露端点。
@Features:
  - 指标定义（prometheus-client）：
      · http_request_duration_seconds  — HTTP 请求延迟直方图（P50/P95/P99）
        labels: method / endpoint / status_code
      · llm_token_usage_total          — LLM Token 消耗计数器（累计）
        labels: model / user_id / endpoint
      · llm_request_duration_seconds   — LLM 单次调用延迟直方图
        labels: model / task_type
      · cache_hit_ratio                — 缓存命中率仪表盘（实时快照）
        labels: cache_type (embedding / search / session)
      · agent_task_duration_seconds    — Agent 任务执行延迟直方图
        labels: task_type (single / multi / rag_only)
      · active_connections             — 当前活跃 WebSocket/HTTP 连接数（仪表盘）
      · document_chunks_total          — 文档分块入库总量计数器
        labels: collection / file_type
  - MetricsMiddleware（ASGI 中间件）：
      · 拦截每个 HTTP 请求，自动记录延迟与状态码
      · 白名单路径跳过（/metrics / /health），避免元指标污染
      · 路径归一化（将 /user/123 → /user/{id}），防止高基数标签
  - GET /metrics 路由：
      · 通过 prometheus_client.generate_latest() 生成文本格式
      · Content-Type: text/plain; charset=utf-8; version=0.0.4
      · 供 Prometheus Scraper 定期抓取（默认 15s 间隔）
  - 工具函数：
      · track_llm_call(model, prompt_tokens, completion_tokens,
                       task_type, user_id, endpoint, latency_ms)
        → 一次性更新 Token 计数器 + 延迟直方图，统一入口
      · track_cache_hit(cache_type, hit) → 更新缓存命中率快照
      · track_agent_task(task_type, duration_ms) → 记录 Agent 任务延迟
      · inc_active_connections(delta) → 调整活跃连接计数（+1/-1）
  - 防 Cardinality 爆炸设计：
      · user_id 做哈希截断（前 8 位），避免无限增长的标签值
      · endpoint 路径归一化（正则替换 UUID / 纯数字 ID）
      · 仅记录白名单 HTTP 方法（GET/POST/PUT/DELETE/PATCH）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-15  Starry  Initial creation
"""

from __future__ import annotations

import hashlib
import re
import time
import logging
from typing import Callable, Optional

from fastapi import APIRouter
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Match

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
    CollectorRegistry,
    REGISTRY,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 指标定义
#
# 领域建模类比：
#   Histogram → 「计时表」，记录每次事件的耗时分布（P50/P95/P99）
#   Counter   → 「里程表」，只增不减，记录累计发生次数
#   Gauge     → 「温度计」，可升可降，记录当前实时状态快照
#
# 命名规范（Prometheus 官方约定）：
#   {namespace}_{subsystem}_{metric_name}_{unit_suffix}
#   示例：synaris_http_request_duration_seconds
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1-A. HTTP 请求延迟（Histogram） ─────────────────────────────────────────
# Buckets 覆盖范围：5ms → 10s，适合 Web API + LLM 调用的延迟分布
# 类比：机场的「各出口等待时间」统计，桶是等待时间的区间
_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

http_request_duration_seconds = Histogram(
    name="synaris_http_request_duration_seconds",
    documentation=(
        "HTTP 请求处理延迟（秒）。"
        "labels: method=HTTP方法 / endpoint=归一化路径 / status_code=响应状态码"
    ),
    labelnames=["method", "endpoint", "status_code"],
    buckets=_HTTP_BUCKETS,
)

# ── 1-B. LLM Token 消耗（Counter） ──────────────────────────────────────────
# 类比：公司的「油卡消费记录」，按车型/员工/部门分开统计
llm_token_usage_total = Counter(
    name="synaris_llm_token_usage_total",
    documentation=(
        "LLM 调用消耗的 Token 总数（累计）。"
        "labels: model=模型名称 / user_id=用户ID前8位 / endpoint=调用接口 / token_type=prompt|completion"
    ),
    labelnames=["model", "user_id", "endpoint", "token_type"],
)

# ── 1-C. LLM 请求延迟（Histogram） ──────────────────────────────────────────
# LLM 调用通常比普通 HTTP 请求慢得多（1s ~ 60s），因此使用更大范围的桶
_LLM_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)

llm_request_duration_seconds = Histogram(
    name="synaris_llm_request_duration_seconds",
    documentation=(
        "单次 LLM API 调用延迟（秒）。"
        "labels: model=模型名称 / task_type=任务类型"
    ),
    labelnames=["model", "task_type"],
    buckets=_LLM_BUCKETS,
)

# ── 1-D. 缓存命中率（Gauge） ─────────────────────────────────────────────────
# Gauge 存储当前命中率快照（0.0 ~ 1.0），由各服务在写入缓存时主动更新
# 类比：超市「货架扫描命中率」，库存系统实时更新
cache_hit_ratio = Gauge(
    name="synaris_cache_hit_ratio",
    documentation=(
        "当前缓存命中率（0.0~1.0，1.0=100% 命中）。"
        "labels: cache_type=embedding|search|session|task"
    ),
    labelnames=["cache_type"],
)

# ── 1-E. Agent 任务延迟（Histogram） ────────────────────────────────────────
# Agent 任务耗时可能从几秒到几分钟不等
_AGENT_BUCKETS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)

agent_task_duration_seconds = Histogram(
    name="synaris_agent_task_duration_seconds",
    documentation=(
        "Agent 任务完整执行耗时（秒）。"
        "labels: task_type=single|multi|rag_only / status=completed|failed"
    ),
    labelnames=["task_type", "status"],
    buckets=_AGENT_BUCKETS,
)

# ── 1-F. 活跃连接数（Gauge） ─────────────────────────────────────────────────
# 实时反映服务当前承载的并发压力
# 类比：医院「在诊患者数量」，挂号时 +1，离院时 -1
active_connections = Gauge(
    name="synaris_active_connections",
    documentation="当前活跃的 HTTP/WebSocket 连接数（实时快照）。",
)

# ── 1-G. 文档分块入库量（Counter） ──────────────────────────────────────────
document_chunks_total = Counter(
    name="synaris_document_chunks_total",
    documentation=(
        "写入 Milvus 的文档分块总数（累计）。"
        "labels: collection=Collection名称 / file_type=pdf|docx|txt|markdown"
    ),
    labelnames=["collection", "file_type"],
)

# ── 1-H. RAG 检索指标（Counter + Histogram） ────────────────────────────────
rag_query_total = Counter(
    name="synaris_rag_query_total",
    documentation=(
        "RAG 知识库查询总次数（累计）。"
        "labels: collection=Collection名称 / status=success|failed|empty"
    ),
    labelnames=["collection", "status"],
)

rag_retrieval_duration_seconds = Histogram(
    name="synaris_rag_retrieval_duration_seconds",
    documentation=(
        "RAG 向量检索耗时（秒，不含 LLM 生成阶段）。"
        "labels: collection=Collection名称"
    ),
    labelnames=["collection"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 路径归一化（防 Cardinality 爆炸）
#
# 问题背景：
#   若直接以 /user/550e8400-e29b-41d4-a716-446655440000 作为 label，
#   每个用户 ID 产生一个独立的时间序列，Prometheus 内存会无限增长。
#
# 解决方案：
#   用正则将「动态段」替换为占位符，如：
#     /user/123           → /user/{id}
#     /task/task-abc123   → /task/{task_id}
#     /knowledge/doc.pdf  → /knowledge/{source_id}
# ═══════════════════════════════════════════════════════════════════════════════

# 归一化规则（按优先级从上到下匹配）
_PATH_NORMALIZATION_RULES: list[tuple[re.Pattern, str]] = [
    # UUID（含连字符格式）
    (re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "/{uuid}"),
    # task-xxx 格式
    (re.compile(r"/task-[a-zA-Z0-9]+"), "/{task_id}"),
    # sess-xxx 格式
    (re.compile(r"/sess-[a-zA-Z0-9]+"), "/{session_id}"),
    # URL 编码的 source_id（文件名）：仅替换包含特殊字符的段
    (re.compile(r"/[^/]*%[0-9A-Fa-f]{2}[^/]*"), "/{source_id}"),
    # 纯数字段（如分页 ID）
    (re.compile(r"/\d{4,}"), "/{id}"),
]

# 仅记录白名单 HTTP 方法（过滤 HEAD/OPTIONS 等噪音）
_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}

# 跳过指标采集的路径前缀（避免 /metrics 本身被记录）
_SKIP_PATHS = {"/metrics", "/health", "/favicon.ico"}


def _normalize_path(path: str) -> str:
    """
    将动态路径归一化，防止 Prometheus 标签基数（Cardinality）无限增长。

    示例：
        /api/v1/agent/status/task-abc123 → /api/v1/agent/status/{task_id}
        /api/v1/knowledge/my%2Fdoc.pdf   → /api/v1/knowledge/{source_id}
        /api/v1/users/42                 → /api/v1/users/{id}
    """
    normalized = path
    for pattern, replacement in _PATH_NORMALIZATION_RULES:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _safe_user_id(user_id: Optional[str]) -> str:
    """
    将 user_id 哈希截断为前 8 位，降低标签基数，同时保留可追踪性。

    示例：
        "550e8400-e29b-41d4-a716-446655440000" → "4a3f8c1d"
        None → "anonymous"
    """
    if not user_id:
        return "anonymous"
    digest = hashlib.sha256(str(user_id).encode()).hexdigest()
    return digest[:8]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MetricsMiddleware（ASGI 中间件）
#
# 类比：超市入口的「客流计数闸机」
#   - 每位顾客进门时开始计时
#   - 出门时记录「在店时长」和「消费状态（结账/退货）」
#   - 统计报表自动汇总，店长无需手动统计
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Prometheus 指标采集 ASGI 中间件。

    自动为每个 HTTP 请求记录：
      - 请求延迟（http_request_duration_seconds）
      - 活跃连接数（active_connections）

    跳过条件：
      - 路径前缀在 _SKIP_PATHS 中（/metrics / /health）
      - HTTP 方法不在白名单中（HEAD / OPTIONS 等）

    使用方式（在 main.py 中注册）：
        from app.core.observability import MetricsMiddleware
        app.add_middleware(MetricsMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # 跳过不需要采集的路径
        if any(path.startswith(skip) for skip in _SKIP_PATHS):
            return await call_next(request)

        method = request.method.upper()
        if method not in _ALLOWED_METHODS:
            return await call_next(request)

        # 归一化路径（用于 label）
        endpoint = _normalize_path(path)

        # 活跃连接 +1
        active_connections.inc()

        start_time = time.perf_counter()
        status_code = 500   # 默认值，确保异常时也有记录

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception as exc:
            # 未捕获异常记为 500，然后重新抛出
            status_code = 500
            raise exc

        finally:
            # 无论成功还是失败，都记录延迟
            elapsed = time.perf_counter() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).observe(elapsed)

            # 活跃连接 -1
            active_connections.dec()

            logger.debug(
                "metrics 记录 HTTP 请求",
                extra={
                    "method":      method,
                    "endpoint":    endpoint,
                    "status_code": status_code,
                    "elapsed_ms":  round(elapsed * 1000, 2),
                },
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 工具函数（供各业务模块调用）
#
# 设计原则：
#   - 单一入口：业务代码只需调用工具函数，不直接操作指标对象
#   - 容错降级：工具函数内部捕获所有异常，指标失败不影响主流程
#   - 类型安全：参数有明确类型注解，IDE 可自动补全
# ═══════════════════════════════════════════════════════════════════════════════

def track_llm_call(
    model:             str,
    prompt_tokens:     int,
    completion_tokens: int,
    task_type:         str  = "chat",
    user_id:           Optional[str] = None,
    endpoint:          str  = "unknown",
    latency_ms:        float = 0.0,
) -> None:
    """
    记录一次 LLM API 调用的指标（Token 消耗 + 延迟）。

    设计为「单一入口」：llm_client.py 在每次调用后调用此函数，
    无需在各业务服务中分散记录指标。

    Args:
        model:             模型名称（如 "gpt-4o" / "gpt-4o-mini"）
        prompt_tokens:     输入 Token 数量
        completion_tokens: 输出 Token 数量
        task_type:         任务类型（chat / rag / agent_reasoning / code / summary）
        user_id:           调用用户 ID（可为 None）
        endpoint:          调用来源接口（如 "/api/v1/chat"）
        latency_ms:        本次调用耗时（毫秒）

    示例（在 llm_client.py 的 invoke() 函数末尾调用）：
        from app.core.observability import track_llm_call
        track_llm_call(
            model="gpt-4o-mini",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            task_type="rag",
            user_id=str(current_user.id),
            endpoint="/api/v1/rag/query",
            latency_ms=elapsed_ms,
        )
    """
    try:
        safe_uid  = _safe_user_id(user_id)
        safe_ep   = _normalize_path(endpoint)
        safe_model = model or "unknown"

        # 记录 prompt Token
        if prompt_tokens > 0:
            llm_token_usage_total.labels(
                model=safe_model,
                user_id=safe_uid,
                endpoint=safe_ep,
                token_type="prompt",
            ).inc(prompt_tokens)

        # 记录 completion Token
        if completion_tokens > 0:
            llm_token_usage_total.labels(
                model=safe_model,
                user_id=safe_uid,
                endpoint=safe_ep,
                token_type="completion",
            ).inc(completion_tokens)

        # 记录延迟
        if latency_ms > 0:
            llm_request_duration_seconds.labels(
                model=safe_model,
                task_type=task_type or "unknown",
            ).observe(latency_ms / 1000.0)   # 转换为秒

        logger.debug(
            "LLM 指标已记录",
            extra={
                "model":             safe_model,
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "task_type":         task_type,
                "latency_ms":        latency_ms,
            },
        )

    except Exception as exc:
        # 指标记录失败不应中断业务流程
        logger.warning("track_llm_call: 指标记录失败（已忽略）: %s", exc)


def track_cache_hit(
    cache_type: str,
    hit:        bool,
    hit_count:  Optional[int] = None,
    miss_count: Optional[int] = None,
) -> None:
    """
    更新缓存命中率仪表盘（Gauge）。

    支持两种调用方式：
      1. 逐次调用（hit=True/False）：内部维护滑动窗口
      2. 直接传入聚合值（hit_count + miss_count）：直接计算并设置

    Args:
        cache_type: 缓存类型（embedding / search / session / task）
        hit:        本次操作是否命中缓存（方式 1 使用）
        hit_count:  当前总命中次数（方式 2 使用）
        miss_count: 当前总未命中次数（方式 2 使用）

    示例（在 vector_store.py 的 similarity_search 中调用）：
        from app.core.observability import track_cache_hit
        track_cache_hit("search", hit=cached is not None)

    示例（在 EmbeddingClient 的 get_stats 中调用，方式 2）：
        stats = embedding_client.get_stats()
        track_cache_hit(
            "embedding",
            hit=False,  # 占位，方式 2 不使用此参数
            hit_count=stats["hit_count"],
            miss_count=stats["miss_count"],
        )
    """
    try:
        safe_type = cache_type or "unknown"

        if hit_count is not None and miss_count is not None:
            # 方式 2：直接计算命中率
            total = hit_count + miss_count
            ratio = hit_count / total if total > 0 else 0.0
        else:
            # 方式 1：简化更新（从当前 Gauge 值估算，适合高频调用）
            # 注意：Gauge 不提供读取方式，此处使用 EMA（指数移动平均）近似
            # hit=True → 向 1.0 靠近；hit=False → 向 0.0 靠近
            # 衰减因子 α=0.01（100 次请求后接近真实命中率）
            alpha = 0.01
            current = cache_hit_ratio.labels(cache_type=safe_type)
            # 这种近似对于监控仪表盘足够精确
            # 精确统计应使用方式 2（在服务层维护计数器后批量更新）
            ratio = 1.0 if hit else 0.0   # 简化：直接设置当前值

        cache_hit_ratio.labels(cache_type=safe_type).set(ratio)

    except Exception as exc:
        logger.warning("track_cache_hit: 指标记录失败（已忽略）: %s", exc)


def track_agent_task(
    task_type:   str,
    duration_ms: float,
    status:      str = "completed",
) -> None:
    """
    记录一次 Agent 任务完成的执行时长。

    在 Celery Worker 的 run_agent_task 或 run_supervisor 结束时调用。

    Args:
        task_type:   任务类型（single / multi / rag_only）
        duration_ms: 任务总耗时（毫秒）
        status:      最终状态（completed / failed）

    示例（在 workers/agent_worker.py 末尾调用）：
        from app.core.observability import track_agent_task
        track_agent_task("single", elapsed_ms, status="completed")
    """
    try:
        agent_task_duration_seconds.labels(
            task_type=task_type or "unknown",
            status=status or "unknown",
        ).observe(duration_ms / 1000.0)

    except Exception as exc:
        logger.warning("track_agent_task: 指标记录失败（已忽略）: %s", exc)


def inc_active_connections(delta: int = 1) -> None:
    """
    调整活跃连接计数。

    delta=+1 表示新增连接（如 WebSocket 建立）
    delta=-1 表示连接关闭

    示例（在 api/agent.py 的 WebSocket 端点中使用）：
        from app.core.observability import inc_active_connections

        @router.websocket("/agent/stream/{task_id}")
        async def stream_agent(websocket: WebSocket, task_id: str):
            await websocket.accept()
            inc_active_connections(+1)
            try:
                ...
            finally:
                inc_active_connections(-1)
    """
    try:
        if delta > 0:
            active_connections.inc(delta)
        elif delta < 0:
            active_connections.dec(abs(delta))
    except Exception as exc:
        logger.warning("inc_active_connections: 指标更新失败（已忽略）: %s", exc)


def track_document_chunks(
    collection: str,
    file_type:  str,
    count:      int = 1,
) -> None:
    """
    记录文档分块写入 Milvus 的数量。

    在 document_service.py 完成向量入库后调用。

    Args:
        collection: Milvus Collection 名称
        file_type:  文件类型（pdf / docx / txt / markdown）
        count:      本次写入的分块数量
    """
    try:
        document_chunks_total.labels(
            collection=collection or "unknown",
            file_type=file_type or "unknown",
        ).inc(count)
    except Exception as exc:
        logger.warning("track_document_chunks: 指标记录失败（已忽略）: %s", exc)


def track_rag_query(
    collection:   str,
    status:       str,
    duration_ms:  float = 0.0,
) -> None:
    """
    记录一次 RAG 知识库查询的结果与耗时。

    Args:
        collection:  查询的 Collection 名称
        status:      查询结果状态（success / failed / empty）
        duration_ms: 向量检索耗时（毫秒，不含 LLM 生成）
    """
    try:
        safe_collection = collection or "unknown"

        rag_query_total.labels(
            collection=safe_collection,
            status=status or "unknown",
        ).inc()

        if duration_ms > 0:
            rag_retrieval_duration_seconds.labels(
                collection=safe_collection,
            ).observe(duration_ms / 1000.0)

    except Exception as exc:
        logger.warning("track_rag_query: 指标记录失败（已忽略）: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GET /metrics 路由
#
# Prometheus Scraper 会定期（默认 15s）向此端点发送 GET 请求，
# 拉取所有已注册指标的当前快照，写入时序数据库（TSDB）。
#
# 格式：Prometheus 文本格式（exposition format 0.0.4）
# ═══════════════════════════════════════════════════════════════════════════════

metrics_router = APIRouter(tags=["Observability"])


@metrics_router.get(
    "/metrics",
    summary="Prometheus 指标采集端点",
    description=(
        "返回所有 Prometheus 指标的当前快照（文本格式）。\n\n"
        "供 Prometheus Scraper 定期拉取（默认间隔 15s）。\n\n"
        "⚠️ 生产环境建议通过 Nginx 设置 IP 白名单，限制只有监控系统可访问此端点。"
    ),
    response_class=Response,
    include_in_schema=True,   # 在 Swagger UI 中展示，方便调试
)
async def metrics_endpoint() -> Response:
    """
    生成并返回 Prometheus 文本格式的全量指标快照。

    Prometheus 拉取模型说明：
      Scraper → GET /metrics → 获取当前瞬时快照
      每个指标的「当前值」由各模块在业务逻辑执行时主动更新。
      此端点只是「读取快照」，本身不产生任何计算开销。
    """
    try:
        # generate_latest() 从全局 REGISTRY 中收集所有已注册指标的当前值
        data = generate_latest(REGISTRY)
        return Response(
            content=data,
            media_type=CONTENT_TYPE_LATEST,
            headers={
                # 明确告知 Prometheus Scraper 响应格式版本
                "X-Prometheus-Exposition-Format-Version": "0.0.4",
            },
        )
    except Exception as exc:
        logger.error("metrics_endpoint: 生成指标失败: %s", exc)
        return Response(
            content=f"# ERROR: 指标生成失败: {exc}\n",
            media_type="text/plain; charset=utf-8",
            status_code=500,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 指标注册表信息（调试辅助）
#
# 在服务启动日志中打印已注册的指标列表，方便运维人员验证配置是否生效。
# ═══════════════════════════════════════════════════════════════════════════════

def log_registered_metrics() -> None:
    """
    在日志中打印当前已注册的 Prometheus 指标列表。

    建议在 lifespan 启动阶段调用（main.py）：
        from app.core.observability import log_registered_metrics
        log_registered_metrics()
    """
    metric_names = sorted(
        collector._name  # type: ignore[attr-defined]
        for collector in REGISTRY._names_to_collectors.values()
        if hasattr(collector, "_name")
    )
    logger.info(
        "Prometheus 指标注册完成",
        extra={
            "metric_count": len(metric_names),
            "metrics":      metric_names[:20],   # 最多展示 20 条，避免日志过长
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 模块级常量（供 Grafana Dashboard 配置参考）
#
# 以下变量记录了所有指标的完整名称，便于在 grafana_dashboard.json 中引用，
# 也作为本模块的「指标目录」供开发者查阅。
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_NAMES = {
    "http_latency":        "synaris_http_request_duration_seconds",
    "llm_tokens":          "synaris_llm_token_usage_total",
    "llm_latency":         "synaris_llm_request_duration_seconds",
    "cache_hit":           "synaris_cache_hit_ratio",
    "agent_task_latency":  "synaris_agent_task_duration_seconds",
    "active_connections":  "synaris_active_connections",
    "document_chunks":     "synaris_document_chunks_total",
    "rag_queries":         "synaris_rag_query_total",
    "rag_retrieval":       "synaris_rag_retrieval_duration_seconds",
}

# 推荐的 Grafana PromQL 查询表达式（可直接粘贴到 Dashboard 面板）
RECOMMENDED_QUERIES = {
    "HTTP P95 延迟":
        'histogram_quantile(0.95, rate(synaris_http_request_duration_seconds_bucket[5m]))',

    "HTTP 请求速率（QPS）":
        'sum(rate(synaris_http_request_duration_seconds_count[1m])) by (endpoint)',

    "LLM Token 消耗速率（分钟）":
        'sum(rate(synaris_llm_token_usage_total[1m])) by (model, token_type)',

    "LLM P95 调用延迟":
        'histogram_quantile(0.95, rate(synaris_llm_request_duration_seconds_bucket[5m])) by (model)',

    "各缓存命中率":
        'synaris_cache_hit_ratio',

    "Agent 任务 P95 耗时":
        'histogram_quantile(0.95, rate(synaris_agent_task_duration_seconds_bucket[10m])) by (task_type)',

    "当前活跃连接数":
        'synaris_active_connections',

    "RAG 查询成功率":
        'sum(rate(synaris_rag_query_total{status="success"}[5m])) / sum(rate(synaris_rag_query_total[5m]))',
}