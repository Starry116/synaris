"""
@File       : task_queue.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Celery 异步任务队列配置与基础设施。
@Features:
  - Celery 应用实例（broker=Redis DB1，backend=Redis DB2，与缓存 DB0 三库隔离）
  - 三优先级队列：
      · high    → Agent 实时任务（用户感知延迟，Worker 优先调度）
      · default → 文档处理任务（后台耗时，不阻塞前台交互）
      · low     → 评估/统计任务（可延迟，资源紧张时自然降级）
  - @celery_task 通用装饰器工厂：
      · 自动注入结构化日志（task_id / task_name / attempt）
      · 统一异常捕获 + 指数退避重试（2^attempt 秒，默认最多 3 次）
      · 任务开始/成功/失败均记录耗时
  - TaskResult(BaseModel)：get_task_status() 的统一返回格式
  - get_task_status(task_id) → TaskResult：
      优先读 Redis 业务快照（agent:result:{task_id}）
      降级读 Celery AsyncResult（原生元数据）

  ── Celery Worker 与 FastAPI 主进程的协作机制 ──────────────────────────────
  ┌──────────────────────────────────────────────────────────────────────┐
  │  FastAPI（异步）           Redis（消息总线）      Celery Worker（多进程）│
  │       │                        │                         │           │
  │ POST /agent/run                │                         │           │
  │       │─ agent:status:pending ─▶│                         │           │
  │       │─ apply_async ──────────────────────────────────── ▶          │
  │       │◀─ {task_id, pending}    │         run_agent_task()│           │
  │       │                        │                         │ 执行中    │
  │ WS /agent/stream/{id}          │                         │           │
  │       │─ SUBSCRIBE ────────────▶│◀── PUBLISH ─────────────│ 每节点    │
  │       │◀─ AgentStepEvent ───────│    agent:events:{id}    │           │
  │       │◀─ {type:done} ──────────│◀── PUBLISH ─────────────│ 完成      │
  │       │                        │    ← 写入 Postgres ───────│           │
  │       │                        │    ← 写入 Redis result ───│           │
  └──────────────────────────────────────────────────────────────────────┘

  关键设计：WebSocket 订阅在 FastAPI 进程，执行在 Worker 进程，
  Redis Pub/Sub 是唯一的跨进程实时通道，无需共享内存。

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from typing import Any, Callable, Optional

from celery import Celery, Task
from celery.result import AsyncResult
from kombu import Exchange, Queue
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. DSN 解析（延迟读取，避免导入时即连接）
# ─────────────────────────────────────────────

def _get_broker_url() -> str:
    """
    Celery Broker：Redis DB1。
    DB0 = 业务缓存，DB1 = Broker，DB2 = Result Backend，三库物理隔离。
    隔离原因：DB0 设置了 maxmemory-policy allkeys-lru，
    LRU 驱逐可能误删未处理的任务消息；DB1/DB2 不设 maxmemory，任务不丢失。
    """
    try:
        from config.settings import get_settings  # type: ignore[import]
        s = get_settings()
        return getattr(s, "celery_broker_url", None) or os.getenv(
            "CELERY_BROKER_URL", "redis://localhost:6379/1"
        )
    except ImportError:
        return os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")


def _get_result_backend() -> str:
    """Celery Result Backend：Redis DB2。"""
    try:
        from config.settings import get_settings  # type: ignore[import]
        s = get_settings()
        return getattr(s, "celery_result_backend", None) or os.getenv(
            "CELERY_RESULT_BACKEND", "redis://localhost:6379/2"
        )
    except ImportError:
        return os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")


# ─────────────────────────────────────────────
# 2. 队列与路由定义
# ─────────────────────────────────────────────

# 队列名称常量（避免散落魔法字符串）
QUEUE_HIGH    = "high"
QUEUE_DEFAULT = "default"
QUEUE_LOW     = "low"

_default_exchange = Exchange("synaris", type="direct")

_TASK_QUEUES = (
    # high：Agent 实时任务。用户点击「提交」后立刻等待响应，延迟敏感。
    Queue(
        QUEUE_HIGH,
        exchange=_default_exchange,
        routing_key=QUEUE_HIGH,
    ),
    # default：文档处理任务。上传后台异步处理，用户不阻塞等待。
    Queue(
        QUEUE_DEFAULT,
        exchange=_default_exchange,
        routing_key=QUEUE_DEFAULT,
    ),
    # low：评估/统计任务（Step 25 eval_service）。定时批量，不影响主流程。
    Queue(
        QUEUE_LOW,
        exchange=_default_exchange,
        routing_key=QUEUE_LOW,
    ),
)

# 任务路由规则：按任务全名前缀自动路由，无需每次 apply_async 时手动指定 queue
_TASK_ROUTES: dict[str, dict] = {
    "workers.agent_worker.*":    {"queue": QUEUE_HIGH},
    "workers.document_worker.*": {"queue": QUEUE_DEFAULT},
    "workers.eval_worker.*":     {"queue": QUEUE_LOW},    # Step 25 预留
}


# ─────────────────────────────────────────────
# 3. Celery 应用实例（模块级单例）
# ─────────────────────────────────────────────

celery_app = Celery(
    "synaris",
    broker=_get_broker_url(),
    backend=_get_result_backend(),
    include=[
        "workers.document_worker",
        "workers.agent_worker",
    ],
)

celery_app.conf.update(
    # ── 序列化 ──────────────────────────────────────────────────────────────
    task_serializer  = "json",
    result_serializer = "json",
    accept_content   = ["json"],

    # ── 时区 ────────────────────────────────────────────────────────────────
    timezone   = "UTC",
    enable_utc = True,

    # ── 队列与路由 ────────────────────────────────────────────────────────
    task_queues       = _TASK_QUEUES,
    task_default_queue = QUEUE_DEFAULT,
    task_routes       = _TASK_ROUTES,

    # ── 可靠性 ────────────────────────────────────────────────────────────
    # task_acks_late=True：任务执行完毕后才向 Broker 确认（ACK）。
    # 若 Worker 在执行中崩溃，Broker 认为消息未被消费，会重新投递给其他 Worker。
    # 类比「快递签收」：快递到手确认，而非送到门口就签收。
    task_acks_late            = True,
    task_reject_on_worker_lost = True,   # Worker 意外退出 → 任务重回队列

    # worker_prefetch_multiplier=1：每个 Worker 子进程同一时刻只预取 1 个任务。
    # 防止少数 Worker 囤积大量任务，导致其他 Worker 空闲（公平调度）。
    worker_prefetch_multiplier = 1,

    # ── 结果保留 ──────────────────────────────────────────────────────────
    # Celery 原生结果保留 3 天，与 Redis agent:result TTL 保持一致
    result_expires = 86400 * 3,

    # ── 监控（Celery Flower 需要） ────────────────────────────────────────
    worker_send_task_events = True,
    task_send_sent_event    = True,

    # ── 内存保护 ──────────────────────────────────────────────────────────
    # 处理 200 个任务后重启子进程，防止长期运行导致内存泄漏（Python 碎片化）
    worker_max_tasks_per_child  = 200,
    # 子进程内存超过 512MB 时强制重启（asyncpg + LangChain 内存占用较大）
    worker_max_memory_per_child = 512_000,   # KB 单位
)


# ─────────────────────────────────────────────
# 4. TaskResult — 统一状态模型
# ─────────────────────────────────────────────

class TaskResult(BaseModel):
    """
    get_task_status() 的统一返回格式。

    整合了 Redis 业务快照（由 Worker 写入）和 Celery 原生结果，
    让调用方（API 层）无需感知底层存储细节。

    状态值说明：
      pending   → 已提交，Worker 尚未拾取
      running   → Worker 正在执行
      waiting_human → 等待人工确认（Human-in-the-Loop）
      completed → 成功完成（result 字段含最终输出）
      failed    → 执行失败（error 字段含错误信息）
      cancelled → 已被取消（revoke 调用后）
    """
    task_id:     str
    status:      str
    result:      Optional[Any]  = None   # 成功时的结果（dict / str）
    error:       Optional[str]  = None   # 失败时的错误描述
    progress:    Optional[int]  = None   # 0-100 进度（文档处理场景）
    tokens_used: Optional[int]  = None   # Token 消耗（Agent 任务场景）
    started_at:  Optional[str]  = None   # ISO 8601 开始时间
    finished_at: Optional[str]  = None   # ISO 8601 结束时间


# ─────────────────────────────────────────────
# 5. get_task_status() — 统一状态查询
# ─────────────────────────────────────────────

def get_task_status(task_id: str) -> TaskResult:
    """
    查询任务当前状态（同步接口，在同步上下文中调用）。

    数据来源优先级：
      ① Redis `agent:result:{task_id}` — 业务快照，由 Worker 写入，内容最全
      ② Redis `doc:progress:{task_id}` — 文档处理进度（仅运行中场景）
      ③ Celery AsyncResult             — Celery 原生元数据（兜底）

    为什么不直接用 Celery AsyncResult？
      Celery 的 result 只存简单 Python 对象，不包含 final_answer、step_log
      等业务字段。Worker 完成后会同时写入 Redis 业务快照，内容更丰富。
    """
    # ── 优先读 Redis 业务快照 ─────────────────────────────────────────────
    snapshot = _sync_fetch_redis_snapshot(task_id)
    if snapshot:
        return TaskResult(
            task_id    = task_id,
            status     = snapshot.get("status", "unknown"),
            result     = snapshot.get("result"),
            error      = snapshot.get("error"),
            progress   = snapshot.get("progress"),
            tokens_used = snapshot.get("tokens_used"),
            started_at  = snapshot.get("started_at"),
            finished_at = snapshot.get("finished_at"),
        )

    # ── 降级：Celery AsyncResult ──────────────────────────────────────────
    ar = AsyncResult(task_id, app=celery_app)

    # Celery 状态 → 业务状态映射
    _STATUS_MAP = {
        "pending": "pending",
        "received": "pending",
        "started":  "running",
        "retry":    "running",
        "success":  "completed",
        "failure":  "failed",
        "revoked":  "cancelled",
    }
    biz_status = _STATUS_MAP.get(ar.status.lower(), ar.status.lower())

    result_val = None
    error_val  = None
    if ar.successful():
        result_val = ar.result
    elif ar.failed():
        error_val = str(ar.result)

    return TaskResult(
        task_id=task_id,
        status=biz_status,
        result=result_val,
        error=error_val,
    )


def _sync_fetch_redis_snapshot(task_id: str) -> Optional[dict]:
    """
    同步方式从 Redis 读取业务快照。

    在同步上下文（如 Celery 任务内部的状态查询）中使用。
    处理「有运行中事件循环」和「没有事件循环」两种场景。
    """
    try:
        from infrastructure.redis_client import get_redis  # type: ignore[import]

        async def _fetch() -> Optional[dict]:
            redis = await get_redis()
            # 优先读完整业务快照
            raw = await redis.get(f"agent:result:{task_id}")
            if raw:
                return json.loads(raw)
            # 文档处理进度（Worker 写入的中间状态）
            prog = await redis.get(f"doc:progress:{task_id}")
            if prog:
                return {"status": "running", "progress": int(prog)}
            return None

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 有运行中的事件循环（如在 FastAPI 中被同步调用）→ 用线程池
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, _fetch()).result(timeout=3.0)
            else:
                return loop.run_until_complete(_fetch())
        except RuntimeError:
            return asyncio.run(_fetch())

    except Exception as exc:
        logger.debug("_sync_fetch_redis_snapshot: 读取失败（非致命）: %s", exc)
        return None


# ─────────────────────────────────────────────
# 6. @celery_task — 通用装饰器工厂
# ─────────────────────────────────────────────

def celery_task(
    queue:       str = QUEUE_DEFAULT,
    max_retries: int = 3,
    **celery_kwargs: Any,
) -> Callable:
    """
    Synaris 通用 Celery 任务装饰器工厂。

    在标准 @celery_app.task 基础上叠加三层能力：
      1. 结构化日志：自动记录 task_id / task_name / attempt / elapsed_ms
      2. 统一重试：异常时指数退避（2^attempt 秒），超过 max_retries 后标记 FAILED
      3. 开箱即用：调用方只需关注业务逻辑，无需写重复的 try/except/retry

    指数退避示意（类比「网络重连间隔」）：
      首次执行失败 → 等 2秒 → 第1次重试
      第1次重试失败 → 等 4秒 → 第2次重试
      第2次重试失败 → 等 8秒 → 第3次重试（最后一次）
      第3次重试失败 → 标记 FAILED，不再重试

    使用方式：
        from infrastructure.task_queue import celery_task, QUEUE_HIGH

        @celery_task(queue=QUEUE_HIGH, max_retries=5)
        def run_agent_task(self, task_id: str, task_config: dict):
            # self 是 Celery Task 实例（bind=True 由装饰器内部设置）
            self.update_state(state="STARTED", meta={"task_id": task_id})
            ...

    Args:
        queue:       目标队列（QUEUE_HIGH / QUEUE_DEFAULT / QUEUE_LOW）
        max_retries: 最大重试次数（不含首次执行，默认 3）
        **celery_kwargs: 透传给 @celery_app.task 的其他参数（如 time_limit）
    """
    def decorator(func: Callable) -> Callable:
        # 自动构造任务全名（供 _TASK_ROUTES 路由规则匹配）
        # 格式：workers.<module_name>.<func_name>
        module_short = func.__module__.split(".")[-1]   # e.g. "document_worker"
        task_full_name = f"workers.{module_short}.{func.__name__}"

        @celery_app.task(
            bind=True,                       # 将 Task 实例注入为第一个参数（self）
            name=task_full_name,
            queue=queue,
            max_retries=max_retries,
            **celery_kwargs,
        )
        @functools.wraps(func)
        def wrapper(self: Task, *args: Any, **kwargs: Any) -> Any:
            task_id   = self.request.id or "local-sync"
            task_name = self.name
            attempt   = self.request.retries   # 0=首次，1=第1次重试，…

            _log = logging.getLogger(func.__module__)
            _log.info(
                "任务开始 | task=%s | task_id=%s | attempt=%d/%d",
                task_name, task_id, attempt + 1, max_retries + 1,
            )

            start_time = time.monotonic()

            try:
                result = func(self, *args, **kwargs)

                elapsed_ms = (time.monotonic() - start_time) * 1000
                _log.info(
                    "任务成功 | task=%s | task_id=%s | elapsed=%.0fms",
                    task_name, task_id, elapsed_ms,
                )
                return result

            except self.MaxRetriesExceededError:
                # 已到达重试上限，由 Celery 框架标记 FAILURE
                elapsed_ms = (time.monotonic() - start_time) * 1000
                _log.error(
                    "任务重试耗尽 | task=%s | task_id=%s | attempts=%d | elapsed=%.0fms",
                    task_name, task_id, max_retries + 1, elapsed_ms,
                )
                raise

            except Exception as exc:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                retry_countdown = 2 ** (attempt + 1)   # 指数退避

                if attempt < max_retries:
                    _log.warning(
                        "任务失败，准备重试 | task=%s | task_id=%s | "
                        "attempt=%d/%d | next_retry_in=%ds | error=%s | elapsed=%.0fms",
                        task_name, task_id,
                        attempt + 1, max_retries + 1,
                        retry_countdown, exc, elapsed_ms,
                    )
                    # countdown 单位为秒；exc 保留原始异常链
                    raise self.retry(exc=exc, countdown=retry_countdown)
                else:
                    _log.error(
                        "任务最终失败 | task=%s | task_id=%s | "
                        "attempts=%d | elapsed=%.0fms | error=%s",
                        task_name, task_id,
                        max_retries + 1, elapsed_ms, exc,
                        exc_info=True,
                    )
                    raise   # 重新抛出，Celery 将任务标记为 FAILURE

        return wrapper
    return decorator


# ─────────────────────────────────────────────
# 7. 便捷工具函数
# ─────────────────────────────────────────────

def revoke_task(task_id: str, terminate: bool = False) -> None:
    """
    撤销（取消）一个 Celery 任务。

    Args:
        task_id:   Celery 任务 ID
        terminate: True → 强制终止正在执行的任务进程（SIGTERM）
                   False → 仅阻止尚未开始的任务（不影响已开始的任务）

    使用场景：
        POST /agent/cancel/{task_id}
        → revoke_task(celery_task_id, terminate=True)
    """
    celery_app.control.revoke(task_id, terminate=terminate, signal="SIGTERM")
    logger.info("revoke_task: 任务已撤销 | task_id=%s | terminate=%s", task_id, terminate)


def get_queue_lengths() -> dict[str, int]:
    """
    获取三个优先级队列的当前积压长度（用于监控和自动扩缩容判断）。

    Returns:
        {"high": N, "default": N, "low": N}

    注意：此函数直接查询 Redis，要求 Broker 为 Redis（不适用于 RabbitMQ）。
    """
    try:
        with celery_app.pool.acquire(block=True, timeout=2) as conn:
            lengths = {}
            for q_name in (QUEUE_HIGH, QUEUE_DEFAULT, QUEUE_LOW):
                lengths[q_name] = conn.default_channel.client.llen(q_name)
            return lengths
    except Exception as exc:
        logger.warning("get_queue_lengths: 查询失败: %s", exc)
        return {QUEUE_HIGH: -1, QUEUE_DEFAULT: -1, QUEUE_LOW: -1}