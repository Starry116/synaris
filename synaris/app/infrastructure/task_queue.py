import os
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from celery import Celery, Task
from celery.result import AsyncResult
from kombu import Exchange, Queue

T = TypeVar("T", bound=Callable[..., Any])


def _get_logger() -> logging.Logger:
    """
    Prefer项目中的结构化日志，如果不存在则退回标准 logging。
    """
    try:
        from app.core.logging import get_logger  # type: ignore

        return cast(logging.Logger, get_logger("task_queue"))
    except Exception:  # pragma: no cover - 仅在尚未实现 logging 时触发
        logger = logging.getLogger("task_queue")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return logger


logger = _get_logger()


def _get_broker_and_backend() -> Dict[str, str]:
    """
    从环境变量中解析 Celery broker / backend。

    - 优先使用 CELERY_BROKER_URL / CELERY_RESULT_BACKEND
    - 其次回退到 REDIS_URL（Step 3 中 RedisConfig 对应的 URL）
    """
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    broker_url = os.getenv("CELERY_BROKER_URL", redis_url)
    backend_url = os.getenv("CELERY_RESULT_BACKEND", redis_url)
    return {"broker_url": broker_url, "backend_url": backend_url}


broker_backend = _get_broker_and_backend()

celery_app = Celery(
    "synaris",
    broker=broker_backend["broker_url"],
    backend=broker_backend["backend_url"],
)

# 基础配置：序列化、时区等
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


# ============
# 队列与路由定义
# ============

default_exchange = Exchange("celery", type="direct")

celery_app.conf.task_queues = (
    Queue("high_priority", default_exchange, routing_key="high_priority"),
    Queue("default", default_exchange, routing_key="default"),
    Queue("low_priority", default_exchange, routing_key="low_priority"),
)

# 默认将任务投递到 default 队列，具体任务可以通过 queue / routing_key 覆盖
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_default_exchange = "celery"
celery_app.conf.task_default_routing_key = "default"

# 为关键任务预留路由（可根据后续需要在其他模块中扩展）
celery_app.conf.task_routes = {
    "app.workers.agent_worker.run_agent_task": {"queue": "high_priority"},
    "app.workers.document_worker.process_document": {"queue": "default"},
}


# ============
# 通用任务基类与装饰器
# ============

class LoggedRetryTask(Task):
    """
    通用 Task 基类：
    - 统一日志记录
    - 失败自动重试（默认最多 3 次）
    """

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    retry_backoff = True
    retry_jitter = True

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        logger.info(
            "Celery task succeeded",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": str(args),
                "kwargs": str(kwargs),
            },
        )

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        logger.error(
            "Celery task failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": str(args),
                "kwargs": str(kwargs),
                "error": str(exc),
            },
        )


def celery_task(
    _func: Optional[T] = None,
    *,
    name: Optional[str] = None,
    queue: Optional[str] = None,
    max_retries: int = 3,
) -> Callable[[T], T]:
    """
    通用任务装饰器：
    - 绑定 LoggedRetryTask
    - 自动记录日志
    - 自动异常重试（最多 3 次，可调整）

    用法示例：
        @celery_task(name="document.process", queue="default")
        def process(...):
            ...
    """

    def decorator(func: T) -> T:
        task_name = name or f"{func.__module__}.{func.__name__}"

        task = celery_app.task(
            name=task_name,
            base=LoggedRetryTask,
            bind=True,
            max_retries=max_retries,
            queue=queue,
        )(func)  # type: ignore[arg-type]

        return cast(T, task)

    if _func is not None:
        return decorator(_func)

    return decorator


# ============
# 任务状态查询
# ============

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    根据 task_id 查询 Celery 任务状态。

    返回示例：
    {
        "id": "...",
        "state": "PENDING" | "STARTED" | "SUCCESS" | "FAILURE" | ...,
        "ready": bool,
        "successful": bool,
        "failed": bool,
        "result": Any,
        "traceback": Optional[str],
    }
    """
    result = AsyncResult(task_id, app=celery_app)
    data: Dict[str, Any] = {
        "id": result.id,
        "state": result.state,
        "ready": result.ready(),
        "successful": result.successful(),
        "failed": result.failed(),
        "traceback": result.traceback,
    }
    # 只有在成功时才返回 result，避免序列化巨大异常对象
    if result.successful():
        data["result"] = result.result
    else:
        data["result"] = None
    return data


__all__ = ["celery_app", "celery_task", "get_task_status"]

