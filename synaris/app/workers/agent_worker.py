import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import redis.asyncio as redis  # type: ignore

from app.infrastructure.task_queue import celery_task


def _get_redis_url() -> str:
    """
    统一从环境变量中获取 Redis 连接地址。
    """
    return os.getenv("REDIS_URL", "redis://redis:6379/0")


def _get_pubsub_channel(task_id: str) -> str:
    """
    Agent 任务状态更新使用的 Redis Pub/Sub 通道。
    WebSocket 层（Step 16）可以订阅该通道，将事件转发给前端。
    """
    return f"agent:task:{task_id}"


async def _publish_event(task_id: str, event: Dict[str, Any]) -> None:
    """
    通过 Redis Pub/Sub 发布 Agent 执行过程中的事件。
    """
    client = redis.from_url(_get_redis_url(), decode_responses=True)
    try:
        channel = _get_pubsub_channel(task_id)
        payload = json.dumps(event, ensure_ascii=False)
        await client.publish(channel, payload)
    finally:
        await client.close()


async def _update_task_status_in_db(
    task_id: Optional[str],
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    更新 PostgreSQL 中的任务状态表（models.task.Task）。

    与 document_worker 中的实现类似，这里也通过延迟导入以避免强耦合。
    """
    try:
        from sqlalchemy import update  # type: ignore

        from app.infrastructure.postgres_client import get_db_session  # type: ignore
        from app.models.task import Task  # type: ignore
    except Exception:
        return

    async with get_db_session() as session:
        stmt = (
            update(Task)
            .where(Task.id == task_id)
            .values(
                status=status,
                result=result,
                error_message=error_message,
            )
        )
        await session.execute(stmt)
        await session.commit()


@celery_task(name="app.workers.agent_worker.run_agent_task", queue="high_priority")
def run_agent_task(task_id: str, task_config: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Agent 任务 Celery worker：
    - 根据 task_config 决定调用单 Agent workflow 还是多 Agent supervisor
    - 每个节点执行后由内部逻辑通过 Redis Pub/Sub 发布状态事件
    - 任务完成后写入 PostgreSQL task 表，并发布最终结果事件
    """

    async def _run() -> Dict[str, Any]:
        start_time = time.perf_counter()

        # 任务开始事件
        await _publish_event(
            task_id,
            {
                "type": "task_started",
                "task_id": task_id,
                "session_id": session_id,
                "config": task_config,
                "timestamp": time.time(),
            },
        )
        await _update_task_status_in_db(task_id=task_id, status="RUNNING")

        mode = (task_config.get("mode") or "single").lower()

        # 延迟导入，避免在对应模块尚未实现时阻塞 worker 启动
        if mode == "multi":
            from app.agents import supervisor  # type: ignore

            if not hasattr(supervisor, "run_supervisor"):
                raise RuntimeError("agents.supervisor.run_supervisor is not implemented yet")

            result_state = await supervisor.run_supervisor(  # type: ignore[attr-defined]
                task_config=task_config,
                session_id=session_id,
                task_id=task_id,
            )
        else:
            from app.agents import workflow  # type: ignore

            if not hasattr(workflow, "run_workflow"):
                raise RuntimeError("agents.workflow.run_workflow is not implemented yet")

            # 约定 run_workflow 接收 task(str) 与 session_id，task_config 作为可选配置字典
            result_state = await workflow.run_workflow(  # type: ignore[attr-defined]
                task=task_config.get("task") or task_config.get("prompt") or "",
                session_id=session_id,
                config=task_config,
                task_id=task_id,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # 将最终结果写入数据库
        await _update_task_status_in_db(
            task_id=task_id,
            status="SUCCESS",
            result={
                "state": result_state,
                "duration_ms": duration_ms,
            },
        )

        # 最终事件：任务完成
        await _publish_event(
            task_id,
            {
                "type": "done",
                "task_id": task_id,
                "session_id": session_id,
                "result": result_state,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            },
        )

        return {
            "ok": True,
            "task_id": task_id,
            "session_id": session_id,
            "result": result_state,
            "duration_ms": duration_ms,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        async def _on_error() -> None:
            await _update_task_status_in_db(
                task_id=task_id,
                status="FAILURE",
                error_message=str(exc),
            )
            await _publish_event(
                task_id,
                {
                    "type": "error",
                    "task_id": task_id,
                    "session_id": session_id,
                    "error": str(exc),
                    "timestamp": time.time(),
                },
            )

        asyncio.run(_on_error())
        # 抛出异常交由 LoggedRetryTask 统一处理重试与日志
        raise


__all__ = ["run_agent_task"]

