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
    优先使用 REDIS_URL，以兼容 Step 3 的配置说明。
    """
    return os.getenv("REDIS_URL", "redis://redis:6379/0")


async def _set_progress(task_id: str, progress: int) -> None:
    """
    将进度信息写入 Redis，key 设计为：
        task:{task_id}:progress -> 0-100
    """
    client = redis.from_url(_get_redis_url(), decode_responses=True)
    try:
        key = f"task:{task_id}:progress"
        await client.set(key, progress, ex=24 * 60 * 60)
    finally:
        await client.close()


async def _update_task_status_in_db(
    task_id: Optional[str],
    user_id: Optional[str],
    status: str,
    progress: int,
    result: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    更新 PostgreSQL 中的任务状态表（models.task.Task）。

    注意：为了避免与尚未完成的 Step 21 强耦合，
    这里通过延迟导入方式调用，如果相关模块尚未实现，则静默跳过。
    """
    try:
        from sqlalchemy import update  # type: ignore

        from app.infrastructure.postgres_client import get_db_session  # type: ignore
        from app.models.task import Task  # type: ignore
    except Exception:
        # 数据库层尚未就绪时不中断任务，只是暂时无法持久化状态
        return

    async with get_db_session() as session:
        stmt = (
            update(Task)
            .where(Task.celery_task_id == task_id)
            .values(
                status=status,
                progress=progress,
                result=result,
                error_message=error_message,
                # started_at / finished_at 字段建议在模型中通过默认值或触发器处理
            )
        )
        await session.execute(stmt)
        await session.commit()


@celery_task(name="app.workers.document_worker.process_document", queue="default")
def process_document(file_path: str, collection_name: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    文档处理 Celery 任务：
    - 调用 document_service 完整处理流程
    - 将任务状态更新到 PostgreSQL
    - 将进度（0-100）写入 Redis
    """

    async def _run() -> Dict[str, Any]:
        start_time = time.perf_counter()

        # 进度 0：任务创建
        from celery import current_task

        task_id = current_task.request.id if current_task else None  # type: ignore[attr-defined]
        if task_id:
            await _set_progress(task_id, 0)
            await _update_task_status_in_db(
                task_id=task_id,
                user_id=user_id,
                status="STARTED",
                progress=0,
            )

        # 延迟导入，避免在服务尚未实现时导致 worker 无法启动
        from app.services import document_service  # type: ignore

        # 约定 document_service 提供 async process_document(...)
        if not hasattr(document_service, "process_document"):
            raise RuntimeError("document_service.process_document is not implemented yet")

        async def progress_callback(pct: int) -> None:
            """供 document_service 在管线中调用的进度回调。"""
            if task_id:
                await _set_progress(task_id, pct)
                await _update_task_status_in_db(
                    task_id=task_id,
                    user_id=user_id,
                    status="RUNNING",
                    progress=pct,
                )

        report = await document_service.process_document(  # type: ignore[attr-defined]
            file_path=file_path,
            collection_name=collection_name,
            user_id=user_id,
            progress_callback=progress_callback,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # 进度 100：任务完成
        if task_id:
            await _set_progress(task_id, 100)
            await _update_task_status_in_db(
                task_id=task_id,
                user_id=user_id,
                status="SUCCESS",
                progress=100,
                result=report,
            )

        return {
            "ok": True,
            "file_path": file_path,
            "collection_name": collection_name,
            "user_id": user_id,
            "report": report,
            "duration_ms": duration_ms,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        # 发生异常时记录失败状态和错误信息
        from celery import current_task

        task_id = None
        try:
            task_id = current_task.request.id  # type: ignore[attr-defined]
        except Exception:
            pass

        async def _on_error() -> None:
            if task_id:
                await _set_progress(task_id, 100)
                await _update_task_status_in_db(
                    task_id=task_id,
                    user_id=user_id,
                    status="FAILURE",
                    progress=100,
                    error_message=str(exc),
                )

        asyncio.run(_on_error())
        # 将异常继续抛出，交给 LoggedRetryTask 处理重试与日志
        raise


__all__ = ["process_document"]

