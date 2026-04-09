"""
@File       : document_worker.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: 文档处理 Celery Worker。
@Features:
  - process_document(task_id, file_path, collection_name, user_id)
      · 调用 document_service 完整处理流水线
        （解析 → 分块 → Embedding → Milvus 存储）
      · 进度报告（0-100%）实时写入 Redis doc:progress:{task_id}
      · 完成/失败后更新 PostgreSQL AgentTask 记录
      · 通过 Redis Pub/Sub 推送阶段性状态事件（与 WebSocket 层对接）
  - 进度分段设计（类比「快递物流节点」）：
      0%   → 任务开始，文件解析中
      20%  → 解析完成，开始分块
      40%  → 分块完成，Embedding 生成中
      80%  → Embedding 完成，写入 Milvus
      100% → 入库完成，任务成功

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from celery import Task

from infrastructure.task_queue import (  # type: ignore[import]
    QUEUE_DEFAULT,
    celery_task,
)

logger = logging.getLogger(__name__)

# ── Redis Key 约定（与 api/agent.py 保持完全一致）─────────────────────────────
_KEY_STATUS  = "agent:status:{}"
_KEY_RESULT  = "agent:result:{}"
_KEY_PROGRESS = "doc:progress:{}"      # 文档处理专属进度 key
_CHANNEL     = "agent:events:{}"
_RESULT_TTL  = 86400 * 3               # 3 天


# ─────────────────────────────────────────────
# 1. 辅助：异步桥接（Worker 运行在同步 Celery 上下文）
# ─────────────────────────────────────────────

def _run(coro):
    """
    在 Celery Worker（同步上下文）中运行异步协程。

    每个 Worker 子进程维护自己的事件循环，避免多进程共享同一循环。
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop closed")
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # 不关闭 loop，留给同进程后续任务复用
            pass


# ─────────────────────────────────────────────
# 2. Redis 辅助函数（同步包装）
# ─────────────────────────────────────────────

def _redis_set(key: str, value: str, ttl: int = _RESULT_TTL) -> None:
    """同步写入 Redis 字符串值（含 TTL）。"""
    async def _go():
        from infrastructure.redis_client import get_redis  # type: ignore[import]
        redis = await get_redis()
        await redis.set(key, value, ex=ttl)
    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_redis_set 失败（非致命）| key=%s | error=%s", key, exc)


def _redis_publish(channel: str, payload: str) -> None:
    """同步向 Redis Pub/Sub 频道发布消息。"""
    async def _go():
        from infrastructure.redis_client import get_redis  # type: ignore[import]
        redis = await get_redis()
        await redis.publish(channel, payload)
    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_redis_publish 失败（非致命）| channel=%s | error=%s", channel, exc)


def _update_progress(task_id: str, progress: int, message: str = "") -> None:
    """
    更新文档处理进度，同时向 Redis Pub/Sub 频道推送进度事件。

    两个写入目标：
      doc:progress:{task_id}   → 供 get_task_status() 轮询读取
      agent:events:{task_id}   → 供 WebSocket 订阅者实时接收
    """
    _redis_set(_KEY_PROGRESS.format(task_id), str(progress), ttl=_RESULT_TTL)

    event = {
        "event_id":  f"doc-prog-{task_id}-{progress}",
        "task_id":   task_id,
        "step_type": "node_end",
        "node_name": "document_processor",
        "content":   message or f"文档处理进度 {progress}%",
        "metadata":  {"progress": progress},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _redis_publish(_CHANNEL.format(task_id), json.dumps(event, ensure_ascii=False))


def _save_result_snapshot(task_id: str, snapshot: dict) -> None:
    """将完整任务快照写入 Redis（供 /agent/status 接口读取）。"""
    _redis_set(
        _KEY_RESULT.format(task_id),
        json.dumps(snapshot, default=str, ensure_ascii=False),
    )
    _redis_set(_KEY_STATUS.format(task_id), snapshot.get("status", "unknown"))


def _publish_done_or_error(task_id: str, success: bool, payload: dict) -> None:
    """推送任务完成或失败的终态事件。"""
    event = {
        "event_id":  f"doc-final-{task_id}",
        "task_id":   task_id,
        "step_type": "done" if success else "error",
        "node_name": "document_processor",
        "content":   "文档处理完成" if success else f"文档处理失败：{payload.get('error', '')}",
        "result":    payload.get("report") if success else None,
        "error":     payload.get("error") if not success else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _redis_publish(_CHANNEL.format(task_id), json.dumps(event, ensure_ascii=False))


# ─────────────────────────────────────────────
# 3. PostgreSQL 辅助（同步包装）
# ─────────────────────────────────────────────

def _pg_mark_started(task_id: str) -> None:
    """将 AgentTask 记录的状态更新为 RUNNING。"""
    async def _go():
        from sqlalchemy import select
        from infrastructure.postgres_client import db_session  # type: ignore[import]
        from models.task import AgentTask                      # type: ignore[import]
        async with db_session() as session:
            result = await session.execute(
                select(AgentTask).where(AgentTask.task_id == task_id)
            )
            task = result.scalar_one_or_none()
            if task:
                task.mark_started()
    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_pg_mark_started 失败（非致命）| task_id=%s | error=%s", task_id, exc)


def _pg_mark_completed(task_id: str, result: dict) -> None:
    """将 AgentTask 记录的状态更新为 COMPLETED，写入结果。"""
    async def _go():
        from sqlalchemy import select
        from infrastructure.postgres_client import db_session  # type: ignore[import]
        from models.task import AgentTask                      # type: ignore[import]
        async with db_session() as session:
            res = await session.execute(
                select(AgentTask).where(AgentTask.task_id == task_id)
            )
            task = res.scalar_one_or_none()
            if task:
                task.mark_completed(result=result)
    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_pg_mark_completed 失败（非致命）| task_id=%s | error=%s", task_id, exc)


def _pg_mark_failed(task_id: str, error: str) -> None:
    """将 AgentTask 记录的状态更新为 FAILED。"""
    async def _go():
        from sqlalchemy import select
        from infrastructure.postgres_client import db_session  # type: ignore[import]
        from models.task import AgentTask                      # type: ignore[import]
        async with db_session() as session:
            res = await session.execute(
                select(AgentTask).where(AgentTask.task_id == task_id)
            )
            task = res.scalar_one_or_none()
            if task:
                task.mark_failed(error=error)
    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_pg_mark_failed 失败（非致命）| task_id=%s | error=%s", task_id, exc)


# ─────────────────────────────────────────────
# 4. process_document — 主任务函数
# ─────────────────────────────────────────────

@celery_task(queue=QUEUE_DEFAULT, max_retries=3)
def process_document(
    self:            Task,
    task_id:         str,
    file_path:       str,
    collection_name: str,
    user_id:         Optional[str] = None,
) -> dict:
    """
    文档处理 Celery 任务（同步包装异步流水线）。

    完整流水线：
        解析（20%）→ 分块（40%）→ Embedding（80%）→ Milvus入库（100%）

    每个阶段完成后：
        1. 更新 Redis doc:progress:{task_id}（供轮询）
        2. 向 agent:events:{task_id} 发布进度事件（供 WebSocket 推送）

    Args:
        task_id:         业务任务 ID（与 PostgreSQL AgentTask.task_id 对应）
        file_path:       文档文件路径（MinIO 路径或本地路径）
        collection_name: 目标 Milvus Collection 名称
        user_id:         提交任务的用户 ID（用于审计）

    Returns:
        处理报告字典（total_chunks / stored / failed / duration_ms）
    """
    start_time = time.monotonic()
    celery_task_id = self.request.id

    logger.info(
        "process_document 开始 | task_id=%s | file=%s | collection=%s",
        task_id, file_path, collection_name,
    )

    # ── 初始化：更新 PostgreSQL + Redis 状态 ─────────────────────────────────
    _pg_mark_started(task_id)
    _update_progress(task_id, 0, f"开始处理文档：{file_path}")
    _save_result_snapshot(task_id, {
        "task_id":   task_id,
        "status":    "running",
        "progress":  0,
        "file_path": file_path,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    report = {}

    try:
        # ── 阶段 1：文档解析（0% → 20%）──────────────────────────────────
        _update_progress(task_id, 5, "正在解析文档格式…")

        async def _parse():
            from services.document_service import DocumentService  # type: ignore[import]
            svc = DocumentService()
            return await svc.parse_document(file_path)

        raw_chunks = _run(_parse())
        _update_progress(task_id, 20, f"解析完成，共 {len(raw_chunks)} 个原始片段")

        # ── 阶段 2：文本分块（20% → 40%）─────────────────────────────────
        _update_progress(task_id, 25, "正在分块处理…")

        async def _split():
            from services.document_service import DocumentService  # type: ignore[import]
            svc = DocumentService()
            return await svc.split_chunks(raw_chunks)

        chunks = _run(_split())
        _update_progress(task_id, 40, f"分块完成，共 {len(chunks)} 个文本块")

        # ── 阶段 3：Embedding 生成（40% → 80%）────────────────────────────
        # 批量 Embedding 耗时最长，提供细粒度子进度
        total_chunks = len(chunks)
        _update_progress(task_id, 45, f"开始生成 Embedding（共 {total_chunks} 块）…")

        batch_size = 20
        embeddings_done = 0

        async def _embed_batch(batch):
            from infrastructure.embedding_client import EmbeddingClient  # type: ignore[import]
            client = EmbeddingClient()
            texts = [c.page_content for c in batch]
            return await client.embed_batch(texts)

        all_embeddings = []
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i: i + batch_size]
            batch_embeddings = _run(_embed_batch(batch))
            all_embeddings.extend(batch_embeddings)
            embeddings_done += len(batch)

            # 子进度：40% → 80% 线性映射
            sub_progress = 40 + int((embeddings_done / total_chunks) * 40)
            _update_progress(
                task_id, sub_progress,
                f"Embedding 进度：{embeddings_done}/{total_chunks}"
            )

        _update_progress(task_id, 80, "Embedding 生成完成，正在写入向量数据库…")

        # ── 阶段 4：Milvus 入库（80% → 100%）─────────────────────────────
        async def _store():
            from services.vector_store import VectorStore  # type: ignore[import]
            from langchain_core.documents import Document  # type: ignore[import]

            vs = VectorStore()
            docs = []
            for chunk, emb in zip(chunks, all_embeddings):
                docs.append(Document(
                    page_content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "source":     file_path,
                        "user_id":    user_id or "",
                        "task_id":    task_id,
                    }
                ))
            inserted_ids = await vs.upsert_documents(
                docs, collection_name, embeddings=all_embeddings
            )
            return len(inserted_ids)

        stored_count = _run(_store())
        elapsed_ms = (time.monotonic() - start_time) * 1000

        report = {
            "total_chunks": total_chunks,
            "stored":       stored_count,
            "failed":       total_chunks - stored_count,
            "duration_ms":  elapsed_ms,
            "file_path":    file_path,
            "collection":   collection_name,
        }

        # ── 成功：更新所有状态 ─────────────────────────────────────────────
        _update_progress(task_id, 100, f"文档处理完成！共写入 {stored_count} 个向量块")

        finished_at = datetime.now(timezone.utc).isoformat()
        final_snapshot = {
            "task_id":     task_id,
            "status":      "completed",
            "progress":    100,
            "result":      report,
            "finished_at": finished_at,
        }
        _save_result_snapshot(task_id, final_snapshot)
        _pg_mark_completed(task_id, result=report)
        _publish_done_or_error(task_id, success=True, payload={"report": report})

        logger.info(
            "process_document 成功 | task_id=%s | stored=%d | elapsed=%.0fms",
            task_id, stored_count, elapsed_ms,
        )
        return report

    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        error_msg  = f"{type(exc).__name__}: {exc}"

        logger.error(
            "process_document 失败 | task_id=%s | elapsed=%.0fms | error=%s",
            task_id, elapsed_ms, error_msg, exc_info=True,
        )

        failed_snapshot = {
            "task_id":     task_id,
            "status":      "failed",
            "progress":    -1,
            "error":       error_msg,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_result_snapshot(task_id, failed_snapshot)
        _pg_mark_failed(task_id, error=error_msg)
        _publish_done_or_error(task_id, success=False, payload={"error": error_msg})

        raise   # 重新抛出，让 @celery_task 装饰器决定是否重试
