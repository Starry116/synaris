"""
@File       : agent_worker.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Agent 任务 Celery Worker。
@Features:
  - run_agent_task(task_id, task_config, session_id)
      · 根据 task_config["mode"] 路由到单 Agent（workflow）或多 Agent（supervisor）
      · 通过 LangGraph astream_events() 流式捕获每个节点执行事件
      · 每节点完成后向 Redis Pub/Sub agent:events:{task_id} 发布 AgentStepEvent
      · WebSocket 订阅者（FastAPI 主进程）实时转发给前端，实现跨进程实时推送
      · 任务完成后写入 PostgreSQL AgentTask 记录（永久归档）
      · 任务完成后写入 Redis agent:result:{task_id}（供 /status 轮询）

  ── 跨进程事件流通道架构 ──────────────────────────────────────────────────
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  Celery Worker 进程                    FastAPI 进程           │
  │  ─────────────────                    ─────────────           │
  │  run_agent_task()                     WS /agent/stream/{id}  │
  │    │                                       │                  │
  │    ├─ LangGraph.astream_events()           │                  │
  │    │   on_chain_end (planner)  ────PUBLISH─▶ SUBSCRIBE        │
  │    │   on_chain_end (executor) ────PUBLISH─▶   ↓              │
  │    │   on_chain_end (observer) ────PUBLISH─▶  send_text()     │
  │    │   ...                                   (to browser)     │
  │    │                                                          │
  │    └─ 任务完成                                                  │
  │        ├─ 写 Postgres                                          │
  │        └─ 写 Redis result + PUBLISH(done event)               │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘

  关键点：两个进程完全独立，唯一的实时通道是 Redis Pub/Sub。
  Worker 只管 PUBLISH，FastAPI 只管 SUBSCRIBE + 转发，职责清晰分离。

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
from typing import Any, Optional

from celery import Task

from infrastructure.task_queue import (  # type: ignore[import]
    QUEUE_HIGH,
    celery_task,
)

logger = logging.getLogger(__name__)

# ── Redis Key 约定（与 api/agent.py 完全一致，任何一处修改需同步）────────────
_KEY_STATUS  = "agent:status:{}"
_KEY_RESULT  = "agent:result:{}"
_CHANNEL     = "agent:events:{}"
_RESULT_TTL  = 86400 * 3   # 3 天


# ─────────────────────────────────────────────
# 1. 事件循环桥接（同步 Worker 调用异步代码）
# ─────────────────────────────────────────────

def _run(coro):
    """
    在 Celery Worker 同步上下文中执行异步协程。

    每个 Worker 子进程拥有独立的事件循环实例（避免多进程间循环共享导致的竞态）。
    循环不在函数内关闭，留给同进程后续任务复用，减少创建开销。
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop is closed")
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


# ─────────────────────────────────────────────
# 2. Redis 辅助（同步包装）
# ─────────────────────────────────────────────

def _redis_set(key: str, value: str, ttl: int = _RESULT_TTL) -> None:
    async def _go():
        from infrastructure.redis_client import get_redis  # type: ignore[import]
        r = await get_redis()
        await r.set(key, value, ex=ttl)
    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_redis_set 失败 | key=%s | error=%s", key, exc)


def _redis_publish(channel: str, payload: str) -> None:
    async def _go():
        from infrastructure.redis_client import get_redis  # type: ignore[import]
        r = await get_redis()
        await r.publish(channel, payload)
    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_redis_publish 失败 | channel=%s | error=%s", channel, exc)


def _publish_event(task_id: str, event_dict: dict) -> None:
    """序列化并发布一条 AgentStepEvent 到 Redis Pub/Sub 频道。"""
    _redis_publish(
        _CHANNEL.format(task_id),
        json.dumps(event_dict, default=str, ensure_ascii=False),
    )


def _make_event(
    task_id:   str,
    step_type: str,
    node_name: str,
    content:   str = "",
    **extra:   Any,
) -> dict:
    """构造 AgentStepEvent 字典（与 schemas/agent.py 的 AgentStepEvent 格式完全对齐）。"""
    evt = {
        "event_id":  f"evt-{task_id[:8]}-{node_name}-{int(time.time()*1000)}",
        "task_id":   task_id,
        "step_type": step_type,
        "node_name": node_name,
        "content":   content[:300] if content else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    evt.update(extra)
    return evt


def _save_result_snapshot(task_id: str, snapshot: dict) -> None:
    """持久化完整任务快照到 Redis（供 GET /agent/status 轮询）。"""
    _redis_set(
        _KEY_RESULT.format(task_id),
        json.dumps(snapshot, default=str, ensure_ascii=False),
    )
    _redis_set(_KEY_STATUS.format(task_id), snapshot.get("status", "unknown"))


# ─────────────────────────────────────────────
# 3. PostgreSQL 辅助（同步包装）
# ─────────────────────────────────────────────

def _pg_update(task_id: str, action: str, **kwargs: Any) -> None:
    """
    更新 PostgreSQL AgentTask 记录。

    action: "started" / "completed" / "failed"
    """
    async def _go():
        from sqlalchemy import select
        from infrastructure.postgres_client import db_session  # type: ignore[import]
        from models.task import AgentTask                      # type: ignore[import]

        async with db_session() as session:
            result = await session.execute(
                select(AgentTask).where(AgentTask.task_id == task_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                logger.warning("_pg_update: 未找到 task_id=%s 的 AgentTask 记录", task_id)
                return

            if action == "started":
                task.mark_started()
                if "celery_task_id" in kwargs:
                    task.celery_task_id = kwargs["celery_task_id"]
            elif action == "completed":
                task.mark_completed(
                    result=kwargs.get("result", {}),
                    tokens_used=kwargs.get("tokens_used", 0),
                )
                if "step_log" in kwargs:
                    task.step_log = kwargs["step_log"]
            elif action == "failed":
                task.mark_failed(error=kwargs.get("error", "未知错误"))

    try:
        _run(_go())
    except Exception as exc:
        logger.warning("_pg_update 失败（非致命）| task_id=%s | action=%s | error=%s",
                       task_id, action, exc)


# ─────────────────────────────────────────────
# 4. LangGraph 事件解析与 Pub/Sub 发布
# ─────────────────────────────────────────────

def _extract_content(node_name: str, output: dict) -> str:
    """
    从 LangGraph 节点输出中提取可读摘要（与 api/agent.py 的同名函数逻辑保持一致）。

    两处存在相同逻辑是有意为之：
    - Worker 在 Celery 进程中发布原始摘要
    - FastAPI 层收到后可以二次展示，但不依赖 FastAPI 做解析
    Worker 必须能独立完成事件内容的生成，不能依赖 FastAPI 进程的函数。
    """
    if node_name == "planner":
        plan = output.get("plan", [])
        return f"生成执行计划：{len(plan)} 步" if plan else "Planner 已完成规划"

    if node_name == "tool_selector":
        pending = output.get("metadata", {}).get("pending_tool", {})
        tool    = pending.get("name", "none")
        reason  = pending.get("reasoning", "")
        return f"选择工具：{tool}｜{reason}"[:300]

    if node_name == "tool_executor":
        results = output.get("tool_results", [])
        if results:
            last = results[-1]
            tool_name = last.get("tool", "?")
            if last.get("error"):
                return f"工具 [{tool_name}] 失败：{last['error']}"[:300]
            return f"工具 [{tool_name}] 完成，输出长度 {len(str(last.get('output', '')))} 字"
        return "工具执行完成"

    if node_name == "observer":
        decision = output.get("metadata", {}).get("observer_decision", "")
        reasoning = output.get("metadata", {}).get("observer_reasoning", "")
        return f"决策：{decision}｜{reasoning}"[:300]

    messages = output.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "content", "") or last_msg.get("content", "")
        return str(content)[:300]

    return f"节点 [{node_name}] 执行完成"


async def _stream_single_agent(
    task_id:    str,
    task_config: dict,
    session_id: str,
) -> dict:
    """
    驱动单 Agent LangGraph 工作流，流式捕获节点事件并实时发布到 Redis。

    Returns:
        最终状态字典（含 final_answer / status / tool_results 等）
    """
    from agents.state import AgentConfig, AgentMode, TaskStatus, initial_state  # type: ignore[import]
    from agents.workflow import _compiled_graph                                  # type: ignore[import]
    from langchain_core.runnables import RunnableConfig                          # type: ignore[import]

    # ── 构建初始状态 ───────────────────────────────────────────────────────
    cfg_dict = task_config.get("config", {})
    agent_cfg = AgentConfig(
        mode=task_config.get("mode", AgentMode.SINGLE),
        **{k: v for k, v in cfg_dict.items()
           if k in ("max_iterations", "timeout_seconds", "enable_human_loop", "allowed_tools")},
    )

    init_state = initial_state(
        task=task_config["task"],
        config=agent_cfg,
        session_id=session_id,
        user_id=task_config.get("user_id"),
    )

    run_config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": agent_cfg.max_iterations * 4 + 10,
    }

    steps: list[dict] = []

    # ── 通知 Worker 已开始 ─────────────────────────────────────────────────
    _publish_event(task_id, _make_event(
        task_id, "node_start", "agent_worker",
        content="Agent 任务开始执行",
    ))

    # ── 流式遍历 LangGraph 事件 ───────────────────────────────────────────
    async for raw_event in _compiled_graph.astream_events(
        init_state, config=run_config, version="v2",
    ):
        event_name: str  = raw_event.get("event", "")
        node_name:  str  = raw_event.get("name", "unknown")
        data:       dict = raw_event.get("data", {})

        # 过滤框架内部节点（不向前端展示）
        if node_name in ("LangGraph", "__start__", "__end__"):
            continue

        if event_name == "on_chain_start":
            _publish_event(task_id, _make_event(
                task_id, "node_start", node_name,
                content=f"节点 [{node_name}] 开始执行",
            ))

        elif event_name == "on_chain_end":
            output  = data.get("output", {})
            content = _extract_content(node_name, output)

            # 若有工具调用，额外发布 tool_call 事件
            tool_results = output.get("tool_results", [])
            if tool_results:
                last = tool_results[-1]
                _publish_event(task_id, _make_event(
                    task_id, "tool_call", node_name,
                    content=str(last.get("output", ""))[:300],
                    tool_name=last.get("tool"),
                    elapsed_ms=last.get("elapsed_ms"),
                ))

            # 若触发 Human-in-the-Loop 中断
            interrupt = output.get("interrupt")
            if interrupt and output.get("status") == "waiting_human":
                q = getattr(interrupt, "question", "") or (interrupt or {}).get("question", "")
                opts = getattr(interrupt, "options", []) or []
                _publish_event(task_id, _make_event(
                    task_id, "interrupt", node_name,
                    content=f"需要人工确认：{q}",
                    interrupt={"question": q, "options": opts},
                ))
                _redis_set(_KEY_STATUS.format(task_id), "waiting_human")

            # node_end 事件
            _publish_event(task_id, _make_event(
                task_id, "node_end", node_name,
                content=content,
            ))

            # 记录到 steps（用于 PostgreSQL step_log）
            steps.append({
                "node_name":  node_name,
                "status":     "failed" if output.get("status") == "failed" else "completed",
                "content":    content[:300],
                "tool_name":  tool_results[-1].get("tool") if tool_results else None,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            })

    # ── 获取最终状态 ────────────────────────────────────────────────────
    final_snapshot = await _compiled_graph.aget_state(run_config)
    final_values   = final_snapshot.values if final_snapshot else {}

    return {
        "final_values": final_values,
        "steps":        steps,
    }


async def _run_multi_agent(
    task_id:     str,
    task_config: dict,
    session_id:  str,
) -> dict:
    """
    驱动多 Agent Supervisor 工作流。

    Supervisor 整体执行完后一次性发布 done 事件（不支持流式节点事件）。
    """
    from agents.supervisor import run_supervisor  # type: ignore[import]

    _publish_event(task_id, _make_event(
        task_id, "node_start", "supervisor",
        content="多 Agent 协作任务开始",
    ))

    final = await run_supervisor(
        task=task_config["task"],
        session_id=session_id,
        user_context=task_config.get("user_context", ""),
    )

    return {
        "final_values": final,
        "steps": [],
    }


# ─────────────────────────────────────────────
# 5. run_agent_task — 主任务函数
# ─────────────────────────────────────────────

@celery_task(queue=QUEUE_HIGH, max_retries=2)
def run_agent_task(
    self:        Task,
    task_id:     str,
    task_config: dict,
    session_id:  str,
) -> dict:
    """
    Agent 任务 Celery Worker（高优先级队列）。

    职责：
      1. 根据 task_config["mode"] 路由到单 Agent 或多 Agent 工作流
      2. 流式捕获 LangGraph 节点事件，实时 PUBLISH 到 Redis Pub/Sub
      3. 任务结束后写入 PostgreSQL（永久归档）和 Redis（供轮询）

    Args:
        task_id:     业务任务 ID（与 PostgreSQL AgentTask.task_id 对应）
        task_config: 任务配置字典（task / mode / config / user_id / user_context）
        session_id:  LangGraph 会话 ID（用于 MemorySaver 断点续跑）

    Returns:
        {"status": "completed", "final_answer": "...", "tokens_used": N}

    task_config 格式：
        {
          "task":         "请帮我分析 Q3 财报",
          "mode":         "single",       # single / multi / rag_only
          "user_id":      "uuid-str",
          "user_context": "销售总监",
          "config": {
            "max_iterations": 10,
            "enable_human_loop": false,
            "allowed_tools": ["rag_retrieval", "web_search"]
          }
        }
    """
    start_time     = time.monotonic()
    celery_task_id = self.request.id
    mode           = task_config.get("mode", "single")

    logger.info(
        "run_agent_task 开始 | task_id=%s | mode=%s | session_id=%s",
        task_id, mode, session_id,
    )

    # ── 初始化状态 ─────────────────────────────────────────────────────────
    _pg_update(task_id, "started", celery_task_id=celery_task_id)
    _save_result_snapshot(task_id, {
        "task_id":    task_id,
        "session_id": session_id,
        "status":     "running",
        "mode":       mode,
        "task":       task_config.get("task", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    _redis_set(_KEY_STATUS.format(task_id), "running")

    try:
        # ── 路由到对应工作流 ───────────────────────────────────────────────
        if mode == "multi":
            execution_result = _run(
                _run_multi_agent(task_id, task_config, session_id)
            )
        else:
            execution_result = _run(
                _stream_single_agent(task_id, task_config, session_id)
            )

        final_values = execution_result.get("final_values", {})
        steps        = execution_result.get("steps", [])

        # ── 提取最终结果字段 ───────────────────────────────────────────────
        # 单 Agent → final_answer；多 Agent → final_output
        if mode == "multi":
            final_answer = final_values.get("final_output", "")
        else:
            final_answer = final_values.get("final_answer", "")

        final_status_raw = final_values.get("status", "completed")
        if hasattr(final_status_raw, "value"):
            final_status = final_status_raw.value
        else:
            final_status = str(final_status_raw)

        error_msg    = final_values.get("error")
        tokens_used  = final_values.get("metadata", {}).get("tokens_used", 0)
        elapsed_ms   = (time.monotonic() - start_time) * 1000
        finished_at  = datetime.now(timezone.utc).isoformat()

        # ── 构建完整结果 ───────────────────────────────────────────────────
        result_dict = {
            "final_answer":  final_answer,
            "tool_results":  final_values.get("tool_results", []),
            "iteration_count": final_values.get("iteration_count", 0),
            "plan":          final_values.get("plan", []),
        }
        if mode == "multi":
            result_dict["worker_results"] = [
                r.model_dump() if hasattr(r, "model_dump") else dict(r)
                for r in (final_values.get("worker_results") or [])
            ]

        # ── 持久化到 Redis ─────────────────────────────────────────────────
        final_snapshot = {
            "task_id":      task_id,
            "session_id":   session_id,
            "status":       final_status,
            "mode":         mode,
            "task":         task_config.get("task", ""),
            "result":       final_answer or None,
            "steps":        steps,
            "tokens_used":  tokens_used,
            "duration_ms":  elapsed_ms,
            "error":        error_msg,
            "started_at":   final_values.get("metadata", {}).get("created_at"),
            "finished_at":  finished_at,
        }
        _save_result_snapshot(task_id, final_snapshot)

        # ── 持久化到 PostgreSQL ────────────────────────────────────────────
        _pg_update(
            task_id, "completed",
            result=result_dict,
            tokens_used=tokens_used,
            step_log=steps,
        )

        # ── 发布终态事件 ───────────────────────────────────────────────────
        if final_status == "completed":
            _publish_event(task_id, _make_event(
                task_id, "done", "agent_worker",
                content="任务完成",
                result=final_answer,
            ))
        else:
            _publish_event(task_id, _make_event(
                task_id, "error", "agent_worker",
                content=f"任务失败：{error_msg or '未知错误'}",
                error=error_msg or "任务执行失败",
            ))

        logger.info(
            "run_agent_task 完成 | task_id=%s | status=%s | tokens=%d | elapsed=%.0fms",
            task_id, final_status, tokens_used, elapsed_ms,
        )
        return {"status": final_status, "final_answer": final_answer, "tokens_used": tokens_used}

    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        error_msg  = f"{type(exc).__name__}: {exc}"

        logger.error(
            "run_agent_task 异常 | task_id=%s | elapsed=%.0fms | error=%s",
            task_id, elapsed_ms, error_msg, exc_info=True,
        )

        # 持久化失败状态
        failed_snapshot = {
            "task_id":     task_id,
            "session_id":  session_id,
            "status":      "failed",
            "mode":        mode,
            "task":        task_config.get("task", ""),
            "error":       error_msg,
            "duration_ms": elapsed_ms,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_result_snapshot(task_id, failed_snapshot)
        _pg_update(task_id, "failed", error=error_msg)
        _publish_event(task_id, _make_event(
            task_id, "error", "agent_worker",
            content=f"任务失败：{error_msg}",
            error=error_msg,
        ))

        raise   # 让 @celery_task 装饰器决定是否重试
