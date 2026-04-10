"""
@File       : agent.py  (api/)
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Agent API 路由层，提供任务提交、状态查询、取消、WebSocket 推送。
@Features:
  - POST /agent/run
      · 异步启动 Agent 任务（asyncio.create_task，后台运行）
      · 立即返回 task_id，客户端通过 WebSocket 或轮询获取进展
      · ⚠️ Step 20 完成后，此处将替换为 Celery 任务提交
  - GET /agent/status/{task_id}
      · 从 Redis 读取任务快照，返回完整 AgentStatusResponse
      · 包含执行轨迹（steps）、Token 消耗、最终结果
  - POST /agent/cancel/{task_id}
      · 取消正在运行的 asyncio.Task
      · 已完成/不存在的任务返回 cancelled=false
  - POST /agent/resume/{task_id}
      · Human-in-the-Loop 恢复接口：用户填写确认内容后调用
      · 内部调用 resume_workflow()，从断点继续执行
  - WebSocket /agent/stream/{task_id}
      · 客户端连接后，实时接收 AgentStepEvent JSON 消息
      · 通过 Redis Pub/Sub 订阅 agent:events:{task_id} 频道
      · 收到 done / error 事件后自动关闭连接
      · 支持重连：若任务已完成，立即推送历史事件 + done

  ──────────────────────────────────────────────────────────────────
  数据流设计：

   客户端                 FastAPI                Redis              LangGraph
     │                      │                      │                    │
     │ POST /agent/run       │                      │                    │
     │──────────────────────▶│  asyncio.create_task │                    │
     │                      │──────────────────────────────────────────▶│
     │◀─ {task_id, pending}  │                      │  astream_events()  │
     │                      │                      │◀──── step events ──│
     │ WS /agent/stream/id   │                      │                    │
     │──────────────────────▶│                      │                    │
     │                      │── SUBSCRIBE ─────────▶│                    │
     │◀── AgentStepEvent ────│◀── PUBLISH ──────────│                    │
     │◀── AgentStepEvent ────│                      │                    │
     │◀── {done, result} ────│                      │                    │
     │                        │                      │                    │
  ──────────────────────────────────────────────────────────────────

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-01  Starry  Initial creation
    ⚠️  Step 20 补丁：POST /agent/run 改为提交 Celery 任务（替换 asyncio.create_task）
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from agents.state import AgentConfig, AgentMode, TaskStatus
from agents.workflow import resume_workflow, run_workflow
from agents.supervisor import run_supervisor
from schemas.agent import (
    AgentCancelResponse,
    AgentRunRequest,
    AgentRunResponse,
    AgentStatusResponse,
    AgentStepEvent,
    HumanResumeRequest,
    StepEventType,
    StepSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

# ── Redis 频道与键名约定 ────────────────────────────────────────────────────────
# agent:status:{task_id}   → TaskStatus 字符串
# agent:result:{task_id}   → JSON 序列化的任务快照（AgentStatusResponse）
# agent:events:{task_id}   → Pub/Sub 频道，推送 AgentStepEvent JSON
_KEY_STATUS  = "agent:status:{}"
_KEY_RESULT  = "agent:result:{}"
_CHANNEL     = "agent:events:{}"
_RESULT_TTL  = 86400 * 3   # 结果保留 3 天（秒）
_WS_POLL_MS  = 100         # WebSocket 轮询 Pub/Sub 的间隔（毫秒）
_WS_TIMEOUT  = 3600        # WebSocket 最长保持连接时间（秒）

# ── 活跃任务注册表（task_id → asyncio.Task）──────────────────────────────────
# ⚠️ Step 20 后，此注册表替换为 Celery 任务 ID 映射
_active_tasks: dict[str, asyncio.Task] = {}


# ─────────────────────────────────────────────
# 1. Redis 辅助函数
# ─────────────────────────────────────────────

async def _get_redis():
    """获取 Redis 客户端（延迟导入，Step 3 完成后可用）。"""
    try:
        from infrastructure.redis_client import get_redis  # type: ignore[import]
        return await get_redis()
    except ImportError:
        logger.warning("redis_client 未就绪，使用内存字典降级（仅开发环境）")
        return _MemoryFallback()


class _MemoryFallback:
    """
    Redis 不可用时的内存降级实现（仅开发/测试环境）。
    生产环境必须使用真实 Redis。
    """
    _store: dict[str, str] = {}
    _pubsub_queue: dict[str, list[str]] = {}

    async def set(self, key: str, value: str, ex: int = None) -> None:
        self._store[key] = value

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def publish(self, channel: str, message: str) -> None:
        self._pubsub_queue.setdefault(channel, []).append(message)

    def pubsub(self) -> "_MemoryPubSub":
        return _MemoryPubSub(self._pubsub_queue)


class _MemoryPubSub:
    def __init__(self, queue: dict[str, list[str]]):
        self._queue = queue
        self._channel: Optional[str] = None

    async def subscribe(self, channel: str) -> None:
        self._channel = channel
        self._queue.setdefault(channel, [])

    async def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 0.1):
        if self._channel and self._queue.get(self._channel):
            data = self._queue[self._channel].pop(0)
            return {"type": "message", "data": data.encode()}
        return None

    async def unsubscribe(self, channel: str = None) -> None:
        pass

    async def aclose(self) -> None:
        pass


async def _set_status(task_id: str, status: TaskStatus) -> None:
    """将任务状态写入 Redis。"""
    redis = await _get_redis()
    await redis.set(_KEY_STATUS.format(task_id), status.value, ex=_RESULT_TTL)


async def _get_status(task_id: str) -> Optional[str]:
    """从 Redis 读取任务状态字符串。"""
    redis = await _get_redis()
    return await redis.get(_KEY_STATUS.format(task_id))


async def _save_result(task_id: str, snapshot: dict[str, Any]) -> None:
    """将完整任务快照（dict）序列化后写入 Redis。"""
    redis = await _get_redis()
    await redis.set(_KEY_RESULT.format(task_id), json.dumps(snapshot, default=str), ex=_RESULT_TTL)


async def _load_result(task_id: str) -> Optional[dict]:
    """从 Redis 加载任务快照（dict）。"""
    redis = await _get_redis()
    raw = await redis.get(_KEY_RESULT.format(task_id))
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


async def _publish_event(task_id: str, event: AgentStepEvent) -> None:
    """将 AgentStepEvent 发布到对应的 Redis Pub/Sub 频道。"""
    redis = await _get_redis()
    await redis.publish(
        _CHANNEL.format(task_id),
        event.model_dump_json(exclude_none=True),
    )


# ─────────────────────────────────────────────
# 2. 后台任务执行器（LangGraph 流式事件桥接）
# ─────────────────────────────────────────────

async def _stream_and_publish(
    task_id:    str,
    session_id: str,
    request:    AgentRunRequest,
    agent_cfg:  AgentConfig,
) -> None:
    """
    后台协程：驱动 LangGraph 执行，并将每个节点事件转化为 AgentStepEvent 推送到 Redis。

    核心机制——「事件桥」：
        LangGraph.astream_events()     →   _stream_and_publish()   →   Redis Pub/Sub
        每个节点完成产生原生 LangGraph 事件   将事件翻译为 AgentStepEvent    WebSocket 订阅后转发给客户端

    LangGraph astream_events() 返回的事件结构（v2 格式）：
        {
          "event":   "on_chain_start" | "on_chain_end" | "on_tool_start" | "on_tool_end" ...,
          "name":    节点名称（如 "planner" / "tool_executor"）,
          "data":    {"input": {...}, "output": {...}}
        }

    Step 20 补丁提示：
        此函数在 Step 20 后需整体迁移到 workers/agent_worker.py（Celery Task），
        Redis Pub/Sub 的 publish 调用保持不变，仅执行方式由 asyncio 改为 Celery。
    """
    start_time = time.monotonic()
    steps: list[StepSummary] = []
    tokens_used = 0

    try:
        await _set_status(task_id, TaskStatus.RUNNING)

        # ── 构建 LangGraph 执行配置 ────────────────────────────────────────
        run_config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": agent_cfg.max_iterations * 4 + 10,
        }

        from agents.state import initial_state, AgentMode as AM

        # ── 按模式选择工作流 ──────────────────────────────────────────────
        if request.mode == AM.MULTI:
            # 多 Agent：使用 Supervisor 子图
            # Supervisor 目前不支持 astream_events，降级为整体执行后推送 done
            await _publish_event(task_id, AgentStepEvent.make_node_start(
                task_id=task_id, node_name="supervisor",
            ))
            final = await run_supervisor(
                task=request.task,
                session_id=session_id,
                user_context=request.user_context,
                graph=None,
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000
            final_output = final.get("final_output", "")
            final_status = final.get("status", TaskStatus.FAILED)
            error = final.get("error")

            steps.append(StepSummary(
                node_name="supervisor",
                status="completed" if final_status == TaskStatus.COMPLETED else "failed",
                content=(final_output[:300] if final_output else error or "")[:300],
                elapsed_ms=elapsed_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        else:
            # 单 Agent：使用 workflow 子图，通过 astream_events 流式获取事件
            init_state = initial_state(
                task=request.task,
                config=agent_cfg,
                session_id=session_id,
            )

            from agents.workflow import _compiled_graph  # type: ignore[import]

            # LangGraph astream_events() 以 v2 格式逐事件 yield
            async for raw_event in _compiled_graph.astream_events(
                init_state, config=run_config, version="v2",
            ):
                event_name: str = raw_event.get("event", "")
                node_name:  str = raw_event.get("name", "unknown")
                data:       dict = raw_event.get("data", {})

                # ── 节点开始 ───────────────────────────────────────────
                if event_name == "on_chain_start" and node_name not in ("LangGraph", "__start__"):
                    step_event = AgentStepEvent.make_node_start(
                        task_id=task_id,
                        node_name=node_name,
                    )
                    await _publish_event(task_id, step_event)

                # ── 节点结束 ───────────────────────────────────────────
                elif event_name == "on_chain_end" and node_name not in ("LangGraph", "__start__"):
                    output = data.get("output", {})
                    node_elapsed = None

                    # 从输出中提取有意义的摘要文本
                    content = _extract_node_content(node_name, output)

                    # 检查是否有工具调用结果
                    tool_results = output.get("tool_results", [])
                    if tool_results:
                        latest = tool_results[-1]
                        tool_evt = AgentStepEvent.make_tool_call(
                            task_id=task_id,
                            node_name=node_name,
                            tool_name=latest.get("tool", "unknown"),
                            result_preview=str(latest.get("output", ""))[:300],
                            elapsed_ms=latest.get("elapsed_ms"),
                        )
                        await _publish_event(task_id, tool_evt)

                    # 检查是否触发 Human-in-the-Loop
                    interrupt = output.get("interrupt")
                    if interrupt and output.get("status") == TaskStatus.WAITING_HUMAN:
                        interrupt_evt = AgentStepEvent.make_interrupt(
                            task_id=task_id,
                            question=getattr(interrupt, "question", "请确认是否继续？"),
                            options=getattr(interrupt, "options", []),
                        )
                        await _publish_event(task_id, interrupt_evt)
                        await _set_status(task_id, TaskStatus.WAITING_HUMAN)

                    step_event = AgentStepEvent.make_node_end(
                        task_id=task_id,
                        node_name=node_name,
                        content=content,
                        elapsed_ms=node_elapsed or 0,
                    )
                    await _publish_event(task_id, step_event)

                    # 记录到 steps（排除内部框架节点）
                    if node_name not in ("LangGraph", "__start__", "__end__"):
                        node_status = "failed" if output.get("status") == TaskStatus.FAILED else "completed"
                        steps.append(StepSummary(
                            node_name=node_name,
                            status=node_status,
                            content=content[:300],
                            tool_name=tool_results[-1].get("tool") if tool_results else None,
                            elapsed_ms=node_elapsed,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ))

            # ── 流式遍历结束，获取最终状态 ─────────────────────────────
            final_snapshot = await _compiled_graph.aget_state(run_config)
            final_values   = final_snapshot.values if final_snapshot else {}
            final_status   = final_values.get("status", TaskStatus.FAILED)
            final_output   = final_values.get("final_answer", "")
            error          = final_values.get("error")

            # 统计 Token（从 metadata 中读取，Step 24 CostService 会写入）
            tokens_used = final_values.get("metadata", {}).get("tokens_used", 0)

        # ── 构建持久化快照 ────────────────────────────────────────────────
        elapsed_total_ms = (time.monotonic() - start_time) * 1000
        finished_at      = datetime.now(timezone.utc).isoformat()

        snapshot = {
            "task_id":    task_id,
            "session_id": session_id,
            "status":     (final_status.value if hasattr(final_status, "value") else str(final_status)),
            "mode":       request.mode.value if hasattr(request.mode, "value") else str(request.mode),
            "task":       request.task,
            "result":     final_output or None,
            "steps":      [s.model_dump() for s in steps],
            "tokens_used": tokens_used,
            "duration_ms": elapsed_total_ms,
            "error":       error,
            "created_at":  datetime.now(timezone.utc).isoformat(),
            "finished_at": finished_at,
        }

        await _save_result(task_id, snapshot)
        await _set_status(task_id, TaskStatus(
            final_status.value if hasattr(final_status, "value") else str(final_status)
        ))

        # ── 推送最终事件 ──────────────────────────────────────────────────
        if final_status == TaskStatus.COMPLETED:
            done_event = AgentStepEvent.make_done(
                task_id=task_id,
                result=final_output or "",
            )
            await _publish_event(task_id, done_event)
            logger.info("_stream_and_publish 完成 | task_id=%s | elapsed=%.2fs",
                        task_id, elapsed_total_ms / 1000)
        else:
            error_event = AgentStepEvent.make_error(
                task_id=task_id,
                error=error or "任务执行失败",
            )
            await _publish_event(task_id, error_event)
            logger.warning("_stream_and_publish 失败 | task_id=%s | error=%s", task_id, error)

    except asyncio.CancelledError:
        # 任务被 /cancel 接口取消
        await _set_status(task_id, TaskStatus.FAILED)
        cancel_event = AgentStepEvent.make_error(task_id=task_id, error="任务已被用户取消")
        await _publish_event(task_id, cancel_event)
        logger.info("_stream_and_publish 被取消 | task_id=%s", task_id)
        raise   # 让 asyncio.Task 感知到 CancelledError

    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error("_stream_and_publish 异常 | task_id=%s | error=%s",
                     task_id, exc, exc_info=True)
        await _set_status(task_id, TaskStatus.FAILED)
        error_event = AgentStepEvent.make_error(
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        await _publish_event(task_id, error_event)

        # 同时持久化错误快照，避免 /status 接口返回 404
        await _save_result(task_id, {
            "task_id":    task_id,
            "session_id": session_id,
            "status":     TaskStatus.FAILED.value,
            "mode":       str(request.mode),
            "task":       request.task,
            "result":     None,
            "steps":      [s.model_dump() for s in steps],
            "tokens_used": 0,
            "duration_ms": elapsed_ms,
            "error":       f"{type(exc).__name__}: {exc}",
            "created_at":  datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })

    finally:
        # 从活跃任务注册表中移除自身
        _active_tasks.pop(task_id, None)


def _extract_node_content(node_name: str, output: dict) -> str:
    """
    从 LangGraph 节点输出中提取可读的摘要文本。

    不同节点产出的「有价值信息」位置不同：
    - planner       → plan[]（执行计划步骤列表）
    - tool_selector → metadata["pending_tool"]["reasoning"]
    - tool_executor → tool_results[-1]["output"] 的前 200 字
    - observer      → metadata["observer_decision"] + reasoning
    """
    if node_name == "planner":
        plan = output.get("plan", [])
        if plan:
            return f"生成执行计划：{len(plan)} 步 → " + " | ".join(plan[:3])
        return "Planner 已生成执行计划"

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
                return f"工具 [{tool_name}] 调用失败：{last['error']}"[:300]
            preview = str(last.get("output", ""))[:200]
            return f"工具 [{tool_name}] 完成，输出：{preview}"
        return "工具执行完成"

    if node_name == "observer":
        decision = output.get("metadata", {}).get("observer_decision", "")
        reasoning = output.get("metadata", {}).get("observer_reasoning", "")
        return f"决策：{decision}｜{reasoning}"[:300]

    if node_name == "human_interrupt":
        interrupt = output.get("interrupt")
        if interrupt:
            q = getattr(interrupt, "question", "") or interrupt.get("question", "")
            return f"等待人工确认：{q}"[:300]
        return "Human-in-the-Loop 挂起"

    # 兜底：取 messages 最后一条
    messages = output.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "content", "") or last_msg.get("content", "")
        return str(content)[:300]

    return f"节点 [{node_name}] 执行完成"


# ─────────────────────────────────────────────
# 3. API 路由实现
# ─────────────────────────────────────────────

@router.post(
    "/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="异步启动 Agent 任务",
    description=(
        "提交一个 Agent 任务。服务器立即返回 task_id，"
        "任务在后台异步执行。\n\n"
        "通过 `GET /agent/status/{task_id}` 轮询，"
        "或连接 `WS /agent/stream/{task_id}` 实时接收进展。\n\n"
        "⚠️ Step 20 完成后，此接口将改为提交 Celery 任务。"
    ),
)
async def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    """
    POST /agent/run

    流程：
    1. 生成 task_id（UUID）
    2. 将初始状态写入 Redis（status=pending）
    3. 启动后台协程 _stream_and_publish（asyncio.create_task）
    4. 立即返回 AgentRunResponse（含 task_id 和 WebSocket URL）
    """
    task_id    = f"task-{uuid.uuid4().hex}"
    session_id = request.session_id or f"sess-{uuid.uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()

    # ── 构建 AgentConfig ───────────────────────────────────────────────────
    try:
        agent_cfg = AgentConfig(
            mode=request.mode,
            session_id=session_id,
            **{k: v for k, v in request.config.items()
               if k in ("max_iterations", "timeout_seconds",
                        "enable_human_loop", "allowed_tools")},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"config 参数无效：{exc}",
        )

    # ── 初始化 Redis 状态 ──────────────────────────────────────────────────
    await _set_status(task_id, TaskStatus.PENDING)

    # 写入初始快照（供 /status 接口在任务尚未完成时使用）
    await _save_result(task_id, {
        "task_id":    task_id,
        "session_id": session_id,
        "status":     TaskStatus.PENDING.value,
        "mode":       request.mode.value if hasattr(request.mode, "value") else str(request.mode),
        "task":       request.task,
        "result":     None,
        "steps":      [],
        "tokens_used": 0,
        "duration_ms": 0.0,
        "error":       None,
        "created_at":  created_at,
        "finished_at": None,
    })

    # ── 启动后台任务 ───────────────────────────────────────────────────────
    # ⚠️ Step 20 补丁：替换为 Celery 任务提交
    #   from workers.agent_worker import run_agent_task
    #   celery_task = run_agent_task.apply_async(
    #       args=[task_id, session_id, request.model_dump(), agent_cfg.model_dump()],
    #       queue="high",
    #   )
    bg_task = asyncio.create_task(
        _stream_and_publish(task_id, session_id, request, agent_cfg),
        name=f"agent-{task_id}",
    )
    _active_tasks[task_id] = bg_task

    logger.info(
        "run_agent: 任务已提交 | task_id=%s | mode=%s | session_id=%s",
        task_id, request.mode, session_id,
    )

    return AgentRunResponse(
        task_id=task_id,
        session_id=session_id,
        status=TaskStatus.PENDING,
        mode=request.mode.value if hasattr(request.mode, "value") else str(request.mode),
        message="任务已提交，正在排队执行。通过 stream_url 连接 WebSocket 获取实时进展。",
        created_at=created_at,
        stream_url=f"/agent/stream/{task_id}",
    )


@router.get(
    "/status/{task_id}",
    response_model=AgentStatusResponse,
    summary="查询 Agent 任务状态",
    description="轮询接口：返回任务的当前状态、执行轨迹、Token 消耗和最终结果。",
)
async def get_agent_status(task_id: str) -> AgentStatusResponse:
    """
    GET /agent/status/{task_id}

    状态来源：Redis key `agent:result:{task_id}`
    若 key 不存在（task_id 无效）→ 返回 404
    """
    snapshot = await _load_result(task_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在或已过期（结果保留 3 天）",
        )

    # 解析 steps 列表（从 dict 还原 StepSummary）
    raw_steps = snapshot.get("steps", [])
    steps = []
    for s in raw_steps:
        try:
            steps.append(StepSummary(**s))
        except Exception:
            pass  # 忽略格式错误的历史步骤

    # 解析 interrupt_question（status=waiting_human 时填写）
    interrupt_question = None
    if snapshot.get("status") == TaskStatus.WAITING_HUMAN.value:
        interrupt_question = snapshot.get("interrupt_question")

    return AgentStatusResponse(
        task_id=task_id,
        session_id=snapshot.get("session_id", ""),
        status=TaskStatus(snapshot.get("status", TaskStatus.PENDING.value)),
        mode=snapshot.get("mode", "single"),
        task=snapshot.get("task", ""),
        result=snapshot.get("result"),
        steps=steps,
        tokens_used=snapshot.get("tokens_used", 0),
        duration_ms=snapshot.get("duration_ms", 0.0),
        error=snapshot.get("error"),
        interrupt_question=interrupt_question,
        created_at=snapshot.get("created_at", ""),
        finished_at=snapshot.get("finished_at"),
    )


@router.post(
    "/cancel/{task_id}",
    response_model=AgentCancelResponse,
    summary="取消正在执行的 Agent 任务",
)
async def cancel_agent(task_id: str) -> AgentCancelResponse:
    """
    POST /agent/cancel/{task_id}

    取消机制：
    - 若任务仍在 _active_tasks 中（asyncio.Task 未完成）→ task.cancel()
    - 若任务已完成或不存在 → 返回 cancelled=false
    """
    task = _active_tasks.get(task_id)

    if task is None or task.done():
        # 检查 Redis 是否有记录（区分「不存在」和「已完成」）
        raw_status = await _get_status(task_id)
        if raw_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务 {task_id} 不存在",
            )
        return AgentCancelResponse(
            task_id=task_id,
            cancelled=False,
            message=f"任务已处于 {raw_status} 状态，无需取消",
        )

    # 取消 asyncio.Task（_stream_and_publish 内部会捕获 CancelledError）
    task.cancel()
    # 等待取消生效（最多 2 秒）
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    logger.info("cancel_agent: 任务已取消 | task_id=%s", task_id)

    return AgentCancelResponse(
        task_id=task_id,
        cancelled=True,
        message="任务已成功取消",
    )


@router.post(
    "/resume/{task_id}",
    response_model=AgentRunResponse,
    summary="Human-in-the-Loop：携带人工响应恢复任务",
    description=(
        "当任务状态为 `waiting_human` 时，调用此接口提交人工确认内容。\n\n"
        "服务器会从断点恢复工作流，继续后续节点的执行。"
    ),
)
async def resume_agent(task_id: str, body: HumanResumeRequest) -> AgentRunResponse:
    """
    POST /agent/resume/{task_id}

    调用 resume_workflow() 从断点恢复，并重新注册后台任务以继续推送事件。
    """
    # 验证任务是否处于 waiting_human 状态
    raw_status = await _get_status(task_id)
    if raw_status is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    if raw_status != TaskStatus.WAITING_HUMAN.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"任务当前状态为 {raw_status}，不是 waiting_human，无法恢复",
        )

    snapshot = await _load_result(task_id) or {}
    session_id = snapshot.get("session_id", task_id)
    created_at = datetime.now(timezone.utc).isoformat()

    await _set_status(task_id, TaskStatus.RUNNING)

    # 异步恢复执行
    async def _resume_task():
        try:
            final_state = await resume_workflow(
                session_id=session_id,
                human_response=body.response,
            )
            final_status = final_state.get("status", TaskStatus.FAILED)
            final_output = final_state.get("final_answer", "")
            error        = final_state.get("error")

            await _save_result(task_id, {
                **snapshot,
                "status":      final_status.value if hasattr(final_status, "value") else str(final_status),
                "result":      final_output or None,
                "error":       error,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            await _set_status(task_id, TaskStatus(
                final_status.value if hasattr(final_status, "value") else str(final_status)
            ))

            if final_status == TaskStatus.COMPLETED:
                await _publish_event(task_id, AgentStepEvent.make_done(task_id, final_output))
            else:
                await _publish_event(task_id, AgentStepEvent.make_error(task_id, error or "恢复后执行失败"))
        except Exception as exc:
            logger.error("resume_agent 异常: %s", exc, exc_info=True)
            await _publish_event(task_id, AgentStepEvent.make_error(task_id, str(exc)))
        finally:
            _active_tasks.pop(task_id, None)

    bg_task = asyncio.create_task(_resume_task(), name=f"agent-resume-{task_id}")
    _active_tasks[task_id] = bg_task

    return AgentRunResponse(
        task_id=task_id,
        session_id=session_id,
        status=TaskStatus.RUNNING,
        mode=snapshot.get("mode", "single"),
        message="任务已恢复执行，继续通过 WebSocket 接收进展。",
        created_at=created_at,
        stream_url=f"/agent/stream/{task_id}",
    )


# ─────────────────────────────────────────────
# 4. WebSocket 实时推送
# ─────────────────────────────────────────────

@router.websocket("/stream/{task_id}")
async def stream_agent(websocket: WebSocket, task_id: str) -> None:
    """
    WebSocket /agent/stream/{task_id}

    ┌─────────────────────────────────────────────────────────┐
    │                  WebSocket 生命周期                       │
    │                                                          │
    │  Client Connect                                          │
    │       ↓                                                  │
    │  [1] 验证 task_id（404 → 拒绝连接）                      │
    │       ↓                                                  │
    │  [2] 检查任务是否已完成                                   │
    │      ├── 已完成 → 推送历史 steps 摘要 + done → 关闭       │
    │      └── 进行中 → 继续订阅                                │
    │       ↓                                                  │
    │  [3] 订阅 Redis Pub/Sub agent:events:{task_id}           │
    │       ↓                                                  │
    │  [4] 轮询循环（100ms 间隔）                               │
    │      ├── 收到事件消息 → 转发给 WebSocket 客户端           │
    │      ├── 收到 done/error → 转发后关闭连接                 │
    │      └── 超时（1h）→ 关闭连接                             │
    └─────────────────────────────────────────────────────────┘

    消息格式（UTF-8 JSON）：
        {"event_id": "...", "task_id": "...", "step_type": "node_end",
         "node_name": "tool_executor", "content": "工具执行完成",
         "timestamp": "2026-03-24T10:30:00Z"}
    """
    # ── 验证 task_id ────────────────────────────────────────────────────────
    raw_status = await _get_status(task_id)
    if raw_status is None:
        await websocket.close(code=4004, reason=f"task_id {task_id} 不存在")
        return

    await websocket.accept()
    logger.info("WebSocket 已连接 | task_id=%s | status=%s", task_id, raw_status)

    # ── 若任务已终态，直接推送历史摘要 + done/error ─────────────────────────
    terminal_statuses = {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value}
    if raw_status in terminal_statuses:
        snapshot = await _load_result(task_id)
        if snapshot:
            # 推送历史步骤
            for step in snapshot.get("steps", []):
                try:
                    hist_event = AgentStepEvent(
                        task_id=task_id,
                        step_type=StepEventType.NODE_END,
                        node_name=step.get("node_name", "unknown"),
                        content=step.get("content", "")[:300],
                    )
                    await websocket.send_bytes(hist_event.to_ws_bytes())
                except Exception:
                    break

        # 推送终态事件
        if raw_status == TaskStatus.COMPLETED.value:
            result = (snapshot or {}).get("result", "")
            done_evt = AgentStepEvent.make_done(task_id=task_id, result=result or "")
        else:
            error = (snapshot or {}).get("error", "任务失败")
            done_evt = AgentStepEvent.make_error(task_id=task_id, error=error or "")

        try:
            await websocket.send_bytes(done_evt.to_ws_bytes())
        except Exception:
            pass
        await websocket.close()
        return

    # ── 订阅 Redis Pub/Sub 频道，实时转发事件 ───────────────────────────────
    redis = await _get_redis()
    pubsub = redis.pubsub()
    channel = _CHANNEL.format(task_id)

    try:
        await pubsub.subscribe(channel)
        deadline = time.monotonic() + _WS_TIMEOUT

        while time.monotonic() < deadline:
            # 接收 Redis 消息（非阻塞，timeout=100ms）
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=_WS_POLL_MS / 1000,
                )
            except Exception as redis_exc:
                logger.warning("WebSocket: Redis 读取异常: %s", redis_exc)
                await asyncio.sleep(_WS_POLL_MS / 1000)
                continue

            if message and message.get("type") == "message":
                raw_data = message.get("data", b"")
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8")

                # 转发原始 JSON 给 WebSocket 客户端
                try:
                    await websocket.send_text(raw_data)
                except WebSocketDisconnect:
                    logger.info("WebSocket 客户端断开 | task_id=%s", task_id)
                    break

                # 检查是否为终态事件（done / error），关闭连接
                try:
                    evt_dict = json.loads(raw_data)
                    step_type = evt_dict.get("step_type", "")
                    if step_type in (StepEventType.DONE.value, StepEventType.ERROR.value):
                        logger.info(
                            "WebSocket 收到终态事件 %s | task_id=%s", step_type, task_id
                        )
                        break
                except (json.JSONDecodeError, KeyError):
                    pass

            else:
                # 无新消息，小憩一下
                await asyncio.sleep(_WS_POLL_MS / 1000)

    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开（客户端主动）| task_id=%s", task_id)

    except Exception as exc:
        logger.error("WebSocket 异常 | task_id=%s | error=%s", task_id, exc)
        try:
            error_evt = AgentStepEvent.make_error(task_id=task_id, error=str(exc))
            await websocket.send_bytes(error_evt.to_ws_bytes())
        except Exception:
            pass

    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket 连接已关闭 | task_id=%s", task_id)