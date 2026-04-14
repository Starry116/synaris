"""
@File       : memory_service.py
@Author     : Starry Hung
@Created    : 2026-04-14
@Version    : 1.0.0
@Description: 四种记忆系统统一管理服务。
@Features:
  ── 记忆类型 1：Short-term Memory（短期对话记忆）────────────────────────────
  - 存储：Redis，key=synaris:mem:short:{session_id}，TTL=2h
  - 内容：最近 20 条消息（role / content / timestamp）
  - async get_recent(session_id, n=10) → List[ShortTermMessage]
  - async append(session_id, message) → None（超 20 条滑动丢弃）

  ── 记忆类型 2：Long-term Memory（长期向量记忆）─────────────────────────────
  - 存储：Milvus Collection synaris_long_term_memory（独立于知识库）
  - 触发：对话轮次超过 10 轮时，LLM 自动生成摘要后写入向量
  - async store_memory(user_id, content, memory_type) → str（向量 ID）
  - async recall(user_id, query, top_k=3) → List[MemoryItem]
  - 摘要策略：调用 LLM（gpt-4o-mini）将历史消息压缩为关键事实

  ── 记忆类型 3：User Profile Memory（用户画像记忆）──────────────────────────
  - 存储：PostgreSQL users.preferences（JSONB/TEXT 字段）
  - 内容：偏好语言 / 领域标签 / 沟通风格 / 常用话题
  - async get_profile(user_id) → UserProfile
  - async update_profile(user_id, updates: dict) → None
  - async extract_and_update_preferences(user_id, messages) → None
    （每 10 次对话触发一次，LLM 自动抽取偏好信息）

  ── 记忆类型 4：Task Memory（任务上下文记忆）────────────────────────────────
  - 存储：Redis（实时，key=synaris:mem:task:{task_id}）+ PostgreSQL（持久化）
  - 内容：LangGraph 各节点中间结果 / 工具调用历史 / 执行计划快照
  - async save_task_context(task_id, node_name, context) → None
  - async get_task_context(task_id) → TaskContext
  - async finalize_task_context(task_id) → None（任务完成后写入 PostgreSQL）

  ── 统一入口：MemoryManager ──────────────────────────────────────────────────
  - 聚合四种记忆的读写操作
  - async build_memory_context(session_id, user_id, task_id) → MemoryContext
    （一次调用组装 AgentState.memory_context 所需的全部内容）
  - async post_conversation_hook(session_id, user_id, messages) → None
    （对话结束时自动触发：长期记忆摘要 + 偏好提取）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-14  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

# ─────────────────────────────────────────────
# 模块日志
# ─────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. 常量与枚举
# ─────────────────────────────────────────────

# Short-term Memory
_SHORT_TTL       = 2 * 60 * 60       # 2 小时
_SHORT_MAX       = 20                # 最多保留消息条数
_SHORT_PREFIX    = "mem:short"

# Long-term Memory
_LONG_COLLECTION = "synaris_long_term_memory"
_LONG_DIM        = 1536              # text-embedding-3-small 维度
_LONG_THRESHOLD  = 0.65              # 召回相关度阈值（IP 度量）
_LONG_SUMMARIZE_THRESHOLD = 10       # 对话轮次超过此值触发摘要

# Task Memory
_TASK_TTL        = 3 * 24 * 60 * 60  # 3 天（与 Celery 结果保留一致）
_TASK_PREFIX     = "mem:task"

# MemoryManager
_BUILD_TIMEOUT   = 5.0               # build_memory_context 最长等待秒数（各类型并发）


class MemoryType(str, Enum):
    """长期记忆的内容类型，用于 Milvus 元数据过滤。"""
    CONVERSATION_SUMMARY = "conversation_summary"  # 对话摘要
    KEY_FACT             = "key_fact"              # 关键事实
    USER_PREFERENCE      = "user_preference"       # 明确偏好
    TASK_RESULT          = "task_result"            # 任务产出摘要


# ─────────────────────────────────────────────
# 2. 数据模型
# ─────────────────────────────────────────────

@dataclass
class ShortTermMessage:
    """
    短期记忆中的单条消息。

    类比通话记录：role = 说话人，content = 内容，timestamp = 时间戳。
    """
    role:      str        # "human" | "ai" | "system" | "tool"
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "ShortTermMessage":
        return cls(
            role=d.get("role", "human"),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class MemoryItem:
    """
    长期记忆召回的单条结果。

    类比档案检索：source_id = 档案编号，content = 内容摘要，score = 相关度。
    """
    source_id:   str
    user_id:     str
    content:     str
    memory_type: str
    score:       float
    created_at:  str

    def to_dict(self) -> dict:
        return {
            "source_id":   self.source_id,
            "user_id":     self.user_id,
            "content":     self.content,
            "memory_type": self.memory_type,
            "score":       round(self.score, 4),
            "created_at":  self.created_at,
        }


@dataclass
class UserProfile:
    """
    用户画像数据模型。

    字段设计类比「个人简历」：
      preferred_language → 偏好语言
      domain_tags        → 擅长领域标签
      communication_style → 沟通风格（简洁/详细/技术性/友好）
      frequent_topics    → 常见话题
      custom             → 任意扩展字段
    """
    user_id:              str
    preferred_language:   str             = "zh"
    domain_tags:          List[str]       = field(default_factory=list)
    communication_style:  str             = "balanced"   # concise / detailed / technical / friendly
    frequent_topics:      List[str]       = field(default_factory=list)
    custom:               dict            = field(default_factory=dict)
    updated_at:           Optional[str]   = None

    def to_dict(self) -> dict:
        return {
            "preferred_language":  self.preferred_language,
            "domain_tags":         self.domain_tags,
            "communication_style": self.communication_style,
            "frequent_topics":     self.frequent_topics,
            "custom":              self.custom,
            "updated_at":          self.updated_at,
        }

    @classmethod
    def from_dict(cls, user_id: str, d: dict) -> "UserProfile":
        return cls(
            user_id=user_id,
            preferred_language=d.get("preferred_language", "zh"),
            domain_tags=d.get("domain_tags", []),
            communication_style=d.get("communication_style", "balanced"),
            frequent_topics=d.get("frequent_topics", []),
            custom=d.get("custom", {}),
            updated_at=d.get("updated_at"),
        )

    @classmethod
    def empty(cls, user_id: str) -> "UserProfile":
        return cls(user_id=user_id)


@dataclass
class NodeContext:
    """单个 LangGraph 节点的执行上下文快照。"""
    node_name:   str
    context:     dict          # 节点产出的任意结构化数据
    saved_at:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TaskContext:
    """
    任务上下文记忆的完整容器。

    类比「施工日志」：按节点顺序记录每道工序的中间产出，
    供后续节点（或人工审查）查阅。
    """
    task_id:    str
    nodes:      List[NodeContext]   = field(default_factory=list)
    tool_calls: List[dict]          = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_node(self, node_name: str) -> Optional[NodeContext]:
        """按名称查找最后一次该节点的上下文（一个任务中同名节点可能执行多次）。"""
        for nc in reversed(self.nodes):
            if nc.node_name == node_name:
                return nc
        return None

    def to_dict(self) -> dict:
        return {
            "task_id":    self.task_id,
            "nodes":      [{"node_name": nc.node_name, "context": nc.context, "saved_at": nc.saved_at}
                           for nc in self.nodes],
            "tool_calls": self.tool_calls,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskContext":
        tc = cls(
            task_id=d.get("task_id", ""),
            created_at=d.get("created_at", ""),
        )
        for n in d.get("nodes", []):
            tc.nodes.append(NodeContext(
                node_name=n.get("node_name", ""),
                context=n.get("context", {}),
                saved_at=n.get("saved_at", ""),
            ))
        tc.tool_calls = d.get("tool_calls", [])
        return tc


@dataclass
class MemoryContext:
    """
    MemoryManager.build_memory_context() 的返回值。

    对应 AgentState.memory_context 字段，在 Step 12 中预留为 Optional[Any]，
    此处提供完整的强类型实现。

    类比「行前简报」：Agent 执行任务前拿到的全部背景信息汇总。
    """
    recent_messages:  List[ShortTermMessage]  = field(default_factory=list)
    long_term_recall: List[MemoryItem]        = field(default_factory=list)
    user_profile:     Optional[UserProfile]   = None
    task_context:     Optional[TaskContext]   = None

    def to_summary_text(self) -> str:
        """
        将 MemoryContext 压缩为可注入 Prompt 的纯文本摘要。
        供 workflow.py 的 planner_node 将其注入 {memory_context} 变量。
        """
        parts: List[str] = []

        if self.user_profile:
            p = self.user_profile
            parts.append(
                f"用户画像：语言偏好={p.preferred_language}，"
                f"沟通风格={p.communication_style}，"
                f"领域标签={', '.join(p.domain_tags) or '未知'}。"
            )

        if self.recent_messages:
            last_n = self.recent_messages[-5:]  # 只取最近 5 条
            parts.append("近期对话片段：")
            for m in last_n:
                role_label = "用户" if m.role == "human" else "助手"
                parts.append(f"  [{role_label}] {m.content[:120]}")

        if self.long_term_recall:
            parts.append("长期记忆关键事实：")
            for item in self.long_term_recall:
                parts.append(f"  - {item.content[:150]}")

        if self.task_context and self.task_context.nodes:
            last_node = self.task_context.nodes[-1]
            parts.append(f"当前任务最新节点：{last_node.node_name}（已执行）。")

        return "\n".join(parts) if parts else "（暂无记忆上下文）"


# ─────────────────────────────────────────────
# 3. Short-term Memory Service
# ─────────────────────────────────────────────

class ShortTermMemoryService:
    """
    短期对话记忆服务（Redis 存储）。

    数据结构：Redis List → JSON 序列化的 ShortTermMessage 列表，整体存储为单个 JSON 数组。
    选择整体存储而非 LPUSH 的理由：
      - 截断（保留最近 N 条）用数组切片即可，LTRIM 对 JSON 数组不适用
      - TTL 刷新与消息写入在同一次 SET 中完成，无竞态
      - 读写都是一次 Redis 往返，延迟更低
    """

    def _key(self, session_id: str) -> str:
        from infrastructure.redis_client import build_key  # type: ignore[import]
        return build_key(_SHORT_PREFIX, session_id)

    async def get_recent(
        self,
        session_id: str,
        n: int = 10,
    ) -> List[ShortTermMessage]:
        """
        获取最近 n 条消息（不足 n 条时全部返回）。

        Args:
            session_id: 会话 ID
            n:          最多返回条数（默认 10）

        Returns:
            按时间正序排列的消息列表（最新的在末尾）
        """
        try:
            from infrastructure.redis_client import get_json  # type: ignore[import]
            raw = await get_json(self._key(session_id))
            if not raw:
                return []
            messages = [ShortTermMessage.from_dict(d) for d in raw]
            return messages[-n:]  # 取末尾 n 条
        except Exception as exc:
            logger.warning(
                "ShortTermMemory.get_recent 失败，返回空列表 | session_id=%s | error=%s",
                session_id, exc,
            )
            return []

    async def append(
        self,
        session_id: str,
        message:    ShortTermMessage,
    ) -> None:
        """
        追加一条消息到短期记忆，超过 _SHORT_MAX 条时自动丢弃最早的消息。
        同时刷新 TTL（每次对话活跃时延长生命周期）。

        Args:
            session_id: 会话 ID
            message:    待追加的消息
        """
        try:
            from infrastructure.redis_client import get_json, set_json  # type: ignore[import]
            existing: List[dict] = await get_json(self._key(session_id)) or []
            existing.append(message.to_dict())

            if len(existing) > _SHORT_MAX:
                existing = existing[-_SHORT_MAX:]
                logger.debug(
                    "短期记忆已截断 | session_id=%s | kept=%d",
                    session_id, _SHORT_MAX,
                )

            await set_json(self._key(session_id), existing, ttl=_SHORT_TTL)
        except Exception as exc:
            logger.warning(
                "ShortTermMemory.append 失败（非致命）| session_id=%s | error=%s",
                session_id, exc,
            )

    async def append_pair(
        self,
        session_id:    str,
        human_content: str,
        ai_content:    str,
    ) -> None:
        """
        一次性追加「用户 + AI」一对消息（减少 Redis 往返次数）。
        这是最常用的调用方式——每轮对话结束后调用一次。
        """
        try:
            from infrastructure.redis_client import get_json, set_json  # type: ignore[import]
            existing: List[dict] = await get_json(self._key(session_id)) or []
            now = datetime.now(timezone.utc).isoformat()
            existing.append({"role": "human", "content": human_content, "timestamp": now})
            existing.append({"role": "ai",    "content": ai_content,    "timestamp": now})

            if len(existing) > _SHORT_MAX:
                existing = existing[-_SHORT_MAX:]

            await set_json(self._key(session_id), existing, ttl=_SHORT_TTL)
        except Exception as exc:
            logger.warning(
                "ShortTermMemory.append_pair 失败（非致命）| session_id=%s | error=%s",
                session_id, exc,
            )

    async def clear(self, session_id: str) -> bool:
        """清空会话短期记忆，返回 True 表示 key 存在并被删除。"""
        try:
            from infrastructure.redis_client import delete  # type: ignore[import]
            count = await delete(self._key(session_id))
            return count > 0
        except Exception as exc:
            logger.warning(
                "ShortTermMemory.clear 失败 | session_id=%s | error=%s",
                session_id, exc,
            )
            return False

    async def message_count(self, session_id: str) -> int:
        """返回当前记忆中的消息条数。"""
        try:
            from infrastructure.redis_client import get_json  # type: ignore[import]
            raw = await get_json(self._key(session_id))
            return len(raw) if raw else 0
        except Exception:
            return 0


# ─────────────────────────────────────────────
# 4. Long-term Memory Service
# ─────────────────────────────────────────────

class LongTermMemoryService:
    """
    长期向量记忆服务（Milvus 独立 Collection）。

    工作原理类比「人的长期记忆形成机制」：
      短期记忆 → 反复激活 / 情绪显著 → 海马体编码 → 长期记忆
    对应：
      短期消息 → 超过 10 轮 → LLM 摘要 → 向量化存储 → 未来相关时召回

    Collection Schema（与知识库 Collection 隔离）：
      id（VARCHAR，auto）/ content（VARCHAR）/ embedding（FLOAT_VECTOR）/
      user_id（VARCHAR）/ memory_type（VARCHAR）/ created_at（INT64）
    """

    async def _ensure_collection(self) -> None:
        """
        确保长期记忆 Collection 存在，不存在则自动创建。
        使用独立 Collection（synaris_long_term_memory）与知识库物理隔离。
        """
        try:
            from infrastructure.milvus_client import get_milvus_client  # type: ignore[import]
            client = get_milvus_client()
            await client.ensure_collection_exists(
                _LONG_COLLECTION,
                description="Synaris 长期向量记忆，存储跨会话关键信息摘要",
            )
        except Exception as exc:
            logger.error("LongTermMemory: 确保 Collection 存在时失败: %s", exc)
            raise

    async def store_memory(
        self,
        user_id:     str,
        content:     str,
        memory_type: MemoryType = MemoryType.CONVERSATION_SUMMARY,
    ) -> str:
        """
        将文本内容向量化后存入长期记忆。

        Args:
            user_id:     所属用户 UUID 字符串
            content:     待存储的文本内容（已摘要）
            memory_type: 记忆类型标签

        Returns:
            存储后的记忆 ID（字符串格式的 Milvus 主键）

        Raises:
            VectorDBError / LLMError：底层存储或 Embedding 失败时
        """
        if not content or not content.strip():
            raise ValueError("长期记忆内容不能为空")

        content = content.strip()
        memory_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)

        try:
            await self._ensure_collection()

            # 生成 Embedding（含 Redis 缓存复用）
            from infrastructure.embedding_client import get_embedding_client  # type: ignore[import]
            embedder = get_embedding_client()
            embedding = await embedder.embed_text(content)

            # 写入 Milvus
            from infrastructure.milvus_client import get_milvus_client  # type: ignore[import]
            from infrastructure.milvus_client import _run_sync           # type: ignore[import]
            from pymilvus import Collection                               # type: ignore[import]

            coll = Collection(_LONG_COLLECTION)
            insert_data = {
                "id":          [memory_id],
                "content":     [content[:2048]],
                "embedding":   [embedding],
                "user_id":     [user_id],
                "memory_type": [memory_type.value],
                "created_at":  [now_ms],
            }
            await _run_sync(coll.insert, insert_data)
            await _run_sync(coll.flush)

            logger.info(
                "长期记忆已存储 | user_id=%s | type=%s | len=%d",
                user_id, memory_type.value, len(content),
            )
            return memory_id

        except Exception as exc:
            logger.error(
                "LongTermMemory.store_memory 失败 | user_id=%s | error=%s",
                user_id, exc,
            )
            raise

    async def recall(
        self,
        user_id: str,
        query:   str,
        top_k:   int = 3,
    ) -> List[MemoryItem]:
        """
        语义召回与 query 最相关的长期记忆（只召回属于该用户的记忆）。

        Args:
            user_id: 所属用户 UUID 字符串（按 user_id 过滤，隔离用户数据）
            query:   当前任务描述或对话内容
            top_k:   返回最多 K 条

        Returns:
            按相关度降序排列的 MemoryItem 列表
        """
        if not query.strip():
            return []

        try:
            await self._ensure_collection()

            from infrastructure.embedding_client import get_embedding_client  # type: ignore[import]
            embedder = get_embedding_client()
            query_vector = await embedder.embed_text(query)

            from infrastructure.milvus_client import _run_sync, METRIC_TYPE, HNSW_EF_SEARCH  # type: ignore[import]
            from pymilvus import Collection  # type: ignore[import]

            coll = Collection(_LONG_COLLECTION)
            await _run_sync(coll.load)

            search_params = {"metric_type": METRIC_TYPE, "params": {"ef": HNSW_EF_SEARCH}}
            safe_user_id  = user_id.replace('"', '\\"')
            expr          = f'user_id == "{safe_user_id}"'

            raw_results = await _run_sync(
                coll.search,
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["content", "user_id", "memory_type", "created_at"],
            )

            items: List[MemoryItem] = []
            for hit in (raw_results[0] if raw_results else []):
                if float(hit.score) < _LONG_THRESHOLD:
                    continue
                entity = hit.entity
                items.append(MemoryItem(
                    source_id=str(hit.id),
                    user_id=entity.get("user_id", ""),
                    content=entity.get("content", ""),
                    memory_type=entity.get("memory_type", ""),
                    score=round(float(hit.score), 4),
                    created_at=str(entity.get("created_at", "")),
                ))

            logger.debug(
                "长期记忆召回完成 | user_id=%s | hits=%d",
                user_id, len(items),
            )
            return items

        except Exception as exc:
            logger.warning(
                "LongTermMemory.recall 失败，返回空列表 | user_id=%s | error=%s",
                user_id, exc,
            )
            return []

    async def summarize_and_store(
        self,
        user_id:  str,
        messages: List[ShortTermMessage],
    ) -> Optional[str]:
        """
        调用 LLM 对短期对话消息进行摘要，然后存入长期记忆。

        触发条件：消息轮次超过 _LONG_SUMMARIZE_THRESHOLD（默认 10 轮）。

        摘要策略类比「会议纪要」：
          - 提取关键决策与结论
          - 记录用户明确表达的偏好或要求
          - 忽略无意义的寒暄与过渡语句

        Returns:
            存储后的 memory_id，若摘要或存储失败则返回 None
        """
        if len(messages) < _LONG_SUMMARIZE_THRESHOLD:
            return None  # 轮次不足，不触发摘要

        # 将消息格式化为摘要输入
        conversation_text = "\n".join(
            f"[{'用户' if m.role == 'human' else '助手'}] {m.content[:200]}"
            for m in messages[-20:]  # 取最近 20 条作为摘要输入
        )

        summarize_prompt = (
            "请将以下对话提炼为简洁的关键信息摘要（100字以内）。\n"
            "重点提取：\n"
            "  1. 用户的核心诉求或问题\n"
            "  2. 用户明确表达的偏好或约束\n"
            "  3. 本次对话达成的结论或决策\n"
            "不要包含：寒暄、过渡语句、无实质内容的确认。\n\n"
            f"对话内容：\n{conversation_text}\n\n"
            "摘要："
        )

        try:
            from langchain_core.messages import HumanMessage  # type: ignore[import]

            # 使用 economy 模型降低摘要成本
            try:
                from infrastructure.llm_client import LLMClientInterface  # type: ignore[import]
                llm_interface = LLMClientInterface()
                summary = await llm_interface.invoke(
                    [HumanMessage(content=summarize_prompt)],
                    model="gpt-4o-mini",
                    temperature=0.0,
                    max_tokens=200,
                )
            except ImportError:
                from langchain_openai import ChatOpenAI  # type: ignore[import]
                import os
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=200)
                resp = await llm.ainvoke([HumanMessage(content=summarize_prompt)])
                summary = resp.content

            if not summary or not summary.strip():
                return None

            memory_id = await self.store_memory(
                user_id=user_id,
                content=summary.strip(),
                memory_type=MemoryType.CONVERSATION_SUMMARY,
            )
            logger.info(
                "对话摘要已写入长期记忆 | user_id=%s | summary_len=%d",
                user_id, len(summary),
            )
            return memory_id

        except Exception as exc:
            logger.warning(
                "LongTermMemory.summarize_and_store 失败（非致命）| user_id=%s | error=%s",
                user_id, exc,
            )
            return None


# ─────────────────────────────────────────────
# 5. User Profile Memory Service
# ─────────────────────────────────────────────

class UserProfileMemoryService:
    """
    用户画像记忆服务（PostgreSQL 存储）。

    画像数据存储在 users.preferences（TEXT/JSONB 字段）中，
    避免新增数据表，与 Step 21 的 User 模型完全兼容。

    数据提取策略（LLM 自动抽取）：
      每 10 次对话后，用 gpt-4o-mini 分析最近的对话记录，
      识别并更新用户偏好字段。成本极低（仅 economy 模型，每 10 轮调用一次）。
    """

    async def get_profile(self, user_id: str) -> UserProfile:
        """
        从 PostgreSQL 读取用户画像。
        用户不存在或画像为空时返回默认画像（不抛异常）。

        Args:
            user_id: 用户 UUID 字符串

        Returns:
            UserProfile 实例
        """
        try:
            from sqlalchemy import select                                      # type: ignore[import]
            from infrastructure.postgres_client import db_session             # type: ignore[import]
            from models.user import User                                       # type: ignore[import]

            async with db_session() as session:
                result = await session.execute(
                    select(User.preferences).where(User.id == uuid.UUID(user_id))
                )
                row = result.scalar_one_or_none()

                if not row:
                    return UserProfile.empty(user_id)

                # preferences 字段存储 JSON 字符串
                try:
                    pref_dict = json.loads(row) if isinstance(row, str) else row
                    return UserProfile.from_dict(user_id, pref_dict or {})
                except (json.JSONDecodeError, TypeError):
                    return UserProfile.empty(user_id)

        except Exception as exc:
            logger.warning(
                "UserProfileMemory.get_profile 失败，返回默认画像 | user_id=%s | error=%s",
                user_id, exc,
            )
            return UserProfile.empty(user_id)

    async def update_profile(
        self,
        user_id: str,
        updates: dict,
    ) -> None:
        """
        更新用户画像（增量合并，不覆盖未传入的字段）。

        Args:
            user_id: 用户 UUID 字符串
            updates: 需要更新的字段字典（只更新传入的 key）
        """
        try:
            from sqlalchemy import select, update as sa_update              # type: ignore[import]
            from infrastructure.postgres_client import db_session           # type: ignore[import]
            from models.user import User                                     # type: ignore[import]

            async with db_session() as session:
                result = await session.execute(
                    select(User.preferences).where(User.id == uuid.UUID(user_id))
                )
                row = result.scalar_one_or_none()

                # 读取现有画像
                try:
                    existing = json.loads(row) if (row and isinstance(row, str)) else (row or {})
                except (json.JSONDecodeError, TypeError):
                    existing = {}

                # 增量合并
                for k, v in updates.items():
                    if isinstance(v, list) and isinstance(existing.get(k), list):
                        # 列表字段：合并去重，保留最新的 20 个
                        merged = list(dict.fromkeys(existing[k] + v))
                        existing[k] = merged[-20:]
                    else:
                        existing[k] = v

                existing["updated_at"] = datetime.now(timezone.utc).isoformat()

                await session.execute(
                    sa_update(User)
                    .where(User.id == uuid.UUID(user_id))
                    .values(preferences=json.dumps(existing, ensure_ascii=False))
                )

                logger.debug(
                    "用户画像已更新 | user_id=%s | keys=%s",
                    user_id, list(updates.keys()),
                )

        except Exception as exc:
            logger.warning(
                "UserProfileMemory.update_profile 失败（非致命）| user_id=%s | error=%s",
                user_id, exc,
            )

    async def extract_and_update_preferences(
        self,
        user_id:  str,
        messages: List[ShortTermMessage],
    ) -> None:
        """
        使用 LLM 从对话中自动提取用户偏好，并增量更新画像。

        触发条件：每 10 次对话（由 MemoryManager.post_conversation_hook 控制调用时机）。

        提取目标：
          - preferred_language：对话中使用的主要语言
          - domain_tags：涉及的技术/业务领域
          - communication_style：用户偏好的回答风格
          - frequent_topics：频繁出现的话题关键词

        Args:
            user_id:  用户 UUID 字符串
            messages: 最近的对话消息列表
        """
        if not messages:
            return

        # 只取最近 10 条用户消息作为分析样本（节省 Token）
        human_messages = [m for m in messages if m.role == "human"][-10:]
        if not human_messages:
            return

        sample_text = "\n".join(f"- {m.content[:150]}" for m in human_messages)

        extract_prompt = (
            "分析以下用户消息，提取用户偏好信息。\n"
            "严格按照 JSON 格式输出，不要输出其他内容：\n"
            "{\n"
            '  "preferred_language": "<zh|en|ja|...>",\n'
            '  "domain_tags": ["<领域1>", "<领域2>"],\n'
            '  "communication_style": "<concise|detailed|technical|friendly>",\n'
            '  "frequent_topics": ["<话题1>", "<话题2>"]\n'
            "}\n\n"
            f"用户消息样本：\n{sample_text}"
        )

        try:
            from langchain_core.messages import HumanMessage  # type: ignore[import]

            try:
                from infrastructure.llm_client import LLMClientInterface  # type: ignore[import]
                llm_interface = LLMClientInterface()
                raw = await llm_interface.invoke(
                    [HumanMessage(content=extract_prompt)],
                    model="gpt-4o-mini",
                    temperature=0.0,
                    max_tokens=200,
                )
            except ImportError:
                from langchain_openai import ChatOpenAI  # type: ignore[import]
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=200)
                resp = await llm.ainvoke([HumanMessage(content=extract_prompt)])
                raw = resp.content

            # 清理 LLM 输出（可能包含 ```json ... ``` 包装）
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                inner = []
                for line in lines[1:]:
                    if line.strip() == "```":
                        break
                    inner.append(line)
                text = "\n".join(inner).strip()

            extracted = json.loads(text)

            # 过滤空值，只更新有内容的字段
            updates = {k: v for k, v in extracted.items() if v and v != "unknown"}
            if updates:
                await self.update_profile(user_id, updates)
                logger.info(
                    "用户偏好已自动提取并更新 | user_id=%s | fields=%s",
                    user_id, list(updates.keys()),
                )

        except Exception as exc:
            logger.warning(
                "UserProfileMemory.extract_and_update_preferences 失败（非致命）| user_id=%s | error=%s",
                user_id, exc,
            )


# ─────────────────────────────────────────────
# 6. Task Memory Service
# ─────────────────────────────────────────────

class TaskMemoryService:
    """
    任务上下文记忆服务（Redis 实时 + PostgreSQL 持久化）。

    两层存储的分工（类比「白板 + 档案柜」）：
      Redis（白板）：实时读写，TTL=3天，任务执行中各节点频繁读写
      PostgreSQL（档案柜）：任务完成后一次性归档，永久保存，支持历史查询

    数据写入流程：
      1. 每个节点执行后 → save_task_context() → 写 Redis
      2. 任务完成后 → finalize_task_context() → 读 Redis → 写 PostgreSQL
    """

    def _key(self, task_id: str) -> str:
        from infrastructure.redis_client import build_key  # type: ignore[import]
        return build_key(_TASK_PREFIX, task_id)

    async def save_task_context(
        self,
        task_id:   str,
        node_name: str,
        context:   dict,
    ) -> None:
        """
        保存单个节点的执行上下文快照到 Redis。

        每次调用追加一个 NodeContext，不覆盖已有节点记录。
        供 LangGraph 各节点在执行完毕后调用，记录中间产出。

        Args:
            task_id:   Celery / asyncio 任务 ID
            node_name: LangGraph 节点名称（如 "planner"）
            context:   该节点产出的任意可序列化数据
        """
        try:
            from infrastructure.redis_client import get_json, set_json  # type: ignore[import]

            raw = await get_json(self._key(task_id))
            if raw:
                tc = TaskContext.from_dict(raw)
            else:
                tc = TaskContext(task_id=task_id)

            tc.nodes.append(NodeContext(node_name=node_name, context=context))

            # 也追加到 tool_calls（若 context 中包含工具调用结果）
            if "tool_results" in context and isinstance(context["tool_results"], list):
                tc.tool_calls.extend(context["tool_results"])

            await set_json(self._key(task_id), tc.to_dict(), ttl=_TASK_TTL)

        except Exception as exc:
            logger.warning(
                "TaskMemory.save_task_context 失败（非致命）| task_id=%s | node=%s | error=%s",
                task_id, node_name, exc,
            )

    async def get_task_context(self, task_id: str) -> Optional[TaskContext]:
        """
        从 Redis 读取任务上下文。
        key 不存在时返回 None（不抛异常）。

        Args:
            task_id: 任务 ID

        Returns:
            TaskContext 实例，或 None
        """
        try:
            from infrastructure.redis_client import get_json  # type: ignore[import]
            raw = await get_json(self._key(task_id))
            if not raw:
                return None
            return TaskContext.from_dict(raw)
        except Exception as exc:
            logger.warning(
                "TaskMemory.get_task_context 失败 | task_id=%s | error=%s",
                task_id, exc,
            )
            return None

    async def finalize_task_context(self, task_id: str) -> None:
        """
        任务完成时将 Redis 中的上下文归档到 PostgreSQL。

        写入目标：AgentTask.step_log 字段（JSONB，存储节点轨迹列表）。
        Step 21 完成后生效，若 PostgreSQL 不可用则静默跳过。

        Args:
            task_id: 业务任务 ID（与 PostgreSQL AgentTask.task_id 对应）
        """
        tc = await self.get_task_context(task_id)
        if not tc:
            logger.debug("TaskMemory.finalize: 无上下文可归档 | task_id=%s", task_id)
            return

        try:
            from sqlalchemy import select                                     # type: ignore[import]
            from infrastructure.postgres_client import db_session            # type: ignore[import]
            from models.task import AgentTask                                # type: ignore[import]

            async with db_session() as session:
                result = await session.execute(
                    select(AgentTask).where(AgentTask.task_id == task_id)
                )
                task = result.scalar_one_or_none()
                if task:
                    task.step_log = [
                        {"node_name": nc.node_name, "context": nc.context, "saved_at": nc.saved_at}
                        for nc in tc.nodes
                    ]
                    logger.info(
                        "任务上下文已归档到 PostgreSQL | task_id=%s | nodes=%d",
                        task_id, len(tc.nodes),
                    )
                else:
                    logger.warning(
                        "TaskMemory.finalize: 未找到 AgentTask 记录 | task_id=%s",
                        task_id,
                    )

        except ImportError:
            # Step 21 未完成时 postgres_client 可能未就绪，静默跳过
            logger.debug("TaskMemory.finalize: PostgreSQL 未就绪，跳过归档（Step 21 后生效）")
        except Exception as exc:
            logger.warning(
                "TaskMemory.finalize_task_context 失败（非致命）| task_id=%s | error=%s",
                task_id, exc,
            )

    async def clear_task_context(self, task_id: str) -> bool:
        """清理 Redis 中的任务上下文（任务归档后可调用以节省内存）。"""
        try:
            from infrastructure.redis_client import delete  # type: ignore[import]
            count = await delete(self._key(task_id))
            return count > 0
        except Exception:
            return False


# ─────────────────────────────────────────────
# 7. MemoryManager — 统一入口
# ─────────────────────────────────────────────

class MemoryManager:
    """
    四种记忆系统的统一管理器（Facade 模式）。

    设计类比「企业档案管理员」：
      - 对外只暴露一个窗口（MemoryManager）
      - 内部自动路由到正确的存储后端（Redis / Milvus / PostgreSQL）
      - 调用方无需关心底层实现细节

    典型使用场景：
      # 1. Agent 任务开始前：构建记忆上下文
      memory = await memory_manager.build_memory_context(
          session_id="sess-xxx", user_id="user-yyy", task_id="task-zzz"
      )
      state["memory_context"] = memory

      # 2. 对话结束时：触发记忆巩固
      await memory_manager.post_conversation_hook(
          session_id="sess-xxx", user_id="user-yyy", messages=messages
      )

      # 3. 任务节点执行后：保存中间状态
      await memory_manager.task.save_task_context(
          task_id="task-zzz", node_name="planner", context=plan_output
      )
    """

    def __init__(self) -> None:
        self.short_term  = ShortTermMemoryService()
        self.long_term   = LongTermMemoryService()
        self.user_profile = UserProfileMemoryService()
        self.task        = TaskMemoryService()

        # 对话计数器（进程内，用于触发偏好提取）
        # key = user_id, value = 对话轮次计数
        self._conversation_counter: dict[str, int] = {}

    async def build_memory_context(
        self,
        session_id:  Optional[str] = None,
        user_id:     Optional[str] = None,
        task_id:     Optional[str] = None,
        query:       str = "",
    ) -> MemoryContext:
        """
        并发组装所有可用的记忆上下文，供 Agent 任务启动前使用。

        并发策略：四种记忆同时发起查询，各自独立超时，
        单个超时不影响其他记忆类型的返回，
        最终等待所有查询完成（总超时 = _BUILD_TIMEOUT）。

        Args:
            session_id: 会话 ID（用于短期记忆查询）
            user_id:    用户 ID（用于长期记忆召回 + 用户画像）
            task_id:    任务 ID（用于任务上下文查询）
            query:      用于长期记忆语义召回的查询词（可为空）

        Returns:
            MemoryContext 实例，包含全部可用记忆
        """
        ctx = MemoryContext()

        # 定义各类型的查询协程（独立容错，不互相影响）
        async def _get_short():
            if not session_id:
                return
            ctx.recent_messages = await self.short_term.get_recent(session_id, n=10)

        async def _get_long():
            if not user_id or not query:
                return
            ctx.long_term_recall = await self.long_term.recall(user_id, query, top_k=3)

        async def _get_profile():
            if not user_id:
                return
            ctx.user_profile = await self.user_profile.get_profile(user_id)

        async def _get_task():
            if not task_id:
                return
            ctx.task_context = await self.task.get_task_context(task_id)

        # 并发执行，总超时 _BUILD_TIMEOUT 秒
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _get_short(),
                    _get_long(),
                    _get_profile(),
                    _get_task(),
                    return_exceptions=True,  # 任一失败不影响其他
                ),
                timeout=_BUILD_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "MemoryManager.build_memory_context 超时（%.1fs），使用已获取的部分记忆",
                _BUILD_TIMEOUT,
            )

        logger.debug(
            "记忆上下文组装完成 | session=%s | short=%d | long=%d | has_profile=%s | has_task=%s",
            session_id,
            len(ctx.recent_messages),
            len(ctx.long_term_recall),
            ctx.user_profile is not None,
            ctx.task_context is not None,
        )
        return ctx

    async def post_conversation_hook(
        self,
        session_id: str,
        user_id:    Optional[str],
        messages:   List[ShortTermMessage],
    ) -> None:
        """
        对话结束时的记忆巩固钩子（异步后台执行，不阻塞响应）。

        执行步骤：
          1. 检查是否达到长期记忆摘要阈值（超过 10 轮）
          2. 检查是否达到偏好提取阈值（每 10 次对话一次）
          3. 若两个条件均满足，并发执行摘要 + 偏好提取

        设计为「低优先级后台任务」，执行失败不影响主流程，
        类比「下班后自动备份」，用户无感知。

        Args:
            session_id: 会话 ID
            user_id:    用户 ID（None 表示匿名用户，跳过长期记忆和画像）
            messages:   本轮对话所有消息
        """
        if not user_id:
            return  # 匿名用户不处理长期记忆

        # 更新对话计数器
        count = self._conversation_counter.get(user_id, 0) + 1
        self._conversation_counter[user_id] = count

        tasks = []

        # 条件 1：触发长期记忆摘要
        if len(messages) >= _LONG_SUMMARIZE_THRESHOLD:
            tasks.append(self.long_term.summarize_and_store(user_id, messages))

        # 条件 2：每 10 次对话触发偏好提取
        if count % 10 == 0:
            tasks.append(
                self.user_profile.extract_and_update_preferences(user_id, messages)
            )

        if tasks:
            # 后台并发执行，不等待结果
            async def _run_hooks():
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as exc:
                    logger.warning("post_conversation_hook 后台任务异常: %s", exc)

            asyncio.ensure_future(_run_hooks())
            logger.debug(
                "post_conversation_hook 已触发 %d 个后台任务 | user_id=%s | count=%d",
                len(tasks), user_id, count,
            )

    async def append_message_pair(
        self,
        session_id:    str,
        human_content: str,
        ai_content:    str,
    ) -> None:
        """
        快捷方法：追加一轮对话到短期记忆。
        供 chat_service.py 在每轮对话结束后调用。

        Args:
            session_id:    会话 ID
            human_content: 用户输入
            ai_content:    AI 回复
        """
        await self.short_term.append_pair(session_id, human_content, ai_content)

    async def get_formatted_context(
        self,
        session_id:  Optional[str] = None,
        user_id:     Optional[str] = None,
        task_id:     Optional[str] = None,
        query:       str = "",
    ) -> str:
        """
        快捷方法：直接获取可注入 Prompt 的记忆文本摘要。
        供 workflow.py 的 planner_node 使用，代替 `memory_context` 变量的人工填写。

        Returns:
            纯文本摘要字符串（适合直接嵌入 Prompt）
        """
        ctx = await self.build_memory_context(
            session_id=session_id,
            user_id=user_id,
            task_id=task_id,
            query=query,
        )
        return ctx.to_summary_text()


# ─────────────────────────────────────────────
# 8. FastAPI 依赖注入
# ─────────────────────────────────────────────

# 进程级单例（所有请求共享同一个 MemoryManager 实例）
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """
    获取 MemoryManager 进程级单例。

    FastAPI Depends 兼容，也可直接调用。
    单例保证对话计数器（_conversation_counter）在进程内跨请求共享，
    使偏好提取的「每 10 次」触发逻辑正确工作。

    使用方式（FastAPI Depends）：
        @router.post("/chat")
        async def chat(
            body: ChatRequest,
            memory: MemoryManager = Depends(get_memory_manager),
        ):
            ctx = await memory.get_formatted_context(session_id=body.session_id)

    使用方式（直接调用）：
        from services.memory_service import get_memory_manager
        memory = get_memory_manager()
        await memory.append_message_pair(session_id, human_text, ai_text)
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        logger.info("MemoryManager 单例已创建")
    return _memory_manager