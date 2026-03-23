"""
@File       : embedding_client.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: OpenAI Embedding 客户端封装，提供向量生成、Redis 缓存与批量并发处理。
@Features:
  - 基于 OpenAI text-embedding-3-small（dim=1536）生成文本向量
  - Redis 缓存：key = "synaris:emb:{sha256(text)}"，TTL=24h
      缓存命中直接返回，避免重复调用 OpenAI API，节省成本
  - embed_text()   单条文本嵌入（优先读缓存）
  - embed_batch()  批量文本嵌入，asyncio.Semaphore 控制最大 10 并发，
                   自动分批（每批 ≤ 20 条，OpenAI 推荐上限）
  - 命中率统计：hit_count / miss_count → cache_hit_ratio 指标
  - 模型调用失败 → 抛出 LLMError(LLM_UNAVAILABLE)
  - get_embedding_client() 工厂函数，兼容 FastAPI Depends

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import List, Optional

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from app.config.settings import get_settings
from app.core.exceptions import ErrorCode, LLMError
from app.core.logging import get_logger
from app.infrastructure.redis_client import build_key, get_json, set_json

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIM: int = 1536
CACHE_TTL: int = 24 * 60 * 60  # 24 小时
CACHE_PREFIX: str = "emb"
BATCH_SIZE: int = 20  # OpenAI 单次请求推荐上限
MAX_CONCURRENCY: int = 10  # asyncio.Semaphore 并发上限


# ---------------------------------------------------------------------------
# 缓存 Key 构造
# ---------------------------------------------------------------------------


def _cache_key(text: str) -> str:
    """
    用 SHA-256 摘要作为缓存 Key，避免长文本直接作为 Key 导致 Redis 内存浪费。

    示例：
        _cache_key("hello world")
        → "synaris:emb:b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576bc76c22892f9b1b"
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return build_key(CACHE_PREFIX, digest)


# ---------------------------------------------------------------------------
# EmbeddingClient
# ---------------------------------------------------------------------------


class EmbeddingClient:
    """
    OpenAI Embedding 客户端，带 Redis 缓存与批量并发处理。

    实例状态：
      _hit_count   — 缓存命中计数（进程级，重启清零）
      _miss_count  — 缓存未命中计数
      _semaphore   — 控制并发 OpenAI API 调用上限
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=getattr(settings, "openai_api_base", None) or None,
            timeout=30.0,
            max_retries=0,  # 重试由上层统一管理
        )
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ── 单条嵌入 ──────────────────────────────────────────────────────

    async def embed_text(self, text: str) -> List[float]:
        """
        对单条文本生成嵌入向量，优先读 Redis 缓存。

        Args:
            text: 待嵌入文本（建议 ≤ 512 字符，对应 chunk_size）

        Returns:
            长度为 1536 的 float 列表

        Raises:
            LLMError: OpenAI API 调用失败
        """
        if not text or not text.strip():
            raise LLMError(
                message="Embedding 输入文本不能为空",
                error_code=ErrorCode.LLM_INVALID_RESPONSE,
            )

        text = text.strip()
        key = _cache_key(text)

        # 1. 查缓存
        cached = await get_json(key)
        if cached is not None:
            self._hit_count += 1
            logger.debug("Embedding 缓存命中", extra={"key": key[:32]})
            return cached

        # 2. 调用 OpenAI
        self._miss_count += 1
        vector = await self._call_openai([text])
        embedding = vector[0]

        # 3. 写缓存
        await set_json(key, embedding, ttl=CACHE_TTL)
        logger.debug(
            "Embedding 已生成并缓存",
            extra={"key": key[:32], "dim": len(embedding)},
        )
        return embedding

    # ── 批量嵌入 ──────────────────────────────────────────────────────

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本嵌入，自动分批 + 并发控制 + 缓存复用。

        处理流程：
          1. 对每条文本查 Redis 缓存，命中则直接使用
          2. 将未命中文本按 BATCH_SIZE(20) 分批
          3. asyncio.Semaphore 控制最大 MAX_CONCURRENCY(10) 个批次并发请求
          4. 合并缓存结果与 API 结果，按原始顺序返回
          5. 将新生成的向量写入 Redis

        Args:
            texts: 待嵌入文本列表，允许为空（返回空列表）

        Returns:
            与 texts 等长的向量列表，顺序严格对应

        Raises:
            LLMError: 任意批次调用失败时抛出（已成功批次不回滚）
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # ── 第一步：查缓存 ────────────────────────────────────────────
        cache_tasks = [get_json(_cache_key(t.strip())) for t in texts]
        cached_results = await asyncio.gather(*cache_tasks, return_exceptions=True)

        for idx, cached in enumerate(cached_results):
            if isinstance(cached, list) and cached:
                results[idx] = cached
                self._hit_count += 1
            else:
                uncached_indices.append(idx)
                uncached_texts.append(texts[idx].strip())

        logger.info(
            "Embedding 批量缓存查询完成",
            extra={
                "total": len(texts),
                "cache_hit": len(texts) - len(uncached_indices),
                "api_needed": len(uncached_indices),
            },
        )

        if not uncached_texts:
            return results  # type: ignore[return-value]

        # ── 第二步：分批并发调用 OpenAI ───────────────────────────────
        batches: List[List[str]] = [
            uncached_texts[i : i + BATCH_SIZE]
            for i in range(0, len(uncached_texts), BATCH_SIZE)
        ]

        async def _fetch_batch(batch: List[str]) -> List[List[float]]:
            async with self._semaphore:
                return await self._call_openai(batch)

        batch_results_nested = await asyncio.gather(
            *[_fetch_batch(b) for b in batches],
            return_exceptions=False,
        )

        # ── 第三步：展平批次结果，回填 results ────────────────────────
        flat_vectors: List[List[float]] = [
            vec for batch in batch_results_nested for vec in batch
        ]

        write_cache_tasks = []
        for order_idx, (orig_idx, text, vector) in enumerate(
            zip(uncached_indices, uncached_texts, flat_vectors)
        ):
            results[orig_idx] = vector
            self._miss_count += 1
            write_cache_tasks.append(set_json(_cache_key(text), vector, ttl=CACHE_TTL))

        # 批量写缓存（并发，失败不中断主流程）
        await asyncio.gather(*write_cache_tasks, return_exceptions=True)

        logger.info(
            "Embedding 批量生成完成",
            extra={
                "total_vectors": len(texts),
                "api_calls": len(batches),
            },
        )
        return results  # type: ignore[return-value]

    # ── OpenAI API 调用 ───────────────────────────────────────────────

    async def _call_openai(self, texts: List[str]) -> List[List[float]]:
        """
        调用 OpenAI Embedding API，返回向量列表（顺序与 texts 对应）。

        Raises:
            LLMError: API 调用失败时转换为 LLMError
        """
        try:
            response = await self._client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                encoding_format="float",
            )
            # response.data 按 index 排序，与 texts 顺序一致
            vectors = [item.embedding for item in response.data]
            logger.debug(
                "OpenAI Embedding API 调用成功",
                extra={
                    "model": EMBEDDING_MODEL,
                    "input_count": len(texts),
                    "total_tokens": response.usage.total_tokens,
                },
            )
            return vectors

        except APITimeoutError as exc:
            raise LLMError(
                message="Embedding API 响应超时",
                error_code=ErrorCode.LLM_TIMEOUT,
                detail={"model": EMBEDDING_MODEL, "error": str(exc)},
            ) from exc
        except RateLimitError as exc:
            raise LLMError(
                message="Embedding API 配额超限，请稍后重试",
                error_code=ErrorCode.LLM_QUOTA_EXCEEDED,
                detail={"model": EMBEDDING_MODEL, "error": str(exc)},
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(
                message="无法连接到 OpenAI Embedding 服务",
                error_code=ErrorCode.LLM_UNAVAILABLE,
                detail={"model": EMBEDDING_MODEL, "error": str(exc)},
            ) from exc
        except Exception as exc:
            raise LLMError(
                message=f"Embedding 生成失败：{exc}",
                error_code=ErrorCode.LLM_UNAVAILABLE,
                detail={"model": EMBEDDING_MODEL, "error": str(exc)},
            ) from exc

    # ── 统计指标 ──────────────────────────────────────────────────────

    @property
    def cache_hit_ratio(self) -> float:
        """
        缓存命中率（0.0 ~ 1.0）。
        进程级统计，重启清零。
        """
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """返回缓存统计摘要，供 /metrics 或日志使用。"""
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "model": EMBEDDING_MODEL,
            "cache_ttl_hours": CACHE_TTL // 3600,
        }

    def reset_stats(self) -> None:
        """重置计数器（测试用途）。"""
        self._hit_count = 0
        self._miss_count = 0


# ---------------------------------------------------------------------------
# 模块级单例 + 依赖注入
# ---------------------------------------------------------------------------

_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """
    获取 EmbeddingClient 模块级单例。

    FastAPI Depends 兼容，也可直接调用。
    单例模式确保 Semaphore 与缓存统计在进程内全局共享。

    用法（FastAPI Depends）：
        @router.post("/embed")
        async def embed(
            text: str,
            client: EmbeddingClient = Depends(get_embedding_client),
        ):
            vector = await client.embed_text(text)

    用法（直接调用）：
        client = get_embedding_client()
        vectors = await client.embed_batch(texts)
    """
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
        logger.debug("EmbeddingClient 实例已创建")
    return _embedding_client
