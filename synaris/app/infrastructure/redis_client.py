"""
@File       : redis_client.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 异步 Redis 客户端封装，提供连接池管理、KV 操作、JSON 序列化及 Pub/Sub 支持。
@Features:
  - 基于 redis.asyncio 的异步连接池，max_connections 从 settings 读取
  - 基础 KV 操作：get / set（带 TTL）/ delete / exists
  - JSON 序列化/反序列化：get_json / set_json（自动处理类型转换）
  - 命名空间 Key 构造工具：build_key(*parts) → "{prefix}:{part1}:{part2}"
  - Pub/Sub 支持：publish(channel, msg) / subscribe(channel) → AsyncIterator
  - 健康检查：ping() → bool
  - 连接失败时记录结构化日志并抛出 CacheConnectionError
  - get_redis_client() 工厂函数，兼容 FastAPI Depends 依赖注入

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, TimeoutError as RedisTimeoutError

from app.config.settings import get_settings
from app.core.exceptions import CacheConnectionError, CacheError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# 连接池单例
# ---------------------------------------------------------------------------

_pool: Optional[aioredis.ConnectionPool] = None
_client: Optional[aioredis.Redis] = None

KEY_NAMESPACE = settings.app_name.lower().replace(" ", "_")  # e.g. "synaris"


def _get_pool() -> aioredis.ConnectionPool:
    """
    懒初始化连接池（进程级单例）。
    多次调用只创建一个连接池实例。
    """
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,          # 统一返回 str，不返回 bytes
            socket_connect_timeout=5,       # 连接超时 5s
            socket_timeout=5,               # 读写超时 5s
            retry_on_timeout=True,
            health_check_interval=30,       # 每 30s 自动 PING 保活
        )
        logger.info(
            "Redis 连接池已初始化",
            extra={
                "redis_url": settings.redis_url.split("@")[-1],  # 隐藏密码
                "max_connections": settings.redis_max_connections,
            },
        )
    return _pool


def get_client() -> aioredis.Redis:
    """
    获取共享 Redis 客户端实例（不阻塞，连接池按需分配连接）。
    """
    global _client
    if _client is None:
        _client = aioredis.Redis(connection_pool=_get_pool())
    return _client


async def close_pool() -> None:
    """
    关闭连接池，供 lifespan shutdown 阶段调用。
    """
    global _pool, _client
    if _client is not None:
        await _client.aclose()
        _client = None
    if _pool is not None:
        await _pool.aclose()
        _pool = None
    logger.info("Redis 连接池已关闭")


# ---------------------------------------------------------------------------
# Key 工具
# ---------------------------------------------------------------------------

def build_key(*parts: str) -> str:
    """
    构造带命名空间前缀的 Redis Key。

    示例：
        build_key("session", "abc123")  →  "synaris:session:abc123"
        build_key("embedding", "sha256:deadbeef")  →  "synaris:embedding:sha256:deadbeef"
    """
    segments = [KEY_NAMESPACE] + [str(p) for p in parts]
    return ":".join(segments)


# ---------------------------------------------------------------------------
# 内部异常转换装饰器
# ---------------------------------------------------------------------------

def _handle_redis_errors(operation: str):
    """
    将 redis-py 底层异常统一转换为项目自定义异常，并写入结构化日志。
    用作上下文管理器（async with）。
    """
    class _Ctx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                return False
            if exc_type in (RedisConnectionError, RedisTimeoutError):
                logger.error(
                    "Redis 连接失败",
                    extra={"operation": operation, "error": str(exc_val)},
                )
                raise CacheConnectionError(
                    f"Redis 连接失败（{operation}）: {exc_val}"
                ) from exc_val
            if issubclass(exc_type, RedisError):
                logger.error(
                    "Redis 操作失败",
                    extra={"operation": operation, "error": str(exc_val)},
                )
                raise CacheError(
                    f"Redis 操作失败（{operation}）: {exc_val}"
                ) from exc_val
            return False  # 其他异常不拦截

    return _Ctx()


# ---------------------------------------------------------------------------
# 核心异步方法
# ---------------------------------------------------------------------------

async def get(key: str) -> Optional[str]:
    """
    获取字符串类型的值。Key 不存在时返回 None。
    """
    async with _handle_redis_errors("GET"):
        value = await get_client().get(key)
        logger.debug("Redis GET", extra={"key": key, "hit": value is not None})
        return value


async def set(key: str, value: str, ttl: Optional[int] = None) -> bool:
    """
    写入字符串值。
    ttl: 过期秒数，None 表示永不过期。
    返回 True 表示写入成功。
    """
    async with _handle_redis_errors("SET"):
        ok = await get_client().set(key, value, ex=ttl)
        logger.debug("Redis SET", extra={"key": key, "ttl": ttl, "ok": bool(ok)})
        return bool(ok)


async def delete(key: str) -> int:
    """
    删除 Key。返回实际删除的 Key 数量（0 或 1）。
    """
    async with _handle_redis_errors("DELETE"):
        count = await get_client().delete(key)
        logger.debug("Redis DELETE", extra={"key": key, "deleted": count})
        return count


async def exists(key: str) -> bool:
    """
    判断 Key 是否存在。
    """
    async with _handle_redis_errors("EXISTS"):
        result = await get_client().exists(key)
        return bool(result)


# ---------------------------------------------------------------------------
# JSON 序列化辅助
# ---------------------------------------------------------------------------

async def get_json(key: str) -> Optional[Any]:
    """
    从 Redis 读取并反序列化 JSON 值。
    Key 不存在或反序列化失败时返回 None。
    """
    async with _handle_redis_errors("GET_JSON"):
        raw = await get_client().get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "Redis JSON 反序列化失败，返回 None",
                extra={"key": key, "error": str(exc)},
            )
            return None


async def set_json(
    key: str,
    obj: Any,
    ttl: Optional[int] = None,
) -> bool:
    """
    将 Python 对象序列化为 JSON 后写入 Redis。
    支持 datetime、UUID 等常见类型（通过 default=str 兜底）。
    """
    async with _handle_redis_errors("SET_JSON"):
        try:
            payload = json.dumps(obj, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.error(
                "Redis JSON 序列化失败",
                extra={"key": key, "error": str(exc)},
            )
            raise CacheError(f"JSON 序列化失败（{key}）: {exc}") from exc

        ok = await get_client().set(key, payload, ex=ttl)
        logger.debug("Redis SET_JSON", extra={"key": key, "ttl": ttl, "ok": bool(ok)})
        return bool(ok)


# ---------------------------------------------------------------------------
# Pub/Sub
# ---------------------------------------------------------------------------

async def publish(channel: str, message: Any) -> int:
    """
    向指定 channel 发布消息。
    message 可为 str 或任意可 JSON 序列化对象。
    返回接收到该消息的订阅者数量。
    """
    async with _handle_redis_errors("PUBLISH"):
        payload = (
            message
            if isinstance(message, str)
            else json.dumps(message, ensure_ascii=False, default=str)
        )
        receivers = await get_client().publish(channel, payload)
        logger.debug(
            "Redis PUBLISH",
            extra={"channel": channel, "receivers": receivers},
        )
        return receivers


async def subscribe(channel: str) -> AsyncIterator[Any]:
    """
    订阅指定 channel，以异步生成器形式逐条 yield 消息内容。

    典型用法（WebSocket 推送）：
        async for message in subscribe("agent:task:abc123"):
            await websocket.send_text(message)

    注意：此函数会持续阻塞直到调用方退出迭代或出现连接错误。
    每次调用会创建独立的 PubSub 连接（不复用主连接池）。
    """
    # PubSub 需要独立连接，不能共享主客户端连接
    pubsub_client: aioredis.Redis = aioredis.Redis.from_pool(_get_pool())
    ps: PubSub = pubsub_client.pubsub()

    try:
        await ps.subscribe(channel)
        logger.info("Redis 订阅已建立", extra={"channel": channel})

        async for raw_message in ps.listen():
            # redis-py 会先推送一条 type=="subscribe" 的确认消息，跳过
            if raw_message["type"] != "message":
                continue

            data = raw_message.get("data", "")
            # 尝试自动 JSON 解析，失败则原样返回字符串
            try:
                yield json.loads(data)
            except (json.JSONDecodeError, TypeError):
                yield data

    except (RedisConnectionError, RedisTimeoutError) as exc:
        logger.error(
            "Redis Pub/Sub 连接断开",
            extra={"channel": channel, "error": str(exc)},
        )
        raise CacheConnectionError(
            f"Redis Pub/Sub 连接断开（channel={channel}）: {exc}"
        ) from exc
    finally:
        await ps.unsubscribe(channel)
        await ps.aclose()
        await pubsub_client.aclose()
        logger.info("Redis 订阅已关闭", extra={"channel": channel})


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

async def ping() -> bool:
    """
    向 Redis 发送 PING 命令，验证连接是否正常。
    返回 True 表示健康，False 表示异常（不抛异常，适合 /health/detailed 调用）。
    """
    try:
        result = await get_client().ping()
        return bool(result)
    except (RedisConnectionError, RedisTimeoutError, RedisError) as exc:
        logger.warning("Redis PING 失败", extra={"error": str(exc)})
        return False


# ---------------------------------------------------------------------------
# FastAPI 依赖注入
# ---------------------------------------------------------------------------

async def get_redis_client() -> AsyncIterator[aioredis.Redis]:
    """
    FastAPI Depends 兼容的依赖注入工厂。

    用法：
        @router.get("/example")
        async def example(redis: aioredis.Redis = Depends(get_redis_client)):
            value = await redis.get("some_key")
    """
    client = get_client()
    try:
        yield client
    except (RedisConnectionError, RedisTimeoutError) as exc:
        logger.error("Redis 连接异常（依赖注入上下文）", extra={"error": str(exc)})
        raise CacheConnectionError(f"Redis 连接异常: {exc}") from exc

"""
（1）代码结构

redis_client.py
├── 连接池管理
│   ├── _get_pool()        懒初始化单例连接池
│   ├── get_client()       获取共享客户端
│   └── close_pool()       lifespan shutdown 清理
│
├── Key 工具
│   └── build_key(*parts)  "synaris:session:abc123"
│
├── 异常转换
│   └── _handle_redis_errors()  RedisError → CacheError/CacheConnectionError
│
├── 基础 KV 操作
│   ├── get / set / delete / exists
│   └── get_json / set_json
│
├── Pub/Sub
│   ├── publish(channel, message)   → int（接收者数）
│   └── subscribe(channel)          → AsyncIterator（独立连接）
│
├── 健康检查
│   └── ping() → bool
│
└── FastAPI 依赖注入
    └── get_redis_client() → AsyncIterator[Redis]

（2）设计决策

Pub/Sub 用独立连接：
Redis 协议要求订阅模式下连接不能再发送普通命令，
因此 subscribe() 每次创建独立的 PubSub 连接，与主连接池隔离，WebSocket 推流结束后自动释放。

decode_responses=True：
连接池统一设置字符串解码，消除代码里的 bytes.decode() 噪音，
get_json / set_json 直接拿到 str 进行 JSON 处理。

_handle_redis_errors 上下文管理器：
集中处理所有 redis-py 异常，避免每个方法重复写 try/except，
同时保证日志结构统一（operation 字段标识来源）。

get_redis_client() 不关闭客户端：
共享连接池模式下，每次请求不应 aclose() 客户端，连接归还连接池由 redis-py 内部管理。

"""