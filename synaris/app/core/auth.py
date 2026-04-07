"""
@File       : auth.py
@Author     : Starry Hung
@Created    : 2026-04-05
@Version    : 1.0.0
@Description: JWT Token + API Key 双认证体系。
@Features:
  ── JWT 认证 ──────────────────────────────────────────────────────────────
  - create_access_token(user_id, role, extra_claims) → 签发 JWT，有效期 24h
  - create_refresh_token(user_id) → 签发刷新 Token，有效期 7d
  - verify_token(token) → TokenPayload（解码并验证签名/有效期/吊销状态）
  - refresh_access_token(refresh_token) → 新 access_token（无感续期）
  - 密钥轮转：从 settings 读取 SECRET_KEY_CURRENT + SECRET_KEY_PREVIOUS，
              验证时依次尝试，实现平滑密钥切换（不强制用户重新登录）
  - Token 吊销：revoke_token(jti) → 将 jti 写入 Redis blacklist（TTL=剩余有效期）

  ── API Key 认证 ──────────────────────────────────────────────────────────
  - generate_api_key() → (raw_key, key_hash, key_prefix)
    raw_key: 32字节随机 hex（格式 sk-syn_{hex}），只返回一次
    key_hash: bcrypt 哈希，存入 PostgreSQL
    key_prefix: 前缀（如 sk-syn_a1b2c3d4），用于列表展示
  - verify_api_key(raw_key, session) → APIKey ORM 对象
    查询前缀匹配 + bcrypt.checkpw 验证，成功后更新 last_used_at

  ── FastAPI 依赖注入 ───────────────────────────────────────────────────────
  - get_current_user(token) → User（JWT 路径，失败抛 401）
  - get_api_key_user(api_key) → User（API Key 路径，失败抛 401）
  - get_current_user_flexible() → User（自动识别 Bearer/X-API-Key，任一有效即可）
  - optional_auth() → User | None（允许匿名访问，无凭证返回 None）

  ── 权限装饰器 ────────────────────────────────────────────────────────────
  - require_role(*roles) → FastAPI Depends 工厂函数（用于路由级权限控制）

  ── 安全规范 ──────────────────────────────────────────────────────────────
  - 所有认证失败统一返回 401，响应体不包含失败原因细节（防止枚举攻击）
  - bcrypt work factor=12（平衡安全性与性能）
  - JWT 使用 HS256 算法，payload 最小化（不含敏感字段）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-05  Starry  Initial creation
"""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Optional

import bcrypt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.postgres_client import get_db_session  # type: ignore[import]
from models.user import APIKey, User, UserRole             # type: ignore[import]

logger = logging.getLogger(__name__)

# ── 常量 ───────────────────────────────────────────────────────────────────────
_JWT_ALGORITHM     = "HS256"
_ACCESS_TOKEN_TTL  = timedelta(hours=24)
_REFRESH_TOKEN_TTL = timedelta(days=7)
_BCRYPT_ROUNDS     = 12          # work factor：2^12 次哈希迭代，约 300ms/次
_API_KEY_PREFIX    = "sk-syn_"   # 格式：sk-syn_{64位hex}，总长 72 字符
_API_KEY_BYTES     = 32          # 随机字节数（32 bytes = 256 bits 熵）
_BLACKLIST_PREFIX  = "auth:blacklist:{}"   # Redis key 格式

# FastAPI 安全方案声明（供 Swagger UI 展示）
_bearer_scheme  = HTTPBearer(auto_error=False)
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


# ─────────────────────────────────────────────
# 1. 配置读取（延迟加载，避免循环依赖）
# ─────────────────────────────────────────────

def _get_secret_keys() -> tuple[str, list[str]]:
    """
    返回 (current_key, all_keys)。

    all_keys 按优先级排列：当前密钥在前，历史密钥在后。
    验证 Token 时依次尝试，签发 Token 只用 current_key。

    密钥轮转流程：
      1. 生成新密钥 → 写入 SECRET_KEY_CURRENT
      2. 旧密钥 → 写入 SECRET_KEY_PREVIOUS（保留 24h，等旧 Token 自然过期）
      3. 24h 后 → 清空 SECRET_KEY_PREVIOUS
    """
    try:
        from config.settings import get_settings  # type: ignore[import]
        s = get_settings()
        current = getattr(s, "secret_key", None) or os.getenv("SECRET_KEY", "")
        previous_raw = getattr(s, "secret_key_previous", None) or os.getenv("SECRET_KEY_PREVIOUS", "")
    except ImportError:
        current      = os.getenv("SECRET_KEY", "dev-insecure-key-change-me")
        previous_raw = os.getenv("SECRET_KEY_PREVIOUS", "")

    if not current:
        raise RuntimeError(
            "SECRET_KEY 未配置。请在 .env 文件中设置 SECRET_KEY（至少 32 字符随机字符串）。\n"
            "生成命令：openssl rand -hex 32"
        )

    previous_keys = [k.strip() for k in previous_raw.split(",") if k.strip()]
    all_keys = [current] + previous_keys
    return current, all_keys


def _get_token_ttl() -> dict[str, timedelta]:
    """从配置读取 Token 有效期（支持自定义，默认 access=24h / refresh=7d）。"""
    try:
        from config.settings import get_settings  # type: ignore[import]
        s = get_settings()
        access_hours  = getattr(s, "jwt_access_hours",  24)
        refresh_days  = getattr(s, "jwt_refresh_days",  7)
        return {
            "access":  timedelta(hours=access_hours),
            "refresh": timedelta(days=refresh_days),
        }
    except ImportError:
        return {"access": _ACCESS_TOKEN_TTL, "refresh": _REFRESH_TOKEN_TTL}


# ─────────────────────────────────────────────
# 2. Pydantic 模型：Token Payload
# ─────────────────────────────────────────────

class TokenPayload(BaseModel):
    """
    JWT Payload 结构体（最小化字段设计）。

    类比「身份证」：
      sub  → 证件号（user_id），主键
      role → 角色（admin/member/viewer），权限判断依据
      jti  → 证件流水号（Token 唯一 ID），用于吊销
      type → Token 类型（access / refresh），防止 refresh token 被用作 access
      exp  → 有效期（由 python-jose 自动验证）
      iat  → 签发时间

    不在 payload 中存储：用户名/邮箱/密码哈希等敏感信息（防止 payload 泄露）。
    """
    sub:  str            # user_id（UUID 字符串）
    role: str            # UserRole 字符串值
    jti:  str            # JWT ID（唯一标识，用于精确吊销）
    type: str            # "access" | "refresh"
    exp:  Optional[int]  = None   # 由 jose 自动处理
    iat:  Optional[int]  = None
    extra: dict[str, Any] = {}    # 扩展字段（如 session_id）


class TokenPair(BaseModel):
    """登录/刷新后返回的 Token 对。"""
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int            # access token 剩余秒数


class APIKeyCreateResult(BaseModel):
    """
    API Key 创建成功后的返回结构。

    raw_key 只返回一次，之后无法从数据库还原，用户必须妥善保存。
    """
    key_id:     str    # APIKey 记录的 UUID
    raw_key:    str    # 完整明文 Key（sk-syn_xxx...），只此一次
    key_prefix: str    # 前缀（用于列表展示）
    name:       str    # Key 名称


# ─────────────────────────────────────────────
# 3. JWT — 签发 / 验证 / 吊销
# ─────────────────────────────────────────────

def create_access_token(
    user_id:      str,
    role:         str,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """
    签发 JWT Access Token。

    Args:
        user_id:      用户 UUID 字符串
        role:         UserRole 枚举值字符串（"admin" / "member" / "viewer"）
        extra_claims: 额外 payload 字段（如 session_id，不放敏感信息）

    Returns:
        HS256 签名的 JWT 字符串
    """
    current_key, _ = _get_secret_keys()
    ttl = _get_token_ttl()["access"]
    now = datetime.now(timezone.utc)

    payload: dict[str, Any] = {
        "sub":   user_id,
        "role":  role,
        "jti":   str(uuid.uuid4()),
        "type":  "access",
        "iat":   int(now.timestamp()),
        "exp":   int((now + ttl).timestamp()),
    }
    if extra_claims:
        # 只允许非敏感的扩展字段，过滤掉可能覆盖标准字段的 key
        reserved = {"sub", "role", "jti", "type", "iat", "exp"}
        payload.update({k: v for k, v in extra_claims.items() if k not in reserved})

    return jwt.encode(payload, current_key, algorithm=_JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """
    签发 JWT Refresh Token（有效期 7d，payload 最小化）。

    Refresh Token 只包含 sub/jti/type/exp，不含 role。
    刷新时需要重新从数据库读取用户信息，确保角色变更即时生效。
    """
    current_key, _ = _get_secret_keys()
    ttl = _get_token_ttl()["refresh"]
    now = datetime.now(timezone.utc)

    payload: dict[str, Any] = {
        "sub":  user_id,
        "jti":  str(uuid.uuid4()),
        "type": "refresh",
        "iat":  int(now.timestamp()),
        "exp":  int((now + ttl).timestamp()),
    }
    return jwt.encode(payload, current_key, algorithm=_JWT_ALGORITHM)


def verify_token(token: str, expected_type: str = "access") -> TokenPayload:
    """
    验证 JWT Token 并返回 Payload。

    验证顺序：
      1. 依次用当前密钥 + 历史密钥尝试解码（支持密钥轮转过渡期）
      2. 检查 type 字段（防止 refresh token 当 access token 使用）
      3. 检查 Redis 黑名单（jti 是否已被吊销）

    所有验证失败都抛出同一个 401 异常，不透露具体原因（防止枚举攻击）。

    Raises:
        HTTPException 401：Token 无效、过期、类型错误或已吊销
    """
    _, all_keys = _get_secret_keys()

    payload_dict: Optional[dict] = None
    for key in all_keys:
        try:
            payload_dict = jwt.decode(token, key, algorithms=[_JWT_ALGORITHM])
            break   # 解码成功，不再尝试后续密钥
        except ExpiredSignatureError:
            # Token 过期，直接抛出（不需要尝试其他密钥）
            logger.info("verify_token: Token 已过期")
            raise _auth_exception()
        except JWTError:
            continue   # 当前密钥不匹配，尝试下一个

    if payload_dict is None:
        logger.warning("verify_token: 所有密钥均无法解码 Token")
        raise _auth_exception()

    # 类型检查（防止 refresh token 冒充 access token）
    if payload_dict.get("type") != expected_type:
        logger.warning(
            "verify_token: Token 类型不匹配 | expected=%s | got=%s",
            expected_type, payload_dict.get("type"),
        )
        raise _auth_exception()

    # Redis 黑名单检查（已吊销的 Token）
    jti = payload_dict.get("jti", "")
    if jti and _is_token_revoked(jti):
        logger.info("verify_token: Token 已被吊销 | jti=%s", jti)
        raise _auth_exception()

    return TokenPayload(
        sub=payload_dict.get("sub", ""),
        role=payload_dict.get("role", UserRole.VIEWER.value),
        jti=jti,
        type=payload_dict.get("type", ""),
        exp=payload_dict.get("exp"),
        iat=payload_dict.get("iat"),
        extra={k: v for k, v in payload_dict.items()
               if k not in {"sub", "role", "jti", "type", "exp", "iat"}},
    )


def revoke_token(jti: str, remaining_seconds: int) -> None:
    """
    将 Token 的 jti 写入 Redis 黑名单（TTL = Token 剩余有效期）。

    用于以下场景：
      - 用户主动登出（Logout）
      - 管理员强制下线某用户
      - 密码修改后使旧 Token 立即失效

    TTL 设为剩余有效期而非固定值，确保黑名单不会无限膨胀。
    """
    try:
        import asyncio
        from infrastructure.redis_client import get_redis  # type: ignore[import]

        async def _revoke():
            redis = await get_redis()
            key   = _BLACKLIST_PREFIX.format(jti)
            await redis.set(key, "1", ex=max(1, remaining_seconds))

        # 同步上下文中触发异步操作
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_revoke())
            else:
                loop.run_until_complete(_revoke())
        except RuntimeError:
            asyncio.run(_revoke())

    except ImportError:
        logger.warning("revoke_token: Redis 未就绪，Token 吊销将在 TTL 到期后自动失效")


def _is_token_revoked(jti: str) -> bool:
    """
    同步检查 jti 是否在 Redis 黑名单中。

    在 FastAPI 的异步依赖中使用时，此处用同步方式查询以避免事件循环嵌套问题。
    对于极高吞吐量场景，可升级为异步版本。
    """
    try:
        import asyncio
        from infrastructure.redis_client import get_redis  # type: ignore[import]

        async def _check() -> bool:
            redis = await get_redis()
            key   = _BLACKLIST_PREFIX.format(jti)
            value = await redis.get(key)
            return value is not None

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在异步上下文中，创建新线程执行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, _check()).result(timeout=1.0)
            else:
                return loop.run_until_complete(_check())
        except RuntimeError:
            return asyncio.run(_check())

    except Exception as exc:
        logger.warning("_is_token_revoked: Redis 查询失败，跳过黑名单检查: %s", exc)
        return False   # Redis 不可用时降级为不检查黑名单（可用性优先）


async def refresh_access_token(
    refresh_token: str,
    session: AsyncSession,
) -> TokenPair:
    """
    使用 Refresh Token 换取新的 Token 对（无感续期）。

    流程：
      1. 验证 Refresh Token（type=refresh）
      2. 从数据库查询用户（确保获取最新的 role/is_active 状态）
      3. 吊销旧 Refresh Token（防止重放攻击）
      4. 签发新的 Access Token + Refresh Token

    Args:
        refresh_token: 客户端持有的 Refresh Token 字符串
        session:       数据库 Session（用于重新查询用户最新状态）

    Returns:
        TokenPair（新的 access_token + refresh_token）

    Raises:
        HTTPException 401：Refresh Token 无效或用户不存在/已被禁用
    """
    payload = verify_token(refresh_token, expected_type="refresh")

    # 重新从数据库获取用户（确保角色/状态是最新的）
    result = await session.execute(
        select(User).where(
            User.id == uuid.UUID(payload.sub),
            User.is_deleted.is_(False),
        )
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        logger.warning("refresh_access_token: 用户不存在或已被禁用 | user_id=%s", payload.sub)
        raise _auth_exception()

    # 吊销旧 Refresh Token（防止同一 Token 被多次使用）
    remaining = (payload.exp or 0) - int(datetime.now(timezone.utc).timestamp())
    if remaining > 0:
        revoke_token(payload.jti, remaining)

    # 签发新 Token 对
    ttl = _get_token_ttl()
    new_access  = create_access_token(str(user.id), user.role)
    new_refresh = create_refresh_token(str(user.id))

    return TokenPair(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=int(ttl["access"].total_seconds()),
    )


# ─────────────────────────────────────────────
# 4. API Key — 生成 / 验证 / 更新
# ─────────────────────────────────────────────

def generate_api_key() -> tuple[str, str, str]:
    """
    生成一个新的 API Key，返回 (raw_key, key_hash, key_prefix)。

    格式：sk-syn_{64位hex}（总长 72 字符，前缀固定，后缀 32 字节随机）

    安全设计：
      raw_key  → 只在调用方此刻拿到，之后无法还原
      key_hash → bcrypt 哈希（work_factor=12），存入 PostgreSQL
      key_prefix → 前 8 位，用于列表展示识别（sk-syn_a1b2...）

    Returns:
        (raw_key, key_hash, key_prefix)
    """
    random_hex = secrets.token_hex(_API_KEY_BYTES)   # 64 位 hex 字符串
    raw_key    = f"{_API_KEY_PREFIX}{random_hex}"     # sk-syn_xxxxxxxx...

    # bcrypt 哈希（cost factor=12，约 300ms，对登录性能可接受，对暴力破解成本极高）
    key_hash   = bcrypt.hashpw(raw_key.encode("utf-8"), bcrypt.gensalt(_BCRYPT_ROUNDS)).decode("utf-8")

    # 前缀：_API_KEY_PREFIX（8字符）+ 随机部分前8字符 = 16字符，用于展示
    key_prefix = raw_key[:len(_API_KEY_PREFIX) + 8]

    return raw_key, key_hash, key_prefix


async def verify_api_key(
    raw_key: str,
    session: AsyncSession,
) -> APIKey:
    """
    验证 API Key 并返回对应的 APIKey ORM 对象。

    查询策略（两步验证）：
      Step 1：按 key_prefix 缩小候选集（前缀匹配，避免全表 bcrypt 比对）
      Step 2：对每个候选记录做 bcrypt.checkpw（精确验证）

    为什么不直接哈希后比对？
      bcrypt 是非确定性哈希（每次 salt 不同），同一明文产生不同哈希，
      无法通过哈希值相等来查询，只能用 checkpw 逐条验证。

    Args:
        raw_key: 请求头 X-API-Key 的值
        session: 数据库 Session

    Returns:
        有效的 APIKey ORM 对象

    Raises:
        HTTPException 401：Key 无效、已禁用或已过期
    """
    if not raw_key or not raw_key.startswith(_API_KEY_PREFIX):
        raise _auth_exception()

    # Step 1：按前缀查找候选 Key（通常只有 1-2 条，极少全表扫描）
    prefix = raw_key[:len(_API_KEY_PREFIX) + 8]
    result = await session.execute(
        select(APIKey)
        .where(
            APIKey.key_prefix == prefix,
            APIKey.is_deleted.is_(False),
        )
        .options()   # 不懒加载关联，减少额外查询
    )
    candidates = result.scalars().all()

    if not candidates:
        logger.info("verify_api_key: 未找到匹配前缀的 Key | prefix=%s", prefix)
        raise _auth_exception()

    # Step 2：对候选集做 bcrypt 精确验证
    matched_key: Optional[APIKey] = None
    for api_key in candidates:
        try:
            if bcrypt.checkpw(raw_key.encode("utf-8"), api_key.key_hash.encode("utf-8")):
                matched_key = api_key
                break
        except Exception:
            continue   # 哈希格式异常，跳过

    if matched_key is None:
        logger.warning("verify_api_key: bcrypt 验证失败 | prefix=%s", prefix)
        raise _auth_exception()

    # 检查有效期和启用状态
    if not matched_key.is_valid():
        logger.info(
            "verify_api_key: Key 已失效 | key_id=%s | is_active=%s | expires_at=%s",
            matched_key.id, matched_key.is_active, matched_key.expires_at,
        )
        raise _auth_exception()

    # 异步更新最后使用时间和请求计数（不阻塞响应，fire and forget）
    try:
        await session.execute(
            update(APIKey)
            .where(APIKey.id == matched_key.id)
            .values(
                last_used_at=datetime.now(timezone.utc),
                request_count=APIKey.request_count + 1,
            )
        )
        # 注意：此处不 commit，由 get_db_session 上下文自动提交
    except Exception as exc:
        logger.warning("verify_api_key: 更新使用记录失败（非致命）: %s", exc)

    return matched_key


async def create_api_key_record(
    user_id: uuid.UUID,
    name:    str,
    session: AsyncSession,
    expires_at: Optional[datetime] = None,
) -> APIKeyCreateResult:
    """
    创建新的 API Key 记录（写入数据库）并返回含明文 Key 的结果。

    此函数是唯一能返回明文 raw_key 的地方，调用方必须立即展示给用户。

    Args:
        user_id:    所属用户 UUID
        name:       Key 用途名称（如"生产环境"）
        session:    数据库 Session
        expires_at: 可选过期时间（None=永不过期）

    Returns:
        APIKeyCreateResult（含 raw_key 明文，只返回一次）
    """
    raw_key, key_hash, key_prefix = generate_api_key()

    api_key = APIKey(
        user_id=user_id,
        name=name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        is_active=True,
        expires_at=expires_at,
    )
    session.add(api_key)
    await session.flush()   # 获取数据库生成的 id，但暂不提交

    logger.info(
        "create_api_key_record: 新 API Key 已创建 | user_id=%s | key_id=%s | name=%s",
        user_id, api_key.id, name,
    )

    return APIKeyCreateResult(
        key_id=str(api_key.id),
        raw_key=raw_key,    # ⚠️ 明文，仅此一次！
        key_prefix=key_prefix,
        name=name,
    )


# ─────────────────────────────────────────────
# 5. FastAPI 依赖注入
# ─────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """
    JWT Bearer Token 认证依赖。

    从 Authorization: Bearer <token> 头提取并验证 JWT，
    返回对应的 User ORM 对象。

    使用方式：
        @router.get("/profile")
        async def get_profile(user: User = Depends(get_current_user)):
            return {"username": user.username}
    """
    if not credentials or not credentials.credentials:
        raise _auth_exception()

    payload = verify_token(credentials.credentials, expected_type="access")

    # 从数据库加载用户（验证账号仍然存在且未被禁用）
    result = await session.execute(
        select(User).where(
            User.id == uuid.UUID(payload.sub),
            User.is_active.is_(True),
            User.is_deleted.is_(False),
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        logger.warning("get_current_user: 用户不存在或已被禁用 | user_id=%s", payload.sub)
        raise _auth_exception()

    return user


async def get_api_key_user(
    api_key: Optional[str] = Security(_api_key_scheme),
    session: AsyncSession  = Depends(get_db_session),
) -> User:
    """
    API Key 认证依赖。

    从 X-API-Key 请求头提取 Key 并验证，返回对应的 User ORM 对象。

    使用方式：
        @router.post("/data")
        async def create_data(user: User = Depends(get_api_key_user)):
            ...
    """
    if not api_key:
        raise _auth_exception()

    matched_key = await verify_api_key(api_key, session)

    # 加载关联用户（eager load，避免懒加载在异步上下文中失效）
    result = await session.execute(
        select(User).where(
            User.id == matched_key.user_id,
            User.is_active.is_(True),
            User.is_deleted.is_(False),
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        logger.warning(
            "get_api_key_user: API Key 对应的用户不存在或已被禁用 | key_id=%s",
            matched_key.id,
        )
        raise _auth_exception()

    return user


async def get_current_user_flexible(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    api_key:     Optional[str]                          = Security(_api_key_scheme),
    session:     AsyncSession                           = Depends(get_db_session),
) -> User:
    """
    双路认证依赖：自动识别 Bearer JWT 或 X-API-Key，任一有效即可。

    优先级：Bearer JWT > X-API-Key

    适用于同时支持两种客户端的接口（如既有 Web 端用 JWT，又有三方系统用 API Key）。

    使用方式：
        @router.post("/agent/run")
        async def run_agent(user: User = Depends(get_current_user_flexible)):
            ...
    """
    # 优先尝试 JWT
    if credentials and credentials.credentials:
        try:
            return await get_current_user(credentials=credentials, session=session)
        except HTTPException:
            pass   # JWT 失败，尝试 API Key

    # 尝试 API Key
    if api_key:
        try:
            return await get_api_key_user(api_key=api_key, session=session)
        except HTTPException:
            pass

    raise _auth_exception()


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    api_key:     Optional[str]                          = Security(_api_key_scheme),
    session:     AsyncSession                           = Depends(get_db_session),
) -> Optional[User]:
    """
    可选认证依赖：有凭证则验证并返回 User，无凭证则返回 None（允许匿名访问）。

    适用场景：公开接口（无需登录），但登录后可获得额外功能（如个性化、更高限速额度）。

    使用方式：
        @router.get("/knowledge/search")
        async def search(
            q: str,
            user: Optional[User] = Depends(optional_auth),
        ):
            limit = 20 if user else 5  # 登录用户返回更多结果
            ...
    """
    if not credentials and not api_key:
        return None

    try:
        return await get_current_user_flexible(
            credentials=credentials, api_key=api_key, session=session
        )
    except HTTPException:
        return None   # 凭证无效时，可选认证不抛异常，直接返回 None


# ─────────────────────────────────────────────
# 6. 权限装饰器
# ─────────────────────────────────────────────

def require_role(*required_roles: str) -> Callable:
    """
    路由级权限控制工厂函数（返回 FastAPI Depends 兼容的依赖）。

    权限继承：admin > member > viewer
    即 require_role("member") 对 admin 用户也放行。

    使用方式（两种等价写法）：
        # 写法 A：直接在 Depends 中使用
        @router.post("/upload")
        async def upload(user: User = Depends(require_role("member"))):
            ...

        # 写法 B：用作装饰器（需要 FastAPI 路由级注入支持）
        AdminDep = Depends(require_role("admin"))

        @router.delete("/users/{uid}")
        async def delete_user(user: User = AdminDep):
            ...

    Args:
        *required_roles: 允许访问的最低角色列表（满足其一即可）

    Returns:
        可被 Depends() 包裹的异步函数，返回经过权限验证的 User
    """
    # 角色优先级映射
    _ROLE_LEVELS: dict[str, int] = {
        UserRole.ADMIN.value:  3,
        UserRole.MEMBER.value: 2,
        UserRole.VIEWER.value: 1,
        # 兼容请求参数中的别名写法
        "admin":    3,
        "user":     2,
        "member":   2,
        "readonly": 1,
        "viewer":   1,
    }

    # 计算所需的最低权限级别
    required_level = max((_ROLE_LEVELS.get(r, 0) for r in required_roles), default=1)

    async def _check_role(
        user: User = Depends(get_current_user_flexible),
    ) -> User:
        user_level = _ROLE_LEVELS.get(user.role, 0)

        if user_level < required_level:
            logger.info(
                "require_role: 权限不足 | user_id=%s | user_role=%s | required=%s",
                user.id, user.role, required_roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": 40300,
                    "message": "权限不足，无法访问此资源",
                    "required_roles": list(required_roles),
                },
            )
        return user

    # 保留原始函数名，便于 FastAPI 生成 OpenAPI 文档时识别
    _check_role.__name__ = f"require_role_{'_or_'.join(required_roles)}"
    return _check_role


# ─────────────────────────────────────────────
# 7. 辅助工具
# ─────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """
    使用 bcrypt 对密码进行哈希（注册/修改密码时调用）。

    Args:
        plain_password: 用户输入的明文密码

    Returns:
        bcrypt 哈希字符串（包含 salt，可直接存入数据库）
    """
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt(_BCRYPT_ROUNDS),
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码是否与哈希值匹配（登录时调用）。

    Args:
        plain_password:   用户输入的明文密码
        hashed_password:  数据库中存储的 bcrypt 哈希

    Returns:
        True=密码正确，False=密码错误
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def _auth_exception() -> HTTPException:
    """
    生成统一的 401 认证失败异常。

    所有认证失败（Token 无效/过期/吊销、API Key 错误/禁用/过期）
    都返回完全相同的响应，防止攻击者通过错误信息细节推断账号状态。

    WWW-Authenticate 头是 HTTP 401 规范要求的标准头，
    告知客户端支持的认证方式。
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": 40100, "message": "认证失败，请检查凭证是否有效"},
        headers={"WWW-Authenticate": "Bearer"},
    )