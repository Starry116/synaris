"""
Synaris · 赛纳睿智能平台
app/config/settings.py — 全局配置管理模块

设计决策说明：
  ┌─────────────────────────────────────────────────────────────────┐
  │  原则                     │  实现方式                           │
  ├─────────────────────────────────────────────────────────────────┤
  │  DRY（不重复自己）         │  _BaseEnvSettings 公共基类          │
  │  Fail Fast（尽早失败）     │  必填项无默认值 + @field_validator   │
  │  可读性                   │  TTL 用乘法（10 * 60）而非魔法数字   │
  │  类型安全                  │  枚举字段用 Literal 约束             │
  │  防意外修改                │  frozen=True 配置不可变              │
  │  防日志泄露                │  敏感字段用 SecretStr               │
  │  全进程单例                │  @lru_cache(maxsize=1)              │
  └─────────────────────────────────────────────────────────────────┘

  不采纳的"常见建议"及原因：
  - env_prefix：字段已带前缀（MILVUS_HOST），再加 prefix 会产生双重前缀 bug
  - Redis 拆成 host/port/password：Synaris 用 Docker Compose 固定地址，URL 已够用
  - MODEL_ROUTER 用 dict：env 里写 JSON 极难维护，改用独立字段 + llm_router 组装
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# 公共基类
# ─────────────────────────────────────────────────────────────────────────────

class _BaseEnvSettings(BaseSettings):
    """所有配置分组的公共基类。

    关键规则：子类绝对不能再声明自己的 model_config。
    pydantic-settings 的 model_config 是「覆盖」语义而非「合并」语义，
    子类一旦声明就会把父类配置整体覆盖，导致 .env 读取失效。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",    # 忽略不属于本分组的字段，避免跨组互相报错
        frozen=True,       # 配置实例不可变，防止运行期被某个模块意外篡改
    )


# ─────────────────────────────────────────────────────────────────────────────
# AppConfig — 应用基础配置
# ─────────────────────────────────────────────────────────────────────────────

class AppConfig(_BaseEnvSettings):
    """FastAPI 应用基础配置。"""

    APP_NAME: str = "Synaris"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # 运行环境：控制日志级别、数据库选择等下游行为
    ENVIRONMENT: Literal["development", "staging", "production"] = "production"

    # 速率限制（slowapi 格式：次数/时间单位）
    RATE_LIMIT: str = "60/minute"

    # CORS 允许的前端来源，多个用英文逗号分隔
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"

    @property
    def cors_origins_list(self) -> list[str]:
        """解析为列表，直接传给 FastAPI CORSMiddleware。"""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIConfig — 大模型接口配置
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIConfig(_BaseEnvSettings):
    """OpenAI API 及多模型路由配置。

    模型三档策略：
      QUALITY  → gpt-4o          用于 Agent 推理、复杂分析
      DEFAULT  → gpt-4o-mini     用于日常对话（BALANCED 策略）
      ECONOMY  → gpt-3.5-turbo   用于简单任务、批量处理

    FALLBACK_MODEL 是主模型不可用时的兜底，必须比 DEFAULT_MODEL 更便宜、
    可用性更高，不能设置为比默认模型更贵的模型。
    """

    # SecretStr：值在内存中加密存储，str() 输出 '**********'，防止日志泄露
    OPENAI_API_KEY: SecretStr = Field(description="OpenAI API 密钥，必填")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_TIMEOUT: int = 60       # 单次请求超时（秒）
    OPENAI_MAX_RETRIES: int = 3    # 请求失败最大重试次数

    # 模型路由（三档 + 兜底），在 core/llm_router.py 中按任务类型组装路由逻辑
    QUALITY_MODEL: str = "gpt-4o"
    DEFAULT_MODEL: str = "gpt-4o-mini"
    ECONOMY_MODEL: str = "gpt-3.5-turbo"
    FALLBACK_MODEL: str = "gpt-3.5-turbo"   # 兜底必须 ≤ DEFAULT_MODEL 的成本

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536   # 必须与 Milvus Collection Schema 一致

    @field_validator("OPENAI_API_KEY", mode="before")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """启动阶段校验，而不是第一次 API 调用时才发现问题（Fail Fast）。"""
        if not v or not str(v).strip():
            raise ValueError(
                "OPENAI_API_KEY 未配置，请在 .env 文件中填写"
            )
        if not str(v).startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY 格式错误，合法的 key 以 'sk-' 开头"
            )
        return v

    def get_api_key(self) -> str:
        """获取 API Key 明文，仅在实际调用 OpenAI 时使用。"""
        return self.OPENAI_API_KEY.get_secret_value()


# ─────────────────────────────────────────────────────────────────────────────
# MilvusConfig — 向量数据库配置
# ─────────────────────────────────────────────────────────────────────────────

class MilvusConfig(_BaseEnvSettings):
    """Milvus 向量数据库配置。

    HNSW 索引参数说明：
      M=16              图的双向链接数，越大召回率越高，内存占用越大
      efConstruction=256 构建索引时的搜索宽度，越大质量越高但构建越慢
      search_ef=64      查询时的搜索宽度，越大召回率越高但延迟越高
    """

    # Docker 环境用服务名 milvus，本地开发改为 localhost
    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: int = 19530

    # Collection 名称
    MILVUS_COLLECTION_NAME: str = "synaris_knowledge"
    MILVUS_MEMORY_COLLECTION: str = "synaris_long_term_memory"

    # HNSW 索引参数
    MILVUS_HNSW_M: int = 16
    MILVUS_HNSW_EF_CONSTRUCTION: int = 256
    MILVUS_SEARCH_EF: int = 64

    # 检索参数
    MILVUS_DEFAULT_TOP_K: int = 5
    MILVUS_SIMILARITY_THRESHOLD: float = 0.70

    @field_validator("MILVUS_SIMILARITY_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"MILVUS_SIMILARITY_THRESHOLD={v} 超出范围，必须在 [0.0, 1.0] 之间"
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# RedisConfig — 缓存与消息队列配置
# ─────────────────────────────────────────────────────────────────────────────

class RedisConfig(_BaseEnvSettings):
    """Redis 缓存、会话与 Pub/Sub 配置。

    为什么用 URL 而不拆成 host/port/password：
    Synaris 基于 Docker Compose 部署，Redis 地址固定，
    redis.asyncio 直接接受 URL 字符串，无需手动拼接。
    拆分只在需要动态切换多实例时才有价值，当前阶段属于过度设计。

    为什么用两个不同的 db（/0 和 /1）：
    缓存数据和 Celery 队列数据隔离，flush 或 debug 时互不影响。
    """

    REDIS_URL: str = "redis://redis:6379/0"           # 主缓存（Docker 服务名 redis）
    REDIS_CELERY_URL: str = "redis://redis:6379/1"    # Celery Broker，独立 db 隔离
    REDIS_MAX_CONNECTIONS: int = 20

    # ── TTL 配置（单位：秒）──────────────────────────────────────────────────
    # 用乘法表达：10 * 60 比魔法数字 600 更直观，reviewer 一眼能看出业务意图
    CACHE_TTL_SHORT: int = 10 * 60            # 600s    检索结果缓存
    CACHE_TTL_LONG: int = 24 * 60 * 60        # 86400s  Embedding 缓存
    CACHE_TTL_SESSION: int = 2 * 60 * 60      # 7200s   聊天历史缓存
    CACHE_TTL_TASK: int = 1 * 60 * 60         # 3600s   任务上下文缓存

    # 聊天历史最大保留条数，超出后自动丢弃最早的消息（滑动窗口）
    CHAT_HISTORY_MAX_MESSAGES: int = 20

    # ── Key 命名空间前缀────────────────────────────────────────────────────
    # 统一管理所有 Redis key 前缀，防止不同业务模块的 key 互相碰撞
    KEY_PREFIX_EMBEDDING: str = "emb"
    KEY_PREFIX_SEARCH: str = "search"
    KEY_PREFIX_CHAT: str = "chat"
    KEY_PREFIX_TASK: str = "task"
    KEY_PREFIX_MEM_SHORT: str = "mem:short"
    KEY_PREFIX_MEM_TASK: str = "mem:task"


# ─────────────────────────────────────────────────────────────────────────────
# PostgresConfig — 结构化数据库配置
# ─────────────────────────────────────────────────────────────────────────────

class PostgresConfig(_BaseEnvSettings):
    """PostgreSQL 异步数据库配置（asyncpg + SQLAlchemy 2.0）。"""

    POSTGRES_URL: str = (
        "postgresql+asyncpg://synaris:synaris_pass@postgres:5432/synaris"
    )
    POSTGRES_POOL_SIZE: int = 10        # 连接池常驻连接数
    POSTGRES_MAX_OVERFLOW: int = 20     # 超出池大小后允许的最大额外连接数
    POSTGRES_POOL_TIMEOUT: int = 30     # 等待可用连接的超时时间（秒）
    POSTGRES_ECHO: bool = False         # 是否打印 SQL，仅用于本地调试，生产必须 False


# ─────────────────────────────────────────────────────────────────────────────
# LogConfig — 日志配置
# ─────────────────────────────────────────────────────────────────────────────

class LogConfig(_BaseEnvSettings):
    """日志系统配置。

    LOG_FORMAT 说明：
      json → 生产环境 / ELK Filebeat 采集，结构化输出方便查询
      text → 本地开发，带颜色的可读格式
    """

    # Literal 约束：非法值（如 VERBOSE）在启动时立即 ValidationError，而非运行时静默失效
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # ELK / Prometheus 服务标识字段，用于多服务日志区分
    LOG_SERVICE_NAME: str = "synaris"
    LOG_ENVIRONMENT: Literal["development", "staging", "production"] = "production"


# ─────────────────────────────────────────────────────────────────────────────
# SecurityConfig — 认证与安全配置
# ─────────────────────────────────────────────────────────────────────────────

class SecurityConfig(_BaseEnvSettings):
    """JWT 与 API Key 认证配置。"""

    # SecretStr：防止 SECRET_KEY 出现在日志、异常信息、repr() 输出中
    SECRET_KEY: SecretStr = Field(description="JWT 签名密钥，必填")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    API_KEY_HEADER_NAME: str = "X-API-Key"

    @field_validator("SECRET_KEY", mode="before")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError(
                "SECRET_KEY 未配置，请执行 `openssl rand -hex 32` 生成"
            )
        if len(str(v)) < 32:
            raise ValueError(
                f"SECRET_KEY 长度为 {len(str(v))} 字符，至少需要 32 字符以保证安全性"
            )
        return v

    def get_secret_key(self) -> str:
        """获取 SECRET_KEY 明文，仅在 JWT 签名/验证时调用。"""
        return self.SECRET_KEY.get_secret_value()


# ─────────────────────────────────────────────────────────────────────────────
# CeleryConfig — 异步任务队列配置
# ─────────────────────────────────────────────────────────────────────────────

class CeleryConfig(_BaseEnvSettings):
    """Celery 任务队列配置。"""

    CELERY_WORKER_CONCURRENCY: int = 4      # Worker 并发进程数
    CELERY_TASK_MAX_RETRIES: int = 3        # 失败最大重试次数
    CELERY_TASK_RETRY_BACKOFF: int = 60     # 指数退避基础间隔（秒）
                                            # 实际间隔 = backoff × 2^(n-1)
                                            # 第1次重试: 60s, 第2次: 120s, 第3次: 240s

    # 三优先级队列名称常量，在 task_queue.py 中引用
    QUEUE_HIGH: str = "high_priority"       # Agent 实时任务
    QUEUE_DEFAULT: str = "default"          # 文档处理
    QUEUE_LOW: str = "low_priority"         # 评估、统计等离线任务


# ─────────────────────────────────────────────────────────────────────────────
# Settings — 聚合主配置类
# ─────────────────────────────────────────────────────────────────────────────

class Settings(_BaseEnvSettings):
    """Synaris 全局配置聚合类。

    通过 get_settings() 获取单例，各模块按需访问对应分组：

        from app.config.settings import get_settings
        s = get_settings()

        s.app.APP_NAME                    # "Synaris"
        s.app.cors_origins_list           # ["http://localhost:3000", ...]
        s.openai.get_api_key()            # "sk-..." 明文（仅在调用时获取）
        s.openai.DEFAULT_MODEL            # "gpt-4o-mini"
        s.redis.CACHE_TTL_SHORT           # 600
        s.redis.KEY_PREFIX_EMBEDDING      # "emb"
        s.security.get_secret_key()       # JWT 密钥明文
        s.log.LOG_LEVEL                   # "INFO"
    """

    app: AppConfig = Field(default_factory=AppConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)

    # ── RAG 文档处理参数（跨模块共享，放顶层便于访问）──────────────────────
    CHUNK_SIZE: int = 512       # 文档分块大小（字符数）
    CHUNK_OVERLAP: int = 50     # 相邻分块重叠字符数，防止语义在块边界处截断
    RAG_RERANK_TOP_K: int = 3   # Reranking 后实际注入 Prompt 的文档数量

    # ── 成本监控 ────────────────────────────────────────────────────────────
    COST_ALERT_DAILY_USD: float = 50.0   # 日费用告警阈值（美元），超出写 WARNING 日志


# ─────────────────────────────────────────────────────────────────────────────
# 单例工厂
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局配置单例。

    @lru_cache 确保 .env 文件在整个进程生命周期内只解析一次，
    无论在多少个模块中调用 get_settings()，始终返回同一个对象。

    单元测试中如需重置（修改环境变量后重新加载）：
        from app.config.settings import get_settings
        get_settings.cache_clear()
    """
    return Settings()
