# =============================================================================
# Dockerfile — Synaris 多阶段构建
#
# 构建阶段划分：
#   builder  → 安装所有编译依赖，构建 Python wheel
#   runtime  → 只复制产物与源码，镜像体积最小化
#
# 覆盖的技术栈：
#   FastAPI + Uvicorn + LangChain + LangGraph + Milvus
#   OpenAI + Redis(asyncio) + PostgreSQL(asyncpg) + SQLAlchemy
#   Celery + Prometheus + RestrictedPython + bcrypt + PyJWT
#
# 安全设计：
#   - 非 root 用户（uid=1000, gid=1000）
#   - pip 安装时 --no-cache-dir 减小层大小
#   - 不在镜像中存储任何密钥，全部通过环境变量注入
#
# 使用方式：
#   docker build -t synaris:latest .
#   docker build --target runtime -t synaris:latest .
# =============================================================================

# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

LABEL maintainer="Synaris Team"
LABEL description="Synaris builder stage — installs Python dependencies"

# 构建时系统依赖（编译 C 扩展：asyncpg、bcrypt、numpy 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 先复制 requirements，利用 Docker 层缓存
COPY requirements.txt .

# 将所有依赖安装到 /install，方便 runtime 精确复制
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        --target /install \
        -r requirements.txt


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Synaris Team"
LABEL version="3.0"
LABEL description="Synaris Enterprise RAG & Multi-Agent AI Platform"

# 运行时系统依赖（最小集）
# curl      → HEALTHCHECK
# libpq5    → asyncpg 运行时动态链接 PostgreSQL
# libgomp1  → numpy / numexpr 并行支持
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── 创建非 root 用户 ─────────────────────────────────────────────────────────
RUN groupadd --gid 1000 synaris \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /bin/bash synaris

WORKDIR /app

# ── 从 builder 复制已安装的 Python 包 ────────────────────────────────────────
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# ── 复制应用源码（按变更频率由低到高排列，最大化缓存命中）──────────────────
COPY alembic.ini        ./
COPY migrations/        ./migrations/
COPY app/               ./app/

# ── 目录权限 ──────────────────────────────────────────────────────────────────
RUN mkdir -p /app/logs /app/tmp \
    && chown -R synaris:synaris /app

# ── 切换到非 root 用户 ────────────────────────────────────────────────────────
USER synaris

# ── 环境变量默认值（可被 docker-compose / K8s 覆盖）─────────────────────────
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1

EXPOSE 8000

# ── 健康检查 ──────────────────────────────────────────────────────────────────
# start_period=20s：给 lifespan 事件（Redis/Milvus 预热）足够时间
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
    CMD curl -f http://localhost:8000/health || exit 1

# ── 启动命令 ──────────────────────────────────────────────────────────────────
# 生产环境通过 docker-compose.prod.yml 覆盖 APP_WORKERS 数量
# --loop uvloop：asyncio 性能提升约 2-4x
# --no-access-log：由 RequestLoggingMiddleware 统一记录，禁用 uvicorn 原生日志
CMD ["sh", "-c", \
     "uvicorn app.main:app \
      --host ${APP_HOST} \
      --port ${APP_PORT} \
      --workers ${APP_WORKERS} \
      --loop uvloop \
      --http h11 \
      --log-level info \
      --no-access-log"]