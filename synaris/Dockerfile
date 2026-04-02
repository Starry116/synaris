# =============================================================================
# Dockerfile — Synaris 多阶段构建
#
# 构建阶段划分（类比「流水线装配」）：
#   builder  → 安装编译依赖、构建 Python wheel，产物干净不含编译工具链
#   runtime  → 只复制 wheel 产物和源码，镜像体积最小化
#
# 安全设计：
#   - 非 root 用户（uid=1000, gid=1000），遵循最小权限原则
#   - 不在镜像中存储任何密钥，全部通过环境变量注入
#   - pip 安装时 --no-cache-dir，减小层大小
#
# 使用方式：
#   docker build -t synaris:latest .
#   docker build --target builder -t synaris:builder .   # 仅 builder 阶段
#
# 环境变量（运行时必须提供）：
#   OPENAI_API_KEY / REDIS_URL / MILVUS_HOST / POSTGRES_DSN 等
#   详见项目根目录 .env.example
# =============================================================================

# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

LABEL maintainer="Synaris Team"
LABEL description="Synaris builder stage — installs Python dependencies"

# 构建时依赖（编译 C 扩展，如 psycopg2-binary、numpy）
# 安装完成后不会出现在 runtime 镜像中
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 先复制依赖文件，利用 Docker 层缓存：
# 只要 requirements.txt 不变，此层不会重建（即使源码变了）
COPY requirements.txt .

# 将所有依赖安装到 /install 目录，方便 runtime 阶段精确复制
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --target /install \
        -r requirements.txt


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Synaris Team"
LABEL version="3.0"
LABEL description="Synaris Enterprise RAG & Multi-Agent AI Platform"

# 运行时系统依赖（最小集）
# curl：HEALTHCHECK 需要；libpq5：asyncpg 运行时动态链接
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# ── 创建非 root 用户 ─────────────────────────────────────────────────────────
# uid/gid 固定为 1000，与常见 CI 环境一致，避免卷挂载权限问题
RUN groupadd --gid 1000 synaris \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /bin/bash synaris

WORKDIR /app

# ── 从 builder 复制已安装的 Python 包 ────────────────────────────────────────
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# ── 复制应用源码 ──────────────────────────────────────────────────────────────
# 按变更频率由低到高排列，最大化利用 Docker 层缓存
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
    # uvicorn 运行参数
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1

EXPOSE 8000

# ── 健康检查 ──────────────────────────────────────────────────────────────────
# interval: 每 30s 检查一次
# timeout:  单次检查超时 10s
# retries:  连续失败 3 次才标记为 unhealthy
# start_period: 容器启动后等待 15s 再开始检查（给 lifespan 事件时间初始化）
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

# ── 启动命令 ──────────────────────────────────────────────────────────────────
# 生产环境通过 docker-compose.prod.yml 覆盖 workers 数量
CMD ["sh", "-c", \
     "uvicorn app.main:app \
      --host ${APP_HOST} \
      --port ${APP_PORT} \
      --workers ${APP_WORKERS} \
      --loop uvloop \
      --http h11 \
      --log-level info \
      --no-access-log"]
