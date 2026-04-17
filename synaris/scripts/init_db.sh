#!/usr/bin/env bash
# =============================================================================
# scripts/init_db.sh — Synaris 数据库完整初始化脚本
#
# 执行顺序：
#   阶段 1 — PostgreSQL
#     1.1  等待 PostgreSQL 就绪
#     1.2  运行 Alembic 迁移（创建/升级所有 ORM 表）
#     1.3  创建默认管理员账号（幂等：已存在则跳过）
#
#   阶段 2 — Milvus Collection
#     2.1  等待 Milvus 就绪（冷启动较慢，最多等 4.5 分钟）
#     2.2  创建知识库 Collection（synaris_knowledge）+ HNSW 索引
#     2.3  创建长期记忆 Collection（synaris_memory）+ HNSW 索引
#
#   阶段 3 — MinIO 存储桶
#     3.1  等待 MinIO 就绪
#     3.2  创建 synaris-docs 和 milvus-bucket 存储桶（幂等）
#
# 使用方式：
#   # 推荐：在已启动的容器中执行
#   docker-compose run --rm app bash scripts/init_db.sh
#
#   # 或直接在宿主机执行（需要配置好 .env）
#   source .env && bash scripts/init_db.sh
#
# 环境变量（从 .env 或 docker-compose 注入）：
#   POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD
#   MILVUS_HOST / MILVUS_PORT
#   MINIO_ENDPOINT / MINIO_ROOT_USER / MINIO_ROOT_PASSWORD / MINIO_BUCKET
#   ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD
# =============================================================================

set -euo pipefail   # 遇到错误立即退出，未定义变量视为错误

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()    { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; \
                echo -e "${BOLD}${BLUE}  $*${NC}"; \
                echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}"; }

# ── 环境变量默认值 ─────────────────────────────────────────────────────────────
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-synaris}"
POSTGRES_USER="${POSTGRES_USER:-synaris}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-synaris_dev}"

MILVUS_HOST="${MILVUS_HOST:-milvus}"
MILVUS_PORT="${MILVUS_PORT:-19530}"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-minio:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
MINIO_BUCKET="${MINIO_BUCKET:-synaris-docs}"

ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@synaris.local}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-Admin@2026!}"

# ── 通用等待函数 ───────────────────────────────────────────────────────────────
wait_for_service() {
    local service_name="$1"
    local check_cmd="$2"
    local max_retries="${3:-60}"
    local retry_interval="${4:-2}"
    local attempt=0

    log_info "等待 ${service_name} 就绪..."

    until eval "$check_cmd" > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_retries ]; then
            log_error "${service_name} 在 $((max_retries * retry_interval)) 秒内未就绪，退出"
            exit 1
        fi
        printf "."
        sleep "$retry_interval"
    done
    echo ""
    log_success "${service_name} 已就绪（等待 $((attempt * retry_interval)) 秒）"
}


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 1：PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════════
log_step "阶段 1：PostgreSQL 迁移与初始化"

# 1.1 等待 PostgreSQL 就绪
wait_for_service "PostgreSQL" \
    "pg_isready -h ${POSTGRES_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB}" \
    60 2

# 1.2 运行 Alembic 迁移
log_info "运行 Alembic 数据库迁移（upgrade head）..."
cd /app   # 确保在项目根目录（alembic.ini 所在位置）

if python -m alembic upgrade head 2>&1; then
    log_success "Alembic 迁移完成"
else
    log_error "Alembic 迁移失败，请检查迁移脚本"
    log_warn "提示：若是首次部署且尚无迁移文件，执行以下命令生成："
    log_warn "  alembic revision --autogenerate -m 'initial migration'"
    exit 1
fi

# 1.3 创建默认管理员账号（幂等）
log_info "初始化默认管理员账号（${ADMIN_USERNAME} / ${ADMIN_EMAIL}）..."

python - <<PYTHON_ADMIN
import asyncio
import sys
import os

async def create_admin():
    try:
        # 动态导入（确保 PYTHONPATH=/app 已设置）
        from app.infrastructure.postgres_client import init_db, close_db, db_session
        from sqlalchemy import select

        await init_db()

        try:
            from app.models.user import User, UserRole
        except ImportError as e:
            print(f"[WARN] models.user 未就绪，跳过管理员创建：{e}", file=sys.stderr)
            return

        async with db_session() as session:
            # 检查是否已存在
            result = await session.execute(
                select(User).where(User.email == "${ADMIN_EMAIL}")
            )
            existing = result.scalar_one_or_none()

            if existing:
                print(f"[SKIP] 管理员 ${ADMIN_EMAIL} 已存在，跳过创建")
                return

            # 使用 bcrypt 哈希密码
            try:
                import bcrypt
                hashed_pw = bcrypt.hashpw(
                    "${ADMIN_PASSWORD}".encode("utf-8"),
                    bcrypt.gensalt(12)
                ).decode("utf-8")
            except ImportError:
                # bcrypt 未安装时，使用 passlib 兜底
                from passlib.context import CryptContext
                ctx = CryptContext(schemes=["bcrypt"])
                hashed_pw = ctx.hash("${ADMIN_PASSWORD}")

            admin = User(
                username="${ADMIN_USERNAME}",
                email="${ADMIN_EMAIL}",
                hashed_password=hashed_pw,
                role=UserRole.ADMIN.value,
                is_active=True,
            )
            session.add(admin)
            # db_session 上下文管理器会自动 commit

        print(f"[OK] 管理员账号已创建：${ADMIN_USERNAME} / ${ADMIN_EMAIL}")

        await close_db()

    except ImportError as e:
        print(f"[WARN] 跳过管理员创建（依赖模块未就绪）：{e}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] 管理员创建失败（非致命）：{e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)

asyncio.run(create_admin())
PYTHON_ADMIN

log_success "PostgreSQL 初始化完成"


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 2：Milvus Collection 初始化
# ═══════════════════════════════════════════════════════════════════════════════
log_step "阶段 2：Milvus Collection & HNSW 索引初始化"

# Milvus 健康检查（HTTP /healthz 端点）
wait_for_service "Milvus" \
    "curl -sf http://${MILVUS_HOST}:9091/healthz" \
    90 3   # 最多等 270 秒（Milvus 冷启动慢）

log_info "连接 Milvus 并初始化 Collection..."

python - <<PYTHON_MILVUS
import sys

try:
    from pymilvus import (
        connections, Collection, CollectionSchema,
        FieldSchema, DataType, utility
    )
except ImportError:
    print("[WARN] pymilvus 未安装，跳过 Milvus 初始化", file=sys.stderr)
    sys.exit(0)

# ── 连接 ────────────────────────────────────────────────────────────────────
MILVUS_HOST = "${MILVUS_HOST}"
MILVUS_PORT = int("${MILVUS_PORT}")
EMBEDDING_DIM = 1536   # text-embedding-3-small 向量维度

try:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=30,
    )
    print(f"[OK] 已连接 Milvus {MILVUS_HOST}:{MILVUS_PORT}")
except Exception as e:
    print(f"[ERROR] Milvus 连接失败：{e}", file=sys.stderr)
    sys.exit(1)

# ── HNSW 索引参数 ────────────────────────────────────────────────────────────
# M=16：每节点最大邻居数，召回率与内存的均衡点
# efConstruction=256：构建质量，越大越好但越慢
HNSW_INDEX = {
    "metric_type": "COSINE",
    "index_type":  "HNSW",
    "params": {"M": 16, "efConstruction": 256}
}

# ── Collection 配置列表 ──────────────────────────────────────────────────────
COLLECTIONS = [
    {
        "name":        "synaris_knowledge",
        "description": "企业知识库 — RAG 检索使用",
    },
    {
        "name":        "synaris_memory",
        "description": "长期记忆 — 跨会话向量记忆",
    },
]

def build_schema(description: str) -> CollectionSchema:
    """构建标准 Schema（与 milvus_client.py 的 _build_collection_schema 保持一致）。"""
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
            auto_id=False,
            description="文档片段 UUID",
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="文档片段文本",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description=f"OpenAI text-embedding-3-small ({EMBEDDING_DIM}维)",
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="来源文件标识",
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="扩展元数据",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description=description,
        enable_dynamic_field=True,
    )

def init_collection(name: str, description: str) -> None:
    """创建 Collection + HNSW 索引（幂等）。"""
    if utility.has_collection(name):
        print(f"[SKIP] Collection '{name}' 已存在")
        col = Collection(name)
        col.load()
        # 若索引不存在则补建（可能是上次创建中断）
        if not col.indexes:
            print(f"[INFO] Collection '{name}' 索引缺失，正在创建...")
            col.create_index(field_name="embedding", index_params=HNSW_INDEX)
            col.load()
            print(f"[OK]   '{name}' 索引已补建")
        return

    print(f"[INFO] 创建 Collection '{name}'...")
    schema = build_schema(description)
    col = Collection(name=name, schema=schema, consistency_level="Strong")
    print(f"[OK]   Collection '{name}' 已创建")

    print(f"[INFO] 为 '{name}' 创建 HNSW 索引（M=16, efConstruction=256）...")
    col.create_index(
        field_name="embedding",
        index_params=HNSW_INDEX,
        index_name=f"{name}_hnsw_idx",
    )
    col.load()
    print(f"[OK]   '{name}' 索引创建完成并已加载到内存")

# 逐一初始化
for cfg in COLLECTIONS:
    try:
        init_collection(**cfg)
    except Exception as e:
        print(f"[ERROR] '{cfg['name']}' 初始化失败：{e}", file=sys.stderr)
        sys.exit(1)

# 打印汇总
print("\n── Collection 状态 ──")
for cfg in COLLECTIONS:
    try:
        col = Collection(cfg["name"])
        print(f"  {cfg['name']}: {col.num_entities} 条记录，{len(col.indexes)} 个索引")
    except Exception:
        pass

connections.disconnect("default")
print("\n[OK] Milvus 初始化完成")
PYTHON_MILVUS

log_success "Milvus Collection 初始化完成"


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 3：MinIO 存储桶初始化
# ═══════════════════════════════════════════════════════════════════════════════
log_step "阶段 3：MinIO 存储桶初始化"

# MinIO 健康检查
wait_for_service "MinIO" \
    "curl -sf http://${MINIO_ENDPOINT}/minio/health/live" \
    30 2

log_info "初始化 MinIO 存储桶..."

python - <<PYTHON_MINIO
import sys

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("[WARN] minio 包未安装，跳过 MinIO 初始化", file=sys.stderr)
    sys.exit(0)

client = Minio(
    "${MINIO_ENDPOINT}",
    access_key="${MINIO_ROOT_USER}",
    secret_key="${MINIO_ROOT_PASSWORD}",
    secure=False,
)

# 需要创建的存储桶
BUCKETS = [
    ("${MINIO_BUCKET}", "ap-east-1"),   # 文档原始文件
    ("milvus-bucket",   "ap-east-1"),   # Milvus 向量数据文件
]

for bucket_name, region in BUCKETS:
    try:
        if client.bucket_exists(bucket_name):
            print(f"[SKIP] 存储桶 '{bucket_name}' 已存在")
        else:
            client.make_bucket(bucket_name, location=region)
            print(f"[OK]   存储桶 '{bucket_name}' 已创建")
    except S3Error as e:
        print(f"[ERROR] '{bucket_name}' 创建失败：{e}", file=sys.stderr)
        sys.exit(1)

print("[OK] MinIO 存储桶初始化完成")
PYTHON_MINIO

log_success "MinIO 初始化完成"


# ═══════════════════════════════════════════════════════════════════════════════
# 完成汇总
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   Synaris 全部初始化完成！                        ║${NC}"
echo -e "${GREEN}${BOLD}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}${BOLD}║                                                  ║${NC}"
echo -e "${GREEN}${BOLD}║  ✓ PostgreSQL 表结构迁移（Alembic）               ║${NC}"
echo -e "${GREEN}${BOLD}║  ✓ 默认管理员账号（${ADMIN_EMAIL}）         ║${NC}"
echo -e "${GREEN}${BOLD}║  ✓ Milvus synaris_knowledge Collection            ║${NC}"
echo -e "${GREEN}${BOLD}║  ✓ Milvus synaris_memory Collection               ║${NC}"
echo -e "${GREEN}${BOLD}║  ✓ MinIO 存储桶（synaris-docs / milvus-bucket）   ║${NC}"
echo -e "${GREEN}${BOLD}║                                                  ║${NC}"
echo -e "${GREEN}${BOLD}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}${BOLD}║  访问地址：                                       ║${NC}"
echo -e "${GREEN}${BOLD}║    API Swagger：  http://localhost:8000/docs      ║${NC}"
echo -e "${GREEN}${BOLD}║    Celery Flower：http://localhost:5555           ║${NC}"
echo -e "${GREEN}${BOLD}║    MinIO Console：http://localhost:9001           ║${NC}"
echo -e "${GREEN}${BOLD}║    Grafana：      http://localhost:3000           ║${NC}"
echo -e "${GREEN}${BOLD}║    Prometheus：   http://localhost:9090           ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"