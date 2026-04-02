#!/usr/bin/env bash
# =============================================================================
# scripts/init_db.sh — Synaris 数据库初始化脚本
#
# 职责（按执行顺序）：
#   1. 等待 PostgreSQL 就绪
#   2. 运行 Alembic 迁移（创建所有 ORM 表）
#   3. 创建默认管理员账号
#   4. 等待 Milvus 就绪
#   5. 创建 Synaris 知识库 Collection（若不存在）
#   6. 创建 HNSW 向量索引（M=16, efConstruction=256）
#   7. 创建长期记忆 Collection（Long-term Memory）
#   8. 等待 MinIO 就绪，初始化存储桶
#
# 使用方式：
#   # 在已启动的服务中执行（推荐）
#   docker-compose run --rm app bash scripts/init_db.sh
#
#   # 或在 docker-compose.yml 中作为 init container 调用
#   # （参见 docker-compose.yml 中 app 服务的注释）
#
# 环境变量（从 .env 或 docker-compose 注入）：
#   POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD
#   MILVUS_HOST / MILVUS_PORT
#   MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY / MINIO_BUCKET
#   ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD（初始管理员，可选）
# =============================================================================

set -euo pipefail    # 遇到错误立即退出，未定义变量视为错误

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'    # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 环境变量默认值 ─────────────────────────────────────────────────────────────
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-synaris}"
POSTGRES_USER="${POSTGRES_USER:-synaris}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-synaris_dev}"

MILVUS_HOST="${MILVUS_HOST:-milvus}"
MILVUS_PORT="${MILVUS_PORT:-19530}"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-minio:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
MINIO_BUCKET="${MINIO_BUCKET:-synaris-docs}"

ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@synaris.local}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-Admin@2026!}"

# ── 通用等待函数 ───────────────────────────────────────────────────────────────
wait_for_service() {
    local service_name="$1"
    local check_cmd="$2"
    local max_retries="${3:-60}"      # 最多等待 60 次（× 2s = 2分钟）
    local retry_interval="${4:-2}"
    local attempt=0

    log_info "等待 ${service_name} 就绪..."

    until eval "$check_cmd" > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_retries ]; then
            log_error "${service_name} 在 $((max_retries * retry_interval)) 秒内未就绪，退出"
            exit 1
        fi
        echo -n "."
        sleep "$retry_interval"
    done
    echo ""
    log_success "${service_name} 已就绪（等待了 $((attempt * retry_interval)) 秒）"
}


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 1：PostgreSQL 初始化
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════"
echo "  阶段 1：PostgreSQL 迁移"
echo "══════════════════════════════════════════"

# 等待 PostgreSQL 响应
wait_for_service "PostgreSQL" \
    "pg_isready -h ${POSTGRES_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB}" \
    60 2

# 运行 Alembic 迁移（创建/升级所有表结构）
log_info "运行 Alembic 数据库迁移..."
cd /app    # 确保在项目根目录（alembic.ini 所在位置）

if python -m alembic upgrade head 2>&1; then
    log_success "Alembic 迁移完成"
else
    log_error "Alembic 迁移失败，请检查迁移脚本"
    exit 1
fi

# 创建默认管理员账号（幂等：若已存在则跳过）
log_info "初始化默认管理员账号（${ADMIN_USERNAME} / ${ADMIN_EMAIL}）..."

python - <<PYTHON_EOF
import asyncio
import os
import sys

async def create_admin():
    try:
        from app.infrastructure.postgres_client import get_db_session
        from app.models.user import User
        from sqlalchemy import select
        import bcrypt

        async with get_db_session() as session:
            # 检查管理员是否已存在
            result = await session.execute(
                select(User).where(User.email == "${ADMIN_EMAIL}")
            )
            existing = result.scalar_one_or_none()

            if existing:
                print(f"[SKIP] 管理员 ${ADMIN_EMAIL} 已存在，跳过创建")
                return

            # 创建管理员
            hashed_pw = bcrypt.hashpw(
                "${ADMIN_PASSWORD}".encode("utf-8"),
                bcrypt.gensalt()
            ).decode("utf-8")

            admin = User(
                username="${ADMIN_USERNAME}",
                email="${ADMIN_EMAIL}",
                hashed_password=hashed_pw,
                role="admin",
                is_active=True,
            )
            session.add(admin)
            await session.commit()
            print(f"[OK] 管理员账号已创建：${ADMIN_USERNAME} / ${ADMIN_EMAIL}")

    except ImportError as e:
        print(f"[WARN] 跳过管理员创建（模块未就绪）：{e}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] 管理员创建失败（非致命）：{e}", file=sys.stderr)

asyncio.run(create_admin())
PYTHON_EOF

log_success "PostgreSQL 初始化完成"


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 2：Milvus Collection 初始化
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════"
echo "  阶段 2：Milvus Collection & 索引初始化"
echo "══════════════════════════════════════════"

# Milvus 健康检查（HTTP /healthz 端点）
wait_for_service "Milvus" \
    "curl -sf http://${MILVUS_HOST}:9091/healthz" \
    90 3    # Milvus 冷启动较慢，最多等待 90 × 3s = 4.5分钟

log_info "连接 Milvus 并初始化 Collection..."

python - <<PYTHON_EOF
import sys
import time

try:
    from pymilvus import (
        connections, Collection, CollectionSchema,
        FieldSchema, DataType, utility
    )
except ImportError:
    print("[WARN] pymilvus 未安装，跳过 Milvus 初始化", file=sys.stderr)
    sys.exit(0)

# ── 连接 Milvus ──────────────────────────────────────────────────────────
MILVUS_HOST = "${MILVUS_HOST}"
MILVUS_PORT = int("${MILVUS_PORT}")

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


# ── Collection 配置 ───────────────────────────────────────────────────────
# HNSW 索引参数（对应 PRD 的技术要求）：
#   M=16:              每个节点最多 16 条邻边（召回率 vs 内存的平衡点）
#   efConstruction=256: 索引构建时的候选集大小（越大索引质量越好，但构建更慢）
HNSW_INDEX_PARAMS = {
    "metric_type": "COSINE",          # 余弦相似度（text-embedding-3-small 推荐）
    "index_type":  "HNSW",
    "params": {
        "M":               16,
        "efConstruction":  256,
    }
}

EMBEDDING_DIM = 1536    # text-embedding-3-small 的向量维度

# 需要初始化的 Collection 列表
COLLECTIONS = [
    {
        "name":        "synaris_knowledge",    # 主知识库（RAG 检索）
        "description": "Synaris 企业知识库 Collection，存储文档 Embedding",
    },
    {
        "name":        "synaris_memory",       # 长期记忆（Multi-session Memory）
        "description": "Synaris 长期记忆 Collection，存储跨会话向量记忆",
    },
]


def create_collection(name: str, description: str) -> None:
    """创建 Collection 并建立 HNSW 索引（幂等：已存在则跳过）。"""

    if utility.has_collection(name):
        print(f"[SKIP] Collection '{name}' 已存在，跳过创建")
        # 验证索引是否已建立
        col = Collection(name)
        col.load()
        indexes = col.indexes
        if indexes:
            print(f"[SKIP] Collection '{name}' 已有索引，跳过")
        else:
            print(f"[INFO] Collection '{name}' 无索引，开始创建...")
            col.create_index(field_name="embedding", index_params=HNSW_INDEX_PARAMS)
            col.load()
            print(f"[OK]   Collection '{name}' 索引已创建")
        return

    # ── 定义 Schema ──────────────────────────────────────────────────────
    # 字段设计对应 milvus_client.py 中的 Schema 定义
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
            auto_id=False,       # 由应用层生成 UUID，保证可溯源
            description="文档片段唯一 ID（UUID）",
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,    # 最大 64KB 文本（chunk_size=512 token 足够）
            description="文档片段原始文本",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description=f"OpenAI text-embedding-3-small 向量（{EMBEDDING_DIM}维）",
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="来源文件名或标识符",
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="扩展元数据（chunk_index / doc_id / created_at 等）",
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description=description,
        enable_dynamic_field=True,    # 允许插入 schema 之外的动态字段
    )

    # ── 创建 Collection ───────────────────────────────────────────────────
    print(f"[INFO] 创建 Collection '{name}'...")
    col = Collection(
        name=name,
        schema=schema,
        consistency_level="Strong",   # 强一致性（写入后立即可读）
    )
    print(f"[OK]   Collection '{name}' 已创建")

    # ── 创建 HNSW 向量索引 ────────────────────────────────────────────────
    print(f"[INFO] 为 '{name}' 创建 HNSW 索引（M=16, efConstruction=256）...")
    col.create_index(
        field_name="embedding",
        index_params=HNSW_INDEX_PARAMS,
        index_name=f"{name}_hnsw_idx",
    )
    print(f"[OK]   HNSW 索引创建完成")

    # ── 为 source 字段创建标量索引（加速按来源过滤的检索）────────────────
    col.create_index(
        field_name="source",
        index_name=f"{name}_source_idx",
    )
    print(f"[OK]   source 字段标量索引创建完成")

    # ── 加载 Collection 到内存（使 Collection 可被检索）───────────────────
    col.load()
    print(f"[OK]   Collection '{name}' 已加载到内存，可接受检索请求")


# 逐一初始化所有 Collection
for col_config in COLLECTIONS:
    try:
        create_collection(**col_config)
    except Exception as e:
        print(f"[ERROR] Collection '{col_config['name']}' 初始化失败：{e}", file=sys.stderr)
        sys.exit(1)

# ── 打印 Collection 统计 ─────────────────────────────────────────────────
print("\n── Milvus Collection 状态 ──")
for col_config in COLLECTIONS:
    col = Collection(col_config["name"])
    stats = col.num_entities
    print(f"  {col_config['name']}: {stats} 条记录，索引数: {len(col.indexes)}")

connections.disconnect("default")
print("\n[OK] Milvus 初始化完成")
PYTHON_EOF

if [ $? -eq 0 ]; then
    log_success "Milvus Collection 初始化完成"
else
    log_error "Milvus 初始化失败"
    exit 1
fi


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 3：MinIO 存储桶初始化
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════"
echo "  阶段 3：MinIO 存储桶初始化"
echo "══════════════════════════════════════════"

# 等待 MinIO 就绪
wait_for_service "MinIO" \
    "curl -sf http://${MINIO_ENDPOINT}/minio/health/live" \
    30 2

log_info "初始化 MinIO 存储桶..."

python - <<PYTHON_EOF
import sys

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("[WARN] minio 包未安装，跳过 MinIO 初始化", file=sys.stderr)
    sys.exit(0)

client = Minio(
    "${MINIO_ENDPOINT}",
    access_key="${MINIO_ACCESS_KEY}",
    secret_key="${MINIO_SECRET_KEY}",
    secure=False,    # 开发环境不用 SSL；生产环境改为 True
)

BUCKETS = [
    ("${MINIO_BUCKET}", "ap-east-1"),     # 文档原始文件
    ("milvus-bucket",   "ap-east-1"),     # Milvus 对象存储（自动使用）
]

for bucket_name, region in BUCKETS:
    try:
        if client.bucket_exists(bucket_name):
            print(f"[SKIP] 存储桶 '{bucket_name}' 已存在")
        else:
            client.make_bucket(bucket_name, location=region)
            print(f"[OK]   存储桶 '{bucket_name}' 已创建")
    except S3Error as e:
        print(f"[ERROR] 存储桶 '{bucket_name}' 初始化失败：{e}", file=sys.stderr)
        sys.exit(1)

print("[OK] MinIO 存储桶初始化完成")
PYTHON_EOF

log_success "MinIO 初始化完成"


# ═══════════════════════════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Synaris 数据库初始化全部完成！              ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  ✓ PostgreSQL 表结构迁移                     ║${NC}"
echo -e "${GREEN}║  ✓ 默认管理员账号                            ║${NC}"
echo -e "${GREEN}║  ✓ Milvus 知识库 Collection + HNSW 索引      ║${NC}"
echo -e "${GREEN}║  ✓ Milvus 长期记忆 Collection + HNSW 索引    ║${NC}"
echo -e "${GREEN}║  ✓ MinIO 存储桶                              ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  访问地址：                                   ║${NC}"
echo -e "${GREEN}║    API：   http://localhost:8000/docs         ║${NC}"
echo -e "${GREEN}║    Flower：http://localhost:5555              ║${NC}"
echo -e "${GREEN}║    MinIO： http://localhost:9001              ║${NC}"
echo -e "${GREEN}║    Grafana：http://localhost:3000             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
