<div align="center">

# SYNARIS

**Enterprise RAG & Multi-Agent AI Platform**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B35?style=flat-square)](https://langchain-ai.github.io/langgraph)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-00A1EA?style=flat-square&logo=milvus&logoColor=white)](https://milvus.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

[快速开始](#-快速开始) · [系统架构](#-系统架构) · [API 文档](#-api-接口速查) · [部署指南](#-生产部署) · [开发指南](#-开发指南)

</div>

---

## 📖 项目简介

**Synaris** 是一个基于 FastAPI + LangChain + LangGraph + Milvus 构建的**企业级 RAG 与多 Agent 智能协作平台**。

为企业内部提供三大核心能力：

| 能力 | 描述 | 关键技术 |
|------|------|---------|
| 🗣️ **AI 聊天** | 多轮对话、流式输出、多模型路由 | FastAPI SSE · Redis 会话缓存 |
| 📚 **RAG 知识库** | 文档上传、语义检索、引用溯源 | Milvus · LangChain LCEL · Reranking |
| 🤖 **Agent 工作流** | 多步推理、工具调用、多 Agent 协作 | LangGraph · Celery · Supervisor 模式 |

### 核心特性

- **9 层清晰架构**：Client → API Gateway → Agent Runtime → Intelligence → Tool System → Memory → LLM Routing → Data & Knowledge → Observability
- **企业级生产标准**：JWT + API Key 双认证、速率限制、全链路追踪、Prometheus 监控
- **4 种记忆系统**：短期对话记忆 / 长期向量记忆 / 用户画像 / 任务上下文
- **完整可观测性**：Token 成本核算、Prompt 版本管理、LLM 评估 Pipeline（幻觉检测）
- **一键 Docker 部署**：10 服务完整编排，含 Celery / PostgreSQL / MinIO / Grafana

---

## ⚡ 快速开始

### 环境要求

- Docker 24.0+ & Docker Compose 2.20+
- （本地开发）Python 3.11+、4GB+ 可用内存

### 三步启动

```bash
# 1. 克隆项目
git clone https://github.com/your-org/synaris.git
cd synaris

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填写：OPENAI_API_KEY

# 3. 一键启动
./scripts/start.sh
```

启动完成后访问：

| 服务 | 地址 |
|------|------|
| API 文档（Swagger） | http://localhost:8000/docs |
| API 文档（ReDoc） | http://localhost:8000/redoc |
| Grafana 监控 | http://localhost:3000 |
| Celery Flower | http://localhost:5555 |
| MinIO 控制台 | http://localhost:9001 |

### 验证安装

```bash
# 健康检查
curl http://localhost:8000/health

# 期望响应
# {"success": true, "data": {"status": "ok", "timestamp": 1234567890}}
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Layer                          │
│         Web App · Mobile · Slack Bot · API Clients          │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST / WebSocket
┌─────────────────────────▼───────────────────────────────────┐
│                    API Gateway Layer                         │
│   FastAPI · JWT + API Key Auth · Rate Limit · TraceID       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Agent Runtime Layer                         │
│         Celery Queue · LangGraph Orchestrator                │
│              Redis Pub/Sub Message Broker                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│               Agent Intelligence Layer                       │
│    Planner · Tool Executor · RAG Engine                      │
│    Supervisor → ResearchAgent → WriterAgent → ReviewerAgent  │
└──────────┬──────────────┬────────────────────────┬──────────┘
           │              │                        │
    ┌──────▼──────┐ ┌─────▼──────┐        ┌───────▼───────┐
    │ Tool System │ │   Memory   │        │  LLM Routing  │
    │  Registry   │ │  4 Types   │        │  Model Router │
    └─────────────┘ └────────────┘        └───────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│               Data & Knowledge Layer                         │
│     Milvus · Redis Cache · MinIO · PostgreSQL               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│             Observability & Evaluation Layer                 │
│    Prometheus · Grafana · Prompt Versioning · Cost Monitor  │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 分类 | 技术 | 版本 |
|------|------|------|
| API 框架 | FastAPI | 0.115+ |
| LLM 编排 | LangChain | 0.3+ |
| Agent 工作流 | LangGraph | 0.2+ |
| 向量数据库 | Milvus | 2.4+ |
| 结构化数据库 | PostgreSQL + SQLAlchemy 2.0 | 16+ |
| 任务队列 | Celery + Redis | 5.3+ |
| 缓存 | Redis | 7+ |
| 大模型接口 | OpenAI API | gpt-4o / gpt-4o-mini |
| 对象存储 | MinIO | RELEASE.2024+ |
| 监控 | Prometheus + Grafana | 2.x / 10.x |
| 部署 | Docker Compose | 2.20+ |

---

## 🗂️ 项目结构

```
synaris/
├── app/
│   ├── config/
│   │   └── settings.py              # 全局配置（pydantic-settings）
│   ├── core/
│   │   ├── auth.py                  # JWT + API Key 双认证
│   │   ├── exceptions.py            # 自定义异常体系
│   │   ├── llm_router.py            # 多模型路由 + 降级
│   │   ├── logging.py               # 结构化 JSON 日志 + TraceID
│   │   ├── observability.py         # Prometheus 指标采集
│   │   └── prompts.py               # 全部 PromptTemplate 集中管理
│   ├── infrastructure/
│   │   ├── redis_client.py          # 异步 Redis 连接池 + Pub/Sub
│   │   ├── milvus_client.py         # Milvus 连接 + HNSW 索引
│   │   ├── llm_client.py            # LangChain 封装 + Streaming
│   │   ├── embedding_client.py      # Embedding 生成 + Redis 缓存
│   │   ├── postgres_client.py       # asyncpg + SQLAlchemy 2.0
│   │   └── task_queue.py            # Celery 三优先级队列配置
│   ├── models/                      # SQLAlchemy ORM 模型
│   │   ├── user.py                  # 用户 + API Key 表
│   │   ├── task.py                  # 异步任务状态表
│   │   └── prompt_version.py        # Prompt 版本管理表
│   ├── schemas/                     # Pydantic 请求/响应模型
│   ├── api/                         # FastAPI 路由层
│   │   ├── health.py                # /health · /health/detailed
│   │   ├── chat.py                  # /chat · /chat/stream
│   │   ├── knowledge.py             # /knowledge/*
│   │   ├── rag.py                   # /rag/query · /rag/query/stream
│   │   └── agent.py                 # /agent/* · WS /agent/stream/{id}
│   ├── services/                    # 业务逻辑层
│   │   ├── chat_service.py
│   │   ├── document_service.py
│   │   ├── rag_service.py
│   │   ├── vector_store.py
│   │   ├── memory_service.py        # 4种记忆类型统一管理
│   │   ├── cost_service.py          # Token 成本核算
│   │   ├── prompt_version_service.py
│   │   └── eval_service.py          # LLM 评估 Pipeline
│   ├── agents/                      # LangGraph Agent 层
│   │   ├── state.py                 # AgentState + TaskStatus
│   │   ├── workflow.py              # 单 Agent 状态机
│   │   ├── supervisor.py            # Supervisor 多 Agent 编排
│   │   ├── workers.py               # Research / Writer / Reviewer
│   │   ├── tool_registry.py         # 工具动态注册中心
│   │   └── tools/                   # 6 个工具实现
│   └── workers/                     # Celery 后台任务
│       ├── document_worker.py
│       └── agent_worker.py
├── migrations/                      # Alembic 数据库迁移
├── tests/                           # pytest 集成测试
├── monitoring/                      # Prometheus + Grafana 配置
├── nginx/                           # 反向代理配置
├── scripts/                         # 运维脚本
├── docs/                            # 项目文档
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
└── .env.example
```

---

## 🌐 API 接口速查

### 认证方式

所有接口（健康检查除外）需要在 Header 中携带以下之一：

```bash
# JWT Token
Authorization: Bearer <your_jwt_token>

# API Key
X-API-Key: <your_api_key>
```

### 接口列表

#### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 存活检查 |
| `GET` | `/health/detailed` | 各组件（Redis/Milvus/Postgres）连通性检查 |
| `GET` | `/metrics` | Prometheus 指标采集端点 |

#### AI 聊天

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat` | 多轮对话（JSON 响应） |
| `POST` | `/api/v1/chat/stream` | 多轮对话（SSE 流式输出） |

```bash
# 示例：普通对话
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_001", "message": "你好，介绍一下 Synaris", "strategy": "BALANCED"}'

# 示例：流式对话
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_001", "message": "详细说明 RAG 的工作原理"}' \
  --no-buffer
```

#### RAG 知识库

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/knowledge/upload` | 上传文档（PDF/Word/TXT/Markdown） |
| `GET` | `/api/v1/knowledge/list` | 知识库文档列表 |
| `DELETE` | `/api/v1/knowledge/{source_id}` | 删除指定文档 |
| `POST` | `/api/v1/knowledge/search` | 测试语义检索 |
| `POST` | `/api/v1/rag/query` | RAG 问答（含引用溯源） |
| `POST` | `/api/v1/rag/query/stream` | RAG 问答（流式输出） |

```bash
# 上传文档
curl -X POST http://localhost:8000/api/v1/knowledge/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@./docs/manual.pdf" \
  -F "collection_name=default"

# RAG 问答
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "系统的安全策略是什么？", "collection": "default"}'
```

#### Agent 任务

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/agent/run` | 提交 Agent 任务（Celery 异步） |
| `GET` | `/api/v1/agent/status/{task_id}` | 查询任务状态与结果 |
| `POST` | `/api/v1/agent/cancel/{task_id}` | 取消运行中的任务 |
| `WS` | `/api/v1/agent/stream/{task_id}` | WebSocket 实时步骤推送 |

```bash
# 提交多 Agent 任务
curl -X POST http://localhost:8000/api/v1/agent/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "调研 2025 年大模型行业趋势并撰写分析报告",
    "mode": "multi",
    "session_id": "sess_001"
  }'

# WebSocket 实时监听
wscat -c "ws://localhost:8000/api/v1/agent/stream/task_abc123" \
  -H "Authorization: Bearer <token>"
```

---

## ⚙️ 环境变量说明

复制 `.env.example` 为 `.env` 并填写以下必填项：

### 必填

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | `sk-...` |
| `SECRET_KEY` | JWT 签名密钥（32位随机字符串） | `openssl rand -hex 32` |
| `POSTGRES_PASSWORD` | PostgreSQL 数据库密码 | `StrongPass123!` |

### 常用配置（已有默认值）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `APP_NAME` | `Synaris` | 应用名称 |
| `APP_VERSION` | `1.0.0` | 版本号 |
| `DEBUG` | `false` | 调试模式 |
| `DEFAULT_MODEL` | `gpt-4o-mini` | 默认 LLM 模型 |
| `FALLBACK_MODEL` | `gpt-3.5-turbo` | 降级模型 |
| `MILVUS_HOST` | `milvus` | Milvus 服务地址 |
| `MILVUS_PORT` | `19530` | Milvus 端口 |
| `REDIS_URL` | `redis://redis:6379/0` | Redis 连接地址 |
| `POSTGRES_URL` | `postgresql+asyncpg://...` | PostgreSQL 连接串 |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | Celery Broker |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LOG_FORMAT` | `json` | 日志格式（`json` / `text`） |
| `RATE_LIMIT` | `60/minute` | API 速率限制 |
| `EMBEDDING_CACHE_TTL` | `86400` | Embedding 缓存时长（秒） |
| `CHAT_HISTORY_TTL` | `7200` | 会话历史缓存时长（秒） |

---

## 🐳 生产部署

### 使用 Docker Compose（推荐）

```bash
# 启动全部服务（生产配置）
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 初始化数据库（首次部署必须执行）
./scripts/init_db.sh

# 验证所有服务健康状态
./scripts/health_check.sh
```

### 服务说明

| 服务 | 镜像 | 端口 | 说明 |
|------|------|------|------|
| `synaris-app` | `synaris:latest` | 8000 | FastAPI 主服务（生产×2副本） |
| `celery-worker` | `synaris:latest` | - | Celery Worker（4并发） |
| `celery-flower` | `mher/flower` | 5555 | Celery 监控仪表盘 |
| `redis` | `redis:7-alpine` | 6379 | 缓存 + 消息队列 |
| `milvus` | `milvusdb/milvus:v2.4.0` | 19530 | 向量数据库 |
| `postgres` | `postgres:16-alpine` | 5432 | 结构化数据库 |
| `minio` | `minio/minio` | 9000/9001 | 文档对象存储 |
| `nginx` | `nginx:alpine` | 80/443 | 反向代理 + 负载均衡 |
| `prometheus` | `prom/prometheus` | 9090 | 指标采集 |
| `grafana` | `grafana/grafana` | 3000 | 监控仪表盘 |

### 资源建议（生产环境）

| 服务 | CPU | 内存 |
|------|-----|------|
| synaris-app × 2 | 1 core × 2 | 2GB × 2 |
| celery-worker | 2 cores | 2GB |
| milvus | 2 cores | 4GB |
| postgres | 1 core | 1GB |
| redis | 0.5 core | 512MB |

### 数据库迁移

```bash
# 生成新的迁移文件
docker exec synaris-app alembic revision --autogenerate -m "add_new_table"

# 执行迁移
docker exec synaris-app alembic upgrade head

# 回滚上一个版本
docker exec synaris-app alembic downgrade -1
```

---

## 🛠️ 开发指南

### 本地环境搭建

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动依赖服务（仅基础设施，不含 app）
docker compose up redis milvus postgres minio -d

# 运行数据库迁移
alembic upgrade head

# 启动开发服务器（热重载）
uvicorn app.main:app --reload --port 8000
```

### 启动 Celery Worker（本地开发）

```bash
# 启动 Worker
celery -A app.infrastructure.task_queue worker \
  --loglevel=info \
  --concurrency=2 \
  -Q high_priority,default,low_priority

# 启动 Flower 监控
celery -A app.infrastructure.task_queue flower --port=5555
```

### 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定模块测试
pytest tests/test_chat_api.py -v
pytest tests/test_rag_pipeline.py -v
pytest tests/test_agent_workflow.py -v

# 生成覆盖率报告
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### 代码规范

```bash
# 格式化
black app/ tests/
isort app/ tests/

# 类型检查
mypy app/

# Lint
ruff check app/
```

---

## 📊 性能指标目标

| 指标 | 目标值 | 监控方式 |
|------|--------|---------|
| 接口响应时间 P95 | ≤ 1.5 秒 | Prometheus Histogram |
| RAG 问答准确率 | ≥ 93% | LLM Evaluation Pipeline |
| Agent 任务完成率 | ≥ 90% | PostgreSQL task 表 |
| 系统可用性 | ≥ 99.95% | 健康检查 + 告警 |
| Redis 缓存命中率 | ≥ 85% | `cache_hit_ratio` 指标 |
| 日查询量支持 | 10 万次+ | 压测验证 |

---

## 🔒 安全说明

- **API Key** 使用 bcrypt 哈希存储，明文仅在创建时显示一次
- **JWT** 支持密钥轮转，有效期 24h
- **DB Query 工具** 通过 AST 解析仅允许 SELECT 语句，防止 SQL 注入
- **外部 API 工具** 实施白名单域名校验，防止 SSRF 攻击
- **Code Executor** 基于 RestrictedPython 沙箱，禁止网络/文件系统访问，执行超时 10s
- 所有敏感配置通过环境变量注入，禁止硬编码

---

## 🗺️ 路线图

- [ ] **v1.1** — 支持 Claude / Gemini 多厂商 LLM 路由
- [ ] **v1.2** — 知识库多租户隔离
- [ ] **v1.3** — Agent 可视化调试界面
- [ ] **v2.0** — Kubernetes Helm Chart 一键部署
- [ ] **v2.1** — 支持本地部署 LLM（Ollama 集成）

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature-name`
3. 提交变更：`git commit -m 'feat: add some feature'`
4. 推送分支：`git push origin feature/your-feature-name`
5. 提交 Pull Request

Commit 信息请遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 规范。

---

## ❓ 常见问题

**Q：启动时 Milvus 连接失败？**

```bash
# 检查 Milvus 是否就绪（通常需要 30-60 秒初始化）
docker logs synaris-milvus | tail -20
# 手动等待并重新初始化
./scripts/init_db.sh
```

**Q：OpenAI API 调用超时？**

检查 `.env` 中的 `OPENAI_BASE_URL`，如使用代理需正确配置。可通过 `FALLBACK_MODEL` 设置备用模型。

**Q：Celery 任务一直处于 PENDING 状态？**

```bash
# 确认 Worker 正在运行
docker ps | grep celery
# 查看 Worker 日志
docker logs synaris-celery-worker
```

**Q：如何查看实时监控？**

访问 Grafana（http://localhost:3000，默认账号 `admin/admin`），导入 `monitoring/grafana_dashboard.json`。

**Q：如何重置所有数据？**

```bash
docker compose down -v   # 删除所有容器和数据卷
./scripts/start.sh       # 重新启动并初始化
```

---

## 📄 许可证

本项目采用 [Apache-2.0 license](LICENSE) 开源许可证。

---

<div align="center">

**Synaris · 赛纳睿智能平台**

*Built with ❤️ by the Synaris Team*

[⬆ 回到顶部](#synaris)

</div>
