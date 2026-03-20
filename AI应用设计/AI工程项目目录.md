enterprise-ai-assistant/
└── app/
    ├── core/
    │   └── auth.py                      # 🆕 JWT + API Key 双认证体系
    │
    ├── infrastructure/
    │   ├── postgres_client.py           # 🆕 异步 PostgreSQL 连接池（asyncpg + SQLAlchemy）
    │   └── task_queue.py                # 🆕 Celery 配置 + Redis Broker
    │
    ├── models/                          # 🆕 SQLAlchemy ORM 数据模型目录
    │   ├── __init__.py
    │   ├── base.py                      # 🆕 DeclarativeBase + 公共字段
    │   ├── user.py                      # 🆕 用户 + API Key 表
    │   ├── session.py                   # 🆕 会话表（聊天/Agent任务）
    │   ├── task.py                      # 🆕 异步任务状态表
    │   └── prompt_version.py            # 🆕 Prompt 版本表（支持A/B测试）
    │
    ├── workers/                         # 🆕 Celery Worker 目录
    │   ├── __init__.py
    │   ├── document_worker.py           # 🆕 文档处理后台任务
    │   └── agent_worker.py              # 🆕 Agent 任务后台执行
    │
    ├── services/
    │   ├── memory_service.py            # 🆕 4种记忆类型统一管理
    │   ├── prompt_version_service.py    # 🆕 Prompt 版本管理 + A/B测试
    │   ├── eval_service.py              # 🆕 LLM 评估 Pipeline
    │   └── cost_service.py              # 🆕 Token 成本核算 + 用量统计
    │
    ├── agents/
    │   ├── tool_registry.py             # 🆕 工具动态注册中心
    │   └── tools/
    │       ├── db_query.py              # 🆕 结构化数据库查询工具
    │       └── external_api.py          # 🆕 外部 API 调用框架
    │
    └── core/
        └── observability.py             # 🆕 Prometheus 指标采集 + 延迟监控