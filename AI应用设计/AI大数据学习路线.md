# AI 工程 8 周系统学习计划

## 总体学习原则

- **每日学习时长**：3 小时  
- **时间分配比例**：  
  - 理论学习 ≈ 30%  
  - 代码实践 / 项目驱动 ≈ 70%

## 第 1 周：Python 与 API 工程基础

**周目标**  
能够独立开发并部署一个简单的 RESTful API 服务，理解 Web 请求的全流程。

### 每日计划

**Day 1**  

- 理论：Python 基础语法、函数、类、模块与包  
- 实践：编写脚本（文件读写 + JSON 序列化/反序列化）

**Day 2**  
- 理论：HTTP 协议基础、REST API 设计规范、JSON 数据格式  
- 实践：使用 `requests` 库完成多种 HTTP 方法的 API 调用（GET / POST / PUT 等）

**Day 3**  
- 安装与环境：FastAPI + Uvicorn  
- 实践：创建第一个 FastAPI 服务，实现 `/hello` 接口

**Day 4**  
- 理论：POST 请求、路径参数、查询参数、请求体、Pydantic 数据验证  
- 实践：开发一个简单的聊天消息接收 API（接收 JSON 格式消息）

**Day 5**  
- 理论：Python 异步编程基础（async / await / asyncio）  
- 实践：将已有接口改造成异步接口

**Day 6**  
- 理论：日志系统（logging 模块）、异常处理与全局异常捕获  
- 实践：为 API 添加结构化日志 + 统一的异常响应格式

**Day 7**  
**小项目**：简易 AI 聊天 API  
技术栈：FastAPI + 任意一个开源/商用 LLM API  
核心功能：接收用户消息 → 调用 LLM → 返回回答

## 第 2 周：大语言模型（LLM）基础与 OpenAI API 实战

**周目标**  
熟练掌握主流大模型 API 的调用方式，理解 Prompt 与生成参数的核心逻辑。

### 每日计划

**Day 8**  
- 核心概念：Prompt 设计、Token 概念、temperature / top_p / max_tokens 等参数含义

**Day 9**  
- 实践：使用官方 SDK 完成最基本的 Completion 和 ChatCompletion 调用

**Day 10**  
- 实践：实现流式输出（streaming）

**Day 11**  
- 实践：实现简单的多轮对话记忆（message history）

**Day 12**  
- 理论：Prompt Engineering 核心技巧（角色扮演、思维链、Few-shot、结构化输出等）

**Day 13**  
- 实践：编写 3–5 种不同场景的 Prompt 模板（可复用）

**Day 14**  
**小项目**：命令行 / Web 版 AI 聊天机器人（带历史记忆）

## 第 3–4 周：RAG（Retrieval-Augmented Generation）系统构建

**周目标**  
能够从零搭建一个具备私有知识库的问答系统（RAG 完整闭环）。

### 核心技术路线**主要技术栈**  

- 向量数据库：Milvus（推荐） / Chroma / Weaviate / Qdrant  
- Embedding 模型：OpenAI / BGE / sentence-transformers / GTE 等  
- 文档处理：常见格式（pdf / docx / md / txt）

### 学习与实践重点（两周）

- 文档清洗与智能切分策略  
- 向量化（Embedding）原理与模型选择  
- 向量相似度检索（cosine / dot / euclidean）  
- 检索后重排序（可选：rerank）  
- 上下文窗口管理与 Prompt 拼接  
- 最终输出控制（json 模式 / 结构化输出）

**第 4 周结束项目**  
**AI 知识库智能问答系统**  
（支持上传文档 / 目录 → 自动构建向量库 → 自然语言提问）

## 第 5 周：LangChain 框架核心使用

**周目标**  
掌握 LangChain 构建复杂 LLM 应用的主要组件，完成标准 RAG Pipeline。

### 核心学习内容

- PromptTemplate / ChatPromptTemplate  
- LLM / ChatModel 调用封装  
- Chain（LLMChain / SequentialChain 等）  
- Document Loaders / Text Splitters  
- VectorStore / Retriever  
- RetrievalQA / ConversationalRetrievalChain

**周项目**  
使用 LangChain 重构第 3–4 周的 RAG 系统

## 第 6 周：Agent 与 LangGraph（有状态多步骤推理）

**周目标**  
理解并实现具备工具调用能力和多步规划的 AI Agent。

### 核心学习内容

- Tool / Tool Calling 机制  
- ReAct / Plan-and-Execute 等 Agent 范式  
- LangGraph 基本概念（StateGraph、Node、Edge）  
- 状态管理、条件分支、人机交互节点

**周项目**  
**AI 搜索 + 工具助手**  
（示例功能：搜索网页 / 调用计算器 / 查询天气 / 知识库问答 等多工具组合）

## 第 7 周：AI 系统性能优化与工程实践

**重点优化方向**

- 向量检索性能（HNSW / IVF / DiskANN 等索引类型）  
- Embedding 缓存 / 结果缓存（Redis / in-memory）  
- Token 消耗控制与成本优化  
- Prompt 压缩 / 上下文总结  
- 延迟优化（prefetch、并行检索、模型蒸馏等）  
- 监控与可观测性（日志、指标、tracing）

## 第 8 周：面试与项目复盘准备

**核心准备内容**

1. 项目亮点提炼（讲 1–2 个最有代表性的项目）  
2. 系统设计题常见方向（RAG / Agent / 搜索 / 推荐 / 多模态等）  
3. AI 工程常见问题  
   - Prompt 工程原理  
   - 向量数据库选型与原理  
   - 上下文长度限制解决方案  
   - Agent 循环死循环 / 幻觉问题  
   - 成本与延迟权衡  
4. 代码实现能力展示（现场 coding 或讲关键代码片段）

建议：  
- 准备 2–3 个不同难度的项目（RAG / Agent / 搜索类）  
- 录制 5–8 分钟的项目讲解视频（方便投递与面试复盘）

祝学习顺利！