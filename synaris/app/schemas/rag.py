"""
@File       : rag.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: RAG 与知识库相关的 Pydantic 请求/响应模型。
@Project    : Synaris
@License    : Apache License 2.0
@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ─── 引用溯源 ───────────────────────────────────────────────


class SourceDoc(BaseModel):
    """RAG 回答引用的文档片段。"""

    source: str = Field(..., description="原始文件路径")
    chunk_index: int = Field(..., description="分块序号")
    page_num: int = Field(0, description="页码（PDF 有效）")
    snippet: str = Field(..., description="内容前 50 字")
    score: float = Field(..., description="语义相似度分数")


# ─── RAG 查询 ───────────────────────────────────────────────


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    collection_name: str = Field("default", description="知识库 Collection 名称")
    top_k: int = Field(5, ge=1, le=20, description="召回文档数")
    score_threshold: float = Field(0.5, ge=0.0, le=1.0, description="相关性阈值")
    session_id: Optional[str] = Field(None, description="会话 ID（用于上下文）")


class RAGResponse(BaseModel):
    answer: str = Field(..., description="LLM 生成的答案")
    sources: List[SourceDoc] = Field(default_factory=list, description="引用文档列表")
    tokens_used: int = Field(0, description="消耗 Token 数")
    collection_name: str = Field("default", description="查询的知识库")
    reranked_count: int = Field(0, description="Reranking 后使用的文档数")


# ─── 知识库管理 ─────────────────────────────────────────────


class KnowledgeUploadResponse(BaseModel):
    source: str
    collection_name: str
    total_chunks: int
    stored: int
    failed: int
    duration_ms: float


class KnowledgeItem(BaseModel):
    source: str = Field(..., description="文件路径（唯一标识）")
    file_type: str
    chunk_count: int
    collection_name: str


class KnowledgeListResponse(BaseModel):
    collection_name: str
    total_documents: int
    items: List[KnowledgeItem]


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    collection_name: str = Field("default")
    top_k: int = Field(5, ge=1, le=20)
