"""
@File       : knowledge.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 知识库管理 API（上传 / 列表 / 删除 / 检索）。
@Features:
  - POST /knowledge/upload  → 文件上传，触发 DocumentService 处理流水线
  - GET  /knowledge/list    → 列出知识库中所有文档
  - DELETE /knowledge/{source_id} → 按 source 删除文档及其向量
  - POST /knowledge/search  → 直接向量检索（调试用，不过 LLM）
  - 支持 collection_name 多知识库隔离
@Project    : Synaris
@License    : Apache License 2.0
@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import os
import tempfile
import urllib.parse
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from core.exceptions import DocumentParseError
from core.logging import get_logger
from schemas.base import ApiResponse
from schemas.rag import (
    KnowledgeItem,
    KnowledgeListResponse,
    KnowledgeSearchRequest,
    KnowledgeUploadResponse,
)
from services.document_service import DocumentService
from services.vector_store import VectorStoreService

logger = get_logger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# ─── 允许上传的 MIME 类型 ────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ─── 依赖注入（生产中由 DI 容器提供） ──────────────────────────────────────────────


def get_document_service() -> DocumentService:
    from dependencies import document_service_instance  # noqa: WPS433

    return document_service_instance


def get_vector_store() -> VectorStoreService:
    from dependencies import vector_store_instance  # noqa: WPS433

    return vector_store_instance


# ─── 路由实现 ───────────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=ApiResponse[KnowledgeUploadResponse],
    summary="上传文档到知识库",
)
async def upload_document(
    file: UploadFile = File(..., description="支持 PDF / DOCX / TXT / Markdown"),
    collection_name: str = Form("default", description="目标知识库名称"),
    doc_service: DocumentService = Depends(get_document_service),
) -> ApiResponse[KnowledgeUploadResponse]:
    """
    上传文档并触发完整处理流水线（parse → split → embed → store）。

    文件先写入临时目录，处理完毕后自动清理。
    """
    # ① 文件校验
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的文件格式：{ext}，仅支持 {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件大小超出限制（最大 {MAX_FILE_SIZE // 1024 // 1024} MB）",
        )

    logger.info(
        "收到文档上传请求", extra={"filename": filename, "collection": collection_name}
    )

    # ② 写入临时文件
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # ③ 调用处理流水线
        report = await doc_service.process_document(tmp_path)

        return ApiResponse(
            success=True,
            message="文档处理完成",
            data=KnowledgeUploadResponse(
                source=filename,
                collection_name=collection_name,
                total_chunks=report.total_chunks,
                stored=report.stored,
                failed=report.failed,
                duration_ms=report.duration_ms,
            ),
        )

    except DocumentParseError as exc:
        logger.warning("文档解析失败", extra={"filename": filename, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    except Exception as exc:
        logger.error("文档上传处理异常", extra={"filename": filename}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档处理失败，请稍后重试",
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get(
    "/list",
    response_model=ApiResponse[KnowledgeListResponse],
    summary="列出知识库文档",
)
async def list_documents(
    collection_name: str = "default",
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> ApiResponse[KnowledgeListResponse]:
    """
    返回指定 Collection 中所有文档的元信息。

    通过查询 Milvus 元数据字段聚合（按 source 去重）。
    """
    try:
        raw_items = await vector_store.list_documents(collection_name=collection_name)
    except Exception as exc:
        logger.error("获取文档列表失败", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    items = [
        KnowledgeItem(
            source=item["source"],
            file_type=item.get("file_type", "unknown"),
            chunk_count=item.get("chunk_count", 0),
            collection_name=collection_name,
        )
        for item in raw_items
    ]

    return ApiResponse(
        success=True,
        message="获取成功",
        data=KnowledgeListResponse(
            collection_name=collection_name,
            total_documents=len(items),
            items=items,
        ),
    )


@router.delete(
    "/{source_id}",
    response_model=ApiResponse[dict],
    summary="删除知识库文档",
)
async def delete_document(
    source_id: str,
    collection_name: str = "default",
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> ApiResponse[dict]:
    """
    按 source（文件路径）删除文档的所有分块向量。

    source_id 需 URL encode，如 `my%2Ffile.pdf`。
    """
    source = urllib.parse.unquote(source_id)
    logger.info(
        "删除知识库文档", extra={"source": source, "collection": collection_name}
    )

    try:
        deleted_count = await vector_store.delete_by_source(
            source=source, collection_name=collection_name
        )
    except Exception as exc:
        logger.error("删除文档失败", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到文档：{source}",
        )

    return ApiResponse(
        success=True,
        message=f"已删除 {deleted_count} 个向量块",
        data={"source": source, "deleted_chunks": deleted_count},
    )


@router.post(
    "/search",
    response_model=ApiResponse[list],
    summary="直接向量检索（调试用）",
)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> ApiResponse[list]:
    """
    直接查询 Milvus，不经过 LLM，返回原始检索结果。
    用于调试知识库召回质量。
    """
    try:
        docs = await vector_store.similarity_search(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    results = [
        {
            "source": d.metadata.get("source"),
            "chunk_index": d.metadata.get("chunk_index"),
            "score": round(d.metadata.get("score", 0.0), 4),
            "snippet": d.page_content[:100],
        }
        for d in docs
    ]

    return ApiResponse(success=True, message="检索完成", data=results)
