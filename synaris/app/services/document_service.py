"""
@File       : document_service.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 文档处理流水线服务（解析 → 分块 → 向量化 → 存储）。
@Features:
  - 多格式文档解析：PDF（PyPDFLoader）/ Word（Docx2txtLoader）/ TXT & Markdown（TextLoader）
  - 解析失败统一抛出 DocumentParseError，附带文件路径与原始错误
  - RecursiveCharacterTextSplitter 分块：chunk_size=512，chunk_overlap=50
  - 元数据保留：source / page_num / chunk_index / file_type / file_size
  - 完整异步处理流水线：parse → split → embed_and_store → 返回处理报告
  - asyncio.Semaphore 控制并发数（默认 5，可通过配置覆盖）
  - 进度回调接口（ProgressCallback），供 Celery Worker / WebSocket 实时推送
  - 处理报告：{ total_chunks, stored, failed, duration_ms, source }

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from core.exceptions import DocumentParseError
from core.logging import get_logger, log_execution_time
from services.vector_store import VectorStoreService

# ─────────────────────────────────────────────
# 模块级别常量 & 类型别名
# ─────────────────────────────────────────────

logger = get_logger(__name__)
settings = get_settings()

# 进度回调签名：(source: str, current: int, total: int, stage: str) -> None
ProgressCallback = Callable[[str, int, int, str], None]


# ─────────────────────────────────────────────
# 文件类型枚举
# ─────────────────────────────────────────────


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"

    @classmethod
    def from_path(cls, path: str | Path) -> "FileType":
        """根据文件扩展名推断类型。"""
        suffix = Path(path).suffix.lower().lstrip(".")
        mapping = {
            "pdf": cls.PDF,
            "docx": cls.DOCX,
            "doc": cls.DOCX,
            "txt": cls.TXT,
            "md": cls.MARKDOWN,
            "markdown": cls.MARKDOWN,
        }
        if suffix not in mapping:
            raise DocumentParseError(
                file_path=str(path),
                reason=f"不支持的文件格式：.{suffix}，仅支持 PDF / DOCX / TXT / Markdown",
            )
        return mapping[suffix]


# ─────────────────────────────────────────────
# 数据结构：处理报告
# ─────────────────────────────────────────────


@dataclass
class ProcessingReport:
    """单个文档的处理结果报告。"""

    source: str
    file_type: str
    total_chunks: int = 0
    stored: int = 0
    failed: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "file_type": self.file_type,
            "total_chunks": self.total_chunks,
            "stored": self.stored,
            "failed": self.failed,
            "duration_ms": round(self.duration_ms, 2),
            "success_rate": (
                round(self.stored / self.total_chunks * 100, 1)
                if self.total_chunks > 0
                else 0.0
            ),
            "error": self.error,
        }


@dataclass
class BatchProcessingReport:
    """批量处理的汇总报告。"""

    total_files: int = 0
    succeeded_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_stored: int = 0
    total_duration_ms: float = 0.0
    reports: List[ProcessingReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "succeeded_files": self.succeeded_files,
            "failed_files": self.failed_files,
            "total_chunks": self.total_chunks,
            "total_stored": self.total_stored,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "reports": [r.to_dict() for r in self.reports],
        }


# ─────────────────────────────────────────────
# 核心服务类
# ─────────────────────────────────────────────


class DocumentService:
    """
    文档处理流水线服务。

    架构类比：
      - parse_document()  → 「原材料入厂」：拆包、识别格式
      - _split_chunks()   → 「切割加工」：将原材料切成标准规格
      - _embed_and_store() → 「入库编码」：贴上语义标签，放入向量仓库
      - process_document() → 「一键全流程」：贯穿以上三道工序
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_concurrency: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # 文本分块器（一次构建，全局复用）
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            # 分隔优先级：段落 → 句子 → 逗号 → 空格 → 字符
            separators=["\n\n", "\n", "。", ".", "，", ",", " ", ""],
            keep_separator=False,
        )

        logger.info(
            "DocumentService 初始化完成",
            extra={
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "max_concurrency": max_concurrency,
            },
        )

    # ─────────────────────────────────────────
    # Step 1 — 文档解析
    # ─────────────────────────────────────────

    async def parse_document(
        self,
        file_path: str | Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Document]:
        """
        异步解析文档，返回 LangChain Document 列表。

        每个 Document 附带元数据：
          - source    : 文件路径（用于溯源）
          - page_num  : 页码（PDF 有效，其余为 0）
          - file_type : 文件格式
          - file_size : 文件大小（字节）

        Raises:
            DocumentParseError: 文件不存在、格式不支持、解析失败时抛出
        """
        file_path = Path(file_path)
        source = str(file_path)

        # 文件存在性校验
        if not file_path.exists():
            raise DocumentParseError(
                file_path=source,
                reason="文件不存在",
            )

        file_type = FileType.from_path(file_path)
        file_size = file_path.stat().st_size

        self._notify(progress_callback, source, 0, 3, "parsing")
        logger.info("开始解析文档", extra={"source": source, "file_type": file_type})

        try:
            # 根据文件类型选择 LangChain 加载器（工厂模式）
            raw_docs = await self._load_with_loader(file_path, file_type)
        except DocumentParseError:
            raise
        except Exception as exc:
            logger.error(
                "文档解析失败",
                extra={"source": source, "error": str(exc)},
                exc_info=True,
            )
            raise DocumentParseError(
                file_path=source,
                reason=f"解析过程中发生异常：{exc}",
            ) from exc

        # 统一注入元数据
        for page_num, doc in enumerate(raw_docs):
            doc.metadata.update(
                {
                    "source": source,
                    "page_num": doc.metadata.get("page", page_num),
                    "file_type": file_type.value,
                    "file_size": file_size,
                }
            )

        self._notify(progress_callback, source, 1, 3, "parsed")
        logger.info(
            "文档解析完成",
            extra={"source": source, "page_count": len(raw_docs)},
        )
        return raw_docs

    async def _load_with_loader(
        self, file_path: Path, file_type: FileType
    ) -> List[Document]:
        """
        在线程池中运行同步加载器，避免阻塞事件循环。

        LangChain 原生加载器均为同步 I/O，
        通过 asyncio.to_thread() 将其移交线程池执行。
        """
        loop = asyncio.get_event_loop()

        if file_type == FileType.PDF:
            loader = PyPDFLoader(str(file_path))
        elif file_type == FileType.DOCX:
            loader = Docx2txtLoader(str(file_path))
        elif file_type in (FileType.TXT, FileType.MARKDOWN):
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            raise DocumentParseError(
                file_path=str(file_path),
                reason=f"内部错误：未知文件类型 {file_type}",
            )

        # 同步 load() 移至线程池执行
        docs: List[Document] = await asyncio.to_thread(loader.load)

        if not docs:
            raise DocumentParseError(
                file_path=str(file_path),
                reason="文档解析结果为空，文件可能已损坏或内容为空",
            )

        return docs

    # ─────────────────────────────────────────
    # Step 2 — 文档分块
    # ─────────────────────────────────────────

    def _split_chunks(self, documents: List[Document]) -> List[Document]:
        """
        将解析后的 Document 列表切分为固定大小的 Chunk。

        切分后每个 Chunk 的元数据额外追加：
          - chunk_index : 该文档内的分块序号（从 0 开始）
          - chunk_size  : 实际字符数
        """
        chunks = self._splitter.split_documents(documents)

        # 按 source 分组，为每个来源的 chunk 独立编号
        source_counter: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            idx = source_counter.get(source, 0)
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            source_counter[source] = idx + 1

        logger.debug(
            "文档分块完成",
            extra={
                "input_pages": len(documents),
                "output_chunks": len(chunks),
            },
        )
        return chunks

    # ─────────────────────────────────────────
    # Step 3 — 向量化 & 存储
    # ─────────────────────────────────────────

    async def _embed_and_store(
        self,
        chunks: List[Document],
        source: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> tuple[int, int]:
        """
        将分块向量化并写入 Milvus。

        Returns:
            (stored_count, failed_count)
        """
        if not chunks:
            return 0, 0

        self._notify(progress_callback, source, 2, 3, "storing")

        try:
            # VectorStoreService.upsert_documents() 内部处理 Embedding + Milvus 写入
            stored_count = await self._vector_store.upsert_documents(chunks)
            failed_count = len(chunks) - stored_count

            logger.info(
                "向量存储完成",
                extra={
                    "source": source,
                    "total": len(chunks),
                    "stored": stored_count,
                    "failed": failed_count,
                },
            )
            return stored_count, failed_count

        except Exception as exc:
            logger.error(
                "向量存储失败",
                extra={"source": source, "error": str(exc)},
                exc_info=True,
            )
            return 0, len(chunks)

    # ─────────────────────────────────────────
    # 完整流水线：parse → split → store
    # ─────────────────────────────────────────

    @log_execution_time
    async def process_document(
        self,
        file_path: str | Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ProcessingReport:
        """
        单文档完整处理流水线（parse → split → embed & store）。

        Returns:
            ProcessingReport 处理报告
        """
        file_path = Path(file_path)
        source = str(file_path)
        file_type = "unknown"
        start_time = time.monotonic()

        report = ProcessingReport(source=source, file_type=file_type)

        try:
            file_type = FileType.from_path(file_path).value
            report.file_type = file_type

            # ① 解析
            raw_docs = await self.parse_document(file_path, progress_callback)

            # ② 分块
            chunks = self._split_chunks(raw_docs)
            report.total_chunks = len(chunks)

            # ③ 向量化 & 存储
            stored, failed = await self._embed_and_store(
                chunks, source, progress_callback
            )
            report.stored = stored
            report.failed = failed

        except DocumentParseError as exc:
            logger.warning(
                "文档处理被中止（解析失败）",
                extra={"source": source, "error": str(exc)},
            )
            report.error = str(exc)
            report.failed = report.total_chunks or 1

        except Exception as exc:
            logger.error(
                "文档处理发生未预期异常", extra={"source": source}, exc_info=True
            )
            report.error = f"未预期异常：{exc}"
            report.failed = report.total_chunks or 1

        finally:
            report.duration_ms = (time.monotonic() - start_time) * 1000
            self._notify(progress_callback, source, 3, 3, "done")
            logger.info("文档处理流水线完成", extra=report.to_dict())

        return report

    # ─────────────────────────────────────────
    # 批量处理
    # ─────────────────────────────────────────

    async def process_documents_batch(
        self,
        file_paths: List[str | Path],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchProcessingReport:
        """
        批量异步处理多个文档。

        使用 asyncio.Semaphore 控制并发上限，
        asyncio.gather() 收集所有结果（return_exceptions=True 防止单个失败中断批次）。

        架构类比：工厂流水线同时运行多条产线，
                  Semaphore 是厂房里的「工位数量」上限。
        """
        batch_start = time.monotonic()
        batch_report = BatchProcessingReport(total_files=len(file_paths))

        async def _process_with_semaphore(fp: str | Path) -> ProcessingReport:
            async with self._semaphore:
                return await self.process_document(fp, progress_callback)

        logger.info(
            "批量处理开始",
            extra={"total_files": len(file_paths)},
        )

        # gather 收集所有任务结果，单个任务异常不影响其他任务
        results = await asyncio.gather(
            *[_process_with_semaphore(fp) for fp in file_paths],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                # gather return_exceptions=True 时未被捕获的异常走此分支
                failed_report = ProcessingReport(
                    source="unknown",
                    file_type="unknown",
                    failed=1,
                    error=str(result),
                )
                batch_report.reports.append(failed_report)
                batch_report.failed_files += 1
            else:
                batch_report.reports.append(result)
                batch_report.total_chunks += result.total_chunks
                batch_report.total_stored += result.stored
                if result.error is None:
                    batch_report.succeeded_files += 1
                else:
                    batch_report.failed_files += 1

        batch_report.total_duration_ms = (time.monotonic() - batch_start) * 1000

        logger.info("批量处理完成", extra=batch_report.to_dict())
        return batch_report

    # ─────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────

    @staticmethod
    def _notify(
        callback: Optional[ProgressCallback],
        source: str,
        current: int,
        total: int,
        stage: str,
    ) -> None:
        """安全调用进度回调（忽略回调自身的异常，不影响主流程）。"""
        if callback is None:
            return
        try:
            callback(source, current, total, stage)
        except Exception as exc:
            logger.warning(
                "进度回调执行失败（已忽略）",
                extra={"error": str(exc)},
            )

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """返回当前支持的文件扩展名列表。"""
        return [".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"]

    @staticmethod
    def is_supported(file_path: str | Path) -> bool:
        """快速判断文件是否被支持，无需实例化 FileType。"""
        return (
            Path(file_path).suffix.lower() in DocumentService.get_supported_extensions()
        )
