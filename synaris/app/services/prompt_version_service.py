"""
@File       : prompt_version_service.py
@Author     : Codex
@Created    : 2026-04-16
@Version    : 1.0.0
@Description: Prompt 版本管理服务。
@Features:
  - 自动创建语义化版本（默认递增 patch）
  - 获取当前主生效版本 / A/B 加权版本
  - 一键回滚到指定版本
  - 严格变量校验与模板渲染
  - 支持外部事务注入，也支持服务内部自行管理 DB Session
"""

from __future__ import annotations

import random
import re
from contextlib import asynccontextmanager
from string import Formatter
from typing import Any, AsyncIterator, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from app.core.exceptions import AppException, ErrorCode, NotFoundError
    from app.core.logging import get_logger
    from app.infrastructure.postgres_client import db_session
    from app.models.prompt_version import PromptVersion
except ImportError:  # pragma: no cover - 兼容当前仓库中混用的导入风格
    from core.exceptions import AppException, ErrorCode, NotFoundError  # type: ignore[no-redef]
    from core.logging import get_logger  # type: ignore[no-redef]
    from infrastructure.postgres_client import db_session  # type: ignore[no-redef]
    from models.prompt_version import PromptVersion  # type: ignore[no-redef]

logger = get_logger(__name__)

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?$"
)
_FORMATTER = Formatter()


def _parse_semver(version: str) -> tuple[int, int, int, str]:
    """解析 semver；非法版本号抛出业务异常。"""
    match = _SEMVER_RE.fullmatch(version.strip())
    if not match:
        raise AppException(
            message=f"非法版本号：{version}，要求形如 1.0.0 或 1.0.0-beta",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    major, minor, patch, prerelease = match.groups()
    return int(major), int(minor), int(patch), prerelease or ""


def _semver_sort_key(version: str) -> tuple[int, int, int, int, str]:
    """稳定版优先于 prerelease，同主次补丁下按 prerelease 字典序排序。"""
    major, minor, patch, prerelease = _parse_semver(version)
    return (
        major,
        minor,
        patch,
        1 if not prerelease else 0,
        prerelease,
    )


def _bump_patch(version: str) -> str:
    major, minor, patch, _ = _parse_semver(version)
    return f"{major}.{minor}.{patch + 1}"


def _extract_field_names(template_content: str) -> list[str]:
    """
    提取模板中的根变量名。

    兼容:
      {question}
      {user[name]}
      {profile.lang}
    缺失检查时只校验根变量 user / profile / question 是否存在。
    """
    names: list[str] = []
    for _, field_name, _, _ in _FORMATTER.parse(template_content):
        if not field_name:
            continue
        root_name = re.split(r"[.\[]", field_name, maxsplit=1)[0].strip()
        if root_name and root_name not in names:
            names.append(root_name)
    return names


class PromptVersionService:
    """PromptVersion 的创建、查询、回滚与渲染服务。"""

    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        self._session = session

    @asynccontextmanager
    async def _session_scope(self) -> AsyncIterator[AsyncSession]:
        if self._session is not None:
            yield self._session
            return

        async with db_session() as session:
            yield session

    async def _list_versions(
        self,
        session: AsyncSession,
        name: str,
        *,
        active_only: bool = False,
    ) -> list[PromptVersion]:
        stmt = select(PromptVersion).where(
            PromptVersion.name == name,
            PromptVersion.is_deleted.is_(False),
        )
        if active_only:
            stmt = stmt.where(PromptVersion.is_active.is_(True))

        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _get_exact_version(
        self,
        session: AsyncSession,
        name: str,
        version: str,
    ) -> Optional[PromptVersion]:
        stmt = select(PromptVersion).where(
            PromptVersion.name == name,
            PromptVersion.version == version,
            PromptVersion.is_deleted.is_(False),
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _next_version(self, session: AsyncSession, name: str) -> str:
        versions = await self._list_versions(session, name)
        if not versions:
            return "1.0.0"

        latest = max(versions, key=lambda item: _semver_sort_key(item.version))
        return _bump_patch(latest.version)

    @staticmethod
    def _validate_weight(weight: float) -> None:
        if weight < 0:
            raise AppException(
                message="ab_test_weight 不能为负数",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

    async def create_version(
        self,
        name: str,
        content: str,
        variables: Optional[dict[str, Any]] = None,
        *,
        version: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Any = None,
        tags: Optional[list[str]] = None,
        activate: bool = True,
        deactivate_others: bool = True,
        ab_test_weight: float = 100.0,
    ) -> PromptVersion:
        """
        创建新 Prompt 版本。

        默认行为：
          - 自动 patch bump
          - 新版本激活
          - 关闭同名旧版本的 active 标记

        若要做 A/B：
          activate=True, deactivate_others=False, ab_test_weight=50.0
        """
        if not name.strip():
            raise AppException(
                message="Prompt 名称不能为空",
                error_code=ErrorCode.VALIDATION_ERROR,
            )
        if not content.strip():
            raise AppException(
                message="Prompt 内容不能为空",
                error_code=ErrorCode.VALIDATION_ERROR,
            )
        self._validate_weight(ab_test_weight)

        async with self._session_scope() as session:
            resolved_version = version or await self._next_version(session, name)
            _parse_semver(resolved_version)

            existing = await self._get_exact_version(session, name, resolved_version)
            if existing is not None:
                raise AppException(
                    message=f"Prompt {name} 的版本 {resolved_version} 已存在",
                    error_code=ErrorCode.INVALID_REQUEST,
                )

            if activate and deactivate_others:
                await session.execute(
                    update(PromptVersion)
                    .where(
                        PromptVersion.name == name,
                        PromptVersion.is_deleted.is_(False),
                    )
                    .values(is_active=False)
                )

            prompt_version = PromptVersion(
                name=name.strip(),
                version=resolved_version,
                content=content,
                variables=variables or {},
                description=description,
                created_by=created_by,
                tags=tags,
                is_active=activate,
                ab_test_weight=ab_test_weight,
            )
            session.add(prompt_version)
            await session.flush()
            await session.refresh(prompt_version)

            logger.info(
                "Prompt 版本已创建",
                extra={
                    "name": name,
                    "version": resolved_version,
                    "activate": activate,
                    "deactivate_others": deactivate_others,
                    "ab_test_weight": ab_test_weight,
                },
            )
            return prompt_version

    async def get_active(self, name: str) -> PromptVersion:
        """
        返回当前主生效版本。

        如果有多个 active 版本，优先返回：
          1. ab_test_weight 更高的
          2. semver 更高的
          3. created_at 更新的
        """
        async with self._session_scope() as session:
            active_versions = await self._list_versions(session, name, active_only=True)
            if not active_versions:
                raise NotFoundError(
                    message=f"未找到 Prompt {name} 的生效版本",
                    error_code=ErrorCode.PROMPT_VERSION_NOT_FOUND,
                )

            if len(active_versions) > 1:
                logger.info(
                    "检测到多个 active Prompt 版本，按主版本规则选择",
                    extra={"name": name, "count": len(active_versions)},
                )

            return max(
                active_versions,
                key=lambda item: (
                    item.ab_test_weight,
                    *_semver_sort_key(item.version),
                    item.created_at.timestamp() if item.created_at else 0.0,
                ),
            )

    async def rollback(self, name: str, version: str) -> None:
        """将指定版本切换为唯一生效版本。"""
        async with self._session_scope() as session:
            target = await self._get_exact_version(session, name, version)
            if target is None:
                raise NotFoundError(
                    message=f"未找到 Prompt {name} 的版本 {version}",
                    error_code=ErrorCode.PROMPT_VERSION_NOT_FOUND,
                )

            await session.execute(
                update(PromptVersion)
                .where(
                    PromptVersion.name == name,
                    PromptVersion.is_deleted.is_(False),
                )
                .values(is_active=False)
            )
            target.is_active = True
            if target.ab_test_weight <= 0:
                target.ab_test_weight = 100.0

            await session.flush()

            logger.info(
                "Prompt 已回滚到指定版本",
                extra={"name": name, "version": version},
            )

    async def get_ab_version(self, name: str) -> PromptVersion:
        """
        从 active 版本中按权重随机选择。

        权重全部为 0 时退化为均匀随机。
        只有一个 active 版本时直接返回。
        """
        async with self._session_scope() as session:
            active_versions = await self._list_versions(session, name, active_only=True)
            if not active_versions:
                return await self.get_active(name)

            if len(active_versions) == 1:
                return active_versions[0]

            positive_weight_versions = [
                item for item in active_versions if item.ab_test_weight > 0
            ]
            candidates = positive_weight_versions or active_versions
            weights = (
                [item.ab_test_weight for item in candidates]
                if positive_weight_versions
                else None
            )
            selected = random.choices(candidates, weights=weights, k=1)[0]

            logger.debug(
                "A/B Prompt 版本已选中",
                extra={
                    "name": name,
                    "selected_version": selected.version,
                    "candidate_count": len(candidates),
                },
            )
            return selected

    def render(
        self,
        template_content: str,
        variables_dict: Optional[dict[str, Any]] = None,
    ) -> str:
        """严格校验变量后渲染模板。"""
        if not template_content:
            raise AppException(
                message="模板内容不能为空",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        values = variables_dict or {}
        required_fields = _extract_field_names(template_content)
        missing = [field for field in required_fields if field not in values]
        if missing:
            raise AppException(
                message=f"Prompt 渲染缺少变量: {', '.join(sorted(missing))}",
                error_code=ErrorCode.VALIDATION_ERROR,
                detail={"required_fields": required_fields},
            )

        try:
            return template_content.format(**values)
        except Exception as exc:  # pragma: no cover - 防止 format 运行时边界问题
            raise AppException(
                message=f"Prompt 渲染失败: {exc}",
                error_code=ErrorCode.INVALID_REQUEST,
            ) from exc


__all__ = ["PromptVersionService"]
