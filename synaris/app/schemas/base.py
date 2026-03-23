"""
@File       : base.py
@Author     : Starry Hung
@Created    : 2026-03-23
@Version    : 1.0.0
@Description: 全局统一 Pydantic 响应模型，供所有 API 路由复用。
@Features:
  - ApiResponse[T]    泛型成功/失败统一响应体
      字段：success / code / message / data / trace_id / timestamp
      类方法：ok() / fail() 快速构造，无需手动填充样板字段
  - PageResponse[T]   分页列表响应（继承 ApiResponse，data 替换为分页结构）
      字段：items / total / page / page_size / total_pages / has_next / has_prev
  - EmptyResponse     无业务数据的成功响应（data=null）
  - 所有模型配置 model_config = ConfigDict(from_attributes=True)
    支持从 SQLAlchemy ORM 对象直接构造

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-23  Starry  Initial creation
"""

from __future__ import annotations

import time
import math
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# 泛型类型变量，代表业务数据载荷类型
T = TypeVar("T")


# ---------------------------------------------------------------------------
# 通用基础模型（所有 Schema 继承此类）
# ---------------------------------------------------------------------------

class SynarisBaseModel(BaseModel):
    """
    项目全局 Pydantic BaseModel。

    统一配置：
      - from_attributes=True  → 支持从 SQLAlchemy ORM 对象构造（替代旧版 orm_mode）
      - populate_by_name=True → 同时支持字段别名与原始字段名赋值
      - use_enum_values=True  → 枚举字段自动转换为其值（int / str）
    """
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )


# ---------------------------------------------------------------------------
# ApiResponse[T] — 统一响应体
# ---------------------------------------------------------------------------

class ApiResponse(SynarisBaseModel, Generic[T]):
    """
    Synaris 统一 API 响应体。

    所有接口返回此结构，确保客户端无需适配不同响应格式。

    字段说明：
      success    - True 表示业务处理成功，False 表示有错误
      code       - 业务错误码（成功时为 0，错误时为 ErrorCode 枚举值）
      message    - 面向用户的描述文字
      data       - 业务数据载荷，失败时为 None
      trace_id   - 全链路追踪 ID（由 TraceID 中间件注入）
      timestamp  - 响应生成时的 Unix 时间戳（秒级，带小数）

    快速构造示例：
        # 成功响应
        return ApiResponse.ok(data={"user_id": "abc"}, message="登录成功")

        # 错误响应
        return ApiResponse.fail(
            code=ErrorCode.LLM_UNAVAILABLE,
            message="AI 服务暂时不可用",
            trace_id=trace_id,
        )
    """

    success: bool = Field(True, description="业务处理是否成功")
    code: int = Field(0, description="业务错误码，0 表示成功")
    message: str = Field("success", description="响应描述")
    data: Optional[T] = Field(None, description="业务数据载荷")
    trace_id: str = Field("", description="全链路追踪 ID")
    timestamp: float = Field(
        default_factory=time.time,
        description="响应生成 Unix 时间戳（秒）",
    )

    # ── 类方法工厂 ──────────────────────────────────────────────────────

    @classmethod
    def ok(
        cls,
        data: Optional[T] = None,
        message: str = "success",
        trace_id: str = "",
    ) -> "ApiResponse[T]":
        """
        构造成功响应。

        示例：
            return ApiResponse.ok(data=user_schema, message="用户信息获取成功")
        """
        return cls(
            success=True,
            code=0,
            message=message,
            data=data,
            trace_id=trace_id,
        )

    @classmethod
    def fail(
        cls,
        code: int,
        message: str,
        data: Optional[Any] = None,
        trace_id: str = "",
    ) -> "ApiResponse[None]":
        """
        构造错误响应。

        示例：
            return ApiResponse.fail(
                code=ErrorCode.LLM_UNAVAILABLE,
                message="AI 服务暂时不可用，请稍后重试",
                trace_id=request.state.trace_id,
            )
        """
        return cls(
            success=False,
            code=int(code),
            message=message,
            data=data,
            trace_id=trace_id,
        )

    @classmethod
    def from_exception(
        cls,
        exc: "AppException",  # type: ignore[name-defined]  # 避免循环导入
        trace_id: str = "",
    ) -> "ApiResponse[None]":
        """
        从 AppException 直接构造错误响应，供异常处理器使用。
        """
        return cls(
            success=False,
            code=int(exc.error_code),
            message=exc.message,
            data=None,
            trace_id=trace_id,
        )


# ---------------------------------------------------------------------------
# 分页数据结构
# ---------------------------------------------------------------------------

class PageData(SynarisBaseModel, Generic[T]):
    """
    分页业务数据载荷（嵌套在 PageResponse.data 中）。

    字段说明：
      items        - 当前页数据列表
      total        - 全量数据总条数
      page         - 当前页码（从 1 开始）
      page_size    - 每页条数
      total_pages  - 总页数（自动计算）
      has_next     - 是否有下一页
      has_prev     - 是否有上一页
    """

    items: List[T] = Field(..., description="当前页数据列表")
    total: int = Field(..., ge=0, description="数据总条数")
    page: int = Field(..., ge=1, description="当前页码（从 1 开始）")
    page_size: int = Field(..., ge=1, le=200, description="每页条数（上限 200）")
    total_pages: int = Field(0, description="总页数")
    has_next: bool = Field(False, description="是否有下一页")
    has_prev: bool = Field(False, description="是否有上一页")

    def model_post_init(self, __context: Any) -> None:
        """自动计算派生字段，避免调用方手动填充。"""
        self.total_pages = math.ceil(self.total / self.page_size) if self.page_size else 0
        self.has_next = self.page < self.total_pages
        self.has_prev = self.page > 1


class PageResponse(SynarisBaseModel, Generic[T]):
    """
    统一分页响应体（与 ApiResponse 结构对齐，data 替换为 PageData）。

    快速构造示例：
        return PageResponse.paginate(
            items=users,
            total=100,
            page=2,
            page_size=10,
            message="用户列表获取成功",
        )
    """

    success: bool = Field(True)
    code: int = Field(0)
    message: str = Field("success")
    data: Optional[PageData[T]] = Field(None, description="分页数据载荷")
    trace_id: str = Field("")
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def paginate(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int,
        message: str = "success",
        trace_id: str = "",
    ) -> "PageResponse[T]":
        """
        快速构造分页成功响应。

        示例：
            return PageResponse.paginate(
                items=task_list,
                total=350,
                page=1,
                page_size=20,
            )
        """
        return cls(
            success=True,
            code=0,
            message=message,
            data=PageData(
                items=items,
                total=total,
                page=page,
                page_size=page_size,
            ),
            trace_id=trace_id,
        )

    @classmethod
    def fail(
        cls,
        code: int,
        message: str,
        trace_id: str = "",
    ) -> "PageResponse[Any]":
        """构造分页查询的错误响应。"""
        return cls(
            success=False,
            code=int(code),
            message=message,
            data=None,
            trace_id=trace_id,
        )


# ---------------------------------------------------------------------------
# EmptyResponse — 无业务数据的成功响应
# ---------------------------------------------------------------------------

class EmptyResponse(SynarisBaseModel):
    """
    无业务数据载荷的成功响应（data 固定为 null）。
    适用于：删除操作、触发异步任务、仅需确认成功的接口。

    快速构造示例：
        return EmptyResponse.ok(message="文档已删除")
        return EmptyResponse.ok()  # message 默认 "success"
    """

    success: bool = Field(True)
    code: int = Field(0)
    message: str = Field("success")
    data: None = Field(None)
    trace_id: str = Field("")
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def ok(
        cls,
        message: str = "success",
        trace_id: str = "",
    ) -> "EmptyResponse":
        return cls(message=message, trace_id=trace_id)

"""

文件关系说明
exceptions.py                    schemas/base.py
──────────────────               ──────────────────────────────
ErrorCode (IntEnum)              SynarisBaseModel
  ↓ 业务码定义                      ↓ 统一 Pydantic 配置基类
AppException (基类)              ApiResponse[T]
  ├── LLMError                     ├── .ok(data, message)
  ├── CacheError                   ├── .fail(code, message)
  │   └── CacheConnectionError     └── .from_exception(exc)
  ├── VectorDBError
  ├── DocumentParseError         PageResponse[T]
  ├── AgentError                   └── .paginate(items, total, ...)
  │   └── ToolExecutionError
  ├── AuthError                  EmptyResponse
  ├── RateLimitError               └── .ok(message)
  ├── NotFoundError
  └── DatabaseError
         ↓
register_exception_handlers(app)
  → 将所有 AppException 转换为 ApiResponse.fail() 格式的 JSONResponse

"""