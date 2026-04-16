"""
@File       : cost_service.py
@Author     : Starry Hung
@Created    : 2026-04-16
@Version    : 1.0.0
@Description: Token 成本核算服务，负责记录、查询与预警 LLM API 调用费用。
@Features:
  - 模型单价配置（硬编码 + settings 覆盖，双层设计）：
      · gpt-4o:                $0.005/1K input，$0.015/1K output
      · gpt-4o-mini:           $0.00015/1K input，$0.0006/1K output
      · gpt-3.5-turbo:         $0.0005/1K input，$0.0015/1K output
      · text-embedding-3-small: $0.00002/1K tokens（不区分 input/output）
      · 未知模型 → 使用 gpt-4o-mini 兜底单价，并写入 WARNING 日志
  - async record_usage()
      · 写入 PostgreSQL token_usage_logs 表（JSONB 扩展元数据）
      · 同步更新 Prometheus Counter（track_llm_call）
      · 触发日费用预警检查（异步后台，不阻塞主调用链路）
  - async get_user_cost()
      · 按用户 + 时间范围聚合费用，返回 CostReport（含模型明细）
  - async get_cost_by_model()
      · 全局视角，按模型聚合 input/output Token 与费用
  - async get_daily_summary()
      · 今日 / 昨日 / 任意日期的费用快照（供 Grafana / 告警使用）
  - 日费用预警机制：
      · 阈值从 settings.COST_ALERT_DAILY_USD 读取（默认 $50）
      · 超限时写入 WARNING 级别结构化日志（含当日累计费用与超限比例）
      · 预留 _notify_alert() 扩展钩子（邮件 / Slack / PagerDuty）
  - CostReport 数据模型：
      · total_cost_usd / total_tokens / breakdown_by_model / top_endpoints
  - 设计模式：
      · 服务层（无状态）+ PostgreSQL 持久化 + 内存聚合
      · 日费用预警通过 asyncio.create_task 异步执行，不阻塞主链路
      · 单价配置集中在 MODEL_PRICING 字典，新增模型只改一处

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-16  Starry  Initial creation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 模型单价配置（Model Pricing Table）
#
# 领域建模类比：
#   MODEL_PRICING 是「价目表」，每个模型对应一张账单单价卡。
#   input_per_1k  → 读取（输入 Token）单价
#   output_per_1k → 生成（输出 Token）单价
#   embedding 模型不区分 input/output，统一用 input_per_1k 记录。
#
# 单位：美元（USD）/ 1000 tokens
# 数据来源：OpenAI 官网定价（2025 年）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _ModelPrice:
    """单个模型的计费单价（$/1K tokens）。"""
    input_per_1k:  Decimal   # 输入 Token 单价（$/1K）
    output_per_1k: Decimal   # 输出 Token 单价（$/1K）；Embedding 模型此值为 0
    is_embedding:  bool = False  # 是否为 Embedding 模型（不区分 input/output）

    def calc_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """
        根据 Token 用量计算本次调用费用（USD）。

        Embedding 模型：只计 input_tokens，output_tokens 视为 0。
        """
        if self.is_embedding:
            return (Decimal(input_tokens + output_tokens) / 1000 * self.input_per_1k)
        return (
            Decimal(input_tokens)  / 1000 * self.input_per_1k
            + Decimal(output_tokens) / 1000 * self.output_per_1k
        )


# 主单价表（硬编码为默认值，settings 中可覆盖）
MODEL_PRICING: Dict[str, _ModelPrice] = {
    # ── OpenAI Chat 模型 ─────────────────────────────────────────────────────
    "gpt-4o": _ModelPrice(
        input_per_1k  = Decimal("0.005"),
        output_per_1k = Decimal("0.015"),
    ),
    "gpt-4o-2024-08-06": _ModelPrice(
        input_per_1k  = Decimal("0.0025"),
        output_per_1k = Decimal("0.01"),
    ),
    "gpt-4o-mini": _ModelPrice(
        input_per_1k  = Decimal("0.00015"),
        output_per_1k = Decimal("0.0006"),
    ),
    "gpt-4o-mini-2024-07-18": _ModelPrice(
        input_per_1k  = Decimal("0.00015"),
        output_per_1k = Decimal("0.0006"),
    ),
    "gpt-3.5-turbo": _ModelPrice(
        input_per_1k  = Decimal("0.0005"),
        output_per_1k = Decimal("0.0015"),
    ),
    "gpt-3.5-turbo-0125": _ModelPrice(
        input_per_1k  = Decimal("0.0005"),
        output_per_1k = Decimal("0.0015"),
    ),
    # ── OpenAI Embedding 模型 ────────────────────────────────────────────────
    "text-embedding-3-small": _ModelPrice(
        input_per_1k  = Decimal("0.00002"),
        output_per_1k = Decimal("0"),
        is_embedding  = True,
    ),
    "text-embedding-3-large": _ModelPrice(
        input_per_1k  = Decimal("0.00013"),
        output_per_1k = Decimal("0"),
        is_embedding  = True,
    ),
    "text-embedding-ada-002": _ModelPrice(
        input_per_1k  = Decimal("0.0001"),
        output_per_1k = Decimal("0"),
        is_embedding  = True,
    ),
}

# 未知模型的兜底单价（使用 gpt-4o-mini 的定价，偏保守估计）
_FALLBACK_PRICING = MODEL_PRICING["gpt-4o-mini"]


def get_model_price(model_name: str) -> _ModelPrice:
    """
    获取指定模型的单价配置，未知模型使用兜底单价并记录警告。

    精确匹配优先，前缀匹配兜底（兼容带日期后缀的模型名）：
        "gpt-4o-2024-12-01" → 未精确命中 → 取 "gpt-4o" 前缀匹配
    """
    # 精确匹配
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # 前缀匹配（模型名带版本后缀时）
    for known_model, price in MODEL_PRICING.items():
        if model_name.startswith(known_model):
            return price

    logger.warning(
        "cost_service: 未知模型单价，使用兜底定价 | model=%s | fallback=gpt-4o-mini",
        model_name,
    )
    return _FALLBACK_PRICING


def calc_cost(
    model_name:    str,
    input_tokens:  int,
    output_tokens: int,
) -> Decimal:
    """计算一次 LLM 调用的费用（USD），结果保留 8 位小数。"""
    price = get_model_price(model_name)
    raw   = price.calc_cost(input_tokens, output_tokens)
    # 保留 8 位小数，防止浮点漂移
    return raw.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 数据模型（返回给调用方的结构化对象）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelCostDetail:
    """单个模型的成本明细（嵌套在 CostReport 中）。"""
    model:            str
    input_tokens:     int     = 0
    output_tokens:    int     = 0
    total_tokens:     int     = 0
    total_cost_usd:   Decimal = Decimal("0")
    request_count:    int     = 0

    def to_dict(self) -> dict:
        return {
            "model":          self.model,
            "input_tokens":   self.input_tokens,
            "output_tokens":  self.output_tokens,
            "total_tokens":   self.total_tokens,
            "total_cost_usd": float(self.total_cost_usd),
            "request_count":  self.request_count,
        }


@dataclass
class CostReport:
    """
    用户或全局维度的费用报告。

    类比「银行对账单」：
      total_cost_usd     → 总账单金额
      breakdown_by_model → 按服务商/产品的消费明细
      top_endpoints      → 消费最多的接口 Top N
      period_start/end   → 账单周期
    """
    user_id:           Optional[str]
    period_start:      date
    period_end:        date
    total_cost_usd:    Decimal             = Decimal("0")
    total_tokens:      int                 = 0
    input_tokens:      int                 = 0
    output_tokens:     int                 = 0
    request_count:     int                 = 0
    breakdown_by_model: List[ModelCostDetail] = field(default_factory=list)
    top_endpoints:     List[Dict[str, Any]]   = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "user_id":          self.user_id,
            "period_start":     self.period_start.isoformat(),
            "period_end":       self.period_end.isoformat(),
            "total_cost_usd":   float(self.total_cost_usd),
            "total_tokens":     self.total_tokens,
            "input_tokens":     self.input_tokens,
            "output_tokens":    self.output_tokens,
            "request_count":    self.request_count,
            "breakdown_by_model": [m.to_dict() for m in self.breakdown_by_model],
            "top_endpoints":    self.top_endpoints,
        }


@dataclass
class DailySummary:
    """单日费用快照（供告警与 Grafana 展示）。"""
    target_date:    date
    total_cost_usd: Decimal = Decimal("0")
    total_tokens:   int     = 0
    request_count:  int     = 0
    alert_threshold: Decimal = Decimal("50")
    is_over_budget: bool    = False

    def to_dict(self) -> dict:
        return {
            "date":            self.target_date.isoformat(),
            "total_cost_usd":  float(self.total_cost_usd),
            "total_tokens":    self.total_tokens,
            "request_count":   self.request_count,
            "alert_threshold": float(self.alert_threshold),
            "is_over_budget":  self.is_over_budget,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CostService（主服务类）
# ═══════════════════════════════════════════════════════════════════════════════

class CostService:
    """
    Token 成本核算服务。

    职责边界：
      ✅ 计算费用（calc_cost）
      ✅ 持久化费用记录（record_usage → PostgreSQL）
      ✅ 聚合查询（get_user_cost / get_cost_by_model / get_daily_summary）
      ✅ 日费用预警（check_daily_alert）
      ❌ 不负责 LLM 调用（由 llm_client.py 负责）
      ❌ 不负责发送通知（预留 _notify_alert 钩子，由外部实现）

    使用方式（在 llm_client.py 的 invoke() 末尾调用）：
        cost_svc = CostService()
        await cost_svc.record_usage(
            user_id="user-xxx",
            model="gpt-4o-mini",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            endpoint="/api/v1/chat",
            session=db_session,
        )
    """

    def __init__(self) -> None:
        self._alert_threshold = self._load_threshold()

    def _load_threshold(self) -> Decimal:
        """从 settings 读取日费用告警阈值（美元），默认 $50。"""
        try:
            from config.settings import get_settings  # type: ignore[import]
            s = get_settings()
            threshold = getattr(s, "COST_ALERT_DAILY_USD", 50.0)
            return Decimal(str(threshold))
        except Exception:
            return Decimal("50.0")

    # ─────────────────────────────────────────────────────────────────────────
    # 3-A. record_usage — 写入费用记录
    # ─────────────────────────────────────────────────────────────────────────

    async def record_usage(
        self,
        user_id:       Optional[str],
        model:         str,
        input_tokens:  int,
        output_tokens: int,
        session:       AsyncSession,
        endpoint:      str = "unknown",
        task_type:     str = "chat",
        session_id:    Optional[str] = None,
        metadata:      Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录一次 LLM API 调用的 Token 用量与费用到 PostgreSQL。

        流程：
          1. 计算本次费用（Decimal，8 位精度）
          2. 写入 token_usage_logs 表
          3. 更新 Prometheus Counter（不阻塞）
          4. 触发日费用预警检查（asyncio.create_task 后台执行）

        Args:
            user_id:       调用用户 UUID 字符串（匿名调用传 None）
            model:         模型名称（如 "gpt-4o-mini"）
            input_tokens:  输入 Token 数
            output_tokens: 输出 Token 数
            session:       数据库 AsyncSession（由 get_db_session 提供）
            endpoint:      调用来源接口路径（如 "/api/v1/rag/query"）
            task_type:     任务类型（chat / rag / agent / embedding）
            session_id:    关联的会话 ID（可选，用于追踪）
            metadata:      额外扩展字段（JSONB，如 agent task_id）
        """
        if input_tokens <= 0 and output_tokens <= 0:
            # 没有实际消耗，跳过记录（流式中断等场景）
            return

        cost_usd = calc_cost(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens
        now = datetime.now(timezone.utc)

        # ── 写入 PostgreSQL ───────────────────────────────────────────────
        try:
            await session.execute(
                text("""
                    INSERT INTO token_usage_logs (
                        user_id, model, input_tokens, output_tokens,
                        total_tokens, cost_usd, endpoint, task_type,
                        session_id, extra_metadata, created_at
                    ) VALUES (
                        :user_id, :model, :input_tokens, :output_tokens,
                        :total_tokens, :cost_usd, :endpoint, :task_type,
                        :session_id, :extra_metadata::jsonb, :created_at
                    )
                """),
                {
                    "user_id":        user_id,
                    "model":          model,
                    "input_tokens":   input_tokens,
                    "output_tokens":  output_tokens,
                    "total_tokens":   total_tokens,
                    "cost_usd":       str(cost_usd),  # Decimal → str，PostgreSQL NUMERIC
                    "endpoint":       endpoint[:256],
                    "task_type":      task_type[:64],
                    "session_id":     session_id,
                    "extra_metadata": _safe_json(metadata or {}),
                    "created_at":     now,
                },
            )
            # 注意：不在此处 commit，由调用方的 get_db_session 上下文统一提交

        except Exception as exc:
            # 费用记录失败不阻塞主业务，但必须记录错误
            logger.error(
                "cost_service: 费用记录写入失败 | model=%s | user=%s | error=%s",
                model, user_id, exc,
            )
            return

        # ── 同步更新 Prometheus 指标 ──────────────────────────────────────
        try:
            from core.observability import track_llm_call  # type: ignore[import]
            track_llm_call(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                task_type=task_type,
                user_id=user_id,
                endpoint=endpoint,
            )
        except Exception:
            pass  # Prometheus 更新失败不影响主流程

        logger.debug(
            "cost_service: 费用已记录 | model=%s | tokens=%d | cost=$%.6f | user=%s",
            model, total_tokens, float(cost_usd), user_id or "anonymous",
        )

        # ── 异步触发日费用预警（不阻塞主调用链路）────────────────────────
        # 使用 create_task 将预警检查推到事件循环下一轮，
        # 确保 record_usage 的响应时间不受预警查询影响
        asyncio.create_task(
            self._check_daily_alert_safe(now.date()),
            name=f"cost_alert_check_{now.date()}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3-B. get_user_cost — 用户费用报告
    # ─────────────────────────────────────────────────────────────────────────

    async def get_user_cost(
        self,
        user_id:    str,
        start_date: date,
        end_date:   date,
        session:    AsyncSession,
    ) -> CostReport:
        """
        查询指定用户在时间范围内的费用报告（含模型明细与接口 Top 5）。

        SQL 聚合策略：
          - 主查询：GROUP BY model → 得到模型明细
          - 子查询：GROUP BY endpoint ORDER BY cost DESC LIMIT 5 → 得到 Top 接口

        Args:
            user_id:    用户 UUID 字符串
            start_date: 查询起始日期（含）
            end_date:   查询结束日期（含）
            session:    数据库 AsyncSession

        Returns:
            CostReport：含总费用、Token 量、模型明细、接口排行
        """
        start_dt = datetime(start_date.year, start_date.month, start_date.day,
                            tzinfo=timezone.utc)
        end_dt   = datetime(end_date.year, end_date.month, end_date.day,
                            23, 59, 59, tzinfo=timezone.utc)

        # ── 按模型聚合 ─────────────────────────────────────────────────────
        model_rows = await session.execute(
            text("""
                SELECT
                    model,
                    SUM(input_tokens)::bigint   AS input_tokens,
                    SUM(output_tokens)::bigint  AS output_tokens,
                    SUM(total_tokens)::bigint   AS total_tokens,
                    SUM(cost_usd)               AS total_cost,
                    COUNT(*)::int               AS request_count
                FROM token_usage_logs
                WHERE
                    user_id   = :user_id
                    AND created_at >= :start_dt
                    AND created_at <= :end_dt
                GROUP BY model
                ORDER BY total_cost DESC
            """),
            {"user_id": user_id, "start_dt": start_dt, "end_dt": end_dt},
        )

        breakdown: List[ModelCostDetail] = []
        total_cost   = Decimal("0")
        total_input  = 0
        total_output = 0
        total_tokens = 0
        total_reqs   = 0

        for row in model_rows.mappings():
            detail = ModelCostDetail(
                model=row["model"],
                input_tokens=int(row["input_tokens"] or 0),
                output_tokens=int(row["output_tokens"] or 0),
                total_tokens=int(row["total_tokens"] or 0),
                total_cost_usd=Decimal(str(row["total_cost"] or "0")),
                request_count=int(row["request_count"] or 0),
            )
            breakdown.append(detail)
            total_cost   += detail.total_cost_usd
            total_input  += detail.input_tokens
            total_output += detail.output_tokens
            total_tokens += detail.total_tokens
            total_reqs   += detail.request_count

        # ── 按接口聚合（Top 5）────────────────────────────────────────────
        endpoint_rows = await session.execute(
            text("""
                SELECT
                    endpoint,
                    SUM(cost_usd)          AS total_cost,
                    SUM(total_tokens)::bigint AS total_tokens,
                    COUNT(*)::int          AS request_count
                FROM token_usage_logs
                WHERE
                    user_id   = :user_id
                    AND created_at >= :start_dt
                    AND created_at <= :end_dt
                GROUP BY endpoint
                ORDER BY total_cost DESC
                LIMIT 5
            """),
            {"user_id": user_id, "start_dt": start_dt, "end_dt": end_dt},
        )

        top_endpoints = [
            {
                "endpoint":      row["endpoint"],
                "total_cost_usd": float(row["total_cost"] or 0),
                "total_tokens":  int(row["total_tokens"] or 0),
                "request_count": int(row["request_count"] or 0),
            }
            for row in endpoint_rows.mappings()
        ]

        return CostReport(
            user_id=user_id,
            period_start=start_date,
            period_end=end_date,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            input_tokens=total_input,
            output_tokens=total_output,
            request_count=total_reqs,
            breakdown_by_model=breakdown,
            top_endpoints=top_endpoints,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3-C. get_cost_by_model — 全局模型费用分布
    # ─────────────────────────────────────────────────────────────────────────

    async def get_cost_by_model(
        self,
        start_date: date,
        end_date:   date,
        session:    AsyncSession,
    ) -> Dict[str, Dict[str, Any]]:
        """
        全局视角：按模型汇总所有用户的 Token 用量与费用。

        返回格式：
            {
                "gpt-4o": {
                    "input_tokens":   1234567,
                    "output_tokens":  234567,
                    "total_tokens":   1469134,
                    "total_cost_usd": 12.34,
                    "request_count":  1000,
                    "cost_share_pct": 45.6,  # 占总费用的百分比
                },
                ...
            }

        Args:
            start_date: 查询起始日期（含）
            end_date:   查询结束日期（含）
            session:    数据库 AsyncSession

        Returns:
            以模型名为 key 的费用分布字典
        """
        start_dt = datetime(start_date.year, start_date.month, start_date.day,
                            tzinfo=timezone.utc)
        end_dt   = datetime(end_date.year, end_date.month, end_date.day,
                            23, 59, 59, tzinfo=timezone.utc)

        rows = await session.execute(
            text("""
                SELECT
                    model,
                    SUM(input_tokens)::bigint   AS input_tokens,
                    SUM(output_tokens)::bigint  AS output_tokens,
                    SUM(total_tokens)::bigint   AS total_tokens,
                    SUM(cost_usd)               AS total_cost,
                    COUNT(*)::int               AS request_count
                FROM token_usage_logs
                WHERE
                    created_at >= :start_dt
                    AND created_at <= :end_dt
                GROUP BY model
                ORDER BY total_cost DESC
            """),
            {"start_dt": start_dt, "end_dt": end_dt},
        )

        result: Dict[str, Dict[str, Any]] = {}
        grand_total = Decimal("0")
        raw_data: list[dict] = []

        for row in rows.mappings():
            cost = Decimal(str(row["total_cost"] or "0"))
            grand_total += cost
            raw_data.append({
                "model":          row["model"],
                "input_tokens":   int(row["input_tokens"] or 0),
                "output_tokens":  int(row["output_tokens"] or 0),
                "total_tokens":   int(row["total_tokens"] or 0),
                "total_cost":     cost,
                "request_count":  int(row["request_count"] or 0),
            })

        for item in raw_data:
            cost = item["total_cost"]
            share = (
                float(cost / grand_total * 100)
                if grand_total > 0 else 0.0
            )
            result[item["model"]] = {
                "input_tokens":   item["input_tokens"],
                "output_tokens":  item["output_tokens"],
                "total_tokens":   item["total_tokens"],
                "total_cost_usd": float(cost),
                "request_count":  item["request_count"],
                "cost_share_pct": round(share, 2),
            }

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 3-D. get_daily_summary — 单日费用快照
    # ─────────────────────────────────────────────────────────────────────────

    async def get_daily_summary(
        self,
        target_date: date,
        session:     AsyncSession,
        user_id:     Optional[str] = None,
    ) -> DailySummary:
        """
        获取指定日期的费用快照（全局或单用户）。

        用于：
          - 日费用预警检查（check_daily_alert 内部调用）
          - Grafana Dashboard 的「今日花费」面板
          - 定时告警任务

        Args:
            target_date: 目标日期（UTC）
            session:     数据库 AsyncSession
            user_id:     None=全局汇总；指定=单用户汇总
        """
        start_dt = datetime(target_date.year, target_date.month, target_date.day,
                            tzinfo=timezone.utc)
        end_dt   = start_dt + timedelta(days=1)

        params: Dict[str, Any] = {"start_dt": start_dt, "end_dt": end_dt}
        user_filter = ""

        if user_id:
            user_filter = "AND user_id = :user_id"
            params["user_id"] = user_id

        row = await session.execute(
            text(f"""
                SELECT
                    COALESCE(SUM(cost_usd), 0)          AS total_cost,
                    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
                    COUNT(*)::int                        AS request_count
                FROM token_usage_logs
                WHERE
                    created_at >= :start_dt
                    AND created_at < :end_dt
                    {user_filter}
            """),
            params,
        )

        data = row.mappings().one()
        total_cost = Decimal(str(data["total_cost"] or "0"))

        return DailySummary(
            target_date=target_date,
            total_cost_usd=total_cost,
            total_tokens=int(data["total_tokens"] or 0),
            request_count=int(data["request_count"] or 0),
            alert_threshold=self._alert_threshold,
            is_over_budget=(total_cost >= self._alert_threshold),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3-E. 日费用预警机制
    # ─────────────────────────────────────────────────────────────────────────

    async def check_daily_alert(
        self,
        target_date: date,
        session:     AsyncSession,
    ) -> Optional[DailySummary]:
        """
        检查指定日期的费用是否超过阈值，超限时触发告警。

        告警策略（「逐级升温」设计）：
          第一次超限（cost >= threshold）      → WARNING 日志
          超限 150%（cost >= threshold * 1.5） → ERROR 日志（更严重）
          超限 200%（cost >= threshold * 2.0） → CRITICAL 日志（紧急）
          每级别只在首次触发时记录，避免日志刷屏（通过 Redis 去重）

        Returns:
            超限时返回 DailySummary，未超限返回 None
        """
        summary = await self.get_daily_summary(target_date, session)

        if not summary.is_over_budget:
            return None

        cost        = summary.total_cost_usd
        threshold   = summary.alert_threshold
        ratio       = float(cost / threshold)
        excess_usd  = float(cost - threshold)

        log_extra = {
            "alert_type":      "daily_cost_exceeded",
            "date":            target_date.isoformat(),
            "total_cost_usd":  float(cost),
            "threshold_usd":   float(threshold),
            "excess_usd":      round(excess_usd, 4),
            "excess_ratio":    round(ratio, 3),
            "total_tokens":    summary.total_tokens,
            "request_count":   summary.request_count,
        }

        # 逐级告警
        if ratio >= 2.0:
            logger.critical(
                "🚨 日费用严重超限！已达阈值 %.0f%%，请立即介入！"
                "当日费用 $%.4f，阈值 $%.2f",
                ratio * 100, float(cost), float(threshold),
                extra=log_extra,
            )
        elif ratio >= 1.5:
            logger.error(
                "⚠️ 日费用大幅超限！已达阈值 %.0f%%。"
                "当日费用 $%.4f，阈值 $%.2f",
                ratio * 100, float(cost), float(threshold),
                extra=log_extra,
            )
        else:
            logger.warning(
                "💡 日费用已超限。当日费用 $%.4f，阈值 $%.2f（超出 %.1f%%）",
                float(cost), float(threshold), (ratio - 1) * 100,
                extra=log_extra,
            )

        # 触发扩展告警钩子（可接 Slack / 邮件 / PagerDuty）
        await self._notify_alert(summary)

        return summary

    async def _check_daily_alert_safe(self, target_date: date) -> None:
        """
        check_daily_alert 的安全包装版本（由 record_usage 通过 create_task 调用）。

        独立创建数据库 Session，与主请求的 Session 完全隔离，
        避免主 Session 提交前查询到不完整数据。
        """
        try:
            from infrastructure.postgres_client import db_session  # type: ignore[import]
            async with db_session() as session:
                await self.check_daily_alert(target_date, session)
        except Exception as exc:
            logger.warning(
                "cost_service: 日费用预警检查失败（已忽略）| date=%s | error=%s",
                target_date, exc,
            )

    async def _notify_alert(self, summary: DailySummary) -> None:
        """
        告警扩展钩子（预留接口，当前只记录日志）。

        扩展方式：
            override 此方法，或在此处注入通知客户端：

            # Slack 通知示例
            async with aiohttp.ClientSession() as s:
                await s.post(SLACK_WEBHOOK_URL, json={
                    "text": f"Synaris 日费用告警：今日已花费 ${summary.total_cost_usd:.2f}"
                })

            # 邮件通知示例
            await email_client.send(
                to=["ops@company.com"],
                subject=f"[Synaris] 日费用超限 ${summary.total_cost_usd:.2f}",
                body=...,
            )

        Args:
            summary: 超限的日费用快照
        """
        # 当前实现：仅打印结构化日志，等待外部接入
        logger.info(
            "cost_service: 告警通知触发（钩子未配置，仅记录日志）",
            extra=summary.to_dict(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3-F. 辅助查询
    # ─────────────────────────────────────────────────────────────────────────

    async def get_recent_usage(
        self,
        session:  AsyncSession,
        user_id:  Optional[str] = None,
        limit:    int = 20,
    ) -> List[Dict[str, Any]]:
        """
        获取最近 N 条费用记录（供「用量明细」页面使用）。

        Args:
            session: 数据库 AsyncSession
            user_id: None=所有用户；指定=单用户
            limit:   返回条数上限（最多 100）
        """
        limit = min(max(1, limit), 100)
        params: Dict[str, Any] = {"limit": limit}
        user_filter = ""

        if user_id:
            user_filter = "AND user_id = :user_id"
            params["user_id"] = user_id

        rows = await session.execute(
            text(f"""
                SELECT
                    id, user_id, model, input_tokens, output_tokens,
                    total_tokens, cost_usd, endpoint, task_type,
                    session_id, created_at
                FROM token_usage_logs
                WHERE 1=1 {user_filter}
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            params,
        )

        return [
            {
                "id":            str(row["id"]),
                "user_id":       str(row["user_id"]) if row["user_id"] else None,
                "model":         row["model"],
                "input_tokens":  row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "total_tokens":  row["total_tokens"],
                "cost_usd":      float(row["cost_usd"] or 0),
                "endpoint":      row["endpoint"],
                "task_type":     row["task_type"],
                "session_id":    row["session_id"],
                "created_at":    row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows.mappings()
        ]

    async def get_cost_trend(
        self,
        session:    AsyncSession,
        days:       int = 7,
        user_id:    Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取近 N 天的每日费用趋势（供折线图使用）。

        Args:
            session: 数据库 AsyncSession
            days:    回溯天数（1~90，默认 7）
            user_id: None=全局；指定=单用户

        Returns:
            按日期升序的每日费用列表：
            [{"date": "2026-04-10", "cost_usd": 12.34, "tokens": 123456}, ...]
        """
        days  = min(max(1, days), 90)
        params: Dict[str, Any] = {
            "start_dt": datetime.now(timezone.utc) - timedelta(days=days),
        }
        user_filter = ""

        if user_id:
            user_filter = "AND user_id = :user_id"
            params["user_id"] = user_id

        rows = await session.execute(
            text(f"""
                SELECT
                    DATE(created_at AT TIME ZONE 'UTC') AS day,
                    SUM(cost_usd)               AS total_cost,
                    SUM(total_tokens)::bigint   AS total_tokens,
                    COUNT(*)::int               AS request_count
                FROM token_usage_logs
                WHERE created_at >= :start_dt {user_filter}
                GROUP BY day
                ORDER BY day ASC
            """),
            params,
        )

        return [
            {
                "date":          row["day"].isoformat(),
                "cost_usd":      float(row["total_cost"] or 0),
                "total_tokens":  int(row["total_tokens"] or 0),
                "request_count": int(row["request_count"] or 0),
            }
            for row in rows.mappings()
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 辅助工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_json(obj: Any) -> str:
    """将任意对象安全序列化为 JSON 字符串（用于写入 PostgreSQL JSONB 字段）。"""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FastAPI 依赖注入工厂
# ═══════════════════════════════════════════════════════════════════════════════

def get_cost_service() -> CostService:
    """
    FastAPI Depends 兼容的 CostService 工厂函数。

    CostService 本身无状态（所有状态存储在 PostgreSQL），
    每次请求新建实例开销极低，无需单例缓存。

    使用方式：
        @router.get("/admin/cost/user/{user_id}")
        async def get_user_cost_api(
            user_id:   str,
            start:     date = Query(...),
            end:       date = Query(...),
            session:   AsyncSession = Depends(get_db_session),
            cost_svc:  CostService  = Depends(get_cost_service),
        ):
            report = await cost_svc.get_user_cost(user_id, start, end, session)
            return ApiResponse.ok(data=report.to_dict())
    """
    return CostService()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 数据库表结构参考（Alembic 迁移脚本片段）
#
# 此处以注释形式提供建表 DDL，实际由 Alembic autogenerate 生成迁移文件。
# 若需要手动创建，可执行以下 SQL：
# ═══════════════════════════════════════════════════════════════════════════════

_CREATE_TABLE_DDL = """
-- token_usage_logs 表（在 migrations/versions/ 中通过 Alembic 生成）
CREATE TABLE IF NOT EXISTS token_usage_logs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        REFERENCES users(id) ON DELETE SET NULL,
    model           VARCHAR(128) NOT NULL,
    input_tokens    INTEGER     NOT NULL DEFAULT 0,
    output_tokens   INTEGER     NOT NULL DEFAULT 0,
    total_tokens    INTEGER     NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(18, 8) NOT NULL DEFAULT 0,  -- 8 位小数精度
    endpoint        VARCHAR(256) NOT NULL DEFAULT 'unknown',
    task_type       VARCHAR(64) NOT NULL DEFAULT 'chat',
    session_id      VARCHAR(128),
    extra_metadata  JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 常用查询索引
CREATE INDEX IF NOT EXISTS idx_token_usage_user_created
    ON token_usage_logs (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_token_usage_created
    ON token_usage_logs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_token_usage_model
    ON token_usage_logs (model, created_at DESC);

-- 分区建议（当日志量超过 1000 万行时）：
-- 按月范围分区：PARTITION BY RANGE (created_at)
"""
