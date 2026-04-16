"""
@File       : eval_service.py
@Author     : Codex
@Created    : 2026-04-16
@Version    : 1.0.0
@Description: Prompt / RAG / Agent 输出评估服务。
@Features:
  - LLM-as-judge：使用 gpt-4o-mini 输出结构化 JSON 评分
  - 评估指标：relevance / faithfulness / quality
  - ground_truth 可选时增加词面重叠指标（token F1 + ROUGE-L）
  - run_regression_eval() 批量回归评估并持久化到 PostgreSQL
  - get_history() 提供历史趋势查询入口
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from app.core.exceptions import ErrorCode, LLMError
    from app.core.logging import get_logger
    from app.infrastructure.llm_client import invoke as llm_invoke
    from app.infrastructure.postgres_client import db_session
    from app.models.eval_run import EvaluationRun
    from app.models.prompt_version import PromptVersion
except ImportError:  # pragma: no cover - 兼容仓库中混用的导入风格
    from core.exceptions import ErrorCode, LLMError  # type: ignore[no-redef]
    from core.logging import get_logger  # type: ignore[no-redef]
    from infrastructure.llm_client import invoke as llm_invoke  # type: ignore[no-redef]
    from infrastructure.postgres_client import db_session  # type: ignore[no-redef]
    from models.eval_run import EvaluationRun  # type: ignore[no-redef]
    from models.prompt_version import PromptVersion  # type: ignore[no-redef]

logger = get_logger(__name__)

_DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
_DEFAULT_PASS_THRESHOLD = 0.70
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

_EVAL_JUDGE_SYSTEM_PROMPT = """你是企业级 LLM 评估裁判。

你将收到 question、answer、context、ground_truth。
请严格输出 JSON，不要输出 Markdown、解释段落或代码块。

评分标准：
1. relevance（0-1）：
   - 回答是否直接回应问题
   - 是否跑题、漏答、答非所问
2. faithfulness（0-1）：
   - 回答是否忠实于给定 context
   - 是否包含 context 中没有明确支持的内容
   - 若未提供 context，请返回 1.0，并在 reasoning 中说明 "context_missing"
3. quality（0-1）：
   - 综合考虑准确性、完整性、清晰度、可执行性
   - 若提供 ground_truth，可参考它判断准确性，但不要机械字面匹配

返回格式：
{
  "relevance": 0.0,
  "faithfulness": 0.0,
  "quality": 0.0,
  "confidence": 0.0,
  "reasoning": "一句到两句简短说明",
  "dimension_scores": {
    "accuracy": 0.0,
    "completeness": 0.0,
    "clarity": 0.0,
    "usefulness": 0.0
  }
}
"""


@dataclass(slots=True)
class EvalScore:
    """单条答案的综合评分结果。"""

    relevance: Optional[float]
    faithfulness: Optional[float]
    quality: float
    judge_quality: float
    confidence: float
    reasoning: str
    latency_ms: float
    lexical_f1: Optional[float] = None
    rouge_l: Optional[float] = None
    dimension_scores: dict[str, float] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        parts = [self.quality]
        if self.relevance is not None:
            parts.append(self.relevance)
        if self.faithfulness is not None:
            parts.append(self.faithfulness)
        return round(sum(parts) / len(parts), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance,
            "faithfulness": self.faithfulness,
            "quality": self.quality,
            "judge_quality": self.judge_quality,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "latency_ms": round(self.latency_ms, 2),
            "lexical_f1": self.lexical_f1,
            "rouge_l": self.rouge_l,
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
        }


@dataclass(slots=True)
class EvalCase:
    """回归评估中的单条样本。"""

    question: str
    answer: str
    context: Optional[str] = None
    ground_truth: Optional[str] = None
    case_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalCaseResult:
    """单条样本的评估结果。"""

    case_name: str
    passed: bool
    latency_ms: float
    score: Optional[EvalScore] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "score": self.score.to_dict() if self.score else None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EvalReport:
    """一次批量回归评估的汇总报告。"""

    run_id: Optional[str]
    judge_model: str
    dataset_size: int
    accuracy: float
    average_relevance: float
    average_faithfulness: float
    average_quality: float
    average_hallucination_rate: float
    p95_latency_ms: float
    started_at: str
    finished_at: str
    results: list[EvalCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "judge_model": self.judge_model,
            "dataset_size": self.dataset_size,
            "accuracy": self.accuracy,
            "average_relevance": self.average_relevance,
            "average_faithfulness": self.average_faithfulness,
            "average_quality": self.average_quality,
            "average_hallucination_rate": self.average_hallucination_rate,
            "p95_latency_ms": self.p95_latency_ms,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [item.to_dict() for item in self.results],
        }


def _clamp_score(value: Any) -> float:
    """将任意输入规范化为 [0, 1] 区间分数。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, score)), 4)


def _tokenize(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _token_f1(prediction: Optional[str], target: Optional[str]) -> float:
    pred_tokens = _tokenize(prediction)
    target_tokens = _tokenize(target)
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    overlap = sum(
        min(pred_counts.get(token, 0), target_counts.get(token, 0))
        for token in pred_counts
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for index, token_b in enumerate(b, start=1):
            current = dp[index]
            if token_a == token_b:
                dp[index] = prev + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            prev = current
    return dp[-1]


def _rouge_l_f1(prediction: Optional[str], target: Optional[str]) -> float:
    pred_tokens = _tokenize(prediction)
    target_tokens = _tokenize(target)
    if not pred_tokens or not target_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, target_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(target_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """从 LLM 输出中抽取 JSON。兼容偶发的 fenced block。"""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()

    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise LLMError(
            message="评估模型未返回可解析的 JSON",
            error_code=ErrorCode.LLM_INVALID_RESPONSE,
            detail={"raw_text": raw_text[:500]},
        )

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(
            message="评估模型返回的 JSON 格式非法",
            error_code=ErrorCode.LLM_INVALID_RESPONSE,
            detail={"raw_text": raw_text[:500]},
        ) from exc


def _compute_p95(latencies_ms: list[float]) -> float:
    if not latencies_ms:
        return 0.0
    ordered = sorted(latencies_ms)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return round(float(ordered[index]), 2)


class EvalService:
    """LLM 输出评估服务。"""

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        *,
        judge_model: str = _DEFAULT_JUDGE_MODEL,
        pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
    ) -> None:
        self._session = session
        self._judge_model = judge_model
        self._pass_threshold = pass_threshold

    @asynccontextmanager
    async def _session_scope(self) -> AsyncIterator[AsyncSession]:
        if self._session is not None:
            yield self._session
            return

        async with db_session() as session:
            yield session

    async def _judge_case(self, case: EvalCase) -> EvalScore:
        payload = {
            "question": case.question,
            "answer": case.answer,
            "context": case.context or "",
            "ground_truth": case.ground_truth or "",
            "metadata": case.metadata,
        }
        messages = [
            {"role": "system", "content": _EVAL_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请评估以下回答，并严格按 JSON 返回：\n"
                    f"{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]

        started = time.perf_counter()
        raw_response = await llm_invoke(
            messages,
            model=self._judge_model,
            temperature=0.0,
            max_tokens=900,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        judgement = _extract_json_payload(raw_response)

        judge_quality = _clamp_score(judgement.get("quality"))
        relevance = _clamp_score(judgement.get("relevance"))
        faithfulness = _clamp_score(judgement.get("faithfulness"))
        confidence = _clamp_score(judgement.get("confidence"))
        reasoning = str(judgement.get("reasoning", "")).strip()

        lexical_f1: Optional[float] = None
        rouge_l: Optional[float] = None
        final_quality = judge_quality

        if case.ground_truth:
            lexical_f1 = _token_f1(case.answer, case.ground_truth)
            rouge_l = _rouge_l_f1(case.answer, case.ground_truth)
            lexical_blend = round((lexical_f1 + rouge_l) / 2, 4)
            final_quality = round(judge_quality * 0.7 + lexical_blend * 0.3, 4)

        dimension_scores_raw = judgement.get("dimension_scores") or {}
        dimension_scores = {
            key: _clamp_score(value)
            for key, value in dimension_scores_raw.items()
        }

        return EvalScore(
            relevance=relevance,
            faithfulness=faithfulness if case.context else None,
            quality=final_quality,
            judge_quality=judge_quality,
            confidence=confidence,
            reasoning=reasoning,
            latency_ms=latency_ms,
            lexical_f1=lexical_f1,
            rouge_l=rouge_l,
            dimension_scores=dimension_scores,
        )

    async def eval_relevance(self, question: str, answer: str) -> float:
        score = await self._judge_case(EvalCase(question=question, answer=answer))
        return score.relevance or 0.0

    async def eval_faithfulness(self, context: str, answer: str) -> float:
        score = await self._judge_case(
            EvalCase(question="N/A", answer=answer, context=context)
        )
        return score.faithfulness or 0.0

    async def eval_answer_quality(
        self,
        question: str,
        answer: str,
        ground_truth: Optional[str] = None,
    ) -> EvalScore:
        return await self._judge_case(
            EvalCase(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
        )

    async def _persist_report(
        self,
        report: EvalReport,
        *,
        run_name: str,
        eval_type: str,
        prompt_name: Optional[str],
        prompt_version_id: Optional[Any],
        metadata: Optional[dict[str, Any]],
    ) -> str:
        async with self._session_scope() as session:
            eval_run = EvaluationRun(
                name=run_name,
                eval_type=eval_type,
                prompt_name=prompt_name,
                prompt_version_id=prompt_version_id,
                judge_model=report.judge_model,
                dataset_size=report.dataset_size,
                completed_cases=sum(
                    1 for item in report.results if item.score is not None
                ),
                failed_cases=sum(1 for item in report.results if item.score is None),
                accuracy=report.accuracy,
                average_relevance=report.average_relevance,
                average_faithfulness=report.average_faithfulness,
                average_quality=report.average_quality,
                average_hallucination_rate=report.average_hallucination_rate,
                p95_latency_ms=report.p95_latency_ms,
                results=[item.to_dict() for item in report.results],
                summary=report.to_dict(),
                extra_payload=metadata or {},
            )
            session.add(eval_run)
            await session.flush()

            target_prompt: Optional[PromptVersion] = None
            if prompt_version_id is not None:
                target_prompt = await session.get(PromptVersion, prompt_version_id)
            elif prompt_name:
                prompt_stmt = (
                    select(PromptVersion)
                    .where(
                        PromptVersion.name == prompt_name,
                        PromptVersion.is_active.is_(True),
                        PromptVersion.is_deleted.is_(False),
                    )
                    .order_by(
                        desc(PromptVersion.ab_test_weight),
                        desc(PromptVersion.created_at),
                    )
                )
                prompt_result = await session.execute(prompt_stmt)
                target_prompt = prompt_result.scalars().first()

            if target_prompt is not None:
                target_prompt.eval_relevance_score = report.average_relevance
                target_prompt.eval_faithfulness_score = report.average_faithfulness

            await session.flush()
            return str(eval_run.id)

    async def run_regression_eval(
        self,
        test_dataset: list[EvalCase],
        *,
        run_name: str = "regression_eval",
        eval_type: str = "regression",
        prompt_name: Optional[str] = None,
        prompt_version_id: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EvalReport:
        """批量跑评估集，并将结果落库。"""
        started_at = datetime.now(timezone.utc)
        results: list[EvalCaseResult] = []
        latencies_ms: list[float] = []
        relevance_scores: list[float] = []
        faithfulness_scores: list[float] = []
        quality_scores: list[float] = []
        passed_count = 0

        for index, case in enumerate(test_dataset, start=1):
            case_name = case.case_name or f"case-{index}"
            case_started = time.perf_counter()
            try:
                score = await self._judge_case(case)
                latencies_ms.append(score.latency_ms)
                if score.relevance is not None:
                    relevance_scores.append(score.relevance)
                if score.faithfulness is not None:
                    faithfulness_scores.append(score.faithfulness)
                quality_scores.append(score.quality)

                passed = (
                    score.overall_score >= self._pass_threshold
                    and (score.faithfulness is None or score.faithfulness >= 0.60)
                )
                if passed:
                    passed_count += 1

                results.append(
                    EvalCaseResult(
                        case_name=case_name,
                        passed=passed,
                        latency_ms=score.latency_ms,
                        score=score,
                        metadata=case.metadata,
                    )
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - case_started) * 1000
                results.append(
                    EvalCaseResult(
                        case_name=case_name,
                        passed=False,
                        latency_ms=latency_ms,
                        error=str(exc),
                        metadata=case.metadata,
                    )
                )
                logger.warning(
                    "评估样本失败，已记录并继续后续样本",
                    extra={"case_name": case_name, "error": str(exc)},
                )

        dataset_size = len(test_dataset)
        finished_at = datetime.now(timezone.utc)
        average_relevance = (
            round(statistics.mean(relevance_scores), 4) if relevance_scores else 0.0
        )
        average_faithfulness = (
            round(statistics.mean(faithfulness_scores), 4)
            if faithfulness_scores
            else 0.0
        )
        average_quality = (
            round(statistics.mean(quality_scores), 4) if quality_scores else 0.0
        )
        accuracy = round(passed_count / dataset_size, 4) if dataset_size else 0.0
        hallucination_rate = (
            round(1 - average_faithfulness, 4) if faithfulness_scores else 0.0
        )

        report = EvalReport(
            run_id=None,
            judge_model=self._judge_model,
            dataset_size=dataset_size,
            accuracy=accuracy,
            average_relevance=average_relevance,
            average_faithfulness=average_faithfulness,
            average_quality=average_quality,
            average_hallucination_rate=hallucination_rate,
            p95_latency_ms=_compute_p95(latencies_ms),
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            results=results,
        )

        run_id = await self._persist_report(
            report,
            run_name=run_name,
            eval_type=eval_type,
            prompt_name=prompt_name,
            prompt_version_id=prompt_version_id,
            metadata=metadata,
        )
        report.run_id = run_id

        logger.info(
            "回归评估完成",
            extra={
                "run_id": run_id,
                "dataset_size": dataset_size,
                "accuracy": accuracy,
                "average_quality": average_quality,
                "p95_latency_ms": report.p95_latency_ms,
            },
        )
        return report

    async def get_history(
        self,
        *,
        prompt_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """读取历史评估记录，供趋势图或后台管理页使用。"""
        async with self._session_scope() as session:
            stmt = (
                select(EvaluationRun)
                .where(EvaluationRun.is_deleted.is_(False))
                .order_by(desc(EvaluationRun.created_at))
                .limit(max(1, limit))
            )
            if prompt_name:
                stmt = stmt.where(EvaluationRun.prompt_name == prompt_name)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "run_id": str(row.id),
                    "name": row.name,
                    "prompt_name": row.prompt_name,
                    "judge_model": row.judge_model,
                    "dataset_size": row.dataset_size,
                    "accuracy": row.accuracy,
                    "average_relevance": row.average_relevance,
                    "average_faithfulness": row.average_faithfulness,
                    "average_quality": row.average_quality,
                    "average_hallucination_rate": row.average_hallucination_rate,
                    "p95_latency_ms": row.p95_latency_ms,
                    "created_at": row.created_at.isoformat()
                    if row.created_at
                    else None,
                }
                for row in rows
            ]


__all__ = [
    "EvalCase",
    "EvalCaseResult",
    "EvalReport",
    "EvalScore",
    "EvalService",
]
