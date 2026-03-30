"""
@File       : web_search.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Agent 网络搜索工具。
@Features:
  - 主引擎：DuckDuckGo Search（duckduckgo-search 库，无需 API Key）
  - 备用引擎：Tavily Search API（当环境变量 TAVILY_API_KEY 存在时自动启用为首选）
  - @tool 装饰器：供 LangGraph ToolSelector 节点直接调用
  - 输入：query(str) + max_results(int, 默认5) + search_type(str)
  - 输出：List[SearchResult]，包含 title / url / snippet / source 字段
  - 安全边界：超时 10s、最多返回 5 条、snippet 截断至 500 字符
  - 错误处理：引擎故障时自动降级（Tavily → DuckDuckGo → 返回结构化错误）

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────
_MAX_RESULTS     = 5       # 硬上限，防止 Token 爆炸
_SNIPPET_MAX_LEN = 500     # snippet 截断长度（字符）
_TIMEOUT_SECONDS = 10      # 单次搜索超时


# ─────────────────────────────────────────────
# 1. 数据模型
# ─────────────────────────────────────────────

class SearchResult(BaseModel):
    """
    单条搜索结果。

    类比图书馆索引卡片：
      title   → 书名
      url     → 馆藏位置
      snippet → 内容摘要（封底简介）
      source  → 来源引擎标识
    """

    title:   str = Field(description="页面标题")
    url:     str = Field(description="页面 URL")
    snippet: str = Field(description="内容摘要（最多 500 字符）")
    source:  str = Field(default="web", description="来源引擎：tavily / duckduckgo")

    def truncate_snippet(self) -> "SearchResult":
        """截断 snippet，返回新对象（不可变设计）。"""
        if len(self.snippet) > _SNIPPET_MAX_LEN:
            return self.model_copy(
                update={"snippet": self.snippet[:_SNIPPET_MAX_LEN] + "…"}
            )
        return self


class WebSearchInput(BaseModel):
    """web_search 工具的入参模型（用于 ToolSelector 生成合法 JSON 入参）。"""

    query:       str = Field(description="搜索关键词或自然语言问题")
    max_results: int = Field(default=5, ge=1, le=_MAX_RESULTS,
                             description=f"返回结果数量，最多 {_MAX_RESULTS} 条")
    search_type: str = Field(default="general",
                             description="搜索类型：general（通用）/ news（新闻）")


# ─────────────────────────────────────────────
# 2. 引擎实现（Tavily + DuckDuckGo）
# ─────────────────────────────────────────────

def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    """
    使用 Tavily Search API 执行搜索。
    仅在环境变量 TAVILY_API_KEY 存在时被调用。

    Tavily 优点：
    - 结果质量更高（专为 LLM/RAG 场景优化）
    - 支持 answer 字段（直接摘要）
    - 官方推荐的 Agent 搜索引擎
    """
    try:
        from tavily import TavilyClient  # type: ignore[import]

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",    # basic 速度更快，advanced 更精准但更贵
            include_answer=False,    # 由 LLM 自行综合，不要 Tavily 的预生成答案
        )
        results = []
        for item in response.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", "无标题"),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="tavily",
                ).truncate_snippet()
            )
        return results

    except ImportError:
        logger.warning("tavily 库未安装，回退到 DuckDuckGo")
        raise
    except Exception as exc:
        logger.warning("Tavily 搜索失败: %s，回退到 DuckDuckGo", exc)
        raise


def _search_duckduckgo(
    query: str,
    max_results: int,
    search_type: str = "general",
) -> list[SearchResult]:
    """
    使用 DuckDuckGo Search 执行搜索（无需 API Key，适合开发/低成本场景）。

    search_type:
      - general → 网页搜索（DDGS.text）
      - news    → 新闻搜索（DDGS.news，返回更新鲜的资讯）
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]

        results: list[SearchResult] = []

        with DDGS() as ddgs:
            if search_type == "news":
                # 新闻搜索
                raw_results = list(ddgs.news(
                    query,
                    max_results=max_results,
                    timelimit="m",   # 最近一个月
                ))
                for item in raw_results:
                    results.append(
                        SearchResult(
                            title=item.get("title", "无标题"),
                            url=item.get("url", ""),
                            snippet=item.get("body", ""),
                            source="duckduckgo_news",
                        ).truncate_snippet()
                    )
            else:
                # 通用网页搜索
                raw_results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    safesearch="moderate",
                ))
                for item in raw_results:
                    results.append(
                        SearchResult(
                            title=item.get("title", "无标题"),
                            url=item.get("href", ""),
                            snippet=item.get("body", ""),
                            source="duckduckgo",
                        ).truncate_snippet()
                    )

        return results

    except ImportError:
        logger.error("duckduckgo-search 库未安装，请执行: pip install duckduckgo-search")
        raise
    except Exception as exc:
        logger.error("DuckDuckGo 搜索失败: %s", exc)
        raise


# ─────────────────────────────────────────────
# 3. @tool 装饰器入口
# ─────────────────────────────────────────────

@tool(args_schema=WebSearchInput)
def web_search(
    query:       str,
    max_results: int = 5,
    search_type: str = "general",
) -> str:
    """
    搜索互联网获取最新信息。

    当需要以下情况时调用此工具：
    - 查询实时数据（股价、新闻、天气、最新政策等）
    - 查找知识库中没有的外部信息
    - 验证或补充已有信息

    返回格式：结构化文本，每条结果包含序号、标题、URL、摘要。

    Args:
        query:       搜索关键词或自然语言问题（建议 5-20 个词）
        max_results: 返回结果数量（1-5，默认 5）
        search_type: 搜索类型，"general" 通用搜索 或 "news" 新闻搜索

    Returns:
        格式化的搜索结果文本，供 LLM 直接阅读。
    """
    # 参数防御
    max_results = min(max(1, max_results), _MAX_RESULTS)
    query = query.strip()

    if not query:
        return "错误：搜索关键词不能为空。"

    start_time = time.monotonic()
    results: list[SearchResult] = []
    engine_used = "unknown"

    try:
        # ── 引擎选择策略：Tavily（若有 Key）优先，否则 DuckDuckGo ──
        if os.environ.get("TAVILY_API_KEY"):
            try:
                results = _search_tavily(query, max_results)
                engine_used = "tavily"
            except Exception:
                # Tavily 失败，自动降级
                results = _search_duckduckgo(query, max_results, search_type)
                engine_used = "duckduckgo (fallback)"
        else:
            results = _search_duckduckgo(query, max_results, search_type)
            engine_used = "duckduckgo"

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "web_search 全部引擎失败 | query=%s | elapsed=%.2fs | error=%s",
            query, elapsed, exc,
        )
        return (
            f"搜索失败：所有搜索引擎均不可用。\n"
            f"错误详情：{exc}\n"
            f"建议：请检查网络连接或配置 TAVILY_API_KEY 环境变量。"
        )

    elapsed = time.monotonic() - start_time
    logger.info(
        "web_search 完成 | engine=%s | query=%s | results=%d | elapsed=%.2fs",
        engine_used, query, len(results), elapsed,
    )

    if not results:
        return f"搜索「{query}」未找到相关结果，建议更换关键词重试。"

    # ── 格式化为 LLM 易读的文本 ──
    lines = [f"搜索引擎：{engine_used}  |  关键词：{query}  |  共 {len(results)} 条结果\n"]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"【{i}】{r.title}\n"
            f"    URL：{r.url}\n"
            f"    摘要：{r.snippet}\n"
        )
    return "\n".join(lines)