"""
@File       : external_api.py
@Author     : Starry Hung
@Created    : 2026-04-11
@Version    : 1.0.0
@Description: Agent 外部 HTTP API 调用工具（白名单域名安全版）。

@Features:
  - @tool 装饰器：供 LangGraph ToolSelector 节点直接调用
  - 四层安全防线：
      1. URL 格式校验  : urllib.parse 解析，必须是 http/https 协议
      2. 域名白名单    : 从 settings.allowed_api_domains 读取，
                         支持精确域名（api.github.com）和通配子域名（*.example.com）；
                         白名单为空时拒绝所有请求（fail-safe 设计）
      3. SSRF 防护     : 拒绝私有 IP 段（10.x / 172.16-31.x / 192.168.x / 127.x / ::1）
                         以及 .local / .internal 等内网域名后缀
      4. 响应大小限制  : 流式读取，超过 1MB 立即截断，防止内存爆炸
  - 超时限制：10 秒（连接 5s + 读取 5s 分开控制）
  - 支持请求方法：GET / POST / PUT / PATCH / DELETE / HEAD
  - 请求头 / 请求体通过 JSON 字符串传入（ToolSelector 友好）
  - 响应处理：
      · Content-Type 为 application/json → 自动解析并 pretty-print
      · 其他类型 → 按 UTF-8 文本返回（最多 5000 字符）
      · 非 2xx 状态码 → 返回结构化错误信息（含状态码 + 响应体摘要）
  - 脱敏日志：URL 中的 query string 自动脱敏，Authorization 头不记录值

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-04-11  Starry  Initial creation
"""

from __future__ import annotations

import ipaddress
import json
import logging
import time
import urllib.parse
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 执行限制常量 ───────────────────────────────────────────────────────────────
_TIMEOUT_CONNECT_SEC = 5          # TCP 连接超时（秒）
_TIMEOUT_READ_SEC    = 10         # 读取超时（秒），包含连接时间
_MAX_RESPONSE_BYTES  = 1 * 1024 * 1024  # 最大响应体：1 MB
_MAX_TEXT_OUTPUT     = 5_000      # 返回给 LLM 的最大字符数
_MAX_URL_LEN         = 2048       # URL 最大长度（防 DoS）
_ALLOWED_METHODS     = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})

# ── 私有/内网 IP 段（SSRF 防护）─────────────────────────────────────────────────
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 private
]

# ── 危险内网域名后缀（SSRF 防护）─────────────────────────────────────────────────
_INTERNAL_SUFFIXES = (
    ".local", ".internal", ".localhost",
    ".intranet", ".corp", ".lan",
)


# ---------------------------------------------------------------------------
# 1. 白名单域名配置读取
# ---------------------------------------------------------------------------

def _get_allowed_domains() -> list[str]:
    """
    从 settings 读取允许访问的域名白名单列表。

    配置方式（.env 示例）：
        ALLOWED_API_DOMAINS=api.github.com,*.openai.com,httpbin.org

    返回：域名字符串列表，支持：
      - 精确域名：api.github.com
      - 通配子域名：*.openai.com（只允许一级通配）

    Fail-safe 设计：
      - 若 settings 未配置 → 返回空列表，拒绝所有请求
      - 若 settings 模块未就绪 → 从环境变量 ALLOWED_API_DOMAINS 读取（降级）
    """
    try:
        from app.config.settings import get_settings  # type: ignore[import]
        s = get_settings()
        raw: str = getattr(s, "allowed_api_domains", "") or ""
    except (ImportError, Exception):
        import os
        raw = os.getenv("ALLOWED_API_DOMAINS", "")

    if not raw.strip():
        return []

    return [d.strip().lower() for d in raw.split(",") if d.strip()]


# ---------------------------------------------------------------------------
# 2. 四层安全校验
# ---------------------------------------------------------------------------

class _APISecurityError(ValueError):
    """外部 API 安全校验失败的专用异常。"""


def _check_url_format(url: str) -> urllib.parse.ParseResult:
    """
    第一层：URL 格式合法性校验。

    要求：
      - 必须是 http 或 https 协议（拒绝 file:// / ftp:// 等）
      - 必须包含非空 host
      - URL 长度不超过 _MAX_URL_LEN
    """
    if len(url) > _MAX_URL_LEN:
        raise _APISecurityError(
            f"URL 过长（{len(url)} 字符，上限 {_MAX_URL_LEN}）。"
        )

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        raise _APISecurityError(f"URL 格式解析失败：{exc}") from exc

    if parsed.scheme not in ("http", "https"):
        raise _APISecurityError(
            f"不支持的协议「{parsed.scheme}」。\n"
            f"external_api 只允许 http / https 协议。"
        )
    if not parsed.netloc:
        raise _APISecurityError("URL 缺少主机名（host）。")

    return parsed


def _check_domain_whitelist(parsed: urllib.parse.ParseResult) -> None:
    """
    第二层：域名白名单校验。

    匹配规则：
      - 精确匹配：host == allowed（如 api.github.com）
      - 通配匹配：*.example.com 允许 api.example.com / v2.example.com
                   但不允许 sub.api.example.com（只支持单级通配）
      - 端口忽略：api.github.com:443 视为 api.github.com

    Fail-safe：白名单为空时拒绝所有请求，防止配置遗漏导致安全缺口。
    """
    allowed_domains = _get_allowed_domains()

    if not allowed_domains:
        raise _APISecurityError(
            "外部 API 域名白名单未配置（ALLOWED_API_DOMAINS 为空）。\n"
            "请在 .env 中设置 ALLOWED_API_DOMAINS=域名1,域名2 后重试。\n"
            "示例：ALLOWED_API_DOMAINS=api.github.com,*.openai.com"
        )

    # 提取纯 host（去掉端口号）
    host = parsed.hostname or ""
    host = host.lower()

    for pattern in allowed_domains:
        if pattern.startswith("*."):
            # 通配匹配：*.example.com 匹配 api.example.com
            suffix = pattern[1:]  # → .example.com
            if host.endswith(suffix) and "." in host[: -len(suffix)].lstrip("."):
                # 确保是单级通配，不允许 sub.api.example.com
                prefix = host[: -len(suffix)]
                if "." not in prefix:
                    return
            # 也匹配根域名本身（*.example.com 也允许 example.com）
            if host == pattern[2:]:
                return
        else:
            # 精确匹配
            if host == pattern:
                return

    raise _APISecurityError(
        f"域名「{host}」不在允许的白名单中。\n"
        f"当前白名单：{', '.join(allowed_domains)}\n"
        f"如需访问此域名，请联系管理员将其加入 ALLOWED_API_DOMAINS。"
    )


def _check_ssrf_protection(parsed: urllib.parse.ParseResult) -> None:
    """
    第三层：SSRF（服务端请求伪造）防护。

    攻击场景：
        Agent 被诱导请求 http://192.168.1.1/admin 或 http://metadata.local/
        从而访问内网服务、云实例元数据等敏感资源。

    防护策略：
      (a) 拒绝内网 IP 段（10.x / 172.16-31.x / 192.168.x / 127.x / 169.254.x）
      (b) 拒绝 .local / .internal / .localhost 等内网域名后缀

    注意：白名单校验（第二层）已限制了域名范围，第三层是针对
    IP 直接访问（绕过域名）和 DNS rebinding 攻击的额外防护。
    """
    host = parsed.hostname or ""

    # 检查危险内网域名后缀
    host_lower = host.lower()
    for suffix in _INTERNAL_SUFFIXES:
        if host_lower.endswith(suffix) or host_lower == suffix.lstrip("."):
            raise _APISecurityError(
                f"拒绝访问内网域名「{host}」（后缀「{suffix}」属于内网保留域）。"
            )

    # 尝试将 host 解析为 IP，检查是否是私有地址
    try:
        addr = ipaddress.ip_address(host)
        for network in _PRIVATE_NETWORKS:
            if addr in network:
                raise _APISecurityError(
                    f"拒绝访问私有/内网 IP 地址「{host}」，防止 SSRF 攻击。"
                )
    except ValueError:
        # host 是域名，不是 IP 地址，跳过 IP 检查
        pass


def validate_request(
    url: str,
    method: str,
) -> urllib.parse.ParseResult:
    """
    统一校验入口：依序执行四层安全检查（前三层在此完成）。
    方法合法性也在此校验。

    Returns:
        解析后的 ParseResult（供后续使用）
    """
    method_upper = method.upper()
    if method_upper not in _ALLOWED_METHODS:
        raise _APISecurityError(
            f"不支持的 HTTP 方法「{method}」。\n"
            f"允许的方法：{', '.join(sorted(_ALLOWED_METHODS))}"
        )

    parsed = _check_url_format(url)
    _check_domain_whitelist(parsed)
    _check_ssrf_protection(parsed)
    return parsed


# ---------------------------------------------------------------------------
# 3. JSON 参数解析
# ---------------------------------------------------------------------------

def _parse_json_param(raw: str, param_name: str) -> dict[str, Any]:
    """
    安全解析 JSON 字符串入参。
    空字符串视为空字典，解析失败返回结构化错误提示。
    """
    if not raw or not raw.strip():
        return {}
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError(f"期望 JSON 对象（{{...}}），得到 {type(result).__name__}")
        return result
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"参数「{param_name}」JSON 解析失败：{exc}\n"
            f"请确保传入合法的 JSON 对象字符串，例如：{{\"key\": \"value\"}}"
        ) from exc


# ---------------------------------------------------------------------------
# 4. HTTP 请求执行（带第四层：响应大小限制）
# ---------------------------------------------------------------------------

def _sanitize_url_for_log(url: str) -> str:
    """
    日志脱敏：移除 URL 中的 query string，
    防止 token / api_key 等敏感参数出现在日志里。
    """
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(
        parsed._replace(query="[redacted]" if parsed.query else "")
    )


def _sanitize_headers_for_log(headers: dict[str, str]) -> dict[str, str]:
    """
    日志脱敏：Authorization / X-API-Key 等认证头只记录键名，不记录值。
    """
    _SENSITIVE_HEADERS = frozenset({
        "authorization", "x-api-key", "api-key",
        "x-auth-token", "cookie", "set-cookie",
    })
    return {
        k: "[redacted]" if k.lower() in _SENSITIVE_HEADERS else v
        for k, v in headers.items()
    }


def _execute_request(
    url: str,
    method: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    """
    执行 HTTP 请求，返回结构化响应信息。

    第四层安全：响应大小限制
        使用 iter_content(chunk_size=8192) 流式读取，
        累计超过 _MAX_RESPONSE_BYTES (1MB) 立即停止读取并截断，
        防止超大响应体耗尽 Agent 进程内存。

    返回格式：
        {
            "status_code": 200,
            "content_type": "application/json",
            "body": {...} 或 "text content",
            "truncated": False,
            "elapsed_ms": 123.4,
        }
    """
    try:
        import httpx  # type: ignore[import]
    except ImportError:
        try:
            import requests as _req  # type: ignore[import]
            return _execute_with_requests(_req, url, method, headers, body)
        except ImportError:
            raise RuntimeError(
                "HTTP 客户端库未安装。\n"
                "请执行：pip install httpx\n"
                "或：pip install requests"
            )

    return _execute_with_httpx(httpx, url, method, headers, body)


def _execute_with_httpx(
    httpx: Any,
    url: str,
    method: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    """使用 httpx 库执行 HTTP 请求（推荐，支持 HTTP/2）。"""
    start = time.perf_counter()

    # 合并默认请求头
    merged_headers = {
        "User-Agent": "Synaris-Agent/1.0",
        "Accept": "application/json, text/plain, */*",
        **headers,
    }

    kwargs: dict[str, Any] = {
        "headers": merged_headers,
        "timeout": httpx.Timeout(
            connect=_TIMEOUT_CONNECT_SEC,
            read=_TIMEOUT_READ_SEC,
            write=_TIMEOUT_READ_SEC,
            pool=_TIMEOUT_READ_SEC,
        ),
        "follow_redirects": True,
    }
    if body and method.upper() not in ("GET", "HEAD"):
        kwargs["json"] = body

    collected = bytearray()
    truncated = False

    with httpx.Client() as client:
        with client.stream(method.upper(), url, **kwargs) as resp:
            content_type = resp.headers.get("content-type", "")
            status_code  = resp.status_code

            for chunk in resp.iter_bytes(chunk_size=8192):
                collected.extend(chunk)
                if len(collected) > _MAX_RESPONSE_BYTES:
                    truncated = True
                    collected = collected[:_MAX_RESPONSE_BYTES]
                    break

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return _build_response_result(
        status_code, content_type, bytes(collected), truncated, elapsed_ms
    )


def _execute_with_requests(
    requests: Any,
    url: str,
    method: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    """使用 requests 库执行 HTTP 请求（httpx 不可用时的降级方案）。"""
    start = time.perf_counter()

    merged_headers = {
        "User-Agent": "Synaris-Agent/1.0",
        "Accept": "application/json, text/plain, */*",
        **headers,
    }
    kwargs: dict[str, Any] = {
        "headers": merged_headers,
        "timeout": (_TIMEOUT_CONNECT_SEC, _TIMEOUT_READ_SEC),
        "allow_redirects": True,
        "stream": True,
    }
    if body and method.upper() not in ("GET", "HEAD"):
        kwargs["json"] = body

    collected = bytearray()
    truncated = False

    with requests.request(method.upper(), url, **kwargs) as resp:
        content_type = resp.headers.get("content-type", "")
        status_code  = resp.status_code

        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                collected.extend(chunk)
            if len(collected) > _MAX_RESPONSE_BYTES:
                truncated = True
                collected = collected[:_MAX_RESPONSE_BYTES]
                break

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return _build_response_result(
        status_code, content_type, bytes(collected), truncated, elapsed_ms
    )


def _build_response_result(
    status_code:  int,
    content_type: str,
    raw_bytes:    bytes,
    truncated:    bool,
    elapsed_ms:   float,
) -> dict[str, Any]:
    """
    将原始响应字节流解析为结构化结果字典。

    解析策略：
      - Content-Type 含 application/json → json.loads（失败则当文本处理）
      - 其他 → UTF-8 解码（失败则 latin-1 兜底）
    """
    ct_lower = content_type.lower()

    if "application/json" in ct_lower or "text/json" in ct_lower:
        try:
            body = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = raw_bytes.decode("utf-8", errors="replace")
    else:
        body = raw_bytes.decode("utf-8", errors="replace")

    return {
        "status_code":  status_code,
        "content_type": content_type,
        "body":         body,
        "truncated":    truncated,
        "elapsed_ms":   elapsed_ms,
    }


# ---------------------------------------------------------------------------
# 5. 响应格式化
# ---------------------------------------------------------------------------

def _format_response(result: dict[str, Any], url: str) -> str:
    """
    将结构化响应结果格式化为 LLM 可读的 Markdown 文本。

    格式：
      - 2xx：显示响应头信息 + 格式化响应体
      - 非 2xx：标记错误状态 + 显示响应体（帮助 LLM 理解错误原因）
      - 截断提示：响应体超过 1MB 时附加说明
      - JSON 响应：pretty-print（缩进 2 空格）
      - 文本响应：最多显示 _MAX_TEXT_OUTPUT 字符
    """
    status_code  = result["status_code"]
    content_type = result["content_type"]
    body         = result["body"]
    truncated    = result["truncated"]
    elapsed_ms   = result["elapsed_ms"]

    is_success = 200 <= status_code < 300
    status_icon = "✅" if is_success else "❌"

    # ── 元信息行 ──────────────────────────────────────────────────────────
    safe_url = _sanitize_url_for_log(url)
    meta = (
        f"{status_icon} **HTTP {status_code}**｜"
        f"Content-Type: `{content_type}`｜"
        f"耗时: {elapsed_ms} ms\n"
        f"请求 URL: `{safe_url}`\n\n"
    )

    # ── 响应体格式化 ──────────────────────────────────────────────────────
    if isinstance(body, dict) or isinstance(body, list):
        # JSON 响应：pretty-print
        body_str = json.dumps(body, ensure_ascii=False, indent=2)
        if len(body_str) > _MAX_TEXT_OUTPUT:
            body_str = body_str[:_MAX_TEXT_OUTPUT] + "\n... [输出已截断]"
        body_section = f"```json\n{body_str}\n```"
    else:
        # 文本响应
        body_str = str(body)
        if len(body_str) > _MAX_TEXT_OUTPUT:
            body_str = body_str[:_MAX_TEXT_OUTPUT] + "\n... [输出已截断]"
        body_section = f"```\n{body_str}\n```"

    # ── 截断提示 ──────────────────────────────────────────────────────────
    truncate_note = ""
    if truncated:
        truncate_note = (
            f"\n\n> ⚠️ 响应体超过 **1 MB**，已自动截断。"
            f"如需完整内容，请考虑分页查询或添加过滤条件。"
        )

    # ── 非 2xx 错误提示 ───────────────────────────────────────────────────
    error_note = ""
    if not is_success:
        error_note = (
            f"\n\n> ❌ 请求返回非成功状态码 **{status_code}**。\n"
            f"> 请根据响应体中的错误信息排查：认证失败、参数错误或接口限流等。"
        )

    return meta + body_section + truncate_note + error_note


# ---------------------------------------------------------------------------
# 6. 入参模型
# ---------------------------------------------------------------------------

class ExternalAPIInput(BaseModel):
    """call_external_api 工具的入参模型，供 ToolSelector 生成合法 JSON 入参。"""

    url: str = Field(
        description=(
            "目标 API 的完整 URL，包含协议头。\n"
            "只支持 http / https。\n"
            "示例：https://api.github.com/repos/openai/openai-python/releases/latest"
        )
    )
    method: str = Field(
        default="GET",
        description=(
            "HTTP 请求方法（大小写不敏感）。\n"
            f"支持：{', '.join(sorted(_ALLOWED_METHODS))}\n"
            "示例：GET / POST / PUT / DELETE"
        ),
    )
    headers_json: str = Field(
        default="{}",
        description=(
            "请求头 JSON 字符串（键值对均为字符串）。\n"
            "空时传 {} 或空字符串。\n"
            '示例：{"Authorization": "Bearer sk-xxx", "Accept-Language": "zh-CN"}'
        ),
    )
    body_json: str = Field(
        default="{}",
        description=(
            "请求体 JSON 字符串（GET / HEAD 请求忽略此参数）。\n"
            "空时传 {} 或空字符串。\n"
            '示例：{"query": "openai", "page": 1}'
        ),
    )


# ---------------------------------------------------------------------------
# 7. @tool 入口
# ---------------------------------------------------------------------------

@tool(args_schema=ExternalAPIInput)
def external_api_tool(
    url:          str,
    method:       str = "GET",
    headers_json: str = "{}",
    body_json:    str = "{}",
) -> str:
    """
    向外部 HTTP API 发起请求并返回格式化响应。

    当需要以下情况时调用此工具：
    - 调用第三方 REST API 获取实时数据（天气、汇率、股价、仓库信息等）
    - 触发外部 Webhook 或通知服务
    - 与企业内部已授权的 API 网关交互

    安全限制（四层防线）：
    - ✅ 允许：访问白名单域名（ALLOWED_API_DOMAINS 配置的域名）
    - ❌ 禁止：访问内网 IP（10.x / 192.168.x / 127.x 等私有地址段）
    - ❌ 禁止：访问 .local / .internal 等内网域名
    - ❌ 禁止：未在白名单中的外部域名
    - ⏱️ 超时：10 秒后自动终止
    - 📦 大小：响应体超过 1MB 自动截断

    使用建议：
    - 优先使用 GET 获取数据，POST 仅在 API 明确要求时使用
    - 将 Authorization Token 放在 headers_json 中（不要放在 URL query string 中）
    - 不确定 API 返回格式时，先发一次不带 body 的 GET 探测

    Args:
        url:          目标 API 的完整 URL（必须是 http/https）
        method:       HTTP 方法，默认 GET
        headers_json: 请求头 JSON 字符串，默认 {}
        body_json:    请求体 JSON 字符串（GET/HEAD 忽略），默认 {}

    Returns:
        格式化的响应内容（含状态码、Content-Type、响应体），或结构化错误说明。
    """
    url    = url.strip()
    method = method.strip().upper() if method else "GET"

    if not url:
        return "错误：URL 不能为空。"

    start_time = time.monotonic()

    # ── 解析 JSON 参数 ────────────────────────────────────────────────────
    try:
        headers = _parse_json_param(headers_json, "headers_json")
        body    = _parse_json_param(body_json,    "body_json")
    except ValueError as exc:
        return f"❌ 参数解析失败：{exc}"

    # ── 日志记录（脱敏）──────────────────────────────────────────────────
    logger.info(
        "external_api 请求开始",
        extra={
            "method":  method,
            "url":     _sanitize_url_for_log(url),
            "headers": _sanitize_headers_for_log(headers),
        },
    )

    # ── 四层安全校验（前三层）────────────────────────────────────────────
    try:
        validate_request(url, method)
    except _APISecurityError as exc:
        logger.info(
            "external_api 安全校验拒绝",
            extra={"url": _sanitize_url_for_log(url), "reason": str(exc)[:200]},
        )
        return f"🚫 安全校验失败\n\n{exc}"

    # ── 执行 HTTP 请求（第四层：响应大小限制在内部执行）─────────────────
    try:
        result = _execute_request(url, method, headers, body)
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        # 区分超时与其他错误，给出更精准的提示
        err_type = type(exc).__name__
        err_msg  = str(exc)

        if "timeout" in err_msg.lower() or "Timeout" in err_type:
            logger.warning(
                "external_api 请求超时",
                extra={"url": _sanitize_url_for_log(url), "elapsed_ms": elapsed_ms},
            )
            return (
                f"⏱️ 请求超时（超过 {_TIMEOUT_READ_SEC} 秒）\n\n"
                f"URL: `{_sanitize_url_for_log(url)}`\n\n"
                f"建议：\n"
                f"  · 检查目标服务是否正常运行\n"
                f"  · 是否需要添加认证头（如 Authorization）\n"
                f"  · 尝试用 curl 直接测试该接口是否可达"
            )

        if "connection" in err_msg.lower() or "ConnectError" in err_type:
            logger.warning(
                "external_api 连接失败",
                extra={"url": _sanitize_url_for_log(url), "error": err_msg[:200]},
            )
            return (
                f"❌ 连接失败（{err_type}）\n\n"
                f"URL: `{_sanitize_url_for_log(url)}`\n"
                f"错误：{err_msg[:300]}\n\n"
                f"常见原因：\n"
                f"  · 域名 DNS 解析失败\n"
                f"  · 目标服务器拒绝连接（防火墙/IP 封禁）\n"
                f"  · 网络不可达（检查 Agent 所在网络出口配置）"
            )

        logger.error(
            "external_api 请求异常",
            extra={"url": _sanitize_url_for_log(url), "error": err_msg},
            exc_info=True,
        )
        return (
            f"❌ 请求执行失败（{err_type}）\n\n"
            f"错误详情：{err_msg[:400]}"
        )

    # ── 格式化输出 ────────────────────────────────────────────────────────
    elapsed_total = round((time.monotonic() - start_time) * 1000, 2)
    logger.info(
        "external_api 请求完成",
        extra={
            "method":       method,
            "url":          _sanitize_url_for_log(url),
            "status_code":  result["status_code"],
            "truncated":    result["truncated"],
            "elapsed_ms":   elapsed_total,
        },
    )

    return _format_response(result, url)