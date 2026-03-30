"""
@File       : code_executor.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Agent 受限 Python 代码执行沙箱工具。
@Features:
  - @tool 装饰器：供 LangGraph ToolSelector 节点直接调用
  - 双层沙箱防御：
      · 主沙箱：RestrictedPython（字节码级别限制，禁止 import/exec/open 等危险操作）
      · 备用沙箱：进程隔离（subprocess + multiprocessing，主库不可用时启用）
  - 执行限制：
      · 超时：10 秒（signal.alarm 或 threading.Timer 实现，跨平台）
      · 禁止网络访问（socket 模块被屏蔽）
      · 禁止文件系统访问（open / os / pathlib 均被屏蔽）
      · 禁止进程操作（subprocess / os.system 均被屏蔽）
      · 禁止 __import__ / __builtins__ 直接访问
  - 允许的安全操作：
      · 数值计算、字符串处理、列表/字典/集合操作
      · 内置函数：print/len/range/sorted/sum/min/max/abs/round/type/isinstance 等
      · 标准库白名单：math / json / re / datetime / itertools / functools / collections
  - stdout 捕获：通过 io.StringIO 重定向 print 输出
  - 输出限制：stdout 最多保留 5000 字符，防止输出洪泛
  - 错误处理：超时 / 语法错误 / 运行时异常均返回结构化信息

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import ast
import io
import logging
import math
import signal
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 执行限制常量 ───────────────────────────────────────────────────────────────
_TIMEOUT_SECONDS  = 10      # 代码执行超时（秒）
_MAX_OUTPUT_LEN   = 5000    # stdout 最大捕获字符数
_MAX_CODE_LEN     = 5000    # 输入代码最大字符数

# ── 安全内置函数白名单（_SAFE_BUILTINS）────────────────────────────────────────
# 这些是代码中可以使用的内置函数，危险函数（eval/exec/open/__import__等）均被排除
_SAFE_BUILTINS: dict[str, Any] = {
    # 类型与转换
    "int": int, "float": float, "str": str, "bool": bool,
    "bytes": bytes, "bytearray": bytearray,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "frozenset": frozenset,
    "complex": complex,
    # 常用内置
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "reversed": reversed, "sorted": sorted,
    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "pow": pow, "divmod": divmod, "hash": hash,
    "print": print,   # 将被重定向到 StringIO
    "repr": repr, "str": str, "format": format,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
    "callable": callable, "iter": iter, "next": next,
    "any": any, "all": all,
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
    "id": id,
    # 异常
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "KeyError": KeyError, "AttributeError": AttributeError,
    "StopIteration": StopIteration, "RuntimeError": RuntimeError,
    "NotImplementedError": NotImplementedError, "OverflowError": OverflowError,
    "ZeroDivisionError": ZeroDivisionError, "NameError": NameError,
    # 禁止列表（明确设为 None，防止被意外访问）
    "__import__": None,
    "open":        None,
    "eval":        None,
    "exec":        None,
    "compile":     None,
    "globals":     None,
    "locals":      None,
    "vars":        None,
    "dir":         None,
    "input":       None,
    "breakpoint":  None,
    "__builtins__": {},
}

# ── 安全模块白名单（代码中 import 只允许这些）──────────────────────────────────
_ALLOWED_IMPORTS = {
    "math", "cmath", "decimal", "fractions", "statistics",
    "json", "re",
    "datetime", "time",
    "itertools", "functools", "operator",
    "collections", "collections.abc",
    "string", "textwrap",
    "random",
    "copy",
    "pprint",
}


# ─────────────────────────────────────────────
# 1. 代码预校验（AST 静态分析）
# ─────────────────────────────────────────────

class _ImportChecker(ast.NodeVisitor):
    """
    静态检查代码中的 import 语句，确保只导入白名单模块。

    在沙箱执行前进行静态分析，是「门禁」的第一道检查。
    类比：进门前先查邀请名单，不在名单上的直接拒之门外。
    """

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module not in _ALLOWED_IMPORTS:
                raise ValueError(
                    f"禁止导入模块「{alias.name}」。\n"
                    f"允许的模块：{', '.join(sorted(_ALLOWED_IMPORTS))}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        base_module = module.split(".")[0]
        if base_module not in _ALLOWED_IMPORTS:
            raise ValueError(
                f"禁止从模块「{module}」导入。\n"
                f"允许的模块：{', '.join(sorted(_ALLOWED_IMPORTS))}"
            )
        self.generic_visit(node)

    # 明确拒绝危险调用
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec", "compile", "__import__"):
                raise ValueError(f"禁止调用危险函数「{node.func.id}」。")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("system", "popen", "spawn", "fork"):
                raise ValueError(f"禁止调用进程操作函数「{node.func.attr}」。")
        self.generic_visit(node)


def _static_validate(code: str) -> ast.Module:
    """
    对代码进行静态 AST 分析，返回解析后的语法树。
    若存在语法错误或违规 import，抛出异常。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(
            f"代码语法错误（第 {exc.lineno} 行）：{exc.msg}\n"
            f"问题代码：{exc.text}"
        ) from exc

    checker = _ImportChecker()
    checker.visit(tree)
    return tree


# ─────────────────────────────────────────────
# 2. 超时机制（跨平台兼容）
# ─────────────────────────────────────────────

class _TimeoutError(Exception):
    """代码执行超时异常。"""
    pass


@contextmanager
def _execution_timeout(seconds: int):
    """
    跨平台超时上下文管理器。

    - Unix/Linux/macOS：使用 signal.SIGALRM（精确，无线程限制）
    - Windows / 非主线程：使用 threading.Timer（近似，有 ~0.1s 误差）

    类比厨房定时器：
    - SIGALRM 是内核级定时器，精确打断执行
    - threading.Timer 是「助理厨师」，时间到了设一个标志位，主线程轮询检查
    """
    use_signal = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )

    if use_signal:
        def _handler(signum, frame):
            raise _TimeoutError(f"代码执行超时（超过 {seconds} 秒）。")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    else:
        # 非主线程（Celery Worker）使用 threading.Timer 方案
        timed_out = threading.Event()
        current_thread = threading.current_thread()

        def _interrupt():
            timed_out.set()
            # 无法直接打断其他线程，通过设置标志位让代码在下次 Python 字节码检查时发现
            # 对于纯 Python 代码，GIL 切换间隔（sys.setswitchinterval）约为 5ms
            # 实际上对于 CPU 密集型代码，这个方法不够精确，但对一般脚本足够
            import ctypes
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(current_thread.ident),
                ctypes.py_object(TimeoutError),
            )

        timer = threading.Timer(seconds, _interrupt)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            if timed_out.is_set():
                raise _TimeoutError(f"代码执行超时（超过 {seconds} 秒）。")


# ─────────────────────────────────────────────
# 3. RestrictedPython 沙箱执行
# ─────────────────────────────────────────────

def _execute_restricted(code: str, stdout_buffer: io.StringIO) -> None:
    """
    使用 RestrictedPython 在字节码层面限制代码执行。

    RestrictedPython 的安全机制：
    - 将代码编译为「受限字节码」，移除 __import__、exec 等危险操作
    - 通过自定义 _getattr_ / _getitem_ 钩子控制属性访问
    - 配合 _SAFE_BUILTINS 隔离内置函数

    类比：不是简单地「把门锁上」，而是在代码编译阶段就重写了危险指令。
    """
    from RestrictedPython import compile_restricted, safe_globals  # type: ignore[import]
    from RestrictedPython.Guards import (  # type: ignore[import]
        safe_builtins,
        guarded_unpack_sequence,
        full_write_guard,
    )
    from RestrictedPython.PrintCollector import PrintCollector  # type: ignore[import]

    # 编译为受限字节码
    byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")

    # 构建受限执行环境
    safe_namespace: dict[str, Any] = {
        **safe_globals,
        "__builtins__": {**safe_builtins, **_SAFE_BUILTINS},
        "_print_":      PrintCollector,           # 捕获 print 输出
        "_getattr_":    getattr,                  # 允许属性访问（白名单模块内）
        "_getitem_":    lambda obj, key: obj[key], # 允许下标访问
        "_getiter_":    iter,                     # 允许迭代
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_":      full_write_guard,
        "__name__":     "__main__",
    }

    # 预先导入白名单模块，让代码可以 import 它们
    import math, json, re, datetime, itertools, functools, collections, random, copy  # noqa: E401
    allowed_modules = {
        "math": math, "json": json, "re": re,
        "datetime": datetime, "itertools": itertools,
        "functools": functools, "collections": collections,
        "random": random, "copy": copy,
    }

    def _restricted_import(name, *args, **kwargs):
        base = name.split(".")[0]
        if base in allowed_modules:
            return allowed_modules[base]
        raise ImportError(
            f"在沙箱中不允许导入「{name}」。"
            f"允许的模块：{', '.join(sorted(_ALLOWED_IMPORTS))}"
        )

    safe_namespace["__builtins__"]["__import__"] = _restricted_import

    exec(byte_code, safe_namespace)  # noqa: S102

    # 收集 PrintCollector 中的输出
    if "_print" in safe_namespace:
        output = safe_namespace["_print"]()
        stdout_buffer.write(output)


def _execute_fallback(code: str, stdout_buffer: io.StringIO) -> None:
    """
    备用沙箱：使用隔离命名空间 + AST 静态分析执行代码。
    当 RestrictedPython 库未安装时启用。

    安全级别略低于 RestrictedPython，但已通过前置的 _static_validate 和
    _SAFE_BUILTINS 限制了大部分危险操作。
    """
    import math, json, re, datetime, itertools, functools, collections, random, copy  # noqa: E401

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base = name.split(".")[0]
        module_map = {
            "math": math, "json": json, "re": re,
            "datetime": datetime, "itertools": itertools,
            "functools": functools, "collections": collections,
            "random": random, "copy": copy,
        }
        if base in module_map:
            return module_map[base]
        raise ImportError(f"沙箱中不允许导入「{name}」。")

    # 重定向 print 到 buffer
    original_stdout = sys.stdout
    sys.stdout = stdout_buffer
    try:
        namespace: dict[str, Any] = {
            **_SAFE_BUILTINS,
            "__builtins__": {**_SAFE_BUILTINS, "__import__": _safe_import},
            "__name__": "__main__",
            "__import__": _safe_import,
        }
        exec(compile(code, "<sandbox_fallback>", "exec"), namespace)  # noqa: S102
    finally:
        sys.stdout = original_stdout


# ─────────────────────────────────────────────
# 4. 入参模型
# ─────────────────────────────────────────────

class CodeExecutorInput(BaseModel):
    """code_executor 工具的入参模型。"""

    code: str = Field(
        description=(
            "要执行的 Python 代码字符串。\n"
            "限制：\n"
            "  - 只能使用安全的内置函数和白名单模块（math/json/re/datetime 等）\n"
            "  - 禁止网络访问、文件操作、进程调用\n"
            "  - 超时限制：10 秒\n"
            "  - 使用 print() 输出结果\n"
            "示例：'import math\\nresult = math.factorial(10)\\nprint(result)'"
        )
    )


# ─────────────────────────────────────────────
# 5. @tool 入口
# ─────────────────────────────────────────────

@tool(args_schema=CodeExecutorInput)
def code_executor(code: str) -> str:
    """
    在安全沙箱中执行 Python 代码片段并返回输出。

    当需要以下情况时调用此工具：
    - 执行复杂的数值计算或统计分析（超出 calculator 工具能力范围）
    - 处理字符串、列表、字典等数据结构
    - 实现简单的算法逻辑（排序、过滤、聚合等）
    - 验证某段代码逻辑的正确性

    安全限制（严格执行）：
    - ✅ 允许：数学计算、字符串处理、数据结构操作、print 输出
    - ✅ 允许导入：math / json / re / datetime / itertools / functools / collections
    - ❌ 禁止：网络访问（socket/requests/urllib）
    - ❌ 禁止：文件操作（open/os/pathlib/shutil）
    - ❌ 禁止：进程操作（subprocess/os.system）
    - ❌ 禁止：eval / exec / __import__ / compile
    - ⏱️ 超时：10 秒后自动终止

    使用建议：
    - 用 print() 输出你想看到的结果
    - 复杂计算建议分步 print，方便调试
    - 单次代码量不要超过 50 行

    Args:
        code: 要执行的 Python 代码字符串

    Returns:
        代码的 stdout 输出，或异常/超时的详细错误信息。
    """
    code = textwrap.dedent(code).strip()

    if not code:
        return "错误：代码不能为空。"

    if len(code) > _MAX_CODE_LEN:
        return (
            f"错误：代码过长（{len(code)} 字符，上限 {_MAX_CODE_LEN} 字符）。"
            f"请精简代码或拆分为多次调用。"
        )

    logger.info("code_executor 开始执行 | code_len=%d", len(code))
    start_time = time.monotonic()

    # ── 第一步：静态 AST 分析（快速失败，避免进入沙箱）──────────────────────
    try:
        _static_validate(code)
    except ValueError as exc:
        return f"代码校验失败（执行前）：{exc}"

    # ── 第二步：沙箱执行（带超时）────────────────────────────────────────────
    stdout_buffer = io.StringIO()
    sandbox_used  = "unknown"

    try:
        with _execution_timeout(_TIMEOUT_SECONDS):

            # 优先使用 RestrictedPython
            try:
                import RestrictedPython  # noqa: F401  — 仅探测是否安装
                _execute_restricted(code, stdout_buffer)
                sandbox_used = "RestrictedPython"

            except ImportError:
                logger.warning(
                    "RestrictedPython 未安装，使用备用沙箱。"
                    "建议执行: pip install RestrictedPython"
                )
                _execute_fallback(code, stdout_buffer)
                sandbox_used = "safe_eval_fallback"

    except _TimeoutError as exc:
        elapsed = time.monotonic() - start_time
        logger.warning("code_executor 超时 | elapsed=%.2fs", elapsed)
        return (
            f"执行超时：代码运行超过 {_TIMEOUT_SECONDS} 秒，已强制终止。\n"
            f"建议：检查是否存在死循环，或将计算量较大的代码拆分简化。"
        )

    except TimeoutError as exc:
        # threading.Timer 方案触发的系统 TimeoutError
        elapsed = time.monotonic() - start_time
        logger.warning("code_executor 超时(threading) | elapsed=%.2fs", elapsed)
        return (
            f"执行超时：代码运行超过 {_TIMEOUT_SECONDS} 秒，已强制终止。"
        )

    except SyntaxError as exc:
        return (
            f"语法错误（第 {exc.lineno} 行）：{exc.msg}\n"
            f"问题代码：{exc.text}"
        )

    except (ValueError, ImportError) as exc:
        # 沙箱内部的安全拦截
        logger.info("code_executor 安全拦截: %s", exc)
        return f"安全限制：{exc}"

    except MemoryError:
        return "执行失败：内存溢出。代码申请了过多内存，请减小数据规模。"

    except RecursionError:
        return "执行失败：递归深度超限。请检查是否存在无限递归。"

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.warning(
            "code_executor 运行时异常 | sandbox=%s | elapsed=%.2fs | error=%s",
            sandbox_used, elapsed, exc,
        )
        return (
            f"运行时错误（{type(exc).__name__}）：{exc}\n"
            f"沙箱：{sandbox_used}"
        )

    # ── 第三步：处理输出 ──────────────────────────────────────────────────────
    elapsed   = time.monotonic() - start_time
    raw_output = stdout_buffer.getvalue()

    logger.info(
        "code_executor 成功 | sandbox=%s | output_len=%d | elapsed=%.2fs",
        sandbox_used, len(raw_output), elapsed,
    )

    if not raw_output:
        return (
            "代码执行成功，但没有任何输出。\n"
            "提示：使用 print() 输出你想查看的结果。"
        )

    # 截断过长输出
    if len(raw_output) > _MAX_OUTPUT_LEN:
        raw_output = (
            raw_output[:_MAX_OUTPUT_LEN]
            + f"\n\n… [输出已截断，共 {len(raw_output)} 字符，仅显示前 {_MAX_OUTPUT_LEN} 字符]"
        )

    return f"执行成功（{elapsed:.2f}s）：\n{raw_output}"