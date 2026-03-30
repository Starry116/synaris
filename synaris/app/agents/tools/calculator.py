"""
@File       : calculator.py
@Author     : Starry Hung
@Created    : 2026-03-24
@Version    : 1.0.0
@Description: Agent 安全数学计算工具。
@Features:
  - @tool 装饰器：供 LangGraph ToolSelector 节点直接调用
  - 三层安全防御（从快到慢，逐层降级）：
      1. AST 白名单校验：遍历语法树，拒绝任何非数学节点（函数调用/导入/赋值等）
      2. numexpr 加速执行：向量化数学运算，性能优于原生 eval
      3. 安全 eval 兜底：numexpr 未安装时的备用路径，使用隔离命名空间执行
  - 支持的运算：四则运算 / 幂运算 / 比较运算 / 三角/对数/常用数学函数
  - 支持常量：pi / e / inf / nan
  - 禁止：函数调用（除白名单）/ 变量赋值 / 导入 / 属性访问 / 条件语句
  - 精度：浮点结果默认保留 10 位有效数字，整数结果原样返回
  - 错误处理：语法错误 / 非法表达式 / 运算溢出均返回结构化错误信息

@Project    : Synaris
@License    : Apache License 2.0

@ChangeLog:
    2026-03-24  Starry  Initial creation
"""

from __future__ import annotations

import ast
import math
import logging
from typing import Any, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 安全数学常量（注入到执行命名空间）───────────────────────────────────────
_SAFE_CONSTANTS: dict[str, Any] = {
    "pi":    math.pi,
    "e":     math.e,
    "inf":   math.inf,
    "nan":   math.nan,
    "tau":   math.tau,
}

# ── 白名单数学函数（只允许这些函数出现在表达式中）──────────────────────────
_SAFE_FUNCTIONS: dict[str, Any] = {
    # 基础
    "abs":     abs,
    "round":   round,
    "min":     min,
    "max":     max,
    "sum":     sum,
    "pow":     pow,
    # 数学模块
    "sqrt":    math.sqrt,
    "ceil":    math.ceil,
    "floor":   math.floor,
    "log":     math.log,
    "log2":    math.log2,
    "log10":   math.log10,
    "exp":     math.exp,
    # 三角函数
    "sin":     math.sin,
    "cos":     math.cos,
    "tan":     math.tan,
    "asin":    math.asin,
    "acos":    math.acos,
    "atan":    math.atan,
    "atan2":   math.atan2,
    # 双曲函数
    "sinh":    math.sinh,
    "cosh":    math.cosh,
    "tanh":    math.tanh,
    # 其他
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd":     math.gcd,
    "hypot":   math.hypot,
}

# 安全命名空间 = 常量 + 函数（禁止 __builtins__ 中的非数学内容）
_SAFE_NAMESPACE: dict[str, Any] = {
    "__builtins__": {},   # 清空内置，防止 __import__ / exec / eval 等调用
    **_SAFE_CONSTANTS,
    **_SAFE_FUNCTIONS,
}

# ── AST 白名单节点类型（只允许这些出现在语法树中）──────────────────────────
_ALLOWED_AST_NODES = (
    # 表达式容器
    ast.Expression,
    ast.Expr,
    # 字面量
    ast.Constant,
    ast.Num,        # Python 3.7 兼容
    # 运算符节点
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    # 运算符本体
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    # 函数调用（仅白名单函数名，在下方额外校验）
    ast.Call,
    ast.Name,
    # 元组/列表（用于 min/max/sum 等多参数函数）
    ast.Tuple,
    ast.List,
    ast.Load,
)


# ─────────────────────────────────────────────
# 1. AST 安全校验器
# ─────────────────────────────────────────────

class _ASTSafetyChecker(ast.NodeVisitor):
    """
    遍历 AST，拒绝任何非数学节点。

    类比机场安检：
    - 每个 AST 节点都要过安检门（visit）
    - 不在白名单中的节点直接触发警报（raise ValueError）
    - 函数调用还要额外检查函数名是否在白名单中
    """

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_AST_NODES):
            node_type = type(node).__name__
            raise ValueError(
                f"不允许的语法节点「{node_type}」，"
                f"计算工具只支持纯数学表达式。"
            )
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """函数调用节点额外检查：函数名必须在白名单中。"""
        # 只允许直接名称调用（如 sqrt(...)），禁止属性调用（如 math.sqrt(...)）
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "不允许属性方式调用函数（如 math.sqrt）。"
                f"请直接使用函数名，例如 sqrt({ast.unparse(node.args[0]) if node.args else '...'})"
            )
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(
                f"函数「{func_name}」不在允许列表中。\n"
                f"允许的函数：{', '.join(sorted(_SAFE_FUNCTIONS.keys()))}"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        """变量名节点检查：只允许白名单常量和函数名。"""
        allowed_names = set(_SAFE_CONSTANTS) | set(_SAFE_FUNCTIONS)
        if node.id not in allowed_names:
            raise ValueError(
                f"未知标识符「{node.id}」。"
                f"允许的常量：{', '.join(sorted(_SAFE_CONSTANTS.keys()))}"
            )
        self.generic_visit(node)


def _ast_validate(expression: str) -> ast.Expression:
    """
    将表达式字符串解析为 AST 并通过安全校验。
    返回解析后的 AST Expression 节点（供后续 eval 使用）。
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc}") from exc

    checker = _ASTSafetyChecker()
    checker.visit(tree)   # 若发现非法节点，抛出 ValueError
    return tree


# ─────────────────────────────────────────────
# 2. 执行后端（numexpr 优先，ast.literal_eval 兜底）
# ─────────────────────────────────────────────

def _execute_with_numexpr(expression: str) -> Union[float, int, bool]:
    """
    使用 numexpr 库执行数学表达式（更快、更安全）。

    numexpr 优点：
    - 内置沙箱，不支持任意 Python 语法
    - 向量化执行，适合大数值计算
    - 不需要我们自己维护安全命名空间
    """
    import numexpr as ne  # type: ignore[import]

    # numexpr 支持部分常量，手动注入 pi/e
    local_dict = {k: v for k, v in _SAFE_CONSTANTS.items()
                  if isinstance(v, (int, float))}
    result = ne.evaluate(expression, local_dict=local_dict)

    # numexpr 返回 numpy scalar，转为 Python 原生类型
    return result.item() if hasattr(result, "item") else float(result)


def _execute_with_safe_eval(expression: str) -> Union[float, int, bool]:
    """
    使用经过 AST 校验的安全 eval 执行表达式（numexpr 不可用时的备用路径）。

    安全保证来自于前置的 _ast_validate：
    - 已通过 AST 白名单校验，只有合法数学节点才能到达这里
    - 使用隔离的 _SAFE_NAMESPACE，__builtins__ 已清空
    """
    tree = _ast_validate(expression)
    code = compile(tree, filename="<calculator>", mode="eval")
    return eval(code, _SAFE_NAMESPACE.copy())  # noqa: S307


# ─────────────────────────────────────────────
# 3. 结果格式化
# ─────────────────────────────────────────────

def _format_result(result: Any, expression: str) -> str:
    """将计算结果格式化为 LLM 友好的字符串。"""
    if isinstance(result, bool):
        return f"{expression} = {'True' if result else 'False'}"

    if isinstance(result, int):
        return f"{expression} = {result}"

    if isinstance(result, float):
        if math.isnan(result):
            return f"{expression} = NaN（结果为非数字，请检查表达式）"
        if math.isinf(result):
            sign = "+" if result > 0 else "-"
            return f"{expression} = {sign}∞（结果为无穷大）"

        # 智能精度：整数结果不显示小数点
        if result == int(result) and abs(result) < 1e15:
            return f"{expression} = {int(result)}"

        # 科学计数法处理超大/超小数
        if abs(result) >= 1e10 or (abs(result) < 1e-4 and result != 0):
            return f"{expression} ≈ {result:.6e}"

        return f"{expression} = {result:.10g}"

    # 其他类型（理论上不应出现）
    return f"{expression} = {result}"


# ─────────────────────────────────────────────
# 4. 入参模型
# ─────────────────────────────────────────────

class CalculatorInput(BaseModel):
    """calculator 工具的入参模型。"""

    expression: str = Field(
        description=(
            "纯数学表达式字符串。"
            "支持：四则运算(+ - * /)、幂运算(**)、括号、"
            "常量(pi/e)、函数(sqrt/sin/cos/log/abs/round 等)。"
            "示例：'sqrt(2**10 + 3**10)'、'sin(pi/6)'、'log(1000, 10)'"
        )
    )


# ─────────────────────────────────────────────
# 5. @tool 入口
# ─────────────────────────────────────────────

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    安全执行数学表达式并返回计算结果。

    当需要以下情况时调用此工具：
    - 精确数值计算（四则运算、幂、开方、三角函数等）
    - 验证数值结论（如确认某公式的计算结果）
    - 避免 LLM 自行估算导致的数值错误

    支持的运算和函数：
        基础：+ - * / // % ** ( )
        常量：pi, e, inf, tau
        函数：sqrt, abs, round, sin, cos, tan, log, log2, log10,
              exp, ceil, floor, min, max, sum, factorial, degrees, radians

    禁止：变量赋值、import、exec、eval、文件操作、网络调用、属性访问（math.xxx）

    Args:
        expression: 数学表达式字符串，如 'sqrt(16) + 2**8'

    Returns:
        格式化的计算结果字符串，或详细的错误说明。
    """
    expression = expression.strip()

    if not expression:
        return "错误：表达式不能为空。"

    # 长度防御（防止超长表达式 DoS）
    if len(expression) > 500:
        return "错误：表达式过长（最多 500 字符），请简化后重试。"

    logger.debug("calculator 执行: %s", expression)

    try:
        # ── 第一步：AST 白名单校验（无论使用哪个后端都要过） ──────────────
        _ast_validate(expression)

        # ── 第二步：优先使用 numexpr，失败则用 safe eval ──────────────────
        try:
            import numexpr  # noqa: F401  — 仅用于探测是否已安装
            result = _execute_with_numexpr(expression)
            backend = "numexpr"
        except (ImportError, Exception) as numexpr_err:
            if not isinstance(numexpr_err, ImportError):
                logger.debug("numexpr 执行失败，回退 safe_eval: %s", numexpr_err)
            result = _execute_with_safe_eval(expression)
            backend = "safe_eval"

        formatted = _format_result(result, expression)
        logger.info("calculator 成功 | backend=%s | expr=%s | result=%s",
                    backend, expression, formatted)
        return formatted

    except ValueError as exc:
        # AST 校验失败或语法错误（用户输入问题）
        logger.info("calculator 校验失败 | expr=%s | reason=%s", expression, exc)
        return f"表达式错误：{exc}"

    except ZeroDivisionError:
        return f"计算错误：除数为零。表达式「{expression}」包含除以零的操作。"

    except OverflowError:
        return f"计算溢出：结果超出浮点数范围。请简化表达式「{expression}」。"

    except Exception as exc:
        logger.warning("calculator 未知错误 | expr=%s | error=%s", expression, exc)
        return f"计算失败：{type(exc).__name__}: {exc}"