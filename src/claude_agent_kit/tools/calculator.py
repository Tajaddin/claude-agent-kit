"""Calculator tool.

Evaluates arithmetic expressions safely (no eval, no imports). Supports
+, -, *, /, %, **, and parentheses. Used in the benchmark to demonstrate
tool calls without network dependencies.

Exponents and integer results are capped: Python integers are unbounded,
so without a cap an expression such as 9**9**9 runs for minutes while
holding the GIL.
"""
from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Any

MAX_EXPONENT = 10_000
MAX_INT_BITS = 10_000  # about 3,000 digits, inside Python's 4,300-digit str() limit


def _pow(base: float, exponent: float) -> float:
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent too large (limit {MAX_EXPONENT})")
    # |base|**e has at least e*(bit_length-1)+1 bits; refuse before computing.
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        if (abs(base).bit_length() - 1) * exponent >= MAX_INT_BITS:
            raise ValueError("result too large")
    return operator.pow(base, exponent)


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: _pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _checked(value: object) -> float:
    if isinstance(value, complex):
        raise ValueError("result is not a real number")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ValueError("result too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is not finite")
    return value


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported literal: {node.value!r}")
        return _checked(node.value)
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return _checked(op(_eval(node.left), _eval(node.right)))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return _checked(op(_eval(node.operand)))
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


@dataclass
class CalculatorTool:
    name: str = "calculator"
    description: str = (
        "Evaluate a single arithmetic expression. Supports +, -, *, /, %, **, "
        "and parentheses. Returns the numeric result or an error string."
    )
    input_schema: dict[str, Any] = None  # set in __post_init__

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate, e.g. '(3 + 4) * 2'",
                    }
                },
                "required": ["expression"],
            }

    def execute(self, params: dict[str, Any]) -> str:
        expression = str(params.get("expression", "")).strip()
        if not expression:
            return "error: expression is required"
        try:
            tree = ast.parse(expression, mode="eval")
            value = _eval(tree)
        except SyntaxError as exc:
            return f"error: invalid expression ({exc.msg})"
        except ZeroDivisionError:
            return "error: division by zero"
        except OverflowError:
            return "error: result too large"
        except ValueError as exc:
            return f"error: {exc}"
        # Render integers as integers, floats with at most 10 significant digits.
        if isinstance(value, int) or value == int(value):
            return str(int(value))
        return f"{value:.10g}"
