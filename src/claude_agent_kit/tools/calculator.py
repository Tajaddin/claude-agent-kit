"""Calculator tool.

Evaluates arithmetic expressions safely (no eval, no imports). Supports
+, -, *, /, %, **, and parentheses. Used in the benchmark to demonstrate
tool calls without network dependencies.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported literal: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval(node.operand))
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
        except ValueError as exc:
            return f"error: {exc}"
        # Render integers as integers, floats with at most 10 significant digits.
        if isinstance(value, int) or value == int(value):
            return str(int(value))
        return f"{value:.10g}"
