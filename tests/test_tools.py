"""Tool tests."""
from __future__ import annotations

import subprocess
import sys

import pytest

from claude_agent_kit.memory import InMemoryStore
from claude_agent_kit.tools import (
    CalculatorTool,
    MemoryRecallTool,
    MemorySaveTool,
    MockSearchTool,
)
from claude_agent_kit.tools.base import dispatch_tool


class TestCalculator:
    def test_addition(self):
        assert CalculatorTool().execute({"expression": "2 + 2"}) == "4"

    def test_precedence(self):
        assert CalculatorTool().execute({"expression": "(3 + 4) * 2"}) == "14"

    def test_division_by_zero(self):
        assert "division by zero" in CalculatorTool().execute({"expression": "1/0"})

    def test_invalid_expression(self):
        out = CalculatorTool().execute({"expression": "1 +"})
        assert out.startswith("error:")

    def test_floats(self):
        assert CalculatorTool().execute({"expression": "10 / 4"}) == "2.5"

    def test_blocks_unsafe(self):
        # imports/identifiers are unsupported
        out = CalculatorTool().execute({"expression": "__import__('os')"})
        assert out.startswith("error:")


class TestCalculatorLimits:
    @pytest.mark.parametrize("expression", ["9**9**9", "2**10**10", "(10**5000)**(10**5000)"])
    def test_huge_powers_are_refused_quickly(self, expression):
        # In a subprocess so an unbounded computation fails the test instead
        # of hanging the suite (big-int pow holds the GIL).
        code = (
            "from claude_agent_kit.tools.calculator import CalculatorTool;"
            f"print(CalculatorTool().execute({{'expression': {expression!r}}}))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        assert out.stdout.startswith("error: "), out.stdout + out.stderr

    def test_large_but_bounded_results_still_work(self):
        assert CalculatorTool().execute({"expression": "2**100"}) == str(2**100)
        assert CalculatorTool().execute({"expression": "2**-2"}) == "0.25"

    def test_result_size_cap_applies_to_every_operator(self):
        out = CalculatorTool().execute({"expression": "2**9000 * 2**9000"})
        assert out.startswith("error: result too large")

    @pytest.mark.parametrize("expression", ["10.0**400", "1e308 * 10", "1e308 * 10 - 1e308 * 10"])
    def test_float_overflow_is_an_error_not_an_exception(self, expression):
        assert CalculatorTool().execute({"expression": expression}).startswith("error: ")


class TestMockSearch:
    def test_substring_match(self):
        out = MockSearchTool().execute({"query": "what is anthropic"})
        assert "Anthropic is" in out

    def test_no_match(self):
        out = MockSearchTool().execute({"query": "obscure-term"})
        assert out == "no results"

    def test_empty_query(self):
        out = MockSearchTool().execute({"query": ""})
        assert out.startswith("error:")


class TestMemoryTools:
    def test_save_then_recall(self):
        store = InMemoryStore()
        save = MemorySaveTool(store=store)
        recall = MemoryRecallTool(store=store)

        out = save.execute({"key": "user.x", "value": "hello"})
        assert "saved" in out
        assert recall.execute({"key": "user.x"}) == "hello"

    def test_prefix_recall(self):
        store = InMemoryStore()
        save = MemorySaveTool(store=store)
        recall = MemoryRecallTool(store=store)
        save.execute({"key": "user.a", "value": "1"})
        save.execute({"key": "user.b", "value": "2"})
        out = recall.execute({"key": "user.*"})
        assert "user.a: 1" in out
        assert "user.b: 2" in out

    def test_missing_key(self):
        recall = MemoryRecallTool(store=InMemoryStore())
        assert "no value found" in recall.execute({"key": "missing"})


class TestDispatch:
    def test_dispatches_to_named_tool(self):
        tools = [CalculatorTool()]
        block = {"type": "tool_use", "id": "tid_1", "name": "calculator", "input": {"expression": "1+1"}}
        result = dispatch_tool(tools, block)
        assert result.tool_use_id == "tid_1"
        assert result.content == "2"
        assert result.is_error is False

    def test_unknown_tool_returns_error(self):
        result = dispatch_tool([CalculatorTool()], {"type": "tool_use", "id": "x", "name": "ghost"})
        assert result.is_error is True
        assert "Unknown tool" in result.content

    def test_exception_becomes_error_result(self):
        class Boom:
            name = "boom"
            description = "raises"
            input_schema = {"type": "object", "properties": {}}

            def execute(self, params):
                raise RuntimeError("explosion")

        result = dispatch_tool([Boom()], {"type": "tool_use", "id": "x", "name": "boom"})
        assert result.is_error is True
        assert "explosion" in result.content
