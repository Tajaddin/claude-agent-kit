"""Tool tests."""
from __future__ import annotations

import contextvars
import subprocess
import sys
import threading
import time

import pytest

from claude_agent_kit.memory import InMemoryStore
from claude_agent_kit.tools import (
    CalculatorTool,
    MemoryRecallTool,
    MemorySaveTool,
    MockSearchTool,
)
from claude_agent_kit.tools.base import dispatch_tool, dispatch_tools


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


class _SleepTool:
    def __init__(self, name: str, seconds: float, reply: str = "done"):
        self.name = name
        self.description = "sleeps"
        self.input_schema = {"type": "object", "properties": {}}
        self.seconds = seconds
        self.reply = reply

    def execute(self, params):
        time.sleep(self.seconds)
        return self.reply


class TestDispatchTimeout:
    def test_slow_tool_returns_an_error_result_at_the_timeout(self):
        block = {"type": "tool_use", "id": "slow_1", "name": "slow", "input": {}}
        start = time.perf_counter()
        result = dispatch_tool([_SleepTool("slow", 5.0)], block, timeout=0.2)
        assert time.perf_counter() - start < 2.0
        assert result.is_error is True
        assert result.tool_use_id == "slow_1"
        assert "timed out after 0.2" in result.content

    def test_fast_tool_is_unaffected_by_the_timeout(self):
        block = {"type": "tool_use", "id": "f", "name": "fast", "input": {}}
        result = dispatch_tool([_SleepTool("fast", 0.0, reply="ok")], block, timeout=5.0)
        assert (result.content, result.is_error) == ("ok", False)


class TestDispatchTools:
    def test_runs_calls_concurrently_and_keeps_their_order(self):
        barrier = threading.Barrier(2)

        class Meet:
            description = "waits for the other call"
            input_schema = {"type": "object", "properties": {}}

            def __init__(self, name):
                self.name = name

            def execute(self, params):
                barrier.wait(timeout=3)
                return self.name

        uses = [
            {"type": "tool_use", "id": "1", "name": "left", "input": {}},
            {"type": "tool_use", "id": "2", "name": "right", "input": {}},
        ]
        results = dispatch_tools([Meet("left"), Meet("right")], uses)
        assert [(r.tool_use_id, r.content, r.is_error) for r in results] == [
            ("1", "left", False),
            ("2", "right", False),
        ]

    def test_tools_see_the_callers_context_variables(self):
        span = contextvars.ContextVar("span", default="none")

        class ReadSpan:
            name = "read_span"
            description = "returns the current span"
            input_schema = {"type": "object", "properties": {}}

            def execute(self, params):
                return span.get()

        uses = [{"type": "tool_use", "id": str(i), "name": "read_span", "input": {}} for i in range(2)]
        token = span.set("turn-7")
        try:
            results = dispatch_tools([ReadSpan()], uses)
        finally:
            span.reset(token)
        assert [r.content for r in results] == ["turn-7", "turn-7"]

    def test_timeout_is_per_call_and_does_not_hold_back_fast_calls(self):
        uses = [
            {"type": "tool_use", "id": "s", "name": "slow", "input": {}},
            {"type": "tool_use", "id": "f", "name": "fast", "input": {}},
        ]
        start = time.perf_counter()
        results = dispatch_tools([_SleepTool("slow", 5.0), _SleepTool("fast", 0.0, reply="ok")], uses, timeout=0.3)
        assert time.perf_counter() - start < 2.0
        assert results[0].is_error is True and "timed out" in results[0].content
        assert (results[1].content, results[1].is_error) == ("ok", False)
