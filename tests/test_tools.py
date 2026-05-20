"""Tool tests."""
from __future__ import annotations

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
