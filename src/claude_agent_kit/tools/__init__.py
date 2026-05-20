"""Example tools the agent uses in the demo + benchmark."""

from claude_agent_kit.tools.base import Tool, ToolResult, dispatch_tool
from claude_agent_kit.tools.calculator import CalculatorTool
from claude_agent_kit.tools.memory_io import MemoryRecallTool, MemorySaveTool
from claude_agent_kit.tools.search import MockSearchTool

__all__ = [
    "CalculatorTool",
    "MemoryRecallTool",
    "MemorySaveTool",
    "MockSearchTool",
    "Tool",
    "ToolResult",
    "dispatch_tool",
]
