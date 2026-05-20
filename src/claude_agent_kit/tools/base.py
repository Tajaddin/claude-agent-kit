"""Tool protocol and dispatch.

Every tool exposes the same shape:
- name + description + input_schema (the Anthropic Messages API tool def)
- execute(input_dict) -> ToolResult

The agent loop reads tool_use blocks from the model, calls dispatch_tool,
then sends tool_result blocks back in the next user message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolResult:
    """One tool's execution output. Becomes a tool_result block."""

    tool_use_id: str
    content: str
    is_error: bool = False

    def as_block(self) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            block["is_error"] = True
        return block


class Tool(Protocol):
    """All tools implement this Protocol."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def execute(self, params: dict[str, Any]) -> str: ...


def dispatch_tool(
    tools: list[Tool],
    tool_use: dict[str, Any],
) -> ToolResult:
    """Run the tool named in tool_use against params, return a ToolResult.

    tool_use is a tool_use block from the Anthropic API:
      {"type": "tool_use", "id": "toolu_...", "name": "...", "input": {...}}
    """
    name = tool_use.get("name", "")
    tool_use_id = tool_use.get("id", "")
    params = tool_use.get("input", {}) or {}

    for tool in tools:
        if tool.name == name:
            try:
                return ToolResult(tool_use_id=tool_use_id, content=tool.execute(params))
            except Exception as exc:  # noqa: BLE001 - surface the error to the model
                return ToolResult(
                    tool_use_id=tool_use_id,
                    content=f"Tool {name} raised: {type(exc).__name__}: {exc}",
                    is_error=True,
                )

    return ToolResult(
        tool_use_id=tool_use_id,
        content=f"Unknown tool: {name}",
        is_error=True,
    )


def tools_as_anthropic_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    """Project tool list into the Anthropic Messages API tools payload."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]
