"""Tool protocol and dispatch.

Every tool exposes the same shape:
- name + description + input_schema (the Anthropic Messages API tool def)
- execute(input_dict) -> ToolResult

The agent loop reads tool_use blocks from the model, runs them with
dispatch_tools (concurrently, results in order), then sends tool_result
blocks back in the next user message.
"""
from __future__ import annotations

import contextvars
import threading
import time
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
    *,
    timeout: float | None = None,
) -> ToolResult:
    """Run the tool named in tool_use against params, return a ToolResult.

    tool_use is a tool_use block from the Anthropic API:
      {"type": "tool_use", "id": "toolu_...", "name": "...", "input": {...}}

    With a timeout the tool runs in a worker thread; see dispatch_tools.
    """
    if timeout is None:
        return _execute(tools, tool_use)
    return dispatch_tools(tools, [tool_use], timeout=timeout)[0]


def dispatch_tools(
    tools: list[Tool],
    tool_uses: list[dict[str, Any]],
    *,
    timeout: float | None = None,
) -> list[ToolResult]:
    """Run every tool_use of one assistant message concurrently.

    Each call gets its own daemon thread, started in a copy of the caller's
    context (so context variables such as tracing spans carry over), and the
    tools must be thread-safe. Results come back in the order of
    `tool_uses`, which is the order the API expects the tool_result blocks.
    With a timeout, every call has `timeout` seconds from the common start,
    and one still running then becomes an is_error result. Python cannot
    stop a thread, so a timed-out call keeps running in the background,
    holding any lock it took; tools that hold the GIL in one long C call
    (e.g. a huge integer power) need their own limits.

    The calls must not depend on each other: the model issued them in one
    message without seeing any result, so this is the parallel tool use the
    Messages API describes.
    """
    slots: list[list[ToolResult]] = [[] for _ in tool_uses]
    workers = [
        threading.Thread(
            target=contextvars.copy_context().run,
            args=(_execute_into, slot, tools, use),
            daemon=True,
        )
        for use, slot in zip(tool_uses, slots)
    ]
    for worker in workers:
        worker.start()
    deadline = None if timeout is None else time.monotonic() + timeout
    results: list[ToolResult] = []
    for use, worker, slot in zip(tool_uses, workers, slots):
        worker.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        results.append(slot[0] if slot else _unfinished(use, timeout))
    return results


def _execute_into(slot: list[ToolResult], tools: list[Tool], tool_use: dict[str, Any]) -> None:
    slot.append(_execute(tools, tool_use))


def _unfinished(tool_use: dict[str, Any], timeout: float | None) -> ToolResult:
    name = tool_use.get("name", "")
    why = f"timed out after {timeout:g}s" if timeout is not None else "did not return a result"
    return ToolResult(tool_use_id=tool_use.get("id", ""), content=f"Tool {name} {why}", is_error=True)


def _execute(tools: list[Tool], tool_use: dict[str, Any]) -> ToolResult:
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
