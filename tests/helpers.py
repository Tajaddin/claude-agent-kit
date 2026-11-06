"""Shared test helpers."""
from __future__ import annotations

from typing import Any


def _blocks(message: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def tool_pairing_errors(messages: list[dict[str, Any]]) -> list[str]:
    """Messages API rules for tool blocks, as a list of violations.

    Every tool_use must be answered by a tool_result in the next message, and
    every tool_result must answer a tool_use in the message right before it.
    """
    errors: list[str] = []
    if messages and messages[0].get("role") != "user":
        errors.append("history does not start with a user message")
    for i, message in enumerate(messages):
        use_ids = {b["id"] for b in _blocks(message, "tool_use")}
        if use_ids and i + 1 < len(messages):
            answered = {b["tool_use_id"] for b in _blocks(messages[i + 1], "tool_result")}
            for missing in sorted(use_ids - answered):
                errors.append(f"message {i}: tool_use {missing} has no tool_result after it")
        result_ids = {b["tool_use_id"] for b in _blocks(message, "tool_result")}
        if result_ids:
            before = {b["id"] for b in _blocks(messages[i - 1], "tool_use")} if i > 0 else set()
            for orphan in sorted(result_ids - before):
                errors.append(f"message {i}: tool_result {orphan} has no tool_use before it")
    return errors
