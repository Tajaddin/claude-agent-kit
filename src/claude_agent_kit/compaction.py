"""Conversation compaction.

When the message history exceeds the context budget, summarize the older
turns into a single compact synopsis. Keep the most recent N turns verbatim
because they are what the agent reasons about next.

The compactor here is deterministic and string-based; for production you
swap _make_summary for an LLM call (Claude Haiku 4.5 works well as a
compactor model).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class CompactionPolicy:
    """Triggers and limits for the compactor."""

    max_history_chars: int = 12000
    keep_recent_turns: int = 6
    summary_prefix: str = "[earlier conversation summary]"


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if isinstance(block, dict):
            total += len(block.get("text", "") or "")
            if "input" in block:
                total += len(str(block["input"]))
            total += _content_chars(block.get("content"))  # tool_result payload
    return total


def _char_count(messages: list[dict[str, Any]]) -> int:
    return sum(_content_chars(m.get("content", "")) for m in messages)


def _has_tool_result(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _default_summarize(turns: list[dict[str, Any]]) -> str:
    """Cheap deterministic summary: list each turn's role and first 120 chars.

    Replace this with an LLM call when running in production. The Agent
    accepts an optional summarize callable in its config so you do not need
    to fork this module.
    """
    parts: list[str] = []
    for m in turns:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(_block_summary(b) for b in content if isinstance(b, dict))
        snippet = str(content).replace("\n", " ").strip()[:120]
        parts.append(f"- {role}: {snippet}")
    return "\n".join(parts)


def _block_summary(block: dict[str, Any]) -> str:
    if block.get("type") == "tool_use":
        return f"[called {block.get('name', '?')}]"
    if block.get("type") == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
        return f"[tool result: {content}]"
    return str(block.get("text", ""))


def compact_history(
    messages: list[dict[str, Any]],
    *,
    policy: CompactionPolicy | None = None,
    summarize: Callable[[list[dict[str, Any]]], str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Compact the message list if it exceeds the policy budget.

    Returns (new_messages, did_compact). If not over budget, returns the
    input unchanged.

    Compaction rule: keep at least the most recent keep_recent_turns
    messages verbatim, summarize everything older into one synthetic user
    message that begins with policy.summary_prefix. The kept part never
    starts on a tool_result: the API rejects one whose tool_use is not in the
    message right before it, so the cut moves earlier to include that
    tool_use.
    """
    policy = policy or CompactionPolicy()
    summarize = summarize or _default_summarize

    if _char_count(messages) <= policy.max_history_chars:
        return messages, False

    keep = max(1, policy.keep_recent_turns)
    if len(messages) <= keep:
        # History is short in *number of turns* but huge in *char count*.
        # We cannot compact further without losing the current question.
        return messages, False

    split = len(messages) - keep
    while split > 0 and _has_tool_result(messages[split]):
        split -= 1
    if split == 0:
        return messages, False

    head = messages[:split]
    tail = messages[split:]
    summary = summarize(head)
    synthetic = {
        "role": "user",
        "content": f"{policy.summary_prefix}\n{summary}",
    }
    return [synthetic, *tail], True
