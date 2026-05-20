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


def _char_count(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", "")) + len(str(block.get("input", "")))
    return total


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
            content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
        snippet = str(content).replace("\n", " ").strip()[:120]
        parts.append(f"- {role}: {snippet}")
    return "\n".join(parts)


def compact_history(
    messages: list[dict[str, Any]],
    *,
    policy: CompactionPolicy | None = None,
    summarize: Callable[[list[dict[str, Any]]], str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Compact the message list if it exceeds the policy budget.

    Returns (new_messages, did_compact). If not over budget, returns the
    input unchanged.

    Compaction rule: keep the most recent keep_recent_turns user/assistant
    pairs verbatim, summarize everything older into one synthetic user
    message that begins with policy.summary_prefix.
    """
    policy = policy or CompactionPolicy()
    summarize = summarize or _default_summarize

    if _char_count(messages) <= policy.max_history_chars:
        return messages, False

    if len(messages) <= policy.keep_recent_turns:
        # History is short in *number of turns* but huge in *char count*.
        # We cannot compact further without losing the current question.
        return messages, False

    head = messages[: -policy.keep_recent_turns]
    tail = messages[-policy.keep_recent_turns :]
    summary = summarize(head)
    synthetic = {
        "role": "user",
        "content": f"{policy.summary_prefix}\n{summary}",
    }
    return [synthetic, *tail], True
