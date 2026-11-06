"""LLM client wrappers.

Two backends:
- AnthropicClient: real Messages API with cache-control support.
- MockClient: deterministic stub that simulates cache hits/misses based on
  whether the system+tools prefix matches a remembered prefix.

Both surface the same `respond(...)` shape so the Agent code does not
branch on the backend.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    """Normalized response shape across backends."""

    content: list[dict[str, Any]]  # Anthropic-style content blocks
    stop_reason: str
    usage: dict[str, int]
    model: str = ""


class LLMClient(Protocol):
    def respond(
        self,
        *,
        system: list[dict[str, Any]] | str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class AnthropicClient:
    """Real Anthropic Messages API client.

    Enables prompt caching automatically when `system` blocks or `tools`
    include cache_control entries. Use cache.build_cached_system and
    cache.build_cached_tools to set those up.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. pip install anthropic, or use MockClient."
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Use MockClient in tests.")
        # The SDK default is 10 minutes per request.
        self._client = Anthropic(api_key=key, timeout=timeout)
        self.model = model

    def respond(
        self,
        *,
        system: list[dict[str, Any]] | str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)
        content_blocks: list[dict[str, Any]] = []
        for block in resp.content:
            # SDK gives typed objects; convert to plain dicts for consistency.
            if hasattr(block, "model_dump"):
                content_blocks.append(block.model_dump())
            else:
                content_blocks.append(dict(block))

        usage_dict = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
        return LLMResponse(
            content=content_blocks,
            stop_reason=resp.stop_reason or "",
            usage=usage_dict,
            model=resp.model,
        )


def _fingerprint(blocks: Any) -> str:
    """Stable fingerprint of a system/tools structure, used by MockClient."""
    if blocks is None:
        return "none"
    return hashlib.sha256(json.dumps(blocks, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class MockClient:
    """Deterministic LLM stub with simulated cache accounting.

    Behavior:
    - The first time we see (system, tools) fingerprint X, the response
      reports cache_creation_input_tokens on the cacheable prefix.
    - Subsequent calls with the same fingerprint report
      cache_read_input_tokens instead. This mirrors Anthropic's API.

    The response content is taken from `script` which maps a substring of
    the latest user message to a list of content blocks (the assistant's
    reply). Default is a single empty text block.
    """

    script: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)
    default_text: str = ""
    call_count: int = 0
    last_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    last_system_fp: str | None = None
    seen_prefixes: set[str] = field(default_factory=set)

    # Approximate token counters so cache stats look realistic. Defaults match a
    # production system prompt + tool definitions + a small few-shot example
    # block (around 3K cached tokens) with light per-turn user content.
    cacheable_prefix_tokens: int = 3000
    fresh_tokens_per_message: int = 40
    output_tokens: int = 80

    @staticmethod
    def _has_cache_control(payload: Any) -> bool:
        """True if any element under payload has a cache_control entry."""
        if payload is None:
            return False
        if isinstance(payload, list):
            return any(MockClient._has_cache_control(p) for p in payload)
        if isinstance(payload, dict):
            if "cache_control" in payload:
                return True
            return any(MockClient._has_cache_control(v) for v in payload.values())
        return False

    def respond(
        self,
        *,
        system: list[dict[str, Any]] | str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages.append(list(messages))

        prefix_fp = _fingerprint([system, tools])
        self.last_system_fp = prefix_fp
        is_hit = prefix_fp in self.seen_prefixes
        self.seen_prefixes.add(prefix_fp)

        # Build content from the script. Matching considers both plain text and
        # tool_result content from the latest message; this lets tests script a
        # response keyed on either the user's question or the tool output.
        latest_user_text = ""
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                latest_user_text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if "text" in b:
                        parts.append(str(b.get("text", "")))
                    if b.get("type") == "tool_result":
                        parts.append(str(b.get("content", "")))
                        parts.append("tool_result")
                latest_user_text = " ".join(parts)

        chosen_content: list[dict[str, Any]] | None = None
        for key, blocks in self.script:
            if key in latest_user_text:
                chosen_content = blocks
                break
        if chosen_content is None:
            chosen_content = [{"type": "text", "text": self.default_text}]

        # When caching is enabled (any cache_control entries on system/tools),
        # only the most recent message counts as fresh input. Everything older
        # is either part of the cached prefix or would itself be marked cached
        # in a production setup. When caching is disabled, every message counts
        # as fresh input on every call.
        caching_enabled = self._has_cache_control(system) or self._has_cache_control(tools)
        if caching_enabled:
            fresh_msgs = 1
        else:
            fresh_msgs = max(1, len(messages))

        usage = {
            "input_tokens": self.fresh_tokens_per_message * fresh_msgs,
            "cache_read_input_tokens": self.cacheable_prefix_tokens if (caching_enabled and is_hit) else 0,
            "cache_creation_input_tokens": (
                self.cacheable_prefix_tokens if (caching_enabled and not is_hit) else 0
            ),
            "output_tokens": self.output_tokens,
        }

        # If any content block is a tool_use, stop reason is tool_use.
        stop_reason = "end_turn"
        for block in chosen_content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                stop_reason = "tool_use"
                break

        return LLMResponse(
            content=chosen_content,
            stop_reason=stop_reason,
            usage=usage,
            model="mock-claude",
        )
