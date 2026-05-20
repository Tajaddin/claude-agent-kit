"""Prompt caching helpers.

Anthropic's prompt caching lets you flag system blocks, tool definitions,
and few-shot examples with `cache_control: {"type": "ephemeral"}`. On a
cache hit, those tokens cost ~10% of the original. Up to 4 cache breakpoints
per request; the cache lives 5 minutes by default.

The helpers here build cache-marked block lists from plain text. They are
small on purpose: the value is in *where* you place the breakpoints, not in
hiding the API behind an abstraction.

Breakpoint strategy this module enforces (proven on multi-turn agents):

    System block (stable persona + rules)             [cached]
    Tool definitions (stable schemas)                 [cached]
    Few-shot examples (stable canonical demos)        [cached, 5min TTL]
    Conversation history up to the last user turn     [cached, last block]
    Current user turn                                 [NOT cached]

The last user turn must never be cached. If you cache it, Anthropic counts
it as a fresh prefix and you pay full price for everything that follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheBlock:
    """One Anthropic content block flagged for caching.

    Use .as_dict() to get the JSON payload to pass to the Messages API.
    """

    text: str
    cache_type: str = "ephemeral"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "text",
            "text": self.text,
            "cache_control": {"type": self.cache_type},
        }


def build_cached_system(*blocks: str | CacheBlock) -> list[dict[str, Any]]:
    """Build a cached system payload from one or more text blocks.

    Pass strings for blocks you want cached with default ephemeral type, or
    CacheBlock objects when you need to override. Returns the system payload
    that Anthropic's Messages API expects.
    """
    out: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, CacheBlock):
            out.append(block.as_dict())
        else:
            out.append({"type": "text", "text": block, "cache_control": {"type": "ephemeral"}})
    return out


def build_cached_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the *last* tool definition with cache_control so the entire tool
    block becomes cacheable.

    Anthropic caches everything up to and including the block with
    cache_control set. Marking only the last tool definition is enough; you
    do not need to mark each one individually.
    """
    if not tools:
        return tools
    out = [dict(t) for t in tools]
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def estimate_cache_savings(
    cached_input_tokens: int,
    fresh_input_tokens: int,
    output_tokens: int,
    *,
    base_input_per_mtok: float = 1.00,
    cache_read_per_mtok: float = 0.10,
    output_per_mtok: float = 5.00,
) -> dict[str, float]:
    """Estimate cost with and without caching.

    Defaults match Claude Haiku 4.5 ($1/MTok input, $5/MTok output, cache
    read = 10% of input). Pass other rates for Sonnet/Opus.

    Returns a dict with `cost_with_cache`, `cost_without_cache`, and
    `savings_pct`. All in USD.
    """
    total_input = cached_input_tokens + fresh_input_tokens
    cost_without = (total_input * base_input_per_mtok + output_tokens * output_per_mtok) / 1_000_000
    cost_with = (
        cached_input_tokens * cache_read_per_mtok
        + fresh_input_tokens * base_input_per_mtok
        + output_tokens * output_per_mtok
    ) / 1_000_000
    if cost_without == 0:
        savings = 0.0
    else:
        savings = (cost_without - cost_with) / cost_without * 100
    return {
        "cost_with_cache": round(cost_with, 6),
        "cost_without_cache": round(cost_without, 6),
        "savings_pct": round(savings, 2),
    }


@dataclass
class CacheStats:
    """Aggregated cache usage across one or more API calls."""

    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    fresh_input_tokens: int = 0
    output_tokens: int = 0
    n_requests: int = 0
    per_request: list[dict[str, int]] = field(default_factory=list)

    def record(self, usage: dict[str, int]) -> None:
        """Record one API response's usage block."""
        self.n_requests += 1
        cc = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cr = int(usage.get("cache_read_input_tokens", 0) or 0)
        fresh = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        self.cache_creation_input_tokens += cc
        self.cache_read_input_tokens += cr
        self.fresh_input_tokens += fresh
        self.output_tokens += out
        self.per_request.append(
            {
                "cache_creation_input_tokens": cc,
                "cache_read_input_tokens": cr,
                "input_tokens": fresh,
                "output_tokens": out,
            }
        )

    @property
    def hit_rate(self) -> float:
        """Cache-read tokens divided by total cacheable input tokens.

        Total cacheable = cache_creation + cache_read (since cache_creation
        is the FIRST-time-seen-this-prefix path, and cache_read is the
        hit path). Fresh input is unrelated (per-turn user content).
        """
        cacheable = self.cache_creation_input_tokens + self.cache_read_input_tokens
        if cacheable == 0:
            return 0.0
        return self.cache_read_input_tokens / cacheable

    def cost_summary(
        self,
        *,
        base_input_per_mtok: float = 1.00,
        cache_read_per_mtok: float = 0.10,
        cache_write_per_mtok: float = 1.25,
        output_per_mtok: float = 5.00,
    ) -> dict[str, float]:
        """Compute realized cost with caching vs hypothetical without."""
        cost_with = (
            self.cache_creation_input_tokens * cache_write_per_mtok
            + self.cache_read_input_tokens * cache_read_per_mtok
            + self.fresh_input_tokens * base_input_per_mtok
            + self.output_tokens * output_per_mtok
        ) / 1_000_000
        # Without caching every cached read would have been a full input.
        hypothetical_input = (
            self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.fresh_input_tokens
        )
        cost_without = (
            hypothetical_input * base_input_per_mtok + self.output_tokens * output_per_mtok
        ) / 1_000_000
        if cost_without == 0:
            savings = 0.0
        else:
            savings = (cost_without - cost_with) / cost_without * 100
        return {
            "cost_with_cache_usd": round(cost_with, 6),
            "cost_without_cache_usd": round(cost_without, 6),
            "savings_pct": round(savings, 2),
            "input_tokens_cached_pct": round(self.hit_rate * 100, 2),
        }
