"""Mock web search tool.

Returns canned snippets from a small corpus so the benchmark stays
deterministic and offline. Real deployments swap this for Tavily, Brave
Search, or the Anthropic web_search tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_DEFAULT_CORPUS: dict[str, list[str]] = {
    "anthropic": [
        "Anthropic is an AI safety company headquartered in San Francisco, "
        "California. Its main product is the Claude family of large language "
        "models.",
    ],
    "claude haiku 4.5": [
        "Claude Haiku 4.5 is a smaller, faster model in the Claude 4 family. "
        "It is optimized for low latency and supports prompt caching, tool "
        "use, and structured outputs.",
    ],
    "prompt caching": [
        "Prompt caching reduces input token cost by reusing a cached prefix "
        "of the request. On Claude, cache reads cost about 10% of normal "
        "input tokens. Up to 4 cache breakpoints per request.",
    ],
}


@dataclass
class MockSearchTool:
    name: str = "search"
    description: str = (
        "Search a small offline corpus for snippets matching the query. "
        "Returns up to 3 snippets concatenated with double newlines. Used "
        "for demos and benchmarks; not for production."
    )
    input_schema: dict[str, Any] | None = None
    corpus: dict[str, list[str]] = field(default_factory=lambda: dict(_DEFAULT_CORPUS))

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Case-insensitive substring match on corpus keys.",
                    }
                },
                "required": ["query"],
            }

    def execute(self, params: dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip().lower()
        if not query:
            return "error: query is required"

        hits: list[str] = []
        for key, snippets in self.corpus.items():
            if key in query or query in key:
                hits.extend(snippets)
        if not hits:
            return "no results"
        return "\n\n".join(hits[:3])
