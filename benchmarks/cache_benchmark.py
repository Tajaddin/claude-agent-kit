"""Cache hit rate benchmark.

Runs a 10-turn conversation against the Agent. The system prompt + tool
definitions + a few-shot example block stay constant across turns, so they
should hit the cache from turn 2 onward.

Two modes:
- live: hits real Anthropic API. Needs ANTHROPIC_API_KEY. Costs ~$0.03 in
  Haiku 4.5 tokens.
- mock: uses MockClient which simulates cache accounting with the same
  prefix-fingerprint logic real Anthropic exposes. Free, deterministic.

Hero number: input-token cost reduction with caching vs without. Theoretical
maximum on a 10-turn conversation with a ~200-token prefix is ~85% savings
(turn 1 pays full prefix as cache_creation; turns 2-10 pay ~10% per read).

Usage:
    python -m benchmarks.cache_benchmark --mode mock
    python -m benchmarks.cache_benchmark --mode live --turns 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from claude_agent_kit.agent import Agent, AgentConfig
from claude_agent_kit.llm import AnthropicClient, LLMClient, MockClient
from claude_agent_kit.memory import InMemoryStore
from claude_agent_kit.tools import (
    CalculatorTool,
    MemoryRecallTool,
    MemorySaveTool,
    MockSearchTool,
)

RESULTS_DIR = Path(__file__).parent / "results"

USER_TURNS = [
    "What is 17 times 23?",
    "Save that as 'math.last_result'.",
    "Search for prompt caching.",
    "What did you find?",
    "Save the key takeaway as 'note.cache_takeaway'.",
    "What is 99 plus 1?",
    "What is the result times 5?",
    "Recall math.last_result.",
    "Recall everything starting with note.*",
    "What was the very first calculation we did this conversation?",
]

DEFAULT_SYSTEM = (
    "You are a careful research assistant working on a long-running task. "
    "Use the provided tools whenever the user asks for arithmetic, search, "
    "or persistent memory. Be concise (1-3 sentences per answer). When you "
    "use a tool, cite the tool name in parentheses at the end of your reply."
)

FEW_SHOT = (
    "Example interactions you should follow:\n"
    "  User: What is 4 plus 5?\n"
    "  Assistant: (calls calculator with expression='4+5', gets '9') The answer is 9. (calculator)\n"
    "  User: Save my favorite color as blue.\n"
    "  Assistant: (calls memory_save key='user.favorite_color', value='blue') "
    "Saved your favorite color. (memory_save)\n"
)


def _build_agent(llm: LLMClient, enable_caching: bool) -> Agent:
    memory = InMemoryStore()
    tools = [
        CalculatorTool(),
        MockSearchTool(),
        MemorySaveTool(store=memory),
        MemoryRecallTool(store=memory),
    ]
    config = AgentConfig(
        system_blocks=[DEFAULT_SYSTEM, FEW_SHOT],
        tools=tools,
        enable_caching=enable_caching,
        max_turn_iterations=4,
    )
    return Agent(llm, config, memory=memory)


def _make_mock_client() -> MockClient:
    """Pre-load script entries so the mock returns reasonable text per turn."""
    script: list = []
    for prompt_key in (
        "What is 17 times 23?",
        "Save that",
        "Search for prompt caching",
        "What did you find?",
        "Save the key takeaway",
        "What is 99 plus 1?",
        "What is the result times 5?",
        "Recall math.last_result",
        "Recall everything starting",
        "very first calculation",
    ):
        script.append((prompt_key, [{"type": "text", "text": "(mock answer)"}]))
    # Defaults match a production-shape agent: system prompt + tool defs + a
    # few-shot example block adds up to ~2500 tokens cacheable; per-turn user
    # content adds ~40 tokens; replies average ~80 output tokens.
    return MockClient(
        script=script,
        default_text="(mock answer)",
        cacheable_prefix_tokens=2500,
        fresh_tokens_per_message=40,
        output_tokens=80,
    )


def run_mode(mode: str, turns: list[str], enable_caching: bool) -> dict:
    if mode == "live":
        llm: LLMClient = AnthropicClient()
    else:
        llm = _make_mock_client()
    agent = _build_agent(llm, enable_caching=enable_caching)
    t0 = time.perf_counter()
    results = agent.run_conversation(turns)
    elapsed = (time.perf_counter() - t0) * 1000

    cost = agent.cache_stats.cost_summary()
    return {
        "mode": mode,
        "caching": enable_caching,
        "n_turns": len(results),
        "n_api_requests": agent.cache_stats.n_requests,
        "elapsed_ms": elapsed,
        "cache_creation_input_tokens": agent.cache_stats.cache_creation_input_tokens,
        "cache_read_input_tokens": agent.cache_stats.cache_read_input_tokens,
        "fresh_input_tokens": agent.cache_stats.fresh_input_tokens,
        "output_tokens": agent.cache_stats.output_tokens,
        "hit_rate_pct": round(agent.cache_stats.hit_rate * 100, 2),
        "cost_with_cache_usd": cost["cost_with_cache_usd"],
        "cost_without_cache_usd": cost["cost_without_cache_usd"],
        "savings_pct": cost["savings_pct"],
        "input_tokens_cached_pct": cost["input_tokens_cached_pct"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache hit rate benchmark for claude-agent-kit.")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--turns", type=int, default=10, help="Number of user turns (max 10).")
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "cache_benchmark.json",
    )
    args = parser.parse_args(argv)

    turns = USER_TURNS[: max(1, min(args.turns, len(USER_TURNS)))]

    print(f"Running {args.mode} mode on {len(turns)} turns, caching ON...", file=sys.stderr)
    on = run_mode(args.mode, turns, enable_caching=True)
    print(f"Running {args.mode} mode on {len(turns)} turns, caching OFF (control)...", file=sys.stderr)
    off = run_mode(args.mode, turns, enable_caching=False)

    summary = {
        "mode": args.mode,
        "caching_on": on,
        "caching_off": off,
        "headline": {
            "cache_hit_rate_pct": on["hit_rate_pct"],
            "input_token_cost_reduction_pct": on["savings_pct"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n=== Cache benchmark summary ===")
    print(json.dumps(summary["headline"], indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
