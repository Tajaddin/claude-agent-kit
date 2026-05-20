"""Command-line entry point.

Usage:
    agent-kit "How many minutes are in three days?"

Requires ANTHROPIC_API_KEY. Wires the agent with the calculator, search,
and memory tools; lets you chat in a single turn from argv.
"""
from __future__ import annotations

import argparse
import sys

from claude_agent_kit.agent import Agent, AgentConfig
from claude_agent_kit.llm import AnthropicClient
from claude_agent_kit.memory import InMemoryStore
from claude_agent_kit.tools import (
    CalculatorTool,
    MemoryRecallTool,
    MemorySaveTool,
    MockSearchTool,
)

DEFAULT_SYSTEM = (
    "You are a careful research assistant. Use tools when needed. "
    "Use memory_save to remember durable facts about the user or task. "
    "Use memory_recall to look up earlier facts before answering. "
    "Be concise. Cite tool outputs when you use them."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="claude-agent-kit CLI demo.")
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true", help="Disable prompt caching for A/B.")
    args = parser.parse_args(argv)

    memory = InMemoryStore()
    tools = [
        CalculatorTool(),
        MockSearchTool(),
        MemorySaveTool(store=memory),
        MemoryRecallTool(store=memory),
    ]
    config = AgentConfig(
        system_blocks=[DEFAULT_SYSTEM],
        tools=tools,
        max_turn_iterations=args.max_iterations,
        enable_caching=not args.no_cache,
    )
    llm = AnthropicClient(model=args.model)
    agent = Agent(llm, config, memory=memory)

    turn = agent.run_turn(args.query)
    print(turn.assistant_text)
    print()
    print(f"--- {turn.iterations} iterations, {turn.elapsed_ms:.0f} ms ---")
    cost = agent.cache_stats.cost_summary()
    print(
        f"cache hit rate: {cost['input_tokens_cached_pct']}%, "
        f"savings: {cost['savings_pct']}% "
        f"(${cost['cost_with_cache_usd']:.4f} cached vs ${cost['cost_without_cache_usd']:.4f} uncached)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
