# claude-agent-kit

> Production patterns for agents on the Anthropic API. **Live Haiku 4.5: 60% cache hit rate, 12% cost reduction on a 10-turn conversation with a 2300-token prefix.** Mock projection at production prefix size (2500 tokens): 90% hit / 66.75% reduction. Prompt caching, memory tool, parallel tool execution, conversation compaction in one minimal package.

[![ci](https://github.com/Tajaddin/claude-agent-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Tajaddin/claude-agent-kit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](#testing)

## Hero metrics

Two benchmarks, both reproducible:
- `python -m benchmarks.cache_benchmark --mode mock` (2 seconds, no API key)
- `python -m benchmarks.cache_benchmark --mode live --turns 10` (40-60 seconds, needs ANTHROPIC_API_KEY, ~$0.05)

| Source | Cacheable prefix size | Cache hit rate | Input-token cost reduction |
|---|---:|---:|---:|
| **Live on Haiku 4.5** (2026-05-20) | ~2334 tokens | **60.26%** | **11.86%** |
| Mock projection (production-shape) | 2500 tokens | 90.0% | 66.75% |

Both numbers are real measurements; they differ because:
1. The mock counts only the most-recent user message as fresh input. The live API also adds assistant + tool-result blocks between turns to the fresh-input column. Each tool round trip adds ~50-300 tokens of fresh input the cache cannot serve.
2. With a longer system prompt (e.g. 9000 tokens), the live hit rate climbs to ~100% on every turn after the first (verified by the diagnostic in `benchmarks/results/cache_benchmark.json`).

Anthropic's prompt-cache floor for Claude Haiku 4.5 is **2048 input tokens** in the cached prefix. Below that, `cache_control` is silently ignored. The padded `FEW_SHOT` in `benchmarks/cache_benchmark.py` crosses that floor by design; if your production agent's system prompt is shorter, expand it (with rules, examples, or a few-shot block) before enabling caching.

The cost number is what your finance team sees on the invoice.

## What you get

Five small modules. Each one solves a problem the Anthropic Python SDK leaves to you:

| Module | Problem solved |
|---|---|
| `cache.py` | Where to place `cache_control` breakpoints (system, tools, history) and how to read back `cache_creation` / `cache_read` usage from the API response. |
| `memory.py` | In-memory and SQLite backends for the agent's `memory_save` / `memory_recall` tools. Persistent across sessions. |
| `compaction.py` | When the message history exceeds a char budget, summarize older turns and keep the recent ones. Pluggable summarizer. |
| `agent.py` | The tool-use loop. Reads `tool_use` blocks, dispatches them, sends `tool_result` blocks back. Stops on `end_turn` or a hard iteration cap. |
| `llm.py` | `AnthropicClient` and a `MockClient` that simulates cache accounting with the same prefix-fingerprint logic the API uses. |

## Why this matters for production

Recruiter signal this maps to:
- **Anthropic SDK production usage** (Caylent, M3 USA, ERP Suites, Pair Team, forgd.ai)
- **Prompt caching** (mentioned in 6+ Senior AI Engineer JDs in 2026 for cost-sensitive workloads)
- **Memory + tool use** (Sekai, Mango Languages, Cohere Agent Infrastructure)
- **Conversation compaction** (every long-context agent JD eventually)

The Anthropic SDK gives you a Messages API. This kit shows you what to put *into* that API.

## Quick start

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
agent-kit "What is 17 times 23, then remember the answer as 'math.last_result'?"
```

The CLI prints the assistant's reply, the iteration count, the cache hit rate, and the realized cost vs hypothetical-without-cache cost.

## Programmatic use

```python
from claude_agent_kit import Agent, AgentConfig
from claude_agent_kit.llm import AnthropicClient
from claude_agent_kit.memory import SqliteMemoryStore
from claude_agent_kit.tools import (
    CalculatorTool, MemoryRecallTool, MemorySaveTool, MockSearchTool,
)

memory = SqliteMemoryStore("./agent.db")
tools = [
    CalculatorTool(),
    MockSearchTool(),
    MemorySaveTool(store=memory),
    MemoryRecallTool(store=memory),
]

config = AgentConfig(
    system_blocks=[
        "You are a careful research assistant. Use tools when needed.",
        # Few-shot example block, cached separately from the system rules.
        "Example: If asked '2+2', call calculator(expression='2+2') first.",
    ],
    tools=tools,
    enable_caching=True,
)

agent = Agent(AnthropicClient(), config, memory=memory)
turn = agent.run_turn("What is 17 times 23?")
print(turn.assistant_text)
print(agent.cache_stats.cost_summary())
```

## Prompt-caching breakpoint strategy

This kit enforces a 4-breakpoint pattern that survives both single-turn and multi-turn workloads:

```
[1] System block: persona, rules, output format            [cache_control]
[2] Tool definitions: schemas, descriptions                [cache_control on last tool]
[3] Few-shot examples: canonical demos                      [cache_control, optional]
[4] Conversation history up to the last user turn          [cache_control on last assistant turn]
    Current user turn                                       (no cache_control, always fresh)
```

The last user turn must never be cached. Anthropic treats a cache_control mark on a non-final message as "everything up to and including this is cacheable". If you mark the current user turn, the model has nothing fresh to read.

`cache.build_cached_system()` handles [1] + [3]. `cache.build_cached_tools()` handles [2]. The agent loop handles the history side automatically.

## Memory tool

```python
from claude_agent_kit.memory import SqliteMemoryStore
from claude_agent_kit.tools import MemorySaveTool, MemoryRecallTool

store = SqliteMemoryStore("./agent.db")  # persists across processes
tools = [MemorySaveTool(store=store), MemoryRecallTool(store=store)]
```

The agent's system prompt should tell Claude to use stable keys (`user.preference.coffee`, `task.deadline`) and to recall before answering. The `memory_recall` tool supports prefix search with a trailing `*`.

## Conversation compaction

When `history_chars > max_history_chars`, `compact_history` summarizes the oldest turns into one synthetic `[earlier conversation summary]` user message and keeps the last N turns verbatim. The summarizer is pluggable; for production you pass a callable that calls Haiku 4.5 with a "summarize this conversation history" prompt.

```python
from claude_agent_kit.compaction import CompactionPolicy

config = AgentConfig(
    system_blocks=[...],
    tools=[...],
    compaction_policy=CompactionPolicy(
        max_history_chars=12000,
        keep_recent_turns=6,
    ),
)
```

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=claude_agent_kit
```

46 tests, 85% coverage. Key tests:

- `test_cache.py::test_cost_summary_reflects_savings` — proves the cost math (>50% savings under production-shape input)
- `test_agent.py::test_tool_use_loop` — proves the tool-call round trip handles `tool_use` -> `tool_result` -> final text correctly
- `test_agent.py::test_max_iterations_safety` — proves the agent bails out of runaway tool loops
- `test_compaction.py::test_compacts_when_over_budget` — proves the summarizer triggers under load
- `test_benchmark.py` — smoke-runs the full cache benchmark to keep the hero number honest

## Docker

```bash
docker compose up bench               # runs the mock-mode cache benchmark
docker compose run --rm agent-kit "your question"
```

## Project layout

```
src/claude_agent_kit/
  state-y modules:
    cache.py            # cache_control helpers + CacheStats
    memory.py           # InMemoryStore + SqliteMemoryStore
    compaction.py       # history shrinker
  runtime:
    llm.py              # AnthropicClient + MockClient
    agent.py            # Agent + AgentConfig + run_turn / run_conversation
    cli.py              # `agent-kit` CLI
  tools/
    base.py             # Tool protocol + dispatch_tool
    calculator.py       # safe AST-based arithmetic
    search.py           # offline corpus search
    memory_io.py        # memory_save + memory_recall

benchmarks/
  cache_benchmark.py    # 10-turn run, cache on vs off
  results/
    cache_benchmark.json
```

## License

MIT
