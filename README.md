# claude-agent-kit

> Production patterns for agents on the Anthropic API. **Live Haiku 4.5: 60% cache hit rate on the cached prefix, 12% cost reduction on a 10-turn conversation with a 2300-token prefix.** Prompt caching, memory tool, parallel tool execution, conversation compaction in one minimal package.

[![ci](https://github.com/Tajaddin/claude-agent-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Tajaddin/claude-agent-kit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](#testing)

## Hero metrics

Two modes of one benchmark:
- `python -m benchmarks.cache_benchmark --mode live --turns 10` (needs ANTHROPIC_API_KEY; the committed run took 22 s with caching and 86 s without, about $0.15 for both passes at list prices)
- `python -m benchmarks.cache_benchmark --mode mock` (no API key, 3 s locally; a simulation, see below; writes `benchmarks/results/cache_benchmark-mock.json`, not committed)

| Source | Cacheable prefix size | Cache hit rate | Input-token cost reduction |
|---|---:|---:|---:|
| **Live on Haiku 4.5** (2026-05-20) | ~2334 tokens | **60.26%** | **11.86%** |
| Mock simulation (not a measurement) | 2500 tokens | 90.0% | 66.75% |

Only the live row is a measurement; its raw usage is in `benchmarks/results/cache_benchmark.json`. Cache hit rate here is cache reads divided by cache reads plus cache writes, so it covers only the cached prefix: of all input tokens in the live run, cache reads were 17% (13,182 of 76,928), and 72% (55,051) were uncached.

The mock row is a simulation. `MockClient` bills only the newest message as fresh input, which assumes the earlier conversation is served from cache. The agent loop does not request that: it sets `cache_control` on the system blocks and tools only, so every request re-sends the history, assistant replies and tool results as uncached input. That is why the live numbers are lower.

Each model has a minimum cacheable prefix length (see Anthropic's prompt-caching docs); below it, `cache_control` is silently ignored. The padded `FEW_SHOT` in `benchmarks/cache_benchmark.py` is there to reach it; if your production agent's system prompt is shorter, expand it (with rules, examples, or a few-shot block) before enabling caching.

The cost number is what your finance team sees on the invoice.

## What you get

Five small modules. Each one solves a problem the Anthropic Python SDK leaves to you:

| Module | Problem solved |
|---|---|
| `cache.py` | `cache_control` breakpoints for the system blocks and tools, and how to read back `cache_creation` / `cache_read` usage from the API response. |
| `memory.py` | In-memory and SQLite backends for the agent's `memory_save` / `memory_recall` tools. Persistent across sessions. |
| `compaction.py` | When the message history (tool results included) exceeds a char budget, summarize older turns and keep the recent ones, never separating a `tool_result` from its `tool_use`. Pluggable summarizer. |
| `agent.py` | The tool-use loop. Runs the `tool_use` blocks of one response concurrently (one thread each, per-tool timeout, results in call order) and sends the `tool_result` blocks back. Stops on `end_turn` or a hard iteration cap. |
| `llm.py` | `AnthropicClient` and a `MockClient` that simulates cache accounting from a fingerprint of the system blocks and tools. |

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

The agent sets these breakpoints:

```
[1] System block: persona, rules, output format            [cache_control]
[2] Tool definitions: schemas, descriptions                [cache_control on last tool]
[3] Few-shot examples: canonical demos                      [cache_control, optional]
    Conversation history                                    (not cached)
```

`cache.build_cached_system()` handles [1] + [3]. `cache.build_cached_tools()` handles [2]. The agent loop does not cache the conversation history, so each request pays full input price for it; the live benchmark above shows the effect. Anthropic's docs cache a multi-turn conversation by marking the last content block of the latest turn on every request.

## Tool execution

The tool calls of one response run concurrently, one thread each, in a copy of the caller's context, so context variables such as tracing spans carry over; results go back in call order. Each call gets `tool_timeout_s` (default 30 s) from the common start, and one still running then is answered with an `is_error` result. Python cannot stop a thread, so a timed-out call keeps running and keeps any lock it holds: a stuck `SqliteMemoryStore` call makes later memory calls time out too, and `close()` waits for it.

Tools must therefore be thread-safe. For tools that are not (a default `sqlite3` connection, a client bound to one thread), set `AgentConfig(parallel_tools=False, tool_timeout_s=None)`: the calls then run one at a time on the calling thread. `parallel_tools=False` with a timeout runs them one at a time, each in a worker thread.

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

87 tests, 88% coverage. Key tests:

- `test_cache.py::test_cost_summary_reflects_savings` — proves the cost math (>50% savings under production-shape input)
- `test_agent.py::test_tool_use_loop` — proves the tool-call round trip handles `tool_use` -> `tool_result` -> final text correctly
- `test_agent.py::test_max_iterations_safety` — proves the agent bails out of runaway tool loops
- `test_compaction.py::test_compacts_when_over_budget` — proves the summarizer triggers under load
- `test_compaction.py::TestToolPairs::test_fuzz_compaction_keeps_every_tool_result_with_its_tool_use` — 3,000 random histories; every compacted result passes the API's tool pairing rules
- `test_agent.py::TestParallelToolCalls` — tool calls in one response meet at a barrier (they only pass if they run at the same time), and results keep call order
- `test_agent.py::TestMaxTokensStop` — a reply cut off inside a tool call is answered with an error result, not run and not left unanswered
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
    cache_benchmark.json  # the live run's record
```

## License

MIT
