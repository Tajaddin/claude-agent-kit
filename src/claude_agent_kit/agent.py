"""Agent loop.

One Agent instance binds:
- An LLMClient (Anthropic or Mock)
- A list of tools
- A memory store (optional)
- A cache config (system + cached tools, set once)
- A compaction policy

The agent exposes two entry points:
- run_turn(user_text): one user message in, final assistant text out, with
  any number of tool-use round trips in between.
- run_conversation(turns): convenience for batch eval; iterates run_turn
  for a scripted list of user turns and returns per-turn stats.

The loop is intentionally simple: read assistant content, run every
tool_use block of that message concurrently (or one at a time, with
parallel_tools=False), append the tool_result blocks
(in tool_use order) back as a single user message, loop until
stop_reason != "tool_use" or the budget runs out. A reply cut off by
max_tokens in the middle of a tool call gets error tool_results instead of
running the call, and the loop continues.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from claude_agent_kit.cache import CacheStats, build_cached_system, build_cached_tools
from claude_agent_kit.compaction import CompactionPolicy, compact_history
from claude_agent_kit.llm import LLMClient
from claude_agent_kit.memory import InMemoryStore, MemoryStore
from claude_agent_kit.tools.base import (
    Tool,
    ToolResult,
    dispatch_tools,
    tools_as_anthropic_schemas,
)


@dataclass
class AgentConfig:
    """Static configuration for one Agent instance."""

    system_blocks: list[str]
    tools: list[Tool]
    max_turn_iterations: int = 10
    max_tokens: int = 1024
    temperature: float = 0.0
    enable_caching: bool = True
    compaction_policy: CompactionPolicy = field(default_factory=CompactionPolicy)
    # A tool call still running after this many seconds is answered with an
    # is_error tool_result. None waits forever.
    tool_timeout_s: float | None = 30.0
    # Run the tool calls of one response concurrently, one thread each, so
    # tools must be thread-safe. False runs them one at a time, and with
    # tool_timeout_s=None on the calling thread.
    parallel_tools: bool = True


@dataclass
class AgentTurn:
    """One user-visible turn: input + final assistant text + observed events."""

    user_text: str
    assistant_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    elapsed_ms: float = 0.0
    compacted: bool = False


class Agent:
    """Stateful agent. Owns one conversation across many user turns."""

    def __init__(
        self,
        llm: LLMClient,
        config: AgentConfig,
        memory: MemoryStore | None = None,
    ):
        self.llm = llm
        self.config = config
        self.memory = memory or InMemoryStore()
        self.history: list[dict[str, Any]] = []
        self.cache_stats = CacheStats()
        self._system_payload = self._build_system_payload()
        self._tools_payload = self._build_tools_payload()

    # --- public API --------------------------------------------------------

    def reset(self) -> None:
        self.history = []
        self.cache_stats = CacheStats()

    def run_turn(self, user_text: str) -> AgentTurn:
        start = time.perf_counter()
        self.history.append({"role": "user", "content": user_text})
        compacted = self._maybe_compact()

        turn = AgentTurn(user_text=user_text, assistant_text="", compacted=compacted)
        for i in range(self.config.max_turn_iterations):
            turn.iterations = i + 1
            response = self.llm.respond(
                system=self._system_payload,
                messages=self.history,
                tools=self._tools_payload,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            self.cache_stats.record(response.usage)
            self.history.append({"role": "assistant", "content": response.content})

            tool_uses = [
                b for b in response.content if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if response.stop_reason == "tool_use":
                tool_results = self._run_tools(tool_uses)
            elif response.stop_reason == "max_tokens" and tool_uses:
                # The reply ran out of tokens inside a tool call, so its input
                # may be incomplete. Do not run it, but answer it: the next
                # request is rejected if a tool_use has no tool_result.
                tool_results = [_cut_off(b) for b in tool_uses]
            else:
                turn.assistant_text = self._extract_text(response.content)
                break

            turn.tool_calls.extend(tool_uses)
            turn.tool_results.extend(tr.as_block() for tr in tool_results)
            self.history.append(
                {"role": "user", "content": [tr.as_block() for tr in tool_results]}
            )
        else:
            turn.assistant_text = "[agent stopped: hit max_turn_iterations]"

        turn.elapsed_ms = (time.perf_counter() - start) * 1000
        return turn

    def run_conversation(self, user_turns: list[str]) -> list[AgentTurn]:
        return [self.run_turn(u) for u in user_turns]

    # --- internals ---------------------------------------------------------

    def _build_system_payload(self) -> list[dict[str, Any]] | str | None:
        if not self.config.system_blocks:
            return None
        if self.config.enable_caching:
            return build_cached_system(*self.config.system_blocks)
        return [{"type": "text", "text": b} for b in self.config.system_blocks]

    def _build_tools_payload(self) -> list[dict[str, Any]] | None:
        if not self.config.tools:
            return None
        schemas = tools_as_anthropic_schemas(self.config.tools)
        if self.config.enable_caching:
            return build_cached_tools(schemas)
        return schemas

    def _maybe_compact(self) -> bool:
        new_history, did_compact = compact_history(
            self.history, policy=self.config.compaction_policy
        )
        if did_compact:
            self.history = new_history
        return did_compact

    def _run_tools(self, tool_uses: list[dict[str, Any]]) -> list[ToolResult]:
        return dispatch_tools(
            self.config.tools,
            tool_uses,
            timeout=self.config.tool_timeout_s,
            parallel=self.config.parallel_tools,
        )

    @staticmethod
    def _extract_text(content: list[dict[str, Any]]) -> str:
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()


def _cut_off(tool_use: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_use_id=tool_use.get("id", ""),
        content=(
            "Not run: the reply hit max_tokens before this tool call was complete. "
            "Retry with a shorter call."
        ),
        is_error=True,
    )
