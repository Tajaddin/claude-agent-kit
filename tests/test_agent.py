"""Agent loop tests with MockClient."""
from __future__ import annotations

import threading
import time

from claude_agent_kit.agent import Agent, AgentConfig
from claude_agent_kit.llm import MockClient
from claude_agent_kit.memory import InMemoryStore, SqliteMemoryStore
from claude_agent_kit.tools import (
    CalculatorTool,
    MemoryRecallTool,
    MemorySaveTool,
    MockSearchTool,
)
from tests.helpers import tool_pairing_errors


def _basic_agent(script=None, default_text="ok"):
    memory = InMemoryStore()
    tools = [CalculatorTool(), MockSearchTool(), MemorySaveTool(store=memory), MemoryRecallTool(store=memory)]
    config = AgentConfig(
        system_blocks=["you are an agent"],
        tools=tools,
        max_turn_iterations=4,
    )
    llm = MockClient(script=script or [], default_text=default_text)
    return Agent(llm, config, memory=memory), llm


class TestAgentBasic:
    def test_single_turn_text_reply(self):
        agent, _ = _basic_agent(default_text="hello back")
        turn = agent.run_turn("hi")
        assert turn.assistant_text == "hello back"
        assert turn.iterations == 1
        assert agent.cache_stats.n_requests == 1

    def test_tool_use_loop(self):
        # First response asks for a tool call, second response is final text.
        script = [
            (
                "what is 2+2",
                [
                    {
                        "type": "tool_use",
                        "id": "tid_1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    }
                ],
            ),
            (
                "tool_result",
                [{"type": "text", "text": "The answer is 4."}],
            ),
        ]
        agent, llm = _basic_agent(script=script)
        turn = agent.run_turn("what is 2+2")
        assert "4" in turn.assistant_text
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0]["name"] == "calculator"
        assert len(turn.tool_results) == 1
        assert turn.iterations == 2

    def test_max_iterations_safety(self):
        # Loop forever via tool_use; the agent should bail after max_turn_iterations.
        looping_block = [
            {"type": "tool_use", "id": "tid_x", "name": "calculator", "input": {"expression": "1+1"}}
        ]
        # Match both the user kickoff and every tool_result so the loop never
        # exits via stop_reason="end_turn".
        script = [
            ("loop forever", looping_block),
            ("tool_result", looping_block),
        ]
        agent, _ = _basic_agent(script=script)
        config_iterations = agent.config.max_turn_iterations
        turn = agent.run_turn("loop forever")
        assert turn.iterations == config_iterations
        assert "max_turn_iterations" in turn.assistant_text


class TestAgentCacheAccounting:
    def test_first_call_records_cache_creation(self):
        agent, _ = _basic_agent(default_text="ok")
        agent.run_turn("hi")
        assert agent.cache_stats.cache_creation_input_tokens > 0
        assert agent.cache_stats.cache_read_input_tokens == 0

    def test_subsequent_calls_record_cache_reads(self):
        agent, _ = _basic_agent(default_text="ok")
        agent.run_turn("turn one")
        agent.run_turn("turn two")
        agent.run_turn("turn three")
        # Turn 1 creates; turns 2-3 should hit the same prefix fingerprint.
        assert agent.cache_stats.cache_read_input_tokens > 0
        assert agent.cache_stats.hit_rate > 0.0

    def test_caching_disabled_changes_prefix_payload(self):
        memory = InMemoryStore()
        tools = [CalculatorTool()]
        cfg_on = AgentConfig(system_blocks=["sys"], tools=tools, enable_caching=True)
        cfg_off = AgentConfig(system_blocks=["sys"], tools=tools, enable_caching=False)
        agent_on = Agent(MockClient(default_text="ok"), cfg_on, memory=memory)
        agent_off = Agent(MockClient(default_text="ok"), cfg_off, memory=memory)
        assert agent_on._system_payload != agent_off._system_payload
        assert agent_on._tools_payload != agent_off._tools_payload


class TestAgentCompaction:
    def test_compaction_runs_when_history_gets_long(self):
        agent, _ = _basic_agent(default_text="ok")
        agent.config.compaction_policy.max_history_chars = 50
        agent.config.compaction_policy.keep_recent_turns = 2
        # Many turns to push past the budget.
        for i in range(8):
            agent.run_turn(f"long turn number {i} with extra padding text " * 5)
        assert agent.config.compaction_policy.max_history_chars == 50
        # History should be trimmed (latest 2 user/assistant pairs plus tool results).
        assert len(agent.history) <= 6


class TestToolExecution:
    def test_slow_tool_times_out_and_the_turn_continues(self):
        class Slow:
            name = "slow"
            description = "sleeps"
            input_schema = {"type": "object", "properties": {}}

            def execute(self, params):
                time.sleep(5)
                return "late"

        script = [
            ("go", [{"type": "tool_use", "id": "s1", "name": "slow", "input": {}}]),
            ("tool_result", [{"type": "text", "text": "gave up"}]),
        ]
        config = AgentConfig(system_blocks=["sys"], tools=[Slow()], tool_timeout_s=0.2)
        agent = Agent(MockClient(script=script), config)
        start = time.perf_counter()
        turn = agent.run_turn("go")
        assert time.perf_counter() - start < 2.0
        assert turn.assistant_text == "gave up"
        assert turn.tool_results[0]["is_error"] is True
        assert "timed out" in turn.tool_results[0]["content"]

    def test_sqlite_memory_tools_work_when_tools_run_off_the_caller_thread(self, tmp_path):
        memory = SqliteMemoryStore(tmp_path / "agent.db")
        script = [
            ("remember", [{"type": "tool_use", "id": "m1", "name": "memory_save", "input": {"key": "user.city", "value": "Tampa"}}]),
            ("tool_result", [{"type": "text", "text": "saved"}]),
        ]
        config = AgentConfig(system_blocks=["sys"], tools=[MemorySaveTool(store=memory)])
        turn = Agent(MockClient(script=script), config, memory=memory).run_turn("remember my city")
        assert turn.tool_results[0].get("is_error") is not True, turn.tool_results
        assert memory.recall("user.city") == "Tampa"


class _BarrierTool:
    """Returns only if `parties` calls are inside execute() at the same time."""

    description = "waits for its siblings"
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name: str, barrier: threading.Barrier):
        self.name = name
        self.barrier = barrier

    def execute(self, params):
        self.barrier.wait(timeout=3)
        return f"{self.name} ran alongside the others"


class TestParallelToolCalls:
    def test_tool_calls_in_one_response_run_concurrently(self):
        barrier = threading.Barrier(3)
        tools = [_BarrierTool(n, barrier) for n in ("a", "b", "c")]
        uses = [{"type": "tool_use", "id": f"id_{n}", "name": n, "input": {}} for n in ("a", "b", "c")]
        script = [("go", uses), ("tool_result", [{"type": "text", "text": "all done"}])]
        agent = Agent(MockClient(script=script), AgentConfig(system_blocks=["sys"], tools=tools))
        turn = agent.run_turn("go")
        assert [r.get("is_error", False) for r in turn.tool_results] == [False, False, False], turn.tool_results
        assert turn.assistant_text == "all done"

    def test_results_keep_the_order_of_the_tool_calls(self):
        class Sleepy:
            description = "sleeps"
            input_schema = {"type": "object", "properties": {}}

            def __init__(self, name, seconds):
                self.name, self.seconds = name, seconds

            def execute(self, params):
                time.sleep(self.seconds)
                return self.name

        tools = [Sleepy("slow", 0.3), Sleepy("fast", 0.0)]
        uses = [
            {"type": "tool_use", "id": "first", "name": "slow", "input": {}},
            {"type": "tool_use", "id": "second", "name": "fast", "input": {}},
        ]
        script = [("go", uses), ("tool_result", [{"type": "text", "text": "ok"}])]
        agent = Agent(MockClient(script=script), AgentConfig(system_blocks=["sys"], tools=tools))
        turn = agent.run_turn("go")
        sent = agent.history[2]["content"]
        assert [(b["tool_use_id"], b["content"]) for b in sent] == [("first", "slow"), ("second", "fast")]
        assert [r["tool_use_id"] for r in turn.tool_results] == ["first", "second"]
        assert tool_pairing_errors(agent.history) == []
