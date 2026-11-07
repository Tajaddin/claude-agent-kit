"""Compaction tests."""
from __future__ import annotations

import random

from claude_agent_kit.compaction import CompactionPolicy, _char_count, compact_history
from tests.helpers import tool_pairing_errors


class TestCompactHistory:
    def test_no_op_when_under_budget(self):
        msgs = [{"role": "user", "content": "hi"}]
        out, did = compact_history(msgs, policy=CompactionPolicy(max_history_chars=1000))
        assert did is False
        assert out == msgs

    def test_compacts_when_over_budget(self):
        msgs = [{"role": "user", "content": "x" * 200} for _ in range(20)]
        out, did = compact_history(
            msgs, policy=CompactionPolicy(max_history_chars=500, keep_recent_turns=4)
        )
        assert did is True
        assert len(out) == 5  # 1 summary + 4 recent
        assert out[0]["content"].startswith("[earlier conversation summary]")

    def test_skips_when_short_in_count_but_long_in_chars(self):
        msgs = [{"role": "user", "content": "x" * 5000}]
        out, did = compact_history(
            msgs,
            policy=CompactionPolicy(max_history_chars=500, keep_recent_turns=4),
        )
        assert did is False
        assert out == msgs

    def test_custom_summarizer(self):
        msgs = [{"role": "user", "content": "x" * 200} for _ in range(20)]
        marker = "CUSTOM_SUMMARY"
        out, did = compact_history(
            msgs,
            policy=CompactionPolicy(max_history_chars=500, keep_recent_turns=4),
            summarize=lambda turns: marker,
        )
        assert did is True
        assert marker in out[0]["content"]


def _random_history(rng: random.Random) -> list[dict]:
    """A valid agent history: plain turns, tool round trips, and turns that
    stopped at the iteration cap (a tool_result followed by a new user turn)."""
    def text() -> str:
        return "w" * rng.choice([5, 80, 600, 4000])

    history: list[dict] = [{"role": "user", "content": text()}]
    next_id = 0
    for _ in range(rng.randint(1, 14)):
        if rng.random() < 0.6:
            ids = [f"t{next_id + k}" for k in range(rng.randint(1, 3))]
            next_id += len(ids)
            blocks = [{"type": "text", "text": text()}] if rng.random() < 0.5 else []
            blocks += [
                {"type": "tool_use", "id": i, "name": "search", "input": {"query": text()}} for i in ids
            ]
            history.append({"role": "assistant", "content": blocks})
            history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": i, "content": "r" * rng.choice([1, 300, 50_000])}
                        for i in ids
                    ],
                }
            )
            if rng.random() < 0.3:
                history.append({"role": "user", "content": text()})
        else:
            history.append({"role": "assistant", "content": [{"type": "text", "text": text()}]})
            history.append({"role": "user", "content": text()})
    if history[-1]["role"] == "assistant" or isinstance(history[-1]["content"], list):
        history.append({"role": "user", "content": text()})
    return history


class TestToolPairs:
    def test_tool_result_content_counts_toward_the_budget(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 50_000}]},
        ]
        assert _char_count(history) >= 50_000

    def test_tool_result_given_as_text_blocks_counts_too(self):
        result = {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "y" * 9000}]}
        assert _char_count([{"role": "user", "content": [result]}]) >= 9000

    def test_tool_heavy_history_gets_compacted(self):
        history = [{"role": "user", "content": "q"}]
        for i in range(4):
            history += [
                {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "search", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 10_000}]},
            ]
        history.append({"role": "user", "content": "next"})
        out, did = compact_history(history, policy=CompactionPolicy(max_history_chars=12_000, keep_recent_turns=2))
        assert did is True
        assert tool_pairing_errors(out) == []

    def test_odd_keep_count_does_not_orphan_a_tool_result(self):
        history = [{"role": "user", "content": "a" * 13_000}]
        for i in range(3):
            history += [
                {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "calculator", "input": {"expression": "1+1"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "2"}]},
            ]
        history += [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}, {"role": "user", "content": "next"}]
        out, did = compact_history(history, policy=CompactionPolicy(keep_recent_turns=5))
        assert did is True
        assert tool_pairing_errors(out) == []
        assert len(out) - 1 >= 5

    def test_fuzz_compaction_keeps_every_tool_result_with_its_tool_use(self):
        rng = random.Random(20261106)
        compacted = 0
        for _ in range(3000):
            history = _random_history(rng)
            assert tool_pairing_errors(history) == []
            keep = rng.randint(1, 10)
            policy = CompactionPolicy(max_history_chars=rng.choice([0, 500, 12_000, 60_000]), keep_recent_turns=keep)
            out, did = compact_history(history, policy=policy)
            assert tool_pairing_errors(out) == [], (keep, [m["role"] for m in history])
            assert out[-1] is history[-1]
            if did:
                compacted += 1
                tail = out[1:]
                assert tail == history[len(history) - len(tail) :]
                assert len(tail) >= keep
        assert compacted > 1000
