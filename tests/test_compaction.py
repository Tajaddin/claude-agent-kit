"""Compaction tests."""
from __future__ import annotations

from claude_agent_kit.compaction import CompactionPolicy, compact_history


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
