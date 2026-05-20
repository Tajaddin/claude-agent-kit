"""Cache helper tests."""
from __future__ import annotations

from claude_agent_kit.cache import (
    CacheBlock,
    CacheStats,
    build_cached_system,
    build_cached_tools,
    estimate_cache_savings,
)


class TestBuildCachedSystem:
    def test_wraps_strings(self):
        payload = build_cached_system("you are an agent")
        assert payload == [
            {"type": "text", "text": "you are an agent", "cache_control": {"type": "ephemeral"}}
        ]

    def test_supports_cache_block(self):
        payload = build_cached_system(CacheBlock(text="hi", cache_type="ephemeral"))
        assert payload[0]["cache_control"]["type"] == "ephemeral"

    def test_multiple_blocks(self):
        payload = build_cached_system("a", "b", "c")
        assert len(payload) == 3
        for block in payload:
            assert "cache_control" in block


class TestBuildCachedTools:
    def test_marks_last_tool(self):
        tools = [{"name": "a"}, {"name": "b"}]
        out = build_cached_tools(tools)
        assert "cache_control" not in out[0]
        assert out[1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_passthrough(self):
        assert build_cached_tools([]) == []


class TestEstimateCacheSavings:
    def test_zero_savings_with_no_cache_reads(self):
        out = estimate_cache_savings(0, 1000, 100)
        assert out["savings_pct"] == 0.0

    def test_savings_with_heavy_cache(self):
        # 1000 cached + 100 fresh, vs 1100 fresh
        out = estimate_cache_savings(1000, 100, 100)
        assert out["cost_with_cache"] < out["cost_without_cache"]
        assert out["savings_pct"] > 50.0


class TestCacheStats:
    def test_records_usage_blocks(self):
        stats = CacheStats()
        stats.record({"input_tokens": 50, "output_tokens": 30, "cache_creation_input_tokens": 200})
        stats.record({"input_tokens": 50, "output_tokens": 30, "cache_read_input_tokens": 200})
        assert stats.n_requests == 2
        assert stats.cache_creation_input_tokens == 200
        assert stats.cache_read_input_tokens == 200
        assert stats.fresh_input_tokens == 100

    def test_hit_rate_zero_when_nothing_cacheable(self):
        stats = CacheStats()
        stats.record({"input_tokens": 100, "output_tokens": 50})
        assert stats.hit_rate == 0.0

    def test_hit_rate_when_only_reads(self):
        stats = CacheStats()
        stats.record({"cache_read_input_tokens": 400, "input_tokens": 50, "output_tokens": 50})
        assert stats.hit_rate == 1.0

    def test_cost_summary_reflects_savings(self):
        stats = CacheStats()
        # Turn 1: cache creation
        stats.record({"cache_creation_input_tokens": 2500, "input_tokens": 40, "output_tokens": 80})
        # Turns 2-10: cache reads
        for _ in range(9):
            stats.record({"cache_read_input_tokens": 2500, "input_tokens": 40, "output_tokens": 80})
        cost = stats.cost_summary()
        # Realistic production-shape agent: ~65% savings from caching.
        assert cost["savings_pct"] > 50.0
        assert cost["input_tokens_cached_pct"] >= 90.0
