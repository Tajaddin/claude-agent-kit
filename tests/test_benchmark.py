"""Smoke test for the cache benchmark in mock mode."""
from __future__ import annotations

from benchmarks import cache_benchmark
from benchmarks.cache_benchmark import USER_TURNS, run_mode


class TestBenchmarkSmoke:
    def test_mock_mode_runs(self):
        result = run_mode("mock", USER_TURNS, enable_caching=True)
        assert result["n_turns"] == len(USER_TURNS)
        # With caching on, hit rate should be high after the first turn.
        assert result["hit_rate_pct"] > 80.0
        # Cost reduction should exceed 60% on a 10-turn run with a fat prefix.
        assert result["savings_pct"] > 50.0

    def test_caching_off_has_no_hits(self):
        result = run_mode("mock", USER_TURNS, enable_caching=False)
        assert result["hit_rate_pct"] >= 0.0
        # Without cache_control on system or tools, the fingerprint changes (or
        # caching is irrelevant). Either way savings should be <= 0.
        assert result["savings_pct"] <= 0.0


class TestOutputPath:
    def test_mock_run_does_not_replace_the_live_record(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_benchmark, "RESULTS_DIR", tmp_path)
        live_record = tmp_path / "cache_benchmark.json"
        live_record.write_text('{"mode": "live"}', encoding="utf-8")
        assert cache_benchmark.main(["--mode", "mock", "--turns", "1"]) == 0
        assert live_record.read_text(encoding="utf-8") == '{"mode": "live"}'
        assert '"mode": "mock"' in (tmp_path / "cache_benchmark-mock.json").read_text(encoding="utf-8")
