"""Memory store tests."""
from __future__ import annotations

import pytest

from claude_agent_kit.memory import InMemoryStore, SqliteMemoryStore


class TestInMemoryStore:
    def test_save_and_recall(self):
        s = InMemoryStore()
        s.save("user.color", "blue")
        assert s.recall("user.color") == "blue"

    def test_missing_key_returns_none(self):
        s = InMemoryStore()
        assert s.recall("nope") is None

    def test_list_keys_prefix(self):
        s = InMemoryStore()
        s.save("user.a", "1")
        s.save("user.b", "2")
        s.save("task.x", "3")
        assert s.list_keys("user.") == ["user.a", "user.b"]

    def test_delete(self):
        s = InMemoryStore()
        s.save("k", "v")
        assert s.delete("k") is True
        assert s.delete("k") is False
        assert s.recall("k") is None


class TestSqliteMemoryStore:
    def test_in_memory_db_works(self):
        s = SqliteMemoryStore(":memory:")
        s.save("k", "v")
        assert s.recall("k") == "v"
        assert s.list_keys() == ["k"]
        s.close()

    def test_persists_to_file(self, tmp_path):
        path = tmp_path / "mem.db"
        s1 = SqliteMemoryStore(path)
        s1.save("a", "1")
        s1.save("b", "2")
        s1.close()

        s2 = SqliteMemoryStore(path)
        assert s2.recall("a") == "1"
        assert s2.recall("b") == "2"
        assert sorted(s2.list_keys()) == ["a", "b"]
        s2.close()

    def test_overwrite_updates_value(self):
        s = SqliteMemoryStore(":memory:")
        s.save("k", "v1")
        s.save("k", "v2")
        assert s.recall("k") == "v2"
        assert s.list_keys() == ["k"]
        s.close()


class TestPrefixSearchParity:
    @pytest.mark.parametrize("store", [InMemoryStore, SqliteMemoryStore])
    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [
            ("user_", ["user_a.x"]),
            ("user.", ["user.b"]),
            ("User", ["User.z"]),
            ("50%", ["50%.off"]),
            ("a\\", ["a\\b"]),
            ("", ["50%.off", "User.z", "a\\b", "user.b", "userXa.y", "user_a.x"]),
        ],
    )
    def test_prefix_is_literal_and_case_sensitive(self, store, prefix, expected):
        s = store()
        for key in ("user_a.x", "userXa.y", "User.z", "user.b", "50%.off", "a\\b"):
            s.save(key, "v")
        assert s.list_keys(prefix) == expected
