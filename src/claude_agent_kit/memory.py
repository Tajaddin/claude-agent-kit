"""Memory store backends for the agent's memory tool.

Two backends:
- InMemoryStore: dict-based, fast, lost on process exit. Default in tests.
- SqliteMemoryStore: persistent, single file, good for CLIs and demos.

Both implement the MemoryStore Protocol below, so the agent's memory tool
does not depend on which backend is wired up.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class MemoryStore(Protocol):
    """Storage protocol for the memory tool.

    All methods are synchronous; the agent calls them inside tool execution
    which is itself synchronous between LLM turns.
    """

    def save(self, key: str, value: str) -> None: ...

    def recall(self, key: str) -> str | None: ...

    def list_keys(self, prefix: str = "") -> list[str]: ...

    def delete(self, key: str) -> bool: ...


@dataclass
class InMemoryStore:
    """Process-local memory. Cleared when the process exits."""

    data: dict[str, str] = field(default_factory=dict)
    _timestamps: dict[str, float] = field(default_factory=dict)

    def save(self, key: str, value: str) -> None:
        self.data[key] = value
        self._timestamps[key] = time.time()

    def recall(self, key: str) -> str | None:
        return self.data.get(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self.data if k.startswith(prefix))

    def delete(self, key: str) -> bool:
        existed = key in self.data
        self.data.pop(key, None)
        self._timestamps.pop(key, None)
        return existed


@dataclass
class SqliteMemoryStore:
    """Persistent memory backed by SQLite.

    One file per agent. The schema is intentionally tiny:

        memory(key TEXT PRIMARY KEY, value TEXT, updated_at REAL)

    Concurrent reads are safe via SQLite's default locking. Writes from
    multiple processes would need WAL mode; left as an exercise. Within one
    process the store may be used from several threads (the agent runs tools
    in worker threads); a lock serializes use of the shared connection.
    """

    path: Path | str = ":memory:"

    def __post_init__(self) -> None:
        self.path = str(self.path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memory ("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL, "
            "updated_at REAL NOT NULL"
            ")"
        )
        self._conn.commit()

    def save(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory(key, value, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, time.time()),
            )
            self._conn.commit()

    def recall(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def list_keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            if prefix:
                rows = self._conn.execute(
                    "SELECT key FROM memory WHERE key LIKE ? ORDER BY key",
                    (f"{prefix}%",),
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT key FROM memory ORDER BY key").fetchall()
        return [r[0] for r in rows]

    def delete(self, key: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memory WHERE key=?", (key,))
            self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def dump(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM memory").fetchall()
        return {k: v for k, v in rows}

    def load_json(self, payload: str) -> int:
        data = json.loads(payload)
        for k, v in data.items():
            self.save(k, v)
        return len(data)
