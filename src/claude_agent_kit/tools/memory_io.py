"""Tools that read and write the agent's memory store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claude_agent_kit.memory import MemoryStore


@dataclass
class MemorySaveTool:
    store: MemoryStore
    name: str = "memory_save"
    description: str = (
        "Save a fact to long-term memory. Use stable keys (e.g. "
        "'user.preference.coffee' or 'project.deadline'). Overwrites existing values."
    )
    input_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Stable identifier for this fact."},
                    "value": {"type": "string", "description": "The fact, in plain prose."},
                },
                "required": ["key", "value"],
            }

    def execute(self, params: dict[str, Any]) -> str:
        key = str(params.get("key", "")).strip()
        value = str(params.get("value", "")).strip()
        if not key or not value:
            return "error: key and value are both required"
        self.store.save(key, value)
        return f"saved key={key}"


@dataclass
class MemoryRecallTool:
    store: MemoryStore
    name: str = "memory_recall"
    description: str = (
        "Recall facts from long-term memory. Pass either an exact key, or a "
        "prefix ending in '*' to list all matching keys with their values."
    )
    input_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Exact key, or prefix ending in '*' for prefix-search.",
                    }
                },
                "required": ["key"],
            }

    def execute(self, params: dict[str, Any]) -> str:
        key = str(params.get("key", "")).strip()
        if not key:
            return "error: key is required"
        if key.endswith("*"):
            prefix = key[:-1]
            keys = self.store.list_keys(prefix)
            if not keys:
                return "no matches"
            lines = [f"{k}: {self.store.recall(k)}" for k in keys]
            return "\n".join(lines)
        value = self.store.recall(key)
        if value is None:
            return f"no value found for key={key}"
        return value
