"""Production patterns for agents on the Anthropic API.

Prompt caching, memory tool, parallel tool execution, conversation compaction.
"""
from claude_agent_kit.agent import Agent, AgentConfig, AgentTurn
from claude_agent_kit.cache import CacheBlock, build_cached_system, build_cached_tools
from claude_agent_kit.compaction import compact_history
from claude_agent_kit.memory import InMemoryStore, MemoryStore, SqliteMemoryStore

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "AgentConfig",
    "AgentTurn",
    "CacheBlock",
    "InMemoryStore",
    "MemoryStore",
    "SqliteMemoryStore",
    "build_cached_system",
    "build_cached_tools",
    "compact_history",
]
