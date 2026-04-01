"""
MCP tool definitions for Muninn.

Exposes memory operations to Mimir (the LLM router) via the Model
Context Protocol.  Each tool maps 1-to-1 to a store function so the
LLM can read and write memories without direct DB access.

Tools registered here:
  remember          — write a new memory
  recall            — semantic KNN search
  list_recent       — chronological list (no embedding required)
  update_memory     — patch content / metadata
  forget            — hard-delete by ID
  add_followup      — schedule a followup after speaking
  get_followups     — list pending followups for a speaker
  dismiss_followup  — mark a followup as handled
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from muninn.config import MuninnConfig
from muninn.db.connection import ConnectionPool
from muninn.store.embeddings import knn_search, store_embedding
from muninn.store.followups import (
    create_followup,
    delete_followup,
    list_followups_for,
)
from muninn.store.memories import (
    create_memory,
    delete_memory,
    get_memory,
    list_memories,
    update_memory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


async def _embed(text: str, config: MuninnConfig) -> list[float]:
    """Compute an embedding via Ollama.

    Args:
        text: Text to embed.
        config: Muninn config with ollama_url and muninn_embed_model.

    Returns:
        Float list.

    Raises:
        httpx.HTTPStatusError: On Ollama error.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{config.ollama_url}/api/embeddings",
            json={"model": config.muninn_embed_model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# Tool schema definitions
# ---------------------------------------------------------------------------

TOOL_REMEMBER = Tool(
    name="remember",
    description=(
        "Store a new memory in the Muninn memory bank. "
        "Use tier='semantic' for facts, 'episodic' for events, "
        "'timeseries' for sensor/metric readings, 'pattern' for behavioural summaries."
    ),
    inputSchema={
        "type": "object",
        "required": ["content", "tier"],
        "properties": {
            "content": {"type": "string", "description": "Natural-language memory text."},
            "tier": {
                "type": "string",
                "enum": ["semantic", "episodic", "timeseries", "pattern"],
                "description": "Memory tier.",
            },
            "source": {"type": "string", "description": "Provenance tag (optional)."},
            "metadata": {
                "type": "object",
                "description": "Optional metadata: who, tags, deadline_utc, ttl_hours.",
            },
        },
    },
)

TOOL_RECALL = Tool(
    name="recall",
    description=(
        "Semantic search across memories using KNN vector similarity. "
        "Returns up to `top_k` most relevant memories for a query."
    ),
    inputSchema={
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "top_k": {"type": "integer", "default": 10, "description": "Max results."},
            "tier": {
                "type": "string",
                "description": "Optional tier filter.",
            },
        },
    },
)

TOOL_LIST_RECENT = Tool(
    name="list_recent",
    description="List the most recently created memories without requiring a query embedding.",
    inputSchema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20},
            "tier": {"type": "string", "description": "Optional tier filter."},
        },
    },
)

TOOL_UPDATE_MEMORY = Tool(
    name="update_memory",
    description="Update the content or metadata of an existing memory by ID.",
    inputSchema={
        "type": "object",
        "required": ["memory_id"],
        "properties": {
            "memory_id": {"type": "string"},
            "content": {"type": "string"},
            "metadata": {"type": "object"},
        },
    },
)

TOOL_FORGET = Tool(
    name="forget",
    description="Hard-delete a memory by ID.",
    inputSchema={
        "type": "object",
        "required": ["memory_id"],
        "properties": {
            "memory_id": {"type": "string"}
        },
    },
)

TOOL_ADD_FOLLOWUP = Tool(
    name="add_followup",
    description=(
        "Schedule a followup for a speaker. Call after Mimir speaks "
        "to check back next time that person is present."
    ),
    inputSchema={
        "type": "object",
        "required": ["who", "spoken_text"],
        "properties": {
            "who": {"type": "string", "description": "Speaker name."},
            "spoken_text": {"type": "string", "description": "What Mimir said."},
            "location": {"type": "string"},
            "memory_id": {"type": "string"},
            "ttl_hours": {"type": "integer", "default": 4},
        },
    },
)

TOOL_GET_FOLLOWUPS = Tool(
    name="get_followups",
    description="List pending followups for a speaker.",
    inputSchema={
        "type": "object",
        "required": ["who"],
        "properties": {
            "who": {"type": "string"},
            "include_expired": {"type": "boolean", "default": False},
        },
    },
)

TOOL_DISMISS_FOLLOWUP = Tool(
    name="dismiss_followup",
    description="Mark a followup as handled and delete it.",
    inputSchema={
        "type": "object",
        "required": ["followup_id"],
        "properties": {
            "followup_id": {"type": "string"}
        },
    },
)

ALL_TOOLS = [
    TOOL_REMEMBER,
    TOOL_RECALL,
    TOOL_LIST_RECENT,
    TOOL_UPDATE_MEMORY,
    TOOL_FORGET,
    TOOL_ADD_FOLLOWUP,
    TOOL_GET_FOLLOWUPS,
    TOOL_DISMISS_FOLLOWUP,
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


async def dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    pool: ConnectionPool,
    config: MuninnConfig,
) -> list[TextContent]:
    """Route an MCP tool call to the appropriate store function.

    Args:
        tool_name: Name of the tool being called.
        arguments: Tool input arguments from the MCP client.
        pool: Database connection pool.
        config: Muninn config.

    Returns:
        List of TextContent results for the MCP response.

    Raises:
        ValueError: If tool_name is unknown.
    """
    if tool_name == "remember":
        mem = await create_memory(
            pool=pool,
            tier=arguments["tier"],
            content=arguments["content"],
            metadata=arguments.get("metadata"),
            source=arguments.get("source"),
        )
        # Embed and store vector asynchronously
        try:
            vec = await _embed(arguments["content"], config)
            await store_embedding(pool, mem["id"], vec, config.muninn_embed_model)
        except Exception as exc:
            logger.warning("Embedding failed for %s: %s", mem["id"], exc)
        return [TextContent(type="text", text=f"Stored memory {mem['id']}")]

    if tool_name == "recall":
        query = arguments["query"]
        top_k = int(arguments.get("top_k", 10))
        tier_filter = arguments.get("tier")

        try:
            vec = await _embed(query, config)
        except Exception as exc:
            return [TextContent(type="text", text=f"Embedding error: {exc}")]

        hits = await knn_search(pool, vec, top_k=top_k * 3)
        results = []
        for hit in hits:
            mem = await get_memory(pool, hit["memory_id"])
            if mem is None:
                continue
            if tier_filter and mem.get("tier") != tier_filter:
                continue
            results.append(
                f"[{mem['tier']}] {mem['content']} (score={hit['distance']:.4f}, id={mem['id']})"
            )
            if len(results) >= top_k:
                break

        text = "\n".join(results) if results else "No memories found."
        return [TextContent(type="text", text=text)]

    if tool_name == "list_recent":
        limit = int(arguments.get("limit", 20))
        tier = arguments.get("tier")
        mems = await list_memories(pool, tier=tier, limit=limit)
        lines = [f"[{m['tier']}] {m['content']} (id={m['id']})" for m in mems]
        text = "\n".join(lines) if lines else "No memories found."
        return [TextContent(type="text", text=text)]

    if tool_name == "update_memory":
        updated = await update_memory(
            pool,
            arguments["memory_id"],
            content=arguments.get("content"),
            metadata=arguments.get("metadata"),
        )
        if updated is None:
            return [TextContent(type="text", text="Memory not found.")]
        return [TextContent(type="text", text=f"Updated memory {updated['id']}")]

    if tool_name == "forget":
        deleted = await delete_memory(pool, arguments["memory_id"])
        msg = "Deleted." if deleted else "Memory not found."
        return [TextContent(type="text", text=msg)]

    if tool_name == "add_followup":
        fu = await create_followup(
            pool=pool,
            who=arguments["who"],
            spoken_text=arguments["spoken_text"],
            location=arguments.get("location"),
            memory_id=arguments.get("memory_id"),
            ttl_hours=int(arguments.get("ttl_hours", config.muninn_followup_ttl_hours)),
        )
        return [TextContent(type="text", text=f"Followup {fu['id']} created for {fu['who']}")]

    if tool_name == "get_followups":
        fus = await list_followups_for(
            pool,
            arguments["who"],
            include_expired=bool(arguments.get("include_expired", False)),
        )
        if not fus:
            return [TextContent(type="text", text="No pending followups.")]
        lines = [f"[{f['id']}] {f['spoken_text']} (expires {f['expires_at']})" for f in fus]
        return [TextContent(type="text", text="\n".join(lines))]

    if tool_name == "dismiss_followup":
        deleted = await delete_followup(pool, arguments["followup_id"])
        msg = "Followup dismissed." if deleted else "Followup not found."
        return [TextContent(type="text", text=msg)]

    raise ValueError(f"Unknown tool: {tool_name!r}")
