"""
Database schema constants shared across Muninn and its clients.

Centralising table names and tier values prevents string literals from
scattering across services and makes migrations easier to reason about.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Table names
# ---------------------------------------------------------------------------

MEMORIES_TABLE = "memories"
EMBEDDINGS_TABLE = "memory_embeddings"
VEC_TABLE = "vec_memories"
FOLLOWUPS_TABLE = "pending_followups"
EMBED_MODELS_TABLE = "embed_models"

# ---------------------------------------------------------------------------
# Memory tiers
# ---------------------------------------------------------------------------

TIER_SEMANTIC = "semantic"      # facts, preferences, relationships
TIER_EPISODIC = "episodic"      # conversation transcripts, events
TIER_TIMESERIES = "timeseries"  # sensor readings, voice activity logs
TIER_PATTERN = "pattern"        # LLM-generated behavioural summaries

ALL_TIERS = [TIER_SEMANTIC, TIER_EPISODIC, TIER_TIMESERIES, TIER_PATTERN]

# ---------------------------------------------------------------------------
# Metadata keys (used in memories.metadata JSON)
# ---------------------------------------------------------------------------

META_DEADLINE = "deadline_utc"  # ISO-8601 — triggers urgency boost in Verdandi
META_TAGS = "tags"              # list[str] — for filtering
META_TTL_HOURS = "ttl_hours"    # explicit expiry override
META_TYPE = "type"              # sub-type within tier, e.g. "voice_activity"
META_WHO = "who"                # speaker / subject entity name
META_LOCATION = "location"      # room name
