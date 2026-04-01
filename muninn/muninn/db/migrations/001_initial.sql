-- Migration 001: initial schema
-- Creates memories, memory_embeddings, vec_memories, and pending_followups tables.

CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    tier         TEXT NOT NULL CHECK(tier IN ('semantic','episodic','timeseries','pattern')),
    content      TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    -- deadline_utc extracted from metadata for indexed deadline queries
    deadline_utc TEXT GENERATED ALWAYS AS
                     (json_extract(metadata, '$.deadline_utc')) VIRTUAL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    expires_at   TEXT,
    source       TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_tier    ON memories(tier);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at)
    WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_deadline ON memories(deadline_utc)
    WHERE deadline_utc IS NOT NULL;

-- Embedding store — one row per (memory, model) pair so multiple embed models coexist.
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    embed_model TEXT NOT NULL,
    embed_dim   INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_memory ON memory_embeddings(memory_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model  ON memory_embeddings(embed_model);

-- pending_followups — Mimir writes here after speaking; read on next ContextEvent.
CREATE TABLE IF NOT EXISTS pending_followups (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT REFERENCES memories(id) ON DELETE CASCADE,
    who         TEXT NOT NULL,
    location    TEXT,
    spoken_text TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_followups_who     ON pending_followups(who);
CREATE INDEX IF NOT EXISTS idx_followups_expires ON pending_followups(expires_at);
