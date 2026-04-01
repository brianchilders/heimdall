-- Migration 002: sqlite-vec virtual table for KNN search.
-- vec0 is provided by the sqlite-vec extension (loaded at connection time).
-- The embedding dimension matches MUNINN_EMBED_DIM (default 768 for nomic-embed-text).
-- If the embed model changes, the vec table must be dropped and recreated via migration 003.

CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    memory_id TEXT,
    embedding float[768]
);
