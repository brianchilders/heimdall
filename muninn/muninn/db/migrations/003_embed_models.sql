-- Migration 003: embed model registry.
-- Tracks which embedding models have been used and how many memories each covers.
-- Verdandi asserts model match at startup via GET /embed-model/active.
-- Model mismatch returns HTTP 409; migrate via POST /embed-model/migrate.

CREATE TABLE IF NOT EXISTS embed_models (
    model_name   TEXT PRIMARY KEY,
    embed_dim    INTEGER NOT NULL,
    first_used   TEXT NOT NULL,
    last_used    TEXT NOT NULL,
    memory_count INTEGER NOT NULL DEFAULT 0
);
