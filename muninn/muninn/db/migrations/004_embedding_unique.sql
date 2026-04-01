-- Migration 004: add UNIQUE constraint to memory_embeddings(memory_id, embed_model).
-- Required for the ON CONFLICT upsert in store_embedding().
-- We recreate the table since SQLite does not support ADD CONSTRAINT.

CREATE TABLE IF NOT EXISTS memory_embeddings_new (
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    embed_model TEXT NOT NULL,
    embed_dim   INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (memory_id, embed_model)
);

INSERT OR IGNORE INTO memory_embeddings_new
    SELECT memory_id, embedding, embed_model, embed_dim, created_at
    FROM memory_embeddings;

DROP TABLE memory_embeddings;

ALTER TABLE memory_embeddings_new RENAME TO memory_embeddings;

CREATE INDEX IF NOT EXISTS idx_embeddings_memory ON memory_embeddings(memory_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model  ON memory_embeddings(embed_model);
