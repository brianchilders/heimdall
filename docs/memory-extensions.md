# memory-extensions — Voice Route API Reference

## Purpose

`memory_extensions/voice_routes.py` is a FastAPI `APIRouter` that adds four
voice-management endpoints to memory-mcp.  These routes manage the lifecycle
of speaker entities: listing unknown voices detected by the pipeline worker,
enrolling them under real names, merging provisional identities into enrolled
ones, and keeping voiceprint embeddings up to date.

---

## Architecture

The module is a self-contained `APIRouter` designed to be spliced into
memory-mcp's `api.py`.  It reads from and writes to memory-mcp's SQLite
database using the same `get_db` dependency pattern as the rest of api.py.

```
pipeline worker        enrollment CLI
       │                      │
       ▼                      ▼
 POST /voices/update_print   POST /voices/enroll
 (after each confident        (human confirms identity)
  voiceprint match)
                              POST /voices/merge
                              (merge provisional → enrolled)

  GET /voices/unknown
  (list all unenrolled voices for human review)
```

---

## Integration with memory-mcp

Add to `api.py`:

```python
from memory_extensions.voice_routes import router as voice_router
from memory_extensions.voice_routes import get_db as _voice_get_db

app.include_router(voice_router)
app.dependency_overrides[_voice_get_db] = get_db   # memory-mcp's own get_db
```

---

## API Reference

### GET `/voices/unknown`

List all unenrolled provisional speaker entities, sorted by detection count
descending.

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Maximum number of results |
| `min_detections` | int | 1 | Exclude entities with fewer detections |

**Response 200**

```json
{
  "result": [
    {
      "entity_name": "unknown_voice_a3f2",
      "first_seen": "2026-03-20",
      "first_seen_room": "kitchen",
      "detection_count": 7,
      "sample_transcript": "I need to pick up groceries tomorrow"
    }
  ],
  "ok": true
}
```

---

### POST `/voices/enroll`

Rename a provisional entity to a confirmed real name.  The entity's primary
key (`entity_id`) is unchanged — all memories, readings, sessions, and
relations remain attached without any data migration.

**Request body**

```json
{
  "entity_name": "unknown_voice_a3f2",
  "new_name": "Brian",
  "display_name": "Brian Childers"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `entity_name` | yes | Current provisional entity name |
| `new_name` | yes | Real name to assign |
| `display_name` | no | Optional full human-readable name stored in meta |

**Response 200**

```json
{
  "result": {
    "enrolled_as": "Brian",
    "entity_id": 42,
    "previous_name": "unknown_voice_a3f2"
  },
  "ok": true
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| 404 | `entity_name` not found |
| 409 | `new_name` already exists as another entity |

**Effect on entity meta**

```json
{
  "status": "enrolled",
  "display_name": "Brian Childers",
  "voiceprint": [...],
  "detection_count": 7,
  "first_seen": "2026-03-20"
}
```

---

### POST `/voices/merge`

Merge a provisional entity into an enrolled entity.  Transfers all linked
data from source to target and deletes the source.

**What is transferred**

| Data | Mechanism |
|------|-----------|
| Tier 1 memories | `UPDATE memories SET entity_id = target` |
| Tier 2 readings | `UPDATE readings SET entity_id = target` |
| Tier 1.5 sessions | `UPDATE sessions SET entity_id = target` (turns cascade via FK) |
| Relations | `UPDATE OR IGNORE` — duplicates silently skipped, cascade-deleted with source |
| Voiceprint | Averaged at equal weight (0.5 : 0.5) |

**Request body**

```json
{
  "source_name": "unknown_voice_b9c1",
  "target_name": "Brian"
}
```

**Response 200**

```json
{
  "result": {
    "source": "unknown_voice_b9c1",
    "target": "Brian",
    "memories_transferred": 12,
    "readings_transferred": 47,
    "sessions_transferred": 3
  },
  "ok": true
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| 404 | `source_name` not found |
| 404 | `target_name` not found |

---

### POST `/voices/update_print`

Update the canonical voiceprint embedding for an entity using a running
average.

**Formula**

```
new_embedding = (1 - weight) * existing + weight * incoming
```

If no voiceprint exists yet, the incoming embedding is stored directly
(weight has no effect on initialization).

**Request body**

```json
{
  "entity_name": "Brian",
  "embedding": [0.01, -0.03, ...],
  "weight": 0.1
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `entity_name` | yes | — | Entity to update |
| `embedding` | yes | — | 256-dim resemblyzer float array |
| `weight` | no | 0.1 | New sample contribution (0 < weight ≤ 1) |

**Response 200**

```json
{
  "result": {
    "entity_name": "Brian",
    "embedding_dim": 256,
    "was_initialized": false
  },
  "ok": true
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| 404 | Entity not found |
| 422 | Embedding is not exactly 256 dimensions |

---

## Configuration

No environment variables.  All configuration (DB path, connection pooling)
is inherited from memory-mcp's existing `get_db` dependency.

---

## Running

This module has no standalone entry point.  It runs as part of memory-mcp.

After splicing in:

```bash
# Verify the routes appear in the schema
curl http://localhost:8900/openapi.json | python -m json.tool | grep "/voices"
```

---

## Testing

```bash
# From the heimdall repo root
pytest memory_extensions/tests/ -v
```

No live services required — tests use an in-memory SQLite database.

---

## Deployment notes

1. Copy `memory_extensions/voice_routes.py` to the memory-mcp server.
2. Add the two lines to memory-mcp's `api.py` (see Integration section above).
3. Restart memory-mcp: `systemctl restart memory-mcp` (or equivalent).
4. Verify with curl:

```bash
curl http://memory-mcp:8900/voices/unknown
```
