"""
voice_routes.py — Speaker identity management for memory-mcp.

Handles the full lifecycle of voiceprint-based speaker enrollment:
listing unknown provisional voices, enrolling them under a real name,
merging duplicates, and updating voiceprint embeddings.

Voiceprints are stored in the existing entity meta JSON column — no schema
changes required.

Convention for provisional entities (created by the pipeline worker):
  entity.name   = "unknown_voice_{8-char-hash}"
  entity.type   = "person"
  entity.meta   = {
    "voiceprint": [...],          # 256-dim float list (unit vector)
    "voiceprint_samples": N,      # utterances averaged into this embedding
    "status": "unenrolled",
    "first_seen": "<ISO timestamp>",
    "first_seen_room": "<room name>",
    "detection_count": N
  }

Wire into memory-mcp api.py with:
    from voice_routes import router as voice_router
    app.include_router(voice_router)

This module is authored in the heimdall repo at memory_extensions/voice_routes.py
and copied to the memory-mcp repo on the Pi 4.  Keep them in sync.
"""

from __future__ import annotations

import json
import logging
import math
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

import server as mem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voices", tags=["voices"])

VOICEPRINT_DIM = 256


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class EnrollRequest(BaseModel):
    """Rename a provisional entity to a confirmed real name."""

    entity_name: str
    new_name: str
    display_name: str | None = None


class MergeRequest(BaseModel):
    """Merge two speaker entities, transferring all data from source to target."""

    source_name: str
    target_name: str


class UpdatePrintRequest(BaseModel):
    """Update a speaker's canonical voiceprint embedding."""

    entity_name: str
    embedding: list[float]
    weight: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_finite(cls, v: list[float]) -> list[float]:
        """Reject NaN and Infinity values that would corrupt the stored embedding."""
        if any(not math.isfinite(x) for x in v):
            raise ValueError("Embedding values must be finite floats (no NaN or Infinity)")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(vec: list[float]) -> list[float]:
    """Return a unit vector.  Resemblyzer embeddings are unit vectors; blending
    them drifts the norm, so we restore it after every averaging operation."""
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _embedding_norm(vec: list[float]) -> float:
    """Return the L2 norm of the vector, rounded to 4 decimal places."""
    return round(sum(x * x for x in vec) ** 0.5, 4)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/unknown")
async def list_unknown(limit: int = 20, min_detections: int = 1):
    """List all provisional (unenrolled) speaker entities with a sample transcript.

    Returns entities whose ``meta.status`` is ``"unenrolled"``, sorted by
    detection count descending.  Includes the most recent sample transcript
    and timestamp from Tier 2 voice_activity readings so the caller can
    identify the voice.

    Args:
        limit: Maximum number of results (default 20).
        min_detections: Exclude entities with fewer detections (default 1).

    Returns:
        JSON with ``result`` list and ``ok: true``.
    """
    db = mem.get_db()
    try:
        rows = db.execute(
            """
            SELECT id, name,
                   json_extract(meta, '$.first_seen')      AS first_seen,
                   json_extract(meta, '$.first_seen_room') AS first_seen_room,
                   CAST(json_extract(meta, '$.detection_count') AS INTEGER) AS detection_count
            FROM entities
            WHERE type = 'person'
              AND json_extract(meta, '$.status') = 'unenrolled'
              AND CAST(json_extract(meta, '$.detection_count') AS INTEGER) >= ?
            ORDER BY detection_count DESC
            LIMIT ?
            """,
            (min_detections, limit),
        ).fetchall()

        results = []
        for row in rows:
            reading = db.execute(
                """
                SELECT MAX(ts) AS last_seen,
                       json_extract(value_json, '$.transcript') AS transcript
                FROM readings
                WHERE entity_id = ? AND metric = 'voice_activity'
                """,
                (row["id"],),
            ).fetchone()

            results.append({
                "entity_name": row["name"],
                "first_seen": row["first_seen"],
                "first_seen_room": row["first_seen_room"],
                "detection_count": row["detection_count"],
                "last_seen": reading["last_seen"] if reading else None,
                "sample_transcript": reading["transcript"] if reading else None,
            })

        return {"result": results, "ok": True}
    finally:
        db.close()


@router.post("/enroll")
async def enroll(req: EnrollRequest):
    """Rename a provisional entity to a real person's name and mark as enrolled.

    The entity's primary key is unchanged — all memories, readings, sessions,
    and relations remain attached via foreign key without any data migration.

    Args:
        req: EnrollRequest with current provisional name and desired new name.

    Returns:
        JSON with entity_id, new name, previous name, and data counts.

    Raises:
        HTTPException 404: Source entity not found.
        HTTPException 409: Target name already exists.
    """
    db = mem.get_db()
    try:
        source = db.execute(
            "SELECT id FROM entities WHERE name = ?", (req.entity_name,)
        ).fetchone()
        if not source:
            raise HTTPException(
                status_code=404, detail=f"Entity '{req.entity_name}' not found"
            )

        conflict = db.execute(
            "SELECT id FROM entities WHERE name = ?", (req.new_name,)
        ).fetchone()
        if conflict:
            raise HTTPException(
                status_code=409, detail=f"Entity '{req.new_name}' already exists"
            )

        n_memories = db.execute(
            "SELECT COUNT(*) FROM memories WHERE entity_id = ?", (source["id"],)
        ).fetchone()[0]
        n_readings = db.execute(
            "SELECT COUNT(*) FROM readings WHERE entity_id = ?", (source["id"],)
        ).fetchone()[0]

        now = time.time()
        if req.display_name is not None:
            db.execute(
                """UPDATE entities
                   SET name = ?, updated = ?,
                       meta = json_set(meta, '$.status', 'enrolled', '$.display_name', ?)
                   WHERE id = ?""",
                (req.new_name, now, req.display_name, source["id"]),
            )
        else:
            db.execute(
                """UPDATE entities
                   SET name = ?, updated = ?,
                       meta = json_set(meta, '$.status', 'enrolled')
                   WHERE id = ?""",
                (req.new_name, now, source["id"]),
            )
        db.commit()

        logger.info("Enrolled provisional entity '%s' as '%s'", req.entity_name, req.new_name)
        return {
            "result": {
                "entity_id": source["id"],
                "entity_name": req.new_name,
                "previous_name": req.entity_name,
                "memories_transferred": n_memories,
                "readings_transferred": n_readings,
            },
            "ok": True,
        }
    finally:
        db.close()


@router.post("/merge")
async def merge(req: MergeRequest):
    """Merge a provisional entity into an enrolled entity.

    Transfers all Tier 1 memories, Tier 2 readings, and active relations
    (``valid_until IS NULL``) from source to target.  Averages voiceprint
    embeddings weighted by sample count so the more-established embedding
    dominates.  Deletes the source entity; relations that could not be
    transferred due to UNIQUE constraint conflicts are cascade-deleted.

    Runs in a single transaction with explicit rollback on error.

    Args:
        req: MergeRequest naming source (to delete) and target (to keep).

    Returns:
        JSON with transfer counts and confirmation of source deletion.

    Raises:
        HTTPException 400: source_name == target_name.
        HTTPException 404: Source or target entity not found.
        HTTPException 500: Unexpected database error (rolls back).
    """
    if req.source_name == req.target_name:
        raise HTTPException(
            status_code=400, detail="source_name and target_name must be different"
        )

    db = mem.get_db()
    try:
        source = db.execute(
            "SELECT id, meta FROM entities WHERE name = ?", (req.source_name,)
        ).fetchone()
        if not source:
            raise HTTPException(
                status_code=404, detail=f"Entity '{req.source_name}' not found"
            )

        target = db.execute(
            "SELECT id, meta FROM entities WHERE name = ?", (req.target_name,)
        ).fetchone()
        if not target:
            raise HTTPException(
                status_code=404, detail=f"Entity '{req.target_name}' not found"
            )

        src_id, tgt_id = source["id"], target["id"]
        src_meta = json.loads(source["meta"])
        tgt_meta = json.loads(target["meta"])

        # Count what we're moving (before the transfer, for response)
        n_memories = db.execute(
            "SELECT COUNT(*) FROM memories WHERE entity_id = ?", (src_id,)
        ).fetchone()[0]
        n_readings = db.execute(
            "SELECT COUNT(*) FROM readings WHERE entity_id = ?", (src_id,)
        ).fetchone()[0]
        n_relations = db.execute(
            """SELECT COUNT(*) FROM relations
               WHERE (entity_a = ? OR entity_b = ?) AND valid_until IS NULL""",
            (src_id, src_id),
        ).fetchone()[0]

        # Transfer Tier 1 and Tier 2 data.
        db.execute(
            "UPDATE memories SET entity_id = ? WHERE entity_id = ?", (tgt_id, src_id)
        )
        db.execute(
            "UPDATE readings SET entity_id = ? WHERE entity_id = ?", (tgt_id, src_id)
        )

        # Transfer active relations only.  UPDATE OR IGNORE silently skips rows
        # that would violate UNIQUE(entity_a, entity_b, rel_type); those rows
        # are deleted by CASCADE when the source entity is removed below.
        db.execute(
            """UPDATE OR IGNORE relations
               SET entity_a = ? WHERE entity_a = ? AND valid_until IS NULL""",
            (tgt_id, src_id),
        )
        db.execute(
            """UPDATE OR IGNORE relations
               SET entity_b = ? WHERE entity_b = ? AND valid_until IS NULL""",
            (tgt_id, src_id),
        )

        # Merge voiceprints — weighted average by sample count so the more
        # established embedding dominates.  Normalize to restore unit vector.
        src_vp = src_meta.get("voiceprint")
        tgt_vp = tgt_meta.get("voiceprint")

        if src_vp and tgt_vp:
            src_n = src_meta.get("voiceprint_samples", 1)
            tgt_n = tgt_meta.get("voiceprint_samples", 1)
            total = src_n + tgt_n
            merged_vp = _normalize(
                [(v * tgt_n + u * src_n) / total for v, u in zip(tgt_vp, src_vp)]
            )
            tgt_meta["voiceprint"] = merged_vp
            tgt_meta["voiceprint_samples"] = total
        elif src_vp and not tgt_vp:
            tgt_meta["voiceprint"] = src_vp
            tgt_meta["voiceprint_samples"] = src_meta.get("voiceprint_samples", 1)

        tgt_meta["detection_count"] = (
            tgt_meta.get("detection_count", 0) + src_meta.get("detection_count", 0)
        )

        now = time.time()
        db.execute(
            "UPDATE entities SET meta = ?, updated = ? WHERE id = ?",
            (json.dumps(tgt_meta), now, tgt_id),
        )

        # Delete source — ON DELETE CASCADE removes any remaining linked rows.
        db.execute("DELETE FROM entities WHERE id = ?", (src_id,))
        db.commit()

        logger.info(
            "Merged '%s' → '%s': %d memories, %d readings, %d relations transferred",
            req.source_name, req.target_name, n_memories, n_readings, n_relations,
        )
        return {
            "result": {
                "target_name": req.target_name,
                "memories_merged": n_memories,
                "readings_merged": n_readings,
                "relations_merged": n_relations,
                "source_deleted": req.source_name,
            },
            "ok": True,
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@router.post("/update_print")
async def update_print(req: UpdatePrintRequest):
    """Update the voiceprint embedding for an entity using a running weighted average.

    Called by the pipeline worker after each confident speaker identification
    to refine the embedding over time.  The new embedding is re-normalized
    after blending so it remains a unit vector.

    Args:
        req: UpdatePrintRequest with entity name, 256-dim embedding, and weight.

    Returns:
        JSON with entity name, total sample count, and current embedding norm.

    Raises:
        HTTPException 404: Entity not found.
        HTTPException 422: Embedding is not exactly 256 dimensions.
    """
    if len(req.embedding) != VOICEPRINT_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Embedding must be {VOICEPRINT_DIM}-dimensional, got {len(req.embedding)}",
        )

    db = mem.get_db()
    try:
        entity = db.execute(
            "SELECT id, meta FROM entities WHERE name = ?", (req.entity_name,)
        ).fetchone()
        if not entity:
            raise HTTPException(
                status_code=404, detail=f"Entity '{req.entity_name}' not found"
            )

        meta = json.loads(entity["meta"])
        existing_vp = meta.get("voiceprint")
        existing_samples = meta.get("voiceprint_samples", 0)

        if not existing_vp:
            new_vp = list(req.embedding)
            new_samples = 1
        else:
            w = req.weight
            blended = [(1.0 - w) * v + w * u for v, u in zip(existing_vp, req.embedding)]
            new_vp = _normalize(blended)
            new_samples = existing_samples + 1

        meta["voiceprint"] = new_vp
        meta["voiceprint_samples"] = new_samples

        now = time.time()
        db.execute(
            "UPDATE entities SET meta = ?, updated = ? WHERE id = ?",
            (json.dumps(meta), now, entity["id"]),
        )
        db.commit()

        logger.debug(
            "Voiceprint updated for '%s' (samples=%d, norm=%.4f)",
            req.entity_name, new_samples, _embedding_norm(new_vp),
        )
        return {
            "result": {
                "entity_name": req.entity_name,
                "voiceprint_samples": new_samples,
                "embedding_norm": _embedding_norm(new_vp),
            },
            "ok": True,
        }
    finally:
        db.close()
