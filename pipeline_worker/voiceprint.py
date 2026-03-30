"""
VoiceprintMatcher — local SQLite cache for speaker voiceprint matching.

Architecture note
-----------------
memory-mcp is the *canonical* store for voiceprint embeddings (held in entity
metadata).  This SQLite database is a runtime cache: it is loaded from
memory-mcp on startup via POST /voices/sync (or manually via
reload_from_memory_mcp) and updated after each confident match without a
round-trip to memory-mcp.

On-disk schema
--------------
    voiceprints (
        entity_name   TEXT PRIMARY KEY,
        embedding     BLOB NOT NULL,   -- float32, VOICEPRINT_DIM × 4 bytes
        sample_count  INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT NOT NULL    -- ISO-8601 UTC
    )
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline_worker.models import ConfidenceLevel, VoiceprintMatch

logger = logging.getLogger(__name__)

VOICEPRINT_DIM = 256

# Default cosine-similarity thresholds; may be overridden via config.
DEFAULT_CONFIDENT_THRESHOLD = 0.85
DEFAULT_PROBABLE_THRESHOLD = 0.70

_DDL = """
CREATE TABLE IF NOT EXISTS voiceprints (
    entity_name   TEXT PRIMARY KEY,
    embedding     BLOB NOT NULL,
    sample_count  INTEGER NOT NULL DEFAULT 1,
    updated_at    TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class StoredVoiceprint:
    """A single voiceprint record retrieved from the local cache."""

    entity_name: str
    embedding: np.ndarray       # shape (VOICEPRINT_DIM,), dtype float32
    sample_count: int
    updated_at: datetime


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class VoiceprintMatcher:
    """Match incoming speaker embeddings against a local voiceprint cache.

    Thread-safety: SQLite is opened in check_same_thread=False mode.  The
    caller is responsible for not issuing concurrent writes from multiple
    threads without external locking.  For the single-process pipeline worker
    this is not an issue.

    Usage::

        with VoiceprintMatcher("voiceprints.sqlite") as matcher:
            match = matcher.match(embedding)
            if match.confidence_level == ConfidenceLevel.CONFIDENT:
                matcher.update_after_match(match.entity_name, embedding)
    """

    def __init__(
        self,
        db_path: str | Path,
        confident_threshold: float = DEFAULT_CONFIDENT_THRESHOLD,
        probable_threshold: float = DEFAULT_PROBABLE_THRESHOLD,
    ) -> None:
        self.db_path = Path(db_path)
        self.confident_threshold = confident_threshold
        self.probable_threshold = probable_threshold
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Open (or create) the SQLite database and ensure the schema exists."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.commit()
        logger.info("Voiceprint DB opened: %s", self.db_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "VoiceprintMatcher":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(
        self,
        entity_name: str,
        embedding: np.ndarray,
        sample_count: int = 1,
    ) -> None:
        """Insert or replace a voiceprint embedding for an entity.

        Args:
            entity_name: The memory-mcp entity name.
            embedding: Unit-norm float32 array of shape (VOICEPRINT_DIM,).
            sample_count: Number of utterances averaged into this embedding.

        Raises:
            ValueError: If embedding has the wrong shape.
        """
        self._validate_embedding(embedding)
        blob = embedding.astype(np.float32).tobytes()
        now = _utcnow()
        self._conn.execute(
            """
            INSERT INTO voiceprints (entity_name, embedding, sample_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_name) DO UPDATE SET
                embedding    = excluded.embedding,
                sample_count = excluded.sample_count,
                updated_at   = excluded.updated_at
            """,
            (entity_name, blob, sample_count, now),
        )
        self._conn.commit()
        logger.debug("Upserted voiceprint for %s (samples=%d)", entity_name, sample_count)

    def get(self, entity_name: str) -> Optional[StoredVoiceprint]:
        """Retrieve a voiceprint by entity name, or None if not found."""
        row = self._conn.execute(
            "SELECT entity_name, embedding, sample_count, updated_at "
            "FROM voiceprints WHERE entity_name = ?",
            (entity_name,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_stored(row)

    def all(self) -> list[StoredVoiceprint]:
        """Return all stored voiceprints."""
        rows = self._conn.execute(
            "SELECT entity_name, embedding, sample_count, updated_at FROM voiceprints"
        ).fetchall()
        return [_row_to_stored(r) for r in rows]

    def delete(self, entity_name: str) -> bool:
        """Remove a voiceprint from the local cache.

        Returns:
            True if a row was deleted, False if the entity was not found.
        """
        cursor = self._conn.execute(
            "DELETE FROM voiceprints WHERE entity_name = ?", (entity_name,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of stored voiceprints."""
        return self._conn.execute("SELECT COUNT(*) FROM voiceprints").fetchone()[0]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, query: np.ndarray) -> VoiceprintMatch:
        """Find the best matching enrolled speaker for a query embedding.

        Computes cosine similarity between the query and every stored
        voiceprint, returns the best match classified by confidence tier.

        If no voiceprints are enrolled, or the best score is below the
        probable threshold, a provisional entity name is returned derived
        from the SHA-256 hash of the query embedding.

        Args:
            query: float32 array of shape (VOICEPRINT_DIM,).

        Returns:
            VoiceprintMatch with entity_name, confidence, and confidence_level.
        """
        self._validate_embedding(query)
        candidates = self.all()

        if not candidates:
            provisional = _provisional_name(query)
            logger.debug("No enrolled speakers — provisional: %s", provisional)
            return VoiceprintMatch(
                entity_name=provisional,
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
            )

        best_name = ""
        best_score = -1.0
        for vp in candidates:
            score = self.cosine_similarity(query, vp.embedding)
            if score > best_score:
                best_score = score
                best_name = vp.entity_name

        level = self.classify_confidence(best_score)
        if level == ConfidenceLevel.UNKNOWN:
            best_name = _provisional_name(query)

        # Clamp to [0, 1]: negative cosine similarity still means "no match",
        # and VoiceprintMatch.confidence has ge=0 constraint.
        clamped_score = max(0.0, best_score)

        logger.debug(
            "Voiceprint match: %s (score=%.3f, level=%s)", best_name, best_score, level.value
        )
        return VoiceprintMatch(
            entity_name=best_name,
            confidence=clamped_score,
            confidence_level=level,
        )

    def update_after_match(
        self,
        entity_name: str,
        new_embedding: np.ndarray,
        weight: float = 0.1,
    ) -> None:
        """Update a stored voiceprint with a running weighted average.

        Called after each confident identification to refine the embedding
        over time.  If no embedding exists yet for the entity, stores the
        new embedding directly.

        Args:
            entity_name: Entity to update.
            new_embedding: Most recent utterance embedding.
            weight: Contribution of the new sample (default 0.1 → 10%).
        """
        self._validate_embedding(new_embedding)
        existing = self.get(entity_name)

        if existing is None:
            self.upsert(entity_name, new_embedding, sample_count=1)
            return

        updated = self.running_average(existing.embedding, new_embedding, weight)
        now = _utcnow()
        self._conn.execute(
            """
            UPDATE voiceprints
            SET embedding    = ?,
                sample_count = sample_count + 1,
                updated_at   = ?
            WHERE entity_name = ?
            """,
            (updated.astype(np.float32).tobytes(), now, entity_name),
        )
        self._conn.commit()
        logger.debug(
            "Updated voiceprint for %s (new sample_count=%d)",
            entity_name,
            existing.sample_count + 1,
        )

    # ------------------------------------------------------------------
    # Pure static helpers (tested independently)
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors.

        Returns 0.0 if either vector is the zero vector (degenerate case).

        Args:
            a: First vector (any dtype, any norm).
            b: Second vector (any dtype, any norm).

        Returns:
            Scalar in [-1.0, 1.0].
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def running_average(
        existing: np.ndarray,
        incoming: np.ndarray,
        weight: float = 0.1,
    ) -> np.ndarray:
        """Weighted running average, re-normalised to unit norm.

        Formula: blended = (1 - weight) * existing + weight * incoming
        Result is L2-normalised so embeddings remain unit vectors.

        Args:
            existing: Current stored embedding.
            incoming: New sample embedding.
            weight: Fraction of incoming to blend in (0.0–1.0).

        Returns:
            Unit-norm float32 array of the same shape.
        """
        blended = (1.0 - weight) * existing + weight * incoming
        norm = float(np.linalg.norm(blended))
        if norm == 0.0:
            return blended.astype(np.float32)
        return (blended / norm).astype(np.float32)

    @staticmethod
    def embedding_hash(embedding: np.ndarray) -> str:
        """Return the first 8 hex characters of the SHA-256 of the embedding.

        Used to generate stable provisional entity names for unknown speakers.
        """
        return hashlib.sha256(embedding.astype(np.float32).tobytes()).hexdigest()[:8]

    def classify_confidence(self, score: float) -> ConfidenceLevel:
        """Map a cosine similarity score to a ConfidenceLevel tier.

        Args:
            score: Cosine similarity in [-1.0, 1.0].

        Returns:
            CONFIDENT, PROBABLE, or UNKNOWN.
        """
        if score >= self.confident_threshold:
            return ConfidenceLevel.CONFIDENT
        if score >= self.probable_threshold:
            return ConfidenceLevel.PROBABLE
        return ConfidenceLevel.UNKNOWN

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> None:
        """Raise ValueError if embedding is not a VOICEPRINT_DIM-dim array."""
        if embedding.shape != (VOICEPRINT_DIM,):
            raise ValueError(
                f"Expected embedding shape ({VOICEPRINT_DIM},), got {embedding.shape}"
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _provisional_name(embedding: np.ndarray) -> str:
    """Generate a provisional entity name from an embedding hash."""
    h = hashlib.sha256(embedding.astype(np.float32).tobytes()).hexdigest()[:8]
    return f"unknown_voice_{h}"


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_stored(row: tuple) -> StoredVoiceprint:
    """Convert a raw SQLite row tuple to a StoredVoiceprint."""
    entity_name, blob, sample_count, updated_at = row
    return StoredVoiceprint(
        entity_name=entity_name,
        embedding=np.frombuffer(blob, dtype=np.float32).copy(),
        sample_count=sample_count,
        updated_at=datetime.fromisoformat(updated_at),
    )
