"""
VoiceprintMatcher — multi-encoder SQLite cache for speaker voiceprint matching.

Architecture
------------
memory-mcp is the canonical store for voiceprint embeddings.  This SQLite
database is a runtime cache with two tables:

  voiceprints       — per-encoder embeddings, rebuilt from memory-mcp or
                      recomputed from enrollment_audio on startup.
  enrollment_audio  — raw enrollment audio retained so embeddings can be
                      recomputed when switching speaker encoder backends.

Switching encoders
------------------
Change SPEAKER_ENCODER in .env and restart the pipeline worker.  On startup:

  1. If voiceprints exist for the new encoder → use them.
  2. If enrollment_audio exists but no voiceprints for the new encoder
     → POST /recompute_embeddings to recompute without re-recording.
  3. If neither → re-enroll speakers.

On-disk schema
--------------
    voiceprints (
        entity_name   TEXT,
        encoder       TEXT,     -- backend name: 'resemblyzer', 'ecapa_tdnn', 'titanet'
        embedding     BLOB,     -- float32 × encoder_dim × 4 bytes
        sample_count  INTEGER,
        updated_at    TEXT      -- ISO-8601 UTC
        PRIMARY KEY (entity_name, encoder)
    )

    enrollment_audio (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_name   TEXT,
        audio_blob    BLOB,     -- float32 mono PCM bytes (not WAV)
        sample_rate   INTEGER,
        duration_s    REAL,
        enrolled_at   TEXT,     -- ISO-8601 UTC
        room          TEXT
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
from pipeline_worker.speaker_encoder import ENCODER_DIMS

logger = logging.getLogger(__name__)

# Default cosine-similarity thresholds; may be overridden via config.
DEFAULT_CONFIDENT_THRESHOLD = 0.85
DEFAULT_PROBABLE_THRESHOLD = 0.70

_DDL = """
CREATE TABLE IF NOT EXISTS voiceprints (
    entity_name   TEXT NOT NULL,
    encoder       TEXT NOT NULL DEFAULT 'resemblyzer',
    embedding     BLOB NOT NULL,
    sample_count  INTEGER NOT NULL DEFAULT 1,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (entity_name, encoder)
);

CREATE TABLE IF NOT EXISTS enrollment_audio (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_name   TEXT NOT NULL,
    audio_blob    BLOB NOT NULL,
    sample_rate   INTEGER NOT NULL DEFAULT 16000,
    duration_s    REAL,
    enrolled_at   TEXT NOT NULL,
    room          TEXT
);
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StoredVoiceprint:
    """A single voiceprint record from the local cache."""

    entity_name: str
    encoder: str
    embedding: np.ndarray  # shape (dim,), dtype float32
    sample_count: int
    updated_at: datetime


@dataclass
class StoredEnrollmentAudio:
    """A raw audio record retained for re-embedding."""

    id: int
    entity_name: str
    audio: np.ndarray  # float32 mono PCM
    sample_rate: int
    duration_s: Optional[float]
    enrolled_at: datetime
    room: Optional[str]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class VoiceprintMatcher:
    """Match incoming speaker embeddings against a local per-encoder voiceprint cache.

    All reads and writes are scoped to the configured encoder_name so that
    embeddings from different backends never mix.

    Thread-safety: SQLite is opened in check_same_thread=False mode.  The
    caller is responsible for not issuing concurrent writes from multiple
    threads without external locking.  For the single-process pipeline worker
    this is not an issue.

    Usage::

        matcher = VoiceprintMatcher("voiceprints.sqlite", encoder_name="ecapa_tdnn")
        match = matcher.match(embedding)
        if match.confidence_level == ConfidenceLevel.CONFIDENT:
            matcher.update_after_match(match.entity_name, embedding)
    """

    def __init__(
        self,
        db_path: str | Path,
        encoder_name: str = "resemblyzer",
        confident_threshold: float = DEFAULT_CONFIDENT_THRESHOLD,
        probable_threshold: float = DEFAULT_PROBABLE_THRESHOLD,
    ) -> None:
        self.db_path = Path(db_path)
        self.encoder_name = encoder_name
        self.confident_threshold = confident_threshold
        self.probable_threshold = probable_threshold
        self._encoder_dim = ENCODER_DIMS.get(encoder_name, 256)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Open (or create) the SQLite database, migrate schema, ensure tables exist."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        _migrate_schema(self._conn)
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.info(
            "Voiceprint DB opened: %s (encoder=%s)", self.db_path, self.encoder_name
        )

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
    # Voiceprint CRUD — scoped to self.encoder_name
    # ------------------------------------------------------------------

    def upsert(
        self,
        entity_name: str,
        embedding: np.ndarray,
        sample_count: int = 1,
    ) -> None:
        """Insert or replace a voiceprint for the current encoder.

        Args:
            entity_name: The memory-mcp entity name.
            embedding: Unit-norm float32 array of shape (encoder_dim,).
            sample_count: Number of utterances averaged into this embedding.

        Raises:
            ValueError: If embedding has the wrong shape for this encoder.
        """
        self._validate_embedding(embedding)
        blob = embedding.astype(np.float32).tobytes()
        now = _utcnow()
        self._conn.execute(
            """
            INSERT INTO voiceprints (entity_name, encoder, embedding, sample_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(entity_name, encoder) DO UPDATE SET
                embedding    = excluded.embedding,
                sample_count = excluded.sample_count,
                updated_at   = excluded.updated_at
            """,
            (entity_name, self.encoder_name, blob, sample_count, now),
        )
        self._conn.commit()
        logger.debug(
            "Upserted voiceprint for %s [%s] (samples=%d)",
            entity_name, self.encoder_name, sample_count,
        )

    def get(self, entity_name: str) -> Optional[StoredVoiceprint]:
        """Retrieve a voiceprint for the current encoder, or None if not found."""
        row = self._conn.execute(
            "SELECT entity_name, encoder, embedding, sample_count, updated_at "
            "FROM voiceprints WHERE entity_name = ? AND encoder = ?",
            (entity_name, self.encoder_name),
        ).fetchone()
        if row is None:
            return None
        return _row_to_stored_voiceprint(row)

    def all(self) -> list[StoredVoiceprint]:
        """Return all stored voiceprints for the current encoder."""
        rows = self._conn.execute(
            "SELECT entity_name, encoder, embedding, sample_count, updated_at "
            "FROM voiceprints WHERE encoder = ?",
            (self.encoder_name,),
        ).fetchall()
        return [_row_to_stored_voiceprint(r) for r in rows]

    def delete(self, entity_name: str) -> bool:
        """Remove a voiceprint for the current encoder.

        Returns:
            True if a row was deleted, False if the entity was not found.
        """
        cursor = self._conn.execute(
            "DELETE FROM voiceprints WHERE entity_name = ? AND encoder = ?",
            (entity_name, self.encoder_name),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of stored voiceprints for the current encoder."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM voiceprints WHERE encoder = ?",
            (self.encoder_name,),
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Enrollment audio CRUD
    # ------------------------------------------------------------------

    def store_enrollment_audio(
        self,
        entity_name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        room: Optional[str] = None,
    ) -> int:
        """Persist raw enrollment audio for future re-embedding.

        Args:
            entity_name: The speaker entity name.
            audio: float32 mono PCM array.
            sample_rate: Audio sample rate.
            room: Optional room name for metadata.

        Returns:
            Row ID of the inserted record.
        """
        audio_blob = audio.astype(np.float32).tobytes()
        duration_s = len(audio) / sample_rate
        now = _utcnow()
        cursor = self._conn.execute(
            """
            INSERT INTO enrollment_audio
                (entity_name, audio_blob, sample_rate, duration_s, enrolled_at, room)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entity_name, audio_blob, sample_rate, duration_s, now, room),
        )
        self._conn.commit()
        logger.debug(
            "Stored enrollment audio for %s (%.1fs, id=%d)",
            entity_name, duration_s, cursor.lastrowid,
        )
        return cursor.lastrowid

    def get_all_enrollment_audio(self) -> list[StoredEnrollmentAudio]:
        """Return all retained enrollment audio records."""
        rows = self._conn.execute(
            "SELECT id, entity_name, audio_blob, sample_rate, duration_s, enrolled_at, room "
            "FROM enrollment_audio ORDER BY enrolled_at"
        ).fetchall()
        return [_row_to_enrollment_audio(r) for r in rows]

    def enrollment_audio_count(self) -> int:
        """Return the number of retained enrollment audio records."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM enrollment_audio"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, query: np.ndarray) -> VoiceprintMatch:
        """Find the best matching enrolled speaker for a query embedding.

        Computes cosine similarity between the query and every stored
        voiceprint for the current encoder.  Returns the best match
        classified by confidence tier.

        If no voiceprints are enrolled, or the best score is below the
        probable threshold, a provisional entity name derived from the
        SHA-256 hash of the query embedding is returned.

        Args:
            query: float32 array of shape (encoder_dim,).

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

        clamped_score = max(0.0, best_score)

        logger.debug(
            "Voiceprint match: %s (score=%.3f, level=%s, encoder=%s)",
            best_name, best_score, level.value, self.encoder_name,
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
        over time.

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
            WHERE entity_name = ? AND encoder = ?
            """,
            (updated.astype(np.float32).tobytes(), now, entity_name, self.encoder_name),
        )
        self._conn.commit()
        logger.debug(
            "Updated voiceprint for %s [%s] (new sample_count=%d)",
            entity_name, self.encoder_name, existing.sample_count + 1,
        )

    # ------------------------------------------------------------------
    # Pure static helpers (tested independently)
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors.

        Returns 0.0 if either vector is the zero vector.

        Args:
            a: First vector.
            b: Second vector.

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
        """Return the first 8 hex characters of the SHA-256 of the embedding."""
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

    def _validate_embedding(self, embedding: np.ndarray) -> None:
        """Raise ValueError if embedding shape does not match this encoder's dim."""
        expected = (self._encoder_dim,)
        if embedding.shape != expected:
            raise ValueError(
                f"Expected embedding shape {expected} for encoder '{self.encoder_name}', "
                f"got {embedding.shape}"
            )


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Migrate old single-encoder schema to multi-encoder schema if needed.

    The old schema had entity_name as sole PRIMARY KEY with no encoder column.
    The new schema has (entity_name, encoder) as composite PRIMARY KEY.
    """
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    if "voiceprints" not in tables:
        return  # fresh database — DDL will create correct schema

    cols = {row[1] for row in conn.execute("PRAGMA table_info(voiceprints)").fetchall()}
    if "encoder" in cols:
        return  # already migrated

    logger.info("Migrating voiceprints table to multi-encoder schema...")
    conn.executescript("""
        ALTER TABLE voiceprints RENAME TO voiceprints_v1;

        CREATE TABLE voiceprints (
            entity_name   TEXT NOT NULL,
            encoder       TEXT NOT NULL DEFAULT 'resemblyzer',
            embedding     BLOB NOT NULL,
            sample_count  INTEGER NOT NULL DEFAULT 1,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY (entity_name, encoder)
        );

        INSERT INTO voiceprints (entity_name, encoder, embedding, sample_count, updated_at)
        SELECT entity_name, 'resemblyzer', embedding, sample_count, updated_at
        FROM voiceprints_v1;

        DROP TABLE voiceprints_v1;
    """)
    conn.commit()
    logger.info("Schema migration complete — existing voiceprints tagged as 'resemblyzer'")


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


def _row_to_stored_voiceprint(row: tuple) -> StoredVoiceprint:
    """Convert a raw SQLite row to a StoredVoiceprint."""
    entity_name, encoder, blob, sample_count, updated_at = row
    return StoredVoiceprint(
        entity_name=entity_name,
        encoder=encoder,
        embedding=np.frombuffer(blob, dtype=np.float32).copy(),
        sample_count=sample_count,
        updated_at=datetime.fromisoformat(updated_at),
    )


def _row_to_enrollment_audio(row: tuple) -> StoredEnrollmentAudio:
    """Convert a raw SQLite row to a StoredEnrollmentAudio."""
    id_, entity_name, audio_blob, sample_rate, duration_s, enrolled_at, room = row
    return StoredEnrollmentAudio(
        id=id_,
        entity_name=entity_name,
        audio=np.frombuffer(audio_blob, dtype=np.float32).copy(),
        sample_rate=sample_rate,
        duration_s=duration_s,
        enrolled_at=datetime.fromisoformat(enrolled_at),
        room=room,
    )
