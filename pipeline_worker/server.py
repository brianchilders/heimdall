"""
Pipeline worker FastAPI application.

Endpoints
---------
POST /ingest                — receive AudioPayload from a room node
POST /enroll                — enroll a speaker from raw audio
POST /reload_voiceprints    — hot-reload voiceprint cache from memory-mcp
POST /recompute_embeddings  — recompute all voiceprints from stored audio
GET  /health                — liveness check

Ingest flow
-----------
1. Validate AudioPayload
2. If audio_clip_b64 present → re-transcribe with Whisper large-v3
3. Compute speaker embedding via configured SpeakerEncoder backend
4. Match embedding against local voiceprint cache → entity_name + confidence_level
5. Write to memory-mcp (record, session, log_turn, extract_and_remember)
6. If CONFIDENT → update local voiceprint cache + POST /voices/update_print
7. If PROBABLE  → POST HA webhook notification
8. Return PipelineResponse

Enroll flow
-----------
1. Validate EnrollRequest (audio_b64 + entity_name)
2. Decode audio
3. Compute embedding via configured SpeakerEncoder
4. Optionally store raw audio for future re-embedding (STORE_ENROLLMENT_AUDIO=true)
5. Upsert embedding in local voiceprint cache (current encoder)
6. Create/confirm entity in memory-mcp
7. Store voiceprint in memory-mcp
8. Return EnrollResponse
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pipeline_worker.diarize import DiarizationFallback, decode_audio
from pipeline_worker.memory_client import MemoryClient
from pipeline_worker.models import (
    AudioPayload,
    ConfidenceLevel,
    EnrollRequest,
    EnrollResponse,
    PipelineResponse,
)
from pipeline_worker.settings import Settings
from pipeline_worker.speaker_encoder import SpeakerEncoder, load_encoder
from pipeline_worker.voiceprint import VoiceprintMatcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state container
# ---------------------------------------------------------------------------


class AppState:
    """Holds all shared service-lifetime objects."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.encoder: SpeakerEncoder = load_encoder(
            settings.speaker_encoder,
            device=settings.speaker_encoder_device,
        )
        self.matcher = VoiceprintMatcher(
            db_path=settings.voiceprint_db,
            encoder_name=settings.speaker_encoder,
            confident_threshold=settings.voiceprint_confident_threshold,
            probable_threshold=settings.voiceprint_probable_threshold,
        )
        self.memory = MemoryClient(settings.memory_mcp_url, token=settings.memory_mcp_token)
        self.fallback = DiarizationFallback()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        settings: Optional Settings instance (defaults to loading from env).

    Returns:
        Configured FastAPI app with all routes registered.
    """
    if settings is None:
        settings = Settings()

    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(settings)
        app.state.app_state = state

        # Auto-recompute voiceprints if we have enrollment audio but no
        # embeddings for the current encoder (e.g. after switching backends).
        if state.matcher.count() == 0 and state.matcher.enrollment_audio_count() > 0:
            logger.info(
                "No voiceprints for encoder=%s — recomputing from %d stored audio samples...",
                settings.speaker_encoder,
                state.matcher.enrollment_audio_count(),
            )
            count = await _recompute_from_audio(state)
            logger.info("Auto-recomputed %d voiceprints", count)

        logger.info(
            "Pipeline worker starting — encoder=%s voiceprints=%d memory-mcp=%s",
            settings.speaker_encoder,
            state.matcher.count(),
            settings.memory_mcp_url,
        )
        yield
        await state.memory.aclose()
        state.matcher.close()
        logger.info("Pipeline worker shut down cleanly")

    app = FastAPI(
        title="Heimdall Pipeline Worker",
        description=(
            "Receives room-node audio payloads, resolves speaker identity, "
            "writes to memory-mcp."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health(request: Request) -> dict:
        """Liveness check.

        Returns encoder, voiceprint cache count, and memory-mcp URL.
        """
        state: AppState = request.app.state.app_state
        return {
            "ok": True,
            "encoder": settings.speaker_encoder,
            "voiceprints_cached": state.matcher.count(),
            "enrollment_audio_stored": state.matcher.enrollment_audio_count(),
            "memory_mcp_url": settings.memory_mcp_url,
        }

    @app.post("/ingest", response_model=PipelineResponse)
    async def ingest(payload: AudioPayload, request: Request) -> PipelineResponse:
        """Receive an AudioPayload from a room node and process it."""
        state: AppState = request.app.state.app_state
        return await _process_payload(payload, state)

    @app.post("/enroll", response_model=EnrollResponse)
    async def enroll(body: EnrollRequest, request: Request) -> EnrollResponse:
        """Enroll a speaker from raw audio.

        The pipeline worker computes the embedding using its configured
        SpeakerEncoder, stores it in the local cache, and syncs to memory-mcp.
        Raw audio is retained for future re-embedding when switching encoders.
        """
        state: AppState = request.app.state.app_state
        return await _enroll_speaker(body, state)

    @app.post("/reload_voiceprints")
    async def reload_voiceprints(request: Request) -> dict:
        """Hot-reload the voiceprint cache from memory-mcp.

        Fetches all enrolled entities from memory-mcp and repopulates the
        local SQLite cache for the current encoder.  Only loads voiceprints
        that match the current encoder's dimension.
        """
        state: AppState = request.app.state.app_state
        count = await _reload_from_memory_mcp(state)
        return {"ok": True, "voiceprints_loaded": count, "encoder": settings.speaker_encoder}

    @app.post("/recompute_embeddings")
    async def recompute_embeddings(request: Request) -> dict:
        """Recompute all voiceprint embeddings from stored enrollment audio.

        Use this after switching SPEAKER_ENCODER to recompute embeddings
        from retained audio without re-recording.

        Returns:
            Count of speakers successfully recomputed.
        """
        state: AppState = request.app.state.app_state
        count = await _recompute_from_audio(state)
        return {"ok": True, "recomputed": count, "encoder": settings.speaker_encoder}

    # ------------------------------------------------------------------
    # Exception handler
    # ------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception in %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "detail": str(exc)})

    return app


# ---------------------------------------------------------------------------
# Core ingest logic
# ---------------------------------------------------------------------------


async def _process_payload(payload: AudioPayload, state: AppState) -> PipelineResponse:
    """Execute the full ingest pipeline for one AudioPayload."""
    flags: list[str] = []
    final_transcript = payload.transcript
    audio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Step 1 — decode audio and re-transcribe if clip present
    # ------------------------------------------------------------------
    if payload.audio_clip_b64:
        logger.info("Audio clip present — running fallback (large-v3 + %s)", state.settings.speaker_encoder)
        try:
            audio = decode_audio(payload.audio_clip_b64)
            transcript = state.fallback.process(payload.audio_clip_b64)
            if transcript:
                final_transcript = transcript
                flags.append("fallback_transcription_used")
        except Exception as exc:
            logger.error("Fallback processing failed: %s", exc)
            flags.append("fallback_failed")

    # ------------------------------------------------------------------
    # Step 2 — compute speaker embedding
    # ------------------------------------------------------------------
    embedding: Optional[np.ndarray] = None

    if audio is not None:
        embedding = state.encoder.embed(audio)
    elif payload.voiceprint is not None:
        # Full-node pre-computed embedding (future path)
        candidate = np.array(payload.voiceprint, dtype=np.float32)
        if candidate.shape == (state.encoder.dim,):
            embedding = candidate
        else:
            logger.warning(
                "Payload voiceprint dim %d does not match encoder %s dim %d — skipping",
                len(payload.voiceprint), state.settings.speaker_encoder, state.encoder.dim,
            )

    # ------------------------------------------------------------------
    # Step 3 — voiceprint matching
    # ------------------------------------------------------------------
    if embedding is not None:
        match = state.matcher.match(embedding)
    else:
        logger.warning("No embedding available — logging as unknown")
        import hashlib
        key = f"{payload.room}:{payload.timestamp.isoformat()}"
        h = hashlib.sha256(key.encode()).hexdigest()[:8]
        from pipeline_worker.models import VoiceprintMatch
        match = VoiceprintMatch(
            entity_name=f"unknown_voice_{h}",
            confidence=0.0,
            confidence_level=ConfidenceLevel.UNKNOWN,
        )

    entity_name = match.entity_name
    confidence_level = match.confidence_level

    if confidence_level == ConfidenceLevel.PROBABLE:
        flags.append("probable_match")
    elif confidence_level == ConfidenceLevel.UNKNOWN:
        flags.append("unknown_speaker")

    logger.info(
        "Ingest: room=%s entity=%s level=%s transcript=%r",
        payload.room,
        entity_name,
        confidence_level.value,
        (final_transcript or "")[:60],
    )

    # ------------------------------------------------------------------
    # Step 4 — write to memory-mcp
    # ------------------------------------------------------------------
    voice_activity_value = {
        "transcript": final_transcript,
        "confidence": match.confidence,
        "doa": payload.doa,
        "room": payload.room,
        "whisper_model": payload.whisper_model,
        "whisper_confidence": payload.whisper_confidence,
        "speaker_confidence": match.confidence,
        "speaker_encoder": state.settings.speaker_encoder,
    }
    if payload.emotion:
        voice_activity_value["emotion"] = payload.emotion.model_dump()

    await state.memory.record(
        entity_name=entity_name,
        metric="voice_activity",
        value=voice_activity_value,
    )

    session_id: Optional[str] = None
    session_result = await state.memory.open_session(entity_name)
    if session_result:
        raw_id = session_result.get("result")
        if isinstance(raw_id, int):
            session_id = str(raw_id)
            await state.memory.log_turn(
                session_id=raw_id,
                role="system",
                content=f"room={payload.room} doa={payload.doa} confidence={confidence_level.value}",
            )
            if final_transcript:
                await state.memory.log_turn(
                    session_id=raw_id,
                    role="user",
                    content=final_transcript,
                )

    if final_transcript:
        await state.memory.extract_and_remember(entity_name, final_transcript)

    # ------------------------------------------------------------------
    # Step 5 — update voiceprint if confident
    # ------------------------------------------------------------------
    if confidence_level == ConfidenceLevel.CONFIDENT and embedding is not None:
        state.matcher.update_after_match(entity_name, embedding)
        await state.memory.update_voiceprint(entity_name, embedding.tolist())

    # ------------------------------------------------------------------
    # Step 6 — HA notification if probable
    # ------------------------------------------------------------------
    if confidence_level == ConfidenceLevel.PROBABLE and state.settings.ha_webhook_url:
        await _notify_ha(state, entity_name, match.confidence, payload.room)

    return PipelineResponse(
        ok=True,
        entity_name=entity_name,
        confidence_level=confidence_level,
        transcript=final_transcript,
        session_id=session_id,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Enroll logic
# ---------------------------------------------------------------------------


async def _enroll_speaker(body: EnrollRequest, state: AppState) -> EnrollResponse:
    """Enroll a speaker: compute embedding, store audio, sync to memory-mcp."""
    audio = decode_audio(body.audio_b64)

    # Compute embedding with the configured encoder
    embedding = state.encoder.embed(audio, body.sample_rate)
    if embedding is None:
        raise ValueError(
            f"Failed to compute embedding for '{body.entity_name}' — "
            "audio may be too short or noisy"
        )

    # Store raw audio for future re-embedding
    audio_stored = False
    if state.settings.store_enrollment_audio:
        state.matcher.store_enrollment_audio(
            entity_name=body.entity_name,
            audio=audio,
            sample_rate=body.sample_rate,
            room=body.room,
        )
        audio_stored = True

    # Update local voiceprint cache
    existing = state.matcher.get(body.entity_name)
    if existing is not None:
        # Blend new sample into existing voiceprint (equal weight with existing)
        weight = 1.0 / (existing.sample_count + 1)
        updated = VoiceprintMatcher.running_average(existing.embedding, embedding, weight)
        state.matcher.upsert(body.entity_name, updated, sample_count=existing.sample_count + 1)
        embedding = updated
    else:
        state.matcher.upsert(body.entity_name, embedding, sample_count=1)

    stored = state.matcher.get(body.entity_name)
    sample_count = stored.sample_count if stored else 1

    # Register entity in memory-mcp
    await state.memory.remember(
        entity_name=body.entity_name,
        entity_type="person",
        fact=f"{body.entity_name} is an enrolled speaker in the Heimdall audio pipeline.",
        category="enrollment",
        source="pipeline_worker_enroll",
        meta={"status": "enrolled", "speaker_encoder": state.settings.speaker_encoder},
    )

    # Store voiceprint in memory-mcp
    await state.memory.update_voiceprint(body.entity_name, embedding.tolist())

    logger.info(
        "Enrolled: entity=%s encoder=%s samples=%d audio_stored=%s",
        body.entity_name, state.settings.speaker_encoder, sample_count, audio_stored,
    )

    return EnrollResponse(
        ok=True,
        entity_name=body.entity_name,
        encoder=state.settings.speaker_encoder,
        embedding_norm=float(np.linalg.norm(embedding)),
        audio_stored=audio_stored,
        sample_count=sample_count,
    )


# ---------------------------------------------------------------------------
# Reload and recompute
# ---------------------------------------------------------------------------


async def _reload_from_memory_mcp(state: AppState) -> int:
    """Reload the local voiceprint cache from memory-mcp enrolled entities.

    Only loads voiceprints whose embedding dimension matches the current
    encoder.  Skips entries with incompatible dimensions.
    """
    try:
        response = await state.memory._client.get("/entities")
        response.raise_for_status()
        entities = response.json().get("entities", [])
    except Exception as exc:
        logger.error("Failed to fetch entities from memory-mcp: %s", exc)
        return 0

    count = 0
    for entity in entities:
        meta = entity.get("meta") or {}
        vp = meta.get("voiceprint")
        if not vp or len(vp) != state.encoder.dim:
            continue
        name = entity.get("name")
        samples = meta.get("voiceprint_samples", 1)
        embedding = np.array(vp, dtype=np.float32)
        state.matcher.upsert(name, embedding, sample_count=samples)
        count += 1

    logger.info(
        "Voiceprint cache reloaded from memory-mcp: %d embeddings (encoder=%s)",
        count, state.settings.speaker_encoder,
    )
    return count


async def _recompute_from_audio(state: AppState) -> int:
    """Recompute voiceprint embeddings from all stored enrollment audio.

    Groups audio records by entity, averages the embeddings, and upserts
    the result into the local cache and memory-mcp.

    Returns:
        Number of entities successfully recomputed.
    """
    records = state.matcher.get_all_enrollment_audio()
    if not records:
        logger.info("No enrollment audio found — nothing to recompute")
        return 0

    # Group by entity
    by_entity: dict[str, list] = {}
    for record in records:
        by_entity.setdefault(record.entity_name, []).append(record)

    count = 0
    for entity_name, entity_records in by_entity.items():
        embeddings = []
        for record in entity_records:
            emb = state.encoder.embed(record.audio, record.sample_rate)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.warning("No valid embeddings for %s — skipping", entity_name)
            continue

        avg = np.mean(embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(avg))
        if norm > 0:
            avg = avg / norm

        state.matcher.upsert(entity_name, avg, sample_count=len(embeddings))
        await state.memory.update_voiceprint(entity_name, avg.tolist())
        count += 1
        logger.info(
            "Recomputed voiceprint for %s: %d clips averaged (encoder=%s)",
            entity_name, len(embeddings), state.settings.speaker_encoder,
        )

    return count


# ---------------------------------------------------------------------------
# HA notification
# ---------------------------------------------------------------------------


async def _notify_ha(
    state: AppState,
    entity_name: str,
    confidence: float,
    room: str,
) -> None:
    """POST a probable-match notification to the Home Assistant webhook."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as ha:
            await ha.post(
                state.settings.ha_webhook_url,
                json={
                    "entity_name": entity_name,
                    "confidence": confidence,
                    "room": room,
                    "message": (
                        f"Probable speaker match: {entity_name} "
                        f"(confidence {confidence:.0%}) in {room}"
                    ),
                },
            )
    except Exception as exc:
        logger.warning("HA notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


# Module-level app instance for uvicorn
app = create_app()
