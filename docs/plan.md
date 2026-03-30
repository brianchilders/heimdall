# Heimdall — Development Plan

## Project Summary

Distributed home audio intelligence system. Room nodes (Pi 5 + Hailo-8 + ReSpeaker USB) capture
speech, identify speakers, and stream structured payloads to the pipeline worker (co-located on
the Pi 5), which resolves identity, extracts facts, and writes to memory-mcp on the Pi 4.

## Guiding Rules

- Every phase: **plan → code → test → verify → document**
- Tests live next to the code they test (`tests/` subfolder per component)
- Docs live in `docs/` (one file per component) plus inline docstrings
- CPU/file fallbacks everywhere so dev and test never require Pi hardware
- Nothing is discarded — unknown speakers get provisional entities immediately

---

## Repository Layout

```
heimdall/
├── docs/
│   ├── plan.md                     # this file
│   ├── memory-extensions.md        # voice route API reference
│   ├── enrollment.md               # enrollment CLI usage
│   ├── pipeline-worker.md          # pipeline worker API + internals
│   ├── room-node.md                # room node setup, hardware, systemd
│   └── payload-schema.md           # canonical AudioPayload format
│
├── memory_extensions/
│   ├── voice_routes.py             # 4 new routes to splice into memory-mcp api.py
│   └── tests/
│       └── test_voice_routes.py
│
├── enrollment/
│   ├── enroll.py                   # CLI: record or load WAV → embed → POST to memory-mcp
│   ├── requirements.txt
│   └── tests/
│       └── test_enroll.py
│
├── pipeline_worker/
│   ├── server.py                   # FastAPI app, port 8001
│   ├── voiceprint.py               # resemblyzer matching + voiceprints.sqlite
│   ├── diarize.py                  # pyannote full diarization (low-confidence fallback)
│   ├── memory_client.py            # HTTP client for memory-mcp (port 8900)
│   ├── models.py                   # Pydantic: AudioPayload, PipelineResponse
│   ├── main.py                     # entry point (uvicorn)
│   ├── requirements.txt
│   └── tests/
│       ├── test_voiceprint.py
│       ├── test_memory_client.py
│       └── test_server.py
│
├── room_node/
│   ├── capture.py                  # sounddevice + Silero VAD
│   ├── doa.py                      # ReSpeaker USB HID → DOA degrees
│   ├── hailo_inference.py          # Whisper + emotion on Hailo-8 (faster-whisper fallback)
│   ├── sender.py                   # package payload + POST to pipeline worker
│   ├── config.py                   # env-driven config (room name, URLs, thresholds)
│   ├── main.py                     # entry point, systemd target
│   ├── requirements.txt
│   └── tests/
│       ├── test_capture.py         # WAV file injection, no hardware needed
│       ├── test_doa.py
│       └── test_sender.py
│
└── CONTEXT.md                      # project context reference
```

---

## Canonical AudioPayload (room node → pipeline worker)

Defined in `pipeline_worker/models.py`, documented in `docs/payload-schema.md`.

```json
{
  "room": "kitchen",
  "timestamp": "2026-03-22T10:30:00Z",
  "transcript": "I need to pick up groceries tomorrow",
  "whisper_confidence": 0.92,
  "whisper_model": "small",
  "doa": 247,
  "emotion": { "valence": 0.6, "arousal": 0.3 },
  "audio_clip_b64": null
}
```

`audio_clip_b64` is populated only when `whisper_confidence < WHISPER_CONFIDENCE_THRESHOLD`
(0.85) or voiceprint match is below the "probable" threshold (0.70). This triggers
large-v3 fallback and full pyannote diarization on the pipeline worker (Pi 5).

---

## Phase 1 — memory-mcp Voice Extensions

**Target machine**: memory-mcp Pi 4 (splice into running memory-mcp)
**No new dependencies** — uses memory-mcp's existing SQLite + entity model

### What

Four new HTTP routes added to memory-mcp via a router in `voice_routes.py`:

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/voices/unknown` | List all entities where `meta.status == "unenrolled"`, with sample transcript |
| `POST` | `/voices/enroll` | Rename provisional entity → real name, set status = "enrolled" |
| `POST` | `/voices/merge` | Merge provisional into existing entity (copy memories/readings, update voiceprint) |
| `POST` | `/voices/update_print` | Replace or update (running average) voiceprint embedding for an entity |

Voiceprints stored as JSON array in entity metadata field `voiceprint` (256-dim float list).
Status tracked in `meta.status`: `"unenrolled"` | `"enrolled"`.

### Steps

1. **Plan** — define Pydantic request/response models, confirm entity metadata schema in memory-mcp
2. **Code** — `memory_extensions/voice_routes.py` as a FastAPI `APIRouter`
3. **Test** — `tests/test_voice_routes.py`: create provisional entity, call each route, assert DB state
4. **Verify** — splice router into memory-mcp `api.py`, `curl` each endpoint against live server
5. **Document** — `docs/memory-extensions.md`: route reference, request/response schemas, integration snippet for api.py

### Acceptance Criteria

- All 4 routes return correct responses with valid entity data
- `/voices/enroll` renames entity and all related memories/readings remain attached (entity_id unchanged)
- `/voices/merge` copies all Tier 1 and Tier 2 data from source to target, deletes source
- `/voices/update_print` computes running average: `new = 0.9 * existing + 0.1 * incoming`
- 100% test coverage on route handlers

---

## Phase 2 — Enrollment CLI

**Target machine**: any machine with a mic (or a WAV file)
**Deps**: `resemblyzer`, `sounddevice`, `httpx`, `numpy`

### What

`enrollment/enroll.py` — CLI tool to enroll a new speaker.

```
python enroll.py --name Brian --room office --duration 10
python enroll.py --name Sarah --wav path/to/sarah.wav
python enroll.py --list               # show enrolled speakers
python enroll.py --unknown            # show unenrolled provisional voices
```

Flow:
1. Record N seconds of audio (or load WAV file)
2. Compute 256-dim resemblyzer embedding
3. `POST /entities` to memory-mcp — create entity of type `person`
4. `POST /voices/update_print` — attach voiceprint embedding
5. `GET /profile/{name}` — verify and print confirmation

### Steps

1. **Plan** — CLI argument design, audio recording parameters (16kHz mono, match ReSpeaker)
2. **Code** — enroll.py with argparse, resemblyzer embed, memory-mcp HTTP calls
3. **Test** — tests/test_enroll.py: mock memory-mcp, test with fixture WAV files, assert correct embedding shape
4. **Verify** — enroll Brian and Sarah against live memory-mcp, confirm via `/profile/{name}`
5. **Document** — `docs/enrollment.md`: usage, flags, WAV file format requirements, troubleshooting

### Acceptance Criteria

- Enrollment from mic and from WAV file both produce valid 256-dim embeddings
- Entity appears in memory-mcp `/profile/{name}` with voiceprint in metadata
- `--list` shows all enrolled speakers with enrollment date
- `--unknown` shows provisional entities from live audio (round-trips with Phase 1)

---

## Phase 3 — Pipeline Worker

**Target machine**: heimdall (Pi 5), port 8001
**Deps**: `fastapi`, `uvicorn`, `resemblyzer`, `pyannote-audio`, `httpx`, `numpy`, `python-dotenv`

### What

FastAPI service that receives `AudioPayload` from room nodes, resolves speaker identity,
and writes to memory-mcp.

```
room node  →  POST /ingest  →  voiceprint match
                                   ↓
                             resolve entity name
                                   ↓
                        POST /record     (Tier 2 — voice_activity reading)
                        POST /log_turn   (Tier 1.5 — transcript turn)
                        POST /extract_and_remember  (Tier 1 — fact extraction)
                                   ↓
                          update voiceprint embedding
```

**voiceprints.sqlite schema** (pipeline_worker local DB — fast runtime matching):
```sql
CREATE TABLE voiceprints (
    entity_name TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,   -- numpy float32 array, 256-dim
    updated_at  TEXT NOT NULL
);
```

**Confidence gating**:
- `>= 0.85` (confident): log under entity name
- `>= 0.70` (probable): log under entity name + send HA notification
- `< 0.70` (unknown): create provisional entity `unknown_voice_{hash}`, log there
- If audio_clip_b64 present: run pyannote diarization + large-v3 whisper fallback first

### Steps

1. **Plan** — finalize `AudioPayload` model, voiceprints.sqlite schema, matching algorithm,
   memory-mcp call sequence
2. **Code**:
   - `models.py` — `AudioPayload`, `PipelineResponse`, `ConfidenceLevel` enum
   - `voiceprint.py` — `VoiceprintMatcher`: load DB, cosine similarity, update running average
   - `memory_client.py` — async httpx client wrapping all memory-mcp endpoints used
   - `diarize.py` — `DiarizationFallback`: pyannote pipeline, only loaded when needed
   - `server.py` — FastAPI app with `/ingest`, `/health`, `/reload_voiceprints`
   - `main.py` — uvicorn entry point
3. **Test**:
   - `test_voiceprint.py` — similarity math, threshold logic, running average update
   - `test_memory_client.py` — all memory-mcp calls with httpx mock
   - `test_server.py` — POST /ingest with fixture payloads, assert correct memory-mcp calls
4. **Verify** — send mock payloads via curl, confirm records appear in memory-mcp
5. **Document** — `docs/pipeline-worker.md`: API reference, deployment, env vars, voiceprint DB schema

### Acceptance Criteria

- `POST /ingest` correctly routes confident/probable/unknown payloads
- Provisional entities created for unknowns with hash-based names
- Voiceprint running average updated after each confident match
- `POST /reload_voiceprints` hot-reloads embeddings without restart (needed after enrollment)
- All three memory-mcp write paths tested with mocked responses
- Service starts cleanly from `.env` via `python main.py`

---

## Phase 4 — Room Node

**Target machine**: heimdall (Pi 5), alongside pipeline_worker
**Deps**: `sounddevice`, `silero-vad`, `hailo`, `faster-whisper`, `httpx`, `numpy`, `python-dotenv`

### What

Six modules running as a single process (systemd service) per room.

**capture.py** — ReSpeaker USB audio capture
- sounddevice stream at 16kHz mono PCM
- Silero VAD gates: buffer speech segments, discard silence
- Yields audio chunks above VAD threshold

**doa.py** — Direction of Arrival from ReSpeaker USB HID
- Reads DOA angle (0–359°) from USB HID interface
- Sampled at utterance start, attached to payload

**hailo_inference.py** — On-device ML
- Primary: Hailo-8 compiled `.hef` models for Whisper small + emotion
- Fallback: `faster-whisper` (CPU) when Hailo unavailable or model not compiled
- Returns: `(transcript, whisper_confidence, emotion_dict)`

**sender.py** — Payload packaging and dispatch
- Builds `AudioPayload` dict
- Attaches base64 audio clip when confidence below threshold
- POST to `BLACKMAGIC_URL/ingest` via httpx with retry (points to localhost:8001 on heimdall)

**config.py** — All config from `.env`
- `ROOM_NAME`, `BLACKMAGIC_URL`, `DEVICE_INDEX`
- `WHISPER_CONFIDENCE_THRESHOLD` (default 0.85)
- `HAILO_ENABLED` (default true, set false for CPU-only dev)

**main.py** — Entry point
- Starts capture loop
- Dispatches VAD segments through inference → sender pipeline
- Handles graceful shutdown (SIGTERM for systemd)

### Hailo-8 Model Note

Hailo models must be compiled to `.hef` format from the Hailo Model Zoo.
`hailo_inference.py` checks for `.hef` file at startup; falls back to faster-whisper
automatically. Initial dev/test uses CPU fallback exclusively — no Hailo hardware needed
until integration testing on Pi.

### Steps

1. **Plan** — VAD buffer sizing, DOA HID protocol (USB vendor/product IDs, HID report format),
   Hailo SDK API surface, payload assembly
2. **Code** — all 6 modules, CPU fallback paths throughout
3. **Test** (no hardware needed):
   - `test_capture.py` — inject WAV file as sounddevice mock, assert VAD segments
   - `test_doa.py` — mock HID device, assert DOA parsing
   - `test_sender.py` — mock httpx, assert payload structure and retry logic
4. **Verify** — full end-to-end on Pi hardware: speak → payload arrives at pipeline worker → appears in memory-mcp
5. **Document**:
   - `docs/room-node.md`: hardware setup, USB HID notes, systemd unit file, env vars
   - Hailo model compilation steps (separate section)
   - `systemd/heimdall-room-node.service` — unit file template

### Acceptance Criteria

- VAD correctly gates on speech, discards silence (verified against WAV fixtures)
- DOA value attached to every payload
- Hailo fallback to faster-whisper transparent (same output shape)
- Payload POSTed successfully to pipeline worker
- Systemd service starts on boot, restarts on failure
- All unit tests pass without Pi hardware

---

## Cross-Cutting: What "Done" Means Per Phase

Every phase is complete when:

1. Code written and linted
2. All tests pass (`pytest` with no skips except hardware-gated)
3. Verified against live service (curl / manual run)
4. Docs written (`docs/<component>.md` + inline docstrings on all public functions)
5. `.env.example` committed with all required variables

---

## Build Order and Dependencies

```
Phase 1: memory-mcp voice extensions
    (no deps — builds on existing memory-mcp)
    ↓
Phase 2: Enrollment CLI
    (needs Phase 1 voice routes to store voiceprints)
    ↓
Phase 3: Pipeline Worker
    (needs Phase 1 routes + Phase 2 enrollments to test matching)
    ↓
Phase 4: Room Node
    (needs Phase 3 running to receive payloads)
```

Phases 1 and 2 can be developed on any machine.
Phases 3 and 4 both target heimdall (Pi 5) but are unit-testable locally without hardware.
