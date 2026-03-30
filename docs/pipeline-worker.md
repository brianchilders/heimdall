# Pipeline Worker

## Purpose

FastAPI service running on the Pi 5 (port 8001), co-located with the room node.
Receives `AudioPayload` from the room node, resolves speaker identity via
voiceprint matching, and writes transcripts and extracted facts to memory-mcp
(running on the Pi 4 at `http://memory-mcp:8900`).

---

## Architecture

```
room node  →  POST /ingest
                  │
                  ├─ audio_clip_b64 present?
                  │    └─ DiarizationFallback.process()
                  │         ├─ large-v3 Whisper re-transcription
                  │         └─ resemblyzer re-embedding
                  │
                  ├─ VoiceprintMatcher.match(embedding)
                  │    ├─ CONFIDENT (≥0.85) → entity name
                  │    ├─ PROBABLE  (≥0.70) → entity name + HA notification
                  │    └─ UNKNOWN   (<0.70) → unknown_voice_{hash}
                  │
                  ├─ POST /record           → memory-mcp Tier 2
                  ├─ POST /open_session     → memory-mcp Tier 1.5
                  ├─ POST /log_turn         → memory-mcp Tier 1.5
                  ├─ POST /extract_and_remember → memory-mcp Tier 1
                  │
                  └─ if CONFIDENT:
                       VoiceprintMatcher.update_after_match()
                       POST /voices/update_print → memory-mcp
```

### Key files

| File | Responsibility |
|------|---------------|
| `server.py` | FastAPI app, route handlers, ingest orchestration |
| `voiceprint.py` | SQLite voiceprint cache, matching, running average |
| `memory_client.py` | Async httpx client for all memory-mcp endpoints |
| `diarize.py` | Lazy-loaded large-v3 Whisper + resemblyzer fallback |
| `settings.py` | Pydantic-settings config from .env |
| `models.py` | AudioPayload, PipelineResponse, ConfidenceLevel |
| `main.py` | uvicorn entry point |

---

## API

### `POST /ingest`

Receive and process one utterance from a room node.

**Request:** `AudioPayload` JSON — see `docs/payload-schema.md`

**Response:**
```json
{
  "ok": true,
  "entity_name": "Brian",
  "confidence_level": "confident",
  "transcript": "I need to pick up groceries tomorrow",
  "session_id": "sess_abc123",
  "flags": []
}
```

Possible `flags`:
- `fallback_transcription_used` — large-v3 replaced room node transcript
- `fallback_failed` — large-v3 failed; original transcript used
- `probable_match` — confidence 0.70–0.85; HA notified if configured
- `unknown_speaker` — new provisional entity created

---

### `GET /health`

```json
{
  "ok": true,
  "voiceprints_cached": 2,
  "memory_mcp_url": "http://localhost:8900"
}
```

---

### `POST /reload_voiceprints`

Hot-reload the local voiceprint SQLite cache from memory-mcp enrolled entities.
Call this after `enroll.py` adds a new speaker — the pipeline worker
immediately starts matching against them without a restart.

```json
{ "ok": true, "voiceprints_loaded": 2 }
```

---

## Configuration

All values from environment or `.env` file.  See `.env.example` for full
documentation.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_MCP_URL` | `http://memory-mcp:8900` | memory-mcp HTTP API (Pi 4) |
| `PIPELINE_PORT` | `8001` | Listening port |
| `VOICEPRINT_DB` | `./voiceprints.sqlite` | Local voiceprint cache path |
| `VOICEPRINT_CONFIDENT_THRESHOLD` | `0.85` | Cosine similarity for confident match |
| `VOICEPRINT_PROBABLE_THRESHOLD` | `0.70` | Cosine similarity for probable match |
| `WHISPER_CONFIDENCE_THRESHOLD` | `0.85` | Trigger large-v3 fallback below this |
| `HF_TOKEN` | `""` | HuggingFace token for pyannote (leave blank to skip diarization) |
| `HA_WEBHOOK_URL` | `""` | Home Assistant webhook for probable-match alerts |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Running

```bash
cd pipeline_worker
cp .env.example .env
# Edit .env — set MEMORY_MCP_URL, PIPELINE_PORT, etc.
pip install -r requirements.txt
python main.py
```

Or directly:
```bash
uvicorn pipeline_worker.server:create_app --factory --host 0.0.0.0 --port 8001
```

---

## Testing

```bash
# Unit + integration tests (no hardware, no live services)
pytest pipeline_worker/tests/ -m "not hardware and not integration" -v

# Verify a specific test file
pytest pipeline_worker/tests/test_voiceprint.py -v

# With live memory-mcp running (integration tests)
pytest pipeline_worker/tests/ -m "integration" -v
```

---

## Deployment

The pipeline worker runs as a systemd service on the Pi 5, alongside the room node.

```ini
# /etc/systemd/system/heimdall-pipeline.service
[Unit]
Description=Heimdall Pipeline Worker
After=network.target

[Service]
User=heimdall
WorkingDirectory=/opt/heimdall/pipeline_worker
EnvironmentFile=/opt/heimdall/pipeline_worker/.env
ExecStart=python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable heimdall-pipeline
sudo systemctl start heimdall-pipeline
sudo journalctl -u heimdall-pipeline -f
```

### First run

1. Install dependencies: `pip install -r requirements.txt`
2. Copy and configure `.env`
3. Enroll at least one speaker via `enrollment/enroll.py`
4. Start the service — voiceprints are loaded from memory-mcp on startup
5. Verify: `curl http://localhost:8001/health`
