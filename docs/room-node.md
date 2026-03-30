# Room Node

## Purpose

Per-room audio capture and inference service.  Two hardware profiles are
supported:

- **Full node** (heimdall — Pi 5 + ReSpeaker XVF3800): Runs faster-whisper,
  emotion model, and Resemblyzer locally.  Ships a rich `AudioPayload` to the
  pipeline worker (localhost:8001); only attaches raw audio when confidence is low.

- **Capture node** (additional Pi, no AI accelerator): VAD + DOA only.  Always
  ships raw audio to the pipeline worker, which runs the full inference stack.
  Minimal dependencies — no ML models on the Pi.

---

## Architecture

### Full Node (heimdall — Pi 5)

```
ReSpeaker USB (4-mic array, hardware AEC/beamforming)
    │
    ├─ sounddevice stream (16kHz mono PCM)
    │
    ├─ Silero VAD — gates on speech, discards silence
    │
    ├─ DOA extraction from USB HID (XMOS XVF-3000/3800 DOAANGLE param)
    │
    ├─ InferenceEngine.run(utterance)
    │    ├─ Whisper (faster-whisper CPU, default small)
    │    │   └─ future: Hailo-8 tiny/base via hailo-whisper
    │    ├─ Emotion model (Hailo-8 .hef)  → valence, arousal
    │    │   └─ fallback: neutral (0.5, 0.5)
    │    └─ Resemblyzer (CPU)             → 256-dim voiceprint embedding
    │
    └─ PayloadSender.send() [node_profile="full"]
         ├─ Build AudioPayload JSON with transcript + embeddings
         ├─ Attach audio_clip_b64 if whisper_confidence < threshold
         └─ POST to localhost:8001/ingest (with retry)
```

### Capture Node (additional Pi)

```
ReSpeaker USB (4-mic array, hardware AEC/beamforming)
    │
    ├─ sounddevice stream (16kHz mono PCM)
    │
    ├─ Silero VAD — gates on speech, discards silence
    │
    ├─ DOA extraction from USB HID
    │
    └─ PayloadSender.send() [node_profile="capture"]
         ├─ Build AudioPayload JSON — NO inference fields
         ├─ Always attach audio_clip_b64
         └─ POST to heimdall.local:8001/ingest (with retry)
```

### Key files

| File | Responsibility |
|------|---------------|
| `config.py` | Pydantic-settings config from .env |
| `capture.py` | sounddevice stream + Silero VAD utterance gating |
| `doa.py` | ReSpeaker USB HID DOA reading (XVF-3000 and XVF-3800) |
| `hailo_inference.py` | Whisper + emotion (Hailo-8 or CPU fallback) + resemblyzer |
| `sender.py` | AudioPayload assembly, base64 audio encoding, httpx POST with retry and offline queue |
| `main.py` | Full node entry point, asyncio loop, SIGTERM handler |
| `capture_node.py` | Capture node entry point (no inference deps) |

---

## Configuration

All values from environment or `.env` file.  See `.env.example` for full docs.

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOM_NAME` | *(required)* | Room identifier, e.g. `kitchen` |
| `NODE_PROFILE` | `full` | `full` (heimdall) or `capture` (additional Pi, no ML) |
| `BLACKMAGIC_URL` | `http://localhost:8001` | Pipeline worker URL (localhost on heimdall) |
| `DEVICE_INDEX` | `0` | sounddevice index for ReSpeaker USB |
| `SAMPLE_RATE` | `16000` | PCM sample rate (Hz) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD speech probability gate |
| `VAD_MIN_SILENCE_MS` | `500` | Silence duration to end an utterance |
| `VAD_SPEECH_PAD_MS` | `100` | Pre-speech padding on each utterance |
| `MAX_UTTERANCE_S` | `30` | Hard cap on utterance length |
| `WHISPER_CONFIDENCE_THRESHOLD` | `0.85` | Attach audio clip below this (full nodes only) |
| `HAILO_ENABLED` | `true` | Set `false` to force CPU fallbacks (full nodes only) |
| `HAILO_WHISPER_HEF` | `./models/whisper_tiny.hef` | Compiled Whisper model — tiny/base supported on Hailo-8 (full nodes only) |
| `HAILO_EMOTION_HEF` | `./models/emotion.hef` | Compiled emotion model (full nodes only) |
| `WHISPER_FALLBACK_MODEL` | `small` | faster-whisper model when Hailo unavailable (full nodes only) |
| `HTTP_MAX_RETRIES` | `3` | POST retries on failure |
| `HTTP_RETRY_BACKOFF_S` | `1.0` | Base backoff seconds (exponential) |
| `OFFLINE_QUEUE_MAXSIZE` | `50` | Max payloads to buffer when pipeline worker is unreachable (oldest evicted when full) |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## Running

### Full node (heimdall — Pi 5)

```bash
cd room_node
cp .env.example .env
# Edit .env:
#   ROOM_NAME=office       ← or wherever heimdall is placed
#   NODE_PROFILE=full
#   BLACKMAGIC_URL=http://localhost:8001
#   HAILO_ENABLED=false
#   WHISPER_FALLBACK_MODEL=small
pip install -r requirements.txt
python main.py
```

### Capture node (additional Pi, no ML)

```bash
cd room_node
cp .env.example .env
# Edit .env:
#   ROOM_NAME=bedroom      ← room this node covers
#   NODE_PROFILE=capture
#   BLACKMAGIC_URL=http://heimdall.local:8001
# Only lightweight deps needed:
pip install silero-vad sounddevice httpx numpy pydantic pydantic-settings
python capture_node.py
```

---

## Testing

Tests use WAV fixture arrays and mocked hardware — no Pi or ReSpeaker required.

```bash
# All unit tests
pytest room_node/tests/ -m "not hardware and not integration" -v

# Hardware-required tests (run on Pi with ReSpeaker attached)
pytest room_node/tests/ -m hardware -v
```

---

## Hardware Setup

### ReSpeaker XVF3800 USB Mic Array

- Plug into USB-A port on heimdall (Pi 5)
- Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to confirm the device appears
- Note the device index — set `DEVICE_INDEX` in `.env`
- Linux: add the user to the `audio` group: `sudo usermod -aG audio $USER`

### Hailo-8 M.2 HAT

**Current status**: Hailo-8 acceleration is not enabled by default. The Pi 5 runs
`faster-whisper` on CPU (`HAILO_ENABLED=false`) which is sufficient for ambient
capture. Hailo support is planned as a future optimization.

**What Hailo-8 actually supports for ASR (as of 2025):**
- Whisper **tiny** and **base** models via Hailo's own
  [`hailo-whisper`](https://github.com/hailo-ai/hailo-whisper) tooling
- Whisper **small** is not currently supported on Hailo-8
- `faster-whisper` does not have an official Hailo backend — Hailo uses their own
  inference pipeline

**When you are ready to enable Hailo-8:**
1. Install HailoRT and the Hailo PCIe driver per the official Hailo installation guide
2. Clone and set up [`hailo-whisper`](https://github.com/hailo-ai/hailo-whisper)
3. Use Whisper `tiny` or `base` (the supported models for Hailo-8)
4. `hailo_inference.py` will need updating to call the `hailo-whisper` API instead
   of `faster-whisper` — the fallback path is already correct
5. Verify hardware: `hailortcli fw-control identify`

### AEC (Acoustic Echo Cancellation)

The XMOS XVF-3000 DSP on the ReSpeaker USB handles AEC in hardware — it
cancels the OpenHome speaker output from the microphone input automatically.
No software AEC configuration required.

---

## Deployment (systemd)

### Full node

```ini
# /etc/systemd/system/heimdall-room-node.service
[Unit]
Description=Heimdall Full Node — Kitchen
After=network.target sound.target

[Service]
User=heimdall
WorkingDirectory=/opt/heimdall/room_node
EnvironmentFile=/opt/heimdall/room_node/.env
ExecStart=python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Capture node

```ini
# /etc/systemd/system/heimdall-capture-node.service
[Unit]
Description=Heimdall Capture Node — Bedroom
After=network.target sound.target
Wants=network-online.target

[Service]
User=heimdall
WorkingDirectory=/opt/heimdall/room_node
ExecStart=/usr/bin/python3 /opt/heimdall/room_node/capture_node.py
Restart=always
RestartSec=5
Environment=ROOM_NAME=bedroom
Environment=NODE_PROFILE=capture
Environment=BLACKMAGIC_URL=http://heimdall.local:8001

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable heimdall-room-node   # or heimdall-capture-node
sudo systemctl start heimdall-room-node
sudo journalctl -u heimdall-room-node -f
```

### Multiple rooms

Deploy one Pi per room.  Each Pi has its own `.env` (or `Environment=` lines)
with a unique `ROOM_NAME`.  A room can be upgraded from capture → full node
by swapping the Pi hardware and setting `NODE_PROFILE=full` — the pipeline
worker requires no changes.

All capture nodes point `BLACKMAGIC_URL` at `http://heimdall.local:8001`.
The full node (heimdall itself) uses `http://localhost:8001`.
