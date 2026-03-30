# Enrollment CLI

## Purpose

Command-line tool to enroll known speakers into the Heimdall voice identity
system.  Records audio from a microphone (or loads a WAV file), computes a
resemblyzer voiceprint embedding, and registers the speaker in memory-mcp.

Must be run before the pipeline worker can identify any speaker by name.
Enroll Brian and Sarah before running any room nodes.

---

## Architecture

```
mic or WAV file
    │
    ├─ load_wav() / record_audio()   → float32 mono 16kHz array
    │
    ├─ compute_embedding()
    │    └─ resemblyzer VoiceEncoder.embed_utterance()  → 256-dim unit-norm
    │
    ├─ POST /remember                → create entity in memory-mcp (Tier 1)
    │
    └─ POST /voices/update_print     → store voiceprint embedding (weight=1.0)
```

After enrollment, call `POST /reload_voiceprints` on the pipeline worker so it
picks up the new embedding without a restart.

---

## Usage

### Enroll from microphone

```bash
cd enrollment
python enroll.py enroll --name Brian --room office --duration 10
```

Speak naturally for 10 seconds.  Longer recordings produce more robust
embeddings — aim for at least 10 seconds of speech.

### Enroll from WAV file

```bash
python enroll.py enroll --name Sarah --wav /path/to/sarah_voice.wav
```

WAV requirements:
- Mono or stereo (stereo is averaged to mono)
- Any sample rate (resampled to 16kHz automatically, requires `scipy`)
- At least **3 seconds** of speech (longer is better)
- Minimum amplitude — not silence

### List enrolled speakers

```bash
python enroll.py list
```

Output:
```
Enrolled speakers (2):
  Name                  Samples  Updated
  --------------------  -------  ------------------------
  Brian                      12  2026-03-22T14:30:00+00:00
  Sarah                       8  2026-03-22T15:00:00+00:00
```

### Show unenrolled provisional voices

```bash
python enroll.py unknown
```

Shows auto-created provisional entities from live audio.  Use this to find
voices detected before enrollment:

```
Unenrolled voices (1):
  Entity                        Detections  Sample transcript
  ------------------------------  ----------  ----------------------------------------
  unknown_voice_a3f2c8d1                 7  'I need to pick up groceries tomorrow'
```

---

## Configuration

Values from environment or `.env` file in the `enrollment/` directory.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_MCP_URL` | `http://memory-mcp:8900` | memory-mcp HTTP API |
| `MEMORY_MCP_TOKEN` | *(empty)* | Bearer token for memory-mcp auth (required if server has auth enabled) |
| `DEFAULT_RECORD_DURATION_S` | `10` | Default mic recording duration |
| `DEVICE_INDEX` | *(system default)* | sounddevice mic device index |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## Running

```bash
cd enrollment
cp .env.example .env
# Edit .env — set MEMORY_MCP_URL and MEMORY_MCP_TOKEN
pip install -r requirements.txt

# Enroll Brian
python enroll.py enroll --name Brian --duration 15

# Enroll Sarah
python enroll.py enroll --name Sarah --wav sarah.wav

# Tell pipeline worker to pick up the new voiceprints
curl -X POST http://heimdall.local:8001/reload_voiceprints
```

---

## Testing

```bash
pytest enrollment/tests/ -m "not hardware and not integration" -v
```

Integration tests (require live memory-mcp):
```bash
pytest enrollment/tests/ -m integration -v
```

---

## WAV Recording Tips

For the best voiceprint quality, record in the same room and conditions where
the person will normally be heard.  Say a variety of phrases naturally — don't
read a word list in a monotone.  Example prompts:

> "The quick brown fox jumps over the lazy dog."
> "Can you turn the lights on in the living room?"
> "What's the weather like today?"
> "I need to pick up some groceries this afternoon."
> "Set a timer for 10 minutes please."

Avoid: whispering, shouting, music in the background, or very short responses
of 1–2 words.
