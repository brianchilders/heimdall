# Supply Chain Security

## Threat Model

The TeamPCP, LiteLLM, and Telnyx incidents share a common pattern:

1. A PyPI package account is compromised (or an abandoned package is re-registered).
2. A new version with malicious code is published — often as a "minor" or "patch" release.
3. Any project using `>=X.Y` or `~=X.Y` silently installs the malicious version on the next `pip install`.

Heimdall is a home automation system with audio capture and memory access.
A compromised dependency has access to everything the process does —
microphone audio, memory-mcp data, Home Assistant tokens.

---

## Defense Strategy

### 1. Two-file model (intent vs. lock)

Every component has two requirements files:

| File | Purpose | Edit? |
|------|---------|-------|
| `requirements.in` | Abstract dependencies — what the project _directly_ needs. Uses `>=` bounds. | Yes — when adding/removing deps |
| `requirements.txt` | Locked file — exact `==` pins for all direct and transitive deps. | **Never** — regenerate with pip-compile |

Installing from `requirements.txt` is deterministic: the same package versions
install every time, regardless of what's been published to PyPI since you locked.

### 2. Version pinning (`==` not `>=`)

`>=` accepts _any_ future version, including a compromised one published tomorrow.
`==` pins to an exact version whose content you have (implicitly) reviewed.

### 3. Hash verification (goal state)

`pip-compile --generate-hashes` adds a cryptographic hash to every entry:

```
fastapi==0.115.0 \
    --hash=sha256:abc123...
```

`pip install --require-hashes -r requirements.txt` then verifies the hash
before extracting any package. Even if a malicious package is published at the
_same_ version number (via a yanked-and-republished release), the hash check
catches it.

**Current state**: requirements.txt files use `==` pins but not hashes yet.
See "Generating a hash-locked file" below.

### 4. Regular CVE auditing with pip-audit

```bash
# Scan the pipeline worker
pip-audit -r pipeline_worker/requirements.txt

# Scan all components
for dir in pipeline_worker room_node enrollment; do
    echo "=== $dir ==="
    pip-audit -r $dir/requirements.txt
done
```

`pip-audit` checks every pinned version against the OSV and PyPI Advisory
databases and reports known CVEs. Run this before every deployment.

---

## Generating / Updating a Lock File

### Prerequisites

```bash
pip install pip-tools
```

### On a development machine (x86)

```bash
cd pipeline_worker
pip-compile --generate-hashes requirements.in -o requirements.txt
```

### On the Pi 5 (platform-specific packages)

The Pi 5 uses ARM64 builds of torch, torchaudio, silero-vad, faster-whisper, etc.
These differ from x86 wheels, so hash pinning must happen on the target platform:

```bash
# On the Pi 5 — after first-time install from requirements.in:
pip-compile --generate-hashes requirements.in -o requirements.txt
# Commit the result as requirements-arm64.txt alongside requirements.txt
```

### Keeping things current

1. Edit `requirements.in` to change a version bound.
2. Run `pip-compile --generate-hashes requirements.in -o requirements.txt`.
3. Review the diff — new packages or version changes warrant a changelog look.
4. Run `pip-audit -r requirements.txt` before committing.
5. Deploy and run tests.

---

## Known Accepted Vulnerabilities

| CVE | Package | Version | Severity | Notes |
|-----|---------|---------|----------|-------|
| CVE-2026-4539 | pygments | 2.19.2 | TBD | Transitive dep of pip-audit only. Not in production runtime. No fix available as of 2026-03-27 — monitor OSV. |

---

## Risk Assessment: Heimdall Dependencies

| Package | Popularity | Last Active | Attack Surface | Action |
|---------|-----------|-------------|----------------|--------|
| fastapi | Very high | Active | HTTP framework | Pinned |
| pydantic | Very high | Active | Data validation | Pinned |
| httpx | Very high | Active | HTTP client | Pinned |
| numpy | Very high | Active | Array math | Pinned |
| resemblyzer | Low | Infrequent | Voice embeddings | **Pinned — watch closely** |
| pyannote.audio | Medium | Active | Speaker diarization | Pin on target |
| silero-vad | Low | Infrequent | VAD model | **Pin on Pi — watch closely** |
| faster-whisper | Medium | Active | ASR | Pin on Pi |
| pyusb | Low | Infrequent | USB HID access | **Pin on Pi — watch closely** |
| torch | Very high | Active | ML runtime | Pinned (dev) |

**Higher-risk packages** are those with lower popularity or infrequent releases —
the same profile as `ctx` (the TeamPCP vector). Pin these first and watch their
release history carefully before upgrading.

---

## Incident Response

If a supply chain compromise is suspected in a pinned dependency:

1. **Do not upgrade.** The compromise is in the new version, not the pinned one.
2. Run `pip-audit -r requirements.txt` to check if the _current_ pin is flagged.
3. If flagged: isolate the affected machine, review what data the process had access to.
4. When a clean version is available: update `requirements.in`, recompile, re-audit, redeploy.
