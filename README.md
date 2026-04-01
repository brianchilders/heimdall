# Ymir
### A Local-First Ambient Intelligence Home Prototype

> **Most AI today lives in apps. Ambient intelligence should live in the environment.**

---

## Overview

**Ymir** is a working prototype exploring what ambient intelligence looks like when it is:

- **Local-first**
- **Privacy-preserving**
- **Memory-enabled**
- **Embedded in the home—not in an app**

This project integrates:

- **OpenHome speakers** — ambient interface layer
- **Home Assistant** — device + sensor orchestration
- **Muninn** — persistent memory store (local SQLite + vector search)
- **Heimdall nodes** — room-level AI presence (voice + awareness)

---

## Services

Each service is named after a figure from Norse mythology. The names are not decorative — they reflect what each service actually does in the system.

| Service | Norse figure | Role | Description |
|---|---|---|---|
| **nornir** | The Norns — weavers of fate | Shared contracts | Canonical data models, wire formats, and constants used across all services. No business logic; pure type definitions. |
| **muninn** | Muninn — Odin's raven of *Memory* | Memory server | Persistent SQLite + vector search store. Stores memories, embeddings, follow-ups, and time-series patterns. The source of truth for everything the system has learned. |
| **verdandi** | Verdandi — the Norn of *the Present* | Recommender engine | Embeds incoming context events via Ollama, runs KNN search against Muninn, and scores candidates by semantic similarity, recency, and deadline urgency. Bridges past memory to present context. |
| **mimir** | Mimir — keeper of *the Well of Wisdom* | Intent router / LLM | Classifies intent domain, retrieves scored memories from Verdandi, prompts a local LLM (Ollama), routes output to TTS or avatar, enforces per-domain cooldown, and writes follow-ups back to Muninn. |
| **heimdall** | Heimdall — *watchman* of Asgard | Inference receiver | Receives audio payloads from room nodes, runs Whisper (transcription) + Resemblyzer (speaker ID) + SpeechBrain (emotion), assembles a `ContextEvent`, and dispatches it to Verdandi → Mimir. |
| **heimdall-node** | Heimdall's ear | Edge capture node | Pi 5 room node. Runs sounddevice capture, Silero VAD, and DOA extraction from the ReSpeaker mic array. Uplinks audio chunks to the Heimdall inference service. |

### Data flow

```
heimdall-node  →  [audio chunk]     →  heimdall
                                            │
                                    [ContextEvent]
                                            │
                                        verdandi  ←→  muninn
                                            │
                                    [ScoredMemory[]]
                                            │
                                          mimir
                                            │
                                    [RoutingResult]
                                            │
                              TTS / OpenHome avatar / silent
```

`nornir` defines `ContextEvent`, `ScoredMemory`, and `RoutingResult` — the wire types passed between every stage.

---

## The Goal

This is **not** an attempt to build a fully autonomous AI home.

Instead, Ymir explores a simpler, more important question:

> Can a home become meaningfully more helpful if it remembers, understands context, and responds locally over time?

---

## Core Capabilities

### A Home That Remembers

Using **Muninn**, the system stores:

- preferences (temperature, lighting, routines)
- environmental patterns
- prior interactions
- contextual signals from the home

The home accumulates intelligence instead of resetting every day.

### Ambient Interaction (Not Commands)

With **OpenHome speakers + Heimdall nodes**:

- voice is context-aware, not command-driven
- responses are grounded in current home state, remembered patterns, and recent interactions

Interaction becomes ambient, not transactional.

### Environmental Awareness (Bounded)

A lightweight perception layer introduces event-based object detection with no continuous surveillance model.

Examples: `package_detected`, `room_occupied`, `garage_state_changed`

Vision becomes input, not monitoring.

---

## Hardware (Room Node)

Each room node is built on:

- Raspberry Pi 5 (16GB)
- Hailo-8 M.2 AI accelerator (26 TOPS)
- ReSpeaker XVF3800 4-mic array (hardware AEC, beamforming, DOA)

See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for setup and hardware verification.

---

## Example Scenarios

**Preference-aware comfort**
> "It's evening, and you usually prefer it warmer here."

System adjusts environment based on learned patterns.

**Context-aware voice**
> "Did I leave the garage open?"

Response uses current state, recent activity, and memory context.

**Intention carryover**
> "Remind me to take that package inside when I get home."

System remembers and triggers when conditions match.

**Event-based awareness**
> "Package detected at front door."

Event → memory → optional action or notification.

---

## Privacy Model

Ymir is designed with privacy as a first principle:

- local processing preferred by default
- minimal raw data retention
- event-based perception (not video storage)
- explainable actions
- no cloud dependency required

---

## Status

**In active development**

- [x] Nornir — shared contracts
- [x] Muninn — memory server
- [x] Verdandi — recommender engine
- [x] Mimir — intent router
- [x] Heimdall — inference receiver
- [x] Heimdall-node — Pi capture node
- [x] Hardware verified (ReSpeaker XVF3800, Logitech C920)
- [ ] End-to-end integration test (node → memory)
- [ ] Object detection pilot (Hailo + Frigate)
- [ ] Full demo scenarios

---

## About

**Brian Childers**
Builder focused on local-first AI systems, memory-driven architectures, and ambient intelligence in real environments.

---

> Ambient intelligence shouldn't feel like using a system.
> It should feel like the environment understands you.
