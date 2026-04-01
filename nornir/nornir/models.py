"""
Shared data models for the Heimdall pipeline.

These dataclasses are the canonical wire format between services.
No business logic lives here — only type definitions.

Flow:
    heimdall-node  →  ContextEvent  →  Verdandi  →  ScoredMemory[]  →  Mimir  →  RoutingResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextEvent:
    """A moment of perception captured by a Heimdall room node.

    Emitted by the Heimdall inference service after processing one utterance.
    Consumed by Verdandi for memory retrieval, then Mimir for routing.

    Attributes:
        who: Speaker identity, e.g. "Brian" or "unknown_voice_a3f2".
        transcript: Whisper-transcribed speech.
        emotion: SpeechBrain emotion label, e.g. "neutral", "happy".
        location: Room identifier, e.g. "kitchen".
        local_time: ISO-8601 timestamp in local timezone.
        speaker_confidence: Speaker ID confidence score 0.0–1.0.
        doa_degrees: Direction of arrival from ReSpeaker (0–359), or None.
        objects_visible: Future: YOLO-detected objects in frame.
        people_detected: Future: video-based person detection.
        activity: Future: coarse action classification.
    """

    who: str
    transcript: str
    emotion: str
    location: str
    local_time: str
    speaker_confidence: float = 0.0
    doa_degrees: int | None = None
    objects_visible: list[str] = field(default_factory=list)
    people_detected: list[str] = field(default_factory=list)
    activity: str | None = None


@dataclass
class ScoredMemory:
    """A memory retrieved from Muninn and scored by Verdandi.

    The composite score combines semantic similarity, recency, and urgency.
    Mimir uses this list to decide whether and what to say.

    Attributes:
        id: Muninn memory UUID.
        content: Natural-language memory text.
        score: Composite score 0.0–1.0 (similarity + recency + urgency).
        similarity: Cosine similarity component.
        recency: Exponential decay component (7-day half-life default).
        urgency: Deadline proximity boost (0.0 unless deadline within 2h).
        meta: Raw metadata dict from Muninn (tags, deadline_utc, tier, etc.).
    """

    id: str
    content: str
    score: float
    similarity: float
    recency: float
    urgency: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """The output decision from Mimir after processing a ContextEvent.

    Attributes:
        spoken_text: Text to speak aloud (empty string if silent decision).
        domain: Matched domain intent key, e.g. "reminder", "greeting".
        memories_used: Muninn memory IDs that influenced the response.
        output_path: "avatar" | "tts_fallback" | "silent".
        latency_ms: Total Mimir processing time in milliseconds.
    """

    spoken_text: str
    domain: str
    memories_used: list[str]
    output_path: str
    latency_ms: int
