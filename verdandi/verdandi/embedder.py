"""
Context embedder.

Converts a ContextEvent into a 768-dim float vector by flattening
the event to natural-language prose and calling Ollama's
/api/embeddings endpoint.

The text template deliberately weights the transcript and speaker
heavily — these are the most semantically meaningful fields for
memory retrieval.
"""

from __future__ import annotations

import logging

import httpx

from nornir.models import ContextEvent
from verdandi.config import VerdandiConfig

logger = logging.getLogger(__name__)


def _context_to_text(event: ContextEvent) -> str:
    """Flatten a ContextEvent to a natural-language embedding prompt.

    Args:
        event: Incoming context event.

    Returns:
        Single string suitable for embedding.
    """
    parts = [
        f"{event.who} in {event.location} said: {event.transcript}.",
        f"Mood: {event.emotion}.",
    ]
    if event.objects_visible:
        parts.append(f"Visible: {', '.join(event.objects_visible)}.")
    if event.activity:
        parts.append(f"Activity: {event.activity}.")
    return " ".join(parts)


async def embed_context(
    event: ContextEvent,
    config: VerdandiConfig,
    client: httpx.AsyncClient | None = None,
) -> list[float]:
    """Embed a ContextEvent via Ollama and return the float vector.

    Args:
        event: Incoming context event to embed.
        config: Verdandi configuration (supplies Ollama URL + model).
        client: Optional existing AsyncClient.  A new one is created if
                not provided (for callers that don't maintain a session).

    Returns:
        Float list of length ``config.embed_dim``.

    Raises:
        httpx.HTTPStatusError: On non-2xx response from Ollama.
        httpx.TimeoutException: If Ollama does not respond within 30s.
    """
    text = _context_to_text(event)
    logger.debug("Embedding context: %r", text[:120])

    async def _call(c: httpx.AsyncClient) -> list[float]:
        resp = await c.post(
            f"{config.ollama_url}/api/embeddings",
            json={"model": config.embed_model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    if client is not None:
        return await _call(client)

    async with httpx.AsyncClient() as c:
        return await _call(c)
