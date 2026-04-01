"""
Recommendation engine.

Core logic:
  1. Embed ContextEvent via Ollama
  2. KNN search against Muninn (pre-computed vector, no re-embedding)
  3. Score each candidate:
       score = W_SIM * similarity + W_REC * recency + W_URG * urgency
  4. Filter by min_score, sort descending, return top_k ScoredMemory

Scoring components
------------------
similarity  — converted from L2 distance of unit-normalised vectors:
              similarity = max(0, 1 - distance² / 2)
              (for unit vectors L2_dist = sqrt(2 - 2*cos_sim))

recency     — linear decay over RECENCY_DAYS with floor at 0:
              recency = max(0, 1 - age_hours / (recency_days * 24))

urgency     — deadline proximity boost:
              if 0 < hours_until_deadline < urgency_window_hours:
                  urgency = 1.0 - hours_until / urgency_window_hours
              else:
                  urgency = 0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from nornir.models import ContextEvent, ScoredMemory
from verdandi.config import VerdandiConfig
from verdandi.embedder import embed_context
from verdandi.memory_client import vector_search

logger = logging.getLogger(__name__)

# Candidate multiplier — request 4× top_k from Muninn before reranking
_CANDIDATE_MULTIPLIER = 4


def _distance_to_similarity(distance: float) -> float:
    """Convert L2 distance of unit vectors to cosine similarity.

    For unit-normalised embeddings:
        L2_distance = sqrt(2 - 2 * cosine_similarity)
    Therefore:
        cosine_similarity = 1 - L2_distance² / 2

    Args:
        distance: L2 distance from sqlite-vec.

    Returns:
        Cosine similarity in [0, 1].
    """
    return max(0.0, 1.0 - (distance ** 2) / 2.0)


def _recency_score(created_at: str, recency_days: float) -> float:
    """Linear recency decay with a floor of 0.

    Args:
        created_at: ISO-8601 creation timestamp string.
        recency_days: Half-life in days (full score at 0, zero at recency_days*2).

    Returns:
        Recency score in [0, 1].
    """
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_hours = (now - created).total_seconds() / 3600.0
        half_life_hours = recency_days * 24.0
        return max(0.0, 1.0 - age_hours / half_life_hours)
    except (ValueError, AttributeError):
        logger.debug("Cannot parse created_at %r — recency=0", created_at)
        return 0.0


def _urgency_score(metadata: dict, urgency_window_hours: float) -> float:
    """Deadline proximity urgency boost.

    A memory scores maximum urgency (approaching 1.0) as its
    deadline_utc approaches, within the urgency window.

    Args:
        metadata: Memory metadata dict (may contain ``deadline_utc``).
        urgency_window_hours: Window before deadline that triggers boost.

    Returns:
        Urgency score in [0, 1].  0.0 if no deadline or deadline passed.
    """
    deadline_str = metadata.get("deadline_utc")
    if not deadline_str:
        return 0.0
    try:
        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_until = (deadline - now).total_seconds() / 3600.0
        if 0.0 < hours_until < urgency_window_hours:
            return 1.0 - hours_until / urgency_window_hours
        return 0.0
    except (ValueError, AttributeError):
        logger.debug("Cannot parse deadline_utc %r — urgency=0", deadline_str)
        return 0.0


def _score_hit(
    hit: dict,
    config: VerdandiConfig,
) -> ScoredMemory:
    """Compute the composite score for one Muninn search result.

    Args:
        hit: Muninn search result dict.
        config: Verdandi config (supplies weights and windows).

    Returns:
        ScoredMemory with all score components filled in.
    """
    similarity = _distance_to_similarity(hit["distance"])
    recency = _recency_score(hit.get("created_at", ""), config.verdandi_recency_days)
    urgency = _urgency_score(hit.get("metadata") or {}, config.verdandi_urgency_window_hours)

    composite = (
        config.verdandi_w_sim * similarity
        + config.verdandi_w_rec * recency
        + config.verdandi_w_urg * urgency
    )

    return ScoredMemory(
        id=hit["memory_id"],
        content=hit["content"],
        score=round(composite, 4),
        similarity=round(similarity, 4),
        recency=round(recency, 4),
        urgency=round(urgency, 4),
        meta=hit.get("metadata") or {},
    )


async def get_recommendations(
    event: ContextEvent,
    config: VerdandiConfig,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> list[ScoredMemory]:
    """Surface the most relevant memories for a ContextEvent.

    Pipeline:
      1. Embed ContextEvent → query vector (Ollama)
      2. KNN search → candidates (Muninn POST /search)
      3. Score each candidate (similarity + recency + urgency)
      4. Filter by min_score, sort descending, return top_k

    Args:
        event: Incoming context event to match against.
        config: Verdandi configuration.
        top_k: Override default top_k from config.
        min_score: Override default min_score from config.
        client: Optional shared AsyncClient (for testing / performance).

    Returns:
        List of ScoredMemory sorted by score descending.
        Empty list if no memories exceed min_score.
    """
    effective_top_k = top_k if top_k is not None else config.verdandi_top_k
    effective_min_score = min_score if min_score is not None else config.verdandi_min_score

    # Step 1: embed
    query_vec = await embed_context(event, config, client=client)

    # Step 2: KNN candidates from Muninn
    candidates = await vector_search(
        query_vec,
        config,
        top_k=effective_top_k * _CANDIDATE_MULTIPLIER,
        client=client,
    )

    if not candidates:
        logger.debug("No KNN candidates returned from Muninn")
        return []

    # Step 3: score
    scored = [_score_hit(h, config) for h in candidates]

    # Step 4: filter + sort + truncate
    above_threshold = [s for s in scored if s.score >= effective_min_score]
    above_threshold.sort(key=lambda s: s.score, reverse=True)
    result = above_threshold[:effective_top_k]

    logger.debug(
        "Recommendations for %r: %d/%d above min_score=%.2f",
        event.who,
        len(result),
        len(scored),
        effective_min_score,
    )
    return result
