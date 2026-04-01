"""
Home Assistant calendar importer.

Reads upcoming calendar events from the Home Assistant REST API and
stores each event as a semantic memory with a deadline_utc field so
Verdandi can apply the urgency boost.

Configuration (via env):
  HA_URL        — e.g. http://homeassistant.local:8123
  HA_TOKEN      — long-lived access token
  HA_CALENDAR_IDS — comma-separated list of calendar entity IDs,
                    e.g. "calendar.family,calendar.work"
  HA_CALENDAR_DAYS — how many days ahead to fetch (default: 14)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from muninn.config import MuninnConfig
from muninn.db.connection import ConnectionPool
from muninn.importers.base import BaseImporter
from muninn.store.memories import create_memory, list_memories

logger = logging.getLogger(__name__)

_SEMANTIC_TIER = "semantic"
_DEFAULT_DAYS_AHEAD = 14


class HACalendarImporter(BaseImporter):
    """Import upcoming calendar events from Home Assistant.

    Events are stored as semantic memories with:
      - ``deadline_utc``: ISO-8601 start time for urgency scoring
      - ``source_event_id``: HA event UID to avoid duplicates

    Args:
        pool: Database connection pool.
        config: Muninn configuration.
        ha_url: Home Assistant base URL.  Defaults to HA_URL env var.
        ha_token: Long-lived access token.  Defaults to HA_TOKEN env var.
        calendar_ids: List of HA calendar entity IDs.  Defaults to
                      HA_CALENDAR_IDS env var (comma-separated).
        days_ahead: Number of days forward to fetch.
    """

    def __init__(
        self,
        pool: ConnectionPool,
        config: MuninnConfig,
        ha_url: Optional[str] = None,
        ha_token: Optional[str] = None,
        calendar_ids: Optional[list[str]] = None,
        days_ahead: int = _DEFAULT_DAYS_AHEAD,
    ) -> None:
        super().__init__(pool, config)
        self._ha_url = (ha_url or os.environ.get("HA_URL", "")).rstrip("/")
        self._ha_token = ha_token or os.environ.get("HA_TOKEN", "")
        raw_ids = os.environ.get("HA_CALENDAR_IDS", "")
        self._calendar_ids: list[str] = calendar_ids or [
            c.strip() for c in raw_ids.split(",") if c.strip()
        ]
        self._days_ahead = days_ahead

        if not self._ha_url:
            raise ValueError("HA_URL is required for HACalendarImporter")
        if not self._ha_token:
            raise ValueError("HA_TOKEN is required for HACalendarImporter")

    @property
    def source_name(self) -> str:
        return "ha_calendar"

    async def _fetch_events(self, calendar_id: str, start: str, end: str) -> list[dict]:
        """Fetch events from a single HA calendar entity.

        Args:
            calendar_id: HA entity ID (e.g. ``calendar.family``).
            start: ISO-8601 start datetime.
            end: ISO-8601 end datetime.

        Returns:
            List of event dicts from HA API.
        """
        url = f"{self._ha_url}/api/calendars/{calendar_id}"
        headers = {"Authorization": f"Bearer {self._ha_token}"}
        params = {"start": start, "end": end}

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, params=params, timeout=15.0)
            resp.raise_for_status()
            return resp.json()

    async def run(self) -> dict[str, int]:
        """Fetch upcoming events and store new ones as memories.

        Returns:
            Dict with ``imported`` and ``skipped`` counts.
        """
        if not self._calendar_ids:
            logger.warning("No HA calendar IDs configured — nothing to import")
            return {"imported": 0, "skipped": 0}

        now = datetime.now(timezone.utc)
        start = now.isoformat()
        end = (now + timedelta(days=self._days_ahead)).isoformat()

        # Build set of already-imported event IDs
        existing = await list_memories(self.pool, tier=_SEMANTIC_TIER, limit=100_000)
        imported_event_ids = {
            m["metadata"].get("source_event_id")
            for m in existing
            if m.get("metadata", {}).get("source_event_id")
        }

        imported = 0
        skipped = 0

        for cal_id in self._calendar_ids:
            try:
                events = await self._fetch_events(cal_id, start, end)
            except httpx.HTTPError as exc:
                logger.error("Failed to fetch %s: %s", cal_id, exc)
                continue

            for event in events:
                uid = event.get("uid") or event.get("id") or ""
                if uid in imported_event_ids:
                    skipped += 1
                    continue

                summary = event.get("summary", "Untitled event")
                description = event.get("description", "")
                start_dt = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date", "")

                content_parts = [f"Calendar event: {summary}"]
                if description:
                    content_parts.append(description)
                if start_dt:
                    content_parts.append(f"When: {start_dt}")

                meta: dict = {"calendar": cal_id}
                if uid:
                    meta["source_event_id"] = uid
                if start_dt:
                    meta["deadline_utc"] = start_dt

                await create_memory(
                    pool=self.pool,
                    tier=_SEMANTIC_TIER,
                    content="\n".join(content_parts),
                    metadata=meta,
                    source=self.source_name,
                )
                imported += 1
                logger.debug("Imported event: %s (%s)", summary, start_dt)

        logger.info(
            "HA Calendar import done: %d imported, %d skipped",
            imported,
            skipped,
        )
        return {"imported": imported, "skipped": skipped}
