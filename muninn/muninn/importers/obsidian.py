"""
Obsidian vault importer.

Walks a local Obsidian vault directory and imports each Markdown note
as a semantic memory.  Notes already present (matched by file path in
metadata) are skipped.

Configuration (via MuninnConfig / env):
  OBSIDIAN_VAULT_PATH  — absolute path to the vault root
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from muninn.config import MuninnConfig
from muninn.db.connection import ConnectionPool
from muninn.importers.base import BaseImporter
from muninn.store.memories import create_memory, list_memories

logger = logging.getLogger(__name__)

_SEMANTIC_TIER = "semantic"
_VAULT_ENV = "OBSIDIAN_VAULT_PATH"


class ObsidianImporter(BaseImporter):
    """Import Markdown notes from an Obsidian vault as semantic memories.

    Each note is stored with ``{"source_path": "<relative/path.md>"}``
    in metadata so re-runs skip already-imported notes.

    Args:
        pool: Database connection pool.
        config: Muninn configuration.
        vault_path: Path to vault root.  Falls back to OBSIDIAN_VAULT_PATH env var.
    """

    def __init__(
        self,
        pool: ConnectionPool,
        config: MuninnConfig,
        vault_path: Optional[str] = None,
    ) -> None:
        super().__init__(pool, config)
        raw = vault_path or os.environ.get(_VAULT_ENV)
        if not raw:
            raise ValueError(
                f"Obsidian vault path not set. "
                f"Pass vault_path= or set {_VAULT_ENV} env var."
            )
        self._vault = Path(raw).expanduser().resolve()

    @property
    def source_name(self) -> str:
        return "obsidian"

    async def run(self) -> dict[str, int]:
        """Walk the vault and import new notes.

        Returns:
            Dict with ``imported`` and ``skipped`` counts.
        """
        if not self._vault.is_dir():
            logger.error("Obsidian vault not found: %s", self._vault)
            return {"imported": 0, "skipped": 0}

        # Build set of already-imported paths
        existing = await list_memories(self.pool, tier=_SEMANTIC_TIER, limit=100_000)
        imported_paths = {
            m["metadata"].get("source_path")
            for m in existing
            if m.get("metadata", {}).get("source_path")
        }

        imported = 0
        skipped = 0

        for md_file in self._vault.rglob("*.md"):
            rel_path = str(md_file.relative_to(self._vault))

            if rel_path in imported_paths:
                skipped += 1
                continue

            try:
                content = md_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", rel_path, exc)
                skipped += 1
                continue

            if not content:
                skipped += 1
                continue

            # Truncate very long notes — embed-model context window limit
            if len(content) > 8_000:
                content = content[:8_000] + " [truncated]"

            title = md_file.stem
            full_content = f"{title}\n\n{content}" if title not in content[:200] else content

            await create_memory(
                pool=self.pool,
                tier=_SEMANTIC_TIER,
                content=full_content,
                metadata={"source_path": rel_path, "title": title},
                source=self.source_name,
            )
            imported += 1
            logger.debug("Imported note: %s", rel_path)

        logger.info(
            "Obsidian import done: %d imported, %d skipped (vault=%s)",
            imported,
            skipped,
            self._vault,
        )
        return {"imported": imported, "skipped": skipped}
