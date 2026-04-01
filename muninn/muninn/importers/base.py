"""
Base class for all Muninn importers.

Importers read from an external source (Obsidian vault, HA calendar,
etc.) and write memories into the Muninn store.  Each importer must
implement :meth:`run` and :attr:`source_name`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from muninn.config import MuninnConfig
from muninn.db.connection import ConnectionPool

logger = logging.getLogger(__name__)


class BaseImporter(ABC):
    """Abstract base for all Muninn importers.

    Args:
        pool: Database connection pool.
        config: Muninn configuration.
    """

    def __init__(self, pool: ConnectionPool, config: MuninnConfig) -> None:
        self.pool = pool
        self.config = config
        self._log = logging.getLogger(f"muninn.importer.{self.source_name}")

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this importer (used as memory source tag)."""

    @abstractmethod
    async def run(self) -> dict[str, int]:
        """Execute the import and return a summary.

        Returns:
            Dict with at minimum ``imported`` and ``skipped`` counts.
        """
