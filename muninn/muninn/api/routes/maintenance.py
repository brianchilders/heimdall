"""
Maintenance trigger routes.

Prefix: /maintenance
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Request

from muninn.maintenance.expire import run_expiry
from muninn.maintenance.reembed import reembed_all
from muninn.maintenance.vacuum import vacuum_db, wal_checkpoint

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/maintenance", tags=["maintenance"])


@router.post("/expire")
async def expire(request: Request) -> dict:
    """Delete all memories and followups past their TTL.

    Returns:
        Dict with ``memories`` and ``followups`` deleted counts.
    """
    return await run_expiry(request.app.state.pool)


@router.post("/vacuum")
async def vacuum(background_tasks: BackgroundTasks, request: Request) -> dict:
    """Run VACUUM in the background to reclaim disk space.

    Returns:
        Acknowledgement with checkpoint info.
    """
    pool = request.app.state.pool
    checkpoint = await wal_checkpoint(pool)
    background_tasks.add_task(vacuum_db, pool)
    return {"status": "vacuum queued", "checkpoint": checkpoint}


@router.post("/reembed")
async def reembed(
    background_tasks: BackgroundTasks, request: Request, force: bool = False
) -> dict:
    """Re-embed memories that lack vectors for the current model.

    Args:
        force: Re-embed even memories that already have a vector.

    Returns:
        Acknowledgement.
    """
    pool = request.app.state.pool
    config = request.app.state.config
    background_tasks.add_task(reembed_all, pool, config, force)
    return {"status": "reembed queued", "model": config.muninn_embed_model, "force": force}
