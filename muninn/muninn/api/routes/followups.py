"""
Followup CRUD routes.

Prefix: /followups
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from muninn.store.followups import (
    create_followup,
    delete_followup,
    get_followup,
    list_followups_for,
)

router = APIRouter(prefix="/followups", tags=["followups"])


class CreateFollowupRequest(BaseModel):
    """Body for POST /followups."""

    who: str
    spoken_text: str
    location: Optional[str] = None
    memory_id: Optional[str] = None
    ttl_hours: int = 4


@router.post("", status_code=201)
async def create(body: CreateFollowupRequest, request: Request) -> dict:
    """Create a pending followup.

    Args:
        body: Followup payload.
        request: FastAPI request.

    Returns:
        Created followup dict.
    """
    return await create_followup(
        pool=request.app.state.pool,
        who=body.who,
        spoken_text=body.spoken_text,
        location=body.location,
        memory_id=body.memory_id,
        ttl_hours=body.ttl_hours,
    )


@router.get("/{followup_id}")
async def read(followup_id: str, request: Request) -> dict:
    """Retrieve a followup by ID.

    Args:
        followup_id: UUID string.
        request: FastAPI request.

    Raises:
        HTTPException 404: If not found.
    """
    fu = await get_followup(request.app.state.pool, followup_id)
    if fu is None:
        raise HTTPException(404, detail="Followup not found")
    return fu


@router.get("")
async def list_for_speaker(
    request: Request,
    who: str = Query(...),
    include_expired: bool = Query(False),
) -> list[dict]:
    """List pending followups for a speaker.

    Args:
        request: FastAPI request.
        who: Speaker name.
        include_expired: Include expired followups.
    """
    return await list_followups_for(
        request.app.state.pool, who, include_expired=include_expired
    )


@router.delete("/{followup_id}", status_code=204)
async def dismiss(followup_id: str, request: Request) -> None:
    """Delete (dismiss) a followup by ID.

    Args:
        followup_id: UUID string.
        request: FastAPI request.

    Raises:
        HTTPException 404: If not found.
    """
    deleted = await delete_followup(request.app.state.pool, followup_id)
    if not deleted:
        raise HTTPException(404, detail="Followup not found")
