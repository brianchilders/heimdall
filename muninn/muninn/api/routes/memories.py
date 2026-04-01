"""
REST routes for memory CRUD.

Prefix: /memories
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from muninn.store.memories import (
    create_memory,
    delete_memory,
    get_memory,
    list_memories,
    update_memory,
)
from nornir.schema import ALL_TIERS

router = APIRouter(prefix="/memories", tags=["memories"])


class CreateMemoryRequest(BaseModel):
    """Body for POST /memories."""

    tier: str
    content: str
    metadata: Optional[dict[str, Any]] = None
    source: Optional[str] = None


class UpdateMemoryRequest(BaseModel):
    """Body for PATCH /memories/{id}."""

    content: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@router.post("", status_code=201)
async def create(body: CreateMemoryRequest, request: Request) -> dict:
    """Create a new memory.

    Args:
        body: Memory payload.
        request: FastAPI request (pool via app.state).

    Returns:
        Created memory dict.

    Raises:
        HTTPException 422: If tier is invalid.
    """
    if body.tier not in ALL_TIERS:
        raise HTTPException(422, detail=f"Invalid tier. Must be one of {ALL_TIERS}")
    return await create_memory(
        pool=request.app.state.pool,
        tier=body.tier,
        content=body.content,
        metadata=body.metadata,
        source=body.source,
    )


@router.get("/{memory_id}")
async def read(memory_id: str, request: Request) -> dict:
    """Retrieve a memory by ID.

    Args:
        memory_id: UUID string.
        request: FastAPI request.

    Raises:
        HTTPException 404: If not found.
    """
    mem = await get_memory(request.app.state.pool, memory_id)
    if mem is None:
        raise HTTPException(404, detail="Memory not found")
    return mem


@router.patch("/{memory_id}")
async def update(memory_id: str, body: UpdateMemoryRequest, request: Request) -> dict:
    """Update content and/or metadata of an existing memory.

    Args:
        memory_id: UUID string.
        body: Fields to update.
        request: FastAPI request.

    Raises:
        HTTPException 404: If not found.
    """
    mem = await update_memory(
        request.app.state.pool, memory_id, content=body.content, metadata=body.metadata
    )
    if mem is None:
        raise HTTPException(404, detail="Memory not found")
    return mem


@router.delete("/{memory_id}", status_code=204)
async def delete(memory_id: str, request: Request) -> None:
    """Hard-delete a memory.

    Args:
        memory_id: UUID string.
        request: FastAPI request.

    Raises:
        HTTPException 404: If not found.
    """
    deleted = await delete_memory(request.app.state.pool, memory_id)
    if not deleted:
        raise HTTPException(404, detail="Memory not found")


@router.get("")
async def list_all(
    request: Request,
    tier: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_expired: bool = Query(False),
) -> list[dict]:
    """List memories with optional filters.

    Args:
        request: FastAPI request.
        tier: Filter by tier.
        limit: Max rows (1–500).
        offset: Pagination offset.
        include_expired: Include expired memories.
    """
    return await list_memories(
        request.app.state.pool,
        tier=tier,
        limit=limit,
        offset=offset,
        include_expired=include_expired,
    )
