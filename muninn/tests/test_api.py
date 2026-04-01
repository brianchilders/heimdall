"""
Integration tests for Muninn FastAPI routes.

Uses httpx.AsyncClient with the FastAPI app directly — no network
required, no running Muninn server needed.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from muninn.api.app import create_app
from muninn.config import MuninnConfig


@pytest_asyncio.fixture
async def client(tmp_path):
    """AsyncClient wired to a fresh Muninn app with its own lifespan."""
    db_path = tmp_path / "test_api.db"
    config = MuninnConfig(
        muninn_db_path=str(db_path),
        ollama_url="http://localhost:11434",
    )
    app = create_app(config=config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # Trigger lifespan startup
        async with app.router.lifespan_context(app):
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "embed_model" in body


# ---------------------------------------------------------------------------
# /memories CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_memory(client):
    resp = await client.post(
        "/memories",
        json={"tier": "semantic", "content": "Brian likes oat milk", "source": "test"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"]
    assert body["tier"] == "semantic"
    assert body["content"] == "Brian likes oat milk"


@pytest.mark.asyncio
async def test_create_memory_invalid_tier(client):
    resp = await client.post(
        "/memories", json={"tier": "invalid", "content": "bad"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_memory(client):
    create_resp = await client.post(
        "/memories", json={"tier": "episodic", "content": "Test event"}
    )
    memory_id = create_resp.json()["id"]

    get_resp = await client.get(f"/memories/{memory_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == memory_id


@pytest.mark.asyncio
async def test_get_memory_not_found(client):
    resp = await client.get("/memories/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_memory(client):
    create_resp = await client.post(
        "/memories", json={"tier": "semantic", "content": "Original"}
    )
    memory_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/memories/{memory_id}", json={"content": "Updated"}
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["content"] == "Updated"


@pytest.mark.asyncio
async def test_delete_memory(client):
    create_resp = await client.post(
        "/memories", json={"tier": "semantic", "content": "To delete"}
    )
    memory_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/memories/{memory_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/memories/{memory_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_list_memories(client):
    for i in range(3):
        await client.post("/memories", json={"tier": "semantic", "content": f"Fact {i}"})

    resp = await client.get("/memories")
    assert resp.status_code == 200
    assert len(resp.json()) >= 3


@pytest.mark.asyncio
async def test_list_memories_tier_filter(client):
    await client.post("/memories", json={"tier": "semantic", "content": "Semantic"})
    await client.post("/memories", json={"tier": "episodic", "content": "Episodic"})

    resp = await client.get("/memories?tier=semantic")
    assert resp.status_code == 200
    assert all(m["tier"] == "semantic" for m in resp.json())


# ---------------------------------------------------------------------------
# /followups CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_followup(client):
    resp = await client.post(
        "/followups",
        json={"who": "Brian", "spoken_text": "Don't forget your dentist appointment."},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["who"] == "Brian"
    assert body["id"]


@pytest.mark.asyncio
async def test_get_followup(client):
    create_resp = await client.post(
        "/followups",
        json={"who": "Aria", "spoken_text": "Check your homework."},
    )
    fu_id = create_resp.json()["id"]

    get_resp = await client.get(f"/followups/{fu_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == fu_id


@pytest.mark.asyncio
async def test_list_followups_for_speaker(client):
    await client.post("/followups", json={"who": "Brian", "spoken_text": "A"})
    await client.post("/followups", json={"who": "Brian", "spoken_text": "B"})
    await client.post("/followups", json={"who": "Aria", "spoken_text": "C"})

    resp = await client.get("/followups?who=Brian")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_dismiss_followup(client):
    create_resp = await client.post(
        "/followups", json={"who": "Brian", "spoken_text": "Temp"}
    )
    fu_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/followups/{fu_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/followups/{fu_id}")
    assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# /maintenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expire_endpoint(client):
    resp = await client.post("/maintenance/expire")
    assert resp.status_code == 200
    body = resp.json()
    assert "memories" in body
    assert "followups" in body
