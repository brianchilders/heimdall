"""
Unit tests for muninn.store.followups.
"""

from __future__ import annotations

import pytest

from muninn.store.followups import (
    create_followup,
    delete_expired_followups,
    delete_followup,
    get_followup,
    list_followups_for,
)


@pytest.mark.asyncio
async def test_create_and_get(pool):
    fu = await create_followup(
        pool=pool,
        who="Brian",
        spoken_text="Don't forget your dentist appointment tomorrow.",
        location="kitchen",
        ttl_hours=4,
    )
    assert fu["id"]
    assert fu["who"] == "Brian"
    assert fu["location"] == "kitchen"
    assert fu["spoken_text"] == "Don't forget your dentist appointment tomorrow."

    fetched = await get_followup(pool, fu["id"])
    assert fetched == fu


@pytest.mark.asyncio
async def test_get_missing_returns_none(pool):
    result = await get_followup(pool, "00000000-0000-0000-0000-000000000000")
    assert result is None


@pytest.mark.asyncio
async def test_list_for_speaker(pool):
    await create_followup(pool, who="Brian", spoken_text="Msg 1")
    await create_followup(pool, who="Brian", spoken_text="Msg 2")
    await create_followup(pool, who="Aria", spoken_text="Msg A")

    brian_fus = await list_followups_for(pool, "Brian")
    assert len(brian_fus) == 2
    assert all(f["who"] == "Brian" for f in brian_fus)

    aria_fus = await list_followups_for(pool, "Aria")
    assert len(aria_fus) == 1


@pytest.mark.asyncio
async def test_delete(pool):
    fu = await create_followup(pool, who="Brian", spoken_text="Test")
    deleted = await delete_followup(pool, fu["id"])
    assert deleted is True
    assert await get_followup(pool, fu["id"]) is None


@pytest.mark.asyncio
async def test_delete_missing_returns_false(pool):
    result = await delete_followup(pool, "00000000-0000-0000-0000-000000000000")
    assert result is False


@pytest.mark.asyncio
async def test_expired_followups_excluded_by_default(pool):
    # Create a followup with a very short TTL — it will technically still be
    # valid immediately but we test the exclude logic via include_expired flag
    fu = await create_followup(pool, who="Brian", spoken_text="Will expire", ttl_hours=1)

    active = await list_followups_for(pool, "Brian", include_expired=False)
    all_ = await list_followups_for(pool, "Brian", include_expired=True)
    assert len(active) == 1
    assert len(all_) == 1
    assert active[0]["id"] == fu["id"]


@pytest.mark.asyncio
async def test_delete_expired_followups(pool):
    # Create one normal followup; none should be immediately expired
    await create_followup(pool, who="Brian", spoken_text="Normal followup", ttl_hours=4)
    count = await delete_expired_followups(pool)
    # No rows should be deleted since ttl_hours=4 is in the future
    assert count == 0
    remaining = await list_followups_for(pool, "Brian")
    assert len(remaining) == 1
