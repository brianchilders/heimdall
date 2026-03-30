"""Tests for memory_extensions.voice_routes.

All tests use an in-memory SQLite database — no live memory-mcp required.
Fixtures are defined in conftest.py.

The conftest wires the voice router into a test FastAPI app and overrides the
``server.get_db`` import so routes use the in-memory DB instead of the real
memory.db on disk.
"""

from __future__ import annotations

import json
import math
import time

import pytest
from httpx import AsyncClient

from memory_extensions.tests.conftest import insert_entity

FAKE_EMBEDDING = [0.01 * (i % 100) for i in range(256)]
UNIT_EMBEDDING = [1.0 / math.sqrt(256)] * 256   # unit vector, all equal components


# ---------------------------------------------------------------------------
# GET /voices/unknown
# ---------------------------------------------------------------------------


class TestListUnknownVoices:
    async def test_empty_db_returns_empty_list(self, client: AsyncClient) -> None:
        resp = await client.get("/voices/unknown")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["result"] == []

    async def test_returns_unenrolled_entities(self, client: AsyncClient, db) -> None:
        insert_entity(
            db,
            "unknown_voice_a3f2",
            meta={"status": "unenrolled", "detection_count": 3},
        )
        resp = await client.get("/voices/unknown")
        result = resp.json()["result"]
        assert len(result) == 1
        assert result[0]["entity_name"] == "unknown_voice_a3f2"
        assert result[0]["detection_count"] == 3

    async def test_excludes_enrolled_entities(self, client: AsyncClient, db) -> None:
        insert_entity(db, "Brian", meta={"status": "enrolled", "detection_count": 10})
        insert_entity(
            db,
            "unknown_voice_b9c1",
            meta={"status": "unenrolled", "detection_count": 2},
        )
        resp = await client.get("/voices/unknown")
        result = resp.json()["result"]
        assert len(result) == 1
        assert result[0]["entity_name"] == "unknown_voice_b9c1"

    async def test_excludes_entities_without_status(
        self, client: AsyncClient, db
    ) -> None:
        insert_entity(db, "kitchen", entity_type="room", meta={})
        resp = await client.get("/voices/unknown")
        assert resp.json()["result"] == []

    async def test_min_detections_filter(self, client: AsyncClient, db) -> None:
        insert_entity(
            db, "unknown_voice_low", meta={"status": "unenrolled", "detection_count": 1}
        )
        insert_entity(
            db, "unknown_voice_high", meta={"status": "unenrolled", "detection_count": 5}
        )
        resp = await client.get("/voices/unknown", params={"min_detections": 3})
        result = resp.json()["result"]
        assert len(result) == 1
        assert result[0]["entity_name"] == "unknown_voice_high"

    async def test_sorted_by_detection_count_descending(
        self, client: AsyncClient, db
    ) -> None:
        insert_entity(db, "voice_a", meta={"status": "unenrolled", "detection_count": 1})
        insert_entity(db, "voice_b", meta={"status": "unenrolled", "detection_count": 7})
        insert_entity(db, "voice_c", meta={"status": "unenrolled", "detection_count": 3})
        resp = await client.get("/voices/unknown")
        names = [v["entity_name"] for v in resp.json()["result"]]
        assert names == ["voice_b", "voice_c", "voice_a"]

    async def test_limit_parameter(self, client: AsyncClient, db) -> None:
        for i in range(5):
            insert_entity(
                db, f"voice_{i}", meta={"status": "unenrolled", "detection_count": i}
            )
        resp = await client.get("/voices/unknown", params={"limit": 2})
        assert len(resp.json()["result"]) == 2

    async def test_includes_sample_transcript(self, client: AsyncClient, db) -> None:
        eid = insert_entity(
            db, "unknown_voice_x", meta={"status": "unenrolled", "detection_count": 1}
        )
        db.execute(
            """INSERT INTO readings(entity_id, metric, value_type, value_json, ts)
               VALUES (?, 'voice_activity', 'composite', ?, ?)""",
            (eid, json.dumps({"transcript": "hello from the kitchen"}), time.time()),
        )
        db.commit()
        resp = await client.get("/voices/unknown")
        assert resp.json()["result"][0]["sample_transcript"] == "hello from the kitchen"

    async def test_sample_transcript_most_recent(self, client: AsyncClient, db) -> None:
        eid = insert_entity(
            db, "unknown_voice_y", meta={"status": "unenrolled", "detection_count": 2}
        )
        now = time.time()
        db.execute(
            "INSERT INTO readings(entity_id, metric, value_type, value_json, ts) VALUES (?,?,?,?,?)",
            (eid, "voice_activity", "composite", json.dumps({"transcript": "older"}), now - 60),
        )
        db.execute(
            "INSERT INTO readings(entity_id, metric, value_type, value_json, ts) VALUES (?,?,?,?,?)",
            (eid, "voice_activity", "composite", json.dumps({"transcript": "newer"}), now),
        )
        db.commit()
        resp = await client.get("/voices/unknown")
        assert resp.json()["result"][0]["sample_transcript"] == "newer"

    async def test_includes_meta_fields(self, client: AsyncClient, db) -> None:
        insert_entity(
            db,
            "unknown_voice_z",
            meta={
                "status": "unenrolled",
                "detection_count": 4,
                "first_seen": "2026-03-20",
                "first_seen_room": "kitchen",
            },
        )
        resp = await client.get("/voices/unknown")
        voice = resp.json()["result"][0]
        assert voice["first_seen"] == "2026-03-20"
        assert voice["first_seen_room"] == "kitchen"

    async def test_no_transcript_when_no_readings(
        self, client: AsyncClient, db
    ) -> None:
        insert_entity(
            db, "unknown_voice_q", meta={"status": "unenrolled", "detection_count": 1}
        )
        resp = await client.get("/voices/unknown")
        assert resp.json()["result"][0]["sample_transcript"] is None

    async def test_includes_last_seen(self, client: AsyncClient, db) -> None:
        eid = insert_entity(
            db, "unknown_voice_ts", meta={"status": "unenrolled", "detection_count": 1}
        )
        ts = time.time()
        db.execute(
            "INSERT INTO readings(entity_id, metric, value_type, value_json, ts) VALUES (?,?,?,?,?)",
            (eid, "voice_activity", "composite", json.dumps({"transcript": "hi"}), ts),
        )
        db.commit()
        resp = await client.get("/voices/unknown")
        assert resp.json()["result"][0]["last_seen"] == pytest.approx(ts, rel=1e-3)


# ---------------------------------------------------------------------------
# POST /voices/enroll
# ---------------------------------------------------------------------------


class TestEnrollVoice:
    async def test_renames_entity(self, client: AsyncClient, db) -> None:
        insert_entity(
            db, "unknown_voice_a3f2", meta={"status": "unenrolled", "detection_count": 5}
        )
        resp = await client.post(
            "/voices/enroll",
            json={"entity_name": "unknown_voice_a3f2", "new_name": "Brian"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["result"]["entity_name"] == "Brian"
        assert body["result"]["previous_name"] == "unknown_voice_a3f2"

        row = db.execute("SELECT name, meta FROM entities WHERE name = 'Brian'").fetchone()
        assert row is not None
        assert json.loads(row["meta"])["status"] == "enrolled"

    async def test_entity_id_unchanged(self, client: AsyncClient, db) -> None:
        eid = insert_entity(
            db, "unknown_voice_id_test", meta={"status": "unenrolled", "detection_count": 1}
        )
        resp = await client.post(
            "/voices/enroll",
            json={"entity_name": "unknown_voice_id_test", "new_name": "Sarah"},
        )
        assert resp.json()["result"]["entity_id"] == eid
        row = db.execute("SELECT id FROM entities WHERE name = 'Sarah'").fetchone()
        assert row["id"] == eid

    async def test_linked_data_stays_attached(self, client: AsyncClient, db) -> None:
        eid = insert_entity(
            db, "unknown_voice_mem", meta={"status": "unenrolled", "detection_count": 3}
        )
        now = time.time()
        db.execute(
            "INSERT INTO memories(entity_id, fact, created, updated) VALUES (?,?,?,?)",
            (eid, "likes coffee", now, now),
        )
        db.execute(
            "INSERT INTO readings(entity_id, metric, value_type, value_json, ts) VALUES (?,?,?,?,?)",
            (eid, "voice_activity", "composite", "{}", now),
        )
        db.commit()

        await client.post(
            "/voices/enroll",
            json={"entity_name": "unknown_voice_mem", "new_name": "Alice"},
        )

        mem_row = db.execute(
            "SELECT entity_id FROM memories WHERE fact='likes coffee'"
        ).fetchone()
        assert mem_row["entity_id"] == eid

    async def test_sets_display_name_in_meta(self, client: AsyncClient, db) -> None:
        insert_entity(db, "unknown_voice_dn", meta={"status": "unenrolled", "detection_count": 1})
        await client.post(
            "/voices/enroll",
            json={
                "entity_name": "unknown_voice_dn",
                "new_name": "Bob",
                "display_name": "Robert Smith",
            },
        )
        row = db.execute("SELECT meta FROM entities WHERE name='Bob'").fetchone()
        assert json.loads(row["meta"])["display_name"] == "Robert Smith"

    async def test_404_when_source_not_found(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/voices/enroll",
            json={"entity_name": "does_not_exist", "new_name": "Dave"},
        )
        assert resp.status_code == 404

    async def test_409_when_target_name_exists(self, client: AsyncClient, db) -> None:
        insert_entity(db, "unknown_voice_409", meta={"status": "unenrolled"})
        insert_entity(db, "ExistingPerson", meta={"status": "enrolled"})
        resp = await client.post(
            "/voices/enroll",
            json={"entity_name": "unknown_voice_409", "new_name": "ExistingPerson"},
        )
        assert resp.status_code == 409

    async def test_old_name_no_longer_exists(self, client: AsyncClient, db) -> None:
        insert_entity(db, "unknown_voice_gone", meta={"status": "unenrolled"})
        await client.post(
            "/voices/enroll",
            json={"entity_name": "unknown_voice_gone", "new_name": "NewPerson"},
        )
        row = db.execute(
            "SELECT id FROM entities WHERE name='unknown_voice_gone'"
        ).fetchone()
        assert row is None

    async def test_response_includes_data_counts(self, client: AsyncClient, db) -> None:
        eid = insert_entity(db, "unknown_voice_cnt", meta={"status": "unenrolled"})
        now = time.time()
        db.execute(
            "INSERT INTO memories(entity_id, fact, created, updated) VALUES (?,?,?,?)",
            (eid, "fact one", now, now),
        )
        db.execute(
            "INSERT INTO readings(entity_id, metric, value_type, value_json, ts) VALUES (?,?,?,?,?)",
            (eid, "voice_activity", "composite", "{}", now),
        )
        db.commit()
        resp = await client.post(
            "/voices/enroll",
            json={"entity_name": "unknown_voice_cnt", "new_name": "Counted"},
        )
        result = resp.json()["result"]
        assert result["memories_transferred"] == 1
        assert result["readings_transferred"] == 1


# ---------------------------------------------------------------------------
# POST /voices/merge
# ---------------------------------------------------------------------------


class TestMergeVoices:
    async def test_transfers_memories(self, client: AsyncClient, db) -> None:
        src_id = insert_entity(db, "voice_src", meta={"status": "unenrolled"})
        tgt_id = insert_entity(db, "Brian", meta={"status": "enrolled"})
        now = time.time()
        db.execute(
            "INSERT INTO memories(entity_id, fact, created, updated) VALUES (?,?,?,?)",
            (src_id, "prefers tea", now, now),
        )
        db.commit()

        resp = await client.post(
            "/voices/merge",
            json={"source_name": "voice_src", "target_name": "Brian"},
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["memories_merged"] == 1

        mem_row = db.execute(
            "SELECT entity_id FROM memories WHERE fact='prefers tea'"
        ).fetchone()
        assert mem_row["entity_id"] == tgt_id

    async def test_transfers_readings(self, client: AsyncClient, db) -> None:
        src_id = insert_entity(db, "voice_src_r", meta={"status": "unenrolled"})
        tgt_id = insert_entity(db, "Carol", meta={"status": "enrolled"})
        db.execute(
            "INSERT INTO readings(entity_id, metric, value_type, value_json, ts) VALUES (?,?,?,?,?)",
            (src_id, "voice_activity", "composite", "{}", time.time()),
        )
        db.commit()

        await client.post(
            "/voices/merge",
            json={"source_name": "voice_src_r", "target_name": "Carol"},
        )

        reading = db.execute(
            "SELECT entity_id FROM readings WHERE entity_id=?", (tgt_id,)
        ).fetchone()
        assert reading is not None

    async def test_source_entity_deleted(self, client: AsyncClient, db) -> None:
        insert_entity(db, "voice_to_delete", meta={"status": "unenrolled"})
        insert_entity(db, "TargetPerson", meta={"status": "enrolled"})
        await client.post(
            "/voices/merge",
            json={"source_name": "voice_to_delete", "target_name": "TargetPerson"},
        )
        row = db.execute(
            "SELECT id FROM entities WHERE name='voice_to_delete'"
        ).fetchone()
        assert row is None

    async def test_sample_weighted_voiceprint_merge(
        self, client: AsyncClient, db
    ) -> None:
        """Voiceprint averaging is weighted by voiceprint_samples, not 50/50.

        Uses orthogonal unit vectors so the direction of the result after
        normalization reveals which embedding dominated:
          - tgt_vp points along dim 0 (9 samples → should dominate)
          - src_vp points along dim 1 (1 sample → minor contribution)
        After weighting (9:1) and normalization, component[0] >> component[1].
        A naive 50/50 average would produce component[0] == component[1].
        """
        tgt_vp = [1.0] + [0.0] * 255   # unit vector along dim 0
        src_vp = [0.0, 1.0] + [0.0] * 254  # unit vector along dim 1
        insert_entity(
            db,
            "voice_vp_src",
            meta={"status": "unenrolled", "voiceprint": src_vp, "voiceprint_samples": 1},
        )
        insert_entity(
            db,
            "WeightedTarget",
            meta={"status": "enrolled", "voiceprint": tgt_vp, "voiceprint_samples": 9},
        )

        await client.post(
            "/voices/merge",
            json={"source_name": "voice_vp_src", "target_name": "WeightedTarget"},
        )

        row = db.execute(
            "SELECT meta FROM entities WHERE name='WeightedTarget'"
        ).fetchone()
        meta = json.loads(row["meta"])
        merged_vp = meta["voiceprint"]
        # After 9:1 weighting and normalization, component[0] (target direction)
        # must be significantly larger than component[1] (source direction).
        assert merged_vp[0] > merged_vp[1] * 5
        assert meta["voiceprint_samples"] == 10

    async def test_voiceprint_is_normalized_after_merge(
        self, client: AsyncClient, db
    ) -> None:
        """Merged voiceprint should remain a unit vector."""
        src_vp = [0.5] * 256
        tgt_vp = [0.8] * 256
        insert_entity(
            db,
            "voice_norm_src",
            meta={"status": "unenrolled", "voiceprint": src_vp, "voiceprint_samples": 3},
        )
        insert_entity(
            db,
            "NormTarget",
            meta={"status": "enrolled", "voiceprint": tgt_vp, "voiceprint_samples": 3},
        )

        await client.post(
            "/voices/merge",
            json={"source_name": "voice_norm_src", "target_name": "NormTarget"},
        )

        row = db.execute("SELECT meta FROM entities WHERE name='NormTarget'").fetchone()
        merged_vp = json.loads(row["meta"])["voiceprint"]
        norm = sum(x * x for x in merged_vp) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    async def test_merges_detection_counts(self, client: AsyncClient, db) -> None:
        insert_entity(
            db, "voice_dc_src", meta={"status": "unenrolled", "detection_count": 5}
        )
        insert_entity(
            db, "DcTarget", meta={"status": "enrolled", "detection_count": 20}
        )
        await client.post(
            "/voices/merge",
            json={"source_name": "voice_dc_src", "target_name": "DcTarget"},
        )
        row = db.execute("SELECT meta FROM entities WHERE name='DcTarget'").fetchone()
        assert json.loads(row["meta"])["detection_count"] == 25

    async def test_uses_source_voiceprint_when_target_has_none(
        self, client: AsyncClient, db
    ) -> None:
        src_vp = list(UNIT_EMBEDDING)
        insert_entity(
            db, "voice_vp_only_src",
            meta={"status": "unenrolled", "voiceprint": src_vp, "voiceprint_samples": 3},
        )
        insert_entity(db, "NoVP", meta={"status": "enrolled"})

        await client.post(
            "/voices/merge",
            json={"source_name": "voice_vp_only_src", "target_name": "NoVP"},
        )

        row = db.execute("SELECT meta FROM entities WHERE name='NoVP'").fetchone()
        meta = json.loads(row["meta"])
        assert meta["voiceprint"] == src_vp
        assert meta["voiceprint_samples"] == 3

    async def test_transfer_active_relations(self, client: AsyncClient, db) -> None:
        src_id = insert_entity(db, "voice_rel_src", meta={"status": "unenrolled"})
        tgt_id = insert_entity(db, "RelTarget", meta={"status": "enrolled"})
        other_id = insert_entity(db, "OtherPerson", meta={"status": "enrolled"})
        db.execute(
            "INSERT INTO relations(entity_a, entity_b, rel_type, created) VALUES (?,?,?,?)",
            (src_id, other_id, "knows", time.time()),
        )
        db.commit()

        await client.post(
            "/voices/merge",
            json={"source_name": "voice_rel_src", "target_name": "RelTarget"},
        )

        rel = db.execute(
            "SELECT entity_a FROM relations WHERE entity_b=? AND rel_type='knows'",
            (other_id,),
        ).fetchone()
        assert rel is not None
        assert rel["entity_a"] == tgt_id

    async def test_duplicate_relations_not_created(
        self, client: AsyncClient, db
    ) -> None:
        src_id = insert_entity(db, "voice_dup_src", meta={"status": "unenrolled"})
        tgt_id = insert_entity(db, "DupTarget", meta={"status": "enrolled"})
        other_id = insert_entity(db, "SharedFriend", meta={"status": "enrolled"})
        now = time.time()
        db.execute(
            "INSERT INTO relations(entity_a, entity_b, rel_type, created) VALUES (?,?,?,?)",
            (src_id, other_id, "knows", now),
        )
        db.execute(
            "INSERT INTO relations(entity_a, entity_b, rel_type, created) VALUES (?,?,?,?)",
            (tgt_id, other_id, "knows", now),
        )
        db.commit()

        await client.post(
            "/voices/merge",
            json={"source_name": "voice_dup_src", "target_name": "DupTarget"},
        )

        count = db.execute(
            "SELECT COUNT(*) FROM relations WHERE entity_a=? AND entity_b=? AND rel_type='knows'",
            (tgt_id, other_id),
        ).fetchone()[0]
        assert count == 1

    async def test_400_same_name(self, client: AsyncClient, db) -> None:
        insert_entity(db, "SamePerson", meta={"status": "enrolled"})
        resp = await client.post(
            "/voices/merge",
            json={"source_name": "SamePerson", "target_name": "SamePerson"},
        )
        assert resp.status_code == 400

    async def test_404_source_not_found(self, client: AsyncClient, db) -> None:
        insert_entity(db, "RealPerson", meta={"status": "enrolled"})
        resp = await client.post(
            "/voices/merge",
            json={"source_name": "ghost", "target_name": "RealPerson"},
        )
        assert resp.status_code == 404

    async def test_404_target_not_found(self, client: AsyncClient, db) -> None:
        insert_entity(db, "voice_orphan", meta={"status": "unenrolled"})
        resp = await client.post(
            "/voices/merge",
            json={"source_name": "voice_orphan", "target_name": "nobody"},
        )
        assert resp.status_code == 404

    async def test_response_field_names(self, client: AsyncClient, db) -> None:
        insert_entity(db, "voice_fields", meta={"status": "unenrolled"})
        insert_entity(db, "FieldTarget", meta={"status": "enrolled"})
        resp = await client.post(
            "/voices/merge",
            json={"source_name": "voice_fields", "target_name": "FieldTarget"},
        )
        result = resp.json()["result"]
        assert "memories_merged" in result
        assert "readings_merged" in result
        assert "relations_merged" in result
        assert result["source_deleted"] == "voice_fields"
        assert result["target_name"] == "FieldTarget"


# ---------------------------------------------------------------------------
# POST /voices/update_print
# ---------------------------------------------------------------------------


class TestUpdateVoiceprint:
    async def test_initializes_when_no_existing(self, client: AsyncClient, db) -> None:
        insert_entity(db, "NewSpeaker", meta={"status": "enrolled"})
        resp = await client.post(
            "/voices/update_print",
            json={"entity_name": "NewSpeaker", "embedding": FAKE_EMBEDDING},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["result"]["voiceprint_samples"] == 1

        row = db.execute("SELECT meta FROM entities WHERE name='NewSpeaker'").fetchone()
        meta = json.loads(row["meta"])
        assert meta["voiceprint"] == FAKE_EMBEDDING
        assert meta["voiceprint_samples"] == 1

    async def test_running_average_default_weight(self, client: AsyncClient, db) -> None:
        existing = [1.0] * 256
        insert_entity(
            db, "AvgSpeaker",
            meta={"status": "enrolled", "voiceprint": existing, "voiceprint_samples": 5},
        )
        incoming = [0.0] * 256

        resp = await client.post(
            "/voices/update_print",
            json={"entity_name": "AvgSpeaker", "embedding": incoming},
        )
        assert resp.json()["result"]["voiceprint_samples"] == 6

        row = db.execute("SELECT meta FROM entities WHERE name='AvgSpeaker'").fetchone()
        stored = json.loads(row["meta"])["voiceprint"]
        # default weight=0.1 → blended = 0.9*1.0 + 0.1*0.0 = 0.9 per component
        # then normalized: norm = 0.9 * sqrt(256) = 14.4, each component = 0.9 / 14.4 ≈ 0.0625
        # the key check: value is positive and embedding is a unit vector
        assert all(v > 0 for v in stored)
        norm = sum(x * x for x in stored) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    async def test_result_is_normalized_unit_vector(self, client: AsyncClient, db) -> None:
        """After update_print, stored embedding must be a unit vector."""
        insert_entity(
            db, "NormSpeaker",
            meta={"status": "enrolled", "voiceprint": [0.5] * 256, "voiceprint_samples": 2},
        )
        await client.post(
            "/voices/update_print",
            json={"entity_name": "NormSpeaker", "embedding": [0.8] * 256},
        )
        row = db.execute("SELECT meta FROM entities WHERE name='NormSpeaker'").fetchone()
        stored = json.loads(row["meta"])["voiceprint"]
        norm = sum(x * x for x in stored) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    async def test_increments_sample_count(self, client: AsyncClient, db) -> None:
        insert_entity(
            db, "CountSpeaker",
            meta={"status": "enrolled", "voiceprint": list(UNIT_EMBEDDING), "voiceprint_samples": 10},
        )
        await client.post(
            "/voices/update_print",
            json={"entity_name": "CountSpeaker", "embedding": list(UNIT_EMBEDDING)},
        )
        row = db.execute("SELECT meta FROM entities WHERE name='CountSpeaker'").fetchone()
        assert json.loads(row["meta"])["voiceprint_samples"] == 11

    async def test_embedding_norm_in_response(self, client: AsyncClient, db) -> None:
        insert_entity(db, "NormResp", meta={"status": "enrolled"})
        resp = await client.post(
            "/voices/update_print",
            json={"entity_name": "NormResp", "embedding": list(UNIT_EMBEDDING)},
        )
        result = resp.json()["result"]
        assert "embedding_norm" in result
        assert abs(result["embedding_norm"] - 1.0) < 0.01

    async def test_preserves_other_meta_fields(self, client: AsyncClient, db) -> None:
        insert_entity(
            db,
            "MetaSpeaker",
            meta={"status": "enrolled", "first_seen_room": "office", "detection_count": 7},
        )
        await client.post(
            "/voices/update_print",
            json={"entity_name": "MetaSpeaker", "embedding": FAKE_EMBEDDING},
        )
        row = db.execute("SELECT meta FROM entities WHERE name='MetaSpeaker'").fetchone()
        meta = json.loads(row["meta"])
        assert meta["first_seen_room"] == "office"
        assert meta["detection_count"] == 7
        assert meta["status"] == "enrolled"

    async def test_404_when_entity_not_found(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/voices/update_print",
            json={"entity_name": "ghost", "embedding": FAKE_EMBEDDING},
        )
        assert resp.status_code == 404

    async def test_rejects_wrong_embedding_dim(self, client: AsyncClient, db) -> None:
        insert_entity(db, "DimCheck", meta={"status": "enrolled"})
        resp = await client.post(
            "/voices/update_print",
            json={"entity_name": "DimCheck", "embedding": [0.1] * 128},
        )
        assert resp.status_code == 422

    def test_validator_rejects_nan_embedding(self) -> None:
        """The Pydantic validator rejects NaN values before they reach the DB.

        Standard JSON cannot encode NaN (it would fail at serialisation before
        even reaching the HTTP layer), but the validator is a defensive check
        for callers that construct the model object directly in Python.
        """
        from pydantic import ValidationError
        from memory_extensions.voice_routes import UpdatePrintRequest

        bad_embedding = FAKE_EMBEDDING[:]
        bad_embedding[0] = float("nan")
        with pytest.raises(ValidationError, match="finite"):
            UpdatePrintRequest(entity_name="x", embedding=bad_embedding)
