"""Tests for pipeline_worker.voiceprint.VoiceprintMatcher."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline_worker.models import ConfidenceLevel
from pipeline_worker.voiceprint import (
    VOICEPRINT_DIM,
    VoiceprintMatcher,
    _provisional_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unit_vec(seed: int = 0) -> np.ndarray:
    """Reproducible random unit-norm float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(VOICEPRINT_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def db(tmp_path) -> VoiceprintMatcher:
    """Fresh in-memory-ish SQLite matcher per test (tmp_path is unique)."""
    with VoiceprintMatcher(tmp_path / "vp.sqlite") as m:
        yield m


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = unit_vec(0)
        assert VoiceprintMatcher.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.zeros(VOICEPRINT_DIM, dtype=np.float32)
        b = np.zeros(VOICEPRINT_DIM, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert VoiceprintMatcher.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        v = unit_vec(1)
        assert VoiceprintMatcher.cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        v = unit_vec(2)
        z = np.zeros(VOICEPRINT_DIM, dtype=np.float32)
        assert VoiceprintMatcher.cosine_similarity(v, z) == 0.0
        assert VoiceprintMatcher.cosine_similarity(z, v) == 0.0

    def test_different_random_vectors_between_neg1_and_1(self):
        a = unit_vec(3)
        b = unit_vec(4)
        sim = VoiceprintMatcher.cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# running_average
# ---------------------------------------------------------------------------


class TestRunningAverage:
    def test_result_is_unit_norm(self):
        a = unit_vec(0)
        b = unit_vec(1)
        result = VoiceprintMatcher.running_average(a, b, weight=0.1)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_weight_zero_returns_existing(self):
        a = unit_vec(0)
        b = unit_vec(1)
        result = VoiceprintMatcher.running_average(a, b, weight=0.0)
        # At weight=0, result == a (normalised, which is already unit norm)
        assert np.allclose(result, a, atol=1e-6)

    def test_weight_one_returns_incoming(self):
        a = unit_vec(0)
        b = unit_vec(1)
        result = VoiceprintMatcher.running_average(a, b, weight=1.0)
        assert np.allclose(result, b, atol=1e-6)

    def test_output_dtype_is_float32(self):
        result = VoiceprintMatcher.running_average(unit_vec(0), unit_vec(1))
        assert result.dtype == np.float32

    def test_result_shape_preserved(self):
        result = VoiceprintMatcher.running_average(unit_vec(0), unit_vec(1))
        assert result.shape == (VOICEPRINT_DIM,)

    def test_repeated_updates_converge(self):
        """After many updates the embedding moves toward the incoming direction."""
        base = unit_vec(0)
        target = unit_vec(1)
        current = base.copy()
        for _ in range(50):
            current = VoiceprintMatcher.running_average(current, target, weight=0.1)
        # After 50 updates the embedding should be closer to target than to base
        sim_target = VoiceprintMatcher.cosine_similarity(current, target)
        sim_base = VoiceprintMatcher.cosine_similarity(current, base)
        assert sim_target > sim_base


# ---------------------------------------------------------------------------
# classify_confidence
# ---------------------------------------------------------------------------


class TestClassifyConfidence:
    def test_confident(self, db):
        assert db.classify_confidence(0.90) == ConfidenceLevel.CONFIDENT
        assert db.classify_confidence(0.85) == ConfidenceLevel.CONFIDENT

    def test_probable(self, db):
        assert db.classify_confidence(0.80) == ConfidenceLevel.PROBABLE
        assert db.classify_confidence(0.70) == ConfidenceLevel.PROBABLE

    def test_unknown(self, db):
        assert db.classify_confidence(0.69) == ConfidenceLevel.UNKNOWN
        assert db.classify_confidence(0.0) == ConfidenceLevel.UNKNOWN
        assert db.classify_confidence(-0.5) == ConfidenceLevel.UNKNOWN

    def test_custom_thresholds(self, tmp_path):
        with VoiceprintMatcher(
            tmp_path / "vp2.sqlite",
            confident_threshold=0.95,
            probable_threshold=0.80,
        ) as m:
            assert m.classify_confidence(0.94) == ConfidenceLevel.PROBABLE
            assert m.classify_confidence(0.95) == ConfidenceLevel.CONFIDENT
            assert m.classify_confidence(0.79) == ConfidenceLevel.UNKNOWN


# ---------------------------------------------------------------------------
# embedding_hash / provisional names
# ---------------------------------------------------------------------------


class TestEmbeddingHash:
    def test_returns_8_hex_chars(self):
        h = VoiceprintMatcher.embedding_hash(unit_vec(0))
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_same_hash(self):
        v = unit_vec(0)
        assert VoiceprintMatcher.embedding_hash(v) == VoiceprintMatcher.embedding_hash(v)

    def test_different_inputs_different_hashes(self):
        assert VoiceprintMatcher.embedding_hash(unit_vec(0)) != VoiceprintMatcher.embedding_hash(
            unit_vec(1)
        )

    def test_provisional_name_format(self):
        name = _provisional_name(unit_vec(0))
        assert name.startswith("unknown_voice_")
        assert len(name) == len("unknown_voice_") + 8


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestCRUD:
    def test_upsert_and_get(self, db):
        v = unit_vec(0)
        db.upsert("Brian", v, sample_count=3)
        stored = db.get("Brian")
        assert stored is not None
        assert stored.entity_name == "Brian"
        assert stored.sample_count == 3
        assert np.allclose(stored.embedding, v, atol=1e-6)

    def test_get_missing_returns_none(self, db):
        assert db.get("nobody") is None

    def test_upsert_replaces_existing(self, db):
        db.upsert("Brian", unit_vec(0))
        db.upsert("Brian", unit_vec(1), sample_count=5)
        stored = db.get("Brian")
        assert stored.sample_count == 5
        assert np.allclose(stored.embedding, unit_vec(1), atol=1e-6)

    def test_all_returns_all(self, db):
        db.upsert("Brian", unit_vec(0))
        db.upsert("Sarah", unit_vec(1))
        assert {vp.entity_name for vp in db.all()} == {"Brian", "Sarah"}

    def test_all_empty(self, db):
        assert db.all() == []

    def test_delete_existing(self, db):
        db.upsert("Brian", unit_vec(0))
        assert db.delete("Brian") is True
        assert db.get("Brian") is None

    def test_delete_missing_returns_false(self, db):
        assert db.delete("nobody") is False

    def test_count(self, db):
        assert db.count() == 0
        db.upsert("Brian", unit_vec(0))
        assert db.count() == 1
        db.upsert("Sarah", unit_vec(1))
        assert db.count() == 2

    def test_wrong_embedding_dim_rejected(self, db):
        bad = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError, match="256"):
            db.upsert("Brian", bad)


# ---------------------------------------------------------------------------
# match()
# ---------------------------------------------------------------------------


class TestMatch:
    def test_no_candidates_returns_unknown(self, db):
        result = db.match(unit_vec(0))
        assert result.confidence_level == ConfidenceLevel.UNKNOWN
        assert result.confidence == 0.0
        assert result.entity_name.startswith("unknown_voice_")

    def test_exact_match_is_confident(self, db):
        v = unit_vec(0)
        db.upsert("Brian", v)
        result = db.match(v)
        assert result.confidence_level == ConfidenceLevel.CONFIDENT
        assert result.entity_name == "Brian"
        assert result.confidence == pytest.approx(1.0, abs=1e-5)

    def test_picks_closest_of_multiple(self, db):
        brian = unit_vec(0)
        sarah = unit_vec(1)
        db.upsert("Brian", brian)
        db.upsert("Sarah", sarah)
        # Query close to Brian
        result = db.match(brian)
        assert result.entity_name == "Brian"

    def test_low_similarity_returns_unknown_entity_name(self, db):
        db.upsert("Brian", unit_vec(0))
        # Negative vector → far away → unknown
        anti = -unit_vec(0)
        result = db.match(anti)
        assert result.confidence_level == ConfidenceLevel.UNKNOWN
        assert result.entity_name.startswith("unknown_voice_")

    def test_probable_range(self, db):
        # Craft an embedding 0.75 similar to stored one
        stored = unit_vec(0)
        db.upsert("Brian", stored)
        # Blend stored with orthogonal to get a vector with known similarity
        orth = unit_vec(99)
        orth -= orth.dot(stored) * stored
        orth /= np.linalg.norm(orth)
        # target_sim = cos(theta): mix = cos * stored + sin * orth
        target_sim = 0.75
        query = (target_sim * stored + np.sqrt(1 - target_sim**2) * orth).astype(np.float32)
        query /= np.linalg.norm(query)
        result = db.match(query)
        assert result.confidence_level == ConfidenceLevel.PROBABLE
        assert result.entity_name == "Brian"


# ---------------------------------------------------------------------------
# update_after_match()
# ---------------------------------------------------------------------------


class TestUpdateAfterMatch:
    def test_creates_if_missing(self, db):
        v = unit_vec(0)
        db.update_after_match("Brian", v)
        stored = db.get("Brian")
        assert stored is not None
        assert np.allclose(stored.embedding, v, atol=1e-6)
        assert stored.sample_count == 1

    def test_increments_sample_count(self, db):
        db.upsert("Brian", unit_vec(0), sample_count=5)
        db.update_after_match("Brian", unit_vec(1))
        stored = db.get("Brian")
        assert stored.sample_count == 6

    def test_updates_embedding_toward_new(self, db):
        original = unit_vec(0)
        new_sample = unit_vec(1)
        db.upsert("Brian", original)
        db.update_after_match("Brian", new_sample, weight=0.5)
        stored = db.get("Brian")
        # New embedding should differ from original
        assert not np.allclose(stored.embedding, original, atol=1e-3)

    def test_result_stays_unit_norm(self, db):
        db.upsert("Brian", unit_vec(0))
        db.update_after_match("Brian", unit_vec(1))
        stored = db.get("Brian")
        assert np.linalg.norm(stored.embedding) == pytest.approx(1.0, abs=1e-5)
