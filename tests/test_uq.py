"""Offline unit tests for components/uq.py (W3).

No real detector/model is loaded: augmentation-consistency is exercised
against small MOCK detector callables (a "stable" one that always reports
the same box+label, and a "jittery" one that reports noisy, half-wrong
detections), so this is fast and fully offline.
"""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from components.uq import annotate_difficulty, augmentation_consistency, difficulty, enrich

REF_BOX = (100, 100, 50, 50)
REF_LABEL = "red block"


def _synthetic_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)


@dataclass
class MockDetection:
    box: tuple
    label: str
    score: Optional[float] = None


class StableDetector:
    """Always reports the reference box/label -- an "easy" object."""

    def __call__(self, image: np.ndarray):
        return [MockDetection(box=REF_BOX, label=REF_LABEL, score=0.95)]


class JitteryDetector:
    """Reports a randomly placed box and often the wrong label -- "tough"."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray):
        x = self._rng.randint(0, 300)
        y = self._rng.randint(0, 300)
        label = self._rng.choice(["red block", "blue block", "unknown thing"])
        return [MockDetection(box=(x, y, 50, 50), label=label, score=0.4)]


class EmptyDetector:
    """Never finds anything -- worst case."""

    def __call__(self, image: np.ndarray):
        return []


# --- difficulty() fusion ----------------------------------------------------


class TestDifficultyFusion:
    def test_monotonic_in_score(self):
        low_score_d = difficulty(score=0.1, consistency=0.8, aspect_ratio=1.0, area=1000)
        high_score_d = difficulty(score=0.9, consistency=0.8, aspect_ratio=1.0, area=1000)
        assert high_score_d < low_score_d

    def test_monotonic_in_consistency(self):
        low_consistency_d = difficulty(score=0.7, consistency=0.1, aspect_ratio=1.0, area=1000)
        high_consistency_d = difficulty(score=0.7, consistency=0.9, aspect_ratio=1.0, area=1000)
        assert high_consistency_d < low_consistency_d

    def test_range_bounded(self):
        for score in (0.0, 0.5, 1.0):
            for consistency in (0.0, 0.5, 1.0):
                d = difficulty(score=score, consistency=consistency, aspect_ratio=5.0, area=1.0)
                assert 0.0 <= d <= 1.0

    def test_extreme_aspect_ratio_increases_difficulty(self):
        normal = difficulty(score=0.8, consistency=0.8, aspect_ratio=1.1, area=1000)
        extreme = difficulty(score=0.8, consistency=0.8, aspect_ratio=6.0, area=1000)
        assert extreme > normal

    def test_extreme_area_increases_difficulty(self):
        normal = difficulty(score=0.8, consistency=0.8, aspect_ratio=1.0, area=5000)
        tiny = difficulty(score=0.8, consistency=0.8, aspect_ratio=1.0, area=5.0)
        huge = difficulty(score=0.8, consistency=0.8, aspect_ratio=1.0, area=500000.0)
        assert tiny > normal
        assert huge > normal

    def test_missing_signals_fall_back_to_neutral(self):
        # No crash, stays in range, and a fully-unknown object is "medium"
        # (score/consistency default neutral 0.5; unknown geometry adds no
        # penalty since there's no evidence of an extreme box).
        d = difficulty()
        assert 0.3 <= d <= 0.6

    def test_custom_weights_respected(self):
        # All weight on score -> consistency changes should not move it.
        d_a = difficulty(score=0.9, consistency=0.0, weights={"score": 1, "consistency": 0, "geometry": 0})
        d_b = difficulty(score=0.9, consistency=1.0, weights={"score": 1, "consistency": 0, "geometry": 0})
        assert d_a == pytest.approx(d_b)


# --- augmentation_consistency() ---------------------------------------------


class TestAugmentationConsistency:
    def test_stable_detector_is_fully_consistent(self):
        image = _synthetic_image()
        result = augmentation_consistency(
            StableDetector(), image, REF_BOX, ref_label=REF_LABEL, n=5, seed=0
        )
        assert result["consistency"] == pytest.approx(1.0)
        assert result["mean_iou"] == pytest.approx(1.0)
        assert result["iou_std"] == pytest.approx(0.0)
        assert result["label_agreement"] == pytest.approx(1.0)

    def test_jittery_detector_is_much_less_consistent(self):
        image = _synthetic_image()
        stable = augmentation_consistency(
            StableDetector(), image, REF_BOX, ref_label=REF_LABEL, n=8, seed=0
        )
        jittery = augmentation_consistency(
            JitteryDetector(), image, REF_BOX, ref_label=REF_LABEL, n=8, seed=0
        )
        assert jittery["consistency"] < stable["consistency"]
        assert jittery["consistency"] < 0.5

    def test_empty_detector_yields_zero_consistency(self):
        image = _synthetic_image()
        result = augmentation_consistency(
            EmptyDetector(), image, REF_BOX, ref_label=REF_LABEL, n=4, seed=0
        )
        assert result["consistency"] == 0.0
        assert result["hit_rate"] == 0.0

    def test_n_is_respected(self):
        image = _synthetic_image()
        result = augmentation_consistency(StableDetector(), image, REF_BOX, n=3, seed=0)
        assert result["n"] == 3


class TestAugmentationRaisesDifficulty:
    def test_tough_object_scores_harder_than_easy_object(self):
        image = _synthetic_image()
        easy_agreement = augmentation_consistency(
            StableDetector(), image, REF_BOX, ref_label=REF_LABEL, n=6, seed=1
        )
        tough_agreement = augmentation_consistency(
            JitteryDetector(), image, REF_BOX, ref_label=REF_LABEL, n=6, seed=1
        )
        # Same score/geometry -- only the augmentation-consistency differs.
        d_easy = difficulty(
            score=0.8, consistency=easy_agreement["consistency"], aspect_ratio=1.0, area=1000
        )
        d_tough = difficulty(
            score=0.8, consistency=tough_agreement["consistency"], aspect_ratio=1.0, area=1000
        )
        assert d_tough > d_easy


# --- annotate_difficulty() / enrich() populate LocatedShape-like objects ----


@dataclass
class MockShape:
    label: str
    box: tuple
    aspect_ratio: float = 1.0
    area: float = 1000.0
    score: Optional[float] = None


@dataclass
class MockLocated:
    label: str
    shape: MockShape
    color: str = ""
    canonical_label: str = ""
    score: Optional[float] = None
    difficulty: Optional[float] = None
    history: Optional[dict] = None


class TestAnnotateDifficulty:
    def test_fields_populated_with_detector(self):
        image = _synthetic_image()
        shape = MockShape(label=REF_LABEL, box=REF_BOX, aspect_ratio=1.0, area=1000, score=0.9)
        obj = MockLocated(label="a red cube", shape=shape)

        annotate_difficulty([obj], image=image, detector_fn=StableDetector(), n=4, seed=0)

        assert obj.difficulty is not None
        assert 0.0 <= obj.difficulty <= 1.0
        assert obj.score == pytest.approx(0.9)  # backfilled from shape.score

    def test_fields_populated_without_detector_fallback(self):
        # No detector_fn/image -> falls back to score + geometry only, still fills difficulty.
        shape = MockShape(label=REF_LABEL, box=REF_BOX, aspect_ratio=1.0, area=1000, score=0.7)
        obj = MockLocated(label="a red cube", shape=shape)

        annotate_difficulty([obj])

        assert obj.difficulty is not None
        assert 0.0 <= obj.difficulty <= 1.0

    def test_does_not_clobber_explicit_score(self):
        shape = MockShape(label=REF_LABEL, box=REF_BOX, score=0.2)
        obj = MockLocated(label="a red cube", shape=shape, score=0.99)

        annotate_difficulty([obj])

        assert obj.score == pytest.approx(0.99)

    def test_tough_object_gets_higher_difficulty_than_easy(self):
        image = _synthetic_image()
        easy_shape = MockShape(label=REF_LABEL, box=REF_BOX, aspect_ratio=1.0, area=1000, score=0.8)
        tough_shape = MockShape(label=REF_LABEL, box=REF_BOX, aspect_ratio=1.0, area=1000, score=0.8)
        easy_obj = MockLocated(label="a red cube", shape=easy_shape)
        tough_obj = MockLocated(label="a red cube", shape=tough_shape)

        annotate_difficulty([easy_obj], image=image, detector_fn=StableDetector(), n=6, seed=2)
        annotate_difficulty([tough_obj], image=image, detector_fn=JitteryDetector(), n=6, seed=2)

        assert tough_obj.difficulty > easy_obj.difficulty


class TestEnrich:
    def test_enrich_sets_canonical_label_score_and_difficulty(self):
        image = _synthetic_image()
        shape = MockShape(label=REF_LABEL, box=REF_BOX, aspect_ratio=1.0, area=1000, score=0.85)
        obj = MockLocated(label="a red cube", shape=shape)

        [enriched] = enrich([obj], image=image, detector_fn=StableDetector(), n=4, seed=0)

        assert enriched.canonical_label == "red block"
        assert enriched.score == pytest.approx(0.85)
        assert enriched.difficulty is not None
        assert 0.0 <= enriched.difficulty <= 1.0

    def test_enrich_without_detector_still_canonicalizes(self):
        shape = MockShape(label="Yellow Blocks!", box=REF_BOX, score=0.5)
        obj = MockLocated(label="Yellow Blocks!", shape=shape)

        [enriched] = enrich([obj])

        assert enriched.canonical_label == "yellow block"
        assert enriched.difficulty is not None
