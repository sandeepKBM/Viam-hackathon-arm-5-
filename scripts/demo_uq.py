#!/usr/bin/env python
"""Demo: easy -> tough UQ ranking (components/uq.py, W3).

Runs `uq.augmentation_consistency` + `uq.difficulty` / `annotate_difficulty`
over a small synthetic scene and prints an easy -> tough ranking table.

Fully OFFLINE: a synthetic (random-noise) image stands in for a captured
frame, and each object gets its own MOCK detector callable instead of a real
model:

  - "red block"  -> a STABLE mock detector: every augmented (jittered/
                    cropped/brightness-shifted) copy of the frame still
                    yields the same box + label. Easy: high confidence,
                    high augmentation-consistency, normal geometry.
  - "yellow block" -> a MODERATE mock detector: small positional jitter,
                    label correct most of the time. Medium.
  - "cup"        -> a JITTERY mock detector: a randomly-placed box and
                    frequently the wrong label (same idea as
                    tests/test_uq.py's JitteryDetector). Tough: low
                    confidence, low augmentation-consistency, and a
                    deliberately elongated/small box (geometry penalty).

No model download, no camera, no robot. Run with:

    .venv/bin/python scripts/demo_uq.py
    .venv/bin/python scripts/demo_uq.py --smoke     # quick CI path (fewer samples)

--- Running this against a REAL image + REAL detector -----------------------
Swap the mocks below for the real pipeline; the *same* `annotate_difficulty`
call shape works with either:

    import cv2
    from components.zeroshot import detect_objects, get_detector
    from components.uq import annotate_difficulty
    from components.shapes import LocatedShape

    bgr = cv2.imread("out/frame.png")
    dets = detect_objects(bgr)  # OWLv2, uses components/prompts.py's ZS_PROMPTS
    objects = [LocatedShape(label=d.label, x=0, y=0, z=0, shape=d, color=d.color)
               for d in dets]
    annotate_difficulty(
        objects, image=bgr, detector_fn=lambda im: get_detector().detect(im), n=5,
    )
    for o in sorted(objects, key=lambda o: o.difficulty):
        print(o.label, o.difficulty)

Moondream (components/vlm.py) has no scored `detect()`-under-jitter loop
built the same way OWLv2's does, but its `MoondreamVLM.detect(bgr, label)`
has the right signature (`image -> list[DetectedShape]`) to drop in as
`detector_fn=lambda im: vlm.detect(im, label)` for a single label at a time.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Run from repo root or elsewhere -- make sure `components` is importable
# regardless of cwd (this script lives in scripts/, one level down).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from components.shapes import DetectedShape, LocatedShape  # noqa: E402
from components.uq import annotate_difficulty  # noqa: E402


def _synthetic_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(240, 240, 3), dtype=np.uint8)


@dataclass
class MockDetection:
    box: tuple
    label: str
    score: Optional[float] = None


class StableDetector:
    """Easy: identical box+label under every augmentation."""

    def __init__(self, box: tuple, label: str):
        self.box, self.label = box, label

    def __call__(self, image: np.ndarray):
        return [MockDetection(box=self.box, label=self.label, score=0.93)]


class ModerateDetector:
    """Medium: small positional jitter, label usually correct."""

    def __init__(self, box: tuple, label: str, seed: int = 7):
        self.box, self.label = box, label
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray):
        x, y, w, h = self.box
        jx = x + self._rng.randint(-6, 6)
        jy = y + self._rng.randint(-6, 6)
        label = self.label if self._rng.random() > 0.15 else "unknown thing"
        return [MockDetection(box=(jx, jy, w, h), label=label, score=0.65)]


class JitteryDetector:
    """Tough: randomly-placed box, frequently the wrong label."""

    def __init__(self, label: str, seed: int = 42):
        self.label = label
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray):
        x = self._rng.randint(0, 300)
        y = self._rng.randint(0, 300)
        label = self._rng.choice([self.label, "red block", "unknown thing"])
        return [MockDetection(box=(x, y, 50, 50), label=label, score=0.35)]


def _make_scene():
    """(canonical label, box, aspect_ratio, area, score, detector_fn)."""
    easy_box = (100, 100, 50, 50)
    medium_box = (60, 60, 45, 45)
    tough_box = (150, 40, 90, 14)  # thin sliver -> geometry penalty too

    return [
        ("red block", easy_box, 1.0, 2500.0, 0.9, StableDetector(easy_box, "red block")),
        (
            "yellow block",
            medium_box,
            1.05,
            2025.0,
            0.7,
            ModerateDetector(medium_box, "yellow block"),
        ),
        ("cup", tough_box, 90 / 14, 1260.0, 0.4, JitteryDetector("cup")),
    ]


def run(n: int, seed: int) -> list[tuple]:
    image = _synthetic_image(seed)
    rows = []
    for label, box, aspect_ratio, area, score, detector_fn in _make_scene():
        shape = DetectedShape(
            label=label, cx=box[0] + box[2] // 2, cy=box[1] + box[3] // 2,
            area=area, vertices=4, aspect_ratio=aspect_ratio, box=box, score=score,
        )
        obj = LocatedShape(label=label, x=0.0, y=0.0, z=0.0, shape=shape, color="")
        annotate_difficulty([obj], image=image, detector_fn=detector_fn, n=n, seed=seed)
        rows.append((label, obj.score, obj.difficulty))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6, help="augmentation-consistency samples per object")
    parser.add_argument("--smoke", action="store_true", help="quick CI path (n=3)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n = 3 if args.smoke else args.n
    rows = run(n, args.seed)
    ranked = sorted(rows, key=lambda r: r[2])  # difficulty ascending: easy -> tough

    header = f"{'rank':>4}  {'object':<14}  {'score':>6}  {'difficulty':>10}"
    print("UQ ranking (easy -> tough) -- augmentation-consistency + difficulty fusion")
    print(header)
    print("-" * len(header))
    for i, (label, score, diff) in enumerate(ranked, start=1):
        print(f"{i:>4}  {label:<14}  {score:>6.2f}  {diff:>10.3f}")

    easiest, hardest = ranked[0], ranked[-1]
    ok = easiest[2] < hardest[2]
    print("-" * len(header))
    print(
        f"easiest: {easiest[0]} (difficulty={easiest[2]:.3f})  |  "
        f"toughest: {hardest[0]} (difficulty={hardest[2]:.3f})"
    )
    if not ok:
        print("warning: expected the jittery/mislabeling object to rank hardest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
