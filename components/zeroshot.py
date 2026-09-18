"""Zero-shot (open-vocabulary) object detection via HuggingFace transformers.

You give it *text* prompts ("red block", "yellow cube") instead of tuning HSV
ranges, and it returns boxes with no task-specific training. Output is a list of
``DetectedShape`` (the same dataclass the OpenCV path produces) so the existing
3D deprojection + pick pipeline in ``components.shapes`` works unchanged.

Default model is OWLv2 (Google), a solid open-vocabulary detector. Grounding DINO
is also supported through the same ``zero-shot-object-detection`` pipeline.

Heavy deps (torch, transformers, pillow) are imported lazily so importing this
module never breaks the OpenCV-only workflow. Install them with:

    pip install -r requirements-zeroshot.txt
"""

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np

from components.shapes import (
    CAMERA_NAME,
    DetectedShape,
    LocatedShape,
    _color_depth_intrinsics,
    _depth_at,
    _pixel_to_world,
)

# google/owlv2-base-patch16-ensemble is a good default; owlvit-base-patch32 is
# lighter/faster; IDEA-Research/grounding-dino-tiny is the other common option.
ZS_MODEL = os.environ.get("ZS_MODEL", "google/owlv2-base-patch16-ensemble")
ZS_THRESHOLD = float(os.environ.get("ZS_THRESHOLD", "0.1"))
# Prompts default to the two blocks this cell sorts. Comma-separate to override,
# e.g. ZS_PROMPTS="red block,yellow block,blue block".
ZS_PROMPTS = tuple(
    p.strip()
    for p in os.environ.get("ZS_PROMPTS", "red block,yellow block").split(",")
    if p.strip()
)

# Map a free-text prompt to the pipeline's color bucket used downstream.
_KNOWN_COLORS = ("red", "yellow", "green", "blue", "orange")


def _color_from_label(label: str) -> str:
    low = label.lower()
    for c in _KNOWN_COLORS:
        if c in low:
            return c
    return ""


class ZeroShotDetector:
    """Lazy-loaded wrapper around a HuggingFace zero-shot detection pipeline."""

    def __init__(
        self,
        model: str = ZS_MODEL,
        prompts: Sequence[str] = ZS_PROMPTS,
        threshold: float = ZS_THRESHOLD,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.prompts = list(prompts)
        self.threshold = threshold
        self._device = device
        self._pipe = None

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        import torch  # noqa: WPS433 (lazy, heavy)
        from transformers import pipeline  # noqa: WPS433

        if self._device is None:
            device = 0 if torch.cuda.is_available() else -1
        else:
            device = self._device
        self._pipe = pipeline(
            task="zero-shot-object-detection",
            model=self.model,
            device=device,
        )
        return self._pipe

    def detect(
        self,
        bgr: np.ndarray,
        prompts: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
    ) -> List[DetectedShape]:
        """Run detection on a BGR (OpenCV) image, return ``DetectedShape``s."""
        from PIL import Image  # noqa: WPS433

        pipe = self._ensure_pipeline()
        queries = list(prompts) if prompts is not None else self.prompts
        thr = self.threshold if threshold is None else threshold

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        raw = pipe(pil, candidate_labels=queries, threshold=thr)

        shapes: List[DetectedShape] = []
        for r in raw:
            b = r["box"]
            x0, y0 = int(b["xmin"]), int(b["ymin"])
            x1, y1 = int(b["xmax"]), int(b["ymax"])
            w, h = max(x1 - x0, 1), max(y1 - y0, 1)
            short, long_ = sorted((w, h))
            label = r["label"]
            shapes.append(
                DetectedShape(
                    label=label,
                    cx=(x0 + x1) // 2,
                    cy=(y0 + y1) // 2,
                    area=float(w * h),
                    vertices=4,
                    aspect_ratio=long_ / max(short, 1e-6),
                    box=(x0, y0, w, h),
                    color=_color_from_label(label) or label,
                )
            )
        shapes.sort(key=lambda s: s.area, reverse=True)
        return shapes


# Module-level singleton so the model is loaded once per process.
_DETECTOR: Optional[ZeroShotDetector] = None


def get_detector() -> ZeroShotDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = ZeroShotDetector()
    return _DETECTOR


def detect_objects(
    bgr: np.ndarray,
    prompts: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
) -> List[DetectedShape]:
    return get_detector().detect(bgr, prompts=prompts, threshold=threshold)


def annotate(bgr: np.ndarray, shapes: List[DetectedShape]) -> np.ndarray:
    out = bgr.copy()
    for s in shapes:
        x, y, w, h = s.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            s.label,
            (x, y - 8 if y > 20 else y + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


async def locate_objects_zeroshot(
    machine,
    camera_name: str = CAMERA_NAME,
    world_frame: str = "world",
    prompts: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
) -> List[LocatedShape]:
    """Zero-shot analogue of ``shapes.locate_block_colors``.

    Detects with text prompts, then deprojects each box centroid to world mm
    using the same depth + intrinsics helpers as the color pipeline.
    """
    from viam.components.camera import Camera

    cam = Camera.from_robot(machine, camera_name)
    bgr, depth_mm, intr = await _color_depth_intrinsics(cam)
    objects = detect_objects(bgr, prompts=prompts, threshold=threshold)
    if not objects:
        return []

    table_depth = 0.0
    if depth_mm is not None:
        valid = depth_mm[depth_mm > 0]
        if valid.size:
            table_depth = float(np.median(valid))

    located: List[LocatedShape] = []
    for s in objects:
        z = _depth_at(depth_mm, s)
        if z <= 0:
            z = table_depth
        if z <= 0:
            print(f"  skip {s.label} px=({s.cx},{s.cy}): no depth")
            continue
        p = await _pixel_to_world(machine, camera_name, s.cx, s.cy, z, intr, world_frame)
        located.append(
            LocatedShape(label=s.label, x=p.x, y=p.y, z=p.z, shape=s, color=s.color)
        )
    return located
