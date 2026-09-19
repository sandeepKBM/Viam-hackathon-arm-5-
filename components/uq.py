"""Uncertainty / difficulty scoring (W3).

Fuses three offline-computable signals into a single per-object
``difficulty in [0, 1]`` (0 = easy/confident, 1 = hard/uncertain):

  (a) detector confidence      -- the raw ``score`` OWLv2 (or any scored
                                   detector) reports for the box.
  (b) augmentation-consistency -- run the SAME detector N times on small
                                   jittered/cropped/brightness-perturbed
                                   copies of the image and measure how much
                                   the box and label agree across runs
                                   (model-agnostic epistemic-uncertainty
                                   proxy; works for OWLv2 and, if it grows a
                                   scored ``detect()``, Moondream too).
  (c) geometry                 -- extreme aspect ratio / extreme area boxes
                                   are harder to grasp confidently even when
                                   the detector is "sure".

The detector used for (b) is an injected callable (``detector_fn: image ->
list[detection]``) so tests can pass a cheap mock instead of loading a real
model. N is exposed as a parameter -- the timing budget (W6) is expected to
bound it at the call site.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Detection = Any  # duck-typed: DetectedShape, or any obj/dict with box/label/score
DetectorFn = Callable[[np.ndarray], Sequence[Detection]]

# --- fusion weights (editable; must sum to something > 0) -----------------
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "score": float(os.environ.get("UQ_WEIGHT_SCORE", 0.4)),
    "consistency": float(os.environ.get("UQ_WEIGHT_CONSISTENCY", 0.4)),
    "geometry": float(os.environ.get("UQ_WEIGHT_GEOMETRY", 0.2)),
}

# --- geometry penalty bounds (editable) ------------------------------------
_ASPECT_EASY_MAX = float(os.environ.get("UQ_ASPECT_EASY_MAX", 1.6))
_AREA_EASY_MIN = float(os.environ.get("UQ_AREA_EASY_MIN", 600.0))
_AREA_EASY_MAX = float(os.environ.get("UQ_AREA_EASY_MAX", 20000.0))

# --- default augmentation-consistency params --------------------------------
DEFAULT_N = int(os.environ.get("UQ_AUG_N", 5))
_MAX_SHIFT_FRAC = float(os.environ.get("UQ_AUG_SHIFT_FRAC", 0.02))
_MAX_CROP_FRAC = float(os.environ.get("UQ_AUG_CROP_FRAC", 0.05))
_MAX_BRIGHTNESS = float(os.environ.get("UQ_AUG_BRIGHTNESS", 0.15))


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _get(obj: Any, name: str, default=None):
    """Duck-typed attribute/dict access used throughout this module."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _geometry_penalty(aspect_ratio: Optional[float], area: Optional[float]) -> float:
    """Extreme aspect ratio / area -> higher penalty in [0, 1]. Unknowns -> 0."""
    terms = []
    if aspect_ratio is not None:
        if aspect_ratio <= _ASPECT_EASY_MAX:
            terms.append(0.0)
        else:
            over = (aspect_ratio - _ASPECT_EASY_MAX) / _ASPECT_EASY_MAX
            terms.append(_clip01(over))
    if area is not None and area > 0:
        if _AREA_EASY_MIN <= area <= _AREA_EASY_MAX:
            terms.append(0.0)
        elif area < _AREA_EASY_MIN:
            terms.append(_clip01((_AREA_EASY_MIN - area) / _AREA_EASY_MIN))
        else:
            terms.append(_clip01((area - _AREA_EASY_MAX) / _AREA_EASY_MAX))
    if not terms:
        return 0.0
    return sum(terms) / len(terms)


def difficulty(
    score: Optional[float] = None,
    consistency: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
    area: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Fuse confidence + augmentation-consistency + geometry into [0, 1].

    Unknown signals fall back to a neutral 0.5 (neither easy nor hard) so a
    missing signal doesn't silently zero the whole score. Monotonic: for a
    fixed ``consistency``/geometry, increasing ``score`` strictly decreases
    (or holds) the result, and likewise for ``consistency``.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    score_term = 1.0 - _clip01(score if score is not None else 0.5)
    consistency_term = 1.0 - _clip01(consistency if consistency is not None else 0.5)
    geometry_term = _geometry_penalty(aspect_ratio, area)

    total_w = w.get("score", 0.0) + w.get("consistency", 0.0) + w.get("geometry", 0.0)
    if total_w <= 0:
        return 0.5
    d = (
        w.get("score", 0.0) * score_term
        + w.get("consistency", 0.0) * consistency_term
        + w.get("geometry", 0.0) * geometry_term
    ) / total_w
    return _clip01(d)


def _augment(
    bgr: np.ndarray,
    rng: np.random.Generator,
    max_shift_frac: float,
    max_crop_frac: float,
    max_brightness: float,
) -> np.ndarray:
    """One jittered copy: small translate + center crop/zoom + brightness."""
    h, w = bgr.shape[:2]
    out = bgr

    dx = int(rng.uniform(-max_shift_frac, max_shift_frac) * w)
    dy = int(rng.uniform(-max_shift_frac, max_shift_frac) * h)
    if dx or dy:
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    crop_frac = rng.uniform(0.0, max_crop_frac)
    if crop_frac > 0:
        cx, cy = int(w * crop_frac / 2), int(h * crop_frac / 2)
        if h - 2 * cy > 0 and w - 2 * cx > 0:
            out = cv2.resize(out[cy : h - cy, cx : w - cx], (w, h), interpolation=cv2.INTER_LINEAR)

    if max_brightness > 0:
        factor = 1.0 + rng.uniform(-max_brightness, max_brightness)
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(bgr.dtype)

    return out


def augmentation_consistency(
    detector_fn: DetectorFn,
    bgr: np.ndarray,
    ref_box: Tuple[float, float, float, float],
    ref_label: Optional[str] = None,
    n: int = DEFAULT_N,
    max_shift_frac: float = _MAX_SHIFT_FRAC,
    max_crop_frac: float = _MAX_CROP_FRAC,
    max_brightness: float = _MAX_BRIGHTNESS,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Run ``detector_fn`` N times under small jitter; measure agreement.

    For each jittered frame, the detection with highest IoU against
    ``ref_box`` is taken as this run's "match" (if the frame yields any
    detections at all). Returns box-IoU stats + label agreement + a fused
    ``consistency`` in [0, 1] (1 = perfectly stable across augmentations).
    """
    rng = np.random.default_rng(seed)
    n = max(int(n), 1)

    ious: List[float] = []
    label_hits = 0
    matched = 0

    for _ in range(n):
        aug = _augment(bgr, rng, max_shift_frac, max_crop_frac, max_brightness)
        dets = list(detector_fn(aug) or [])
        if not dets:
            ious.append(0.0)
            continue
        best = max(dets, key=lambda d: _iou(ref_box, _get(d, "box", (0, 0, 0, 0))))
        best_box = _get(best, "box", (0, 0, 0, 0))
        ious.append(_iou(ref_box, best_box))
        matched += 1
        best_label = str(_get(best, "label", "") or "")
        if ref_label is None or best_label.lower() == str(ref_label).lower():
            label_hits += 1

    mean_iou = float(np.mean(ious)) if ious else 0.0
    iou_std = float(np.std(ious)) if ious else 0.0
    label_agreement = label_hits / n
    hit_rate = matched / n
    consistency = _clip01(max(0.0, mean_iou - iou_std) * label_agreement * hit_rate)

    return {
        "mean_iou": mean_iou,
        "iou_std": iou_std,
        "label_agreement": label_agreement,
        "hit_rate": hit_rate,
        "consistency": consistency,
        "n": n,
    }


def annotate_difficulty(
    objects: Sequence[Any],
    image: Optional[np.ndarray] = None,
    detector_fn: Optional[DetectorFn] = None,
    n: int = DEFAULT_N,
    weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Sequence[Any]:
    """Fill ``.difficulty`` (and backfill ``.score`` if unset) on each object.

    ``objects`` are duck-typed ``LocatedShape``-like: each optionally has a
    ``.shape`` (``DetectedShape``-like, with ``.box``/``.label``/``.score``/
    ``.aspect_ratio``/``.area``) and its own ``.score``/``.difficulty``.
    When ``detector_fn`` and ``image`` are both given, augmentation-
    consistency is computed per object (N detector calls each) and folded
    into the fusion; otherwise difficulty falls back to score + geometry.
    """
    for obj in objects:
        shape = _get(obj, "shape", None)

        score = _get(obj, "score", None)
        if score is None:
            score = _get(shape, "score", None)

        aspect_ratio = _get(shape, "aspect_ratio", None)
        if aspect_ratio is None:
            aspect_ratio = _get(obj, "aspect_ratio", None)

        area = _get(shape, "area", None)
        if area is None:
            area = _get(obj, "area", None)

        consistency = None
        box = _get(shape, "box", None)
        if detector_fn is not None and image is not None and box is not None:
            label = _get(shape, "label", None)
            agreement = augmentation_consistency(
                detector_fn, image, box, ref_label=label, n=n, seed=seed
            )
            consistency = agreement["consistency"]

        d = difficulty(
            score=score,
            consistency=consistency,
            aspect_ratio=aspect_ratio,
            area=area,
            weights=weights,
        )

        if isinstance(obj, dict):
            obj["difficulty"] = d
            if obj.get("score") is None:
                obj["score"] = score
        else:
            obj.difficulty = d
            if getattr(obj, "score", None) is None:
                obj.score = score

    return objects


def enrich(
    objects: Sequence[Any],
    image: Optional[np.ndarray] = None,
    detector_fn: Optional[DetectorFn] = None,
    n: int = DEFAULT_N,
    weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Sequence[Any]:
    """Thin end-to-end entry point: canonicalize + score/difficulty.

    Sets ``.canonical_label`` (W2) and ``.score``/``.difficulty`` (W3) on
    each object in place, and returns the same list for convenience.
    """
    from components.canonicalize import canonical_key

    for obj in objects:
        canonical_key(obj)
    return annotate_difficulty(
        objects, image=image, detector_fn=detector_fn, n=n, weights=weights, seed=seed
    )
