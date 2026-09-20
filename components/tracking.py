"""Object-tracking consumer: turn a Viam object-tracking vision service's
per-frame detections into `LocatedShape`s carrying a persistent `track_id`,
and match successive frames' detections against what the pipeline already
knows about -- so it can SKIP re-planning/re-capturing an object it's
already committed to picking.

WHY THIS EXISTS
----------------
Every detector this repo already has (`components/shapes.py`'s color/shape
finder, `components/zeroshot.py`'s OWLv2, `components/vlm.py`'s Moondream)
is per-frame stateless: run it twice on two frames of the same static scene
and you get two unrelated sets of detections with no notion that "block #2
in frame 1" and "block #2 in frame 2" are the same physical object. That is
fine for a single snapshot-then-plan pipeline, but it breaks down the moment
perception runs more than once per task (retries, re-looks via
`components/active_perception.py`, or simply polling while the arm is
mid-task): every new frame looks like a brand-new scene, so a naive pipeline
would re-plan (and, for policies that recapture between steps, re-detect)
every object on every frame, including the one it is already mid-grasp on.

Viam's object-tracking vision service (see README "Registry modules") fixes
the identity half of that problem: it wraps a detector and assigns each
detected object a persistent id that survives across frames as long as the
object keeps being seen. This module is the thin, offline-testable consumer
of that:

  - `get_tracked_objects` -- calls the tracker, wraps each detection as a
    `LocatedShape` with `.track_id` set (the additive field added to
    `components/shapes.py`'s `LocatedShape` for exactly this).
  - `match_to_known` -- pure/sync: given this frame's tracked objects and the
    pipeline's previously-known ones, decide which are "the same object"
    (by `track_id`, falling back to nearest-XY when a detection carries no
    id) so the caller knows which ones to SKIP re-planning for.

WORLD-FRAME CAVEAT
--------------------
A 2D detection (bounding box only, no depth) has no inherent 3D position.
`components/shapes.py`'s real world-frame pipeline (`locate_shapes_3d`,
`locate_block_colors`) gets there by ALSO sampling a depth image at the
detection's pixel and deprojecting through the camera intrinsics + a
`transform_pose` call -- machinery that assumes the detector's camera frame
gives you a depth-aligned RGB image, not just a bbox from a `Vision`
service. A tracking service, called purely through
`get_detections_from_camera`, hands back only 2D detections, so this module
does NOT invent a fake world position: it reports the detection's pixel
center as `x`/`y` (in the CAMERA's 2D pixel frame, not world mm) and `z=0.0`,
clearly marked below. A caller that needs real world XYZ for a tracked
object should fuse this module's `track_id` with a depth-aware locate call
(e.g. run `components.shapes.locate_shapes_3d` for the 3D position and match
its output to this module's tracked detections by pixel proximity /
timestamp) -- that fusion is intentionally left to the caller rather than
silently faked here, since a wrong "world" position derived from raw pixel
coordinates would be actively misleading to a downstream IK/motion call.

`match_to_known` itself never depends on x/y being world-mm -- nearest-XY
matching works the same whether x/y are pixels or millimeters, as long as
both `new` and `known` are expressed in the same units, which they will be
if both come from this module (or both come from a shared depth-aware
locate call once a caller does that fusion).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from viam.services.vision import VisionClient

from components.shapes import CAMERA_NAME, DetectedShape, LocatedShape

# Default max XY distance (same units as the LocatedShape x/y being compared
# -- pixels if these came straight from get_tracked_objects, world-mm if the
# caller fused in a depth-aware locate) for the nearest-XY fallback to count
# two detections as "the same object" when no track_id is available.
DEFAULT_XY_MATCH_TOLERANCE = float(os.environ.get("TRACKING_XY_TOLERANCE", "40.0"))


# ---------------------------------------------------------------------------
# Tracker -> LocatedShape
# ---------------------------------------------------------------------------


def _extract_label_and_track_id(det: Any) -> Tuple[str, Optional[str]]:
    """Pull `(label, track_id)` out of one tracker detection. Handles the
    several conventions real object-tracking vision services (and this
    module's tests) use to carry a persistent id, since the plain Viam
    `Detection` proto itself has no dedicated track-id field:

    1. A duck-typed attribute directly on the detection: `.track_id`,
       `.object_id`, or `.tracker_id` (what a mock/test detection, or a
       tracker module that subclasses/wraps `Detection`, would set).
    2. An `.extra` mapping (Viam's generic per-call extra-params channel)
       carrying `"track_id"` / `"id"` / `"object_id"`.
    3. Embedded in `class_name` itself, the most common real-world
       convention for trackers built on top of a plain detector: e.g.
       `"cup#14"`, `"cup::14"`, `"cup|14"` -- split on the first separator
       found, label on the left, id on the right.

    Returns `(label, None)` if no id can be found by any of the above --
    the detection is still usable, just not trackable (nearest-XY fallback
    only).
    """
    for attr in ("track_id", "object_id", "tracker_id"):
        val = getattr(det, attr, None)
        if val is not None and val != "":
            return str(getattr(det, "class_name", "") or ""), str(val)

    extra = getattr(det, "extra", None)
    if isinstance(extra, dict):
        for key in ("track_id", "id", "object_id"):
            if extra.get(key) not in (None, ""):
                return str(getattr(det, "class_name", "") or ""), str(extra[key])

    class_name = str(getattr(det, "class_name", "") or "")
    for sep in ("#", "::", "|"):
        if sep in class_name:
            label, _, tid = class_name.rpartition(sep)
            if label and tid:
                return label, tid

    return class_name, None


def tracked_shapes_from_detections(dets: Sequence[Any]) -> List[LocatedShape]:
    """Pure/sync core of `get_tracked_objects`: wrap already-fetched tracker
    detections as `LocatedShape`s with `.track_id` attached. `x`/`y` are the
    detection's pixel-space bbox center, `z=0.0` -- see the module docstring
    ("WORLD-FRAME CAVEAT") for why this doesn't fabricate a world position.
    Exposed separately so tests (and callers who already have detections in
    hand) don't need any asyncio/Viam plumbing.
    """
    out: List[LocatedShape] = []
    for det in dets:
        label, track_id = _extract_label_and_track_id(det)
        x_min = float(getattr(det, "x_min", 0.0))
        y_min = float(getattr(det, "y_min", 0.0))
        x_max = float(getattr(det, "x_max", x_min))
        y_max = float(getattr(det, "y_max", y_min))
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w, h = max(x_max - x_min, 0.0), max(y_max - y_min, 0.0)
        short, long_ = sorted((w, h)) if (w or h) else (0.0, 0.0)
        shape = DetectedShape(
            label=label,
            cx=int(cx),
            cy=int(cy),
            area=w * h,
            vertices=4,
            aspect_ratio=(long_ / short) if short > 1e-9 else 1.0,
            box=(x_min, y_min, w, h),
            score=getattr(det, "confidence", None),
        )
        out.append(
            LocatedShape(
                label=label,
                x=cx,
                y=cy,
                z=0.0,
                shape=shape,
                track_id=track_id,
            )
        )
    return out


async def get_tracked_objects(
    machine: Any,
    tracker_name: str,
    camera_name: str = CAMERA_NAME,
    *,
    world_frame: str = "world",
) -> List[LocatedShape]:
    """Call the Viam object-tracking vision service's
    `get_detections_from_camera(camera_name)` and return each detection as a
    `LocatedShape` with `.track_id` attached (see `tracked_shapes_from_detections`
    for the pure conversion logic and the pixel-vs-world caveat).

    `world_frame` is accepted for API symmetry with `components.shapes`'s
    `locate_shapes_3d`/`locate_block_colors` (and so a caller doing the
    depth-fusion described in the module docstring has an obvious place to
    pass it through) but is currently unused here -- this function alone
    never calls `transform_pose`, since without a depth sample there is no
    camera-frame 3D point to transform into `world_frame` in the first
    place.

    `machine` is duck-typed exactly like `components/perception3d.py`'s
    `get_object_grasps`/`components/vision.py`'s `detect`: passed straight to
    `VisionClient.from_robot`, so tests monkeypatch the module-level
    `VisionClient` name with a fake `from_robot(...).get_detections_from_camera(...)`.
    """
    del world_frame  # accepted for signature symmetry; see docstring
    tracker = VisionClient.from_robot(machine, tracker_name)
    dets = await tracker.get_detections_from_camera(camera_name)
    return tracked_shapes_from_detections(dets)


# ---------------------------------------------------------------------------
# Frame-to-frame identity matching
# ---------------------------------------------------------------------------


@dataclass
class TrackMatchResult:
    """Structured view of `match_to_known`'s return dict -- not what the
    function returns (a plain `dict`, per the integration contract), but a
    convenient typed accessor if a caller wants one. `match_to_known` builds
    its dict from exactly these fields."""

    matched: Dict[int, int] = field(default_factory=dict)     # new_index -> known_index
    new: List[int] = field(default_factory=list)               # new_index list, no match found
    method: Dict[int, str] = field(default_factory=dict)       # new_index -> "track_id"|"nearest_xy"
    skip_replan: List[int] = field(default_factory=list)       # == sorted(matched.keys())


def match_to_known(
    new: Sequence[LocatedShape],
    known: Sequence[LocatedShape],
    *,
    xy_tolerance: float = DEFAULT_XY_MATCH_TOLERANCE,
) -> dict:
    """Match this frame's tracked objects (`new`) against the pipeline's
    previously-known ones (`known`), so the caller can tell which objects
    are the SAME physical object it is already handling (skip re-plan /
    re-capture for those) vs. genuinely new.

    Matching rule, in priority order (as specified):

    1. **By `track_id`** -- if a `new` object has a non-empty `track_id` and
       some *unclaimed* `known` object shares that exact id, they're the
       same object. This is authoritative: if a `new` object HAS a
       track_id but it doesn't match any known object's id, it is treated
       as new -- NOT XY-fallback-matched onto some other object -- because a
       tracker-assigned id is a stronger identity signal than position and a
       real id mismatch usually means a genuinely different (or
       re-acquired-as-new) object.
    2. **Nearest-XY fallback** -- only for `new` objects with NO `track_id`
       (or whose source didn't track identity at all): match to the closest
       unclaimed `known` object within `xy_tolerance` (Euclidean, same units
       as `.x`/`.y` on both sides -- see the module docstring's world-frame
       caveat). No unclaimed candidate within tolerance -> new.

    Each `known` object is claimed by at most one `new` object (first
    match wins, in `new`'s order) so two new detections can't both silently
    claim the same prior object.

    Returns a dict:
        {
          "matched": {new_index: known_index, ...},  # same object as before
          "new": [new_index, ...],                    # genuinely new objects
          "skip_replan": [new_index, ...],             # == sorted(matched)
          "method": {new_index: "track_id" | "nearest_xy", ...},
        }

    Pure and synchronous -- no I/O, so fully unit-testable.
    """
    matched: Dict[int, int] = {}
    method: Dict[int, str] = {}
    new_indices: List[int] = []

    known_by_track_id: Dict[str, int] = {}
    for ki, k in enumerate(known):
        tid = getattr(k, "track_id", None)
        if tid:
            known_by_track_id.setdefault(str(tid), ki)

    unclaimed_known = set(range(len(known)))

    remaining: List[int] = []
    for ni, n in enumerate(new):
        tid = getattr(n, "track_id", None)
        if tid:
            ki = known_by_track_id.get(str(tid))
            if ki is not None and ki in unclaimed_known:
                matched[ni] = ki
                method[ni] = "track_id"
                unclaimed_known.discard(ki)
                continue
            # Had an id, but no (unclaimed) known object shares it: an
            # authoritative id mismatch, not a "no id" case -- don't fall
            # through to XY matching (see docstring).
            new_indices.append(ni)
            continue
        remaining.append(ni)

    for ni in remaining:
        n = new[ni]
        best_ki: Optional[int] = None
        best_d: Optional[float] = None
        for ki in unclaimed_known:
            k = known[ki]
            d = math.hypot(n.x - k.x, n.y - k.y)
            if d <= xy_tolerance and (best_d is None or d < best_d):
                best_d = d
                best_ki = ki
        if best_ki is not None:
            matched[ni] = best_ki
            method[ni] = "nearest_xy"
            unclaimed_known.discard(best_ki)
        else:
            new_indices.append(ni)

    return {
        "matched": matched,
        "new": new_indices,
        "skip_replan": sorted(matched.keys()),
        "method": method,
    }
