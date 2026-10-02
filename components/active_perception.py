"""Active perception: UQ-triggered "center-and-zoom" re-look.

Ideas from "Strategic Vantage Selection" / ActiveVLA / GraspView: when the
detector's own uncertainty (components/uq.py's ``.difficulty``) says a
reading is shaky, don't trust the far-away/off-center pixels -- move the
eye-in-hand (wrist) camera CLOSER and CENTERED over the object, re-perceive,
and keep whichever reading (the original wide shot or the zoomed-in one) is
more confident. This is exactly the hook UQ (W3) was built for: the win here
comes from better pixels (bigger, centered object in frame), not a different
detector or model -- ``vision``'s re-detect call is whatever detector
produced the original scene, and ``detector_fn`` (if given) is the same
augmentation-consistency probe ``components.uq`` already knows how to run.

This module only ever COMPOSES existing pieces:
  - ``components.uq.enrich``            -- canonicalize + re-score + re-difficulty
                                            the zoomed-in reading, exactly the
                                            same way ``pipeline.enrich_and_seed``
                                            scores the original scene.
  - ``components.canonicalize``         -- via ``uq.enrich`` -- so "cup" vs
                                            "mug" compare as the same object
                                            (color-sort's own canonicalize.py
                                            delegates to
                                            components.constants.normalize_object,
                                            e.g. "mug" -> "cup").
  - ``components.safety.in_workspace``  -- the zoom pose is safety-checked
                                            before the arm ever moves.
  - ``components.constants``            -- MIN_Z / TRAVEL_Z / PICK_ORIENTATION,
                                            imported (never edited) to stay
                                            consistent with the rest of the
                                            real-arm-tuned stack.
  - ``arm.move_to_position(x, y, z, **PICK_ORIENTATION)`` -- position control
                                            only, the same primitive
                                            ``components/pickplace.py`` and
                                            ``components/arm.py`` already use.

Nothing here is imported by any of those modules, so wiring this in is
strictly additive: existing detectors/skills/arm code are unmodified and
this whole pass is skippable (an empty/absent ``scene`` uncertainty list is
a no-op, and callers that never import this module see no behavior change).

ADAPTATION NOTE (color-sort's manipulation API)
--------------------------------------------------
Unlike components/retry.py, components/skills.py and components/pipeline.py,
this module never calls ``components.pickplace.PickPlace`` at all -- it only
needs a bare ``arm.move_to_position(x, y, z, **kw)`` coroutine (color-sort's
``components.arm.ArmComponent`` provides exactly that signature, with
``o_x``/``o_y``/``o_z``/``theta``/``check_workspace``/``floor`` all accepted
as the ``**kw`` this module passes via ``PICK_ORIENTATION``) and a
``vision.locate_shapes()`` coroutine (color-sort's
``components.vision.VisionComponent.locate_shapes()`` matches this contract
exactly). No changes were needed to port this module -- it composes the
same building blocks color-sort already exposes under the same names.

DEPENDENCY CONTRACT (duck-typed, so tests can pass mocks)
----------------------------------------------------------
``arm``    -- needs one coroutine: ``move_to_position(x, y, z, **kw)``
              (matches ``components.arm.ArmComponent`` / ``MockArm`` in
              ``tests/test_active_perception.py``).

``vision`` -- needs one coroutine: ``locate_shapes()`` returning a sequence
              of fresh, world-located detections (duck-typed like
              ``components.shapes.LocatedShape``: ``.x``/``.y``/``.z``/
              ``.label``/``.score``), captured from the CAMERA'S CURRENT
              POSE -- i.e. whatever ``arm.move_to_position`` just did. This
              is the same contract ``components.vision.VisionComponent``
              already fulfills (``locate_shapes``/``locate_blocks`` both
              re-capture + re-detect from wherever the arm currently is).
              Callers wire whichever detector produced the original scene as
              this coroutine.

RE-LOOK TRIGGER
----------------
An object re-looks iff ``object.difficulty >= difficulty_threshold``
(default 0.5, the same "uncertain" cutoff UQ documents). Confident objects
(below threshold) are left completely alone -- no camera move, no re-detect
call -- by construction (they're never added to the candidate list below).

ZOOM-POSE COMPUTATION (how it centers + how the safe height is chosen)
------------------------------------------------------------------------
Centering: the wrist camera is commanded straight above the object's own
current best-known (x, y) -- ``arm.move_to_position(obj.x, obj.y, zoom_z,
**PICK_ORIENTATION)`` -- so the object sits in the middle of the new frame
(the whole reason a re-look helps: same detector, bigger + centered pixels).

Height: ``zoom_height_mm`` is a single override for the whole call (pass a
fixed number to force one height for every re-look); when left at the
default ``None``, each object gets its own safe height:

    zoom_z = clip(obj.z + ZOOM_CLEARANCE_MM, MIN_Z, TRAVEL_Z)

i.e. clearance above the object's own top (``ZOOM_CLEARANCE_MM``, default
80mm) but never below the hard floor (``MIN_Z``) and never above/farther
than the normal travel height (``TRAVEL_Z``) -- so the computed pose is
always a genuine "zoom in" (closer than the ordinary capture height), never
farther away, and it's SAFETY-CHECKED (``in_workspace`` + ``z >= MIN_Z``)
before the arm is ever commanded there; an unsafe pose is skipped (no move,
object left untouched).

VOTE / ADOPT RULE
-------------------
After the re-look, the zoomed-in detection nearest the object's original
(x, y) (within ``match_tolerance_mm``, default 60mm -- since the arm just
hovered directly over it, the true match should be very close) is compared
against the original reading:

    adopt the zoom reading iff its difficulty is strictly lower, OR
    (difficulty ties AND its raw detector score is higher).

On adopt: ``label``, ``canonical_label``, ``score``, ``difficulty`` and the
refined ``(x, y)`` are overwritten with the zoom reading's values (z, color,
shape and history are left as-is -- this pass only refines identity/
confidence + planar position, never the grasp height). If no zoom detection
matches within tolerance, the original reading is kept unchanged.

TIMING BUDGET
---------------
``max_relooks`` (default 3, mirrors the W6 timing-budget pattern used
elsewhere in this repo, e.g. ``uq``'s ``n``) caps how many objects get a
re-look per call: the most-uncertain objects (highest ``.difficulty``) are
served first, and once ``max_relooks`` of them have been attempted, the rest
are left as-is even if they're above ``difficulty_threshold`` -- so a scene
with many uncertain objects can't blow the per-task timing budget.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence

from components import uq
from components.constants import MIN_Z, PICK_ORIENTATION, TRAVEL_Z
from components.safety import in_workspace

# --- tunables (editable; env-overridable like the rest of this repo) -----
DEFAULT_DIFFICULTY_THRESHOLD = float(
    os.environ.get("ACTIVE_PERCEPTION_DIFFICULTY_THRESHOLD", 0.5)
)
DEFAULT_MAX_RELOOKS = int(os.environ.get("ACTIVE_PERCEPTION_MAX_RELOOKS", 3))
ZOOM_CLEARANCE_MM = float(os.environ.get("ACTIVE_PERCEPTION_ZOOM_CLEARANCE_MM", 80.0))
MATCH_TOLERANCE_MM = float(os.environ.get("ACTIVE_PERCEPTION_MATCH_TOLERANCE_MM", 60.0))


def _get(obj: Any, name: str, default=None):
    """Duck-typed attribute/dict access (mirrors components.uq._get)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _default_zoom_height(obj_z: float) -> float:
    """A safe zoom height: clearance above the object's own top, clamped
    between MIN_Z (hard floor) and TRAVEL_Z (normal/farther capture height)
    so the result is always a genuine zoom-IN, never farther away."""
    candidate = obj_z + ZOOM_CLEARANCE_MM
    return float(min(max(candidate, MIN_Z), TRAVEL_Z))


def zoom_pose(obj: Any, zoom_height_mm: Optional[float] = None) -> tuple:
    """The (x, y, z) the wrist camera should move to for a closer, centered
    look at ``obj``: directly above its current (x, y), at ``zoom_height_mm``
    if given, else the per-object safe default (see module docstring)."""
    x = float(_get(obj, "x"))
    y = float(_get(obj, "y"))
    if zoom_height_mm is not None:
        z = float(zoom_height_mm)
    else:
        z = _default_zoom_height(float(_get(obj, "z", MIN_Z) or MIN_Z))
    return x, y, z


def pose_is_safe(x: float, y: float, z: float) -> bool:
    """SAFETY-check a candidate zoom pose before the arm moves there:
    inside the taught workspace polygon and above the hard floor."""
    return in_workspace(x, y) and z >= MIN_Z


def _nearest_match(
    target: Any, candidates: Sequence[Any], tolerance_mm: float
) -> Optional[Any]:
    """The re-detected object nearest ``target``'s (x, y), within tolerance.

    The arm just hovered directly over the object's last-known position, so
    the true re-detection should land very close by; anything farther than
    ``tolerance_mm`` is treated as "didn't see it" rather than a false match.
    """
    tx, ty = float(_get(target, "x")), float(_get(target, "y"))
    best: Optional[Any] = None
    best_d: Optional[float] = None
    for c in candidates:
        cx, cy = float(_get(c, "x")), float(_get(c, "y"))
        d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if d <= tolerance_mm and (best_d is None or d < best_d):
            best, best_d = c, d
    return best


def _more_confident(candidate: Any, original: Any) -> bool:
    """VOTE rule: prefer strictly lower difficulty; on a tie, prefer the
    higher raw detector score. Unknown difficulty is treated as maximally
    uncertain (1.0) so a scored candidate always beats an unscored one."""
    cd = _get(candidate, "difficulty", None)
    od = _get(original, "difficulty", None)
    cd = 1.0 if cd is None else float(cd)
    od = 1.0 if od is None else float(od)
    if cd != od:
        return cd < od
    cs = _get(candidate, "score", None) or 0.0
    os_ = _get(original, "score", None) or 0.0
    return cs > os_


def _adopt(original: Any, candidate: Any) -> None:
    """Overwrite the object in place with the more-confident zoom reading:
    identity (label/canonical_label), confidence (score/difficulty) and the
    refined planar position (x, y). z/color/shape/history are untouched --
    this pass refines WHAT the object is and WHERE it is in-plane, not the
    grasp height."""
    original.label = _get(candidate, "label", original.label)
    original.canonical_label = _get(candidate, "canonical_label", original.canonical_label)
    original.score = _get(candidate, "score", None)
    original.difficulty = _get(candidate, "difficulty", None)
    original.x = float(_get(candidate, "x"))
    original.y = float(_get(candidate, "y"))


async def refine_uncertain(
    scene: Sequence[Any],
    arm: Any,
    vision: Any,
    *,
    difficulty_threshold: float = DEFAULT_DIFFICULTY_THRESHOLD,
    zoom_height_mm: Optional[float] = None,
    max_relooks: int = DEFAULT_MAX_RELOOKS,
    detector_fn: Optional[Callable] = None,
) -> Sequence[Any]:
    """UQ-triggered center-and-zoom active-perception pass.

    For the most-uncertain objects in ``scene`` (``.difficulty >=
    difficulty_threshold``, highest difficulty first, capped at
    ``max_relooks``): compute a closer/centered zoom pose, safety-check it,
    move the arm there (position control), re-detect via ``vision``,
    re-score via ``components.uq`` (reusing ``detector_fn`` for the same
    augmentation-consistency signal the original scene was scored with),
    then adopt the more-confident reading (see module docstring for the
    exact vote rule). Confident objects (below threshold) are never touched
    -- no move, no re-detect. Mutates and returns ``scene``.
    """
    candidates: List[Any] = [
        obj for obj in scene if (_get(obj, "difficulty", None) or 0.0) >= difficulty_threshold
    ]
    candidates.sort(key=lambda o: _get(o, "difficulty", 0.0) or 0.0, reverse=True)
    targets = candidates[: max(int(max_relooks), 0)]

    for obj in targets:
        label = _get(obj, "canonical_label", None) or _get(obj, "label", "?")
        x, y, z = zoom_pose(obj, zoom_height_mm)

        if not pose_is_safe(x, y, z):
            print(
                f"[active-perception] skip {label!r}: zoom pose "
                f"({x:.1f}, {y:.1f}, {z:.1f}) is unsafe -- not moving"
            )
            continue

        print(
            f"[active-perception] re-look at {label!r} "
            f"(difficulty={_get(obj, 'difficulty', 0.0):.2f}) -> "
            f"zoom pose ({x:.1f}, {y:.1f}, {z:.1f})"
        )
        await arm.move_to_position(x, y, z, **PICK_ORIENTATION)

        redetected = list(await vision.locate_shapes() or [])
        if not redetected:
            print(f"[active-perception]   no re-detection at zoom pose for {label!r}")
            continue

        uq.enrich(redetected, image=None, detector_fn=detector_fn)

        match = _nearest_match(obj, redetected, MATCH_TOLERANCE_MM)
        if match is None:
            print(f"[active-perception]   zoom re-detection didn't match {label!r} within tolerance")
            continue

        if _more_confident(match, obj):
            before_d = _get(obj, "difficulty", None)
            before_label = _get(obj, "canonical_label", None) or _get(obj, "label", "")
            _adopt(obj, match)
            print(
                f"[active-perception]   adopted zoom reading for {before_label!r}: "
                f"difficulty {before_d} -> {obj.difficulty:.2f}, "
                f"label={obj.canonical_label!r}"
            )
        else:
            print(f"[active-perception]   kept original reading for {label!r} (zoom not more confident)")

    return scene
