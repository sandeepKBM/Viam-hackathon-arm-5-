"""Declutter-aware pick planning.

This is a hand-authored GEOMETRIC HEURISTIC, not a learned policy. Given a
TARGET object and the list of ALL currently-detected objects, it decides
whether the target's top-down grasp is BLOCKED by a neighboring object, and
if so, produces an ordered "move blocker(s) aside, then pick the target"
plan.

Everything here is pure and deterministic: no robot I/O, no camera calls, no
async, no Viam imports. It is safe (and intended) to unit test offline with
synthetic objects. The execution side (actually driving the arm/gripper to
carry out a plan) lives in components/pickplace.py, which consumes the
`DeclutterPlan` produced here and calls the existing ArmComponent /
GripperComponent primitives.

Objects are duck-typed: anything with numeric `.x` and `.y` attributes
(mm, in the arm's world frame -- same convention as
components.shapes.LocatedShape) works as a target or a blocker. A `.color`
attribute is used only for a deterministic tie-break when ordering multiple
blockers; it is optional.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from components.safety import in_workspace

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Approximate half-width (mm) of the (open) gripper fingers plus a small
# margin for the block footprint itself. There is no gripper/CAD datasheet
# wired into this repo, so this is a conservative hand-picked estimate, not
# a measured value: it is sized so that two ~30-50mm cube blocks sitting
# this close together would have their footprints overlap the descending
# finger sweep of a top-down grasp. If a real gripper geometry becomes
# available, replace this constant (or override via env var) with a value
# derived from it.
GRIPPER_CLEARANCE_MM = float(os.environ.get("GRIPPER_CLEARANCE_MM", 35.0))

# Minimum separation (mm) a candidate temp clear-zone must keep from every
# other known object position (and from the target) so that relocating a
# blocker there doesn't just create a new collision. Defaults to the same
# radius as the grasp clearance -- same physical justification.
CLEAR_ZONE_MARGIN_MM = float(
    os.environ.get("CLEAR_ZONE_MARGIN_MM", GRIPPER_CLEARANCE_MM)
)

# Search pattern used to find a temp clear-zone near a blocker: try a ring
# of candidate points at increasing radii and angles around the blocker's
# current position, and take the first one that is both in-workspace and
# clear of every other known position.
_SEARCH_RADII_MM: Tuple[float, ...] = (60.0, 90.0, 120.0, 160.0, 200.0, 260.0)
_SEARCH_ANGLES_DEG: Tuple[float, ...] = tuple(range(0, 360, 30))


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveAside:
    """Relocate a blocking object out of the way before the real pick."""

    obj: object
    to_xy: Tuple[float, float]


@dataclass(frozen=True)
class Pick:
    """Perform the actual top-down pick of `obj` (the original target)."""

    obj: object


DeclutterAction = Union[MoveAside, Pick]


@dataclass
class DeclutterPlan:
    """An ordered list of actions: zero or more MoveAside, then one Pick."""

    actions: List[DeclutterAction]

    @property
    def blocked(self) -> bool:
        return any(isinstance(a, MoveAside) for a in self.actions)

    @property
    def blockers_moved(self) -> List[object]:
        return [a.obj for a in self.actions if isinstance(a, MoveAside)]


class DeclutterPlanError(RuntimeError):
    """Raised when no safe temp clear-zone can be found for a blocker."""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _xy(obj) -> Tuple[float, float]:
    return (float(obj.x), float(obj.y))


def find_blockers(
    target,
    objects: Sequence,
    clearance: float = GRIPPER_CLEARANCE_MM,
) -> List:
    """Return the objects (excluding the target itself) whose position lies
    within `clearance` of the target's grasp point (target.x, target.y),
    i.e. close enough that a descending top-down gripper aimed at the
    target would be expected to collide with them.
    """
    tx, ty = _xy(target)
    blockers = []
    for obj in objects:
        if obj is target:
            continue
        ox, oy = _xy(obj)
        if _dist(ox, oy, tx, ty) <= clearance:
            blockers.append(obj)
    return blockers


def is_blocked(
    target,
    objects: Sequence,
    clearance: float = GRIPPER_CLEARANCE_MM,
) -> bool:
    return len(find_blockers(target, objects, clearance)) > 0


def _is_clear(x: float, y: float, occupied: Sequence[Tuple[float, float]], margin: float) -> bool:
    return all(_dist(x, y, ox, oy) >= margin for ox, oy in occupied)


def _find_temp_zone(
    blocker,
    occupied: Sequence[Tuple[float, float]],
    margin: float = CLEAR_ZONE_MARGIN_MM,
    radii: Sequence[float] = _SEARCH_RADII_MM,
    angles_deg: Sequence[float] = _SEARCH_ANGLES_DEG,
) -> Optional[Tuple[float, float]]:
    """Search a ring pattern around `blocker` for a spot that is inside the
    workspace and at least `margin` away from every position in `occupied`.
    Returns the (x, y) of the first hit, or None if nothing in the search
    pattern works.
    """
    bx, by = _xy(blocker)
    for r in radii:
        for deg in angles_deg:
            rad = math.radians(deg)
            cx = bx + r * math.cos(rad)
            cy = by + r * math.sin(rad)
            if not in_workspace(cx, cy):
                continue
            if _is_clear(cx, cy, occupied, margin):
                return (cx, cy)
    return None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def plan_declutter(
    target,
    objects: Sequence,
    clearance: float = GRIPPER_CLEARANCE_MM,
    clear_zone_margin: float = CLEAR_ZONE_MARGIN_MM,
) -> DeclutterPlan:
    """Build a DeclutterPlan for picking `target` out of `objects`.

    `objects` should be the full set of currently-detected objects (it may
    or may not include `target` itself -- if present, it is excluded from
    consideration as its own blocker by identity).

    - No blockers: plan is just [Pick(target)].
    - One or more blockers: plan is [MoveAside(blocker_1, zone_1), ...,
      Pick(target)], blockers ordered closest-to-target first (deterministic
      tie-break on color then position) so the nearest obstruction is
      cleared first.
    - If no safe temp clear-zone can be found for some blocker, raises
      DeclutterPlanError -- callers should treat this as "cannot safely
      declutter this target right now" rather than attempting the grasp.

    Pure function: does not mutate `target`, `objects`, or any of their
    elements. The target's own grasp point (target.x, target.y) is never
    changed -- the plan only ever moves *other* objects.
    """
    blockers = find_blockers(target, objects, clearance)
    if not blockers:
        return DeclutterPlan(actions=[Pick(target)])

    # Every currently-known position is "occupied" and must be avoided when
    # choosing a temp clear-zone: the target, and every detected object
    # (including blockers not yet moved). This list only grows as blockers
    # get relocated (their new temp-zone position is added too), which is a
    # deliberately conservative choice -- we never reuse a spot we've just
    # vacated or one another blocker still occupies.
    occupied: List[Tuple[float, float]] = [_xy(target)]
    for obj in objects:
        if obj is target:
            continue
        occupied.append(_xy(obj))

    tx, ty = _xy(target)
    blockers_sorted = sorted(
        blockers,
        key=lambda o: (
            _dist(o.x, o.y, tx, ty),
            getattr(o, "color", "") or "",
            float(o.x),
            float(o.y),
        ),
    )

    actions: List[DeclutterAction] = []
    for blocker in blockers_sorted:
        # A blocker's own current position obviously can't block its own
        # search (it will vacate it), so exclude it from the occupied set
        # for its own search only.
        own_xy = _xy(blocker)
        search_occupied = [p for p in occupied if p != own_xy]
        zone = _find_temp_zone(blocker, search_occupied, margin=clear_zone_margin)
        if zone is None:
            raise DeclutterPlanError(
                f"no clear temp zone found for blocker at "
                f"({blocker.x:.1f}, {blocker.y:.1f}) while clearing target "
                f"at ({tx:.1f}, {ty:.1f})"
            )
        actions.append(MoveAside(obj=blocker, to_xy=zone))
        occupied.append(zone)

    actions.append(Pick(target))
    return DeclutterPlan(actions=actions)
