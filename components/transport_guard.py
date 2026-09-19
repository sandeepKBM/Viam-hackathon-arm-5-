"""Transport-collision guard: keep the HELD object clear of the table
during LIFT + CARRY, not just the gripper.

THE PROBLEM
-----------
`components/fast_planner.py` (and `components/declutter.py` before it) only
reasons about the GRIPPER's path: lift straight up to a fixed `TRAVEL_Z`,
translate, descend. That is correct for an EMPTY gripper, or for a target
object that is still sitting on the table. It is wrong for the CARRY leg of
a pick: once an object is grasped, it becomes rigidly attached tool
geometry that hangs BELOW the TCP -- exactly what Viam's motion-service
`world_state` attached-geometry mechanism models for a real motion planner.
A fixed travel height picked only to clear the gripper can still drag the
held object's underside through a neighboring object that is taller than
the gripper's own clearance but shorter than the (gripper + held object)
stack.

This module fixes that with pure geometry -- no sampling planner, no
point-cloud/robot/camera I/O, fully offline-testable with synthetic
`ObjectBox` extents:

  (a) `safe_transport_height` -- lift high enough that the held object's
      BOTTOM clears the tallest remaining obstacle by a margin, given how
      tall the held object itself is.
  (b) `transport_path_clear` -- a swept-box check of the (gripper + held
      object) footprint along the straight XY carry segment, against every
      other object's footprint and height.
  (c) `plan_transport` -- ties both together into a lift/move/lower
      waypoint list that REPLACES a fixed-`TRAVEL_Z` carry with a
      held-object-aware one, raising the transport height further if the
      naive safe height still doesn't clear the corridor, or raising a
      documented failure if no reachable height does.

Everything here is duck-typed / dataclass-based and pure Python + numpy
only (no Viam imports, no `components.constants`/`components.safety`
dependency) so it can be unit tested with plain synthetic boxes and wired
into the live stack by whoever computes real object extents from a point
cloud (see `box_from_points`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

XY = Tuple[float, float]
Pose = Dict[str, float]
Joints = List[float]
IkFn = Callable[[Pose], Sequence[float]]


# ---------------------------------------------------------------------------
# Object extents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectBox:
    """Axis-aligned world-frame extents of one object -- the same shape a
    live pipeline would compute from a segmented point cloud (see
    `box_from_points`), but plain enough to hand-construct in a test.

    `cx`, `cy`: footprint center (mm, arm/world XY frame).
    `half_x`, `half_y`: footprint half-extents (mm). A round object can pass
        equal `half_x == half_y` and be treated as its bounding square --
        conservative, and this module never needs the exact circle.
    `z_bottom`, `z_top`: world Z of the object's bottom/top (mm).
    """

    cx: float
    cy: float
    half_x: float
    half_y: float
    z_bottom: float
    z_top: float

    @property
    def height(self) -> float:
        """Object's own height (top - bottom). This is what makes a TALL
        held object need a higher TCP than a short one to clear the same
        obstacle -- see `safe_transport_height`."""
        return self.z_top - self.z_bottom

    def footprint_radius(self) -> float:
        """Conservative bounding-circle radius of the footprint, used
        wherever a single scalar "how far this object's footprint reaches
        from its center" is needed (e.g. combining with the gripper's half
        width for a swept corridor)."""
        return math.hypot(self.half_x, self.half_y)


def box_from_points(points: Any) -> ObjectBox:
    """Axis-aligned `ObjectBox` from an (N, 3) array of XYZ points (the
    same shape `components.grasp_affordance.classify_grasp` consumes,
    e.g. parsed from one `viam.services.vision.PointCloudObject`'s
    `.point_cloud`). Pure numpy, no Viam/camera dependency -- this is the
    one line that turns a live segmented cloud into the `ObjectBox` this
    module's geometry runs on.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"box_from_points expects a nonempty (N, 3) array, got shape {pts.shape}")
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    cx = (mins[0] + maxs[0]) / 2.0
    cy = (mins[1] + maxs[1]) / 2.0
    half_x = (maxs[0] - mins[0]) / 2.0
    half_y = (maxs[1] - mins[1]) / 2.0
    return ObjectBox(
        cx=float(cx),
        cy=float(cy),
        half_x=float(half_x),
        half_y=float(half_y),
        z_bottom=float(mins[2]),
        z_top=float(maxs[2]),
    )


# ---------------------------------------------------------------------------
# Safe transport height
# ---------------------------------------------------------------------------


def safe_transport_height(
    held: ObjectBox,
    obstacles: Sequence[ObjectBox],
    *,
    grip_z: float,
    margin_mm: float = 20.0,
) -> float:
    """TCP height at which the HELD object's bottom clears the tallest
    obstacle by `margin_mm`.

    FORMULA
    -------
    This module's grasp convention (matching `fast_planner`'s top-down
    `PICK_ORIENTATION`) is that the TCP sits at the held object's TOP when
    it is grasped, so at any TCP height `Z` the object's bottom is at
    `Z - held.height` (`held.height = held.z_top - held.z_bottom`, the
    object's OWN height). For the bottom to clear the tallest obstacle's
    top by `margin_mm`:

        Z - held.height >= max(o.z_top for o in obstacles) + margin_mm
        Z >= max(o.z_top for o in obstacles) + margin_mm + held.height

    So the held object's height enters ADDITIVELY on the required TCP
    height -- a taller held object needs a correspondingly higher TCP to
    put its (now lower, relative to the TCP) bottom above the same
    obstacle. With no obstacles, the only floor is the object's own pick
    height.

    Only the GLOBAL tallest obstacle is used (a safe, conservative
    default -- clearing the tallest obstacle in the scene clears every
    obstacle whether or not it turns out to sit under the transport
    corridor; `transport_path_clear` below does the corridor-aware check
    for callers that want a tighter height).

    Never returns below `grip_z` (the height the object was picked from --
    lifting below where it was already picked can't be "transport", and
    with no obstacles there is nothing to clear anyway).
    """
    tallest_top = max((o.z_top for o in obstacles), default=None)
    if tallest_top is None:
        return grip_z
    required = tallest_top + margin_mm + held.height
    return max(required, grip_z)


# ---------------------------------------------------------------------------
# Swept-path collision check
# ---------------------------------------------------------------------------


def _point_to_aabb_dist_xy(
    x: float, y: float, cx: float, cy: float, half_x: float, half_y: float
) -> float:
    """2D Euclidean distance from point (x, y) to an axis-aligned box
    centered at (cx, cy) with half-extents (half_x, half_y). Zero if the
    point is inside (or on) the box."""
    dx = max(0.0, abs(x - cx) - half_x)
    dy = max(0.0, abs(y - cy) - half_y)
    return math.hypot(dx, dy)


def _segment_to_aabb_dist_xy(
    p0: XY, p1: XY, cx: float, cy: float, half_x: float, half_y: float
) -> float:
    """Minimum 2D distance between the segment p0->p1 and an axis-aligned
    box. `t -> distance(point_on_segment(t), box)` is a convex,
    unimodal function of `t` (distance-to-a-convex-set of a point moving
    along a line is convex), so a simple ternary search over `t in [0, 1]`
    finds the minimum to machine precision without needing a closed-form
    segment/rectangle-distance formula.
    """
    x0, y0 = p0
    x1, y1 = p1
    if x0 == x1 and y0 == y1:
        return _point_to_aabb_dist_xy(x0, y0, cx, cy, half_x, half_y)

    def dist_at(t: float) -> float:
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        return _point_to_aabb_dist_xy(x, y, cx, cy, half_x, half_y)

    lo, hi = 0.0, 1.0
    for _ in range(100):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if dist_at(m1) < dist_at(m2):
            hi = m2
        else:
            lo = m1
    return min(dist_at(lo), dist_at(hi), dist_at(0.0), dist_at(1.0))


def transport_path_clear(
    start_xy: XY,
    dest_xy: XY,
    held: ObjectBox,
    obstacles: Sequence[ObjectBox],
    *,
    transport_z: float,
    tcp_to_object_bottom: float,
    gripper_half_width_mm: float,
    margin_mm: float = 10.0,
) -> Tuple[bool, List[ObjectBox]]:
    """Sweep the (gripper + held object) footprint along the straight XY
    segment `start_xy -> dest_xy` at TCP height `transport_z`, and report
    whether it stays clear of every obstacle.

    An obstacle collides if BOTH hold:

      1. XY: its footprint comes within
         `(swept_half_width + margin_mm)` of the segment, where
         `swept_half_width = max(gripper_half_width_mm, held.half_x,
         held.half_y)` -- the wider of the gripper's own reach and the
         held object's footprint, since a wide held object can stick out
         past the fingers. Computed as a pure 2D segment-to-AABB distance
         (`_segment_to_aabb_dist_xy`) against the obstacle's own box.
      2. Z: the held object's z-band at `transport_z` overlaps (within
         `margin_mm`) the obstacle's height -- i.e. the obstacle's `z_top`
         reaches up to (or into) the held object's underside:
         `obstacle.z_top + margin_mm > transport_z - tcp_to_object_bottom`.
         `tcp_to_object_bottom` is the fixed rigid offset from the TCP
         down to the held object's bottom while it's grasped (pass
         `held.height` for the module's top-down-grasp convention, or an
         explicit value if the grasp geometry differs).

    An obstacle whose footprint is off the corridor (XY test fails) is
    ignored regardless of height -- it is never in the way. An obstacle
    under the corridor whose top sits below the held object's swept
    underside (Z test fails) is also ignored -- short obstacles simply
    pass beneath the held object.

    Returns `(True, [])` if the whole segment is clear, else
    `(False, colliding)` with `colliding` the list of obstacles (in input
    order) that would be hit.
    """
    swept_half_width = max(gripper_half_width_mm, held.half_x, held.half_y)
    held_bottom_at_transport = transport_z - tcp_to_object_bottom

    colliding: List[ObjectBox] = []
    for obstacle in obstacles:
        xy_dist = _segment_to_aabb_dist_xy(
            start_xy, dest_xy, obstacle.cx, obstacle.cy, obstacle.half_x, obstacle.half_y
        )
        xy_conflict = xy_dist < (swept_half_width + margin_mm)
        if not xy_conflict:
            continue
        z_conflict = (obstacle.z_top + margin_mm) > held_bottom_at_transport
        if z_conflict:
            colliding.append(obstacle)

    return (len(colliding) == 0, colliding)


# ---------------------------------------------------------------------------
# Full transport plan: lift -> move -> lower
# ---------------------------------------------------------------------------


class TransportBlockedError(RuntimeError):
    """Raised by `plan_transport` when no reachable transport height (up to
    `max_transport_z`) keeps the swept (gripper + held object) footprint
    clear of the corridor's obstacles. This is the documented "clear
    FAILURE" -- tabletop transport does not attempt to route AROUND an
    obstacle, only OVER it; if going higher can't clear the path (an
    obstacle taller than any sane lift height sitting squarely in the
    corridor), the caller must re-plan the approach/destination rather
    than execute a guessed path.
    """


@dataclass
class TransportWaypoint:
    """One Cartesian waypoint of a transport plan, with its solved joint
    target if an `ik_fn` was supplied to `plan_transport` (mirroring
    `components.fast_planner`'s pose-then-joints style)."""

    pose: Pose
    joints: Optional[Joints] = None


def plan_transport(
    start_xy: XY,
    dest_xy: XY,
    held: ObjectBox,
    obstacles: Sequence[ObjectBox],
    *,
    pick_z: float,
    gripper_half_width_mm: float,
    margin_mm: float = 20.0,
    path_margin_mm: float = 10.0,
    tcp_to_object_bottom: Optional[float] = None,
    max_transport_z: Optional[float] = None,
    max_raise_attempts: int = 25,
    raise_step_mm: float = 25.0,
    ik_fn: Optional[IkFn] = None,
) -> List[TransportWaypoint]:
    """Build the held-object-aware LIFT -> CARRY -> LOWER waypoints for
    moving a grasped object from `start_xy` to `dest_xy`.

    This REPLACES a fixed-`TRAVEL_Z` carry (as used by
    `components.fast_planner._pick_poses`) with one that accounts for the
    object now hanging below the TCP:

      1. Start at `safe_transport_height(held, obstacles, grip_z=pick_z,
         margin_mm=margin_mm)` -- clears the global tallest obstacle by
         construction (see that function's docstring for the formula),
         under `safe_transport_height`'s built-in top-down-grasp
         assumption that the TCP sits at the held object's top, i.e. a
         TCP-to-object-bottom offset of exactly `held.height`.
      2. Verify the straight-line XY carry at that height is actually
         clear with `transport_path_clear`, using the REAL
         `tcp_to_object_bottom` (defaults to `held.height`, matching step
         1's assumption, but a caller can pass the actual measured offset
         -- e.g. the gripper grasped partway down the object rather than
         at its very top, so the object hangs lower below the TCP than
         `held.height` alone implies). When the real offset matches the
         assumption, this step normally just reconfirms what step 1
         already guarantees; when it's larger, this is what catches the
         shortfall and drives step 3.
      3. If blocked, raise the transport height by `raise_step_mm` and
         recheck, up to `max_raise_attempts` times or `max_transport_z`
         (default: `pick_z + raise_step_mm * max_raise_attempts`).
         Raising the height above the tallest colliding obstacle's own
         `z_top` (plus margin and held-object height) always clears it,
         so this loop terminates as soon as the height exceeds every
         obstacle actually near the corridor.
      4. If still blocked at the height ceiling, raise
         `TransportBlockedError` -- documented failure, no attempt to
         route around the obstacle.

    Returns three `TransportWaypoint`s: lift straight up at `start_xy` to
    the transport height, move over to `dest_xy` at that height, descend
    to `pick_z` at `dest_xy`. If `ik_fn` is given, each waypoint's
    `.joints` is populated via `ik_fn(pose)` (a plain synchronous
    callable, e.g. a mock in tests or `components.fast_planner.solve_ik`'s
    underlying `ik_fn`); if `ik_fn` is None, `.joints` stays `None` and
    the caller gets Cartesian poses only.
    """
    if max_transport_z is None:
        max_transport_z = pick_z + raise_step_mm * max_raise_attempts

    z = safe_transport_height(held, obstacles, grip_z=pick_z, margin_mm=margin_mm)
    if tcp_to_object_bottom is None:
        tcp_to_object_bottom = held.height

    attempts = 0
    while True:
        clear, colliding = transport_path_clear(
            start_xy,
            dest_xy,
            held,
            obstacles,
            transport_z=z,
            tcp_to_object_bottom=tcp_to_object_bottom,
            gripper_half_width_mm=gripper_half_width_mm,
            margin_mm=path_margin_mm,
        )
        if clear:
            break
        attempts += 1
        if attempts > max_raise_attempts or z >= max_transport_z:
            raise TransportBlockedError(
                f"plan_transport: no transport height up to {max_transport_z:.1f}mm "
                f"clears the corridor {start_xy} -> {dest_xy}; still colliding with "
                f"{len(colliding)} obstacle(s) at z={z:.1f}mm "
                f"(tallest blocking z_top={max(o.z_top for o in colliding):.1f}mm)"
            )
        # Raising past the tallest currently-colliding obstacle's top (plus
        # margin and the held object's own height) is guaranteed to clear
        # it; step toward that directly instead of a blind fixed increment
        # when it would take more than one step to get there.
        needed = max(o.z_top for o in colliding) + margin_mm + tcp_to_object_bottom
        z = max(z + raise_step_mm, needed)
        z = min(z, max_transport_z)

    sx, sy = start_xy
    dx, dy = dest_xy
    poses: List[Pose] = [
        {"x": float(sx), "y": float(sy), "z": float(z)},  # lift straight up at start
        {"x": float(dx), "y": float(dy), "z": float(z)},  # carry over at transport z
        {"x": float(dx), "y": float(dy), "z": float(pick_z)},  # lower to place
    ]
    if ik_fn is None:
        return [TransportWaypoint(pose=p) for p in poses]
    return [TransportWaypoint(pose=p, joints=list(ik_fn(p))) for p in poses]
