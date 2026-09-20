"""Deterministic, fast pick/transit planner -- replaces per-call BiRRT
sampling-based motion planning for tabletop picks.

THE PROBLEM
-----------
Today, every `arm.move_to_position(x, y, z)` call (used throughout
`components/pickplace.py` and `components/retry.py`) makes the Viam xArm
service run a FRESH sampling-based (RRT-family) motion plan internally --
seconds of latency, and stochastic (the exact path/timing varies run to
run). Bin drops, by contrast, are already fast: `arm.go_to("bin1")` /
`ArmComponent.move_to_joints()` call `move_to_joint_positions` directly,
which is a joint-space interpolated move with NO planning step. This
module makes the pick portion of a task move like the bins already do.

WHY WE DON'T NEED A SAMPLING PLANNER HERE
------------------------------------------
Sampling planners (RRT/BiRRT) exist to solve the hard case: find *some*
collision-free path through a cluttered, unknown-shaped configuration
space. That hard case does not exist for a decluttered tabletop pick:

  1. `components/declutter.py` (`plan_declutter`) already MOVES any
     neighboring object that would block the top-down grasp OUT OF THE
     WAY before the target pick is attempted. By the time this module's
     plan executes, the straight-line path to the target is, by
     construction, clear.
  2. The workspace is a flat, taught tabletop with one dominant obstacle
     class (the table itself, at `components.constants.MIN_Z`) and a
     fixed, generous `TRAVEL_Z` well above every known object. A
     "lift to travel height -> translate at travel height -> descend
     straight down" path (a single inverted-U) cannot intersect the
     table or a decluttered neighbor: the only way to hit something is to
     move at a Z the object could occupy, and we only ever do that
     directly above the target (post-declutter, empty) or above the
     start point (which the arm just vacated).
  3. Orientation is fixed (top-down grasp, `PICK_ORIENTATION`) for the
     whole pick -- there's no wrist-flip search, no basin-of-attraction
     ambiguity to resolve; IK for a fixed-orientation pose directly above
     a taught, bounded workspace has a well-behaved, effectively unique
     "elbow-up" solution.

So the plan only needs three deterministic Cartesian waypoints -- lift,
translate, descend -- each converted to a joint target by a single IK
solve (`ik_fn`, injected so tests can use a mock and the real deployment
wraps Viam's `compute_inverse_kinematics`; see `make_ik_fn` below). No
sampling, no randomness, no per-call search: the SAME inputs always
produce the SAME waypoints, and execution is a straight
`move_to_joint_positions` per waypoint -- exactly the "bins are fast"
recipe, applied to the pick itself.

WHY THIS IS ALSO MORE ROBUST THAN BiRRT HERE, NOT JUST FASTER
---------------------------------------------------------------
A sampling planner's output is not fully reproducible run to run (random
tree growth), so a flaky pick can fail differently each retry and is hard
to reason about after the fact. The deterministic plan below always
produces the same three waypoints for the same (start xy, target xy,
pick_z, travel_z) -- failures are attributable to detection/grasp issues
(which `components/retry.py` already handles), not to planner noise. It
also can't produce a "creative" but undesirable path (e.g. skimming close
to a neighboring object at a weird angle while satisfying its random
tree) -- the path shape is fixed and inspectable ahead of time.

TASK-LEVEL PLANNING (precompute the whole task up front)
----------------------------------------------------------
`plan_task_waypoints` / `plan_task_waypoints_async` extend the same idea
from a single pick to an ENTIRE skill sequence (the ordered
`components.skills.SkillCall` list `components.policy.plan_task` already
produces): every waypoint's IK is solved BEFORE any motion starts, so
execution is a single back-to-back stream of `move_to_joint_positions`
calls with zero planning gaps between skills. This module's pure
"build pose specs -> solve IK -> return joints" split (see
`_task_step_specs` / `_solve_specs` / `_solve_specs_async`) is what makes
that possible: planning has no side effects and doesn't touch the arm, so
it can all run up front, in one pass, offline-testable with a mock
`ik_fn`.

TWO FURTHER LATENCY-HIDING LEVERS (documented here for the record; not
needed for this repo's pick latency today, and not implemented beyond
the small cache below, since a fixed, small, decluttered tabletop scene
does not need them yet)
----------------------------------------------------------------------
  (a) PIPELINING: `plan_task_waypoints_async` uses an async `ik_fn`
      because a real `compute_inverse_kinematics` call is a network RPC.
      A further step (not implemented here) is to overlap solving step
      N+1's waypoints with *executing* step N's motion -- e.g. an
      executor that does
      `next_task = asyncio.create_task(plan_one_step_async(...))` right
      after kicking off `fast_move(arm, current_step.waypoints)`, then
      awaits both -- so IK latency for the next step is hidden behind the
      arm's physical motion time for the current one instead of adding
      up serially. This only pays off once a single pose's real IK solve
      time is comparable to a waypoint's motion time; for this repo's
      pick geometry the whole-task precompute above is already enough.
  (b) IK / ROADMAP CACHE: the workspace here is fixed (same table, same
      `TRAVEL_Z`, same `PICK_ORIENTATION`, same taught bins), so many
      poses recur across picks (every pick's "lift to travel height"
      waypoint sits at the same Z with only XY varying over a bounded
      grid; every place ends at one of two fixed bin poses). `cache_ik_fn`
      below wraps any `ik_fn` with a small rounded-pose memo dict so a
      repeated pose becomes an O(1) lookup instead of a fresh solve --
      the minimal version of an "IK/roadmap cache" for this fixed scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from components.constants import (
    BIN_1_POSE,
    BIN_2_POSE,
    COLOR_BINS,
    HOME_POSE,
    MIN_Z,
    PICK_ORIENTATION,
    TAUGHT_JOINTS,
    TRAVEL_Z,
)
from components.safety import assert_in_workspace, clamp_z

XY = Tuple[float, float]
Pose = Dict[str, float]
Joints = List[float]
IkFn = Callable[[Pose], Sequence[float]]
AsyncIkFn = Callable[[Pose], Awaitable[Sequence[float]]]

# Cartesian (x, y) of every taught joint pose this module needs to chain
# a multi-step task plan through -- built locally from the existing pose
# constants in components.constants (no edits to that module needed).
_TAUGHT_XY: Dict[str, XY] = {
    "home": (float(HOME_POSE["x"]), float(HOME_POSE["y"])),
    "bin1": (float(BIN_1_POSE["x"]), float(BIN_1_POSE["y"])),
    "bin2": (float(BIN_2_POSE["x"]), float(BIN_2_POSE["y"])),
}


# ---------------------------------------------------------------------------
# Pose / IK plumbing
# ---------------------------------------------------------------------------


def pick_pose(
    x: float,
    y: float,
    z: float,
    orientation: Optional[Dict[str, float]] = None,
    *,
    check_workspace: bool = True,
) -> Pose:
    """Build the pose dict handed to `ik_fn`. Applies the same safety
    checks the rest of the codebase applies before any real move
    (`components.safety.assert_in_workspace` / `clamp_z` against
    `components.constants.MIN_Z`) so a bad waypoint is rejected here, at
    plan time, rather than only discovered when it would have driven the
    arm through the table.

    `check_workspace=False` skips the in-workspace check for a pose that
    describes where the arm ALREADY is (e.g. the "lift straight up at
    current XY" waypoint's starting XY) rather than a new location being
    commanded -- the arm's current position (which may legitimately be a
    taught pose outside the pick/place tabletop polygon, e.g. a bin or
    home) was already valid when it got there; only NEW XY targets this
    planner asks the arm to move to (the target/destination waypoints)
    need the workspace check.
    """
    orientation = dict(PICK_ORIENTATION if orientation is None else orientation)
    if check_workspace:
        assert_in_workspace(x, y)
    z = clamp_z(float(z))
    pose = {"x": float(x), "y": float(y), "z": z}
    pose.update(orientation)
    return pose


def solve_ik(ik_fn: IkFn, pose: Pose) -> Joints:
    return list(ik_fn(pose))


async def solve_ik_async(ik_fn: AsyncIkFn, pose: Pose) -> Joints:
    return list(await ik_fn(pose))


def make_ik_fn(arm: Any) -> AsyncIkFn:
    """Build a real `ik_fn` by wrapping Viam's `compute_inverse_kinematics`.

    Returns an ASYNC callable (a real IK solve is a network RPC through
    the Viam SDK, unlike the synchronous mock `ik_fn` used in tests /
    offline planning) -- pair it with `plan_pick_waypoints_async` /
    `plan_task_waypoints_async`, not the synchronous `plan_pick_waypoints`.

    Looks for `compute_inverse_kinematics` on `arm` directly, then on
    `arm._arm` (the raw `viam.components.arm.Arm` client
    `components.arm.ArmComponent` wraps), so this works whether `arm` is
    an `ArmComponent` or a raw Viam arm client. As of writing, the
    `viam` Python SDK pinned in this repo's `.venv` does not expose
    `compute_inverse_kinematics` on the `Arm` component (only
    `get_kinematics`); rather than silently returning nonsense, the
    returned callable raises a clear `RuntimeError` the first time it's
    called against a build without it, naming exactly what's missing.
    Offline/tests should inject a mock `ik_fn` instead of this wrapper.
    """

    async def _ik_fn(pose: Pose) -> Joints:
        compute = getattr(arm, "compute_inverse_kinematics", None)
        if compute is None:
            raw = getattr(arm, "_arm", None)
            compute = (
                getattr(raw, "compute_inverse_kinematics", None)
                if raw is not None
                else None
            )
        if compute is None:
            raise RuntimeError(
                "fast_planner.make_ik_fn: no compute_inverse_kinematics found on "
                f"{arm!r} (checked the object itself and .{'_arm'!s}); this Viam "
                "SDK/robot build doesn't expose IK yet. Inject a mock ik_fn for "
                "offline use, or a callable wrapping whatever IK entry point your "
                "SDK version provides, instead of make_ik_fn(arm)."
            )
        result = await compute(pose)
        return list(result)

    return _ik_fn


def make_local_ik_fn(urdf_path: Optional[str] = None, **solver_kwargs: Any) -> IkFn:
    """Build a real, OFFLINE-usable `ik_fn` via `components.ik`'s local
    numerical IK solver (ikpy, against the bundled xArm5 URDF) -- the fast
    planner's actual IK source now that `make_ik_fn(arm)` above is known to
    raise on this repo's `viam-sdk` (no `compute_inverse_kinematics`).

    Unlike `make_ik_fn`, this is SYNCHRONOUS (no network RPC involved -- the
    whole solve happens locally against the URDF's kinematic model), so it
    is exactly the `IkFn` shape `plan_pick_waypoints`/`plan_task_waypoints`
    (and `components.retry.RetryController(ik_fn=...)` /
    `components.pickplace.PickPlace(ik_fn=..., use_fast_planner=True)`,
    which both call the sync `plan_pick_waypoints`/`solve_ik`, not the
    `_async` variants) already expect -- pass this straight into any of
    them. See `components.ik.make_local_ik_fn`'s own docstring for the URDF
    / tolerance / seeding details. `ikpy` is imported lazily inside
    `components.ik`, so importing this module doesn't require it -- only
    calling `make_local_ik_fn` does.
    """
    from components.ik import make_local_ik_fn as _make_local_ik_fn

    return _make_local_ik_fn(urdf_path, **solver_kwargs)


def cache_ik_fn(ik_fn: IkFn, precision_mm: float = 0.5) -> IkFn:
    """Wrap a synchronous `ik_fn` with a small memo cache keyed by a
    rounded pose -- the "IK/roadmap cache" lever described in the module
    docstring. Poses are rounded to `precision_mm` before hashing, so
    near-identical repeated poses (e.g. the same bin-lift waypoint,
    computed for every pick) hit the cache instead of re-solving.
    Deterministic and pure -- safe to use in the offline planner path.
    """
    cache: Dict[Tuple[float, ...], Joints] = {}

    def _key(pose: Pose) -> Tuple[float, ...]:
        return tuple(
            round(float(pose[k]) / precision_mm) * precision_mm
            for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")
        )

    def _cached(pose: Pose) -> Joints:
        key = _key(pose)
        hit = cache.get(key)
        if hit is not None:
            return list(hit)
        joints = list(ik_fn(pose))
        cache[key] = joints
        return list(joints)

    return _cached


# ---------------------------------------------------------------------------
# Single-pick planning: up -> over -> down
# ---------------------------------------------------------------------------


def _pick_poses(
    start_xy: XY,
    target_xy: XY,
    pick_z: float,
    travel_z: float,
    orientation: Optional[Dict[str, float]],
    *,
    held: Optional[Any] = None,
    transport_obstacles: Optional[Sequence[Any]] = None,
    transport_margin_mm: float = 20.0,
) -> List[Pose]:
    if pick_z < MIN_Z:
        raise ValueError(f"pick_z={pick_z:.2f} is below the Z-floor MIN_Z={MIN_Z:.2f}")
    sx, sy = start_xy
    tx, ty = target_xy

    effective_travel_z = travel_z
    if held is not None:
        # Opt-in transport-collision guard (components/transport_guard.py):
        # when the caller identifies a HELD object (an `ObjectBox`, or
        # anything with the same `.height`/`z_top` shape) and the other
        # objects on the table, raise the travel height so the held
        # object's BOTTOM clears the tallest obstacle, instead of only
        # clearing the (empty) gripper at a fixed `travel_z`. Lazily
        # imported so this module has no hard dependency on
        # transport_guard/numpy when the hook isn't used, and `held=None`
        # (the default everywhere below) leaves behavior byte-for-byte
        # unchanged.
        from components.transport_guard import safe_transport_height

        effective_travel_z = max(
            travel_z,
            safe_transport_height(
                held,
                transport_obstacles or [],
                grip_z=pick_z,
                margin_mm=transport_margin_mm,
            ),
        )

    return [
        # (a) lift straight up at current XY -- the arm is already there
        # (it may be a taught pose outside the pick/place polygon, e.g. a
        # bin or home; see pick_pose's check_workspace docstring), so this
        # is not a new location being commanded.
        pick_pose(sx, sy, effective_travel_z, orientation, check_workspace=False),
        pick_pose(tx, ty, effective_travel_z, orientation),  # (b) move over target
        pick_pose(tx, ty, pick_z, orientation),  # (c) descend straight down to pick_z
    ]


def plan_pick_waypoints(
    start_joints: Sequence[float],
    start_xy: XY,
    target_xy: XY,
    ik_fn: IkFn,
    *,
    pick_z: float = MIN_Z,
    travel_z: float = TRAVEL_Z,
    orientation: Optional[Dict[str, float]] = None,
    held: Optional[Any] = None,
    transport_obstacles: Optional[Sequence[Any]] = None,
    transport_margin_mm: float = 20.0,
) -> List[Joints]:
    """Produce the deterministic 3-waypoint joint-space plan for one pick:
    (a) lift straight up to `travel_z` at the arm's CURRENT xy
        (`start_xy`),
    (b) move over the target at `travel_z`,
    (c) descend straight down to `pick_z`.

    Each waypoint's joint config comes from `ik_fn(pose) -> joints`
    (injected -- tests pass a mock, `make_ik_fn(arm)` wraps the real
    Viam IK for deployment). Pure and deterministic: same inputs always
    produce the same three joint targets, no sampling/search involved.

    `start_joints` is accepted (rather than only `start_xy`) so a caller
    -- or a smarter `ik_fn` -- can use it as an IK seed hint for solution
    continuity (many IK solvers prefer the branch nearest a seed config);
    this planner's own waypoint geometry only needs `start_xy` (the
    arm's current Cartesian position), since `start_joints` alone can't
    be turned into an (x, y) without a forward-kinematics model this
    module deliberately doesn't carry. It is not required to be accurate
    for the plan's correctness -- pass `list(start_joints)` unchanged
    from `arm.get_joint_positions()` if available, or `[]` otherwise.

    OPT-IN TRANSPORT-COLLISION GUARD: `held`/`transport_obstacles` default
    to `None`, which leaves this function's behavior exactly as before
    (the fixed `travel_z` is used verbatim). When a caller is planning a
    leg where the gripper is ALREADY holding an object -- e.g. this
    `target_xy` is really a drop-off/relocation destination, not a fresh
    pick -- pass `held` (a `components.transport_guard.ObjectBox`
    describing the held object) and `transport_obstacles` (the other
    objects still on the table); the travel height used for waypoints (a)
    and (b) is then `max(travel_z, safe_transport_height(held,
    transport_obstacles, grip_z=pick_z, margin_mm=transport_margin_mm))`
    (see `components.transport_guard.safe_transport_height`), so the held
    object's own height is accounted for on top of the usual travel
    clearance. `pick_z` here is the height the object was picked from
    (used as `safe_transport_height`'s `grip_z`), not a new descent depth.
    This module does not itself call
    `components.transport_guard.transport_path_clear` -- for a corridor-
    specific check (rather than the global-tallest-obstacle default this
    hook applies), call `components.transport_guard.plan_transport`
    directly instead of this function.

    Raises `ValueError` if `pick_z` is below `components.constants.MIN_Z`
    (the Z-floor) or if `start_xy`/`target_xy` fall outside the taught
    workspace (via `components.safety.assert_in_workspace`) -- checked
    here, at PLAN time, before anything is sent to the arm.
    """
    del start_joints  # see docstring: not needed for this planner's own geometry
    poses = _pick_poses(
        start_xy,
        target_xy,
        pick_z,
        travel_z,
        orientation,
        held=held,
        transport_obstacles=transport_obstacles,
        transport_margin_mm=transport_margin_mm,
    )
    return [solve_ik(ik_fn, p) for p in poses]


async def plan_pick_waypoints_async(
    start_joints: Sequence[float],
    start_xy: XY,
    target_xy: XY,
    ik_fn: AsyncIkFn,
    *,
    pick_z: float = MIN_Z,
    travel_z: float = TRAVEL_Z,
    orientation: Optional[Dict[str, float]] = None,
    held: Optional[Any] = None,
    transport_obstacles: Optional[Sequence[Any]] = None,
    transport_margin_mm: float = 20.0,
) -> List[Joints]:
    """Same as `plan_pick_waypoints`, for an ASYNC `ik_fn` (e.g.
    `make_ik_fn(arm)`, wrapping a real network IK call). See
    `plan_pick_waypoints`'s docstring for the opt-in `held`/
    `transport_obstacles` transport-collision guard."""
    del start_joints
    poses = _pick_poses(
        start_xy,
        target_xy,
        pick_z,
        travel_z,
        orientation,
        held=held,
        transport_obstacles=transport_obstacles,
        transport_margin_mm=transport_margin_mm,
    )
    return [await solve_ik_async(ik_fn, p) for p in poses]


# ---------------------------------------------------------------------------
# Executor: direct joint moves, no planning
# ---------------------------------------------------------------------------


async def fast_move(
    arm: Any, waypoints: Sequence[Sequence[float]], *, timeout: float = 30
) -> None:
    """Run `waypoints` in order via direct joint-space interpolation --
    NO planning per call, exactly like the already-fast taught-bin moves.

    Duck-types `arm`: prefers a raw `move_to_joint_positions(joints,
    timeout=...)` (the Viam SDK primitive itself, and what the offline
    tests mock), and falls back to `move_to_joints(joints, timeout=...)`
    (`components.arm.ArmComponent`'s existing wrapper, which itself calls
    `move_to_joint_positions` under the hood -- see components/arm.py) so
    this works unmodified against `PickPlace.arm` in the real stack
    without needing any change to components/arm.py. Never calls
    `move_to_position` -- that is the whole point (no RRT plan per call).
    """
    move_fn = getattr(arm, "move_to_joint_positions", None)
    if move_fn is None:
        move_fn = getattr(arm, "move_to_joints", None)
    if move_fn is None:
        raise AttributeError(
            "fast_move: arm exposes neither move_to_joint_positions nor "
            "move_to_joints -- need a direct joint-move primitive (no planning)"
        )
    for waypoint in waypoints:
        await move_fn(list(waypoint), timeout=timeout)


# ---------------------------------------------------------------------------
# Whole-task planning: precompute every waypoint before any motion
# ---------------------------------------------------------------------------

# A pending waypoint is either an IK solve to do ("ik", pose) or an
# already-known joint target that needs no IK ("joints", joints) -- e.g.
# a taught bin drop, exactly as fast (and exactly as planning-free) today.
_WaypointSpec = Tuple[str, Any]


@dataclass
class TaskStep:
    """One skill call's precomputed joint-waypoint group, in task order."""

    skill: str
    waypoints: List[Joints] = field(default_factory=list)
    end_xy: XY = (0.0, 0.0)


def _bin_name_for(skill: str, params: Any) -> str:
    if skill == "place":
        return params.bin
    color = getattr(params.object, "color", None)
    bin_name = COLOR_BINS.get(color)
    if bin_name is None:
        raise ValueError(f"no bin mapped for color {color!r}")
    return bin_name


def _pick_place_specs(
    cur_xy: XY,
    target_xy: XY,
    bin_name: str,
    pick_z: float,
    travel_z: float,
    orientation: Optional[Dict[str, float]],
) -> Tuple[List[_WaypointSpec], XY]:
    specs: List[_WaypointSpec] = [
        ("ik", p) for p in _pick_poses(cur_xy, target_xy, pick_z, travel_z, orientation)
    ]
    tx, ty = target_xy
    specs.append(("ik", pick_pose(tx, ty, travel_z, orientation)))  # lift after grab
    bin_joints = TAUGHT_JOINTS.get(bin_name)
    if bin_joints is None:
        raise ValueError(f"unknown taught bin {bin_name!r}")
    specs.append(("joints", list(bin_joints)))  # already-taught drop, no IK needed
    end_xy = _TAUGHT_XY.get(bin_name, target_xy)
    return specs, end_xy


def _move_aside_specs(
    cur_xy: XY,
    src_xy: XY,
    dst_xy: XY,
    pick_z: float,
    travel_z: float,
    orientation: Optional[Dict[str, float]],
) -> Tuple[List[_WaypointSpec], XY]:
    specs: List[_WaypointSpec] = [
        ("ik", p) for p in _pick_poses(cur_xy, src_xy, pick_z, travel_z, orientation)
    ]
    sx, sy = src_xy
    specs.append(("ik", pick_pose(sx, sy, travel_z, orientation)))  # lift blocker
    specs += [("ik", p) for p in _pick_poses(src_xy, dst_xy, pick_z, travel_z, orientation)]
    dx, dy = dst_xy
    specs.append(("ik", pick_pose(dx, dy, travel_z, orientation)))  # lift after release
    return specs, dst_xy


def _task_step_specs(
    skill_sequence: Sequence[Any],
    start_xy: XY,
    *,
    pick_z: float,
    travel_z: float,
    orientation: Optional[Dict[str, float]],
) -> List[Tuple[str, List[_WaypointSpec], XY]]:
    """Pure planning pass: turn a `components.skills.SkillCall` sequence
    into an ordered list of (skill_name, waypoint_specs, end_xy), with NO
    IK solved yet and NO arm I/O. Threads the arm's expected Cartesian
    position (`cur_xy`) through the whole task so each step's "lift at
    current xy" waypoint is correct even though the arm never actually
    moves during planning.
    """
    from components.declutter import MoveAside, Pick, plan_declutter

    steps: List[Tuple[str, List[_WaypointSpec], XY]] = []
    cur_xy = start_xy
    for call in skill_sequence:
        skill = getattr(call, "skill", None)
        params = getattr(call, "params", None)

        if skill in ("pick", "place"):
            target_xy = (float(params.object.x), float(params.object.y))
            bin_name = _bin_name_for(skill, params)
            specs, end_xy = _pick_place_specs(
                cur_xy, target_xy, bin_name, pick_z, travel_z, orientation
            )
            steps.append((skill, specs, end_xy))
            cur_xy = end_xy

        elif skill == "move_aside":
            src_xy = (float(params.object.x), float(params.object.y))
            dst_xy = (float(params.to_xy[0]), float(params.to_xy[1]))
            specs, end_xy = _move_aside_specs(
                cur_xy, src_xy, dst_xy, pick_z, travel_z, orientation
            )
            steps.append((skill, specs, end_xy))
            cur_xy = end_xy

        elif skill == "declutter":
            plan = plan_declutter(params.target, params.all_objects)
            for action in plan.actions:
                if isinstance(action, MoveAside):
                    src_xy = (float(action.obj.x), float(action.obj.y))
                    dst_xy = (float(action.to_xy[0]), float(action.to_xy[1]))
                    specs, end_xy = _move_aside_specs(
                        cur_xy, src_xy, dst_xy, pick_z, travel_z, orientation
                    )
                    steps.append(("declutter:move_aside", specs, end_xy))
                    cur_xy = end_xy
                elif isinstance(action, Pick):
                    target_xy = (float(action.obj.x), float(action.obj.y))
                    bin_name = COLOR_BINS.get(getattr(action.obj, "color", None))
                    if bin_name is None:
                        raise ValueError(
                            f"no bin mapped for color {getattr(action.obj, 'color', None)!r}"
                        )
                    specs, end_xy = _pick_place_specs(
                        cur_xy, target_xy, bin_name, pick_z, travel_z, orientation
                    )
                    steps.append(("declutter:pick", specs, end_xy))
                    cur_xy = end_xy

        else:
            raise NotImplementedError(
                f"fast_planner: skill {skill!r} has no fast-path waypoint plan"
            )

    return steps


def plan_task_waypoints(
    skill_sequence: Sequence[Any],
    start_joints: Sequence[float],
    start_xy: XY,
    ik_fn: IkFn,
    *,
    pick_z: float = MIN_Z,
    travel_z: float = TRAVEL_Z,
    orientation: Optional[Dict[str, float]] = None,
) -> List[TaskStep]:
    """PRECOMPUTE the full joint-waypoint list for an entire task (every
    skill call `components.policy.plan_task` produced -- picks, places,
    move_asides, declutters) up front: every `ik_fn` call happens here,
    before the caller executes a single motion. The returned `TaskStep`
    list can then be streamed straight through `fast_move_task` (or
    flattened with `flatten_task_waypoints`) with zero planning in the
    execution loop -- back-to-back `move_to_joint_positions` calls only.

    `skill_sequence` items are duck-typed as `components.skills.SkillCall`
    (`.skill: str`, `.params: <that skill's params dataclass>`) -- pass
    the list `components.policy.plan_task(...)` returns. Bin drops within
    "pick"/"place"/"declutter" steps use `components.constants.TAUGHT_JOINTS`
    directly (no IK call at all), matching the fact bin drops are
    already planning-free today. `declutter` steps are expanded via
    `components.declutter.plan_declutter` into their concrete
    move-aside(s) + final pick, exactly like `PickPlace.pick_with_declutter`
    does at execution time -- but here it's all resolved to joints ahead
    of time, offline.

    `start_joints`/`start_xy` are the arm's actual current joints/xy
    before the task starts (see `plan_pick_waypoints` for why both are
    accepted). Raises `NotImplementedError` for any skill this module
    has no fast-path geometry for (currently only "descend_until_contact",
    itself unimplemented in `components/skills.py`), and the same
    `ValueError`s as `plan_pick_waypoints` for an out-of-workspace or
    below-floor waypoint -- both raised during planning, before any
    motion.
    """
    step_specs = _task_step_specs(
        skill_sequence, start_xy, pick_z=pick_z, travel_z=travel_z, orientation=orientation
    )
    steps: List[TaskStep] = []
    for skill, specs, end_xy in step_specs:
        joints = [
            solve_ik(ik_fn, payload) if kind == "ik" else list(payload)
            for kind, payload in specs
        ]
        steps.append(TaskStep(skill=skill, waypoints=joints, end_xy=end_xy))
    return steps


async def plan_task_waypoints_async(
    skill_sequence: Sequence[Any],
    start_joints: Sequence[float],
    start_xy: XY,
    ik_fn: AsyncIkFn,
    *,
    pick_z: float = MIN_Z,
    travel_z: float = TRAVEL_Z,
    orientation: Optional[Dict[str, float]] = None,
) -> List[TaskStep]:
    """Same as `plan_task_waypoints`, for an ASYNC `ik_fn` (e.g.
    `make_ik_fn(arm)`)."""
    step_specs = _task_step_specs(
        skill_sequence, start_xy, pick_z=pick_z, travel_z=travel_z, orientation=orientation
    )
    steps: List[TaskStep] = []
    for skill, specs, end_xy in step_specs:
        joints = []
        for kind, payload in specs:
            if kind == "ik":
                joints.append(await solve_ik_async(ik_fn, payload))
            else:
                joints.append(list(payload))
        steps.append(TaskStep(skill=skill, waypoints=joints, end_xy=end_xy))
    return steps


def flatten_task_waypoints(steps: Sequence[TaskStep]) -> List[Joints]:
    """Flatten a `plan_task_waypoints(...)` result into one ordered joint
    -waypoint list, for a caller that just wants to stream everything
    through `fast_move` without caring about per-skill grouping."""
    flat: List[Joints] = []
    for step in steps:
        flat.extend(step.waypoints)
    return flat


async def fast_move_task(
    arm: Any, steps: Sequence[TaskStep], *, timeout: float = 30
) -> None:
    """Execute an already-fully-planned task (see `plan_task_waypoints`):
    streams every step's waypoints through `fast_move`, in order, with no
    planning in this loop -- all IK was solved before this function was
    ever called."""
    await fast_move(arm, flatten_task_waypoints(steps), timeout=timeout)
