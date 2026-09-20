"""Offline unit tests for components/fast_planner.py.

No Viam connection, no robot, no camera: a mock `ik_fn` (pose dict ->
joints) and a mock arm exercise the deterministic waypoint geometry and
the direct-joint-move executor. Mirrors the dependency-injection test
style already used elsewhere (components/declutter.py, components/retry.py).
"""

import asyncio
from dataclasses import dataclass, field
from typing import List

import pytest

from components.constants import MIN_Z, TRAVEL_Z
from components.declutter import GRIPPER_CLEARANCE_MM
from components.fast_planner import (
    TaskStep,
    cache_ik_fn,
    fast_move,
    fast_move_task,
    flatten_task_waypoints,
    make_ik_fn,
    pick_pose,
    plan_pick_waypoints,
    plan_task_waypoints,
    solve_ik,
)
from components.skills import (
    DeclutterParams,
    MoveAsideParams,
    PickParams,
    PlaceParams,
    SkillCall,
)

BASE_X, BASE_Y = 200.0, 100.0
OTHER_X, OTHER_Y = 300.0, -50.0


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeObj:
    x: float
    y: float
    color: str = "red"
    label: str = "cube"


def mock_ik_fn(pose):
    """Deterministic fake IK: joints just encode the pose so tests can
    assert on waypoint order/content without a real kinematics model."""
    return [pose["x"], pose["y"], pose["z"]]


class RecordingArm:
    """Raw-Viam-style mock: exposes move_to_joint_positions only (no
    move_to_position at all), so any accidental planned-move call would
    raise AttributeError instead of silently succeeding."""

    def __init__(self):
        self.joint_moves: List[list] = []

    async def move_to_joint_positions(self, joints, timeout=30):
        self.joint_moves.append(list(joints))


class ArmComponentStyleArm:
    """Mimics components.arm.ArmComponent's public surface: move_to_joints
    (not move_to_joint_positions) plus a leftover move_to_position, to
    confirm fast_move prefers the joint-space primitive and never touches
    the planned one."""

    def __init__(self):
        self.joint_moves: List[list] = []
        self.cartesian_moves: List[tuple] = []

    async def move_to_joints(self, joints, timeout=60):
        self.joint_moves.append(list(joints))

    async def move_to_position(self, x, y, z, **kw):
        self.cartesian_moves.append((x, y, z))


def async_test(fn):
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# plan_pick_waypoints: up -> over -> down, deterministic, no sampling
# ---------------------------------------------------------------------------


def test_plan_pick_waypoints_order_and_count():
    start_xy = (BASE_X, BASE_Y)
    target_xy = (OTHER_X, OTHER_Y)
    waypoints = plan_pick_waypoints([], start_xy, target_xy, mock_ik_fn)

    assert len(waypoints) == 3
    up, over, down = waypoints
    # (a) lift straight up to travel_z at current XY
    assert up == [start_xy[0], start_xy[1], TRAVEL_Z]
    # (b) move over target at travel_z
    assert over == [target_xy[0], target_xy[1], TRAVEL_Z]
    # (c) descend to pick_z
    assert down == [target_xy[0], target_xy[1], MIN_Z]


def test_plan_pick_waypoints_is_pure_and_deterministic():
    start_xy = (BASE_X, BASE_Y)
    target_xy = (OTHER_X, OTHER_Y)
    a = plan_pick_waypoints([], start_xy, target_xy, mock_ik_fn)
    b = plan_pick_waypoints([], start_xy, target_xy, mock_ik_fn)
    assert a == b  # same inputs -> same output, every time (no sampling)


def test_plan_pick_waypoints_no_move_when_start_equals_target_xy():
    xy = (BASE_X, BASE_Y)
    waypoints = plan_pick_waypoints([], xy, xy, mock_ik_fn)
    up, over, down = waypoints
    assert up[:2] == over[:2] == down[:2] == [xy[0], xy[1]]


# ---------------------------------------------------------------------------
# Safety: descend never below the Z-floor
# ---------------------------------------------------------------------------


def test_pick_z_below_floor_is_rejected():
    with pytest.raises(ValueError):
        plan_pick_waypoints(
            [], (BASE_X, BASE_Y), (OTHER_X, OTHER_Y), mock_ik_fn, pick_z=MIN_Z - 5.0
        )


def test_pick_z_at_floor_is_allowed():
    waypoints = plan_pick_waypoints(
        [], (BASE_X, BASE_Y), (OTHER_X, OTHER_Y), mock_ik_fn, pick_z=MIN_Z
    )
    assert waypoints[-1][-1] == MIN_Z


def test_pick_pose_clamps_at_the_z_floor():
    # Even a caller that tries to sneak a lower z through pick_pose directly
    # gets clamped, not silently allowed through to the arm.
    pose = pick_pose(BASE_X, BASE_Y, MIN_Z - 50.0)
    assert pose["z"] == MIN_Z


def test_out_of_workspace_xy_is_rejected():
    with pytest.raises(ValueError):
        plan_pick_waypoints([], (BASE_X, BASE_Y), (1e6, 1e6), mock_ik_fn)


# ---------------------------------------------------------------------------
# fast_move: only move_to_joint_positions, no planning, no move_to_position
# ---------------------------------------------------------------------------


@async_test
async def test_fast_move_calls_only_move_to_joint_positions():
    arm = RecordingArm()
    waypoints = plan_pick_waypoints([], (BASE_X, BASE_Y), (OTHER_X, OTHER_Y), mock_ik_fn)
    await fast_move(arm, waypoints)
    assert arm.joint_moves == waypoints
    assert not hasattr(arm, "move_to_position")


@async_test
async def test_fast_move_prefers_move_to_joints_and_never_calls_move_to_position():
    arm = ArmComponentStyleArm()
    waypoints = plan_pick_waypoints([], (BASE_X, BASE_Y), (OTHER_X, OTHER_Y), mock_ik_fn)
    await fast_move(arm, waypoints)
    assert arm.joint_moves == waypoints
    assert arm.cartesian_moves == []  # move_to_position was never called


@async_test
async def test_fast_move_matches_segment_count():
    arm = RecordingArm()
    waypoints = plan_pick_waypoints([], (BASE_X, BASE_Y), (OTHER_X, OTHER_Y), mock_ik_fn)
    await fast_move(arm, waypoints)
    assert len(arm.joint_moves) == len(waypoints) == 3


@async_test
async def test_fast_move_raises_clearly_with_no_joint_primitive():
    class NoJointArm:
        async def move_to_position(self, x, y, z, **kw):
            pass

    with pytest.raises(AttributeError):
        await fast_move(NoJointArm(), [[0, 0, 0]])


# ---------------------------------------------------------------------------
# cache_ik_fn: repeated poses become lookups, not solves
# ---------------------------------------------------------------------------


def test_cache_ik_fn_avoids_resolving_repeated_poses():
    calls = {"count": 0}

    def counting_ik_fn(pose):
        calls["count"] += 1
        return [pose["x"], pose["y"], pose["z"]]

    cached = cache_ik_fn(counting_ik_fn)
    pose = pick_pose(BASE_X, BASE_Y, TRAVEL_Z)
    first = solve_ik(cached, pose)
    second = solve_ik(cached, pose)
    third = solve_ik(cached, pick_pose(BASE_X, BASE_Y, TRAVEL_Z))  # same pose, new dict

    assert first == second == third
    assert calls["count"] == 1


def test_cache_ik_fn_still_solves_distinct_poses():
    calls = {"count": 0}

    def counting_ik_fn(pose):
        calls["count"] += 1
        return [pose["x"], pose["y"], pose["z"]]

    cached = cache_ik_fn(counting_ik_fn)
    solve_ik(cached, pick_pose(BASE_X, BASE_Y, TRAVEL_Z))
    solve_ik(cached, pick_pose(OTHER_X, OTHER_Y, TRAVEL_Z))
    assert calls["count"] == 2


# ---------------------------------------------------------------------------
# make_ik_fn: clear degrade when no real IK is available (offline)
# ---------------------------------------------------------------------------


@async_test
async def test_make_ik_fn_degrades_clearly_without_compute_inverse_kinematics():
    class ArmWithNoIk:
        pass

    ik_fn = make_ik_fn(ArmWithNoIk())
    with pytest.raises(RuntimeError):
        await ik_fn(pick_pose(BASE_X, BASE_Y, TRAVEL_Z))


@async_test
async def test_make_ik_fn_wraps_a_present_compute_inverse_kinematics():
    class ArmWithIk:
        async def compute_inverse_kinematics(self, pose):
            return [pose["x"], pose["y"], pose["z"]]

    ik_fn = make_ik_fn(ArmWithIk())
    joints = await ik_fn(pick_pose(BASE_X, BASE_Y, TRAVEL_Z))
    assert joints == [BASE_X, BASE_Y, TRAVEL_Z]


# ---------------------------------------------------------------------------
# plan_task_waypoints: precompute an ENTIRE task's IK before any motion
# ---------------------------------------------------------------------------


def _skill_seq():
    red = FakeObj(BASE_X, BASE_Y, color="red")
    yellow = FakeObj(OTHER_X, OTHER_Y, color="yellow")
    return [
        SkillCall(skill="pick", params=PickParams(object=red)),
        SkillCall(skill="pick", params=PickParams(object=yellow)),
    ]


def test_plan_task_waypoints_precomputes_every_ik_call_up_front():
    events: List[str] = []

    def recording_ik_fn(pose):
        events.append("ik")
        return [pose["x"], pose["y"], pose["z"]]

    steps = plan_task_waypoints(
        _skill_seq(), [], (BASE_X, BASE_Y), recording_ik_fn
    )

    # 2 pick skills * 4 IK poses each (up, over, down, lift-after-grab) = 8;
    # the bin drops themselves need NO IK (taught joints), matching the
    # fact bin drops are already planning-free today.
    assert len(events) == 8
    assert all(e == "ik" for e in events)

    # Now separately execute -- this must add NO further ik calls, proving
    # planning (all IK) already fully happened before any "motion".
    arm = RecordingArm()

    async def run():
        await fast_move_task(arm, steps)

    asyncio.run(run())
    assert len(events) == 8  # unchanged: execution did not solve any IK
    assert len(arm.joint_moves) == len(flatten_task_waypoints(steps))


def test_plan_task_waypoints_all_ik_before_first_move():
    """Directly asserts ordering: every ik_fn call happens strictly before
    the first move_to_joint_positions call, by sharing one event log
    across both the mock ik_fn and the mock arm."""
    events: List[str] = []

    def recording_ik_fn(pose):
        events.append("ik")
        return [pose["x"], pose["y"], pose["z"]]

    class LoggingArm:
        async def move_to_joint_positions(self, joints, timeout=30):
            events.append("move")

    steps = plan_task_waypoints(_skill_seq(), [], (BASE_X, BASE_Y), recording_ik_fn)
    # All planning happened above; nothing has moved yet.
    assert "move" not in events
    ik_count = events.count("ik")
    assert ik_count > 0

    asyncio.run(fast_move_task(LoggingArm(), steps))

    first_move_index = events.index("move")
    assert all(e == "ik" for e in events[:first_move_index])
    assert events[:ik_count] == ["ik"] * ik_count


def test_plan_task_waypoints_includes_taught_bin_joints_without_ik():
    from components.constants import BIN_1_JOINTS

    events: List[str] = []

    def recording_ik_fn(pose):
        events.append(pose)
        return [pose["x"], pose["y"], pose["z"]]

    red_only = [SkillCall(skill="pick", params=PickParams(object=FakeObj(BASE_X, BASE_Y, color="red")))]
    steps = plan_task_waypoints(red_only, [], (BASE_X, BASE_Y), recording_ik_fn)
    assert len(steps) == 1
    step = steps[0]
    assert isinstance(step, TaskStep)
    # up, over, down, lift-after-grab (4 IK'd waypoints) + taught bin drop.
    assert len(step.waypoints) == 5
    assert step.waypoints[-1] == list(BIN_1_JOINTS)
    assert len(events) == 4  # the bin waypoint never called ik_fn


def test_plan_task_waypoints_move_aside_then_pick():
    obj = FakeObj(BASE_X, BASE_Y, color="red")
    seq = [
        SkillCall(
            skill="move_aside",
            params=MoveAsideParams(object=obj, to_xy=(BASE_X + 80.0, BASE_Y + 80.0)),
        )
    ]
    steps = plan_task_waypoints(seq, [], (BASE_X, BASE_Y), mock_ik_fn)
    assert len(steps) == 1
    assert steps[0].skill == "move_aside"
    # up/over/down to the blocker, lift, up/over/down to the drop zone, lift.
    assert len(steps[0].waypoints) == 8
    assert steps[0].end_xy == (BASE_X + 80.0, BASE_Y + 80.0)


def test_plan_task_waypoints_expands_declutter_via_plan_declutter():
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    blocker = FakeObj(BASE_X + GRIPPER_CLEARANCE_MM * 0.5, BASE_Y, color="red")
    seq = [
        SkillCall(
            skill="declutter",
            params=DeclutterParams(target=target, all_objects=[target, blocker]),
        )
    ]
    steps = plan_task_waypoints(seq, [], (BASE_X, BASE_Y), mock_ik_fn)
    skills = [s.skill for s in steps]
    assert skills == ["declutter:move_aside", "declutter:pick"]


def test_plan_task_waypoints_unknown_skill_raises_clearly():
    class Weird:
        skill = "descend_until_contact"
        params = object()

    with pytest.raises(NotImplementedError):
        plan_task_waypoints([Weird()], [], (BASE_X, BASE_Y), mock_ik_fn)


def test_flatten_task_waypoints_matches_manual_concat():
    steps = plan_task_waypoints(_skill_seq(), [], (BASE_X, BASE_Y), mock_ik_fn)
    manual = []
    for s in steps:
        manual.extend(s.waypoints)
    assert flatten_task_waypoints(steps) == manual
