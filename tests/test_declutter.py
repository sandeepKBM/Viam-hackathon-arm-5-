"""Offline unit tests for components/declutter.py.

No Viam connection, no robot, no camera -- purely synthetic objects and
plain geometry, exercising the pure planner functions.
"""

from dataclasses import dataclass

import pytest

from components.declutter import (
    CLEAR_ZONE_MARGIN_MM,
    GRIPPER_CLEARANCE_MM,
    DeclutterPlanError,
    MoveAside,
    Pick,
    find_blockers,
    is_blocked,
    plan_declutter,
)
from components.safety import in_workspace


@dataclass(frozen=True)
class FakeObj:
    """Minimal synthetic stand-in for components.shapes.LocatedShape.

    Only exposes what the declutter planner actually reads (x, y, color) so
    these tests don't need to construct a real LocatedShape / DetectedShape
    (and don't pull in cv2/numpy through components.shapes).
    """

    x: float
    y: float
    color: str = "red"
    label: str = "cube"


# A point comfortably inside the taught workspace polygon (verified against
# components.safety.WORKSPACE_CORNERS).
BASE_X, BASE_Y = 200.0, 100.0


def test_unobstructed_target_plan_is_just_pick():
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    # another object far enough away to not be a blocker
    far = FakeObj(BASE_X + 500, BASE_Y + 500, color="red")

    assert not is_blocked(target, [target, far])

    plan = plan_declutter(target, [target, far])
    assert plan.actions == [Pick(target)]
    assert plan.blocked is False
    assert plan.blockers_moved == []


def test_blocker_within_clearance_adds_move_aside_before_pick():
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    # well within GRIPPER_CLEARANCE_MM of the target's grasp point
    blocker = FakeObj(BASE_X + GRIPPER_CLEARANCE_MM * 0.5, BASE_Y, color="red")

    assert is_blocked(target, [target, blocker])
    assert find_blockers(target, [target, blocker]) == [blocker]

    plan = plan_declutter(target, [target, blocker])

    assert plan.blocked is True
    assert len(plan.actions) == 2
    assert isinstance(plan.actions[0], MoveAside)
    assert plan.actions[0].obj is blocker
    assert isinstance(plan.actions[-1], Pick)
    assert plan.actions[-1].obj is target
    assert plan.blockers_moved == [blocker]


def test_chosen_temp_zone_is_in_workspace_and_clear_of_other_objects():
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    blocker = FakeObj(BASE_X + 10, BASE_Y + 5, color="red")
    other = FakeObj(BASE_X - 300, BASE_Y - 100, color="green")

    plan = plan_declutter(target, [target, blocker, other])
    move = plan.actions[0]
    assert isinstance(move, MoveAside)
    zx, zy = move.to_xy

    assert in_workspace(zx, zy)

    def dist(ax, ay, bx, by):
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    # Clear of the target
    assert dist(zx, zy, target.x, target.y) >= CLEAR_ZONE_MARGIN_MM
    # Clear of the unrelated other object
    assert dist(zx, zy, other.x, other.y) >= CLEAR_ZONE_MARGIN_MM
    # Not literally on top of the blocker's own original spot
    assert dist(zx, zy, blocker.x, blocker.y) > 0


def test_target_grasp_point_unchanged_by_planning():
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    blocker = FakeObj(BASE_X + 5, BASE_Y, color="red")
    orig_xy = (target.x, target.y)

    plan = plan_declutter(target, [target, blocker])

    assert (target.x, target.y) == orig_xy
    pick_action = plan.actions[-1]
    assert isinstance(pick_action, Pick)
    assert pick_action.obj is target
    assert (pick_action.obj.x, pick_action.obj.y) == orig_xy


def test_multiple_blockers_all_moved_before_pick_closest_first():
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    near = FakeObj(BASE_X + 5, BASE_Y, color="red")
    mid = FakeObj(BASE_X + 15, BASE_Y, color="red")
    far_but_still_blocking = FakeObj(BASE_X, BASE_Y + GRIPPER_CLEARANCE_MM - 1, color="green")

    objects = [target, near, mid, far_but_still_blocking]
    plan = plan_declutter(target, objects)

    move_actions = [a for a in plan.actions if isinstance(a, MoveAside)]
    assert {a.obj for a in move_actions} == {near, mid, far_but_still_blocking}
    assert isinstance(plan.actions[-1], Pick)
    assert plan.actions[-1].obj is target

    # closest-to-target ordering
    ordered_objs = [a.obj for a in move_actions]
    assert ordered_objs[0] is near
    assert ordered_objs[1] is mid

    # No two blockers were sent to the same temp zone, and each temp zone is
    # clear of the target and of the other (still-unmoved-at-plan-time)
    # objects' original positions.
    zones = [a.to_xy for a in move_actions]
    assert len(set(zones)) == len(zones)
    for zx, zy in zones:
        assert in_workspace(zx, zy)


def test_no_clear_zone_raises_declutter_plan_error():
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    blocker = FakeObj(BASE_X + 5, BASE_Y, color="red")

    # Force every candidate ring point to look "occupied" by using a margin
    # far larger than the workspace itself -- no point can ever be clear.
    with pytest.raises(DeclutterPlanError):
        plan_declutter(target, [target, blocker], clear_zone_margin=1e9)


def test_target_not_required_to_be_in_objects_list():
    # Caller may pass a target that isn't itself present in `objects`.
    target = FakeObj(BASE_X, BASE_Y, color="yellow")
    blocker = FakeObj(BASE_X + 5, BASE_Y, color="red")

    plan = plan_declutter(target, [blocker])
    assert plan.blocked is True
    assert plan.actions[-1].obj is target
