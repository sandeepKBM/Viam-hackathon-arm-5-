"""Offline unit tests for components/policy.py (+ components/skills.py).

No Viam connection, no robot, no camera, no network/LLM access -- purely
synthetic scene objects and the default rule-based stub planner (or a
hand-written fake planner, to exercise rejection paths).
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from components.declutter import GRIPPER_CLEARANCE_MM
from components.policy import plan_task, rule_based_stub_planner
from components.safety import in_workspace
from components.skills import (
    SKILL_REGISTRY,
    MoveAsideParams,
    PickParams,
    PlaceParams,
    SkillValidationError,
    make_skill_call,
    skill_menu,
)


@dataclass(frozen=True)
class FakeObj:
    """Minimal synthetic stand-in for components.shapes.LocatedShape --
    only exposes the attributes the skill registry / policy actually read,
    same approach as tests/test_declutter.py's FakeObj."""

    x: float
    y: float
    color: str = "red"
    label: str = "cube"
    canonical_label: str = "red block"
    z: float = 0.0
    difficulty: Optional[float] = None
    score: Optional[float] = None
    history: Optional[dict] = None


# A point comfortably inside the taught workspace polygon (same anchor used
# by tests/test_declutter.py, verified against components.safety.WORKSPACE_CORNERS).
BASE_X, BASE_Y = 200.0, 100.0

# A point far outside the taught workspace polygon.
OUTSIDE_X, OUTSIDE_Y = 100_000.0, 100_000.0


def _scene(*objs):
    return list(objs)


# ---------------------------------------------------------------------------
# Skill menu / registry sanity
# ---------------------------------------------------------------------------


def test_skill_menu_lists_all_five_skills_with_docs():
    names = {s["name"] for s in skill_menu()}
    assert names == {"pick", "place", "move_aside", "declutter", "descend_until_contact"}
    for spec in skill_menu():
        assert spec["doc"], f"{spec['name']} missing a planner-visible docstring"
        assert spec["params"]


# ---------------------------------------------------------------------------
# Per-skill safety validation (in_workspace / Z-floor)
# ---------------------------------------------------------------------------


def test_make_skill_call_rejects_out_of_workspace_pick():
    obj = FakeObj(OUTSIDE_X, OUTSIDE_Y, color="red")
    with pytest.raises(SkillValidationError):
        make_skill_call("pick", object=obj)


def test_make_skill_call_accepts_in_workspace_pick():
    obj = FakeObj(BASE_X, BASE_Y, color="red")
    call = make_skill_call("pick", object=obj)
    assert call.skill == "pick"
    assert isinstance(call.params, PickParams)
    assert call.params.object is obj


def test_move_aside_rejects_out_of_workspace_destination():
    obj = FakeObj(BASE_X, BASE_Y, color="red")
    with pytest.raises(SkillValidationError):
        make_skill_call("move_aside", object=obj, to_xy=(OUTSIDE_X, OUTSIDE_Y))


def test_move_aside_accepts_in_workspace_destination():
    obj = FakeObj(BASE_X, BASE_Y, color="red")
    call = make_skill_call("move_aside", object=obj, to_xy=(BASE_X + 80.0, BASE_Y + 80.0))
    assert isinstance(call.params, MoveAsideParams)
    assert in_workspace(*call.params.to_xy)


def test_place_rejects_non_default_bin():
    obj = FakeObj(BASE_X, BASE_Y, color="red")  # default bin is bin1
    with pytest.raises(SkillValidationError):
        make_skill_call("place", object=obj, bin="bin2")


def test_place_accepts_default_bin():
    obj = FakeObj(BASE_X, BASE_Y, color="red")
    call = make_skill_call("place", object=obj, bin="bin1")
    assert isinstance(call.params, PlaceParams)


def test_unknown_skill_name_rejected():
    with pytest.raises(SkillValidationError):
        make_skill_call("teleport", object=FakeObj(BASE_X, BASE_Y))


def test_bad_params_type_rejected():
    with pytest.raises(SkillValidationError):
        make_skill_call("pick", nonexistent_field=123)


def test_descend_until_contact_validated_but_not_executable():
    call = make_skill_call("descend_until_contact", x=BASE_X, y=BASE_Y)
    assert call.skill == "descend_until_contact"
    with pytest.raises(SkillValidationError):
        make_skill_call("descend_until_contact", x=OUTSIDE_X, y=OUTSIDE_Y)


# ---------------------------------------------------------------------------
# plan_task with the default (offline) rule-based stub planner
# ---------------------------------------------------------------------------


def test_stub_planner_produces_valid_plan_for_sort_goal():
    scene = _scene(
        FakeObj(BASE_X, BASE_Y, color="red", label="cube", difficulty=0.2),
        FakeObj(BASE_X + 150, BASE_Y + 100, color="yellow", label="cube", difficulty=0.8),
        FakeObj(BASE_X - 100, BASE_Y + 150, color="green", label="cube", difficulty=0.1),
    )
    plan = plan_task("sort the red and yellow blocks", scene)

    assert len(plan) == 2  # only red + yellow named -> green excluded
    for call in plan:
        assert call.skill in SKILL_REGISTRY
    colors_touched = {
        getattr(call.params.object if hasattr(call.params, "object") else call.params.target, "color")
        for call in plan
    }
    assert colors_touched == {"red", "yellow"}


def test_stub_planner_orders_easy_difficulty_first():
    easy = FakeObj(BASE_X, BASE_Y, color="red", difficulty=0.9)
    mid = FakeObj(BASE_X + 200, BASE_Y + 50, color="red", difficulty=0.5)
    hard = FakeObj(BASE_X - 50, BASE_Y - 200, color="red", difficulty=0.1)
    # deliberately scrambled input order
    scene = _scene(easy, mid, hard)

    plan = plan_task("sort the red blocks", scene)

    assert len(plan) == 3
    assert all(call.skill == "pick" for call in plan)
    difficulties = [call.params.object.difficulty for call in plan]
    assert difficulties == sorted(difficulties)
    assert difficulties == [0.1, 0.5, 0.9]


def test_stub_planner_uses_declutter_when_blocked():
    target = FakeObj(BASE_X, BASE_Y, color="red", difficulty=0.3)
    # well within GRIPPER_CLEARANCE_MM -> blocks the target's grasp
    blocker = FakeObj(BASE_X + GRIPPER_CLEARANCE_MM * 0.5, BASE_Y, color="green")
    scene = _scene(target, blocker)

    plan = plan_task("sort the red blocks", scene)

    assert len(plan) == 1
    call = plan[0]
    assert call.skill == "declutter"
    assert call.params.target is target
    assert blocker in call.params.all_objects


def test_stub_planner_clear_a_path_goal_targets_named_object():
    target = FakeObj(BASE_X, BASE_Y, color="red", label="cube", canonical_label="red cube")
    blocker = FakeObj(BASE_X + GRIPPER_CLEARANCE_MM * 0.5, BASE_Y, color="green", label="triangle")
    other = FakeObj(BASE_X - 400, BASE_Y - 400, color="yellow", label="cube")
    scene = _scene(target, blocker, other)

    plan = plan_task("clear a path to the red cube", scene)

    assert len(plan) == 1
    assert plan[0].skill == "declutter"
    assert plan[0].params.target is target


def test_every_emitted_call_is_in_the_skill_menu():
    scene = _scene(
        FakeObj(BASE_X, BASE_Y, color="red", difficulty=0.4),
        FakeObj(BASE_X + 250, BASE_Y + 50, color="yellow", difficulty=0.6),
    )
    plan = plan_task("sort the red and yellow blocks", scene)
    menu_names = {s["name"] for s in skill_menu()}
    assert plan  # sanity: non-empty
    assert all(call.skill in menu_names for call in plan)


def test_empty_scene_produces_empty_plan():
    assert plan_task("sort the red and yellow blocks", []) == []


# ---------------------------------------------------------------------------
# Rejection / repair: out-of-menu and unsafe planner output never survives
# plan_task's validation, even from a (fake, injected) misbehaving planner.
# ---------------------------------------------------------------------------


def test_out_of_menu_call_is_rejected_not_raised():
    scene = _scene(FakeObj(BASE_X, BASE_Y, color="red"))

    def rogue_planner(goal, scene_view, menu):
        return [{"skill": "teleport", "params": {"object_id": "obj0"}}]

    plan = plan_task("sort the red blocks", scene, planner=rogue_planner)
    assert plan == []  # rejected, not raised, and not silently accepted


def test_out_of_workspace_call_is_rejected():
    scene = _scene(FakeObj(BASE_X, BASE_Y, color="red"))

    def rogue_planner(goal, scene_view, menu):
        return [
            {
                "skill": "move_aside",
                "params": {"object_id": "obj0", "to_xy": [OUTSIDE_X, OUTSIDE_Y]},
            }
        ]

    plan = plan_task("sort the red blocks", scene, planner=rogue_planner)
    assert plan == []


def test_unknown_object_id_is_rejected():
    scene = _scene(FakeObj(BASE_X, BASE_Y, color="red"))

    def rogue_planner(goal, scene_view, menu):
        return [{"skill": "pick", "params": {"object_id": "not-a-real-id"}}]

    plan = plan_task("sort the red blocks", scene, planner=rogue_planner)
    assert plan == []


def test_valid_and_invalid_calls_from_same_planner_partially_survive():
    good = FakeObj(BASE_X, BASE_Y, color="red")
    scene = _scene(good)

    def mixed_planner(goal, scene_view, menu):
        return [
            {"skill": "pick", "params": {"object_id": "obj0"}},  # valid
            {"skill": "not_a_skill", "params": {}},  # invalid: out of menu
        ]

    plan = plan_task("sort the red blocks", scene, planner=mixed_planner)
    assert len(plan) == 1
    assert plan[0].skill == "pick"
    assert plan[0].params.object is good


def test_planner_is_injectable_and_receives_menu_and_scene_view():
    seen = {}

    def spy_planner(goal, scene_view, menu):
        seen["goal"] = goal
        seen["scene_view"] = scene_view
        seen["menu"] = menu
        return []

    scene = _scene(FakeObj(BASE_X, BASE_Y, color="red", difficulty=0.5, score=0.9))
    plan_task("sort the red blocks", scene, planner=spy_planner)

    assert seen["goal"] == "sort the red blocks"
    assert seen["scene_view"][0]["object_id"] == "obj0"
    assert seen["scene_view"][0]["color"] == "red"
    assert seen["scene_view"][0]["difficulty"] == 0.5
    assert {m["name"] for m in seen["menu"]} == set(SKILL_REGISTRY)


def test_default_planner_is_the_rule_based_stub():
    # plan_task with no planner kwarg behaves identically to passing the
    # stub explicitly -- confirms the offline default really is used.
    scene = _scene(FakeObj(BASE_X, BASE_Y, color="red", difficulty=0.5))
    a = plan_task("sort the red blocks", scene)
    b = plan_task("sort the red blocks", scene, planner=rule_based_stub_planner)
    assert [c.skill for c in a] == [c.skill for c in b]
