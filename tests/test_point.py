"""Offline tests for the point/locate skill (components/point.py).

Exercises PointSkill (aim without grasping), the natural-request -> target
matcher, and the workspace safety check -- all with mocks, no robot.
"""

import asyncio

import pytest

from components.audio import ScriptedAudio
from components.constants import PICK_ORIENTATION, TRAVEL_Z
from components.point import PointSkill, describe_position, point_at, resolve_pointing_target
from components.shapes import LocatedShape


class FakeArm:
    def __init__(self):
        self.position_moves = []

    async def move_to_position(self, x, y, z, **kw):
        self.position_moves.append((x, y, z, kw))


def _scene():
    return [
        LocatedShape(label="red block", x=200.0, y=100.0, z=179.8, color="red"),
        LocatedShape(label="yellow block", x=300.0, y=-50.0, z=179.8, color="yellow"),
    ]


def test_point_at_moves_above_target_without_grasping():
    arm = FakeArm()
    target = _scene()[0]
    asyncio.run(PointSkill(arm).point_at(target))

    assert len(arm.position_moves) == 1
    x, y, z, kw = arm.position_moves[0]
    assert (x, y) == (target.x, target.y)
    # Raised well above ordinary travel height -- an aim/hover, not a descent.
    assert z > TRAVEL_Z
    # Position control only: same downward-aimed orientation as a pick hover.
    for k, v in PICK_ORIENTATION.items():
        assert kw[k] == v
    # PointSkill has no gripper at all -- it is structurally incapable of
    # grasping.
    assert not hasattr(arm, "grab")


def test_point_at_rejects_out_of_workspace_target():
    arm = FakeArm()
    target = LocatedShape(label="red block", x=10000.0, y=10000.0, z=179.8, color="red")
    with pytest.raises(ValueError):
        asyncio.run(PointSkill(arm).point_at(target))
    assert arm.position_moves == []  # rejected before any move was issued


def test_point_at_announces_position_via_audio():
    arm = FakeArm()
    audio = ScriptedAudio()
    target = _scene()[0]
    asyncio.run(PointSkill(arm, audio).point_at(target))
    assert audio.said  # something was spoken
    assert "red block" in audio.said[0].lower()


def test_functional_point_at_wrapper():
    arm = FakeArm()
    target = _scene()[1]
    asyncio.run(point_at(arm, target))
    assert len(arm.position_moves) == 1


def test_resolve_pointing_target_matches_color_and_noun():
    scene = _scene()
    match = resolve_pointing_target("where is the red block", scene)
    assert match is scene[0]

    match2 = resolve_pointing_target("bring me the yellow block", scene)
    assert match2 is scene[1]


def test_resolve_pointing_target_no_match_returns_none():
    scene = _scene()
    assert resolve_pointing_target("where is the green triangle", scene) is None
    assert resolve_pointing_target("anything", []) is None


def test_describe_position_is_deterministic_string():
    # Pure function, no robot state -- just a sanity check it returns text
    # for both sides of the workspace centroid.
    left = describe_position(200.0, 600.0)
    right = describe_position(200.0, -600.0)
    assert isinstance(left, str) and isinstance(right, str)
    assert left != right
