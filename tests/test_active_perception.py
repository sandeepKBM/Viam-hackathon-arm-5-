"""Offline tests for the UQ-triggered active-perception re-look
(components/active_perception.py): a mock arm (records move_to_position
calls) + a mock vision that returns an ambiguous low-confidence detection at
the default pose and a clear high-confidence one from a zoomed-in pose. No
robot, no camera, no models -- pure duck-typed mocks.
"""

import asyncio

from components import uq
from components.active_perception import refine_uncertain, zoom_pose, pose_is_safe
from components.constants import MIN_Z, TRAVEL_Z
from components.shapes import LocatedShape


class MockArm:
    """Records every move_to_position call; nothing else needed offline."""

    def __init__(self):
        self.moves = []

    async def move_to_position(self, x, y, z, **kw):
        self.moves.append((x, y, z))


class ZoomAwareVision:
    """Mock vision: `locate_shapes()` returns a CLEAR, high-confidence
    detection near (target_x, target_y) once the arm has actually moved
    close to it (mimicking the real "bigger/centered pixels -> better
    detection" effect a zoom-in produces); otherwise (arm hasn't zoomed
    there, or this object isn't the one under test) it returns nothing, so
    only a genuine re-look for the right object adopts a new reading.
    """

    def __init__(self, arm, target_x, target_y, label="mug", score=0.92):
        self.arm = arm
        self.target_x = target_x
        self.target_y = target_y
        self.label = label
        self.score = score
        self.calls = 0

    async def locate_shapes(self):
        self.calls += 1
        if not self.arm.moves:
            return []
        x, y, z = self.arm.moves[-1]
        near = abs(x - self.target_x) < 1.0 and abs(y - self.target_y) < 1.0
        zoomed_in = z < TRAVEL_Z  # closer than the normal/farther capture height
        if not (near and zoomed_in):
            return []
        return [
            LocatedShape(label=self.label, x=self.target_x, y=self.target_y, z=MIN_Z, score=self.score)
        ]


def _ambiguous_object(x=250.0, y=50.0, label="cup", score=0.2):
    """An uncertain reading: low score alone (via uq.enrich, no detector_fn/
    image) already pushes difficulty well above the default 0.5 threshold."""
    obj = LocatedShape(label=label, x=x, y=y, z=MIN_Z, score=score)
    uq.enrich([obj])
    return obj


def _confident_object(x=300.0, y=-100.0, label="red block", score=0.97):
    obj = LocatedShape(label=label, x=x, y=y, z=MIN_Z, score=score)
    uq.enrich([obj])
    return obj


def test_zoom_pose_centers_on_object_and_is_closer_than_travel():
    obj = _ambiguous_object()
    x, y, z = zoom_pose(obj)
    assert x == obj.x and y == obj.y  # centered directly over the object
    assert MIN_Z <= z < TRAVEL_Z  # a genuine zoom-IN: closer, never farther


def test_uncertain_object_triggers_move_and_adopts_more_confident_reading():
    obj = _ambiguous_object(x=250.0, y=50.0, label="cup", score=0.2)
    assert obj.difficulty >= 0.5  # sanity: this is the "uncertain" case

    arm = MockArm()
    vision = ZoomAwareVision(arm, target_x=obj.x, target_y=obj.y, label="mug", score=0.92)

    scene = [obj]
    before_difficulty = obj.difficulty
    result = asyncio.run(refine_uncertain(scene, arm, vision))

    assert result is scene
    assert len(arm.moves) == 1  # exactly one re-look move
    mx, my, mz = arm.moves[0]
    assert mx == obj.x and my == obj.y  # centered
    assert MIN_Z <= mz < TRAVEL_Z  # closer than the normal capture height

    # More-confident zoom reading adopted: difficulty dropped...
    assert obj.difficulty < before_difficulty
    # ...and the label stabilized onto the shared canonical vocabulary even
    # though the VLM named it differently across views (cup vs mug).
    assert obj.canonical_label == "cup"
    assert obj.score == 0.92


def test_confident_object_triggers_no_move():
    obj = _confident_object()
    assert obj.difficulty < 0.5  # sanity: this is the "confident" case

    arm = MockArm()
    vision = ZoomAwareVision(arm, target_x=obj.x, target_y=obj.y)

    before = (obj.label, obj.x, obj.y, obj.score, obj.difficulty)
    asyncio.run(refine_uncertain([obj], arm, vision))

    assert arm.moves == []  # never moved for a confident object
    assert (obj.label, obj.x, obj.y, obj.score, obj.difficulty) == before


def test_unsafe_zoom_pose_is_skipped_without_moving():
    # Well outside the taught workspace polygon (see components/constants.py)
    # -> in_workspace(x, y) is False -> pose_is_safe must reject it.
    obj = _ambiguous_object(x=5000.0, y=5000.0, label="can", score=0.2)
    assert not pose_is_safe(*zoom_pose(obj))

    arm = MockArm()
    vision = ZoomAwareVision(arm, target_x=obj.x, target_y=obj.y)

    before = (obj.label, obj.x, obj.y, obj.difficulty)
    asyncio.run(refine_uncertain([obj], arm, vision))

    assert arm.moves == []  # unsafe -> never moved
    assert (obj.label, obj.x, obj.y, obj.difficulty) == before  # untouched


def test_max_relooks_caps_number_of_relooks():
    objs = [
        _ambiguous_object(x=200.0 + 10 * i, y=100.0, label="soda can", score=0.1 + 0.01 * i)
        for i in range(5)
    ]
    arm = MockArm()

    class NullVision:
        async def locate_shapes(self):
            return []

    asyncio.run(refine_uncertain(objs, arm, NullVision(), max_relooks=2))

    assert len(arm.moves) == 2  # capped, even though all 5 are uncertain
