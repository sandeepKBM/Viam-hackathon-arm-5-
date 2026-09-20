"""Offline tests for deliver-to-person (components/assistant.py).

Exercises deliver_to_person composing components.retry.RetryController (the
same grasp state machine components.pipeline._run_pick uses) with
components.handover.HandoverSkill (unmodified) -- all with mocks, no robot,
no audio hardware.
"""

import asyncio

from components.assistant import deliver_to_person
from components.audio import ScriptedAudio
from components.experience_store import ExperienceStore
from components.pipeline import TaskContext
from components.retry import RetryController
from components.shapes import LocatedShape


class FakeArm:
    def __init__(self):
        self.moves = []
        self.joint_moves = []
        self.homed = 0

    async def move_to_position(self, x, y, z, **kw):
        self.moves.append((x, y, z))

    async def move_to_joints(self, joints, **kw):
        self.joint_moves.append(list(joints))

    async def go_home(self, **kw):
        self.homed += 1


class FakeGripper:
    def __init__(self, grab_result=True):
        self.grab_result = grab_result
        self.opens = 0
        self.grabs = 0

    async def open(self, **kw):
        self.opens += 1

    async def grab(self, **kw):
        self.grabs += 1
        return self.grab_result

    async def is_holding_something(self, **kw):
        return self.grab_result


def _target():
    return LocatedShape(
        label="red block", x=200.0, y=100.0, z=179.8, color="red",
        canonical_label="red block",
    )


def _ctx(tmp_path, grab_result=True):
    arm = FakeArm()
    gripper = FakeGripper(grab_result=grab_result)
    store = ExperienceStore(path=str(tmp_path / "experience.json"))
    retry = RetryController(arm, gripper)
    ctx = TaskContext(arm=arm, gripper=gripper, pickplace=None, store=store, retry=retry)
    return ctx, arm, gripper, store


def test_deliver_grasps_then_releases_on_yes(tmp_path):
    ctx, arm, gripper, store = _ctx(tmp_path, grab_result=True)
    audio = ScriptedAudio(answers=[True])

    result = asyncio.run(deliver_to_person(_target(), ctx, audio=audio))

    assert result.skill == "deliver"
    assert result.success is True
    # Grasped via the retry path (gripper.grab/open called), then handed
    # over (a final open() to release into the person's hand -- 2 opens:
    # the pre-grasp open + the hand-over release).
    assert gripper.grabs >= 1
    assert gripper.opens >= 2
    assert any("ready" in q.lower() for q in audio.asked)
    assert store.get_history("red block") is not None
    assert store.get_history("red block")["attempts"][-1]["placement_success"] is True


def test_deliver_grasps_but_holds_on_no(tmp_path):
    ctx, arm, gripper, store = _ctx(tmp_path, grab_result=True)
    audio = ScriptedAudio(answers=[False])

    result = asyncio.run(deliver_to_person(_target(), ctx, audio=audio))

    assert result.success is False
    assert result.failure_type == "handover_declined_or_lost"
    # Grasped, but never released -- opens == 1 is just the pre-grasp open.
    assert gripper.opens == 1
    hist = store.get_history("red block")
    assert hist["attempts"][-1]["grasp_success"] is True
    assert hist["attempts"][-1]["placement_success"] is False


def test_deliver_never_hands_over_on_failed_grasp(tmp_path):
    ctx, arm, gripper, store = _ctx(tmp_path, grab_result=False)
    audio = ScriptedAudio(answers=[True])

    result = asyncio.run(deliver_to_person(_target(), ctx, audio=audio))

    assert result.success is False
    # HandoverSkill was never invoked -- no "are you ready" question asked.
    assert audio.asked == []
    hist = store.get_history("red block")
    assert hist["attempts"][-1]["grasp_success"] is False
    assert hist["attempts"][-1]["placement_success"] is None


def test_deliver_goes_home_after(tmp_path):
    ctx, arm, gripper, store = _ctx(tmp_path, grab_result=True)
    audio = ScriptedAudio(answers=[True])
    asyncio.run(deliver_to_person(_target(), ctx, audio=audio))
    assert arm.homed == 1
