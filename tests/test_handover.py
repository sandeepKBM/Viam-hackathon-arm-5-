"""Offline tests for the hand-over skill + reactive hold (components/handover.py).

Exercises HandoverSkill (the inheritable base), its functional wrapper, and the
reactive re-grab -- all with mocks, no robot / no audio hardware.
"""

import asyncio

from components.audio import ScriptedAudio
from components.handover import HandoverSkill, handover, ensure_holding


class FakeArm:
    def __init__(self):
        self.joint_moves = []

    async def move_to_joints(self, joints, **kw):
        self.joint_moves.append(list(joints))


class FakeGripper:
    """`holding_seq` = what is_holding_something() returns on successive calls.
    `recover_on_grab` = whether a grab() restores the hold (a real re-grab that
    catches the object) or not (object is genuinely gone)."""

    def __init__(self, holding_seq=None, recover_on_grab=True):
        self._seq = list(holding_seq or [True])
        self._i = 0
        self._recover = recover_on_grab
        self.opens = 0
        self.grabs = 0

    async def open(self, **kw):
        self.opens += 1

    async def grab(self, **kw):
        self.grabs += 1
        if self._recover:
            self._seq = self._seq[: self._i] + [True] * (len(self._seq) - self._i)
        return self._recover

    async def is_holding_something(self, **kw):
        val = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return val


def test_handover_releases_on_yes():
    arm, grip = FakeArm(), FakeGripper(holding_seq=[True, True, True])
    audio = ScriptedAudio(answers=[True])
    ok = asyncio.run(HandoverSkill(arm, grip, audio).run("red block"))
    assert ok is True
    assert grip.opens == 1                       # dropped into the hand
    assert arm.joint_moves                        # moved to the marked pose
    assert any("ready" in q.lower() for q in audio.asked)


def test_handover_holds_on_no():
    arm, grip = FakeArm(), FakeGripper(holding_seq=[True, True, True])
    audio = ScriptedAudio(answers=[False])
    ok = asyncio.run(handover(arm, grip, audio, object_label="cup"))  # functional wrapper
    assert ok is False
    assert grip.opens == 0                        # did NOT release


def test_handover_aborts_if_not_holding():
    # object is gone and a re-grab can't recover it -> abort, never release
    arm, grip = FakeArm(), FakeGripper(holding_seq=[False], recover_on_grab=False)
    audio = ScriptedAudio(answers=[True])
    ok = asyncio.run(HandoverSkill(arm, grip, audio).run("pen"))
    assert ok is False
    assert grip.opens == 0


def test_reactive_hold_regrabs_on_slip():
    grip = FakeGripper(holding_seq=[False, True], recover_on_grab=True)
    held = asyncio.run(ensure_holding(grip, retries=2))
    assert held is True
    assert grip.grabs >= 1                         # reacted by re-grabbing


def test_reactive_hold_unknown_status_assumes_held():
    class NoStatusGripper:
        async def grab(self, **kw):
            return True
    assert asyncio.run(ensure_holding(NoStatusGripper(), retries=1)) is True


def test_subclass_can_override_hooks():
    """A downstream agent (e.g. a fetch skill) inherits and overrides hooks."""
    events = []

    class VisionHandover(HandoverSkill):
        async def present_pose(self):
            events.append("aim_at_detected_hand")   # override: aim at a hand

        async def confirm(self, object_label):
            events.append("wait_for_hand_ready")     # override: vision trigger
            return True

    arm, grip = FakeArm(), FakeGripper(holding_seq=[True, True, True])
    ok = asyncio.run(VisionHandover(arm, grip).run("can"))
    assert ok is True and events == ["aim_at_detected_hand", "wait_for_hand_ready"]
    assert grip.opens == 1
