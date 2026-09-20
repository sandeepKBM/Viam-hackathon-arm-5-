"""Hand-over skill (assistant): give a held object to a person.

Designed to be INHERITED and built on. A `fetch` skill (which may live in
another system) composes with this as:

    fetch(label)  =  detect(label)  ->  pick(object)  ->  HandoverSkill.run(label)

`HandoverSkill` orchestrates the hand-over as a sequence of small overridable
HOOKS, so a downstream agent extends behaviour by subclassing and overriding a
hook -- not by rewriting the flow:

    ensure_holding()  reactive hold: re-grab if the grip status says we slipped
    present_pose()    bring the object into the person's reach   [OVERRIDE]
    confirm()         ask + get the go-ahead to release          [OVERRIDE]
    release()         let go                                     [OVERRIDE]

Extension points (where real implementations plug in):
  * confirm()      -> real audio: components.audio.AudioIO subclass wired to
                      Viam `audio_out` (TTS says the question) + `audio_in`
                      (STT hears "yes"). Or a button / vision hand-detector.
  * present_pose() -> compute a pose from a detected person/hand instead of the
                      fixed, marked HANDOVER_JOINTS.
  * release()      -> "tug to release": read arm joint torque/current (via the
                      xarm_velocity module's get_joint_torques) and open when the
                      person pulls; or a compliant open. (Stock Viam has no
                      continuous grip-force loop -- that needs a custom gripper
                      module; see components/audio.py + the xarm_velocity module.)

Everything is dependency-injected (arm, gripper, audio) and async, so it runs
offline with mocks. No robot, no audio hardware required to exercise the logic.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from components.audio import AudioIO
from components.constants import HOME_JOINTS


def _env_joints(name: str, default: List[float]) -> List[float]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    try:
        return [float(v) for v in raw.split(",")]
    except ValueError:
        return list(default)


# MARKED hand-over pose: where the gripper presents the object to the person.
# TODO: TEACH this on the real robot -- a comfortable pose in the person's
# reach, gripper facing them. Placeholder = HOME_JOINTS (known-valid); override
# via env HANDOVER_JOINTS="j1,j2,j3,j4,j5".
HANDOVER_JOINTS: List[float] = _env_joints("HANDOVER_JOINTS", HOME_JOINTS)


class HandoverSkill:
    """Present a held object to a person and release it only on confirmation.

    Subclass and override `present_pose` / `confirm` / `release` to build on
    this without touching the `run()` orchestration."""

    def __init__(
        self,
        arm: Any,
        gripper: Any,
        audio: Optional[AudioIO] = None,
        *,
        handover_joints: Optional[List[float]] = None,
        regrab_retries: int = 2,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.audio = audio or AudioIO()
        self.handover_joints = (
            handover_joints if handover_joints is not None else HANDOVER_JOINTS
        )
        self.regrab_retries = regrab_retries

    # -- reactive hold ----------------------------------------------------
    async def _grip_status(self) -> Optional[bool]:
        """True/False from the gripper's grip status, or None if it can't
        report (handles a plain bool or Viam's HoldingStatus)."""
        fn = getattr(self.gripper, "is_holding_something", None)
        if not callable(fn):
            return None
        try:
            res = await fn()
        except Exception:
            return None
        return bool(getattr(res, "is_holding_something", res))

    async def ensure_holding(self) -> bool:
        """Reactive hold: if the grip status says we lost the object, RE-GRAB,
        up to `regrab_retries` times. Unknown status -> assume held (don't
        thrash)."""
        for _ in range(self.regrab_retries + 1):
            status = await self._grip_status()
            if status is None or status:
                return True
            await self.gripper.grab()
        status = await self._grip_status()
        return True if status is None else bool(status)

    # -- overridable hooks ------------------------------------------------
    async def present_pose(self) -> None:
        """Bring the object into the person's reach. OVERRIDE to aim at a
        detected hand instead of the fixed marked pose."""
        await self.arm.move_to_joints(self.handover_joints)

    async def confirm(self, object_label: str) -> bool:
        """Ask the person and return their yes/no. OVERRIDE for real
        audio_in->STT, a button, or a vision hand-ready trigger."""
        return self.audio.ask_yes_no(
            f"I have the {object_label}. Are you ready for me to drop it into your hand?"
        )

    async def release(self) -> None:
        """Let go of the object. OVERRIDE for tug-to-release / compliant open."""
        await self.gripper.open()

    # -- orchestration (don't usually need to override this) --------------
    async def run(self, object_label: str = "object") -> bool:
        """Full hand-over. Returns True if released into the person's hand,
        False if we couldn't (dropped it) or the person declined (we keep it)."""
        if not await self.ensure_holding():
            self.audio.speak(f"I don't seem to be holding the {object_label} anymore.")
            return False

        self.audio.speak(f"Bringing the {object_label} to you.")
        await self.present_pose()

        if not await self.ensure_holding():
            self.audio.speak(f"I lost the {object_label} on the way, let me try again.")
            return False

        if await self.confirm(object_label):
            await self.release()
            self.audio.speak("There you go.")
            return True

        self.audio.speak(f"Okay, I'll keep holding the {object_label} until you're ready.")
        return False


# Functional convenience wrapper (same behaviour, for callers that don't want
# to instantiate the class). A fetch skill can use either.
async def handover(
    arm: Any,
    gripper: Any,
    audio: Optional[AudioIO] = None,
    *,
    object_label: str = "object",
    handover_joints: Optional[List[float]] = None,
) -> bool:
    return await HandoverSkill(
        arm, gripper, audio, handover_joints=handover_joints
    ).run(object_label)


async def ensure_holding(gripper: Any, *, retries: int = 2) -> bool:
    """Standalone reactive-hold helper (re-grabs on a reported slip)."""
    skill = HandoverSkill(arm=None, gripper=gripper, regrab_retries=retries)
    return await skill.ensure_holding()
