"""Point / locate skill (assistant): aim the gripper at an object to
INDICATE it, without grasping. Position control only.

Composes existing primitives -- does NOT reimplement them:
  - components.constants (TRAVEL_Z / PICK_ORIENTATION) for the raised,
    downward-aimed "point" pose, the same orientation
    components.pickplace.PickPlace._above uses for its travel-height hover.
  - components.safety.in_workspace for the pre-move safety check, exactly
    like PickPlace.pick_and_place does before it commands a move.
  - components.canonicalize for matching a natural-language request
    ("where is the red block") to a LocatedShape-like object in a scene.
  - components.audio.AudioIO (optional) to announce the object's position
    relative to the workspace.

    PointSkill(arm, audio).point_at(target)   -- hover + aim, no grab
    resolve_pointing_target(request, scene)   -- "where is the red block" -> target

Everything is dependency-injected (arm, audio) and async, so it runs offline
with mocks -- no robot, no camera required to exercise the logic.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from components.audio import AudioIO
from components.canonicalize import canonicalize_label
from components.constants import PICK_ORIENTATION, TRAVEL_Z, WORKSPACE_CORNERS
from components.safety import in_workspace

# How far above TRAVEL_Z the gripper hovers when just POINTING (never
# picking): a clearly-raised pose, always an upward move from TRAVEL_Z, never
# a descent -- so it can never be mistaken for (or drift into) a grasp
# approach. Override via env if a taller pose reads better on the real arm.
POINT_HEIGHT_MM = float(os.environ.get("POINT_HEIGHT_MM", TRAVEL_Z + 40.0))

_WORKSPACE_CENTER_X = sum(x for x, _ in WORKSPACE_CORNERS) / len(WORKSPACE_CORNERS)
_WORKSPACE_CENTER_Y = sum(y for _, y in WORKSPACE_CORNERS) / len(WORKSPACE_CORNERS)

# Deadband (mm) around the workspace centroid before calling a direction out
# by name, so near-center objects get "straight ahead" instead of a jittery
# left/right call.
_DIRECTION_DEADBAND_MM = 20.0


def describe_position(x: float, y: float) -> str:
    """A short natural-language direction for (x, y) relative to the
    workspace's centroid (components.constants.WORKSPACE_CORNERS) -- e.g.
    "to your left, further out". Pure/offline, no robot state needed."""
    dy = y - _WORKSPACE_CENTER_Y
    dx = x - _WORKSPACE_CENTER_X
    if dy > _DIRECTION_DEADBAND_MM:
        lr = "to your left"
    elif dy < -_DIRECTION_DEADBAND_MM:
        lr = "to your right"
    else:
        lr = "straight ahead"
    if dx > _DIRECTION_DEADBAND_MM:
        fb = "further out"
    elif dx < -_DIRECTION_DEADBAND_MM:
        fb = "closer in"
    else:
        fb = ""
    return f"{lr}, {fb}" if fb else lr


class PointSkill:
    """Aim the gripper above a target's (x, y) at a raised POINT height to
    indicate it, WITHOUT grasping. Position control only
    (arm.move_to_position); never calls a gripper. Safety-checks the target
    against the taught workspace before moving, exactly like
    components.pickplace.PickPlace does for a pick."""

    def __init__(
        self,
        arm: Any,
        audio: Optional[AudioIO] = None,
        *,
        point_height: float = POINT_HEIGHT_MM,
    ) -> None:
        self.arm = arm
        self.audio = audio
        self.point_height = point_height

    async def point_at(self, target: Any, *, announce: bool = True) -> None:
        """Move to a pointing pose above `target`. Never touches a gripper.
        Raises ValueError if the target is outside the taught workspace --
        checked BEFORE any move is issued (components.safety.in_workspace)."""
        x, y = float(target.x), float(target.y)
        if not in_workspace(x, y):
            raise ValueError(
                f"point target at ({x:.1f}, {y:.1f}) is outside the workspace"
            )

        label = (
            getattr(target, "canonical_label", "")
            or canonicalize_label(getattr(target, "label", ""))
            or getattr(target, "color", "")
            or "object"
        )
        if announce and self.audio is not None:
            self.audio.speak(f"The {label} is {describe_position(x, y)}.")

        # Position control only: a single hover move at the raised point
        # height, aimed downward with the same PICK_ORIENTATION the pick
        # approach uses -- indicates the object without descending toward
        # it (no gripper call, ever, in this method).
        await self.arm.move_to_position(x, y, self.point_height, **PICK_ORIENTATION)


async def point_at(
    arm: Any,
    target: Any,
    audio: Optional[AudioIO] = None,
    **kwargs: Any,
) -> None:
    """Functional convenience wrapper around PointSkill.point_at."""
    await PointSkill(arm, audio, **kwargs).point_at(target)


def resolve_pointing_target(request: str, scene: Sequence[Any]) -> Optional[Any]:
    """Match a natural-language request ("where is the red block", "bring me
    the red block") to the best object in `scene`.

    Reuses components.canonicalize's label normalization (the same
    color+noun vocabulary the rest of the system keys on) so this doesn't
    reimplement its own matching rules: prefers a canonical "<color> <noun>"
    match, falls back to a bare color match, then None if nothing hits.
    Pure/offline -- no model, no network.
    """
    req_l = (request or "").lower()
    if not scene:
        return None

    best: Optional[Any] = None
    best_score = 0
    for obj in scene:
        canon = getattr(obj, "canonical_label", "") or canonicalize_label(
            getattr(obj, "label", "")
        )
        color = (getattr(obj, "color", "") or "").lower()
        score = 0
        if canon and canon in req_l:
            score = 2
        elif color and color in req_l:
            score = 1
        if score > best_score:
            best_score = score
            best = obj

    return best
