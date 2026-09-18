"""Thin wrapper around the Viam ``Arm`` component.

Target hardware: a UFactory XL15 (UFactory xArm family), driven through the
Viam `Arm` component/API -- this module does not talk to UFactory's own SDK
directly.

TODO: fill in DOF, joint limits, and reach from the UFactory XL15 datasheet
once available; do not hard-code guessed numbers here. See
`arm5.controls.safety.SafetyLimits` for where those limits get enforced.
"""

from __future__ import annotations

from viam.components.arm import Arm, JointPositions, Pose
from viam.robot.client import RobotClient


class ArmController:
    """Convenience wrapper around a single Viam ``Arm`` resource.

    Usage:
        arm = ArmController.from_robot(robot, "arm0")
        pose = await arm.get_end_position()
    """

    def __init__(self, arm: Arm) -> None:
        self._arm = arm

    @classmethod
    def from_robot(cls, robot: RobotClient, name: str) -> "ArmController":
        """Look up the named arm resource on ``robot`` and wrap it."""
        return cls(Arm.from_robot(robot, name))

    async def get_end_position(self) -> Pose:
        """Return the arm's current end-effector pose."""
        return await self._arm.get_end_position()

    async def move_to_position(self, pose: Pose) -> None:
        """Move the end effector to ``pose``.

        TODO: decide how obstacles / world_state get threaded through here
        vs. going through `arm5.planning.classical.motion` instead.
        """
        await self._arm.move_to_position(pose)

    async def get_joint_positions(self) -> JointPositions:
        """Return the arm's current joint positions."""
        return await self._arm.get_joint_positions()

    async def move_to_joint_positions(self, positions: JointPositions) -> None:
        """Move the arm to the given joint positions.

        TODO: add joint-limit validation via `arm5.controls.safety` before
        issuing the move.
        """
        await self._arm.move_to_joint_positions(positions)
