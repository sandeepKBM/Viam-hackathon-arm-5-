"""Thin wrapper around the Viam ``Gripper`` component."""

from __future__ import annotations

from viam.components.gripper import Gripper
from viam.robot.client import RobotClient


class GripperController:
    """Convenience wrapper around a single Viam ``Gripper`` resource.

    Usage:
        gripper = GripperController.from_robot(robot, "gripper0")
        await gripper.open()
    """

    def __init__(self, gripper: Gripper) -> None:
        self._gripper = gripper

    @classmethod
    def from_robot(cls, robot: RobotClient, name: str) -> "GripperController":
        """Look up the named gripper resource on ``robot`` and wrap it."""
        return cls(Gripper.from_robot(robot, name))

    async def open(self) -> None:
        """Open the gripper fully."""
        await self._gripper.open()

    async def grab(self) -> bool:
        """Close the gripper and return True if an object was grasped.

        TODO: expose grip-force / timeout tuning once hardware is on hand.
        """
        return await self._gripper.grab()

    async def stop(self) -> None:
        """Stop any in-progress gripper motion (e-stop-adjacent)."""
        await self._gripper.stop()
