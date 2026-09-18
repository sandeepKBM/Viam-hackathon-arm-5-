import os

from viam.components.gripper import Gripper
from viam.robot.client import RobotClient


class GripperComponent:
    def __init__(self, machine: RobotClient, name: str | None = None) -> None:
        self.name = name or os.environ.get("GRIPPER_NAME", "gripper")
        self._gripper = Gripper.from_robot(robot=machine, name=self.name)

    async def open(self, timeout: float = 10) -> None:
        await self._gripper.open(timeout=timeout)

    async def grab(self, timeout: float = 10) -> bool:
        return bool(await self._gripper.grab(timeout=timeout))
