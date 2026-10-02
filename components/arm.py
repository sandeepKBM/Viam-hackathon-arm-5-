import os

from viam.components.arm import Arm
from viam.proto.common import Pose
from viam.proto.component.arm import JointPositions
from viam.robot.client import RobotClient

from components.constants import HOME_JOINTS, MIN_Z, TAUGHT_JOINTS
from components.safety import in_workspace, make_pose


class ArmComponent:
    def __init__(self, machine: RobotClient, name: str | None = None) -> None:
        self.machine = machine
        self.name = name or os.environ.get("ARM_NAME", "arm")
        self._arm = Arm.from_robot(robot=machine, name=self.name)

    async def get_joint_positions(self, timeout: float = 10) -> list[float]:
        positions = await self._arm.get_joint_positions(timeout=timeout)
        return list(positions.values)

    async def get_end_position(self, timeout: float = 10) -> Pose:
        return await self._arm.get_end_position(timeout=timeout)

    async def go_home(self, timeout: float = 60) -> None:
        await self.move_to_joints(HOME_JOINTS, timeout=timeout)

    async def move_to_joints(self, joints: list[float], timeout: float = 60) -> None:
        await self._arm.move_to_joint_positions(
            JointPositions(values=joints), timeout=timeout
        )

    async def go_to(self, name: str, timeout: float = 60) -> None:
        joints = TAUGHT_JOINTS.get(name)
        if joints is None:
            raise ValueError(f"unknown taught pose {name!r}; expected one of {list(TAUGHT_JOINTS)}")
        await self.move_to_joints(joints, timeout=timeout)

    async def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        o_x: float = 0.0,
        o_y: float = 0.0,
        o_z: float = -1.0,
        theta: float = 0.0,
        timeout: float = 30,
        check_workspace: bool = True,
        floor: float | None = MIN_Z,
    ) -> None:
        pose = make_pose(
            x,
            y,
            z,
            o_x=o_x,
            o_y=o_y,
            o_z=o_z,
            theta=theta,
            check_workspace=check_workspace,
            floor=floor,
        )
        await self._arm.move_to_position(pose, timeout=timeout)

    async def workspace_status(self, timeout: float = 10) -> dict:
        pose = await self.get_end_position(timeout=timeout)
        return {
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "in_workspace": in_workspace(pose.x, pose.y),
            "above_floor": pose.z >= MIN_Z,
        }
