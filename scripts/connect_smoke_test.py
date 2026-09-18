"""Smoke test: connect to the robot, list resources, print arm end position.

Usage:
    export ROBOT_ADDRESS=...
    export ROBOT_API_KEY=...
    export ROBOT_API_KEY_ID=...
    export ROBOT_ARM_NAME=arm   # optional, defaults to "arm"
    python scripts/connect_smoke_test.py
"""

from __future__ import annotations

import asyncio
import os

from arm5.hardware.arm import ArmController
from arm5.hardware.robot import connect


async def main() -> None:
    """Connect to the robot and print basic diagnostics.

    TODO: also exercise the camera/vision wrappers once resource names for
    those are settled (see `config/robot.example.json`).
    """
    robot = await connect()
    try:
        print("Resources:")
        for resource in robot.resource_names:
            print(f"  {resource}")

        arm_name = os.environ.get("ROBOT_ARM_NAME", "arm")
        arm = ArmController.from_robot(robot, arm_name)
        end_position = await arm.get_end_position()
        print(f"Arm '{arm_name}' end position: {end_position}")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
