import asyncio
import json

import boot  # noqa: F401
from components.arm import ArmComponent
from components.connection import connect_machine


async def main() -> None:
    machine = await connect_machine()
    try:
        arm = ArmComponent(machine)
        joints = await arm.get_joint_positions()
        print(json.dumps({"arm": arm.name, "joint_positions_deg": joints}, indent=2))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
