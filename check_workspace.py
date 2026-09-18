import asyncio
import json

from dotenv import load_dotenv

from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import FLOOR_Z, MAX_Y, WORKSPACE_CORNERS


async def main() -> None:
    load_dotenv()
    machine = await connect_machine()
    try:
        arm = ArmComponent(machine)
        status = await arm.workspace_status()
        print(
            json.dumps(
                {
                    "floor_z_mm": FLOOR_Z,
                    "max_y_mm": MAX_Y,
                    "workspace_corners_mm": WORKSPACE_CORNERS,
                    "current": status,
                },
                indent=2,
            )
        )
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
