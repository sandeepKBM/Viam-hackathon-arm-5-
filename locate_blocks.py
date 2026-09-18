import asyncio
import json

from dotenv import load_dotenv

from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import COLOR_BINS, MIN_Z
from components.pickplace import pick_order
from components.safety import in_workspace
from components.vision import VisionComponent


async def main() -> None:
    load_dotenv()
    machine = await connect_machine()
    try:
        arm = ArmComponent(machine)
        print("Moving to home...")
        await arm.go_home()
        vision = VisionComponent(machine)
        blocks = await vision.locate_blocks()
        plan = [
            {
                "color": b.color,
                "bin": COLOR_BINS.get(b.color),
                "xy_mm": [round(b.x, 1), round(b.y, 1)],
                "world_z_mm": round(b.z, 1),
                "pick_z_mm": MIN_Z,
                "in_workspace": in_workspace(b.x, b.y),
                "bbox": b.shape.box if b.shape else None,
            }
            for b in pick_order(blocks)
        ]
        print(json.dumps({"count": len(plan), "plan": plan}, indent=2))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
