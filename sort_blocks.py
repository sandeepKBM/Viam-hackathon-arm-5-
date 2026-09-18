import asyncio
import json

from dotenv import load_dotenv

from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import COLOR_BINS, MIN_Z
from components.gripper import GripperComponent
from components.pickplace import PickPlace, pick_order
from components.vision import VisionComponent


async def main() -> None:
    load_dotenv()
    machine = await connect_machine()
    try:
        arm = ArmComponent(machine)
        gripper = GripperComponent(machine)
        vision = VisionComponent(machine)
        sorter = PickPlace(arm, gripper)

        print("Moving to home...")
        await arm.go_home()
        await gripper.open()

        print("Detecting red / yellow blocks...")
        blocks = await vision.locate_blocks()
        plan = [
            {
                "color": b.color,
                "bin": COLOR_BINS.get(b.color),
                "xy_mm": [round(b.x, 1), round(b.y, 1)],
                "pick_z_mm": MIN_Z,
                "bbox": b.shape.box if b.shape else None,
            }
            for b in pick_order(blocks)
        ]
        print(json.dumps({"count": len(plan), "plan": plan}, indent=2))
        if not plan:
            print("No red or yellow blocks found.")
            return

        results = await sorter.sort_blocks(blocks)
        print(json.dumps(results, indent=2))
        await arm.go_home()
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
