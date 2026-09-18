"""Home the arm, run zero-shot detection on the live camera, print world XY.

No grasp — the zero-shot analogue of locate_blocks.py. Use it to sanity-check
that prompts + depth deprojection land where you expect before sorting.

    python locate_objects_zeroshot.py
    ZS_PROMPTS="red block,yellow block,blue block" python locate_objects_zeroshot.py
"""

import asyncio
import json

from dotenv import load_dotenv

from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import COLOR_BINS, MIN_Z
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
        objects = await vision.locate_objects_zeroshot()
        plan = [
            {
                "label": o.label,
                "color": o.color,
                "bin": COLOR_BINS.get(o.color),
                "xy_mm": [round(o.x, 1), round(o.y, 1)],
                "world_z_mm": round(o.z, 1),
                "pick_z_mm": MIN_Z,
                "in_workspace": in_workspace(o.x, o.y),
                "bbox": o.shape.box if o.shape else None,
            }
            for o in objects
        ]
        print(json.dumps({"count": len(plan), "plan": plan}, indent=2))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
