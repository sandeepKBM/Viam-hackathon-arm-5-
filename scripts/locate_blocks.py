import asyncio
import json

import boot  # noqa: F401
from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import COLOR_BINS
from components.pickplace import pick_orientation, pick_order, tcp_pick_z
from components.safety import in_workspace
from components.vision import VisionComponent


async def main() -> None:
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
                "depth_mm": round(b.depth_mm, 1),
                "world_z_mm": round(b.z, 1),
                "pick_z_mm": round(tcp_pick_z(b), 1),
                "in_workspace": in_workspace(b.x, b.y),
                "bbox": b.shape.box if b.shape else None,
                "aspect": round(b.shape.aspect_ratio, 2) if b.shape else None,
                "long_yaw_deg": round(b.yaw, 1),
                "pick_theta_deg": round(pick_orientation(b)["theta"], 1),
            }
            for b in pick_order(blocks)
        ]
        print(json.dumps({"count": len(plan), "plan": plan}, indent=2))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
