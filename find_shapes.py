import asyncio
import json

from dotenv import load_dotenv

from components.arm import ArmComponent
from components.connection import connect_machine
from components.vision import VisionComponent


async def main() -> None:
    load_dotenv()
    machine = await connect_machine()
    try:
        arm = ArmComponent(machine)
        print("Moving to home...")
        await arm.go_home()
        vision = VisionComponent(machine)
        shapes_2d = await vision.find_shapes()
        print("== 2D (OpenCV on cam) ==")
        if not shapes_2d:
            print("  (no red shapes)")
        for s in shapes_2d:
            print(
                json.dumps(
                    {
                        "label": s.label,
                        "center_px": [s.cx, s.cy],
                        "box": s.box,
                        "aspect_ratio": round(s.aspect_ratio, 2),
                    }
                )
            )

        print("\n== 3D world (mm) ==")
        try:
            located = await vision.locate_shapes()
            if not located:
                print("  (none)")
            for s in located:
                print(json.dumps({"label": s.label, "x": s.x, "y": s.y, "z": s.z}))
        except Exception as exc:
            print(f"  locate skipped: {exc}")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
