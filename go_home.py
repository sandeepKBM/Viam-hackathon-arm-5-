import asyncio

from dotenv import load_dotenv

from components.arm import ArmComponent
from components.connection import connect_machine


async def main() -> None:
    load_dotenv()
    machine = await connect_machine()
    try:
        arm = ArmComponent(machine)
        print("Moving to home...")
        await arm.go_home()
        pose = await arm.get_end_position()
        print(f"home: x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f}")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
