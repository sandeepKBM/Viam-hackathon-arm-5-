import argparse
import asyncio
import json

import boot  # noqa: F401
from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import COLOR_BINS
from components.gripper import GripperComponent
from components.pickplace import PickPlace, pick_order, tcp_pick_z
from components.vision import VisionComponent


async def main(color_bins: dict | None = None, counts: dict | None = None) -> None:
    bins = color_bins or COLOR_BINS
    machine = await connect_machine()
    try:
        arm = ArmComponent(machine)
        gripper = GripperComponent(machine)
        vision = VisionComponent(machine)
        sorter = PickPlace(arm, gripper)

        print("Moving to home...")
        await arm.go_home()
        await gripper.hold_open()

        colors = tuple(bins)
        print(f"Detecting {', '.join(colors)}...")
        print(f"routing: {bins}")
        if counts:
            print(f"counts: {counts}")
        blocks = await vision.locate_blocks(colors=colors)
        used: dict[str, int] = {}
        planned = []
        for b in pick_order(blocks):
            if b.color not in bins:
                continue
            limit = counts.get(b.color) if counts and b.color in counts else None
            if limit is not None and used.get(b.color, 0) >= limit:
                continue
            planned.append(b)
            used[b.color] = used.get(b.color, 0) + 1
        plan = [
            {
                "color": b.color,
                "bin": bins.get(b.color),
                "xy_mm": [round(b.x, 1), round(b.y, 1)],
                "region_px": [round(b.u, 1), round(b.v, 1)],
                "depth_mm": round(b.depth_mm, 1),
                "world_z_mm": round(b.z, 1),
                "pick_z_mm": round(tcp_pick_z(b), 1),
                "bbox": b.shape.box if b.shape else None,
            }
            for b in planned
        ]
        print(json.dumps({"count": len(plan), "plan": plan}, indent=2))
        if not plan:
            print("No matching objects found.")
            return

        results = await sorter.sort_blocks(blocks, bins, counts)
        print(json.dumps(results, indent=2))
        await arm.go_home()
    finally:
        await machine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        action="append",
        metavar="COLOR=BIN",
        help="override routing, e.g. --route can=dropoff --route cup=handoff",
    )
    args = parser.parse_args()
    routes = dict(COLOR_BINS)
    if args.route:
        routes = {}
        for item in args.route:
            color, _, dest = item.partition("=")
            routes[color.strip().lower()] = dest.strip().lower()
    asyncio.run(main(routes))
