import argparse
import asyncio
import json
import time

import boot  # noqa: F401
from components.affordances import GOALS, resolve_goal
from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import COLOR_BINS
from components.debug_view import prediction_from_block, publish
from components.gripper import GripperComponent
from components.pickplace import PickPlace, pick_order, tcp_pick_z
from components.voice import moves_to_plan
from components.vision import VisionComponent


def _planned(blocks, bins: dict, counts: dict | None) -> list:
    used: dict[str, int] = {}
    planned = []
    for block in pick_order(blocks):
        if block.color not in bins:
            continue
        limit = counts.get(block.color) if counts and block.color in counts else None
        if limit is not None and used.get(block.color, 0) >= limit:
            continue
        planned.append(block)
        used[block.color] = used.get(block.color, 0) + 1
    return planned


def _outcome(results: dict | None) -> dict:
    results = results or {"placed": [], "skipped": []}
    placed = results.get("placed") or []
    skipped = results.get("skipped") or []
    if placed and not skipped:
        return {
            "ok": True,
            "moved": True,
            "say": "Done.",
            "results": results,
        }
    if skipped and not placed:
        err = str(skipped[0].get("error") or "")
        say = "I could not pick that up." if "grab" in err else "That task failed."
        return {"ok": False, "moved": True, "say": say, "results": results}
    if placed:
        return {"ok": False, "moved": True, "say": "Partly done.", "results": results}
    return {"ok": False, "moved": False, "say": "Nothing to pick.", "results": results}


async def main(
    color_bins: dict | None = None,
    counts: dict | None = None,
    goal: str | None = None,
    timings: dict | None = None,
    _vision=None,
    _sorter=None,
    _skip_motion_setup: bool = False,
) -> dict:
    timings = timings if timings is not None else {}
    pre_start = time.perf_counter()
    bins = dict(color_bins) if color_bins else {}
    if goal and not bins:
        spec = GOALS.get(goal)
        if spec is None:
            return {"ok": False, "moved": False, "say": "I am not sure what you want."}
        detect_colors = spec["candidates"]
    elif bins:
        detect_colors = tuple(bins)
    else:
        bins = dict(COLOR_BINS)
        detect_colors = tuple(bins)

    machine = None
    if _vision is None or _sorter is None:
        machine = await connect_machine()
    try:
        if machine is not None:
            arm = ArmComponent(machine)
            gripper = GripperComponent(machine)
            vision = _vision or VisionComponent(machine)
            sorter = _sorter or PickPlace(arm, gripper)
            if not _skip_motion_setup:
                print("Moving to home...")
                await arm.go_home()
                await gripper.hold_open()
        else:
            vision = _vision
            sorter = _sorter

        print(f"Detecting {', '.join(detect_colors)}...")
        if bins:
            print(f"routing: {bins}")
        if counts:
            print(f"counts: {counts}")
        vis_start = time.perf_counter()
        blocks = await vision.locate_blocks(colors=detect_colors)
        timings["vision_ms"] = round((time.perf_counter() - vis_start) * 1000.0, 2)

        if goal and not color_bins:
            resolved = resolve_goal(goal, blocks)
            timings["resolution_ms"] = round(resolved["resolution_ms"], 3)
            print(f"  goal={goal} resolution_ms={timings['resolution_ms']:.3f}")
            if not resolved["ok"]:
                timings["total_pre_motion_ms"] = round(
                    (time.perf_counter() - pre_start) * 1000.0, 2
                )
                publish(
                    [prediction_from_block(b, pick_z=tcp_pick_z(b)) for b in blocks],
                    None,
                    context={"task": "sort", "goal": goal},
                )
                print(resolved["say"])
                return {
                    "ok": False,
                    "moved": False,
                    "say": resolved["say"],
                    "results": {"placed": [], "skipped": []},
                    "timings": timings,
                }
            bins, counts = moves_to_plan(resolved["moves"])

        planned = _planned(blocks, bins, counts)
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
        predictions = [
            prediction_from_block(b, bins.get(b.color), tcp_pick_z(b)) for b in blocks
        ]
        target = (
            prediction_from_block(planned[0], bins.get(planned[0].color), tcp_pick_z(planned[0]))
            if planned
            else None
        )
        publish(
            predictions,
            target,
            context={"task": "sort", "bins": bins, "counts": counts or {}, "goal": goal},
        )
        timings["total_pre_motion_ms"] = round((time.perf_counter() - pre_start) * 1000.0, 2)
        print(
            "  timings "
            + json.dumps(
                {k: timings.get(k) for k in ("vision_ms", "resolution_ms", "total_pre_motion_ms")}
            )
        )
        if not planned:
            print("No matching objects found.")
            return {
                "ok": False,
                "moved": False,
                "say": "I cannot see that object.",
                "results": {"placed": [], "skipped": []},
                "timings": timings,
            }

        results = await sorter.sort_blocks(blocks, bins, counts)
        print(json.dumps(results, indent=2))
        if machine is not None and not _skip_motion_setup:
            await arm.go_home()
        out = _outcome(results)
        out["timings"] = timings
        return out
    finally:
        if machine is not None:
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
