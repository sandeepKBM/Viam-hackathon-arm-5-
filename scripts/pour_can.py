"""Standalone side-pick of the soda can, then pour into the cup.

Not wired into voice or the orchestrator. Tune here, then integrate later.

  python scripts/pour_can.py              # detect + print the plan (no motion)
  python scripts/pour_can.py --go         # execute the pour

Pick TCP sits 1.5 inches above the taught table floor (MIN_Z). Side approach
uses the taught dropoff (horizontal) wrist and comes in along Y, away from
the cup. The pour station is 3.5 inches along world X from the cup, and
tilting starts there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from dataclasses import asdict, dataclass

import boot  # noqa: F401
from components.arm import ArmComponent
from components.connection import connect_machine
from components.constants import DROPOFF_POSE, MIN_Z, TRAVEL_Z, WORKSPACE_CORNERS
from components.gripper import GripperComponent
from components.safety import in_workspace
from components.shapes import LocatedShape
from components.vision import VisionComponent

INCH_MM = 25.4
PICK_HEIGHT_IN = float(os.environ.get("POUR_PICK_HEIGHT_IN", "1.5"))
POUR_X_OFFSET_IN = float(os.environ.get("POUR_X_OFFSET_IN", "3.5"))
TILT_DEG = float(os.environ.get("POUR_TILT_DEG", "90"))
APPROACH_IN = float(os.environ.get("POUR_APPROACH_IN", "3"))
TILT_STEPS = int(os.environ.get("POUR_TILT_STEPS", "4"))
HOLD_S = float(os.environ.get("POUR_HOLD_S", "1.5"))
# Flange-to-jaw offset along the tool axis. Top-down picks hide this in MIN_Z;
# a side grasp has to pull the flange back so the pads land on the can.
STICKOUT_MM = float(os.environ.get("POUR_STICKOUT_MM", "90"))
SIDE_THETA = float(os.environ.get("POUR_SIDE_THETA", str(DROPOFF_POSE["theta"])))


def mm(inches: float) -> float:
    return float(inches) * INCH_MM


def _nudge_in_workspace(x: float, y: float, label: str) -> tuple[float, float]:
    if in_workspace(x, y):
        return x, y
    cx = sum(p[0] for p in WORKSPACE_CORNERS) / len(WORKSPACE_CORNERS)
    cy = sum(p[1] for p in WORKSPACE_CORNERS) / len(WORKSPACE_CORNERS)
    for t in (0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2):
        nx = x + (cx - x) * t
        ny = y + (cy - y) * t
        if in_workspace(nx, ny):
            print(f"  nudged {label} ({x:.1f},{y:.1f}) -> ({nx:.1f},{ny:.1f})")
            return nx, ny
    raise ValueError(f"{label} xy=({x:.1f},{y:.1f}) is outside the workspace")


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    o_x: float
    o_y: float
    o_z: float
    theta: float


@dataclass
class PourPlan:
    can: dict
    cup: dict
    pick_z: float
    pour_z: float
    x_sign: float
    tool_x: float
    x_offset_mm: float
    waypoints: list[Waypoint]


def _side_ori(tool_x: float, tilt_deg: float) -> dict:
    """Horizontal X-approach. tilt_deg=0 points along tool_x; 90 tips down."""
    rad = math.radians(tilt_deg)
    return {
        "o_x": tool_x * math.cos(rad),
        "o_y": 0.0,
        "o_z": -math.sin(rad),
        "theta": SIDE_THETA,
    }


def _wp(name: str, x: float, y: float, z: float, ori: dict) -> Waypoint:
    return Waypoint(name=name, x=x, y=y, z=z, **ori)


def _xyz(block: LocatedShape) -> dict:
    return {
        "color": block.color,
        "xy_mm": [round(block.x, 1), round(block.y, 1)],
        "world_z_mm": round(block.z, 1),
        "depth_mm": round(block.depth_mm, 1),
        "bbox": block.shape.box if block.shape else None,
        "in_workspace": in_workspace(block.x, block.y),
    }


def _area(block: LocatedShape) -> float:
    return block.shape.area if block.shape else 0.0


def _on_table(block: LocatedShape) -> bool:
    # Tall cans/bottles often sample the lid (closer depth, world z ~150mm).
    if block.color in {"can", "bottle"} and -20 <= block.z <= 250:
        return True
    if block.depth_mm > 0 and not (400 <= block.depth_mm <= 650):
        print(
            f"  skip {block.color} depth={block.depth_mm:.0f}mm "
            f"xy=({block.x:.1f},{block.y:.1f}) (off table)"
        )
        return False
    return True


def _pick_source_and_cup(blocks: list[LocatedShape]) -> tuple[LocatedShape, LocatedShape]:
    blocks = [b for b in blocks if _on_table(b)]
    cups = [b for b in blocks if b.color == "cup"]
    cans = [b for b in blocks if b.color == "can"]
    bottles = [b for b in blocks if b.color == "bottle"]
    if cups:
        cup = max(cups, key=_area)
        pool = cans or bottles
        if not pool:
            raise RuntimeError("no soda can or bottle in view")
        source = max(pool, key=_area)
        if source.color == "bottle":
            print("  no soda can; using bottle")
        return source, cup
    if bottles and cans:
        print("  no cup label; using bottle as source and can as cup")
        return max(bottles, key=_area), max(cans, key=_area)
    raise RuntimeError("need a pour source and a cup in view")


def build_plan(
    can: LocatedShape,
    cup: LocatedShape,
    *,
    pick_height_in: float = PICK_HEIGHT_IN,
    x_offset_in: float = POUR_X_OFFSET_IN,
    tilt_deg: float = TILT_DEG,
    approach_in: float = APPROACH_IN,
) -> PourPlan:
    can_x, can_y = _nudge_in_workspace(can.x, can.y, "can")
    cup_x, cup_y = _nudge_in_workspace(cup.x, cup.y, "cup")

    pick_z = MIN_Z + mm(pick_height_in)
    pour_z = pick_z
    x_sign = 1.0 if can_x >= cup_x else -1.0
    x_offset = mm(x_offset_in)
    approach = mm(approach_in)
    pour_x = cup_x + x_sign * x_offset
    if not in_workspace(pour_x, cup_y):
        x_sign = -x_sign
        pour_x = cup_x + x_sign * x_offset
        print(f"  pour X flipped to stay in workspace (sign={x_sign:.0f})")

    # Prefer tool +X (approach from -X). That pose already reached the can.
    tool_x = 1.0
    grasp_x = can_x - tool_x * STICKOUT_MM
    standoff_x = grasp_x - tool_x * approach
    if not (in_workspace(grasp_x, can_y) and in_workspace(standoff_x, can_y)):
        tool_x = -1.0
        grasp_x = can_x - tool_x * STICKOUT_MM
        standoff_x = grasp_x - tool_x * approach
        print("  side approach flipped to -X to stay in workspace")

    upright = _side_ori(tool_x, 0.0)
    tilted = _side_ori(tool_x, tilt_deg)

    if not in_workspace(pour_x, cup_y):
        raise ValueError(
            f"pour station ({pour_x:.1f},{cup_y:.1f}) is outside the workspace"
        )
    if not in_workspace(grasp_x, can_y) or not in_workspace(standoff_x, can_y):
        raise ValueError(
            f"side approach ({standoff_x:.1f},{can_y:.1f}) -> "
            f"({grasp_x:.1f},{can_y:.1f}) is outside the workspace"
        )

    waypoints = [
        _wp("travel_standoff", standoff_x, can_y, TRAVEL_Z, upright),
        _wp("travel_grasp", grasp_x, can_y, TRAVEL_Z, upright),
        _wp("side_grasp", grasp_x, can_y, pick_z, upright),
        _wp("lift_can", grasp_x, can_y, TRAVEL_Z, upright),
        _wp("travel_pour", pour_x, cup_y, TRAVEL_Z, upright),
        _wp("pour_station", pour_x, cup_y, pour_z, upright),
        _wp("tilt", pour_x, cup_y, pour_z, tilted),
        _wp("untilt", pour_x, cup_y, pour_z, upright),
        _wp("retreat", pour_x, cup_y, TRAVEL_Z, upright),
    ]
    return PourPlan(
        can=_xyz(can),
        cup=_xyz(cup),
        pick_z=pick_z,
        pour_z=pour_z,
        x_sign=x_sign,
        tool_x=tool_x,
        x_offset_mm=x_offset,
        waypoints=waypoints,
    )


def _plan_json(plan: PourPlan) -> dict:
    return {
        "can": plan.can,
        "cup": plan.cup,
        "pick_z_mm": round(plan.pick_z, 1),
        "pour_z_mm": round(plan.pour_z, 1),
        "x_sign": plan.x_sign,
        "tool_x": plan.tool_x,
        "x_offset_mm": round(plan.x_offset_mm, 1),
        "waypoints": [asdict(wp) for wp in plan.waypoints],
    }


async def _move(arm: ArmComponent, wp: Waypoint) -> None:
    print(
        f"  {wp.name}: xy=({wp.x:.1f},{wp.y:.1f}) z={wp.z:.1f} "
        f"o=({wp.o_x:.2f},{wp.o_y:.2f},{wp.o_z:.2f}) theta={wp.theta:.1f}",
        flush=True,
    )
    await arm.move_to_position(
        wp.x,
        wp.y,
        wp.z,
        o_x=wp.o_x,
        o_y=wp.o_y,
        o_z=wp.o_z,
        theta=wp.theta,
        timeout=60,
        floor=MIN_Z,
    )


async def execute(
    arm: ArmComponent,
    gripper: GripperComponent,
    plan: PourPlan,
    *,
    tilt_deg: float,
    hold_s: float,
    release: bool,
) -> None:
    by_name = {wp.name: wp for wp in plan.waypoints}
    await gripper.open_full()
    await _move(arm, by_name["travel_standoff"])
    await _move(arm, by_name["travel_grasp"])
    await _move(arm, by_name["side_grasp"])
    grasp = await gripper.grab()
    await _move(arm, by_name["lift_can"])
    if not grasp.holding:
        await gripper.open_full()
        raise RuntimeError("gripper did not grab the can")

    await _move(arm, by_name["travel_pour"])
    await _move(arm, by_name["pour_station"])

    station = by_name["pour_station"]
    steps = max(1, TILT_STEPS)
    for i in range(1, steps + 1):
        deg = tilt_deg * i / steps
        await _move(arm, _wp(f"tilt_{deg:.0f}", station.x, station.y, station.z, _side_ori(plan.tool_x, deg)))
    if hold_s > 0:
        await asyncio.sleep(hold_s)
    await _move(arm, by_name["untilt"])
    await _move(arm, by_name["retreat"])
    if release:
        await gripper.open_full()
    await arm.go_home()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Side-pick a soda can and pour into the cup")
    parser.add_argument("--go", action="store_true", help="execute the pour (default is dry-run)")
    parser.add_argument("--release", action="store_true", help="open the gripper after retreating")
    parser.add_argument("--pick-height", type=float, default=PICK_HEIGHT_IN, help="inches above MIN_Z")
    parser.add_argument("--x-offset", type=float, default=POUR_X_OFFSET_IN, help="inches from cup in X")
    parser.add_argument("--tilt", type=float, default=TILT_DEG, help="pour tilt degrees")
    parser.add_argument("--hold", type=float, default=HOLD_S, help="seconds to hold the tilt")
    args = parser.parse_args()

    machine = None
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            machine = await connect_machine()
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            print(f"  connect attempt {attempt}/3 failed: {exc}", flush=True)
            await asyncio.sleep(2)
    if machine is None:
        raise RuntimeError(f"could not reach the machine: {last_err}")
    try:
        arm = ArmComponent(machine)
        gripper = GripperComponent(machine)
        vision = VisionComponent(machine)

        if args.go:
            print("Moving to home for a clear view...")
            await arm.go_home()
            await gripper.hold_open()

        print("Detecting can + cup...")
        blocks = await vision.locate_blocks(colors=("can", "cup", "bottle"))
        can, cup = _pick_source_and_cup(blocks)
        plan = build_plan(
            can,
            cup,
            pick_height_in=args.pick_height,
            x_offset_in=args.x_offset,
            tilt_deg=args.tilt,
        )
        print(json.dumps(_plan_json(plan), indent=2))
        if not args.go:
            print("dry-run only; pass --go to execute")
            return
        await execute(
            arm,
            gripper,
            plan,
            tilt_deg=args.tilt,
            hold_s=args.hold,
            release=args.release,
        )
        print("pour finished")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
