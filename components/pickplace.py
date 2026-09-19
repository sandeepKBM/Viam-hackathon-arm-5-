import os
from typing import Dict, List

from components.arm import ArmComponent
from components.debug_view import prediction_from_block, publish, save_failure
from components.constants import (
    COLOR_BINS,
    FLOOR_PICK_OBJECTS,
    FLOOR_Z,
    MIN_Z,
    PEN_PICK_Z_OFFSET,
    PICK_ORIENTATION,
    TABLE_Z,
    TAUGHT_POSES,
    TRAVEL_Z,
)
from components.gripper import GripperComponent
from components.safety import clamp_z, in_workspace
from components.shapes import CUBE_AR_MAX, LocatedShape

# Extra yaw if the jaws still land on the short ends (try 90).
GRIPPER_YAW_OFFSET = float(os.environ.get("GRIPPER_YAW_OFFSET", 0))


def tcp_pick_z(block: LocatedShape) -> float:
    """TCP pick height.

    Bottle, can, and pen use the taught floor (MIN_Z). Depth still sets XY.
    Blocks keep the depth-derived world Z plus the floor offset.
    """
    if block.color == "pen":
        return MIN_Z - PEN_PICK_Z_OFFSET
    if block.color in FLOOR_PICK_OBJECTS:
        return clamp_z(MIN_Z)
    if block.z == 0.0 and block.depth_mm <= 0:
        return clamp_z(FLOOR_Z)
    return clamp_z(block.z + (FLOOR_Z - TABLE_Z))


def _nearest_periodic(value: float, ref: float, period: float = 180.0) -> float:
    half = period / 2.0
    delta = (value - ref + half) % period - half
    return ref + delta


def pick_orientation(block: LocatedShape) -> dict:
    """Yaw the downward TCP so the jaws close on the long faces.

    Parallel-jaw grippers repeat every 180°, so the chosen theta stays near
    the taught home wrist instead of spinning joint 6. Cubes keep home.
    """
    ori = dict(PICK_ORIENTATION)
    shape = block.shape
    if shape is None or shape.aspect_ratio <= CUBE_AR_MAX:
        return ori
    target = -block.yaw + GRIPPER_YAW_OFFSET
    ori["theta"] = _nearest_periodic(target, PICK_ORIENTATION["theta"])
    return ori


def pick_order(blocks: List[LocatedShape]) -> List[LocatedShape]:
    rank = {"yellow": 0, "red": 1, "can": 2, "cup": 3, "airpods": 4, "pen": 5, "bottle": 6}
    return sorted(
        blocks,
        key=lambda b: (
            rank.get(b.color, 9),
            -(b.shape.area if b.shape else 0.0),
        ),
    )


class PickPlace:
    def __init__(self, arm: ArmComponent, gripper: GripperComponent) -> None:
        self.arm = arm
        self.gripper = gripper

    async def _above(
        self,
        x: float,
        y: float,
        z: float,
        orientation: dict | None = None,
        check_workspace: bool = True,
        floor: float | None = MIN_Z,
    ) -> None:
        ori = orientation or PICK_ORIENTATION
        await self.arm.move_to_position(
            x,
            y,
            z,
            timeout=60,
            check_workspace=check_workspace,
            floor=floor,
            **ori,
        )

    async def _place_at(self, name: str, travel_ori: dict) -> dict | None:
        dest = TAUGHT_POSES.get(name)
        if dest is None:
            await self.arm.go_to(name)
            return None
        dest = dict(dest)
        print(
            f"  travel z={TRAVEL_Z:.1f} -> {name} "
            f"xy=({dest['x']:.1f}, {dest['y']:.1f})"
        )
        await self._above(
            dest["x"],
            dest["y"],
            TRAVEL_Z,
            travel_ori,
            check_workspace=False,
        )
        place_ori = {
            "o_x": dest["o_x"],
            "o_y": dest["o_y"],
            "o_z": dest["o_z"],
            "theta": dest["theta"],
        }
        print(f"  descend {name} z={dest['z']:.1f}")
        await self._above(
            dest["x"],
            dest["y"],
            dest["z"],
            place_ori,
            check_workspace=False,
            floor=None if dest["z"] < MIN_Z else MIN_Z,
        )
        return dest

    async def hand_to_human(self) -> dict | None:
        dest = TAUGHT_POSES["handoff"]
        travel_ori = {
            "o_x": dest["o_x"],
            "o_y": dest["o_y"],
            "o_z": dest["o_z"],
            "theta": dest["theta"],
        }
        placed = await self._place_at("handoff", travel_ori)
        await self.gripper.open_full()
        return placed

    async def pick_and_place(
        self, block: LocatedShape, color_bins: Dict[str, str] | None = None
    ) -> bool:
        bins = color_bins or COLOR_BINS
        bin_name = bins.get(block.color)
        if bin_name is None:
            raise ValueError(f"no bin mapped for color {block.color!r}")
        if not in_workspace(block.x, block.y):
            raise ValueError(
                f"{block.color} at ({block.x:.1f}, {block.y:.1f}) is outside the workspace"
            )

        z = tcp_pick_z(block)
        ori = pick_orientation(block)
        aspect = block.shape.aspect_ratio if block.shape else 1.0
        print(
            f"  pick {block.color} -> {bin_name}  "
            f"region_px=({block.u:.0f},{block.v:.0f}) "
            f"xy=({block.x:.1f}, {block.y:.1f}) "
            f"depth={block.depth_mm:.0f}mm world_z={block.z:.1f} pick_z={z:.1f}  "
            f"long_yaw={block.yaw:.1f} theta={ori['theta']:.1f} ar={aspect:.2f}"
        )
        await self.gripper.open_full()
        await self._above(block.x, block.y, TRAVEL_Z, ori)
        await self._above(
            block.x,
            block.y,
            z,
            ori,
            floor=None if z < MIN_Z else MIN_Z,
        )
        grasp = await self.gripper.grab()
        await self._above(block.x, block.y, TRAVEL_Z, ori)
        if not grasp.holding:
            await self.gripper.open_full()
            return False
        dest = None
        try:
            dest = await self._place_at(bin_name, ori)
        except Exception:
            if bin_name in {"dropoff", "handoff"}:
                await self.gripper.open_full()
            raise
        if bin_name in {"dropoff", "handoff"}:
            await self.gripper.open_full()
        else:
            await self.gripper.hold_open()
        dest = dest or TAUGHT_POSES.get(bin_name)
        if dest is not None:
            await self._above(
                dest["x"], dest["y"], TRAVEL_Z, ori, check_workspace=False
            )
        print(f"  placed in {bin_name}")
        return True

    async def sort_blocks(
        self,
        blocks: List[LocatedShape],
        color_bins: Dict[str, str] | None = None,
        counts: Dict[str, int | None] | None = None,
    ) -> dict:
        bins = color_bins or COLOR_BINS
        results = {"placed": [], "skipped": []}
        used: Dict[str, int] = {}
        wanted: List[LocatedShape] = []
        for block in pick_order(blocks):
            if block.color not in bins:
                continue
            limit = counts.get(block.color) if counts and block.color in counts else None
            if limit is not None and used.get(block.color, 0) >= limit:
                continue
            wanted.append(block)
            used[block.color] = used.get(block.color, 0) + 1
        if counts:
            print(
                "  request: "
                + ", ".join(
                    f"{name} x{('all' if counts.get(name) is None else counts.get(name))}"
                    for name in bins
                ),
                flush=True,
            )
        remaining = list(wanted)
        while remaining:
            block = remaining.pop(0)
            dest = bins[block.color]
            target = prediction_from_block(block, dest, tcp_pick_z(block))
            publish(
                [prediction_from_block(b, bins.get(b.color), tcp_pick_z(b)) for b in wanted],
                target,
                context={"task": "sort", "bins": bins, "counts": counts or {}},
            )
            try:
                ok = await self.pick_and_place(block, bins)
            except Exception as exc:
                print(f"  skip {block.color}: {exc}")
                results["skipped"].append(
                    {"color": block.color, "bin": dest, "x": block.x, "y": block.y, "error": str(exc)}
                )
                save_failure(
                    target,
                    {
                        "xy_mm": [block.x, block.y],
                        "pick_z_mm": tcp_pick_z(block),
                        "theta": pick_orientation(block).get("theta"),
                    },
                    str(exc),
                )
                await self.arm.go_home()
                continue
            if ok:
                results["placed"].append(
                    {"color": block.color, "bin": dest, "x": block.x, "y": block.y}
                )
                if remaining:
                    print("  stay at travel height for the next object", flush=True)
            else:
                error = "gripper did not grab"
                grasp = self.gripper.last_grasp
                results["skipped"].append(
                    {
                        "color": block.color,
                        "bin": dest,
                        "x": block.x,
                        "y": block.y,
                        "error": error,
                    }
                )
                save_failure(
                    target,
                    {
                        "xy_mm": [block.x, block.y],
                        "pick_z_mm": tcp_pick_z(block),
                        "theta": pick_orientation(block).get("theta"),
                        "jaws": None if grasp is None else grasp.pos,
                        "holding": None if grasp is None else grasp.holding,
                        "torque": None if grasp is None else grasp.torque,
                    },
                    error,
                )
        return results
