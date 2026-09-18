import os
from typing import Dict, List

from components.arm import ArmComponent
from components.constants import COLOR_BINS, FLOOR_Z, PICK_ORIENTATION, TABLE_Z, TRAVEL_Z
from components.gripper import GripperComponent
from components.safety import clamp_z, in_workspace
from components.shapes import CUBE_AR_MAX, LocatedShape

# Extra yaw if the jaws still land on the short ends (try 90).
GRIPPER_YAW_OFFSET = float(os.environ.get("GRIPPER_YAW_OFFSET", 0))


def tcp_pick_z(block: LocatedShape) -> float:
    """TCP pick height from this block's depth-derived world Z.

    `block.z` is the object surface in world (from the depth image). FLOOR_Z is
    the taught TCP height at the table, so FLOOR_Z - TABLE_Z is the gripper
    offset. The floor clamp still refuses anything below MIN_Z / FLOOR_Z.
    """
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
    rank = {"yellow": 0, "red": 1}
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

    async def _above(self, x: float, y: float, z: float, orientation: dict | None = None) -> None:
        ori = orientation or PICK_ORIENTATION
        await self.arm.move_to_position(x, y, z, timeout=60, **ori)

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
            f"xy=({block.x:.1f}, {block.y:.1f}) "
            f"depth={block.depth_mm:.0f}mm world_z={block.z:.1f} pick_z={z:.1f}  "
            f"long_yaw={block.yaw:.1f} theta={ori['theta']:.1f} ar={aspect:.2f}"
        )
        await self.gripper.open_full()
        await self._above(block.x, block.y, TRAVEL_Z, ori)
        await self._above(block.x, block.y, z, ori)
        grasp = await self.gripper.grab()
        await self._above(block.x, block.y, TRAVEL_Z, ori)
        if not grasp.holding:
            await self.gripper.hold_open()
            return False
        await self.arm.go_to(bin_name)
        await self.gripper.hold_open()
        print(f"  placed in {bin_name}")
        return True

    async def sort_blocks(
        self,
        blocks: List[LocatedShape],
        color_bins: Dict[str, str] | None = None,
    ) -> dict:
        bins = color_bins or COLOR_BINS
        results = {"placed": [], "skipped": []}
        wanted = [b for b in pick_order(blocks) if b.color in bins]
        for block in wanted:
            dest = bins[block.color]
            try:
                ok = await self.pick_and_place(block, bins)
            except Exception as exc:
                print(f"  skip {block.color}: {exc}")
                results["skipped"].append(
                    {"color": block.color, "bin": dest, "x": block.x, "y": block.y, "error": str(exc)}
                )
                await self.arm.go_home()
                continue
            if ok:
                results["placed"].append(
                    {"color": block.color, "bin": dest, "x": block.x, "y": block.y}
                )
            else:
                results["skipped"].append(
                    {
                        "color": block.color,
                        "bin": dest,
                        "x": block.x,
                        "y": block.y,
                        "error": "gripper did not grab",
                    }
                )
            await self.arm.go_home()
        return results
