from typing import List

from components.arm import ArmComponent
from components.constants import COLOR_BINS, MIN_Z, PICK_ORIENTATION, TRAVEL_Z
from components.gripper import GripperComponent
from components.safety import in_workspace
from components.shapes import LocatedShape


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

    async def _above(self, x: float, y: float, z: float) -> None:
        await self.arm.move_to_position(x, y, z, timeout=60, **PICK_ORIENTATION)

    async def pick_and_place(self, block: LocatedShape) -> bool:
        bin_name = COLOR_BINS.get(block.color)
        if bin_name is None:
            raise ValueError(f"no bin mapped for color {block.color!r}")
        if not in_workspace(block.x, block.y):
            raise ValueError(
                f"{block.color} at ({block.x:.1f}, {block.y:.1f}) is outside the workspace"
            )

        print(
            f"  pick {block.color} -> {bin_name}  "
            f"xy=({block.x:.1f}, {block.y:.1f}) z={MIN_Z:.1f}"
        )
        await self.gripper.open()
        await self._above(block.x, block.y, TRAVEL_Z)
        await self._above(block.x, block.y, MIN_Z)
        grabbed = await self.gripper.grab()
        print(f"  grabbed: {grabbed}")
        await self._above(block.x, block.y, TRAVEL_Z)
        if not grabbed:
            return False
        await self.arm.go_to(bin_name)
        await self.gripper.open()
        print(f"  placed in {bin_name}")
        return True

    async def sort_blocks(self, blocks: List[LocatedShape]) -> dict:
        results = {"placed": [], "skipped": []}
        for block in pick_order(blocks):
            try:
                ok = await self.pick_and_place(block)
            except Exception as exc:
                print(f"  skip {block.color}: {exc}")
                results["skipped"].append(
                    {"color": block.color, "x": block.x, "y": block.y, "error": str(exc)}
                )
                await self.arm.go_home()
                continue
            dest = COLOR_BINS[block.color]
            if ok:
                results["placed"].append(
                    {"color": block.color, "bin": dest, "x": block.x, "y": block.y}
                )
            else:
                results["skipped"].append(
                    {
                        "color": block.color,
                        "x": block.x,
                        "y": block.y,
                        "error": "gripper did not grab",
                    }
                )
            await self.arm.go_home()
        return results
