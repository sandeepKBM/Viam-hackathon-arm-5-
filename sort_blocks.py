import asyncio
import json

from dotenv import load_dotenv

from components.arm import ArmComponent
from components.connection import close_machine, get_machine
from components.constants import COLOR_BINS, MIN_Z
from components.experience_store import ExperienceStore, resolve_key
from components.gripper import GripperComponent
from components.pickplace import PickPlace, pick_order
from components.vision import VisionComponent


def record_sort_results(store: ExperienceStore, blocks, results: dict) -> None:
    """Persist each sort_blocks() outcome to the per-object experience
    store (W1), keyed by canonical_label (fall back to color+label via
    resolve_key() when canonical_label hasn't been populated yet).

    `blocks` is the same list passed into sorter.sort_blocks(blocks); each
    result dict in results["placed"]/results["skipped"] carries the same
    color/x/y sort_blocks() read off of one of those blocks, so we match
    them back up by that (color, x, y) key to recover the LocatedShape
    (and therefore its canonical_label / label) for each outcome.
    """
    by_xy = {(b.color, b.x, b.y): b for b in blocks}

    for entry in results.get("placed", []):
        block = by_xy.get((entry.get("color"), entry.get("x"), entry.get("y")))
        key = resolve_key(block) if block is not None else (entry.get("color") or "unknown")
        store.record_attempt(
            key,
            xy=(entry.get("x", 0.0), entry.get("y", 0.0)),
            grasp_success=True,
            placement_success=True,
            plan_params={"bin": entry.get("bin")},
        )

    for entry in results.get("skipped", []):
        block = by_xy.get((entry.get("color"), entry.get("x"), entry.get("y")))
        key = resolve_key(block) if block is not None else (entry.get("color") or "unknown")
        store.record_attempt(
            key,
            xy=(entry.get("x", 0.0), entry.get("y", 0.0)),
            grasp_success=False,
            placement_success=False,
            failure_type=entry.get("error"),
        )


async def main() -> None:
    load_dotenv()
    # Shared client: pay the ~4s cloud handshake once, reuse it.
    machine = await get_machine()
    try:
        arm = ArmComponent(machine)
        gripper = GripperComponent(machine)
        vision = VisionComponent(machine)
        sorter = PickPlace(arm, gripper)

        # NOTE: perception here stays LOCAL on purpose. This is the color-sort
        # path (needs LocatedShape.color + shape.box), but the on-machine
        # grasp-service detections() returns label/xyz/score with NO color --
        # routing through it would silently break color binning. The fast
        # on-machine path is wired into the general pick flow (run_pick_and_place)
        # instead. Connection reuse above is the win that applies here.

        print("Moving to home...")
        await arm.go_home()
        await gripper.open()

        print("Detecting red / yellow blocks...")
        blocks = await vision.locate_blocks()
        plan = [
            {
                "color": b.color,
                "bin": COLOR_BINS.get(b.color),
                "xy_mm": [round(b.x, 1), round(b.y, 1)],
                "pick_z_mm": MIN_Z,
                "bbox": b.shape.box if b.shape else None,
            }
            for b in pick_order(blocks)
        ]
        print(json.dumps({"count": len(plan), "plan": plan}, indent=2))
        if not plan:
            print("No red or yellow blocks found.")
            return

        results = await sorter.sort_blocks(blocks)
        print(json.dumps(results, indent=2))

        experience = ExperienceStore()
        record_sort_results(experience, blocks, results)

        await arm.go_home()
    finally:
        await close_machine()


if __name__ == "__main__":
    asyncio.run(main())
