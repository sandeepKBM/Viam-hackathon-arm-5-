from typing import List, Optional

from components import fast_planner
from components.arm import ArmComponent
from components.constants import COLOR_BINS, HOME_POSE, MIN_Z, PICK_ORIENTATION, TRAVEL_Z
from components.declutter import MoveAside, Pick, plan_declutter
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
    def __init__(
        self,
        arm: ArmComponent,
        gripper: GripperComponent,
        *,
        ik_fn: Optional[fast_planner.IkFn] = None,
        use_fast_planner: bool = False,
    ) -> None:
        """`ik_fn` and `use_fast_planner` are optional, additive knobs for
        the deterministic fast-path planner (components/fast_planner.py) --
        both default to the prior behavior (planned `_above()` cartesian
        moves via `arm.move_to_position`, one BiRRT plan per call), so
        existing callers (`PickPlace(arm, gripper)`) are unaffected.

        - `ik_fn`: a synchronous `pose(dict) -> joints` callable (a mock in
          tests, or `components.fast_planner.make_ik_fn(arm)`'s real
          wrapper called once synchronously -- see `pick_and_place_fast`).
          Required for the fast path; with none injected the fast path
          degrades with a clear `RuntimeError` rather than silently
          falling back or doing something wrong.
        - `use_fast_planner`: when True (and `ik_fn` is set), the normal
          `pick_and_place()` entry point itself delegates to
          `pick_and_place_fast()`, so `pick_with_declutter`/`sort_blocks`
          get the fast path too without any other code change. Defaults
          to False -- opt-in only.
        """
        self.arm = arm
        self.gripper = gripper
        self.ik_fn = ik_fn
        self.use_fast_planner = use_fast_planner

    async def _above(self, x: float, y: float, z: float) -> None:
        await self.arm.move_to_position(x, y, z, timeout=60, **PICK_ORIENTATION)

    async def _current_xy(self) -> tuple:
        """Best-effort current end-effector (x, y), used only as the
        "lift straight up at current XY" starting point for the fast
        planner. Falls back to the taught home pose's xy if the arm
        doesn't expose `get_end_position` (e.g. a minimal test double)."""
        get_end = getattr(self.arm, "get_end_position", None)
        if get_end is None:
            return (float(HOME_POSE["x"]), float(HOME_POSE["y"]))
        pose = await get_end()
        return (float(pose.x), float(pose.y))

    async def pick_and_place(self, block: LocatedShape) -> bool:
        if self.use_fast_planner and self.ik_fn is not None:
            return await self.pick_and_place_fast(block)

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

    async def pick_and_place_fast(self, block: LocatedShape) -> bool:
        """Opt-in deterministic fast path (components/fast_planner.py):
        replaces the three planned `_above()` cartesian moves (each a
        fresh BiRRT solve) with a 4-waypoint direct joint-space plan --
        up at current xy, over target, down to pick_z, lift after grab --
        executed with `move_to_joint_positions` only (no
        `move_to_position`, no per-call planning). The bin drop itself
        (`arm.go_to`) is unchanged: it was already joint-space/fast.

        Requires an `ik_fn` to have been injected via the constructor
        (`PickPlace(..., ik_fn=...)`); degrades with a clear
        `RuntimeError` -- not a silent fallback -- if none is available
        (e.g. running fully offline with no mock/real IK wired up).
        """
        if self.ik_fn is None:
            raise RuntimeError(
                "pick_and_place_fast requires PickPlace(..., ik_fn=...) -- no "
                "ik_fn was injected (offline run with no mock/real IK available)"
            )
        bin_name = COLOR_BINS.get(block.color)
        if bin_name is None:
            raise ValueError(f"no bin mapped for color {block.color!r}")
        if not in_workspace(block.x, block.y):
            raise ValueError(
                f"{block.color} at ({block.x:.1f}, {block.y:.1f}) is outside the workspace"
            )

        target_xy = (float(block.x), float(block.y))
        print(
            f"  [fast] pick {block.color} -> {bin_name}  "
            f"xy=({block.x:.1f}, {block.y:.1f}) z={MIN_Z:.1f}"
        )
        await self.gripper.open()
        start_xy = await self._current_xy()
        pick_waypoints = fast_planner.plan_pick_waypoints(
            [], start_xy, target_xy, self.ik_fn
        )
        await fast_planner.fast_move(self.arm, pick_waypoints)
        grabbed = await self.gripper.grab()
        print(f"  grabbed: {grabbed}")
        lift_pose = fast_planner.pick_pose(target_xy[0], target_xy[1], TRAVEL_Z)
        lift_joints = fast_planner.solve_ik(self.ik_fn, lift_pose)
        await fast_planner.fast_move(self.arm, [lift_joints])
        if not grabbed:
            return False
        await self.arm.go_to(bin_name)
        await self.gripper.open()
        print(f"  placed in {bin_name}")
        return True

    async def _move_aside(self, obj: LocatedShape, to_xy: tuple) -> None:
        """Relocate a blocking object to a temp clear-zone using the same
        top-down grab/lift/place primitives as pick_and_place, but without
        touching a color bin. Reuses existing arm/gripper methods, so the
        Z-floor and workspace-safety checks already enforced by
        components.safety (via ArmComponent.move_to_position -> make_pose)
        apply here too.
        """
        x, y = to_xy
        print(
            f"  declutter: move {obj.color or obj.label} blocker "
            f"({obj.x:.1f}, {obj.y:.1f}) -> ({x:.1f}, {y:.1f})"
        )
        await self.gripper.open()
        await self._above(obj.x, obj.y, TRAVEL_Z)
        await self._above(obj.x, obj.y, MIN_Z)
        grabbed = await self.gripper.grab()
        await self._above(obj.x, obj.y, TRAVEL_Z)
        if not grabbed:
            raise RuntimeError(
                f"declutter: failed to grab blocker at ({obj.x:.1f}, {obj.y:.1f})"
            )
        await self._above(x, y, TRAVEL_Z)
        await self._above(x, y, MIN_Z)
        await self.gripper.open()
        await self._above(x, y, TRAVEL_Z)

    async def pick_with_declutter(
        self, target: LocatedShape, all_objects: List[LocatedShape]
    ) -> bool:
        """Declutter-aware pick: plan (pure, offline-testable geometry in
        components.declutter) then execute. If the target's top-down grasp
        is blocked by neighbors, moves each blocker to a clear temp zone
        first, then falls through to the normal pick_and_place for the
        target. Raises components.declutter.DeclutterPlanError if no safe
        temp zone can be found for a blocker.
        """
        plan = plan_declutter(target, all_objects)
        for action in plan.actions:
            if isinstance(action, MoveAside):
                await self._move_aside(action.obj, action.to_xy)
            elif isinstance(action, Pick):
                return await self.pick_and_place(action.obj)
        return False

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
