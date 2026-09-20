#!/usr/bin/env python
"""run_pick_and_place.py -- the polished, real-arm-ready pick-and-place
entry point: connect -> perceive -> plan -> execute (with memory/retries)
-> summarize, using the FAST planner end to end (no per-call RRT).

    .venv/bin/python run_pick_and_place.py                       # live robot
    .venv/bin/python run_pick_and_place.py --goal "clear a path to the red block"
    .venv/bin/python run_pick_and_place.py --dry-run              # offline, no Viam connection

WHAT THIS TIES TOGETHER
------------------------
  1. `components.connection.connect_machine` -- connect to the real Viam
     machine (skipped entirely under `--dry-run`; see below).
  2. `components.vision.VisionComponent.locate_blocks` -- perceive the
     scene from the real camera (color-based red/yellow blocks, matching
     `sort_blocks.py`'s default detector; `--detector zeroshot`/`vlm` swap
     in the other two `VisionComponent` detectors).
  3. `components.fast_planner.make_local_ik_fn` -- the fast planner's real
     IK source (`components/ik.py`: numerical IK via `ikpy` against the
     bundled `assets/xarm5.urdf`), since the Viam SDK pinned here exposes
     no `compute_inverse_kinematics`.
  4. `components.pickplace.PickPlace(..., ik_fn=ik_fn, use_fast_planner=True)`
     and `components.retry.RetryController(..., ik_fn=ik_fn)` -- POSITION
     CONTROL throughout (`move_to_joint_positions`), with the pick itself
     driven by `fast_planner`'s deterministic up/over/down joint plan
     instead of three `move_to_position` (RRT) calls per pick.
  5. `components.pipeline.run_task` -- the full enrich -> seed -> plan
     (rule-based by default) -> execute-with-retries -> record loop,
     including declutter (`components.declutter`) and the self-improving
     `components.experience_store.ExperienceStore`.
  6. A clean per-object summary: picked / decluttered / skipped, retries
     and escalations, and the calibrated plan the experience store has
     learned for each object so far.

`--dry-run` swaps in a small synthetic scene and a mock arm/gripper (no
Viam connection, no camera, no robot) so this whole loop -- perceive ->
plan -> fast pick via local IK -> record -- is OFFLINE-verifiable. The
`--dry-run` experience store is a temp file (never touches
`data/experience.json`), so repeated dry runs don't accumulate state.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# Runnable from any cwd (mirrors scripts/*.py's own sys.path guard).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import fast_planner
from components.assistant import deliver_to_person
from components.audio import AudioIO, ScriptedAudio
from components.constants import MIN_Z, TRAVEL_Z
from components.declutter import MoveAside, plan_declutter
from components.active_perception import refine_uncertain
from components.experience_store import ExperienceStore
from components.pickplace import PickPlace
from components.pipeline import (
    StepResult,
    TaskContext,
    enrich_and_seed,
    execute_plan_with_memory,
    plan_task,
    run_task,
)
from components.point import resolve_pointing_target
from components.retry import RetryController
from components.shapes import LocatedShape

DEFAULT_GOAL = "sort the blocks"


# ---------------------------------------------------------------------------
# IK wiring: the fast planner's real, offline-capable IK source
# ---------------------------------------------------------------------------


def build_fast_ik_fn(urdf_path: Optional[str] = None) -> fast_planner.IkFn:
    """The one IK source this whole script's fast path runs on: local
    numerical IK (components/ik.py, via ikpy) against the bundled xArm5
    URDF (assets/xarm5.urdf by default). See components/ik.py's module
    docstring for why: the Viam SDK pinned in this repo has no
    `compute_inverse_kinematics`, so `fast_planner.make_ik_fn(arm)` alone
    always raises -- this is what actually makes `use_fast_planner=True` /
    `RetryController(ik_fn=...)` usable end to end, live or offline.
    """
    return fast_planner.make_local_ik_fn(urdf_path)


# ---------------------------------------------------------------------------
# Declutter adapter for RetryController's escalation ladder
# ---------------------------------------------------------------------------


async def _retry_declutter(target: Any, all_objects: Sequence[Any], pickplace: PickPlace) -> None:
    """`RetryController`'s "declutter" escalation level wants a callable
    that just clears blockers around `target` (it makes its own next pick
    attempt itself, right after) -- so this runs `plan_declutter`'s
    MoveAside steps only, via `PickPlace._move_aside` (the same primitive
    `PickPlace.pick_with_declutter` uses), without also picking `target`
    itself (that would race with RetryController's own subsequent attempt).
    """
    plan = plan_declutter(target, all_objects)
    for action in plan.actions:
        if isinstance(action, MoveAside):
            await pickplace._move_aside(action.obj, action.to_xy)


# ---------------------------------------------------------------------------
# --active-perception opt-in: UQ-triggered center-and-zoom re-look
# (components/active_perception.py) inserted AFTER enrich_and_seed (W2/W3/W1)
# and BEFORE planning (W5). Additive: default (no --active-perception) still
# calls components.pipeline.run_task exactly as before.
# ---------------------------------------------------------------------------


class _PerceiveVision:
    """Adapter satisfying `active_perception.refine_uncertain`'s `vision`
    duck-type contract (one coroutine, `locate_shapes()`), backed by
    whichever detector `--detector`/`--dry-run` selected -- reusing the
    exact perceive() coroutine so a zoom re-look re-runs the SAME detector
    that produced the original scene, just from the arm's new (closer,
    centered) pose, instead of switching detectors mid-task."""

    def __init__(self, perceive_fn: Any) -> None:
        self._perceive_fn = perceive_fn

    async def locate_shapes(self) -> List[LocatedShape]:
        return await self._perceive_fn()


async def _run_task_with_active_perception(
    goal: str, scene: Sequence[Any], ctx: TaskContext, vision_relook: Any
) -> List[StepResult]:
    """Same loop as `components.pipeline.run_task`, with one addition: a
    UQ-triggered active-perception re-look between enrich/seed and planning.
    Composes the exact same pieces `run_task` does (`enrich_and_seed`,
    `plan_task`, `execute_plan_with_memory`) plus
    `active_perception.refine_uncertain` -- no rewriting of any of them."""
    enriched = enrich_and_seed(scene, image=None, store=ctx.store)
    refined = await refine_uncertain(enriched, ctx.arm, vision_relook)
    plan = plan_task(goal, refined)
    return await execute_plan_with_memory(plan, refined, ctx)


# ---------------------------------------------------------------------------
# Live machine wiring
# ---------------------------------------------------------------------------


@dataclass
class Runtime:
    """Everything main() needs to run one task, plus how to tear it down."""

    ctx: TaskContext
    perceive: Any  # async () -> List[LocatedShape]
    on_start: Any  # async () -> None  (home + open, before perceiving)
    on_finish: Any  # async () -> None  (home, after the task)
    close: Any  # async () -> None  (release the machine connection, if any)
    vision_relook: Any = None  # async duck-type: locate_shapes() -> List[LocatedShape]
    # (--active-perception's re-detect dependency; None-safe default so
    # existing Runtime construction elsewhere/tests is unaffected).


async def _build_live_runtime(ik_fn: fast_planner.IkFn, detector: str, store_path: Optional[str]) -> Runtime:
    from components.arm import ArmComponent
    from components.connection import close_machine, get_machine
    from components.gripper import GripperComponent
    from components.vision import VisionComponent

    # Shared client: pay the ~4s cloud handshake ONCE and reuse it (per-call RTT
    # is only ~36ms once connected). See components/connection.get_machine.
    machine = await get_machine()
    arm = ArmComponent(machine)
    gripper = GripperComponent(machine)
    vision = VisionComponent(machine)
    pickplace = PickPlace(arm, gripper, ik_fn=ik_fn, use_fast_planner=True)
    store = ExperienceStore(path=store_path) if store_path else ExperienceStore()
    retry = RetryController(
        arm,
        gripper,
        declutter=lambda target, objs: _retry_declutter(target, objs, pickplace),
        ik_fn=ik_fn,
    )
    ctx = TaskContext(arm=arm, gripper=gripper, pickplace=pickplace, store=store, retry=retry)

    # Opt-in fast perception: when ON_MACHINE_PERCEPTION=1 and the grasp-service
    # module is deployed on the machine, perceive via a ~36ms do_command instead
    # of pulling images (~2s) across the relay. Falls back to the local detector
    # on ANY failure, so default (unset) behavior is unchanged.
    from components.perception_router import build_router
    router = build_router(machine)

    async def _local_perceive() -> List[LocatedShape]:
        if detector == "zeroshot":
            return await vision.locate_objects_zeroshot()
        if detector == "vlm":
            return await vision.locate_objects_vlm()
        return await vision.locate_blocks()

    async def _perceive() -> List[LocatedShape]:
        return await router.detections(_local_perceive)

    async def _on_start() -> None:
        print("Moving to home...")
        await arm.go_home()
        await gripper.open()

    async def _on_finish() -> None:
        await arm.go_home()

    async def _close() -> None:
        # Clears the shared singleton so a later get_machine() reconnects cleanly.
        await close_machine()

    return Runtime(
        ctx=ctx,
        perceive=_perceive,
        on_start=_on_start,
        on_finish=_on_finish,
        close=_close,
        vision_relook=_PerceiveVision(_perceive),
    )


# ---------------------------------------------------------------------------
# Offline (--dry-run) wiring: no Viam connection, no camera, no robot
# ---------------------------------------------------------------------------


class MockArm:
    """Offline stand-in for ArmComponent: records every call, accepts the
    exact same primitives the fast planner / PickPlace / RetryController
    drive against a real arm (position control -- move_to_joints /
    go_to / go_home -- plus the legacy move_to_position PickPlace's
    declutter _move_aside still uses)."""

    def __init__(self) -> None:
        self.joint_moves: List[list] = []
        self.cartesian_moves: List[Tuple[float, float, float]] = []
        self.bin_visits: List[str] = []
        self.homes = 0
        self._xy = (250.0, 50.0)

    async def get_joint_positions(self, timeout: float = 10) -> List[float]:
        return [0.0] * 5

    async def get_end_position(self, timeout: float = 10):
        @dataclass
        class _Pose:
            x: float
            y: float
            z: float = TRAVEL_Z

        return _Pose(x=self._xy[0], y=self._xy[1])

    async def go_home(self, timeout: float = 60) -> None:
        self.homes += 1
        self._xy = (250.0, 50.0)

    async def move_to_joints(self, joints: list, timeout: float = 60) -> None:
        self.joint_moves.append(list(joints))

    async def move_to_joint_positions(self, joints: list, timeout: float = 30) -> None:
        self.joint_moves.append(list(joints))

    async def go_to(self, name: str, timeout: float = 60) -> None:
        self.bin_visits.append(name)

    async def move_to_position(self, x: float, y: float, z: float, **kw) -> None:
        self.cartesian_moves.append((x, y, z))
        self._xy = (x, y)


class MockGripper:
    """Offline stand-in for GripperComponent: always grabs successfully so
    --dry-run exercises the full happy path end to end."""

    def __init__(self, grab_result: bool = True) -> None:
        self.grab_result = grab_result
        self.opens = 0
        self.grabs = 0

    async def open(self, timeout: float = 10) -> None:
        self.opens += 1

    async def grab(self, timeout: float = 10) -> bool:
        self.grabs += 1
        return self.grab_result

    async def is_holding_something(self) -> bool:
        return self.grab_result


def _mock_scene() -> List[LocatedShape]:
    """A small synthetic tabletop scene for --dry-run: two blocks inside
    the taught workspace, at positions verified reachable by the local IK
    solver at both MIN_Z and TRAVEL_Z (see tests/test_ik.py) -- exercises
    the real fast-planner/local-IK path, not just the planning logic.

    Scores are set so `--dry-run --active-perception` has something to
    demonstrate offline: the yellow block's low score pushes its UQ
    `.difficulty` (components/uq.py) above the default 0.5 "uncertain"
    threshold (an ambiguous low-confidence read, like a VLM torn between
    "cup"/"mug"), while the red block's high score keeps it comfortably
    confident -- so the re-look triggers for exactly one of the two."""
    return [
        LocatedShape(label="red block", x=300.0, y=-100.0, z=MIN_Z, color="red", score=0.93),
        LocatedShape(label="yellow block", x=250.0, y=50.0, z=MIN_Z, color="yellow", score=0.2),
    ]


class _MockZoomVision:
    """Offline stand-in for the eye-in-hand re-detect call
    `active_perception.refine_uncertain` makes after zooming in: reports the
    same synthetic objects but with a boosted score, mimicking a real
    detector doing better on the bigger/centered pixels a zoom-in provides.
    Lets `--dry-run --active-perception` demonstrate a real re-look +
    confidence improvement end to end without a camera/model."""

    def __init__(self, scene_fn: Any, boost: float = 0.55) -> None:
        self._scene_fn = scene_fn
        self._boost = boost

    async def locate_shapes(self) -> List[LocatedShape]:
        boosted = self._scene_fn()
        for obj in boosted:
            if obj.score is not None:
                obj.score = min(0.99, obj.score + self._boost)
        return boosted


def _build_dry_run_runtime(ik_fn: fast_planner.IkFn, store_path: str) -> Runtime:
    arm = MockArm()
    gripper = MockGripper()
    pickplace = PickPlace(arm, gripper, ik_fn=ik_fn, use_fast_planner=True)
    store = ExperienceStore(path=store_path)
    retry = RetryController(
        arm,
        gripper,
        declutter=lambda target, objs: _retry_declutter(target, objs, pickplace),
        ik_fn=ik_fn,
    )
    ctx = TaskContext(arm=arm, gripper=gripper, pickplace=pickplace, store=store, retry=retry)

    async def _perceive() -> List[LocatedShape]:
        return _mock_scene()

    async def _on_start() -> None:
        print("[dry-run] Moving to home...")
        await arm.go_home()
        await gripper.open()

    async def _on_finish() -> None:
        await arm.go_home()

    async def _close() -> None:
        return None

    return Runtime(
        ctx=ctx,
        perceive=_perceive,
        on_start=_on_start,
        on_finish=_on_finish,
        close=_close,
        vision_relook=_MockZoomVision(_mock_scene),
    )


# ---------------------------------------------------------------------------
# Deliver-to-person opt-in: routes ONE matched target through
# components.assistant.deliver_to_person (grasp via the same ctx.retry state
# machine run_task/_run_pick uses, then components.handover.HandoverSkill)
# instead of the normal run_task -> _run_pick -> color-bin plan. Additive:
# the default (bin sort via run_task) is completely unchanged when --deliver
# isn't passed.
# ---------------------------------------------------------------------------


async def _run_deliver(
    scene: List[LocatedShape], ctx: TaskContext, goal: str, *, dry_run: bool
) -> List[StepResult]:
    """Enrich/seed the scene (same W2/W3/W1 step run_task itself does), pick
    the target the goal refers to (components.point.resolve_pointing_target,
    reusing the canonicalize label-matching the rest of the system keys on;
    falls back to the first detected object if the goal names no known
    color/label), then hand it to a person via deliver_to_person instead of
    binning it."""
    enriched = enrich_and_seed(scene, image=None, store=ctx.store)
    target = resolve_pointing_target(goal, enriched) or (enriched[0] if enriched else None)
    if target is None:
        print("[deliver] no objects detected to deliver.")
        return []

    label = target.canonical_label or target.color or target.label
    print(f"[deliver] target: {label}  xy=({target.x:.1f}, {target.y:.1f})")

    # Dry-run demonstrates a completed hand-over end to end (scripted "yes");
    # a live run uses the real AudioIO stub (asks, defaults to holding until
    # real audio_in/STT is wired up -- see components/audio.py).
    audio = ScriptedAudio(answers=[True]) if dry_run else AudioIO()
    result = await deliver_to_person(target, ctx, audio=audio, scene=enriched)
    if isinstance(audio, ScriptedAudio):
        # ScriptedAudio (unlike the base AudioIO stub) logs instead of
        # printing -- surface the hand-over dialogue so --dry-run --deliver
        # visibly shows the hand-over, not just the summary line. HandoverSkill
        # speaks once before asking and once after, so said[0] -> asked[0] ->
        # said[1:] reproduces the real order.
        if audio.said:
            print(f"[deliver]   \U0001F50A {audio.said[0]}")
        for q in audio.asked:
            print(f"[deliver]   \U0001F50A? {q}")
        for s in audio.said[1:]:
            print(f"[deliver]   \U0001F50A {s}")
    return [result]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: Sequence[StepResult], ctx: TaskContext) -> None:
    print("\n=== pick-and-place summary ===")
    if not results:
        print("  (no steps planned)")
        return

    placed = [r for r in results if r.skill == "pick" and r.success]
    declutter_skills = {"declutter", "declutter:move_aside", "declutter:pick", "move_aside"}
    decluttered = [r for r in results if r.skill in declutter_skills]
    skipped = [r for r in results if not r.success]

    for r in results:
        status = "OK" if r.success else "FAILED"
        line = f"  [{status}] {r.skill:22s} {r.key}"
        if r.attempts:
            line += f"  attempts={r.attempts}"
        if r.escalations:
            line += f"  escalations={list(r.escalations)}"
        if r.failure_type:
            line += f"  reason={r.failure_type}"
        print(line)

        calibrated = ctx.store.get_calibration(r.key)
        print(
            f"      calibrated_plan: xy_offset={tuple(calibrated['xy_offset'])} "
            f"pick_z_offset={calibrated['pick_z_offset']:.2f} "
            f"retry_budget={calibrated['retry_budget']}"
        )

    print(
        f"\n  {len(placed)} placed, {len(decluttered)} declutter step(s), "
        f"{len(skipped)} skipped/failed (of {len(results)} total steps)"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goal", default=DEFAULT_GOAL, help=f"task goal (default: {DEFAULT_GOAL!r})")
    parser.add_argument(
        "--detector",
        choices=("color", "zeroshot", "vlm"),
        default="color",
        help="live-run perception path (ignored under --dry-run, which uses a synthetic scene)",
    )
    parser.add_argument("--urdf", default=None, help="override the bundled xArm5 URDF path used for local IK")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run fully offline: synthetic scene + mock arm/gripper, no Viam connection",
    )
    parser.add_argument(
        "--deliver",
        action="store_true",
        help=(
            "hand the goal-matched target to a person (components.assistant."
            "deliver_to_person: grasp via the normal retry path, then "
            "components.handover.HandoverSkill) instead of sorting into a "
            "color bin. The target is matched from --goal (e.g. "
            "--goal 'bring me the red block'); default (no --deliver) is "
            "unchanged bin-sort behavior."
        ),
    )
    parser.add_argument(
        "--active-perception",
        action="store_true",
        help=(
            "UQ-triggered center-and-zoom re-look (components.active_perception."
            "refine_uncertain): for objects UQ flags as uncertain "
            "(.difficulty >= 0.5), move the wrist camera closer + centered over "
            "the object, re-detect, and keep the more-confident reading -- runs "
            "after enrich/seed and before planning, in both --dry-run and live "
            "runs (not the --deliver path). Default (no flag) is unchanged "
            "behavior."
        ),
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    ik_fn = build_fast_ik_fn(args.urdf)

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _build_dry_run_runtime(ik_fn, store_path=str(Path(tmp) / "experience.json"))
            await runtime.on_start()
            print("[dry-run] Perceiving scene (synthetic)...")
            scene = await runtime.perceive()
            print(f"[dry-run] {len(scene)} object(s) detected: "
                  f"{[(o.color, round(o.x, 1), round(o.y, 1)) for o in scene]}")
            if args.deliver:
                results = await _run_deliver(scene, runtime.ctx, args.goal, dry_run=True)
            elif args.active_perception:
                results = await _run_task_with_active_perception(
                    args.goal, scene, runtime.ctx, runtime.vision_relook
                )
            else:
                results = await run_task(args.goal, scene, image=None, ctx=runtime.ctx)
            print_summary(results, runtime.ctx)
            await runtime.on_finish()
            await runtime.close()
        return 0

    from dotenv import load_dotenv

    load_dotenv()
    runtime = await _build_live_runtime(ik_fn, args.detector, store_path=None)
    try:
        await runtime.on_start()
        print("Perceiving scene...")
        scene = await runtime.perceive()
        if not scene:
            print("No objects detected.")
            return 0
        print(f"{len(scene)} object(s) detected: "
              f"{[(o.color, round(o.x, 1), round(o.y, 1)) for o in scene]}")
        if args.deliver:
            results = await _run_deliver(scene, runtime.ctx, args.goal, dry_run=False)
        elif args.active_perception:
            results = await _run_task_with_active_perception(
                args.goal, scene, runtime.ctx, runtime.vision_relook
            )
        else:
            results = await run_task(args.goal, scene, image=None, ctx=runtime.ctx)
        print_summary(results, runtime.ctx)
        await runtime.on_finish()
        return 0
    finally:
        await runtime.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
