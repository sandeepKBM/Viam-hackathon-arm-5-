"""Integration merge (W1-W6): the end-to-end perceive -> enrich -> plan ->
execute-with-memory loop that ties the six workstreams into one flow, wired
to color-sort's manipulation stack.

    enrich   (W2/W3)  uq.enrich       -> .canonical_label + .score + .difficulty
    seed     (W1)     store.seed       -> attaches .history from past attempts
    plan     (W5)     policy.plan_task -> a validated, safety-checked skill sequence
    execute  (W4)     pick calls run through RetryController (difficulty-budgeted,
                      calibrated-plan-biased) directly against arm/gripper, then
                      placed via arm.go_to(bin)+gripper.open(); declutter/
                      move_aside/place/handoff calls run through their skill
                      adapters (components/skills.py) against color-sort's
                      components.pickplace.PickPlace
    record   (W1)     store.record_attempt after each step -> updates calibration
    budget   (W6)     the UQ sample count `n` can be bounded from data/timing.json

Everything is dependency-injected (arm, gripper, pickplace, store, retry,
detector) so this runs OFFLINE with mocks -- no robot, no camera. The same
TaskContext, populated with color-sort's real ArmComponent/GripperComponent/
PickPlace/ExperienceStore, runs it on hardware.

ADAPTATION NOTE (color-sort's PickPlace)
------------------------------------------
`RetryController` (components/retry.py) never calls `PickPlace` -- it drives
`arm`/`gripper` directly for the grasp+lift, the same low-level primitives
`PickPlace.pick_and_place` itself uses internally. That means `_run_pick`
below is unchanged in shape from the source this was ported from: on a
successful grasp it completes the place step itself via
`ctx.arm.go_to(bin_name)` + `ctx.gripper.open()` -- both of which exist on
color-sort's `components.arm.ArmComponent` / `components.gripper.GripperComponent`
with the same signatures, so no adaptation was needed here. `ctx.pickplace`
(color-sort's real `PickPlace`) is used only by the declutter/move_aside/
place/handoff branch in `execute_plan_with_memory`, via the already-adapted
skill adapters in components/skills.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from components import uq
from components.constants import COLOR_BINS
from components.experience_store import ExperienceStore, resolve_key
from components.policy import Planner, plan_task
from components.retry import RetryController
from components.skills import execute_call


@dataclass
class TaskContext:
    """The injected runtime the loop executes against. On hardware: real
    components.arm.ArmComponent / components.gripper.GripperComponent /
    components.pickplace.PickPlace. In tests: mocks."""

    arm: Any
    gripper: Any
    pickplace: Any                       # color-sort's PickPlace, for declutter/move_aside/place/handoff adapters
    store: ExperienceStore
    retry: RetryController
    detector: Optional[Callable[[Any], Any]] = None  # async re-detect (retry/uq)


@dataclass
class StepResult:
    skill: str
    key: str
    success: bool
    attempts: int = 0
    escalations: List[str] = field(default_factory=list)
    failure_type: Optional[str] = None


def enrich_and_seed(
    objects: Sequence[Any],
    image: Any,
    store: ExperienceStore,
    *,
    detector_fn: Optional[Callable[[Any], Any]] = None,
    n: Optional[int] = None,
) -> Sequence[Any]:
    """W2/W3 enrich (canonical_label + score + difficulty) then W1 seed
    (attach .history). Mutates the objects in place and returns them. `n`
    (UQ sample count) defaults to uq's own default when None -- pass a value
    derived from the W6 timing budget to bound augmentation cost."""
    if n is None:
        uq.enrich(objects, image=image, detector_fn=detector_fn)
    else:
        uq.enrich(objects, image=image, detector_fn=detector_fn, n=n)
    for obj in objects:
        store.seed(obj)
    return objects


async def _run_pick(target: Any, scene: Sequence[Any], ctx: TaskContext) -> StepResult:
    """Execute one pick through the retry state machine (W4), biased by the
    object's calibrated plan (W1) and budgeted by its difficulty (W3); on a
    successful grasp, place into the color bin (color-sort's
    components.constants.COLOR_BINS); record the outcome (W1)."""
    key = resolve_key(target)
    calibrated = ctx.store.get_calibration(key)
    difficulty = getattr(target, "difficulty", None)

    outcome = await ctx.retry.run(
        target,
        difficulty=difficulty,
        calibrated_plan=calibrated,
        all_objects=list(scene),
    )

    placement_success: Optional[bool] = None
    if outcome.success:
        # RetryController grasped and lifted the object to travel height;
        # complete the pick-and-place by dropping it in its color bin
        # (color-sort's ArmComponent.go_to(name) drives to the taught
        # bin1/bin2/dropoff/handoff joint pose).
        bin_name = COLOR_BINS.get(getattr(target, "color", None))
        if bin_name is not None:
            await ctx.arm.go_to(bin_name)
            await ctx.gripper.open()
            placement_success = True

    ctx.store.record_attempt(
        key,
        xy=(float(target.x), float(target.y)),
        grasp_success=outcome.success,
        placement_success=placement_success,
        failure_type=outcome.failure_type,
        plan_params={
            "difficulty": difficulty,
            "xy_offset": (calibrated or {}).get("xy_offset", (0.0, 0.0)),
            "pick_z_offset": (calibrated or {}).get("pick_z_offset", 0.0),
        },
    )

    go_home = getattr(ctx.arm, "go_home", None)
    if callable(go_home):
        await go_home()

    return StepResult(
        skill="pick",
        key=key,
        success=outcome.success,
        attempts=outcome.attempts,
        escalations=list(outcome.escalations),
        failure_type=outcome.failure_type,
    )


async def execute_plan_with_memory(
    plan: Sequence[Any], scene: Sequence[Any], ctx: TaskContext
) -> List[StepResult]:
    """Run a validated plan (from policy.plan_task): route `pick` through the
    retry state machine, and declutter/move_aside/place/handoff through their
    validated skill adapters (against color-sort's PickPlace); record every
    step's outcome to the experience store."""
    results: List[StepResult] = []
    for call in plan:
        if call.skill == "pick":
            target = call.params.object
            try:
                results.append(await _run_pick(target, scene, ctx))
            except Exception as exc:
                # A hard failure below RetryController's own success/failure
                # return isn't fatal to the run -- same resilience policy as
                # the declutter/move_aside/place/handoff branch below, so one
                # bad target can't abort an otherwise-valid multi-object plan.
                key = resolve_key(target)
                failure_type = f"pick_error: {exc}"
                ctx.store.record_attempt(
                    key,
                    xy=(float(target.x), float(target.y)),
                    grasp_success=False,
                    failure_type=failure_type,
                    plan_params={"skill": "pick"},
                )
                results.append(
                    StepResult(skill="pick", key=key, success=False, failure_type=failure_type)
                )
            continue

        # declutter / move_aside / place / handoff: use the already-validated
        # skill adapter (components/skills.py), driven against ctx.pickplace.
        target = getattr(call.params, "target", None) or getattr(call.params, "object", None)
        key = resolve_key(target) if target is not None else call.skill
        failure_type: Optional[str] = None
        try:
            ret = await execute_call(call, ctx.pickplace)
            success = ret is not False  # adapters return bool or None(ok)
        except Exception as exc:  # an adapter failure isn't fatal to the run
            success = False
            failure_type = f"{call.skill}_error: {exc}"

        if target is not None:
            ctx.store.record_attempt(
                key,
                xy=(float(target.x), float(target.y)),
                grasp_success=success,
                failure_type=failure_type,
                plan_params={"skill": call.skill},
            )
        results.append(
            StepResult(skill=call.skill, key=key, success=success, failure_type=failure_type)
        )
    return results


async def run_task(
    goal: str,
    objects: Sequence[Any],
    image: Any,
    ctx: TaskContext,
    *,
    planner: Optional[Planner] = None,
    n: Optional[int] = None,
) -> List[StepResult]:
    """The full loop: enrich + seed the scene (W2/W3/W1), plan the task
    (W5, `planner` defaults to the offline rule-based stub), then execute with
    memory + retries (W4/W1). Returns a per-step result list."""
    scene = enrich_and_seed(objects, image, ctx.store, detector_fn=ctx.detector, n=n)
    plan = plan_task(goal, scene, planner=planner)
    return await execute_plan_with_memory(plan, scene, ctx)
