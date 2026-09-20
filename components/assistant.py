"""Deliver-to-person (assistant): grasp a target through the existing
retry/fast pick path, then hand it to a person instead of dropping it in a
color bin.

Composes -- does NOT rewrite -- the two skills this sits between:

    deliver_to_person(target, ctx, audio=...)
        = grasp(target)              components.retry.RetryController, the
                                      exact state machine
                                      components.pipeline._run_pick already
                                      drives for the grasp portion of a pick
        -> HandoverSkill.run(label)  components.handover.HandoverSkill,
                                      unmodified: present + confirm + release

`components.pipeline._run_pick` itself is untouched; this module is an
ADDITIVE alternative "back half" (hand-over instead of a bin drop) that
plugs into the same `TaskContext` / `ExperienceStore` recording shape it
uses, so a caller can route a specific target through either path without
either module knowing about the other.

Everything is dependency-injected (ctx.arm/ctx.gripper/ctx.retry, audio) and
async, so it runs OFFLINE with mocks -- no robot, no audio hardware.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from components.audio import AudioIO
from components.canonicalize import canonicalize_label
from components.experience_store import resolve_key
from components.handover import HandoverSkill
from components.pipeline import StepResult, TaskContext


def _label_of(target: Any) -> str:
    return (
        getattr(target, "canonical_label", "")
        or canonicalize_label(getattr(target, "label", ""))
        or getattr(target, "color", "")
        or "object"
    )


async def deliver_to_person(
    target: Any,
    ctx: TaskContext,
    *,
    audio: Optional[AudioIO] = None,
    scene: Optional[Sequence[Any]] = None,
    handover_cls: type = HandoverSkill,
) -> StepResult:
    """Grasp `target` via `ctx.retry` (identical call shape to
    `components.pipeline._run_pick`'s grasp step: same difficulty budgeting,
    same calibrated-plan bias, same declutter escalation), then -- on a
    successful grasp -- hand it to a person via `HandoverSkill.run()` instead
    of `_run_pick`'s color-bin drop. Records the outcome to `ctx.store`
    exactly like `_run_pick` does (so the experience store's calibration
    keeps learning regardless of which "back half" a pick used).

    Returns a `StepResult` with `skill="deliver"`; `success` is True only if
    the object was actually released into the person's hand (a declined or
    lost hand-over is `success=False`, matching `HandoverSkill.run`'s own
    True/False contract).
    """
    key = resolve_key(target)
    calibrated = ctx.store.get_calibration(key)
    difficulty = getattr(target, "difficulty", None)

    outcome = await ctx.retry.run(
        target,
        difficulty=difficulty,
        calibrated_plan=calibrated,
        all_objects=list(scene or []),
    )

    delivered = False
    handover_failure: Optional[str] = None
    if outcome.success:
        handover = handover_cls(ctx.arm, ctx.gripper, audio)
        delivered = await handover.run(_label_of(target))
        if not delivered:
            handover_failure = "handover_declined_or_lost"

    ctx.store.record_attempt(
        key,
        xy=(float(target.x), float(target.y)),
        grasp_success=outcome.success,
        placement_success=delivered if outcome.success else None,
        failure_type=outcome.failure_type or handover_failure,
        plan_params={
            "difficulty": difficulty,
            "xy_offset": (calibrated or {}).get("xy_offset", (0.0, 0.0)),
            "pick_z_offset": (calibrated or {}).get("pick_z_offset", 0.0),
            "mode": "deliver",
        },
    )

    go_home = getattr(ctx.arm, "go_home", None)
    if callable(go_home):
        await go_home()

    return StepResult(
        skill="deliver",
        key=key,
        success=bool(delivered),
        attempts=outcome.attempts,
        escalations=list(outcome.escalations),
        failure_type=outcome.failure_type or handover_failure,
    )
