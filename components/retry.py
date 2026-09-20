"""Preplanned grasp retries (W4).

A bounded recovery state machine that wraps a single pick attempt (the
primitives `gripper.grab()` / `gripper.open()` / `arm.move_to_position()`
already used by `components.pickplace.PickPlace`). On failure it:

  1. re-detects the target (it may have shifted / been bumped), via an
     injected `detector` callback;
  2. re-centers the approach on the fresh detection;
  3. escalates through `nudge -> declutter -> skip`, one level per
     subsequent failure, up to a difficulty-derived retry budget.

Design notes:
  - `LocatedShape.difficulty` (W3, easy=0..tough=1) sets the retry budget
    *up front* -- it is the authoritative signal when available. When it
    isn't, the experience store's `calibrated_plan.retry_budget` (W1) is
    used as a fallback prior.
  - The experience store's `calibrated_plan` (`pick_z_offset`, `xy_offset`,
    `grip_params`) biases the *first* attempt only -- later attempts rely
    on live re-detection instead of a stale historical bias.
  - `arm`, `gripper`, and `detector` are injected dependencies (duck-typed)
    so tests can use plain mocks -- no Viam connection, no robot, no
    camera. Failure signal is `gripper.grab() -> bool` being falsy, and,
    when the gripper exposes it, `gripper.is_holding_something() -> bool`
    also being falsy.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from components import fast_planner
from components.constants import MIN_Z, TRAVEL_Z

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MIN_RETRY_BUDGET = int(os.environ.get("RETRY_MIN_BUDGET", 1))
MAX_RETRY_BUDGET = int(os.environ.get("RETRY_MAX_BUDGET", 4))
DEFAULT_RETRY_BUDGET = int(os.environ.get("RETRY_DEFAULT_BUDGET", 2))

# Deterministic re-center nudge applied at the "nudge" escalation level
# (mm). Small and axis-symmetric -- just enough to shake loose a marginal
# miss without leaving the object's immediate vicinity.
NUDGE_STEP_MM = float(os.environ.get("RETRY_NUDGE_STEP_MM", 5.0))


def difficulty_to_budget(
    difficulty: float,
    min_budget: int = MIN_RETRY_BUDGET,
    max_budget: int = MAX_RETRY_BUDGET,
) -> int:
    """Map a W3 difficulty score in [0, 1] (easy -> tough) linearly onto a
    retry budget in [min_budget, max_budget]. Difficulty is clamped to
    [0, 1] first so an out-of-range score can't blow the budget out."""
    d = max(0.0, min(1.0, float(difficulty)))
    return min_budget + round(d * (max_budget - min_budget))


# ---------------------------------------------------------------------------
# States / escalation ladder
# ---------------------------------------------------------------------------


class RetryState(str, Enum):
    ATTEMPTING = "attempting"
    REDETECTING = "redetecting"
    RECENTERING = "recentering"
    ESCALATING = "escalating"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    EXHAUSTED = "exhausted"


# Escalation severity order used on successive failures within one run.
# "skip" is terminal: it ends the run immediately (deliberately giving up)
# rather than waiting for the budget to run out.
ESCALATION_ORDER: tuple = ("nudge", "declutter", "skip")


@dataclass
class RetryOutcome:
    success: bool
    attempts: int
    state: RetryState
    escalations: List[str] = field(default_factory=list)
    target: Any = None
    failure_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_xy(obj: Any, x: float, y: float) -> Any:
    """Return a copy of `obj` with .x/.y updated, without mutating the
    original (dataclasses.replace for real dataclasses -- e.g.
    components.shapes.LocatedShape -- shallow copy + setattr otherwise, so
    plain test doubles work too)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, x=x, y=y)
    clone = copy.copy(obj)
    clone.x = x
    clone.y = y
    return clone


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class RetryController:
    """Bounded recovery state machine wrapping a single pick.

    Dependencies (all injected so tests use mocks):
      arm       -- needs async move_to_position(x, y, z, **kwargs)
      gripper   -- needs async open(), async grab(**grip_params) -> bool;
                   optionally async is_holding_something() -> bool
      detector  -- optional async callable(target) -> updated target-like
                   object (or None if nothing new was seen -- the previous
                   target is kept). Models "re-detect, object may have
                   shifted".
      declutter -- optional async callable(target, all_objects) -> Any,
                   invoked at the "declutter" escalation level to clear a
                   blocker before the next attempt (e.g.
                   PickPlace.pick_with_declutter's planning step). Errors
                   are swallowed (logged) so a declutter failure doesn't
                   abort the whole retry run -- the next attempt is still
                   made and may simply fail again.
    """

    def __init__(
        self,
        arm: Any,
        gripper: Any,
        detector: Optional[Callable[[Any], Awaitable[Any]]] = None,
        declutter: Optional[Callable[[Any, Sequence[Any]], Awaitable[Any]]] = None,
        *,
        travel_z: float = TRAVEL_Z,
        pick_z: float = MIN_Z,
        ik_fn: Optional[fast_planner.IkFn] = None,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.detector = detector
        self.declutter = declutter
        self.travel_z = travel_z
        self.pick_z = pick_z
        # Optional, additive fast-path knob (components/fast_planner.py):
        # when set, `_attempt` drives the arm with a deterministic
        # up/over/down joint-space plan (`move_to_joint_positions`, no
        # per-call BiRRT) instead of `arm.move_to_position`. Defaults to
        # None, which preserves the exact prior `move_to_position` behavior
        # -- opt-in only, nothing changes for existing callers/tests.
        self.ik_fn = ik_fn

    def budget_for(
        self, difficulty: Optional[float], calibrated_plan: Optional[Dict[str, Any]]
    ) -> int:
        """Difficulty (W3) is authoritative when available -- it "fronts
        the run" with a budget before any attempt is made. Otherwise fall
        back to the experience store's calibrated retry_budget (W1), then
        to DEFAULT_RETRY_BUDGET."""
        if difficulty is not None:
            return difficulty_to_budget(difficulty)
        if calibrated_plan and "retry_budget" in calibrated_plan:
            return int(calibrated_plan["retry_budget"])
        return DEFAULT_RETRY_BUDGET

    async def run(
        self,
        target: Any,
        *,
        difficulty: Optional[float] = None,
        calibrated_plan: Optional[Dict[str, Any]] = None,
        all_objects: Optional[Sequence[Any]] = None,
    ) -> RetryOutcome:
        """Attempt to pick `target`, retrying with escalating recovery on
        failure, up to a difficulty-derived budget. Stops immediately on
        the first successful attempt, or on the "skip" escalation level
        (whichever comes first)."""
        budget = max(1, self.budget_for(difficulty, calibrated_plan))
        current = target
        escalations: List[str] = []
        attempt = 0

        while True:
            attempt += 1
            bias = calibrated_plan if attempt == 1 else None
            success = await self._attempt(current, bias)
            if success:
                return RetryOutcome(
                    success=True,
                    attempts=attempt,
                    state=RetryState.SUCCEEDED,
                    escalations=escalations,
                    target=current,
                )
            if attempt >= budget:
                return RetryOutcome(
                    success=False,
                    attempts=attempt,
                    state=RetryState.EXHAUSTED,
                    escalations=escalations,
                    target=current,
                    failure_type="retry_budget_exhausted",
                )

            current = await self._redetect(current)
            current = self._recenter(current)

            level = ESCALATION_ORDER[min(len(escalations), len(ESCALATION_ORDER) - 1)]
            escalations.append(level)
            if level == "nudge":
                current = self._nudge(current)
            elif level == "declutter":
                await self._run_declutter(current, all_objects or [])
            elif level == "skip":
                return RetryOutcome(
                    success=False,
                    attempts=attempt,
                    state=RetryState.SKIPPED,
                    escalations=escalations,
                    target=current,
                    failure_type="escalated_skip",
                )

    # -- pipeline steps ------------------------------------------------

    async def _attempt(self, target: Any, bias: Optional[Dict[str, Any]]) -> bool:
        x, y = float(target.x), float(target.y)
        z = self.pick_z
        grip_params: Dict[str, Any] = {}
        if bias:
            dx, dy = bias.get("xy_offset", (0.0, 0.0))
            x += float(dx)
            y += float(dy)
            z = self.pick_z + float(bias.get("pick_z_offset", 0.0))
            grip_params = dict(bias.get("grip_params") or {})

        await self.gripper.open()
        if self.ik_fn is not None:
            # Fast path: deterministic up/over/down joint plan, no
            # planning per call. Retry attempts are always a local
            # descent onto (x, y) -- the "current xy" and "target xy" for
            # this micro-move are the same point.
            waypoints = fast_planner.plan_pick_waypoints(
                [], (x, y), (x, y), self.ik_fn, pick_z=z, travel_z=self.travel_z
            )
            await fast_planner.fast_move(self.arm, waypoints)
        else:
            await self.arm.move_to_position(x, y, self.travel_z)
            await self.arm.move_to_position(x, y, z)
        try:
            grabbed = await self.gripper.grab(**grip_params) if grip_params else await self.gripper.grab()
        except TypeError:
            # gripper doesn't accept grip_params kwargs -- fall back.
            grabbed = await self.gripper.grab()
        holding = True
        is_holding_fn = getattr(self.gripper, "is_holding_something", None)
        if callable(is_holding_fn):
            holding = await is_holding_fn()
        if self.ik_fn is not None:
            lift_pose = fast_planner.pick_pose(x, y, self.travel_z)
            lift_joints = fast_planner.solve_ik(self.ik_fn, lift_pose)
            await fast_planner.fast_move(self.arm, [lift_joints])
        else:
            await self.arm.move_to_position(x, y, self.travel_z)
        return bool(grabbed) and bool(holding)

    async def _redetect(self, target: Any) -> Any:
        if self.detector is None:
            return target
        try:
            updated = await self.detector(target)
        except Exception:
            return target
        return updated if updated is not None else target

    def _recenter(self, target: Any) -> Any:
        # Re-centering is already achieved by using the fresh detection's
        # coordinates (from _redetect) for the next attempt; this hook
        # exists so a geometry-aware re-centering strategy can be dropped
        # in later without changing the state machine shape.
        return target

    def _nudge(self, target: Any) -> Any:
        return _with_xy(target, float(target.x) + NUDGE_STEP_MM, float(target.y))

    async def _run_declutter(self, target: Any, all_objects: Sequence[Any]) -> None:
        if self.declutter is None:
            return
        try:
            await self.declutter(target, all_objects)
        except Exception:
            # A failed declutter attempt shouldn't abort the whole retry
            # run -- the next pick attempt is still made (and may fail
            # again, eventually reaching the "skip" escalation).
            pass
