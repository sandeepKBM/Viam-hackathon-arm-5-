#!/usr/bin/env python
"""Demo: the self-improving pick loop converging (W1 experience_store + W4
retry, driven through components/pipeline.py's own building blocks).

Fully OFFLINE and deterministic: mock arm/gripper (same shape as
tests/test_pipeline.py's FakeArm/FakeGripper), a tmp-dir ExperienceStore, no
robot, no camera, no model download. Run with:

    .venv/bin/python scripts/demo_selfimprove.py
    .venv/bin/python scripts/demo_selfimprove.py --smoke     # quick CI path
    .venv/bin/python scripts/demo_selfimprove.py --episodes 12

What this demonstrates
-----------------------
Every episode re-perceives the SAME real-world object (a red block whose
true grasp point is offset from the naive detection centroid by exactly
`TRUE_XY_ERROR_MM` -- a stand-in for ordinary camera/arm calibration drift).
A naive first attempt at the raw detected position always misses; whether
the pick still SUCCEEDS depends on whether components.retry.RetryController
has enough of a retry budget to reach its one "nudge" escalation (a fixed
+RETRY_NUDGE_STEP_MM correction) before giving up.

That budget is exactly components.experience_store.ExperienceStore's
"self-building" `calibrated_plan.retry_budget` -- recomputed after every
recorded attempt from the object's recent failure rate (more recent misses
-> bigger budget; a clean streak -> the budget settles back down). This demo
sets it up so that signal is the thing actually driving success:

  - Episode 1: the object has no history yet, so the store hands out a
    LOW budget (1) -- there's no room for the nudge to fire, so the pick
    fails outright.
  - That recorded failure immediately pushes the calibrated budget way up
    (the store reacts to a 100% recent failure rate).
  - With budget >= 2, the very next episode's miss-then-nudge sequence
    succeeds, and the budget then settles down to the minimum that's still
    sufficient (2) -- a converged, self-improving policy.

Why `.difficulty` is cleared per-episode: `components.pipeline.enrich_and_seed`
(the same W2/W3 step `run_task` calls) always fills in a numeric
`.difficulty` via `components.uq.annotate_difficulty`, and
`components.retry.RetryController.budget_for` treats a live `.difficulty` as
AUTHORITATIVE over the experience store's calibrated retry_budget (see its
docstring). Since this demo's synthetic objects carry no shape/box/detector
signal, that difficulty would just be a flat constant every episode -- so to
actually exercise (and show) the store's *own* retry_budget calibration
adapting from real attempt history, this demo clears `.difficulty` after the
enrich step, letting RetryController fall back to
`ExperienceStore.get_calibration(...)["retry_budget"]`, exactly as its
fallback path is documented to do. Everything else runs through the real,
unmodified pipeline: `enrich_and_seed` -> `plan_task` ->
`execute_plan_with_memory` (the same three calls `pipeline.run_task` composes).

Note on `xy_offset`/`pick_z_offset`: `components/pipeline.py` records each
attempt's plan_params with the offset that was FETCHED going into that pick
(the prior bias applied to attempt 1), not a freshly-measured correction --
so, as wired today, those two calibration fields are stable audit fields
(they stay at their seed value here) rather than the thing that visibly
adapts; `retry_budget` is. Both are still printed every episode for
visibility into the full calibrated_plan the store is maintaining.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Run from repo root or elsewhere -- make sure `components` is importable
# regardless of cwd (this script lives in scripts/, one level down).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tunables for ExperienceStore's calibration rollup -- set *before* importing
# components.experience_store (directly or transitively) so its module-level
# defaults pick them up. This only affects this demo process.
os.environ.setdefault("EXPERIENCE_DEFAULT_RETRY_BUDGET", "1")  # unseen object -> no retry margin
os.environ.setdefault("EXPERIENCE_MIN_RETRY_BUDGET", "2")      # never settle below "just enough"
os.environ.setdefault("EXPERIENCE_MAX_RETRY_BUDGET", "4")
os.environ.setdefault("EXPERIENCE_RECENT_WINDOW", "5")

from components.constants import MIN_Z, TRAVEL_Z  # noqa: E402
from components.experience_store import ExperienceStore  # noqa: E402
from components.pipeline import TaskContext, enrich_and_seed, execute_plan_with_memory  # noqa: E402
from components.policy import plan_task  # noqa: E402
from components.retry import NUDGE_STEP_MM, RetryController  # noqa: E402
from components.shapes import LocatedShape  # noqa: E402

# The naive detection centroid is always this far from the TRUE grasp point
# (mm) -- matches RetryController's single "nudge" escalation step exactly,
# so whether a pick succeeds hinges entirely on whether the retry budget
# gives it a chance to fire.
TRUE_XY_ERROR_MM = (NUDGE_STEP_MM, 0.0)
GRAB_TOLERANCE_MM = 1.0
OBJECT_XY = (200.0, 100.0)


class SimArm:
    """Records every commanded position; test_pipeline.py's FakeArm shape."""

    def __init__(self) -> None:
        self.moves: list[tuple[float, float, float]] = []
        self.bins: list[str] = []
        self.homed = 0
        self.last_pick_xy: tuple[float, float] | None = None

    async def move_to_position(self, x, y, z, **kw):
        self.moves.append((x, y, z))
        if abs(z - MIN_Z) < 1e-6:  # the descend-to-grasp move, not the travel move
            self.last_pick_xy = (x, y)

    async def go_to(self, name, **kw):
        self.bins.append(name)

    async def go_home(self, **kw):
        self.homed += 1


class SimGripper:
    """Grasp succeeds iff the arm's last descend position is within
    GRAB_TOLERANCE_MM of the TRUE grasp point -- a simple physical stand-in
    for "the naive detection was off by a fixed calibration error"."""

    def __init__(self, arm: SimArm, object_xy: tuple[float, float]):
        self.arm = arm
        self.true_xy = (object_xy[0] + TRUE_XY_ERROR_MM[0], object_xy[1] + TRUE_XY_ERROR_MM[1])
        self.opens = 0
        self.grabs = 0

    async def open(self, **kw):
        self.opens += 1

    async def grab(self, **kw) -> bool:
        self.grabs += 1
        if self.arm.last_pick_xy is None:
            return False
        dx = self.arm.last_pick_xy[0] - self.true_xy[0]
        dy = self.arm.last_pick_xy[1] - self.true_xy[1]
        return (dx * dx + dy * dy) ** 0.5 <= GRAB_TOLERANCE_MM

    async def is_holding_something(self, **kw) -> bool:
        return True


async def run_episode(store: ExperienceStore):
    """One perceive -> enrich -> plan -> execute-with-memory pass, mirroring
    components.pipeline.run_task's own body (enrich_and_seed -> plan_task ->
    execute_plan_with_memory) with one addition: clear .difficulty after
    enrich so RetryController's retry_budget fallback (calibrated_plan, from
    the experience store) is what actually governs this pick -- see the
    module docstring for why."""
    arm = SimArm()
    gripper = SimGripper(arm, OBJECT_XY)
    retry = RetryController(arm, gripper, travel_z=TRAVEL_Z, pick_z=MIN_Z)
    ctx = TaskContext(arm=arm, gripper=gripper, pickplace=None, store=store, retry=retry)

    obj = LocatedShape(label="red block", x=OBJECT_XY[0], y=OBJECT_XY[1], z=MIN_Z, color="red")
    calib_before = store.get_calibration("red block")  # what THIS episode will actually use

    scene = enrich_and_seed([obj], image=None, store=store, detector_fn=None)
    raw_difficulty = scene[0].difficulty
    for o in scene:
        o.difficulty = None  # let RetryController fall back to calib_before["retry_budget"]

    plan = plan_task("sort the red blocks", scene, planner=None)
    results = await execute_plan_with_memory(plan, scene, ctx)

    return results[0], calib_before, raw_difficulty


def _fmt_xy(xy) -> str:
    return f"({float(xy[0]):+.1f}, {float(xy[1]):+.1f})"


async def main_async(n_episodes: int) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(path=str(Path(tmp) / "experience.json"))

        header = (
            f"{'ep':>3}  {'w3 difficulty':>13}  {'retry_budget':>12}  "
            f"{'xy_offset':>14}  {'z_offset':>9}  {'attempts':>8}  {'result':>8}"
        )
        print("Self-improving pick loop -- red block, repeated episodes")
        print(f"(true grasp point is {NUDGE_STEP_MM:.1f}mm off the naive detection; "
              f"retry_budget is the calibrated field that adapts -- see module docstring)")
        print(header)
        print("-" * len(header))

        successes: list[bool] = []
        for ep in range(1, n_episodes + 1):
            result, calib_before, raw_difficulty = await run_episode(store)
            successes.append(result.success)
            print(
                f"{ep:>3}  {raw_difficulty:>13.2f}  {calib_before['retry_budget']:>12d}  "
                f"{_fmt_xy(calib_before['xy_offset']):>14}  "
                f"{calib_before['pick_z_offset']:>9.2f}  {result.attempts:>8d}  "
                f"{'OK' if result.success else 'MISS':>8}"
            )

        final = store.get_calibration("red block")
        early_fail = not successes[0]
        # Every episode from #2 onward should succeed once the store has
        # reacted to episode 1's failure -- check the whole streak, not just
        # a fixed-size tail, so this also holds for a short --smoke run.
        rest_succeeded = all(successes[1:]) if len(successes) > 1 else True
        print("-" * len(header))
        print(
            f"final calibrated_plan for 'red block': retry_budget="
            f"{final['retry_budget']}, xy_offset={_fmt_xy(final['xy_offset'])}, "
            f"pick_z_offset={final['pick_z_offset']:.2f}"
        )
        print(
            f"converged: episode 1 {'failed' if early_fail else 'succeeded'} "
            f"(retry_budget=1, no margin for the nudge) -> episodes 2.."
            f"{len(successes)} {'all succeeded' if rest_succeeded else 'still missing sometimes'} "
            f"(retry_budget settled at {final['retry_budget']})"
        )
        return early_fail and rest_succeeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8, help="number of repeated episodes")
    parser.add_argument("--smoke", action="store_true", help="quick CI path (3 episodes)")
    args = parser.parse_args()

    n_episodes = 3 if args.smoke else args.episodes
    converged = asyncio.run(main_async(n_episodes))
    if not converged:
        print("warning: did not observe the expected fail-then-converge pattern")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
