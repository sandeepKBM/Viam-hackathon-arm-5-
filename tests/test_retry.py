"""Offline unit tests for components/retry.py.

No Viam connection, no robot, no camera -- arm/gripper/detector/declutter
are all injected fakes, exercising the pure state-machine logic of
RetryController (mirrors the dependency-injection style already used
elsewhere, e.g. components/declutter.py's duck-typed objects).

The repo's test suite doesn't depend on pytest-asyncio, so async test
bodies are driven with a tiny `asyncio.run(...)` wrapper (`@async_test`
below) rather than the `@pytest.mark.asyncio` decorator -- no new test
dependency required.
"""

import asyncio
import functools
from dataclasses import dataclass

import pytest

from components.retry import (
    DEFAULT_RETRY_BUDGET,
    MAX_RETRY_BUDGET,
    MIN_RETRY_BUDGET,
    RetryController,
    RetryState,
    difficulty_to_budget,
)


@dataclass
class FakeTarget:
    x: float
    y: float
    color: str = "red"
    label: str = "cube"


class FakeArm:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float, float]] = []

    async def move_to_position(self, x, y, z, **kwargs):
        self.moves.append((x, y, z))


class FakeGripper:
    """grab() pops results off a preset queue; defaults to True (grabbed)
    once the queue is exhausted. holding_queue works the same way for
    is_holding_something(); pass holding_queue=None to omit the method
    entirely (gripper that doesn't report holding status)."""

    def __init__(self, grab_results, holding_results=None):
        self._grab_results = list(grab_results)
        self._has_holding = holding_results is not None
        self._holding_results = list(holding_results) if holding_results is not None else []
        self.open_calls = 0
        self.grab_calls = []

    async def open(self):
        self.open_calls += 1

    async def grab(self, **kwargs):
        self.grab_calls.append(kwargs)
        return self._grab_results.pop(0) if self._grab_results else True

    async def is_holding_something(self):
        return self._holding_results.pop(0) if self._holding_results else True


def async_test(coro_fn):
    """Run an `async def test_...` body to completion via asyncio.run, so
    these tests work under plain pytest without the pytest-asyncio plugin."""

    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro_fn(*args, **kwargs))

    return wrapper


class GripperWithoutHoldingCheck:
    """A gripper that doesn't expose is_holding_something at all."""

    def __init__(self, grab_results):
        self._grab_results = list(grab_results)
        self.open_calls = 0

    async def open(self):
        self.open_calls += 1

    async def grab(self, **kwargs):
        return self._grab_results.pop(0) if self._grab_results else True


# ---------------------------------------------------------------------------
# difficulty -> budget mapping
# ---------------------------------------------------------------------------


def test_difficulty_zero_maps_to_min_budget():
    assert difficulty_to_budget(0.0) == MIN_RETRY_BUDGET


def test_difficulty_one_maps_to_max_budget():
    assert difficulty_to_budget(1.0) == MAX_RETRY_BUDGET


def test_difficulty_out_of_range_is_clamped():
    assert difficulty_to_budget(-5.0) == MIN_RETRY_BUDGET
    assert difficulty_to_budget(5.0) == MAX_RETRY_BUDGET


# ---------------------------------------------------------------------------
# stop on first success
# ---------------------------------------------------------------------------


@async_test
async def test_succeeds_immediately_without_retry():
    arm, gripper = FakeArm(), FakeGripper(grab_results=[True])
    controller = RetryController(arm, gripper)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=0.5)

    assert outcome.success is True
    assert outcome.attempts == 1
    assert outcome.state == RetryState.SUCCEEDED
    assert outcome.escalations == []


@async_test
async def test_stops_on_first_success_after_one_failure():
    arm, gripper = FakeArm(), FakeGripper(grab_results=[False, True])
    controller = RetryController(arm, gripper)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=1.0)  # budget=4

    assert outcome.success is True
    assert outcome.attempts == 2
    assert outcome.state == RetryState.SUCCEEDED
    assert outcome.escalations == ["nudge"]
    # No third attempt was made after success.
    assert len(gripper.grab_calls) == 2


# ---------------------------------------------------------------------------
# escalation order
# ---------------------------------------------------------------------------


@async_test
async def test_escalation_order_is_nudge_then_declutter_then_skip():
    arm = FakeArm()
    gripper = FakeGripper(grab_results=[False, False, False, False])
    declutter_calls = []

    async def declutter(target, all_objects):
        declutter_calls.append((target, list(all_objects)))

    controller = RetryController(arm, gripper, declutter=declutter)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=1.0, all_objects=["blocker"])

    assert outcome.success is False
    assert outcome.state == RetryState.SKIPPED
    assert outcome.escalations == ["nudge", "declutter", "skip"]
    # skip is terminal -- stops before a 4th attempt even though budget=4 allowed one.
    assert outcome.attempts == 3
    assert len(declutter_calls) == 1
    assert declutter_calls[0][1] == ["blocker"]


# ---------------------------------------------------------------------------
# budget respected
# ---------------------------------------------------------------------------


@async_test
async def test_budget_of_one_means_no_retry_at_all():
    arm, gripper = FakeArm(), FakeGripper(grab_results=[False, False, False])
    controller = RetryController(arm, gripper)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=0.0)  # budget=1

    assert outcome.success is False
    assert outcome.state == RetryState.EXHAUSTED
    assert outcome.attempts == 1
    assert outcome.escalations == []


@async_test
async def test_budget_exhausted_before_reaching_skip_escalation():
    arm, gripper = FakeArm(), FakeGripper(grab_results=[False, False, False])
    controller = RetryController(arm, gripper)
    # No difficulty -> falls back to calibrated_plan.retry_budget = 3.
    outcome = await controller.run(
        FakeTarget(0.0, 0.0), difficulty=None, calibrated_plan={"retry_budget": 3}
    )

    assert outcome.success is False
    assert outcome.state == RetryState.EXHAUSTED
    assert outcome.attempts == 3
    # Escalated twice (nudge, declutter) before running out of budget --
    # never reached "skip".
    assert outcome.escalations == ["nudge", "declutter"]


@async_test
async def test_default_budget_used_when_neither_difficulty_nor_plan_given():
    arm, gripper = FakeArm(), FakeGripper(grab_results=[False] * 10)
    controller = RetryController(arm, gripper)
    outcome = await controller.run(FakeTarget(0.0, 0.0))

    assert outcome.attempts == DEFAULT_RETRY_BUDGET
    assert outcome.state == RetryState.EXHAUSTED


# ---------------------------------------------------------------------------
# calibrated_plan biases only the first attempt
# ---------------------------------------------------------------------------


@async_test
async def test_calibrated_plan_offset_applied_only_to_first_attempt():
    arm = FakeArm()
    gripper = FakeGripper(grab_results=[False, True])
    controller = RetryController(arm, gripper, pick_z=100.0, travel_z=200.0)
    calibrated_plan = {"pick_z_offset": -5.0, "xy_offset": [2.0, -3.0], "grip_params": {}}

    outcome = await controller.run(
        FakeTarget(10.0, 10.0), difficulty=1.0, calibrated_plan=calibrated_plan
    )

    assert outcome.success is True
    # First attempt: travel move then pick move, both biased.
    first_travel, first_pick = arm.moves[0], arm.moves[1]
    assert first_pick == (12.0, 7.0, 95.0)
    assert first_travel == (12.0, 7.0, 200.0)

    # Second attempt (after redetect+nudge, no bias): unbiased pick_z, and
    # x shifted by the nudge step relative to the (unbiased) original x.
    # moves layout is [travel, pick, travel] per attempt -> attempt 2's
    # pick move is index 4 (0,1,2 = attempt 1; 3,4,5 = attempt 2).
    second_pick = arm.moves[4]
    assert second_pick[2] == 100.0  # no pick_z_offset this time
    assert second_pick[0] == pytest.approx(10.0 + 5.0)  # NUDGE_STEP_MM default


# ---------------------------------------------------------------------------
# re-detection updates the target used for subsequent attempts
# ---------------------------------------------------------------------------


@async_test
async def test_redetect_shifts_target_before_next_attempt():
    arm = FakeArm()
    gripper = FakeGripper(grab_results=[False, True])

    async def detector(target):
        return FakeTarget(target.x + 50.0, target.y + 50.0)

    controller = RetryController(arm, gripper, detector=detector, pick_z=100.0, travel_z=200.0)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=1.0)

    assert outcome.success is True
    assert outcome.target.x == pytest.approx(50.0 + 5.0)  # redetected + nudge
    assert outcome.target.y == pytest.approx(50.0)


@async_test
async def test_redetect_none_keeps_previous_target():
    arm = FakeArm()
    gripper = FakeGripper(grab_results=[False, True])

    async def detector(target):
        return None  # nothing new seen

    controller = RetryController(arm, gripper, detector=detector)
    outcome = await controller.run(FakeTarget(3.0, 4.0), difficulty=1.0)

    assert outcome.target.x == pytest.approx(3.0 + 5.0)
    assert outcome.target.y == pytest.approx(4.0)


@async_test
async def test_detector_exception_does_not_abort_run():
    arm = FakeArm()
    gripper = FakeGripper(grab_results=[False, True])

    async def bad_detector(target):
        raise RuntimeError("camera glitch")

    controller = RetryController(arm, gripper, detector=bad_detector)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=1.0)

    assert outcome.success is True
    assert outcome.attempts == 2


# ---------------------------------------------------------------------------
# declutter failures don't abort the run
# ---------------------------------------------------------------------------


@async_test
async def test_declutter_exception_does_not_abort_run():
    arm = FakeArm()
    gripper = FakeGripper(grab_results=[False, False, True])

    async def flaky_declutter(target, all_objects):
        raise RuntimeError("no clear zone")

    controller = RetryController(arm, gripper, declutter=flaky_declutter)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=1.0)

    assert outcome.success is True
    assert outcome.attempts == 3
    assert outcome.escalations == ["nudge", "declutter"]


# ---------------------------------------------------------------------------
# failure signal: grab() True but not actually holding something
# ---------------------------------------------------------------------------


@async_test
async def test_is_holding_something_false_overrides_grab_true():
    arm = FakeArm()
    # grab() reports True both times, but is_holding_something() says False
    # the first time (e.g. block slipped) -- that attempt must count as a
    # failure.
    gripper = FakeGripper(grab_results=[True, True], holding_results=[False, True])
    controller = RetryController(arm, gripper)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=1.0)

    assert outcome.attempts == 2
    assert outcome.success is True


@async_test
async def test_gripper_without_is_holding_something_still_works():
    arm = FakeArm()
    gripper = GripperWithoutHoldingCheck(grab_results=[True])
    controller = RetryController(arm, gripper)
    outcome = await controller.run(FakeTarget(0.0, 0.0), difficulty=1.0)

    assert outcome.success is True
    assert outcome.attempts == 1
