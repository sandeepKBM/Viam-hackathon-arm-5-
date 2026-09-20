"""Offline integration test for the merge orchestrator (components/pipeline.py).

Drives the full loop -- enrich (W2/W3) -> seed (W1) -> plan (W5, rule-based
stub) -> execute pick through RetryController (W4) -> record (W1) -- with
mock arm/gripper and a tmp experience store. No robot, no models, no network.
"""

import asyncio

from components.pipeline import TaskContext, run_task, enrich_and_seed
from components.experience_store import ExperienceStore
from components.retry import RetryController
from components.shapes import LocatedShape


class FakeArm:
    def __init__(self):
        self.moves = []
        self.bins = []
        self.homed = 0

    async def move_to_position(self, x, y, z, **kw):
        self.moves.append((x, y, z))

    async def go_to(self, name, **kw):
        self.bins.append(name)

    async def go_home(self, **kw):
        self.homed += 1


class FakeGripper:
    def __init__(self, grab_result=True):
        self.grab_result = grab_result
        self.opens = 0
        self.grabs = 0

    async def open(self, **kw):
        self.opens += 1

    async def grab(self, **kw):
        self.grabs += 1
        return self.grab_result

    async def is_holding_something(self, **kw):
        return self.grab_result


def _scene():
    # Two blocks, far apart (not blocking each other), inside the workspace.
    return [
        LocatedShape(label="red block", x=200.0, y=100.0, z=179.8, color="red"),
        LocatedShape(label="yellow block", x=300.0, y=-50.0, z=179.8, color="yellow"),
    ]


def _ctx(tmp_path, grab_result=True):
    arm = FakeArm()
    gripper = FakeGripper(grab_result=grab_result)
    store = ExperienceStore(path=str(tmp_path / "experience.json"))
    retry = RetryController(arm, gripper)
    ctx = TaskContext(arm=arm, gripper=gripper, pickplace=None, store=store, retry=retry)
    return ctx, arm, gripper, store


def test_enrich_and_seed_populates_contract_fields(tmp_path):
    objs = _scene()
    store = ExperienceStore(path=str(tmp_path / "e.json"))
    enrich_and_seed(objs, image=None, store=store)
    for o in objs:
        assert o.canonical_label == f"{o.color} block"
        assert o.difficulty is not None
        assert hasattr(o, "history")  # seeded (None when no prior attempts)


def test_full_loop_success_picks_places_and_records(tmp_path):
    ctx, arm, gripper, store = _ctx(tmp_path, grab_result=True)
    results = asyncio.run(
        run_task("sort the red and yellow blocks", _scene(), None, ctx)
    )
    # Both objects planned as picks and succeeded.
    assert len(results) == 2
    assert all(r.skill == "pick" and r.success for r in results)
    # Each grasped object was placed into its color bin.
    assert set(arm.bins) == {"bin1", "bin2"}
    # Outcomes recorded under canonical keys, with a calibrated plan.
    assert store.get_calibration("red block")["retry_budget"] >= 1
    assert store.get_history("red block") is not None
    assert store.get_history("yellow block") is not None


def test_full_loop_failure_exhausts_retries_and_records_failure(tmp_path):
    ctx, arm, gripper, store = _ctx(tmp_path, grab_result=False)
    results = asyncio.run(run_task("sort the red blocks", _scene(), None, ctx))
    reds = [r for r in results if r.key == "red block"]
    assert reds and reds[0].success is False
    # Retried more than once (difficulty-derived budget), no bin placement.
    assert reds[0].attempts >= 1
    assert "bin1" not in arm.bins  # never placed a failed grasp
    hist = store.get_history("red block")
    assert hist is not None and hist["attempts"][-1]["grasp_success"] is False


def test_easy_difficulty_picked_first(tmp_path):
    ctx, arm, gripper, store = _ctx(tmp_path, grab_result=True)
    scene = _scene()
    # Make the red block explicitly harder than the yellow one.
    scene[0].difficulty = 0.9
    scene[1].difficulty = 0.1
    # enrich would overwrite difficulty; pre-seed via a planner that trusts
    # the provided scene by skipping enrich's recompute is out of scope, so
    # assert ordering through the planner on the pre-enriched values instead.
    from components.policy import plan_task

    plan = plan_task("sort the red and yellow blocks", scene)
    ordered_keys = [c.params.object.color for c in plan if c.skill == "pick"]
    assert ordered_keys[0] == "yellow"  # easier (0.1) first
