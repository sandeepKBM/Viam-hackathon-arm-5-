"""The Claude stub is a seam, not an implementation: it conforms to the planner
interface, raises a clear NotImplementedError until filled, and a subclass that
fills the one method drops straight into policy.plan_task."""

import pytest

from components.claude_stub import ClaudePlanner, ClaudeOracle
from components.policy import plan_task
from components.shapes import LocatedShape


def test_planner_stub_raises_until_wired():
    with pytest.raises(NotImplementedError):
        ClaudePlanner()("sort the blocks", [], [])


def test_oracle_stub_raises_until_wired():
    with pytest.raises(NotImplementedError):
        ClaudeOracle().verify_label(image=None, box=(0, 0, 1, 1))


def test_filled_planner_drops_into_plan_task():
    """A future agent only fills _call_claude; the seam then works end-to-end."""

    class WiredClaude(ClaudePlanner):
        def _call_claude(self, prompt):
            # pretend Claude returned a valid plan referencing obj0
            return '[{"skill": "pick", "params": {"object_id": "obj0"}}]'

    scene = [LocatedShape(label="red block", x=200.0, y=100.0, z=179.8, color="red")]
    plan = plan_task("bring me the red block", scene, planner=WiredClaude())
    assert len(plan) == 1 and plan[0].skill == "pick"
