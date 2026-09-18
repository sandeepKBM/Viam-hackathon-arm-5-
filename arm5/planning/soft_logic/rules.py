"""Soft-constraint / rule scoring and planner arbitration.

This module holds the hand-written heuristics, behavior rules, and
arbitration logic (the "soft logic") that we will write to bias or veto
candidate outputs from `arm5.planning.classical` and `arm5.planning.learned`.
Unlike those planners, rules here are not learned or solved for -- they are
authored directly (e.g. "prefer trajectories that keep the gripper above the
table", "veto anything that enters the no-go zone").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from arm5.planning.base import Trajectory


@dataclass
class Rule:
    """A single hand-written soft constraint.

    `score_fn` should return a float in roughly [-1, 1]: negative values
    penalize a candidate trajectory, positive values reward it. A rule may
    also be marked `hard=True` to indicate it should veto (not just
    penalize) a violating candidate.
    """

    name: str
    score_fn: Callable[[Trajectory], float]
    hard: bool = False
    weight: float = 1.0


@dataclass
class RuleSet:
    """A collection of `Rule`s used to arbitrate between planner candidates."""

    rules: List[Rule]

    def score(self, candidate: Trajectory) -> float:
        """Return a weighted sum of all rule scores for `candidate`.

        TODO: define veto semantics precisely (e.g. return -inf if any
        `hard` rule fails) and decide whether scoring happens per-waypoint
        or on the trajectory as a whole.
        """
        raise NotImplementedError("TODO: implement weighted rule scoring")

    def choose_best(self, candidates: Sequence[Trajectory]) -> Trajectory:
        """Pick the highest-scoring, non-vetoed trajectory among `candidates`.

        This is the arbitration entry point between `classical` and
        `learned` planner outputs.
        """
        raise NotImplementedError("TODO: implement candidate arbitration")
