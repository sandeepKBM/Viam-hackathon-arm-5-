"""Framework-agnostic learned-policy interface, adapted to the Planner ABC.

Deliberately has no dependency on a specific ML framework (torch/jax/etc.)
so the interface can be pinned down before the actual model is chosen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from arm5.planning.base import Planner, Trajectory


class LearnedPolicy(ABC):
    """Interface for a learned control/planning policy (e.g. IL or RL)."""

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model weights from `checkpoint_path`.

        TODO: decide checkpoint format once a framework is chosen.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, obs: Dict[str, Any]) -> Sequence[float]:
        """Compute an action (e.g. joint velocities/targets) given `obs`.

        TODO: define the `obs` schema (proprioception + vision features).
        """
        raise NotImplementedError


class LearnedPlanner(Planner):
    """Adapts a `LearnedPolicy` to the `Planner` interface.

    TODO: implement `plan()`, likely by rolling the policy forward
    open-loop from `start` toward `goal` to produce a `Trajectory`, or by
    treating this as a closed-loop controller instead (see
    `arm5.controls.controllers`).
    """

    def __init__(self, policy: LearnedPolicy) -> None:
        self._policy = policy

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        context: Dict[str, Any],
    ) -> Trajectory:
        raise NotImplementedError("TODO: roll out self._policy to build a Trajectory")
