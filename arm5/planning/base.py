"""Shared Planner interface and Trajectory/Waypoint data model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class Waypoint:
    """A single point along a trajectory."""

    joint_positions_rad: Sequence[float]
    time_from_start_s: float = 0.0


@dataclass
class Trajectory:
    """An ordered sequence of waypoints produced by a planner."""

    waypoints: List[Waypoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Planner(ABC):
    """Common interface implemented by classical, learned, and soft-logic planners."""

    @abstractmethod
    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        context: Dict[str, Any],
    ) -> Trajectory:
        """Plan a trajectory from `start` to `goal`.

        Args:
            start: starting joint positions (radians).
            goal: goal joint positions (radians), or another
                planner-specific goal representation (e.g. a cartesian pose)
                carried in `context`.
            context: planner-specific extras (obstacles, camera detections,
                rule weights, etc).

        TODO: settle on whether `goal` is always joint-space or whether
        cartesian goals get normalized upstream before reaching planners.
        """
        raise NotImplementedError
