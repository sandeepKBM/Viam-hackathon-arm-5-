"""Wrapper around the Viam ``MotionClient`` service, and a classical planner.

Notes on IK/RRT:
    The Viam `MotionClient.move` call performs IK + collision-aware path
    planning (RRT-connect under the hood, in the Viam motion service) on the
    *server* side given a destination pose and optional `world_state`
    obstacles. This module does not reimplement IK/RRT locally -- it is a
    thin client over that service. A from-scratch local planner (e.g. for
    offline testing without a robot) would live in a separate module.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from viam.components.arm import Pose
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from arm5.planning.base import Planner, Trajectory


class ArmMotionClient:
    """Convenience wrapper around a single Viam ``MotionClient`` resource."""

    def __init__(self, motion: MotionClient) -> None:
        self._motion = motion

    @classmethod
    def from_robot(cls, robot: RobotClient, name: str = "builtin") -> "ArmMotionClient":
        """Look up the named motion service resource on ``robot`` and wrap it."""
        return cls(MotionClient.from_robot(robot, name))

    async def move(self, component_name: str, destination: Pose) -> bool:
        """Ask the Viam motion service to move `component_name` to `destination`.

        TODO: thread through `world_state` obstacles and `constraints` once
        `arm5.controls.safety` workspace bounds are finalized.
        """
        raise NotImplementedError("TODO: build PoseInFrame from `destination` and call self._motion.move")


class ClassicalPlanner(Planner):
    """`Planner` implementation backed by the Viam motion service (IK/RRT).

    TODO: implement `plan()` by calling into `ArmMotionClient`, or by doing
    local IK/RRT if an offline (no-robot) planner is needed.
    """

    def __init__(self, motion_client: ArmMotionClient) -> None:
        self._motion_client = motion_client

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        context: Dict[str, Any],
    ) -> Trajectory:
        raise NotImplementedError("TODO: implement classical plan() via ArmMotionClient")
