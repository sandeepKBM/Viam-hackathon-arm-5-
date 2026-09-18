"""Abstract controller interface and a simple setpoint controller stub.

Control-rate assumption: these controllers assume they are driven by an
external fixed-rate loop (e.g. 30-100 Hz) that calls `step()` once per tick
with a wall-clock `dt`. No internal timer/thread is started here -- the
caller (a script or `arm5.planning` executor) owns the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class Controller(ABC):
    """Base class for something that turns a setpoint + current state into a command."""

    @abstractmethod
    def set_target(self, target: Sequence[float]) -> None:
        """Set the desired joint or cartesian target."""
        raise NotImplementedError

    @abstractmethod
    def step(self, current: Sequence[float], dt: float) -> Sequence[float]:
        """Compute one control command given the current state and timestep `dt` (s)."""
        raise NotImplementedError


class SetpointController(Controller):
    """Minimal joint/cartesian setpoint controller stub (e.g. for a P/PD loop).

    TODO: implement actual gain-based tracking; currently a pass-through
    stub so the interface can be wired up end-to-end before the control law
    is written.
    """

    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain
        self._target: Sequence[float] = []

    def set_target(self, target: Sequence[float]) -> None:
        self._target = target

    def step(self, current: Sequence[float], dt: float) -> Sequence[float]:
        """TODO: replace with real P/PD control law using `self._gain` and `dt`."""
        raise NotImplementedError("TODO: implement setpoint tracking control law")
