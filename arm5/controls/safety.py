"""Joint/cartesian limit checks, an e-stop hook, and workspace bounds.

These checks are meant to gate every command before it is sent to
`arm5.hardware`, independent of whatever planner produced it.

Target hardware is a UFactory XL15 arm. Its DOF count, per-joint limits,
reach, and payload are NOT hard-coded here: they must be confirmed from the
UFactory XL15 datasheet before this module can enforce real limits. Do not
guess these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

# PLACEHOLDER: XL15 per-joint (min, max) limits, in radians. Left empty
# until confirmed from the UFactory XL15 datasheet -- do not guess DOF or
# per-joint ranges here.
XL15_JOINT_LIMITS_RAD: Optional[Sequence[Tuple[float, float]]] = None

# PLACEHOLDER: XL15 cartesian workspace bounds, in meters. Left as None
# until confirmed from the UFactory XL15 datasheet / reach spec.
XL15_WORKSPACE_BOUNDS_M: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None


@dataclass
class SafetyLimits:
    """Static safety configuration for a given arm.

    Defaults are intentionally empty/None placeholders (see
    `XL15_JOINT_LIMITS_RAD` / `XL15_WORKSPACE_BOUNDS_M` above) -- populate
    them from the confirmed UFactory XL15 datasheet, not a guess.
    """

    joint_limits_rad: Sequence[Tuple[float, float]] = field(default_factory=list)
    """Per-joint (min, max) limits, in radians. TODO: fill from XL15 datasheet."""

    workspace_bounds_m: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    """((x_min, x_max), (y_min, y_max), (z_min, z_max)) cartesian workspace bounds, in meters.
    TODO: fill from XL15 datasheet / reach spec."""

    def check_joint_positions(self, positions_rad: Sequence[float]) -> bool:
        """Return True if `positions_rad` is within `joint_limits_rad`.

        TODO: raise a descriptive exception (vs. bool) once callers are
        wired up, so violations can be logged with the offending joint.
        """
        raise NotImplementedError("TODO: implement per-joint limit checking")

    def check_cartesian_position(self, xyz_m: Tuple[float, float, float]) -> bool:
        """Return True if `xyz_m` is within `workspace_bounds_m`."""
        raise NotImplementedError("TODO: implement workspace bounds checking")


class EStop:
    """Simple e-stop hook: a callback that can halt motion immediately.

    TODO: wire this to a real trigger (hardware button, keyboard interrupt,
    watchdog timeout) and to `arm5.hardware.arm.ArmController.stop` /
    `arm5.hardware.gripper.GripperController.stop`.
    """

    def __init__(self, on_trigger: Optional[Callable[[], None]] = None) -> None:
        self._on_trigger = on_trigger
        self._triggered = False

    def trigger(self) -> None:
        """Trip the e-stop and invoke the registered callback, if any."""
        self._triggered = True
        if self._on_trigger is not None:
            self._on_trigger()

    @property
    def is_triggered(self) -> bool:
        return self._triggered
