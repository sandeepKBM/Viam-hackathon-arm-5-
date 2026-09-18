"""Joint/cartesian limit checks, an e-stop hook, and workspace bounds.

These checks are meant to gate every command before it is sent to
`arm5.hardware`, independent of whatever planner produced it.

Target hardware is a UFactory xArm 5 (5-DOF). The DOF count is known
(``XARM5_DOF = 5``), but per-joint limits, reach, and payload are NOT
hard-coded here: they must be confirmed from the UFactory xArm 5 datasheet
before this module can enforce real limits. Do not guess these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

# UFactory xArm 5 is a 5-DOF arm (per project owner). See
# `check_joint_command_length` for the (only currently-enforced) guard that
# uses it.
XARM5_DOF: int = 5


def check_joint_command_length(positions_rad: Sequence[float], dof: int = XARM5_DOF) -> None:
    """Raise ``ValueError`` if a joint command does not have exactly ``dof`` values.

    This is the one safety guard that is fully enforceable today: it does not
    depend on datasheet limits, only on the known DOF count. Range/limit
    checks remain TODO (see ``SafetyLimits``) until the xArm 5 limits are
    confirmed.
    """
    n = len(positions_rad)
    if n != dof:
        raise ValueError(f"Expected {dof} joint values for xArm 5, got {n}.")

# PLACEHOLDER: xArm 5 per-joint (min, max) limits, in radians -- a length-5
# sequence once filled. Left empty until confirmed from the UFactory xArm 5
# datasheet; do not guess per-joint ranges here.
XARM5_JOINT_LIMITS_RAD: Optional[Sequence[Tuple[float, float]]] = None

# PLACEHOLDER: xArm 5 cartesian workspace bounds, in meters. Left as None
# until confirmed from the UFactory xArm 5 datasheet / reach spec.
XARM5_WORKSPACE_BOUNDS_M: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None


@dataclass
class SafetyLimits:
    """Static safety configuration for a given arm.

    Defaults are ``None`` on purpose (see `XARM5_JOINT_LIMITS_RAD` /
    `XARM5_WORKSPACE_BOUNDS_M` above): with limits unset the checks below
    fail **closed** (raise) rather than silently approving motion. Populate
    them from the confirmed UFactory xArm 5 datasheet, not a guess. When
    filled, `joint_limits_rad` should have length `XARM5_DOF` (5).
    """

    joint_limits_rad: Optional[Sequence[Tuple[float, float]]] = None
    """Per-joint (min, max) limits, in radians (length `XARM5_DOF`).
    TODO: fill from xArm 5 datasheet."""

    workspace_bounds_m: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    """((x_min, x_max), (y_min, y_max), (z_min, z_max)) cartesian workspace bounds, in meters.
    TODO: fill from xArm 5 datasheet / reach spec."""

    def check_joint_positions(self, positions_rad: Sequence[float]) -> bool:
        """Return True if `positions_rad` is within `joint_limits_rad`.

        Fails closed: enforces DOF length now, and refuses to pass a command
        while per-joint limits are unset rather than approving it blindly.

        TODO: implement the actual per-joint range check (and prefer raising
        a descriptive exception naming the offending joint) once the xArm 5
        limits are filled in.
        """
        check_joint_command_length(positions_rad)
        if not self.joint_limits_rad:
            raise NotImplementedError(
                "joint_limits_rad not configured; refusing to approve motion "
                "(fill from the xArm 5 datasheet)."
            )
        raise NotImplementedError("TODO: implement per-joint range checking")

    def check_cartesian_position(self, xyz_m: Tuple[float, float, float]) -> bool:
        """Return True if `xyz_m` is within `workspace_bounds_m`.

        Fails closed while `workspace_bounds_m` is unset.
        """
        if self.workspace_bounds_m is None:
            raise NotImplementedError(
                "workspace_bounds_m not configured; refusing to approve motion "
                "(fill from the xArm 5 datasheet)."
            )
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
