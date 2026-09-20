"""xarm_velocity: a Viam module adding velocity control, servo streaming, and
torque readout to xArm arms (capabilities the stock UFactory module and
Viam's Arm API don't expose), with configurable DOF (5/6/7).

Importing this package (or `xarm_velocity.arm`) never requires the `xarm`
SDK -- it's only imported lazily inside `arm._load_real_xarm_api()`, and only
when no fake backend has been injected. See `fake_backend.FakeXArm` for the
offline-testable stand-in used by `xarm_velocity/tests/`.
"""

from .arm import XArmVelocityArm

__all__ = ["XArmVelocityArm"]
