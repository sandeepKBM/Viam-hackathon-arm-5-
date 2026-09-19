# arm5.controls

Owns the low(er)-level control loop that turns planner setpoints into
commands sent to `arm5.hardware`.

Key files:
- `controllers.py` -- `Controller` ABC + `SetpointController` stub for
  joint/cartesian setpoint tracking. Assumes a fixed control rate (see the
  module docstring) rather than adaptive timing.
- `safety.py` -- joint/cartesian limit checks, an e-stop hook, and
  workspace-bounds checking, meant to gate every command before it reaches
  hardware.

TODO: pick an actual control rate for the hackathon rig, implement the
setpoint-tracking math, and decide whether safety checks run in-process or
as a separate watchdog.
