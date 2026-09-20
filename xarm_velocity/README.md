# xarm_velocity

A Viam module that extends xArm control with capabilities the stock
`viam-modules/viam-ufactory` module and Viam's `Arm` API don't expose:

- **Joint velocity control** (xArm mode 4, `vc_set_joint_velocity`, firmware >= 1.6.8)
- **Cartesian velocity control** (xArm mode 5, `vc_set_cartesian_velocity`)
- **Servo-angle streaming** (xArm mode 1, `set_servo_angle_j`, ~250 Hz)
- **Joint torque / current readout** (`set_report_tau_or_i` + `get_joints_torque`) --
  the xArm has no torque **control** mode, only read-only torque/current telemetry.
- **Configurable DOF** (5 / 6 / 7) -- the stock module doesn't support xArm5.

Model: `hack:xarm-velocity:arm`, API: `rdk:component:arm` (a real `viam.components.arm.Arm`
subclass, so it still works for normal position-mode motion planning).

## Why this exists

Viam's `Arm` API is position-only (`move_to_position` / `move_to_joint_positions`);
there is no velocity or torque method on the API. The xArm hardware itself has no
torque-*control* mode either -- its `XArmAPI.set_mode()` modes are: `0` = position,
`1` = servo/streaming-position, `4` = joint velocity, `5` = cartesian velocity.
Torque/current are read-only. This module fills that gap by exposing the extra
xArm-SDK capabilities through `do_command`, while remaining a drop-in `Arm` for
everything position-mode.

## Files

- `arm.py` -- the `XArmVelocityArm` component (the actual logic).
- `fake_backend.py` -- `FakeXArm`, a recording stand-in for `xarm.wrapper.XArmAPI`
  used by the test suite (and safe to use for any offline dry run).
- `watchdog.py` -- the dead-man timer used to auto-stop stale velocity/servo streams.
- `main.py` -- module entrypoint; registers `XArmVelocityArm` with a Viam `Module`.
- `meta.json`, `run.sh`, `requirements.txt` -- module packaging/deployment.
- `tests/` -- offline unit tests (FakeXArm only, no hardware, no `xarm` SDK).

## Config attributes

| attribute             | type    | required | default | notes                                   |
|------------------------|---------|----------|---------|------------------------------------------|
| `host`                 | string  | yes      | --      | xArm controller IP                       |
| `dof`                  | int     | no       | `6`     | must be `5`, `6`, or `7`                 |
| `max_joint_velocity`   | float   | no       | `30.0`  | deg/s (or rad/s if `is_radian`); per-joint clamp, must be > 0 |
| `watchdog_timeout_s`   | float   | no       | `0.2`   | dead-man timeout, must be > 0            |
| `is_radian`            | bool    | no       | `false` | units for angles/velocities in/out       |

Example machine config fragment:

```json
{
  "name": "arm1",
  "api": "rdk:component:arm",
  "model": "hack:xarm-velocity:arm",
  "attributes": {
    "host": "192.168.1.185",
    "dof": 6,
    "max_joint_velocity": 20.0,
    "watchdog_timeout_s": 0.2,
    "is_radian": false
  }
}
```

`validate_config` rejects missing/invalid `host`, `dof` outside `{5,6,7}`, and
non-positive `max_joint_velocity` / `watchdog_timeout_s` at config time (before
the arm ever connects).

## Standard Arm API

`get_end_position`, `move_to_position`, `move_to_joint_positions`,
`get_joint_positions`, `stop`, `is_moving` are implemented as thin position-mode
(`set_mode(0)`) wrappers over the xArm SDK, so the component works as a normal
arm for motion planning. `get_end_position`/`move_to_position` use a best-effort
yaw-only RPY -> orientation-vector projection for telemetry/rough placement --
for exact, round-trippable state use `get_joint_positions` /
`move_to_joint_positions`. `get_kinematics` is not implemented (raises
`MethodNotImplementedError`); this module's focus is velocity/servo/torque, not
kinematics files.

## do_command API

All extras are dispatched through `do_command({"command": ..., ...})`:

- **`set_joint_velocity`** -- `{"command": "set_joint_velocity", "velocities": [<dof floats>], "duration": <s, optional>}`
  Switches to xArm mode 4 and calls `vc_set_joint_velocity`. Each element is
  clamped to `+/- max_joint_velocity`. Feeds the watchdog. `duration`, if given,
  is an optional convenience that auto-stops after that many seconds (in
  *addition* to, not instead of, the watchdog). Returns
  `{"ok", "mode", "velocities" (post-clamp), "clamped" (bool)}`.

- **`set_cartesian_velocity`** -- `{"command": "set_cartesian_velocity", "velocity": [vx,vy,vz,wx,wy,wz]}`
  Switches to xArm mode 5 and calls `vc_set_cartesian_velocity`. Feeds the
  watchdog. Returns `{"ok", "mode", "velocity"}`.

- **`servo_joint`** -- `{"command": "servo_joint", "angles": [<dof floats>]}`
  Switches to xArm mode 1 and calls `set_servo_angle_j` (intended to be called
  at up to ~250 Hz by the client for smooth streaming). Feeds the watchdog.
  Returns `{"ok", "mode", "angles"}`.

- **`get_joint_torques`** -- `{"command": "get_joint_torques"}`
  Calls `set_report_tau_or_i(0)` then `get_joints_torque()`. Returns
  `{"ok", "code", "torques": [<dof floats>]}`.

- **`get_joint_currents`** -- `{"command": "get_joint_currents"}`
  Calls `set_report_tau_or_i(1)` then `get_joints_torque()`. Returns
  `{"ok", "code", "currents": [<dof floats>]}`.

- **`set_control_mode`** -- `{"command": "set_control_mode", "mode": "position"|"velocity"|"cartesian_velocity"|"servo"}`
  Explicitly transitions mode (`set_mode` + `set_state(0)`) without issuing a
  motion command. Switching to a non-streaming mode (`position`) also disarms
  the watchdog. Returns `{"ok", "mode"}`.

- **`stop`** -- `{"command": "stop"}`
  Zeros any active velocity stream, calls `set_state(4)`, disarms the watchdog,
  and resets mode to `position`. Returns `{"ok": true}`. (Also reachable via the
  standard `Arm.stop()` method.)

Unknown `command` values raise `ValueError`. Wrong-length `velocities` /
`velocity` / `angles` raise `ValueError` (checked against configured `dof`, or
6 for cartesian velocity). Velocity/servo commands issued before the arm is
connected/enabled raise `RuntimeError`.

## Safety design

1. **Per-joint velocity clamp** -- every element of `set_joint_velocity`'s
   `velocities` is clamped to `[-max_joint_velocity, +max_joint_velocity]`
   before being sent to the SDK; the response reports whether clamping
   occurred. Array-length mismatches (vs. configured `dof`) are rejected
   outright rather than silently truncated/padded.

2. **Dead-man watchdog** (`watchdog.py`) -- a `threading.Timer`-based timer,
   independent of the asyncio event loop. `set_joint_velocity`,
   `set_cartesian_velocity`, and `servo_joint` all call `watchdog.feed()` on
   every command, which (re)arms a `watchdog_timeout_s` countdown. If no fresh
   streaming command arrives before it elapses, the timer fires
   `_on_watchdog_timeout()` on a background thread: it zeros the active
   velocity stream, calls `set_state(4)` (stop), and resets mode to
   `position`. **Velocity/servo control only continues while the client keeps
   sending fresh commands; any gap longer than `watchdog_timeout_s` halts the
   arm automatically.** Explicit `stop`/`set_control_mode(position)` disarm the
   watchdog cleanly (no stray timeout after an intentional stop).

3. **Enable + mode gating** -- `_ensure_enabled()` raises `RuntimeError` if the
   backend never connected/enabled. `_ensure_mode()` only calls `set_mode` +
   `set_state(0)` when actually changing mode (no redundant SDK calls), keeping
   mode transitions clean and observable.

## Fake vs. real backend

`XArmVelocityArm.__init__` accepts an optional `backend_factory: Callable[[XArmVelocityArm], Any]`.

- **Production** (module loaded normally via `main.py`/`Module.from_args()`):
  no factory is passed, so `_connect()` lazily does
  `from xarm.wrapper import XArmAPI` (only place the SDK is imported) and
  constructs the real `XArmAPI(host, is_radian=...)`.
- **Tests**: `tests/test_arm.py` passes `backend_factory=lambda self: FakeXArm(...)`,
  so `XArmAPI` is never imported and the `xarm` package doesn't need to be
  installed. `FakeXArm` (in `fake_backend.py`) implements the subset of
  `XArmAPI` this module calls (`set_mode`, `set_state`, `motion_enable`,
  `vc_set_joint_velocity`, `vc_set_cartesian_velocity`, `set_servo_angle_j`,
  `set_servo_angle`, `get_servo_angle`, `set_position`, `get_position`,
  `set_report_tau_or_i`, `get_joints_torque`, ...) and records every call in
  `self.calls` for assertions.

Because of this, both `import xarm_velocity` and
`.venv/bin/python -m pytest xarm_velocity/tests/` work with zero hardware and
the `xarm` package absent.

## Deploying to a machine

1. `xarm-python-sdk` must actually be installed for real hardware use (it's in
   `requirements.txt`; `run.sh` creates a local `.venv` and installs it). The
   controller firmware must be **>= 1.6.8** for `vc_set_joint_velocity` (mode 4)
   to be available -- earlier firmware only supports modes 0/1/2.
2. Upload this directory as a local/private module (module ID
   `hack:xarm-velocity`, entrypoint `run.sh`) in the Viam app, or reference it
   as a `local` module pointing at this path in the machine's JSON config.
3. Add a component with `"api": "rdk:component:arm"`,
   `"model": "hack:xarm-velocity:arm"`, and the attributes documented above.
4. **xArm5 note**: set `"dof": 5`. The stock UFactory module does not support
   the 5-DOF xArm; this module treats DOF as a plain config value used purely
   for array-length validation and readout slicing (`get_joint_positions`,
   `get_joint_torques`, etc. all size to `dof`), so 5/6/7-DOF arms are all
   handled the same way.

## Running the tests

```
cd /common/users/ss5772/viam_5
.venv/bin/python -m pytest xarm_velocity/tests/ -q
```
