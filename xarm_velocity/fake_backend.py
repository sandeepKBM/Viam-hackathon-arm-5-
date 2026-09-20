"""In-process fake standing in for `xarm.wrapper.XArmAPI`.

Implements the subset of the real xArm-Python-SDK surface that
`xarm_velocity.arm.XArmVelocityArm` calls, so the module (and its tests) can
run with no hardware and no `xarm` SDK installed. Every call is recorded in
`self.calls` (a list of `(method_name, args, kwargs)` tuples) so tests can
assert exactly what the component asked the arm to do.

The real backend is only ever imported lazily, inside
`xarm_velocity.arm._load_real_xarm_api()`, so importing this module (or the
component module) never requires the `xarm` package to be installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class FakeXArm:
    """A minimal, stateful stand-in for XArmAPI.

    Only implements what XArmVelocityArm needs:
      - connection / lifecycle: connect, disconnect, motion_enable
      - mode/state: set_mode, set_state, get_mode, get_state
      - position control: set_position, set_servo_angle, get_position
      - joint readout: get_servo_angle
      - servo streaming (mode 1): set_servo_angle_j
      - joint velocity (mode 4): vc_set_joint_velocity
      - cartesian velocity (mode 5): vc_set_cartesian_velocity
      - torque/current readout: set_report_tau_or_i, get_joints_torque
      - stop: emergency_stop / stop / set_state(4)
    """

    def __init__(self, host: str = "0.0.0.0", is_radian: bool = False, dof: int = 6, **kwargs):
        self.host = host
        self.is_radian = is_radian
        self.dof = dof
        self.connected = False
        self.enabled = False
        self.mode = 0
        self.state = 0
        self.report_tau_or_i = 0  # 0 = torque, 1 = current (mirrors xArm SDK semantics)

        self._joint_angles: List[float] = [0.0] * dof
        self._joint_velocities: List[float] = [0.0] * dof
        self._cartesian_velocity: List[float] = [0.0] * 6
        self._position: List[float] = [200.0, 0.0, 200.0, 0.0, 0.0, 0.0]  # x,y,z,roll,pitch,yaw

        self.calls: List[Tuple[str, tuple, dict]] = []

    # -- bookkeeping -----------------------------------------------------
    def _record(self, name: str, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))

    def calls_named(self, name: str) -> List[Tuple[tuple, dict]]:
        return [(a, k) for (n, a, k) in self.calls if n == name]

    # -- lifecycle ---------------------------------------------------------
    def connect(self, *args, **kwargs):
        self._record("connect", *args, **kwargs)
        self.connected = True
        return 0

    def disconnect(self, *args, **kwargs):
        self._record("disconnect", *args, **kwargs)
        self.connected = False
        return 0

    def motion_enable(self, enable: bool = True, servo_id: Optional[int] = None, **kwargs):
        self._record("motion_enable", enable, servo_id=servo_id, **kwargs)
        self.enabled = bool(enable)
        return 0

    def clean_error(self, *args, **kwargs):
        self._record("clean_error", *args, **kwargs)
        return 0

    def clean_warn(self, *args, **kwargs):
        self._record("clean_warn", *args, **kwargs)
        return 0

    # -- mode / state --------------------------------------------------------
    def set_mode(self, mode: int, **kwargs):
        self._record("set_mode", mode, **kwargs)
        self.mode = mode
        return 0

    def set_state(self, state: int = 0, **kwargs):
        self._record("set_state", state, **kwargs)
        self.state = state
        if state == 4:
            self._joint_velocities = [0.0] * self.dof
            self._cartesian_velocity = [0.0] * 6
        return 0

    def get_mode(self):
        self._record("get_mode")
        return 0, self.mode

    def get_state(self):
        self._record("get_state")
        return 0, self.state

    # -- position control (mode 0) --------------------------------------
    def set_position(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None, is_radian=None, wait=False, **kwargs):
        self._record(
            "set_position",
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            is_radian=is_radian,
            wait=wait,
            **kwargs,
        )
        self._position = [
            x if x is not None else self._position[0],
            y if y is not None else self._position[1],
            z if z is not None else self._position[2],
            roll if roll is not None else self._position[3],
            pitch if pitch is not None else self._position[4],
            yaw if yaw is not None else self._position[5],
        ]
        return 0

    def get_position(self, is_radian: Optional[bool] = None):
        self._record("get_position", is_radian=is_radian)
        return 0, list(self._position)

    def set_servo_angle(self, angle=None, is_radian=None, wait=False, **kwargs):
        self._record("set_servo_angle", angle=angle, is_radian=is_radian, wait=wait, **kwargs)
        if angle is not None:
            self._joint_angles = list(angle)
        return 0

    def get_servo_angle(self, is_radian: Optional[bool] = None):
        self._record("get_servo_angle", is_radian=is_radian)
        return 0, list(self._joint_angles)

    # -- servo streaming (mode 1) -----------------------------------------
    def set_servo_angle_j(self, angles, is_radian=None, **kwargs):
        self._record("set_servo_angle_j", angles, is_radian=is_radian, **kwargs)
        self._joint_angles = list(angles)
        return 0

    # -- joint velocity (mode 4) ------------------------------------------
    def vc_set_joint_velocity(self, velocities, is_radian=None, duration=None, **kwargs):
        self._record("vc_set_joint_velocity", velocities, is_radian=is_radian, duration=duration, **kwargs)
        self._joint_velocities = list(velocities)
        return 0

    # -- cartesian velocity (mode 5) --------------------------------------
    def vc_set_cartesian_velocity(self, velocity, is_radian=None, is_tool_coord=False, duration=None, **kwargs):
        self._record(
            "vc_set_cartesian_velocity",
            velocity,
            is_radian=is_radian,
            is_tool_coord=is_tool_coord,
            duration=duration,
            **kwargs,
        )
        self._cartesian_velocity = list(velocity)
        return 0

    # -- torque / current readout ------------------------------------------
    def set_report_tau_or_i(self, tau_or_i: int = 0, **kwargs):
        self._record("set_report_tau_or_i", tau_or_i, **kwargs)
        self.report_tau_or_i = tau_or_i
        return 0

    def get_joints_torque(self):
        self._record("get_joints_torque")
        # Deterministic fake readout: torque/current proportional to joint index.
        values = [round(0.1 * (i + 1), 3) for i in range(self.dof)]
        return 0, values

    # -- stop --------------------------------------------------------------
    def emergency_stop(self, *args, **kwargs):
        self._record("emergency_stop", *args, **kwargs)
        self.set_state(4)
        return 0

    def stop(self, *args, **kwargs):
        self._record("stop", *args, **kwargs)
        self.set_state(4)
        return 0
