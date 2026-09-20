"""XArmVelocityArm: a Viam `Arm` component with velocity/servo/torque extras.

Extends xArm control past what the stock `viam-modules/viam-ufactory` module
and Viam's `Arm` API expose:

  - joint velocity streaming (xArm mode 4, `vc_set_joint_velocity`)
  - cartesian velocity streaming (xArm mode 5, `vc_set_cartesian_velocity`)
  - servo-angle streaming (xArm mode 1, `set_servo_angle_j`, ~250 Hz)
  - joint torque / current readout (`set_report_tau_or_i` + `get_joints_torque`)
  - a configurable DOF (5 / 6 / 7) so xArm5 -- which the stock module does not
    support -- works too

The standard `Arm` methods (`move_to_position`, `move_to_joint_positions`,
`get_joint_positions`, `get_end_position`, `stop`, `is_moving`) are kept as
thin position-mode wrappers so this component still behaves like a normal
arm for motion planning. The extras are exposed only through `do_command`,
since they are outside Viam's `Arm` API surface.

The real xArm SDK (`xarm.wrapper.XArmAPI`) is imported lazily -- only inside
`_load_real_xarm_api()`, and only when a real (non-injected) backend is
needed -- so this module imports fine, and is fully unit-testable, on a
machine with no `xarm` package installed (e.g. import it and swap in
`fake_backend.FakeXArm` for tests).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components import KinematicsReturn
from viam.errors import MethodNotImplementedError
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.proto.component.arm import JointPositions
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .watchdog import Watchdog

LOGGER = getLogger(__name__)

DEFAULT_DOF = 6
VALID_DOF = (5, 6, 7)
DEFAULT_MAX_JOINT_VELOCITY_DEG_S = 30.0
DEFAULT_WATCHDOG_TIMEOUT_S = 0.2
MIN_FIRMWARE_FOR_VELOCITY_MODE = "1.6.8"


class ControlMode:
    """Logical control modes this component understands, mapped to xArm SDK `set_mode()` values."""

    POSITION = "position"
    SERVO = "servo"
    VELOCITY = "velocity"
    CARTESIAN_VELOCITY = "cartesian_velocity"


_MODE_TO_XARM_MODE: Dict[str, int] = {
    ControlMode.POSITION: 0,
    ControlMode.SERVO: 1,
    ControlMode.VELOCITY: 4,
    ControlMode.CARTESIAN_VELOCITY: 5,
}

_STREAMING_MODES = (ControlMode.SERVO, ControlMode.VELOCITY, ControlMode.CARTESIAN_VELOCITY)


def _load_real_xarm_api():
    """Lazily import the real xArm SDK. Only called when no fake backend was injected."""
    from xarm.wrapper import XArmAPI  # noqa: PLC0415 -- intentionally lazy

    return XArmAPI


class XArmVelocityArm(Arm):
    """Modular xArm Arm component with velocity control, servo streaming, and torque readout."""

    MODEL: ClassVar[Model] = Model(ModelFamily("hack", "xarm-velocity"), "arm")

    def __init__(self, name: str, *, backend_factory: Optional[Callable[["XArmVelocityArm"], Any]] = None):
        super().__init__(name)
        self.logger = LOGGER

        # config (filled in by reconfigure())
        self.host: str = ""
        self.dof: int = DEFAULT_DOF
        self.max_joint_velocity: float = DEFAULT_MAX_JOINT_VELOCITY_DEG_S
        self.watchdog_timeout_s: float = DEFAULT_WATCHDOG_TIMEOUT_S
        self.is_radian: bool = False

        # `backend_factory(self) -> backend` lets tests inject a FakeXArm instead of the
        # real XArmAPI. Production code leaves this None and gets the real SDK.
        self._backend_factory = backend_factory
        self._arm: Any = None

        self._mode: str = ControlMode.POSITION
        self._enabled: bool = False
        self._moving: bool = False
        self._watchdog = Watchdog(self.watchdog_timeout_s, self._on_watchdog_timeout)

    # ------------------------------------------------------------------
    # Viam resource lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        instance = cls(config.name)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        attrs = struct_to_dict(config.attributes)

        host = attrs.get("host")
        if not host or not str(host).strip():
            raise ValueError("'host' attribute (xArm controller IP) is required")

        dof = attrs.get("dof", DEFAULT_DOF)
        try:
            dof_int = int(dof)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'dof' must be an integer, got {dof!r}") from exc
        if dof_int not in VALID_DOF:
            raise ValueError(f"'dof' must be one of {VALID_DOF}, got {dof_int}")

        max_v = attrs.get("max_joint_velocity", DEFAULT_MAX_JOINT_VELOCITY_DEG_S)
        try:
            max_v_f = float(max_v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'max_joint_velocity' must be a number, got {max_v!r}") from exc
        if max_v_f <= 0:
            raise ValueError(f"'max_joint_velocity' must be > 0, got {max_v_f}")

        wd = attrs.get("watchdog_timeout_s", DEFAULT_WATCHDOG_TIMEOUT_S)
        try:
            wd_f = float(wd)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'watchdog_timeout_s' must be a number, got {wd!r}") from exc
        if wd_f <= 0:
            raise ValueError(f"'watchdog_timeout_s' must be > 0, got {wd_f}")

        is_radian = attrs.get("is_radian", False)
        if not isinstance(is_radian, bool):
            raise ValueError(f"'is_radian' must be a boolean, got {is_radian!r}")

        return []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        # validate_config already raises ValueError for bad config; re-run it here too so
        # reconfigure() is safe to call directly (e.g. from tests) without a separate validate step.
        self.validate_config(config)
        attrs = struct_to_dict(config.attributes)

        self.host = str(attrs["host"])
        self.dof = int(attrs.get("dof", DEFAULT_DOF))
        self.max_joint_velocity = float(attrs.get("max_joint_velocity", DEFAULT_MAX_JOINT_VELOCITY_DEG_S))
        self.watchdog_timeout_s = float(attrs.get("watchdog_timeout_s", DEFAULT_WATCHDOG_TIMEOUT_S))
        self.is_radian = bool(attrs.get("is_radian", False))

        self._watchdog.stop()
        self._watchdog = Watchdog(self.watchdog_timeout_s, self._on_watchdog_timeout)

        if self._arm is None:
            self._connect()

        self.logger.info(
            "xarm-velocity reconfigured: host=%s dof=%d max_joint_velocity=%.2f watchdog_timeout_s=%.3f is_radian=%s",
            self.host,
            self.dof,
            self.max_joint_velocity,
            self.watchdog_timeout_s,
            self.is_radian,
        )

    def _connect(self) -> None:
        if self._backend_factory is not None:
            self._arm = self._backend_factory(self)
        else:
            XArmAPI = _load_real_xarm_api()
            self._arm = XArmAPI(self.host, is_radian=self.is_radian)
        self._arm.motion_enable(enable=True)
        self._arm.set_mode(_MODE_TO_XARM_MODE[ControlMode.POSITION])
        self._arm.set_state(0)
        self._mode = ControlMode.POSITION
        self._enabled = True

    async def close(self):
        self._watchdog.stop()
        if self._arm is not None:
            disconnect = getattr(self._arm, "disconnect", None)
            if disconnect is not None:
                disconnect()

    # ------------------------------------------------------------------
    # Standard Arm API -- thin position-mode wrappers
    # ------------------------------------------------------------------
    async def get_end_position(self, *, extra=None, timeout=None, **kwargs) -> Pose:
        _, pose = self._arm.get_position(is_radian=self.is_radian)
        x, y, z, roll, pitch, yaw = pose[:6]
        if not self.is_radian:
            roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
        # NOTE: this is a best-effort RPY -> orientation-vector projection (yaw only), good
        # enough for status/telemetry. This module's focus -- and what's exercised precisely --
        # is velocity/servo/torque control; use get_joint_positions/move_to_joint_positions for
        # exact, round-trippable arm state.
        return Pose(x=x, y=y, z=z, o_x=0.0, o_y=0.0, o_z=1.0, theta=math.degrees(yaw))

    async def move_to_position(self, pose: Pose, *, extra=None, timeout=None, **kwargs):
        self._ensure_enabled()
        self._ensure_mode(ControlMode.POSITION)
        self._moving = True
        try:
            yaw = math.radians(pose.theta) if not self.is_radian else pose.theta
            self._arm.set_position(
                x=pose.x,
                y=pose.y,
                z=pose.z,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
                is_radian=self.is_radian,
                wait=True,
            )
        finally:
            self._moving = False

    async def move_to_joint_positions(self, positions: JointPositions, *, extra=None, timeout=None, **kwargs):
        angles = list(positions.values)
        self._require_length(angles, self.dof, "positions")
        self._ensure_enabled()
        self._ensure_mode(ControlMode.POSITION)
        self._moving = True
        try:
            self._arm.set_servo_angle(angle=angles, is_radian=self.is_radian, wait=True)
        finally:
            self._moving = False

    async def get_joint_positions(self, *, extra=None, timeout=None, **kwargs) -> JointPositions:
        _, angles = self._arm.get_servo_angle(is_radian=self.is_radian)
        return JointPositions(values=list(angles[: self.dof]))

    async def stop(self, *, extra=None, timeout=None, **kwargs):
        self._do_stop()

    async def is_moving(self) -> bool:
        return self._moving

    async def get_kinematics(self, *, extra=None, timeout=None, **kwargs) -> KinematicsReturn:
        raise MethodNotImplementedError("get_kinematics")

    # ------------------------------------------------------------------
    # do_command: the velocity / servo / torque extras
    # ------------------------------------------------------------------
    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd == "set_joint_velocity":
            result = self._cmd_set_joint_velocity(command)
            duration = command.get("duration")
            if duration:
                # Optional convenience: auto-stop after `duration` seconds. This is in
                # addition to, not instead of, the watchdog -- the watchdog is what
                # protects against a client that never calls back at all.
                asyncio.create_task(self._auto_stop_after(float(duration)))
            return result
        if cmd == "set_cartesian_velocity":
            return self._cmd_set_cartesian_velocity(command)
        if cmd == "servo_joint":
            return self._cmd_servo_joint(command)
        if cmd == "get_joint_torques":
            return self._cmd_get_joint_torques()
        if cmd == "get_joint_currents":
            return self._cmd_get_joint_currents()
        if cmd == "set_control_mode":
            return self._cmd_set_control_mode(command)
        if cmd == "stop":
            return self._cmd_stop()
        raise ValueError(f"unknown do_command 'command': {cmd!r}")

    def _cmd_set_joint_velocity(self, command: Mapping[str, ValueTypes]):
        velocities = command.get("velocities")
        if velocities is None:
            raise ValueError("'velocities' is required for set_joint_velocity")
        velocities = [float(v) for v in velocities]
        self._require_length(velocities, self.dof, "velocities")
        clamped = [self._clamp_joint_velocity(v) for v in velocities]

        self._ensure_enabled()
        self._ensure_mode(ControlMode.VELOCITY)
        self._arm.vc_set_joint_velocity(clamped, is_radian=self.is_radian)
        self._watchdog.feed()
        self._moving = any(v != 0.0 for v in clamped)

        duration = command.get("duration")
        result: Dict[str, ValueTypes] = {
            "ok": True,
            "mode": self._mode,
            "velocities": clamped,
            "clamped": clamped != velocities,
        }
        return result

    def _cmd_set_cartesian_velocity(self, command: Mapping[str, ValueTypes]):
        velocity = command.get("velocity")
        if velocity is None:
            raise ValueError("'velocity' ([vx,vy,vz,wx,wy,wz]) is required for set_cartesian_velocity")
        velocity = [float(v) for v in velocity]
        self._require_length(velocity, 6, "velocity")

        self._ensure_enabled()
        self._ensure_mode(ControlMode.CARTESIAN_VELOCITY)
        self._arm.vc_set_cartesian_velocity(velocity, is_radian=self.is_radian)
        self._watchdog.feed()
        self._moving = any(v != 0.0 for v in velocity)

        return {"ok": True, "mode": self._mode, "velocity": velocity}

    def _cmd_servo_joint(self, command: Mapping[str, ValueTypes]):
        angles = command.get("angles")
        if angles is None:
            raise ValueError("'angles' is required for servo_joint")
        angles = [float(a) for a in angles]
        self._require_length(angles, self.dof, "angles")

        self._ensure_enabled()
        self._ensure_mode(ControlMode.SERVO)
        self._arm.set_servo_angle_j(angles, is_radian=self.is_radian)
        self._watchdog.feed()
        self._moving = True

        return {"ok": True, "mode": self._mode, "angles": angles}

    async def _auto_stop_after(self, duration_s: float) -> None:
        await asyncio.sleep(duration_s)
        self._do_stop()

    def _cmd_get_joint_torques(self):
        self._arm.set_report_tau_or_i(0)  # 0 = torque
        code, torques = self._arm.get_joints_torque()
        return {"ok": code == 0, "code": code, "torques": list(torques[: self.dof])}

    def _cmd_get_joint_currents(self):
        self._arm.set_report_tau_or_i(1)  # 1 = current
        code, currents = self._arm.get_joints_torque()
        return {"ok": code == 0, "code": code, "currents": list(currents[: self.dof])}

    def _cmd_set_control_mode(self, command: Mapping[str, ValueTypes]):
        mode = command.get("mode")
        if mode not in _MODE_TO_XARM_MODE:
            raise ValueError(f"unknown mode {mode!r}; expected one of {list(_MODE_TO_XARM_MODE)}")
        self._ensure_enabled()
        self._ensure_mode(mode)
        if mode not in _STREAMING_MODES:
            self._watchdog.stop()
        return {"ok": True, "mode": self._mode}

    def _cmd_stop(self):
        self._do_stop()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Safety: clamp, length checks, mode transitions, watchdog
    # ------------------------------------------------------------------
    def _clamp_joint_velocity(self, v: float) -> float:
        limit = self.max_joint_velocity
        if v > limit:
            return limit
        if v < -limit:
            return -limit
        return v

    @staticmethod
    def _require_length(values: Sequence[Any], expected: int, field: str) -> None:
        if len(values) != expected:
            raise ValueError(f"'{field}' must have length {expected}, got {len(values)}")

    def _ensure_enabled(self) -> None:
        if not self._enabled or self._arm is None:
            raise RuntimeError("arm is not enabled/connected; cannot accept motion commands")

    def _ensure_mode(self, mode: str) -> None:
        if self._mode == mode:
            return
        xarm_mode = _MODE_TO_XARM_MODE[mode]
        self._arm.set_mode(xarm_mode)
        self._arm.set_state(0)
        self._mode = mode

    def _do_stop(self) -> None:
        self._watchdog.stop()
        try:
            if self._arm is not None:
                if self._mode == ControlMode.VELOCITY:
                    self._arm.vc_set_joint_velocity([0.0] * self.dof, is_radian=self.is_radian)
                elif self._mode == ControlMode.CARTESIAN_VELOCITY:
                    self._arm.vc_set_cartesian_velocity([0.0] * 6, is_radian=self.is_radian)
                self._arm.set_state(4)
        finally:
            self._moving = False
            self._mode = ControlMode.POSITION

    def _on_watchdog_timeout(self) -> None:
        """Dead-man handler: fires on a background thread when a streaming command goes stale."""
        self.logger.warning(
            "xarm-velocity watchdog timeout (%.3fs, mode=%s): no fresh command, stopping arm",
            self.watchdog_timeout_s,
            self._mode,
        )
        try:
            if self._arm is not None:
                if self._mode == ControlMode.VELOCITY:
                    self._arm.vc_set_joint_velocity([0.0] * self.dof, is_radian=self.is_radian)
                elif self._mode == ControlMode.CARTESIAN_VELOCITY:
                    self._arm.vc_set_cartesian_velocity([0.0] * 6, is_radian=self.is_radian)
                self._arm.set_state(4)
        except Exception:  # noqa: BLE001 -- watchdog must never raise into the timer thread
            self.logger.exception("xarm-velocity watchdog stop failed")
        finally:
            self._moving = False
            self._mode = ControlMode.POSITION


Registry.register_resource_creator(
    Arm.API,
    XArmVelocityArm.MODEL,
    ResourceCreatorRegistration(XArmVelocityArm.new, XArmVelocityArm.validate_config),
)
