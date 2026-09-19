import os
from dataclasses import dataclass

from viam.components.gripper import Gripper
from viam.robot.client import RobotClient

# UFactory two-finger range is 0 (closed) to 850 (fully open).
# Place uses a ready gap; pick opens fully, then torque-limits the close.
FULL_OPEN_POS = 850.0
OPEN_POS = float(os.environ.get("GRIPPER_OPEN_POS", 520))
GRASP_POS = float(os.environ.get("GRIPPER_GRASP_POS", 0))
GRASP_TORQUE = float(os.environ.get("GRIPPER_TORQUE", 15))
GRASP_SPEED = float(os.environ.get("GRIPPER_SPEED", 2000))
POS_SLACK = 25.0


@dataclass
class Grasp:
    holding: bool
    pos: float
    torque: float


class GripperComponent:
    def __init__(self, machine: RobotClient, name: str | None = None) -> None:
        self.name = name or os.environ.get("GRIPPER_NAME", "gripper")
        self._gripper = Gripper.from_robot(robot=machine, name=self.name)
        self.last_grasp: Grasp | None = None

    async def do(self, command: dict) -> dict:
        return dict(await self._gripper.do_command(command))

    async def get_pos(self) -> float:
        resp = await self.do({"get": True})
        for key in ("pos", "position", "gripper_position"):
            if key in resp and resp[key] is not None:
                return float(resp[key])
        return -1.0

    async def set_pos(self, pos: float, force: bool = False) -> float:
        pos = max(0.0, min(850.0, float(pos)))
        if not force:
            current = await self.get_pos()
            if current >= 0 and abs(current - pos) <= POS_SLACK:
                return current
        resp = await self.do({"set": pos})
        for key in ("position", "pos", "gripper_position"):
            if key in resp and resp[key] is not None:
                return float(resp[key])
        return pos

    async def hold_open(self) -> None:
        await self.set_pos(OPEN_POS)

    async def open_full(self) -> None:
        try:
            await self._gripper.open(timeout=10)
        except Exception as exc:
            print(f"  gripper.open failed ({exc}); setting 850")
        pos = await self.set_pos(FULL_OPEN_POS, force=True)
        print(f"  gripper open jaws={pos:.0f}/850")

    async def open(self, timeout: float = 10) -> None:
        await self.open_full()

    async def _holding(self, timeout: float) -> bool | None:
        try:
            holding = await self._gripper.is_holding_something(timeout=timeout)
            if isinstance(holding, bool):
                return holding
            status = getattr(holding, "is_holding_something", None)
            if status is not None:
                return bool(status)
        except Exception:
            pass
        return None

    async def grab(self, timeout: float = 10) -> Grasp:
        try:
            await self.do(
                {
                    "grab_with_torque": {
                        "position": GRASP_POS,
                        "speed": GRASP_SPEED,
                        "torque": GRASP_TORQUE,
                    }
                }
            )
        except Exception as exc:
            print(f"  grab_with_torque unavailable ({exc}); closing to {GRASP_POS:.0f}")
            await self.set_pos(GRASP_POS)

        pos = await self.get_pos()
        holding = await self._holding(timeout)
        if holding is None:
            holding = pos > GRASP_POS + POS_SLACK
        self.last_grasp = Grasp(holding=holding, pos=pos, torque=GRASP_TORQUE)
        print(
            f"  grasp jaws={pos:.0f}/850  "
            f"torque={GRASP_TORQUE:.0f}%  holding={holding}"
        )
        return self.last_grasp
