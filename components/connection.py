import os

from viam.components.camera import Camera  # noqa: F401
from viam.components.gripper import Gripper  # noqa: F401
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient  # noqa: F401


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing {name}. Copy .env.example to .env and paste values "
            "from the machine CONNECT tab in the Viam app."
        )
    return value


async def connect_machine() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=_require_env("API_KEY"),
        api_key_id=_require_env("API_KEY_ID"),
    )
    opts.check_connection_interval = 0
    opts.attempt_reconnect_interval = 0
    return await RobotClient.at_address(_require_env("MACHINE_ADDRESS"), opts)
