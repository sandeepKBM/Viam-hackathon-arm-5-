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


def _live_connection_enabled() -> bool:
    return os.environ.get("VIAM_ALLOW_LIVE", "").strip().lower() in ("1", "true", "yes")


async def connect_machine() -> RobotClient:
    # --- Credential-hygiene guard ---
    # Refuse to open a LIVE connection to the real machine unless the user has
    # EXPLICITLY opted in with VIAM_ALLOW_LIVE=1. This blocks scripts/agents
    # (and subagents) from connecting to the arm accidentally -- having the
    # API key in .env is no longer enough on its own. Offline/dry-run/sim/tests
    # never call this, so they are unaffected.
    if not _live_connection_enabled():
        raise RuntimeError(
            "Live Viam connection is DISABLED (credential-hygiene guard). "
            "Set VIAM_ALLOW_LIVE=1 to explicitly enable connecting to the real "
            "machine in THIS session. This prevents accidental connects by "
            "scripts/agents; offline/dry-run/sim needs no connection."
        )
    opts = RobotClient.Options.with_api_key(
        api_key=_require_env("API_KEY"),
        api_key_id=_require_env("API_KEY_ID"),
    )
    opts.check_connection_interval = 0
    opts.attempt_reconnect_interval = 0
    return await RobotClient.at_address(_require_env("MACHINE_ADDRESS"), opts)
