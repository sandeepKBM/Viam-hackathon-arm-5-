import os

from viam.components.camera import Camera  # noqa: F401
from viam.components.gripper import Gripper  # noqa: F401
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.services.vision import VisionClient  # noqa: F401

# Default viam-server gRPC port. When connecting locally you dial the machine's
# LAN address (e.g. 192.168.1.150:8080) or, if the port is reverse-tunneled to
# this host, localhost:8080.
_DEFAULT_LOCAL_ADDRESS = "localhost:8080"


def _env(*names: str, default: str | None = None) -> str | None:
    """First non-empty value among NAMES.

    Supports both the ``VIAM_``-prefixed names used in the live ``.env`` and the
    legacy unprefixed names from ``.env.example`` -- so either populates the
    same setting without a silent break.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _require_env(*names: str) -> str:
    value = _env(*names)
    if not value:
        raise RuntimeError(
            f"Missing {' / '.join(names)}. Copy .env.example to .env and paste "
            "values from the machine CONNECT tab in the Viam app."
        )
    return value


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _live_connection_enabled() -> bool:
    return _flag("VIAM_ALLOW_LIVE")


def _local_mode() -> bool:
    return _flag("VIAM_LOCAL")


async def connect_machine() -> RobotClient:
    # --- Credential-hygiene guard (unchanged) ---
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

    if _local_mode():
        return await _connect_local()
    return await _connect_cloud()


async def _connect_cloud() -> RobotClient:
    """Connect via the Viam cloud FQDN (WebRTC signaling through app.viam.com).

    This is the default and works from anywhere -- both the client and the
    machine reach out to Viam cloud, so no shared LAN is required.
    """
    opts = RobotClient.Options.with_api_key(
        api_key=_require_env("VIAM_API_KEY", "API_KEY"),
        api_key_id=_require_env("VIAM_API_KEY_ID", "API_KEY_ID"),
    )
    opts.check_connection_interval = 0
    opts.attempt_reconnect_interval = 0
    return await RobotClient.at_address(
        _require_env("VIAM_MACHINE_ADDRESS", "MACHINE_ADDRESS"), opts
    )


async def _connect_local() -> RobotClient:
    """Connect DIRECTLY to viam-server over gRPC, bypassing Viam cloud signaling.

    Enable with ``VIAM_LOCAL=1``. Intended for:
      (a) this client running on the same host/LAN as viam-server, dialing the
          machine's LAN address (e.g. ``192.168.1.150:8080``), or
      (b) the machine's gRPC port reverse-tunneled to this host, dialing
          ``localhost:8080``.

    WebRTC is always disabled here: a single forwarded TCP port (or a direct LAN
    dial) cannot carry WebRTC's separately-negotiated peer connection, so we use
    plain gRPC. Address form is HOST:PORT via ``VIAM_MACHINE_LOCAL_ADDRESS``
    (default ``localhost:8080``).

    Auth: viam-server normally still requires the API key even locally, so we
    keep API-key auth by default. If your machine's network config explicitly
    allows insecure local connections, set ``VIAM_LOCAL_INSECURE=1`` to drop TLS
    and credentials entirely.
    """
    address = _env(
        "VIAM_MACHINE_LOCAL_ADDRESS",
        "MACHINE_LOCAL_ADDRESS",
        default=_DEFAULT_LOCAL_ADDRESS,
    )

    if _flag("VIAM_LOCAL_INSECURE", default=False):
        # No TLS, no credentials -- only works if viam-server is configured to
        # allow insecure connections. Try this only if the API-key path below
        # fails with a TLS/handshake error.
        dial = DialOptions(disable_webrtc=True, insecure=True)
    else:
        # Keep API-key auth, but force a direct (non-WebRTC) gRPC connection.
        dial = DialOptions(
            disable_webrtc=True,
            auth_entity=_require_env("VIAM_API_KEY_ID", "API_KEY_ID"),
            credentials=Credentials(
                type="api-key",
                payload=_require_env("VIAM_API_KEY", "API_KEY"),
            ),
        )

    opts = RobotClient.Options(
        dial_options=dial,
        check_connection_interval=0,
        attempt_reconnect_interval=0,
    )
    return await RobotClient.at_address(address, opts)


# ---------------------------------------------------------------------------
# Shared connection (connect ONCE, reuse everywhere)
# ---------------------------------------------------------------------------
# Measured: the connect handshake over Viam cloud costs ~4s (WebRTC/ICE + auth),
# paid every time you open a client. Per-call RTT after that is only ~36ms. So
# the single biggest latency win is to NEVER reconnect: open one RobotClient at
# startup and reuse it for the whole run. get_machine() caches the client;
# warm_machine() lets you pay the handshake up front (e.g. overlap it with model
# loading) instead of blocking the first command.

_shared_machine: RobotClient | None = None
_shared_lock: "object | None" = None


async def get_machine() -> RobotClient:
    """Return a process-wide shared RobotClient, connecting once on first use.

    Reuse this instead of calling connect_machine() repeatedly -- each
    connect_machine() pays the full ~4s cloud handshake, while the shared client
    pays it once. Safe to await from multiple coroutines: concurrent first-callers
    are serialized so only one connection is opened.
    """
    global _shared_machine, _shared_lock
    import asyncio

    if _shared_machine is not None:
        return _shared_machine
    if _shared_lock is None:
        _shared_lock = asyncio.Lock()
    async with _shared_lock:  # type: ignore[union-attr]
        if _shared_machine is None:
            _shared_machine = await connect_machine()
        return _shared_machine


async def warm_machine() -> RobotClient:
    """Eagerly open the shared connection (alias of get_machine()).

    Call at program start so the ~4s handshake overlaps with imports / model
    loading rather than stalling the first real command.
    """
    return await get_machine()


async def close_machine() -> None:
    """Close and clear the shared connection, if any."""
    global _shared_machine
    if _shared_machine is not None:
        try:
            await _shared_machine.close()
        finally:
            _shared_machine = None
