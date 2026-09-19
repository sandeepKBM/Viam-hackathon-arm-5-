"""Helper for connecting to a Viam robot from environment variables.

Usage:
    import asyncio
    from arm5.hardware.robot import connect

    async def main():
        robot = await connect()
        print(robot.resource_names)
        await robot.close()

    asyncio.run(main())

Environment variables:
    ROBOT_ADDRESS    -- e.g. "my-machine-main.abcd1234.viam.cloud"
    ROBOT_API_KEY    -- Viam API key secret
    ROBOT_API_KEY_ID -- Viam API key id
"""

from __future__ import annotations

import os

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions


async def connect() -> RobotClient:
    """Connect to a Viam robot using credentials from the environment.

    Raises:
        RuntimeError: if any of ROBOT_ADDRESS, ROBOT_API_KEY, or
            ROBOT_API_KEY_ID is not set.

    TODO: support reading credentials from `config/robot.example.json` (or a
    local, non-example override) as a fallback when env vars are unset.
    """
    address = os.environ.get("ROBOT_ADDRESS")
    api_key = os.environ.get("ROBOT_API_KEY")
    api_key_id = os.environ.get("ROBOT_API_KEY_ID")

    if not address or not api_key or not api_key_id:
        raise RuntimeError(
            "ROBOT_ADDRESS, ROBOT_API_KEY, and ROBOT_API_KEY_ID must all be set"
        )

    credentials = Credentials(type="api-key", payload=api_key)
    dial_options = DialOptions(auth_entity=api_key_id, credentials=credentials)
    options = RobotClient.Options(
        refresh_interval=0,
        dial_options=dial_options,
    )
    return await RobotClient.at_address(address, options)
