"""Entry point for the xarm_velocity Viam module.

Run directly (as `run.sh` does) or via `viam-server`'s module manager, which
invokes this with the module's UNIX socket path as argv[1]
(`Module.from_args()` picks that up automatically).
"""

import asyncio
import os
import sys

# Allow running this file directly (`python main.py`) regardless of cwd, and make the
# sibling `arm`/`fake_backend`/`watchdog` modules importable as a package either way.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from viam.components.arm import Arm
from viam.module.module import Module

from xarm_velocity.arm import XArmVelocityArm


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(Arm.API, XArmVelocityArm.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
