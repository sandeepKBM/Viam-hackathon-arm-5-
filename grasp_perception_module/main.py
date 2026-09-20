"""Entry point for the grasp_perception_module Viam module.

Run directly (as `run.sh` does) or via `viam-server`'s module manager, which
invokes this with the module's UNIX socket path as argv[1]
(`Module.from_args()` picks that up automatically).
"""

import asyncio
import os
import sys

# Allow running this file directly (`python main.py`) regardless of cwd, make the
# sibling `service`/`fake_backend` modules importable as a package, AND put the
# repo root on sys.path so the reused `components.*` geometry imports resolve.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from viam.module.module import Module
from viam.services.generic import Generic

from grasp_perception_module.service import GraspPerceptionService


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(Generic.API, GraspPerceptionService.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
