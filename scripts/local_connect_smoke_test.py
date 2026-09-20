#!/usr/bin/env python3
"""Smoke-test a LOCAL/direct connection to the arm (bypasses Viam cloud).

Verifies the "gateway" is up: connects via ``components.connection.connect_machine``
in local mode (WebRTC disabled) and prints the machine's resource names, so you
can confirm the reverse tunnel / LAN dial works before running any motion.

Prereqs (on the VM):
  1. Reverse tunnel up (run ``scripts/tunnel_to_arm.sh`` on your laptop), OR run
     this on a host that shares the arm's LAN.
  2. ``.env`` has VIAM_API_KEY / VIAM_API_KEY_ID.
  3. Opt in explicitly:  ``export VIAM_ALLOW_LIVE=1 VIAM_LOCAL=1``
     and point at the tunnel:  ``export VIAM_MACHINE_LOCAL_ADDRESS=localhost:8080``
     (or the arm's LAN IP:port, e.g. 192.168.1.150:8080).

Run:
  VIAM_ALLOW_LIVE=1 VIAM_LOCAL=1 python scripts/local_connect_smoke_test.py
"""
import asyncio
import os
import sys

# Make the repo root importable when run as `python scripts/local_connect_smoke_test.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv


async def main() -> int:
    load_dotenv()
    # Convenience default: this script is *for* local mode.
    os.environ.setdefault("VIAM_LOCAL", "1")

    if os.environ.get("VIAM_ALLOW_LIVE", "").strip().lower() not in ("1", "true", "yes", "on"):
        print(
            "Refusing to connect: set VIAM_ALLOW_LIVE=1 to explicitly enable a "
            "live connection (credential-hygiene guard).",
            file=sys.stderr,
        )
        return 2

    # Imported here so the guard message above prints without SDK import cost.
    from components.connection import connect_machine

    addr = os.environ.get("VIAM_MACHINE_LOCAL_ADDRESS", "localhost:8080")
    insecure = os.environ.get("VIAM_LOCAL_INSECURE", "0").strip().lower() in ("1", "true", "yes", "on")
    print(f"[local] dialing {addr}  (webrtc disabled, insecure={insecure}) ...")

    machine = await connect_machine()
    try:
        print("Connected. Resources:")
        for name in machine.resource_names:
            print(f"  - {name}")
    finally:
        await machine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
