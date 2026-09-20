"""Thin westeros-side client for the on-machine grasp-perception module.

WHY THIS EXISTS (measured)
--------------------------
Running perception from westeros means pulling the raw sensor payload across the
Viam cloud relay: ``camera.get_point_cloud()`` is ~12.5 s for a 14.7 MB cloud,
``camera.get_images()`` ~2 s. Control calls are only ~36 ms. So we push the
perception onto the machine as a Viam **generic service** (model
``hackathons:grasp-perception:grasp-service``, built under ``grasp_perception_module/``)
that reads the camera LOCALLY and returns only a small computed result. This
client is the westeros half: it calls the service's ``do_command`` and gets back
a few hundred bytes (~36 ms) instead of a multi-megabyte cloud.

The ``do_command`` contract (kept in lockstep with the module's ``service.py``):

    {"cmd": "health"}
        -> {"ok": true, "camera": str, "segmenter": str, "detector": str}

    {"cmd": "detections"}
        -> {"ok": true, "objects": [{"label": str,
                                     "world_xyz_mm": [x, y, z],
                                     "score": float}, ...]}

    {"cmd": "localize", "object": str, "hint_xy": [x, y] (optional)}
        -> {"ok": true, "world_xyz_mm": [x, y, z],
            "grasp_type": "top_down"|"side"|"inside_outside",
            "approach_vec": [ax, ay, az], "yaw_deg": float,
            "label": str, "n_points": int}
        -> {"ok": false, "error": str}

Only small JSON crosses the network here; raw clouds/images never do.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from viam.robot.client import RobotClient
from viam.services.generic import Generic as GenericService

# Service name as configured on the machine. Override with GRASP_SERVICE_NAME if
# the machine config names it differently.
DEFAULT_SERVICE_NAME = os.environ.get("GRASP_SERVICE_NAME", "grasp-service").strip() or "grasp-service"


class GraspServiceUnavailable(RuntimeError):
    """Raised when the on-machine grasp-perception service isn't reachable.

    Usually means the module hasn't been deployed / configured on the machine
    yet (see grasp_perception_module/README.md for the machine-config snippet).
    """


class GraspServiceClient:
    """Calls the on-machine grasp-perception generic service.

    Parameters
    ----------
    machine:
        A connected ``RobotClient`` (reuse the shared one from
        ``components.connection.get_machine()`` -- do not open a fresh
        connection per call).
    name:
        The generic-service name as configured on the machine.
    """

    def __init__(self, machine: RobotClient, name: str = DEFAULT_SERVICE_NAME) -> None:
        self._machine = machine
        self._name = name
        try:
            self._svc = GenericService.from_robot(machine, name)
        except Exception as exc:  # noqa: BLE001 -- surface a clear, actionable error
            raise GraspServiceUnavailable(
                f"grasp-perception service {name!r} not found on the machine. "
                "Deploy grasp_perception_module/ as a local module and add it to "
                "the machine config (see that module's README)."
            ) from exc

    async def _do(self, command: Dict[str, Any]) -> Dict[str, Any]:
        result = await self._svc.do_command(command)
        # Viam returns a plain dict (proto Struct decoded); normalize to dict.
        return dict(result) if result is not None else {}

    async def health(self) -> Dict[str, Any]:
        """Readiness probe -- confirms the module is up and which resources it uses."""
        return await self._do({"cmd": "health"})

    async def detections(self) -> List[Dict[str, Any]]:
        """All currently detected objects with world-frame centers (small payload)."""
        res = await self._do({"cmd": "detections"})
        if not res.get("ok"):
            raise GraspServiceUnavailable(res.get("error", "detections failed"))
        return list(res.get("objects", []))

    async def localize(
        self,
        obj: str,
        hint_xy: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Localize + classify a grasp for ``obj`` ON THE MACHINE.

        Returns the small grasp dict (world_xyz_mm, grasp_type, approach_vec,
        yaw_deg, label, n_points). Raises on ``{"ok": false}``. This replaces the
        slow westeros-side ``get_point_cloud`` pull with one ~36 ms round-trip.
        """
        cmd: Dict[str, Any] = {"cmd": "localize", "object": obj}
        if hint_xy is not None:
            cmd["hint_xy"] = [float(hint_xy[0]), float(hint_xy[1])]
        res = await self._do(cmd)
        if not res.get("ok"):
            raise GraspServiceUnavailable(res.get("error", f"localize({obj!r}) failed"))
        return res
