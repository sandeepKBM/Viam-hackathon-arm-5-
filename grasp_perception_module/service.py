"""GraspPerceptionService: on-machine (in-process) perception for the arm.

WHY THIS EXISTS -- THE NETWORK BOUNDARY
----------------------------------------
Client code runs on a remote VM (westeros) and reaches the arm only through the
Viam cloud / TURN relay. Measured over that relay:

  - control calls (`arm.is_moving`, `arm.get_end_position`): ~36 ms  -- fine
  - `camera.get_images()`:                                  ~2,000 ms
  - `camera.get_point_cloud()`:                            ~12,500 ms for a
    14.7 MB cloud                                            <-- the bottleneck

Point clouds and images are heavy; shipping them across the relay dominates the
loop. This module is a Viam **generic service** (`rdk:service:generic`, model
`hackathons:grasp-perception:grasp-service`) that runs INSIDE `viam-server` on the
arm's own computer. It touches the camera and vision services through LOCAL,
in-process resource handles (no relay), runs detection + grasp localization
there, and returns only a tiny JSON result over `do_command`. That turns the
~12.5 s point-cloud fetch + off-box classification into a ~0.04 s round trip:
the 14.7 MB never crosses the network -- only a handful of floats do.

WHAT IT DOES
------------
Pure orchestration. It does NOT reinvent any geometry -- it reads the LOCAL
camera's segmented point clouds via the machine's vision `segmenter`
(`get_object_point_clouds`) and hands them to the repo's existing perception
code:

  - `components.perception3d.grasps_from_point_cloud_objects` -- PCD parse
    (`parse_viam_pcd`) + segmenter->grasp bridge (per-object `classify_grasp`
    with a geometry-bbox fallback for sparse/unparseable clouds).
  - `components.grasp_affordance.classify_grasp` -- top_down / side /
    inside_outside grasp geometry (called transitively via perception3d).

`components.shapes.CAMERA_NAME` and `components.constants.FLOOR_Z` supply the
defaults for the camera name and floor height so nothing is hardcoded here.

UNITS / FRAME ASSUMPTIONS (state them, never silently change them)
-----------------------------------------------------------------
  - Distances OUT are millimetres. `world_xyz_mm` is taken straight from
    `ObjectGrasp.center_xyz`, which `components.perception3d` sources from the
    segmented object's Viam `Geometry.center` (Viam geometries are in mm),
    falling back to `classify_grasp`'s own cloud centroid. This module adds NO
    unit scaling of its own -- it reuses perception3d/shapes' existing convention.
  - Frame: whatever frame the configured `segmenter` emits its geometries in.
    If the segmenter is fed by a camera placed in the machine's frame system,
    those centers are already world-frame; otherwise they are camera-frame.
    This service does not run `transform_pose` (it holds only camera/vision
    handles, not a RobotClient), so frame correctness is a LIVE-MACHINE check
    -- see README "Live-machine checks".
  - `floor_z` (mm) is read from config / `FLOOR_Z` env and exposed to callers
    for placement math; the geometry functions above are z-agnostic.

CONTROL RATE
------------
This is a request/response perception service, not a control loop -- it has no
internal control rate. Each `do_command` performs one synchronous capture +
classify. The caller sets the rate by how often it polls (typically once per
pick, not in a tight servo loop). Nothing here streams or actuates the arm.

do_command CONTRACT (small JSON only -- the network boundary; keep it exact)
----------------------------------------------------------------------------
  {"cmd": "health"}
      -> {"ok": true, "camera": <name>, "segmenter": <name>, "detector": <name>}

  {"cmd": "detections"}
      -> {"ok": true, "objects": [
             {"label": str, "world_xyz_mm": [x, y, z], "score": float}, ...]}

  {"cmd": "localize", "object": <label>, "hint_xy": [x, y]?}
      -> success: {"ok": true, "world_xyz_mm": [x, y, z],
                   "grasp_type": "top_down"|"side"|"inside_outside",
                   "approach_vec": [ax, ay, az], "yaw_deg": float,
                   "label": str, "n_points": int}
      -> failure: {"ok": false, "error": <reason>}

Never returns raw point clouds or images -- only computed small results.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.vision import Vision
from viam.utils import ValueTypes, struct_to_dict

# The reused geometry lives at the repo root (added to sys.path by main.py; also
# ensured here so `import components...` works when the class is imported directly,
# e.g. from tests). No new geometry is defined in this module.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from components.constants import FLOOR_Z as _DEFAULT_FLOOR_Z  # noqa: E402
from components.perception3d import (  # noqa: E402
    ObjectGrasp,
    grasps_from_point_cloud_objects,
)
from components.shapes import CAMERA_NAME as _DEFAULT_CAMERA_NAME  # noqa: E402

LOGGER = getLogger(__name__)

DEFAULT_WORLD_FRAME = os.environ.get("WORLD_FRAME", "world")


class GraspPerceptionService(Generic):
    """On-machine grasp perception, exposed only through `do_command`.

    Subclasses Viam's `Generic` service so it registers as
    `rdk:service:generic` / `hackathons:grasp-perception:grasp-service`. See the
    module docstring for the full rationale, unit/frame assumptions, and the
    exact `do_command` contract.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("hackathons", "grasp-perception"), "grasp-service")

    def __init__(self, name: str):
        super().__init__(name)
        self.logger = LOGGER

        # config (filled in by reconfigure())
        self.camera_name: str = ""
        self.segmenter_name: str = ""
        self.detector_name: str = ""
        self.floor_z: float = float(_DEFAULT_FLOOR_Z)
        self.world_frame: str = DEFAULT_WORLD_FRAME

        # LOCAL, in-process resource handles (grabbed from `dependencies`).
        self._camera: Optional[Any] = None
        self._segmenter: Optional[Any] = None
        self._detector: Optional[Any] = None

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
        """Validate attributes and declare required dependencies.

        Returns the resource NAMES this service depends on (camera, segmenter,
        detector) so `viam-server` wires their LOCAL handles into
        `dependencies` before `new`/`reconfigure` runs. Raises `ValueError`
        for missing/blank required attributes (config-time, before any capture).
        """
        attrs = struct_to_dict(config.attributes)

        camera = str(attrs.get("camera", "") or "").strip()
        if not camera:
            raise ValueError("'camera' attribute (local camera name) is required")

        segmenter = str(attrs.get("segmenter", "") or "").strip()
        if not segmenter:
            raise ValueError(
                "'segmenter' attribute (vision service exposing "
                "get_object_point_clouds) is required"
            )

        detector = str(attrs.get("detector", "") or "").strip()
        if not detector:
            raise ValueError(
                "'detector' attribute (vision service exposing get_detections) is required"
            )

        floor_z = attrs.get("floor_z", _DEFAULT_FLOOR_Z)
        try:
            float(floor_z)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'floor_z' must be a number (mm), got {floor_z!r}") from exc

        # Dependencies: the camera and the two vision services. viam-server uses
        # these to build the resource graph and pass in LOCAL handles.
        return [camera, segmenter, detector]

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        # Re-run validation so reconfigure() is safe to call directly (e.g. from
        # tests) without a separate validate step.
        self.validate_config(config)
        attrs = struct_to_dict(config.attributes)

        self.camera_name = str(attrs["camera"]).strip()
        self.segmenter_name = str(attrs["segmenter"]).strip()
        self.detector_name = str(attrs["detector"]).strip()
        self.floor_z = float(attrs.get("floor_z", _DEFAULT_FLOOR_Z))
        self.world_frame = str(attrs.get("world_frame", DEFAULT_WORLD_FRAME)).strip() or DEFAULT_WORLD_FRAME

        # Grab LOCAL, in-process handles for the camera and vision services. These
        # talk to the resources in the SAME viam-server process -- no cloud/TURN
        # relay -- which is the whole point of running this on the machine.
        self._camera = dependencies.get(Camera.get_resource_name(self.camera_name))
        self._segmenter = dependencies.get(Vision.get_resource_name(self.segmenter_name))
        self._detector = dependencies.get(Vision.get_resource_name(self.detector_name))

        self.logger.info(
            "grasp-perception reconfigured: camera=%s segmenter=%s detector=%s floor_z=%.3f world_frame=%s",
            self.camera_name,
            self.segmenter_name,
            self.detector_name,
            self.floor_z,
            self.world_frame,
        )

    # ------------------------------------------------------------------
    # do_command: the (only) network-facing surface
    # ------------------------------------------------------------------
    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("cmd")
        if cmd == "health":
            return self._cmd_health()
        if cmd == "detections":
            return await self._cmd_detections()
        if cmd == "localize":
            return await self._cmd_localize(command)
        # Keep failures inside the JSON contract rather than raising across the
        # RPC boundary, so a client always gets a parseable {"ok": false, ...}.
        return {"ok": False, "error": f"unknown cmd {cmd!r}; expected 'health', 'detections', or 'localize'"}

    def _cmd_health(self) -> Dict[str, ValueTypes]:
        return {
            "ok": True,
            "camera": self.camera_name,
            "segmenter": self.segmenter_name,
            "detector": self.detector_name,
        }

    async def _cmd_detections(self) -> Dict[str, ValueTypes]:
        try:
            grasps = await self._segment_grasps()
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the RPC
            return {"ok": False, "error": f"detection failed: {exc}"}

        objects: List[Dict[str, ValueTypes]] = []
        for g in grasps:
            objects.append(
                {
                    "label": g.label,
                    "world_xyz_mm": _xyz_list(g.center_xyz),
                    # No 2D detector confidence flows through the point-cloud
                    # path, so surface the grasp classifier's own heuristic
                    # confidence as the per-object score (see README).
                    "score": round(float(g.grasp.confidence), 4),
                }
            )
        return {"ok": True, "objects": objects}

    async def _cmd_localize(self, command: Mapping[str, ValueTypes]) -> Dict[str, ValueTypes]:
        label = command.get("object")
        if not label or not str(label).strip():
            return {"ok": False, "error": "'object' (target label) is required for localize"}
        target = str(label).strip().lower()

        hint_xy = command.get("hint_xy")
        hint: Optional[Tuple[float, float]] = None
        if hint_xy is not None:
            try:
                hx, hy = (float(v) for v in list(hint_xy)[:2])
                hint = (hx, hy)
            except (TypeError, ValueError):
                return {"ok": False, "error": f"'hint_xy' must be [x, y], got {hint_xy!r}"}

        try:
            grasps = await self._segment_grasps()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"localize failed: {exc}"}

        matches = [g for g in grasps if target in (g.label or "").lower()]
        if not matches:
            seen = sorted({g.label for g in grasps})
            return {"ok": False, "error": f"object {str(label)!r} not found; saw {seen}"}

        if hint is not None:
            chosen = min(matches, key=lambda g: _xy_dist2(g.center_xyz, hint))  # type: ignore[arg-type]
        else:
            # Deterministic default: highest-confidence match.
            chosen = max(matches, key=lambda g: float(g.grasp.confidence))

        return _localize_result(chosen)

    # ------------------------------------------------------------------
    # Internal: LOCAL capture -> existing geometry (no reinvented logic)
    # ------------------------------------------------------------------
    async def _segment_grasps(self) -> List[ObjectGrasp]:
        """Read the LOCAL segmented point clouds and classify each into a grasp
        proposal using the repo's existing `components.perception3d` bridge.

        This is the in-process, no-relay capture: `get_object_point_clouds`
        runs against the camera in the same `viam-server`, and the heavy PCD
        never leaves this process -- only the small `ObjectGrasp` list does.
        """
        if self._segmenter is None:
            raise RuntimeError(
                f"segmenter {self.segmenter_name!r} not available in dependencies"
            )
        objects = await self._segmenter.get_object_point_clouds(self.camera_name)
        return grasps_from_point_cloud_objects(objects)


# ---------------------------------------------------------------------------
# Small pure helpers (marshalling only -- no geometry)
# ---------------------------------------------------------------------------


def _xyz_list(xyz: Sequence[float]) -> List[float]:
    return [round(float(xyz[0]), 3), round(float(xyz[1]), 3), round(float(xyz[2]), 3)]


def _xy_dist2(xyz: Sequence[float], hint: Tuple[float, float]) -> float:
    dx = float(xyz[0]) - hint[0]
    dy = float(xyz[1]) - hint[1]
    return dx * dx + dy * dy


def _yaw_deg_from_axis(grasp_axis: Sequence[float]) -> float:
    """Yaw (deg) of the horizontal line the gripper pads close along. Taken from
    the grasp_axis' XY projection; 0 for a degenerate (vertical/zero) axis."""
    ax, ay = float(grasp_axis[0]), float(grasp_axis[1])
    if abs(ax) < 1e-9 and abs(ay) < 1e-9:
        return 0.0
    return round(math.degrees(math.atan2(ay, ax)), 3)


def _localize_result(g: ObjectGrasp) -> Dict[str, ValueTypes]:
    grasp = g.grasp
    return {
        "ok": True,
        "world_xyz_mm": _xyz_list(g.center_xyz),
        "grasp_type": grasp.grasp_type.value,
        "approach_vec": [round(float(v), 6) for v in grasp.approach],
        "yaw_deg": _yaw_deg_from_axis(grasp.grasp_axis),
        "label": g.label,
        "n_points": int(g.n_points),
    }


Registry.register_resource_creator(
    Generic.API,
    GraspPerceptionService.MODEL,
    ResourceCreatorRegistration(GraspPerceptionService.new, GraspPerceptionService.validate_config),
)
