"""In-process fakes standing in for the LOCAL camera + vision services.

`GraspPerceptionService` normally grabs, from `dependencies`, real Viam
resource handles for a camera and two vision services (a `segmenter` exposing
`get_object_point_clouds` and a `detector` exposing `get_detections`). These
fakes implement just the subset the service calls, returning SYNTHETIC point
clouds, so the service (and its tests) run with no live machine, no network,
and no torch/CUDA.

DESIGN NOTES
------------
- The synthetic clouds are the SAME structured families the repo's own
  `tests/test_perception3d.py` / `tests/test_grasp_affordance.py` use (dense
  Cartesian cube/box -> TOP_DOWN, tall solid cylinder -> SIDE, dense polar
  hollow tube -> INSIDE_OUTSIDE), encoded as REAL Viam-style ascii PCD bytes.
  Reusing the known-good generators keeps `classify_grasp` deterministic here.
- They are emitted in MILLIMETRES, self-consistent with the mm-scale
  `Geometry.center`/`box.dims_mm` fields, so the whole reused geometry pipeline
  (`parse_viam_pcd` -> `classify_grasp`) is exercised end to end at the scale
  `classify_grasp`'s gripper-width thresholds expect. On real hardware the
  camera's PCD may be in metres -- that unit question is a live-machine check
  (see README), not something these offline fakes assert.
- `build_dependencies(...)` returns a mapping keyed by the SAME `ResourceName`s
  the service looks up (`Camera.get_resource_name` / `Vision.get_resource_name`),
  so tests drive the service's real `reconfigure` dependency-grabbing path
  rather than reaching past it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic point-cloud generators (mm). Mirrors tests/test_perception3d.py.
# ---------------------------------------------------------------------------


def solid_cube(side_xy: float = 70.0, height: float = 45.0, n_per_axis: int = 12) -> np.ndarray:
    """Dense, solid cube -> compact, no cavity -> TOP_DOWN."""
    xs = np.linspace(0.0, side_xy, n_per_axis)
    ys = np.linspace(0.0, side_xy, n_per_axis)
    zs = np.linspace(0.0, height, n_per_axis)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def solid_cylinder(radius: float = 25.0, height: float = 130.0, n_xy: int = 61, n_z: int = 25) -> np.ndarray:
    """Dense, solid (filled-disk cross-section) cylinder -> tall -> SIDE."""
    xs = np.linspace(-radius, radius, n_xy)
    ys = np.linspace(-radius, radius, n_xy)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    mask = X**2 + Y**2 <= radius**2
    xy = np.stack([X[mask], Y[mask]], axis=1)
    zs = np.linspace(0.0, height, n_z)
    pts = np.empty((xy.shape[0] * n_z, 3), dtype=np.float64)
    for i, z in enumerate(zs):
        pts[i * xy.shape[0] : (i + 1) * xy.shape[0], :2] = xy
        pts[i * xy.shape[0] : (i + 1) * xy.shape[0], 2] = z
    return pts


def hollow_tube(
    inner_r: float = 25.0,
    outer_r: float = 35.0,
    height: float = 70.0,
    n_theta: int = 160,
    n_radii: int = 6,
    n_z: int = 30,
) -> np.ndarray:
    """Dense open cylinder (annulus cross-section) -> INSIDE_OUTSIDE."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radii = np.linspace(inner_r, outer_r, n_radii)
    zs = np.linspace(0.0, height, n_z)
    pts = []
    for z in zs:
        for r in radii:
            for t in thetas:
                pts.append((r * np.cos(t), r * np.sin(t), z))
    return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Real Viam-style ascii PCD encoding (matches tests/test_perception3d.py).
# ---------------------------------------------------------------------------


def encode_ascii_pcd(points: np.ndarray) -> bytes:
    n = points.shape[0]
    header = (
        "# .PCD v0.7\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    body = "\n".join(f"{x} {y} {z}" for x, y, z in points)
    return (header + body + "\n").encode("ascii")


def make_point_cloud_object(
    points: np.ndarray,
    *,
    label: str = "",
    center: Optional[Tuple[float, float, float]] = None,
    dims: Optional[Tuple[float, float, float]] = None,
) -> SimpleNamespace:
    """Duck-typed `PointCloudObject`: `.point_cloud` (ascii PCD bytes) plus
    `.geometries.geometries[0]` carrying `.label`/`.center`/`.box.dims_mm`,
    exactly the shape `components.perception3d` reads."""
    raw = encode_ascii_pcd(points)
    geom = SimpleNamespace(
        label=label,
        center=(SimpleNamespace(x=center[0], y=center[1], z=center[2]) if center else None),
        box=(SimpleNamespace(dims_mm=SimpleNamespace(x=dims[0], y=dims[1], z=dims[2])) if dims else None),
    )
    geoms_in_frame = SimpleNamespace(reference_frame="cam", geometries=[geom])
    return SimpleNamespace(point_cloud=raw, geometries=geoms_in_frame)


def default_scene() -> List[SimpleNamespace]:
    """A deterministic three-object tabletop: one of each grasp type, at
    distinct world_xyz_mm centers so `localize` + `hint_xy` are testable."""
    return [
        make_point_cloud_object(
            solid_cube(side_xy=70.0, height=45.0),
            label="block",
            center=(120.0, -40.0, 25.0),
        ),
        make_point_cloud_object(
            solid_cylinder(radius=25.0, height=130.0),
            label="bottle",
            center=(300.0, 60.0, 65.0),
        ),
        make_point_cloud_object(
            hollow_tube(inner_r=25.0, outer_r=35.0, height=70.0),
            label="cup",
            center=(220.0, 180.0, 35.0),
        ),
    ]


# ---------------------------------------------------------------------------
# Fake resource handles (record calls like xarm_velocity.fake_backend.FakeXArm).
# ---------------------------------------------------------------------------


class FakeCamera:
    """Minimal stand-in for the LOCAL camera. The service currently reads the
    cloud through the segmenter, but a camera dependency is still wired, so
    this exists so `dependencies` carries a real handle for it and so a fake
    `get_point_cloud`/`get_images` is available for future use/tests."""

    def __init__(self, name: str = "cam", objects: Optional[Sequence[SimpleNamespace]] = None):
        self.name = name
        self._objects = list(objects) if objects is not None else default_scene()
        self.calls: List[str] = []

    async def get_point_cloud(self, **kwargs) -> Tuple[bytes, str]:
        self.calls.append("get_point_cloud")
        # A single merged synthetic cloud (all objects), ascii PCD.
        merged = np.concatenate(
            [
                np.frombuffer(b"", dtype=np.float64).reshape(0, 3),
                *[_pco_points(o) for o in self._objects],
            ]
        )
        return encode_ascii_pcd(merged), "pointcloud/pcd"

    async def get_images(self, **kwargs):
        self.calls.append("get_images")
        return [], SimpleNamespace()


class FakeSegmenter:
    """Stand-in for the `segmenter` vision service. Implements only
    `get_object_point_clouds(camera_name)`, returning the synthetic scene."""

    def __init__(self, name: str = "vision-segment", objects: Optional[Sequence[SimpleNamespace]] = None):
        self.name = name
        self._objects = list(objects) if objects is not None else default_scene()
        self.calls: List[Tuple[str, tuple]] = []

    def set_objects(self, objects: Sequence[SimpleNamespace]) -> None:
        self._objects = list(objects)

    async def get_object_point_clouds(self, camera_name: str, **kwargs) -> List[SimpleNamespace]:
        self.calls.append(("get_object_point_clouds", (camera_name,)))
        return list(self._objects)


class FakeDetector:
    """Stand-in for the `detector` vision service (2D detections). Wired as a
    dependency + reported by `health`; not required by the point-cloud path."""

    def __init__(self, name: str = "shape-detector", detections: Optional[Sequence[Any]] = None):
        self.name = name
        self._detections = list(detections) if detections is not None else []
        self.calls: List[Tuple[str, tuple]] = []

    async def get_detections_from_camera(self, camera_name: str, **kwargs) -> List[Any]:
        self.calls.append(("get_detections_from_camera", (camera_name,)))
        return list(self._detections)

    async def get_detections(self, image, **kwargs) -> List[Any]:
        self.calls.append(("get_detections", (image,)))
        return list(self._detections)


def _pco_points(obj: SimpleNamespace) -> np.ndarray:
    from components.perception3d import parse_viam_pcd  # local import: repo root on sys.path

    return parse_viam_pcd(obj)


def build_dependencies(
    camera: Optional[FakeCamera] = None,
    segmenter: Optional[FakeSegmenter] = None,
    detector: Optional[FakeDetector] = None,
) -> Dict[Any, Any]:
    """Build a `dependencies` mapping keyed by the SAME `ResourceName`s
    `GraspPerceptionService.reconfigure` looks up, so tests exercise the real
    dependency-grabbing path instead of monkeypatching around it."""
    from viam.components.camera import Camera
    from viam.services.vision import Vision

    camera = camera or FakeCamera()
    segmenter = segmenter or FakeSegmenter()
    detector = detector or FakeDetector()

    return {
        Camera.get_resource_name(camera.name): camera,
        Vision.get_resource_name(segmenter.name): segmenter,
        Vision.get_resource_name(detector.name): detector,
    }
