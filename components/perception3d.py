"""Bridge from Viam's live 3D segmenter output to
`components.grasp_affordance.classify_grasp` -- REAL point clouds instead of
the synthetic ones that module's own tests use.

WHY THIS EXISTS
----------------
`components/grasp_affordance.py` already answers "what grasp does this
object's point cloud afford" (top_down / side / inside_outside), and its
tests prove the geometry logic against hand-built synthetic clouds. What it
does NOT do is get those points from the robot: its `points_from_pcd()` is
an ASCII-only stub, explicitly documented as "not required by, or exercised
in, the geometry/decision logic".

This module is that missing piece. Viam's vision service exposes
`VisionClient.get_object_point_clouds(camera_name)` -- backed, per the repo's
README "Registry modules" section, by the built-in `detections-to-segments`
segmenter fed from a detector (e.g. `viam-labs:EfficientDet-COCO`). It returns
one `viam.services.vision.PointCloudObject` per detected object, each
carrying:

  - `.point_cloud` -- raw PCD (Point Cloud Data) bytes for JUST that object
    (segmented out of the full scene), typically `DATA ascii` for small/debug
    output or `DATA binary` for real sensor output.
  - `.geometries` -- a `GeometriesInFrame` with a `.geometries` list of
    `Geometry` protos giving a coarse 3D bounding box (`.box.dims_mm`), a
    world/camera-frame `.center` pose, and an optional `.label`.

THE PARTIAL-VIEW PROBLEM THIS FIXES
------------------------------------
A single camera frame only ever sees one side of an object -- one oblique
depth image, one silhouette. Naively deprojecting that single frame's pixels
(as `components/shapes.py`'s 2D-detection-plus-depth path does) gives you an
incomplete, one-sided point cloud: a cup's far wall, its interior, its
underside are all missing. `classify_grasp`'s INSIDE_OUTSIDE logic in
particular needs to see BOTH the near and far rim of an opening to recognize
an annulus at all. The segmenter service is expected to fuse/complete the
object's cloud (e.g. from multiple views, a learned shape prior, or simply a
denser single-shot depth capture than a handful of 2D keypoints) and hand
back one cloud that represents the FULL object, not one partial frame. That
is the actual fix here: `get_object_grasps` doesn't do anything geometrically
different from `classify_grasp` -- it just feeds it a materially better input.

PIPELINE
--------
    VisionClient.get_object_point_clouds(camera)
        -> per object: parse_viam_pcd(obj.point_cloud) -> (N, 3) numpy
        -> classify_grasp(points)                       -> GraspAffordance
        -> ObjectGrasp(label, center_xyz, grasp, n_points)

`get_object_grasps` is dependency-injected on `machine`/the Viam vision
client (duck-typed, see `VisionClient` usage below) so it is fully
offline-testable: tests monkeypatch `components.perception3d.VisionClient`
with a fake whose `from_robot(...)` returns a fake client exposing an async
`get_object_point_clouds(camera_name)`.

Executability caveat: same as `grasp_affordance` itself -- this only answers
"what grasp does the geometry afford", never "can the 5-DOF arm reach it".
Run the result through `components.ik`/`components.fast_planner` before
committing to a pick.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from viam.services.vision import VisionClient

from components.grasp_affordance import GraspAffordance, GraspType, classify_grasp

Tuple3 = Tuple[float, float, float]

# Below this many parsed points, classify_grasp's own PCA/occupancy-grid
# features are too noisy to trust (its hard floor is 8 points; this default
# is a much more conservative "don't even try" threshold) -- fall back to
# the segmenter's own coarse geometry bbox instead. Overridable via env for
# whoever tunes this against a real segmenter's typical point density.
SPARSE_POINTS_THRESHOLD = int(os.environ.get("PERCEPTION3D_SPARSE_MIN_POINTS", "40"))

# Confidence assigned to every geometry-bbox fallback grasp -- deliberately
# lower than classify_grasp's own worst-case (0.15 "too wide" fallback) is
# NOT guaranteed, so this is set independently: a bbox aspect-ratio guess is
# strictly less informed than even a bad classify_grasp() run on real points.
GEOMETRY_FALLBACK_CONFIDENCE = 0.2


class PCDParseError(ValueError):
    """Raised when bytes claiming to be a Viam PCD point cloud can't be
    parsed into an (N, 3) xyz array -- malformed header, unsupported field
    types, or (for `binary_compressed`) a compression scheme this module
    doesn't implement. Always raised with a specific, actionable reason;
    never fails silently."""


# ---------------------------------------------------------------------------
# PCD parsing: ascii + binary (uncompressed). binary_compressed (LZF) is
# explicitly NOT implemented -- see the NotImplementedError raised below.
# ---------------------------------------------------------------------------

# PCD SIZE/TYPE pair -> numpy dtype string (little-endian, which is what
# every PCD producer this repo has seen -- PCL, Open3D, RealSense drivers --
# emits on x86/ARM Linux and macOS).
_NUMPY_TYPE = {
    ("F", 4): "f4",
    ("F", 8): "f8",
    ("U", 1): "u1",
    ("U", 2): "u2",
    ("U", 4): "u4",
    ("U", 8): "u8",
    ("I", 1): "i1",
    ("I", 2): "i2",
    ("I", 4): "i4",
    ("I", 8): "i8",
}


def _pcd_bytes(pcd: Any) -> bytes:
    """Extract raw PCD bytes from a `PointCloudObject` (via `.point_cloud`)
    or accept `bytes`/`bytearray` directly."""
    raw = getattr(pcd, "point_cloud", pcd)
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    raise TypeError(
        f"parse_viam_pcd expects PCD bytes or an object with a `.point_cloud` "
        f"bytes attribute (e.g. a PointCloudObject), got {type(raw)!r}"
    )


def _parse_header(raw: bytes):
    """Read PCD header lines up to and including the `DATA <mode>` line.
    Returns (fields, sizes, types, counts, points_n, data_mode, body_offset)."""
    fields: List[str] = []
    sizes: List[int] = []
    types: List[str] = []
    counts: List[int] = []
    points_n: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    data_mode: Optional[str] = None

    pos = 0
    while True:
        nl = raw.find(b"\n", pos)
        if nl == -1:
            raise PCDParseError("malformed PCD: no DATA line found before end of input")
        line = raw[pos:nl]
        pos = nl + 1
        text = line.decode("ascii", errors="ignore").strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("FIELDS"):
            fields = text.split()[1:]
        elif text.startswith("SIZE"):
            sizes = [int(v) for v in text.split()[1:]]
        elif text.startswith("TYPE"):
            types = text.split()[1:]
        elif text.startswith("COUNT"):
            counts = [int(v) for v in text.split()[1:]]
        elif text.startswith("WIDTH"):
            width = int(text.split()[1])
        elif text.startswith("HEIGHT"):
            height = int(text.split()[1])
        elif text.startswith("POINTS"):
            points_n = int(text.split()[1])
        elif text.startswith("DATA"):
            parts = text.split()
            data_mode = parts[1] if len(parts) > 1 else ""
            break

    if not fields:
        raise PCDParseError("malformed PCD: missing FIELDS header")
    if not counts:
        counts = [1] * len(fields)
    if not sizes or not types or len(sizes) != len(fields) or len(types) != len(fields):
        raise PCDParseError("malformed PCD: missing or mismatched SIZE/TYPE header")
    if points_n is None:
        points_n = (width or 0) * (height or 1)
    if data_mode is None:
        raise PCDParseError("malformed PCD: missing DATA line")

    return fields, sizes, types, counts, points_n, data_mode, pos


def _xyz_indices(fields: Sequence[str]) -> Tuple[int, int, int]:
    try:
        return fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError as exc:
        raise PCDParseError(f"PCD FIELDS {list(fields)} does not contain x/y/z") from exc


def _parse_ascii_body(body: bytes, xi: int, yi: int, zi: int) -> np.ndarray:
    text = body.decode("ascii", errors="ignore")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rows.append((float(parts[xi]), float(parts[yi]), float(parts[zi])))
    if not rows:
        raise PCDParseError("ASCII PCD DATA section contained no points")
    return np.asarray(rows, dtype=np.float64)


def _structured_dtype(fields: Sequence[str], sizes: Sequence[int], types: Sequence[str], counts: Sequence[int]) -> np.dtype:
    """Build a numpy structured dtype matching the PCD FIELDS/SIZE/TYPE/COUNT
    header, so one `np.frombuffer` call parses every point in the buffer at
    once (vectorized, no per-point Python loop)."""
    seen: dict = {}
    descr = []
    for name, size, typ, count in zip(fields, sizes, types, counts):
        code = _NUMPY_TYPE.get((typ.upper(), size))
        if code is None:
            raise PCDParseError(f"unsupported PCD field TYPE/SIZE: {typ}{size} (field {name!r})")
        # PCD padding fields are conventionally named "_" and may repeat;
        # numpy structured dtypes require unique field names.
        n = seen.get(name, 0)
        seen[name] = n + 1
        field_name = name if n == 0 else f"{name}__dup{n}"
        if count == 1:
            descr.append((field_name, "<" + code))
        else:
            descr.append((field_name, "<" + code, (count,)))
    return np.dtype(descr)


def _parse_binary_body(
    body: bytes,
    fields: Sequence[str],
    sizes: Sequence[int],
    types: Sequence[str],
    counts: Sequence[int],
    points_n: int,
    xi: int,
    yi: int,
    zi: int,
) -> np.ndarray:
    dt = _structured_dtype(fields, sizes, types, counts)
    point_size = dt.itemsize
    if point_size <= 0:
        raise PCDParseError("malformed PCD: zero-size point record")
    available = len(body) // point_size
    n = points_n if points_n and points_n <= available else available
    if n <= 0:
        raise PCDParseError(
            f"binary PCD DATA section has no complete points "
            f"(body={len(body)} bytes, point_size={point_size} bytes)"
        )
    arr = np.frombuffer(body, dtype=dt, count=n)
    x_name, y_name, z_name = dt.names[xi], dt.names[yi], dt.names[zi]
    xyz = np.empty((n, 3), dtype=np.float64)
    xyz[:, 0] = arr[x_name].astype(np.float64)
    xyz[:, 1] = arr[y_name].astype(np.float64)
    xyz[:, 2] = arr[z_name].astype(np.float64)
    return xyz


def parse_viam_pcd(pcd: Any) -> np.ndarray:
    """Parse one Viam `PointCloudObject`'s point cloud (or raw PCD bytes)
    into an `(N, 3)` float64 xyz numpy array, ready for
    `components.grasp_affordance.classify_grasp`.

    Handles both PCD `DATA` modes this repo's segmenter is expected to ever
    emit:

    - `ascii` -- whitespace-separated float rows, one per line (extends
      `grasp_affordance.points_from_pcd`'s ascii-only path: same format,
      generalized header parsing).
    - `binary` -- raw little-endian point records packed per the
      `FIELDS`/`SIZE`/`TYPE`/`COUNT` header, parsed vectorized via a numpy
      structured dtype (no per-point Python loop).

    `binary_compressed` (PCL's LZF-compressed binary mode) is explicitly NOT
    implemented -- raises `NotImplementedError` with a clear pointer to
    `pypcd4`/`open3d` (or reconfiguring the segmenter to emit `binary`) --
    rather than silently mis-parsing compressed bytes as raw ones.

    Raises `PCDParseError` for any other malformed/unsupported header, and
    `TypeError` if `pcd` isn't PCD bytes or an object with a `.point_cloud`
    bytes attribute.
    """
    raw = _pcd_bytes(pcd)
    fields, sizes, types, counts, points_n, data_mode, body_off = _parse_header(raw)
    xi, yi, zi = _xyz_indices(fields)
    body = raw[body_off:]

    if data_mode == "ascii":
        return _parse_ascii_body(body, xi, yi, zi)
    if data_mode == "binary":
        return _parse_binary_body(body, fields, sizes, types, counts, points_n, xi, yi, zi)
    if data_mode == "binary_compressed":
        raise NotImplementedError(
            "parse_viam_pcd does not implement 'binary_compressed' (LZF-compressed) "
            "PCD. Either configure the segmenter to emit 'binary' or 'ascii' PCD, or "
            "parse with a dedicated library (e.g. `pypcd4`, `open3d.io.read_point_cloud`) "
            "and pass the resulting (N, 3) array to classify_grasp() directly."
        )
    raise PCDParseError(f"unsupported PCD DATA mode: {data_mode!r}")


# ---------------------------------------------------------------------------
# Geometry-bbox fallback (sparse/unparseable clouds)
# ---------------------------------------------------------------------------


def _first_geometry(obj: Any) -> Optional[Any]:
    geoms_in_frame = getattr(obj, "geometries", None)
    geoms = getattr(geoms_in_frame, "geometries", None) if geoms_in_frame is not None else None
    if not geoms:
        return None
    return geoms[0]


def _label_from_geometry(geom: Optional[Any]) -> str:
    if geom is None:
        return ""
    return str(getattr(geom, "label", "") or "")


def _vec3(v: Any) -> Optional[Tuple3]:
    if v is None:
        return None
    x, y, z = getattr(v, "x", None), getattr(v, "y", None), getattr(v, "z", None)
    if x is None or y is None or z is None:
        return None
    return (float(x), float(y), float(z))


def _center_from_geometry(geom: Optional[Any]) -> Optional[Tuple3]:
    if geom is None:
        return None
    return _vec3(getattr(geom, "center", None))


def _dims_from_geometry(geom: Optional[Any]) -> Optional[Tuple3]:
    if geom is None:
        return None
    box = getattr(geom, "box", None)
    dims = getattr(box, "dims_mm", None) if box is not None else None
    return _vec3(dims)


def _bbox_grasp_fallback(obj: Any, *, points: Optional[np.ndarray] = None, note: str = "") -> GraspAffordance:
    """A low-confidence GraspAffordance derived purely from the segmenter's
    coarse `.geometries` bounding box (or, absent that, the raw points'
    min/max extent) -- for when the cloud is too sparse/malformed to run
    `classify_grasp`'s PCA/occupancy-grid features on. Mirrors
    `classify_grasp`'s own aspect-ratio heuristic (tall -> SIDE, else
    TOP_DOWN) at a coarser resolution, since a bbox alone can't detect an
    INSIDE_OUTSIDE cavity."""
    geom = _first_geometry(obj)
    dims = _dims_from_geometry(geom)
    center = _center_from_geometry(geom)

    if dims is not None:
        dx, dy, dz = (abs(v) for v in dims)
    elif points is not None and points.shape[0] >= 1:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        dx, dy, dz = (float(v) for v in (maxs - mins))
        if center is None:
            mid = (mins + maxs) / 2.0
            center = (float(mid[0]), float(mid[1]), float(mid[2]))
    else:
        dx = dy = dz = 0.0

    if center is None:
        center = (0.0, 0.0, 0.0)

    minor, major = sorted((dx, dy))
    minor = minor if minor > 1e-6 else max(major, 1e-6)
    is_tall = dz > 1.3 * minor if minor > 1e-6 else False

    if is_tall:
        grasp_type = GraspType.SIDE
        approach: Tuple3 = (1.0, 0.0, 0.0)
        grasp_axis: Tuple3 = (0.0, 1.0, 0.0)
    else:
        grasp_type = GraspType.TOP_DOWN
        approach = (0.0, 0.0, -1.0)
        grasp_axis = (1.0, 0.0, 0.0)

    return GraspAffordance(
        grasp_type=grasp_type,
        approach=approach,
        grasp_axis=grasp_axis,
        width_mm=float(minor),
        center=center,
        confidence=GEOMETRY_FALLBACK_CONFIDENCE,
        note=note or "geometry-bbox fallback (sparse/unparseable point cloud)",
    )


# ---------------------------------------------------------------------------
# Per-object grasp result + the pure classification step (testable without
# any Viam/async machinery -- see get_object_grasps below for the I/O shim).
# ---------------------------------------------------------------------------


@dataclass
class ObjectGrasp:
    """One segmented object's grasp proposal, bridging a live Viam
    `PointCloudObject` to `components.grasp_affordance`'s classifier."""

    label: str
    center_xyz: Tuple3
    grasp: GraspAffordance
    n_points: int
    note: str = ""


def _grasp_for_point_cloud_object(obj: Any, index: int) -> ObjectGrasp:
    """Classify one `PointCloudObject`: parse its cloud, run
    `classify_grasp` if there are enough points, else fall back to the
    geometry bbox. Pure/synchronous -- no I/O -- so it's directly unit
    testable; `get_object_grasps` below is just this plus the Viam call."""
    geom = _first_geometry(obj)
    label = _label_from_geometry(geom) or f"object_{index}"
    geom_center = _center_from_geometry(geom)

    try:
        points = parse_viam_pcd(obj)
    except Exception as exc:
        grasp = _bbox_grasp_fallback(obj, note=f"PCD parse failed ({exc}) -> geometry bbox fallback")
        center = geom_center or grasp.center
        return ObjectGrasp(label=label, center_xyz=center, grasp=grasp, n_points=0, note=grasp.note)

    n = int(points.shape[0])
    if n < SPARSE_POINTS_THRESHOLD:
        grasp = _bbox_grasp_fallback(
            obj, points=points, note=f"sparse cloud ({n} pts < {SPARSE_POINTS_THRESHOLD}) -> geometry bbox fallback"
        )
    else:
        try:
            grasp = classify_grasp(points)
        except ValueError as exc:
            grasp = _bbox_grasp_fallback(
                obj, points=points, note=f"classify_grasp rejected cloud ({exc}) -> geometry bbox fallback"
            )

    center = geom_center or grasp.center
    return ObjectGrasp(label=label, center_xyz=center, grasp=grasp, n_points=n, note=grasp.note)


def grasps_from_point_cloud_objects(objects: Sequence[Any]) -> List[ObjectGrasp]:
    """Pure/synchronous core of `get_object_grasps`: classify a list of
    already-fetched `PointCloudObject`s (or duck-typed equivalents). Exposed
    separately so tests can exercise the parsing + classification + fallback
    logic without any asyncio/Viam plumbing."""
    return [_grasp_for_point_cloud_object(obj, i) for i, obj in enumerate(objects)]


async def get_object_grasps(machine: Any, segmenter_name: str, camera_name: str) -> List[ObjectGrasp]:
    """Call the Viam segmenter's `get_object_point_clouds(camera_name)`, then
    classify each returned object's cloud into a grasp proposal (see the
    module docstring for the full parse -> classify -> fallback pipeline).

    `machine` is passed straight to `VisionClient.from_robot` -- duck-typed
    like the rest of this repo's `*_from_robot` call sites (`components/
    vision.py`, `components/shapes.py`), so tests monkeypatch the
    module-level `VisionClient` name with a fake exposing `from_robot(...)
    .get_object_point_clouds(camera_name)` and never need a real RobotClient.
    """
    segmenter = VisionClient.from_robot(machine, segmenter_name)
    objects = await segmenter.get_object_point_clouds(camera_name)
    return grasps_from_point_cloud_objects(objects)
