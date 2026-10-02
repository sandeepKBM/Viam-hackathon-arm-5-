"""Collect obstacle geometries and camera point clouds for motion planning."""

from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np

from viam.proto.common import (
    Geometry,
    GeometriesInFrame,
    PointCloud,
    Pose,
    RectangularPrism,
    Vector3,
    WorldState,
)
from viam.robot.client import RobotClient

from components.shapes import CAMERA_NAME

VOXEL_MM = float(os.environ.get("OBSTACLE_VOXEL_MM", 40))
MAX_VOXELS = int(os.environ.get("OBSTACLE_MAX_VOXELS", 400))

OBSTACLE_NAMES = tuple(
    n.strip()
    for n in os.environ.get(
        "OBSTACLE_NAMES", "table,wall-front,wall-side,ceiling"
    ).split(",")
    if n.strip()
)
MOTION_NAME = os.environ.get("MOTION_NAME", "builtin")
COLLISION_BUFFER_MM = float(os.environ.get("COLLISION_BUFFER_MM", 20))
# Fallback boxes from the cell fragment translations if get_geometries is empty.
STATIC_BOXES = (
    ("table", Pose(x=0, y=0, z=-123), Vector3(x=1400, y=1400, z=80)),
    ("wall-front", Pose(x=740, y=0, z=300), Vector3(x=40, y=1400, z=800)),
    ("wall-side", Pose(x=0, y=-500, z=300), Vector3(x=1400, y=40, z=800)),
    ("ceiling", Pose(x=0, y=0, z=1050), Vector3(x=1600, y=1600, z=40)),
)

_cache: tuple[float, WorldState, list[str], list[dict]] | None = None
_CACHE_S = float(os.environ.get("OBSTACLE_CACHE_S", 3))


def _box(label: str, center: Pose, dims: Vector3) -> Geometry:
    return Geometry(
        center=center,
        box=RectangularPrism(dims_mm=dims),
        label=label,
    )


async def _geometries_of(machine: RobotClient, name: str) -> list[Geometry]:
    from viam.components.generic import Generic

    try:
        res = Generic.from_robot(machine, name)
        found = list(await res.get_geometries(timeout=8))
    except Exception:
        return []
    for geo in found:
        if not geo.label:
            geo.label = name
    return found


def _parse_pcd(data: bytes) -> np.ndarray:
    """Read Viam/RealSense PCD (ascii or binary little-endian XYZ)."""
    if not data:
        return np.zeros((0, 3), dtype=np.float32)
    header, sep, body = data.partition(b"\nDATA ")
    if not sep:
        return np.zeros((0, 3), dtype=np.float32)
    meta = {}
    for raw in header.splitlines():
        line = raw.decode("ascii", errors="ignore").strip()
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        meta[key.upper()] = rest.strip()
    fields = meta.get("FIELDS", "x y z").split()
    sizes = [int(s) for s in meta.get("SIZE", "4 4 4").split()]
    types = meta.get("TYPE", "F F F").split()
    counts = [int(c) for c in meta.get("COUNT", "1 1 1").split()]
    n_pts = int(float(meta.get("POINTS") or meta.get("WIDTH") or 0))
    data_line, _, payload = body.partition(b"\n")
    kind = data_line.decode("ascii", errors="ignore").strip().lower()
    stride = sum(s * c for s, c in zip(sizes, counts))
    if n_pts <= 0 or stride <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    xi = fields.index("x") if "x" in fields else 0
    yi = fields.index("y") if "y" in fields else 1
    zi = fields.index("z") if "z" in fields else 2

    def _offset(idx: int) -> int:
        return sum(s * c for s, c in zip(sizes[:idx], counts[:idx]))

    ox, oy, oz = _offset(xi), _offset(yi), _offset(zi)
    if kind.startswith("ascii"):
        rows = [ln.split() for ln in payload.splitlines()[:n_pts] if ln.strip()]
        if not rows:
            return np.zeros((0, 3), dtype=np.float32)
        cols = np.array(rows, dtype=np.float32)
        pts = cols[:, [xi, yi, zi]]
        return pts[np.isfinite(pts).all(axis=1)]

    raw = payload[: n_pts * stride]
    n_pts = len(raw) // stride
    raw = raw[: n_pts * stride]
    rec = np.frombuffer(raw, dtype=np.uint8).reshape(n_pts, stride)
    pts = np.column_stack(
        (
            rec[:, ox : ox + sizes[xi]].view("<f4").reshape(-1),
            rec[:, oy : oy + sizes[yi]].view("<f4").reshape(-1),
            rec[:, oz : oz + sizes[zi]].view("<f4").reshape(-1),
        )
    ).astype(np.float32, copy=False)
    finite = np.isfinite(pts).all(axis=1)
    nonzero = np.linalg.norm(pts, axis=1) > 1e-6
    return pts[finite & nonzero]


def _write_binary_pcd(points: np.ndarray) -> bytes:
    """Binary little-endian XYZ PCD — the format Geometry.pointcloud accepts."""
    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")
    return header + pts.tobytes(order="C")


def voxelize_points(
    points: np.ndarray, voxel_mm: float = VOXEL_MM, max_voxels: int = MAX_VOXELS
) -> tuple[np.ndarray, float]:
    """Occupied voxel centers (mm) and the resolution used."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32), voxel_mm
    size = max(float(voxel_mm), 1.0)
    cells = np.unique(np.floor(pts / size).astype(np.int32), axis=0)
    while cells.shape[0] > max_voxels:
        size *= 1.5
        cells = np.unique(np.floor(pts / size).astype(np.int32), axis=0)
    centers = (cells.astype(np.float32) + 0.5) * size
    return centers, size


def voxels_to_boxes(centers: np.ndarray, size: float, label: str) -> list[Geometry]:
    """Planner-accepted box geometries — one RectangularPrism per occupied voxel."""
    half = float(size)
    boxes: list[Geometry] = []
    for i, c in enumerate(centers):
        boxes.append(
            Geometry(
                label=f"{label}-voxel-{i}",
                center=Pose(x=float(c[0]), y=float(c[1]), z=float(c[2])),
                box=RectangularPrism(dims_mm=Vector3(x=half, y=half, z=half)),
            )
        )
    return boxes


def voxels_payload(centers: np.ndarray, size: float, label: str) -> dict:
    """JSON/Struct-safe voxel payload for arm extra."""
    cells = np.floor(np.asarray(centers, dtype=np.float64) / size).astype(int)
    return {
        "label": label,
        "type": "voxels",
        "resolution_mm": float(size),
        "count": int(len(centers)),
        "occupied": [[int(a), int(b), int(c)] for a, b, c in cells.tolist()],
        "centers_mm": [[float(x), float(y), float(z)] for x, y, z in centers.tolist()],
        "box": {"type": "box", "x": float(size), "y": float(size), "z": float(size)},
    }


def pcd_to_voxels(data: bytes, label: str) -> tuple[list[Geometry], dict | None, bytes]:
    points = _parse_pcd(data)
    centers, size = voxelize_points(points)
    boxes = voxels_to_boxes(centers, size, label)
    payload = voxels_payload(centers, size, label) if len(centers) else None
    pcd = _write_binary_pcd(centers) if len(centers) else b""
    return boxes, payload, pcd


async def _point_cloud(
    machine: RobotClient, name: str
) -> tuple[list[Geometry], dict | None]:
    from viam.components.camera import Camera

    try:
        cam = Camera.from_robot(machine, name)
        data, _mime = await cam.get_point_cloud(timeout=15)
    except Exception:
        return [], None
    if not data:
        return [], None
    boxes, payload, voxel_pcd = pcd_to_voxels(data, name)
    geos = list(boxes)
    if voxel_pcd:
        geos.append(
            Geometry(
                label=f"{name}-voxels-pcd",
                pointcloud=PointCloud(point_cloud=voxel_pcd),
            )
        )
    print(
        f"  voxels {name}: {0 if payload is None else payload['count']} cells "
        f"@ {0 if payload is None else payload['resolution_mm']:.0f}mm "
        f"(binary PCD {len(voxel_pcd)} B)",
        flush=True,
    )
    return geos, payload


def _world_state(frames: dict[str, list[Geometry]]) -> WorldState:
    return WorldState(
        obstacles=[
            GeometriesInFrame(reference_frame=frame, geometries=geos)
            for frame, geos in frames.items()
            if geos
        ]
    )


async def collect_world_state(
    machine: RobotClient,
) -> tuple[WorldState, list[str], list[dict]]:
    """Voxelized point clouds + geometries for obstacles and the cell camera."""
    global _cache
    now = time.monotonic()
    if _cache and now - _cache[0] < _CACHE_S:
        return _cache[1], list(_cache[2]), [dict(v) for v in _cache[3]]

    frames: dict[str, list[Geometry]] = {}
    labels: list[str] = []
    voxels: list[dict] = []

    async def _add(name: str, frame: str) -> None:
        geos = await _geometries_of(machine, name)
        cloud_geos, payload = await _point_cloud(machine, name)
        geos.extend(cloud_geos)
        if payload:
            voxels.append(payload)
        if geos:
            frames.setdefault(frame, []).extend(geos)
            labels.extend(g.label or name for g in geos)

    for name in OBSTACLE_NAMES:
        await _add(name, name)

    cam = os.environ.get("CAMERA_NAME", CAMERA_NAME)
    if cam not in OBSTACLE_NAMES:
        await _add(cam, cam)

    if not any(frames.values()):
        frames["world"] = [_box(n, c, d) for n, c, d in STATIC_BOXES]
        labels = [g.label for g in frames["world"]]
        print("  obstacles: using static table/wall/ceiling boxes", flush=True)

    state = _world_state(frames)
    _cache = (now, state, labels, voxels)
    print(f"  obstacles: {', '.join(labels) or 'none'}", flush=True)
    return state, labels, voxels


def _box_extra(label: str, size: float, center) -> dict:
    return {
        "label": label,
        "type": "box",
        "x": float(size),
        "y": float(size),
        "z": float(size),
        "translation": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        },
    }


def extra_for(
    world_state: WorldState | None, voxels: list[dict] | None = None
) -> dict:
    extra: dict = {"collision_buffer_mm": COLLISION_BUFFER_MM}
    extra.update(voxels_to_extra(voxels or []))
    if world_state is None:
        return extra
    extra["obstacle_frames"] = [o.reference_frame for o in world_state.obstacles]
    extra["obstacle_labels"] = [
        g.label
        for o in world_state.obstacles
        for g in o.geometries
        if g.label
    ]
    extra["has_point_clouds"] = any(
        g.HasField("pointcloud") for o in world_state.obstacles for g in o.geometries
    )
    extra["has_voxels"] = bool(voxels)
    return extra


def voxels_to_extra(voxels: list[dict]) -> dict:
    """Struct-safe extra accepted by move_to_joint_positions."""
    obstacles: list[dict] = []
    for cloud in voxels:
        size = float(cloud.get("resolution_mm") or cloud.get("box", {}).get("x") or VOXEL_MM)
        centers = cloud.get("centers_mm") or []
        label = str(cloud.get("label") or "voxels")
        for i, center in enumerate(centers):
            obstacles.append(_box_extra(f"{label}-voxel-{i}", size, center))
    return {"voxels": list(voxels), "obstacles": obstacles} if voxels else {}


def accept_move_extra(extra: dict | None) -> dict:
    """Normalize voxels / boxes / binary PCD into Struct-safe joint extra."""
    if not extra:
        return {}
    out = {k: v for k, v in extra.items() if k not in {"point_cloud", "pcd", "world_state"}}
    voxels = list(out.get("voxels") or [])
    pcd = extra.get("point_cloud") or extra.get("pcd")
    if isinstance(pcd, (bytes, bytearray)):
        _boxes, payload, _voxel_pcd = pcd_to_voxels(bytes(pcd), str(extra.get("label") or "pcd"))
        if payload:
            voxels.append(payload)
    if voxels:
        packed = voxels_to_extra(voxels)
        out["voxels"] = packed["voxels"]
        existing = [o for o in (out.get("obstacles") or []) if isinstance(o, dict)]
        seen = {(o.get("label"), tuple((o.get("translation") or {}).values())) for o in existing}
        for box in packed["obstacles"]:
            key = (box["label"], tuple(box["translation"].values()))
            if key not in seen:
                existing.append(box)
        out["obstacles"] = existing
        out["has_voxels"] = True
    # Drop anything protobuf / bytes that Struct cannot encode.
    clean: dict = {}
    for key, value in out.items():
        if isinstance(value, (bytes, bytearray)):
            continue
        if hasattr(value, "SerializeToString"):
            continue
        clean[key] = value
    return clean


def pose_from_dict(values: dict) -> Pose:
    return Pose(
        x=float(values["x"]),
        y=float(values["y"]),
        z=float(values["z"]),
        o_x=float(values.get("o_x", 0)),
        o_y=float(values.get("o_y", 0)),
        o_z=float(values.get("o_z", -1)),
        theta=float(values.get("theta", 0)),
    )


def iter_labels(world_state: WorldState) -> Iterable[str]:
    for frame in world_state.obstacles:
        for geo in frame.geometries:
            yield geo.label or frame.reference_frame
