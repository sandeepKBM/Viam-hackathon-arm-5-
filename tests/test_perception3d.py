"""Offline verification of components/perception3d.py: the PCD-parsing and
segmenter-to-grasp bridge, against synthetic clouds encoded as real
Viam-style PCD bytes (ascii AND binary) -- no robot, no camera, no live
Viam. `VisionClient` is monkeypatched with a fake exposing
`from_robot(...).get_object_point_clouds(...)`, so `get_object_grasps`
itself is exercised end to end.
"""

from __future__ import annotations

import asyncio
import functools
import struct
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

import components.perception3d as perception3d
from components.grasp_affordance import GraspType
from components.perception3d import (
    PCDParseError,
    SPARSE_POINTS_THRESHOLD,
    get_object_grasps,
    grasps_from_point_cloud_objects,
    parse_viam_pcd,
)


def async_test(fn):
    """Local `asyncio.run` wrapper -- matches this repo's convention
    (tests/test_retry.py, tests/test_fast_planner.py) of not depending on
    pytest-asyncio."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


# ---------------------------------------------------------------------------
# Synthetic clouds (reused across ascii/binary encodings and grasp checks).
# ---------------------------------------------------------------------------


def _solid_cube(side_xy: float, height: float, n_per_axis: int = 12) -> np.ndarray:
    xs = np.linspace(0.0, side_xy, n_per_axis)
    ys = np.linspace(0.0, side_xy, n_per_axis)
    zs = np.linspace(0.0, height, n_per_axis)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _solid_cylinder(radius: float, height: float, n_xy: int = 61, n_z: int = 20) -> np.ndarray:
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


def _hollow_tube(inner_r, outer_r, height, n_theta=160, n_radii=6, n_z=30) -> np.ndarray:
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
# PCD encoders -- turn an (N,3) numpy array into real Viam-style PCD bytes,
# both DATA modes.
# ---------------------------------------------------------------------------


def _encode_ascii_pcd(points: np.ndarray) -> bytes:
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


def _encode_binary_pcd(points: np.ndarray, *, with_rgb: bool = False) -> bytes:
    n = points.shape[0]
    if with_rgb:
        fields, size, typ, count = "x y z rgb", "4 4 4 4", "F F F U", "1 1 1 1"
    else:
        fields, size, typ, count = "x y z", "4 4 4", "F F F", "1 1 1"
    header = (
        "# .PCD v0.7\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {size}\n"
        f"TYPE {typ}\n"
        f"COUNT {count}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")
    body = bytearray()
    for x, y, z in points:
        body += struct.pack("<fff", float(x), float(y), float(z))
        if with_rgb:
            body += struct.pack("<I", 0xFF00FF)
    return header + bytes(body)


# ---------------------------------------------------------------------------
# parse_viam_pcd round-trip
# ---------------------------------------------------------------------------


def test_parse_ascii_round_trip():
    pts = _solid_cube(side_xy=70.0, height=40.0)
    raw = _encode_ascii_pcd(pts)
    parsed = parse_viam_pcd(raw)
    assert parsed.shape == pts.shape
    assert np.allclose(np.sort(parsed, axis=0), np.sort(pts, axis=0), atol=1e-3)


def test_parse_binary_round_trip_xyz_only():
    pts = _hollow_tube(inner_r=20.0, outer_r=30.0, height=60.0, n_theta=40, n_radii=3, n_z=10)
    raw = _encode_binary_pcd(pts)
    parsed = parse_viam_pcd(raw)
    assert parsed.shape == pts.shape
    assert np.allclose(np.sort(parsed, axis=0), np.sort(pts, axis=0), atol=1e-4)


def test_parse_binary_round_trip_with_extra_rgb_field():
    # A real depth-camera PCD usually carries rgb/rgba too; the parser must
    # still recover just x/y/z correctly given the right stride.
    pts = _solid_cylinder(radius=20.0, height=90.0, n_xy=25, n_z=8)
    raw = _encode_binary_pcd(pts, with_rgb=True)
    parsed = parse_viam_pcd(raw)
    assert parsed.shape == pts.shape
    assert np.allclose(np.sort(parsed, axis=0), np.sort(pts, axis=0), atol=1e-4)


def test_parse_accepts_point_cloud_object_wrapper():
    pts = _solid_cube(side_xy=50.0, height=30.0, n_per_axis=6)
    raw = _encode_ascii_pcd(pts)
    fake_obj = SimpleNamespace(point_cloud=raw)
    parsed = parse_viam_pcd(fake_obj)
    assert parsed.shape == pts.shape


def test_parse_binary_compressed_not_implemented():
    header = (
        "# .PCD v0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary_compressed\n"
    ).encode("ascii")
    with pytest.raises(NotImplementedError):
        parse_viam_pcd(header + b"\x00" * 16)


def test_parse_rejects_malformed_header():
    with pytest.raises(PCDParseError):
        parse_viam_pcd(b"not a pcd file at all\n")


def test_parse_rejects_non_bytes_input():
    with pytest.raises(TypeError):
        parse_viam_pcd(12345)


# ---------------------------------------------------------------------------
# grasps_from_point_cloud_objects / get_object_grasps
# ---------------------------------------------------------------------------


def _pco(points: np.ndarray, *, label: str = "", binary: bool = False, center=None, dims=None):
    """Build a duck-typed PointCloudObject: `.point_cloud` PCD bytes +
    `.geometries.geometries[0]` with `.label`/`.center`/`.box.dims_mm`."""
    raw = _encode_binary_pcd(points) if binary else _encode_ascii_pcd(points)
    geom = SimpleNamespace(
        label=label,
        center=SimpleNamespace(x=center[0], y=center[1], z=center[2]) if center else None,
        box=SimpleNamespace(dims_mm=SimpleNamespace(x=dims[0], y=dims[1], z=dims[2])) if dims else None,
    )
    geoms_in_frame = SimpleNamespace(reference_frame="cam", geometries=[geom])
    return SimpleNamespace(point_cloud=raw, geometries=geoms_in_frame)


def test_grasps_from_objects_cube_is_top_down_ascii():
    pts = _solid_cube(side_xy=70.0, height=50.0)
    obj = _pco(pts, label="cube1", binary=False, center=(35.0, 35.0, 25.0))
    [result] = grasps_from_point_cloud_objects([obj])
    assert result.label == "cube1"
    assert result.grasp.grasp_type is GraspType.TOP_DOWN
    assert result.center_xyz == pytest.approx((35.0, 35.0, 25.0))
    assert result.n_points == pts.shape[0]
    assert result.note == ""


def test_grasps_from_objects_cylinder_is_side_binary():
    pts = _solid_cylinder(radius=25.0, height=130.0, n_xy=61, n_z=25)
    obj = _pco(pts, label="cyl1", binary=True)
    [result] = grasps_from_point_cloud_objects([obj])
    assert result.grasp.grasp_type is GraspType.SIDE
    assert result.n_points == pts.shape[0]


def test_grasps_from_objects_hollow_cup_is_inside_outside():
    pts = _hollow_tube(inner_r=25.0, outer_r=35.0, height=70.0)
    obj = _pco(pts, label="cup1", binary=True)
    [result] = grasps_from_point_cloud_objects([obj])
    assert result.grasp.grasp_type is GraspType.INSIDE_OUTSIDE
    assert result.n_points == pts.shape[0]


def test_grasps_falls_back_to_geometry_bbox_when_sparse():
    # Fewer points than SPARSE_POINTS_THRESHOLD -> geometry-bbox fallback,
    # not classify_grasp (which would need >= 8 points anyway, but this is
    # well under the module's own sparse-cloud threshold).
    n = max(SPARSE_POINTS_THRESHOLD - 5, 8)
    pts = _solid_cube(side_xy=60.0, height=20.0, n_per_axis=4)[:n]
    obj = _pco(pts, label="sparse1", binary=False, dims=(60.0, 60.0, 20.0), center=(30.0, 30.0, 10.0))
    [result] = grasps_from_point_cloud_objects([obj])
    assert result.n_points == n
    assert "sparse" in result.note.lower()
    assert result.grasp.confidence < 0.3
    assert result.grasp.grasp_type is GraspType.TOP_DOWN  # 60x60x20 -> compact, not tall
    assert result.center_xyz == pytest.approx((30.0, 30.0, 10.0))


def test_grasps_falls_back_to_geometry_bbox_when_pcd_unparseable():
    bad_obj = SimpleNamespace(
        point_cloud=b"garbage, not a pcd\n",
        geometries=SimpleNamespace(
            geometries=[
                SimpleNamespace(
                    label="mystery",
                    center=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                    box=SimpleNamespace(dims_mm=SimpleNamespace(x=40.0, y=40.0, z=100.0)),
                )
            ]
        ),
    )
    [result] = grasps_from_point_cloud_objects([bad_obj])
    assert result.label == "mystery"
    assert result.n_points == 0
    assert "parse failed" in result.note.lower()
    assert result.grasp.grasp_type is GraspType.SIDE  # 40x40 footprint, 100 tall -> tall
    assert result.center_xyz == pytest.approx((1.0, 2.0, 3.0))


def test_grasps_without_geometries_uses_points_bbox_center():
    pts = _solid_cube(side_xy=40.0, height=20.0, n_per_axis=4)
    raw = _encode_ascii_pcd(pts)
    obj = SimpleNamespace(point_cloud=raw, geometries=None)
    [result] = grasps_from_point_cloud_objects([obj])
    assert result.label == "object_0"
    # No geometry -> center comes from classify_grasp's own centroid.
    assert result.center_xyz[0] == pytest.approx(20.0, abs=2.0)
    assert result.center_xyz[1] == pytest.approx(20.0, abs=2.0)


def test_multiple_objects_get_distinct_results():
    cube = _pco(_solid_cube(side_xy=70.0, height=50.0), label="cube1")
    cup = _pco(_hollow_tube(inner_r=25.0, outer_r=35.0, height=70.0), label="cup1", binary=True)
    results = grasps_from_point_cloud_objects([cube, cup])
    assert [r.label for r in results] == ["cube1", "cup1"]
    assert results[0].grasp.grasp_type is GraspType.TOP_DOWN
    assert results[1].grasp.grasp_type is GraspType.INSIDE_OUTSIDE


# ---------------------------------------------------------------------------
# get_object_grasps: the async Viam-facing wrapper, monkeypatching VisionClient.
# ---------------------------------------------------------------------------


class _FakeSegmenterClient:
    def __init__(self, objects):
        self._objects = objects

    async def get_object_point_clouds(self, camera_name: str):
        assert camera_name == "cam"
        return self._objects


class _FakeVisionClient:
    last_from_robot_args = None

    @classmethod
    def from_robot(cls, machine, name):
        cls.last_from_robot_args = (machine, name)
        return machine.segmenter_client


@async_test
async def test_get_object_grasps_end_to_end(monkeypatch):
    cube = _pco(_solid_cube(side_xy=70.0, height=50.0), label="cube1")
    cyl = _pco(_solid_cylinder(radius=25.0, height=130.0), label="cyl1", binary=True)
    fake_client = _FakeSegmenterClient([cube, cyl])
    machine = SimpleNamespace(segmenter_client=fake_client)

    monkeypatch.setattr(perception3d, "VisionClient", _FakeVisionClient)

    results = await get_object_grasps(machine, "shape-segmenter", "cam")

    assert _FakeVisionClient.last_from_robot_args == (machine, "shape-segmenter")
    assert [r.label for r in results] == ["cube1", "cyl1"]
    assert results[0].grasp.grasp_type is GraspType.TOP_DOWN
    assert results[1].grasp.grasp_type is GraspType.SIDE


@async_test
async def test_get_object_grasps_empty_scene(monkeypatch):
    fake_client = _FakeSegmenterClient([])
    machine = SimpleNamespace(segmenter_client=fake_client)
    monkeypatch.setattr(perception3d, "VisionClient", _FakeVisionClient)

    results = await get_object_grasps(machine, "shape-segmenter", "cam")
    assert results == []
