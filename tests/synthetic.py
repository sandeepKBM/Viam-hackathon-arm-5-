"""Synthetic aligned RGB-D scenes and a calibrated setup for offline tests.

A tiny vectorized ray caster: table plane, upright bottles (body + neck),
upright solid cans, hollow cups, and lying (horizontal) cylinders. Depth is
z-depth in the camera frame (what RealSense reports), masks are exact
object silhouettes (stand-ins for SAM).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from components.calibration import ACCEPTANCE, SCHEMA_VERSION, compute_setup_hash
from components.rgbd import CameraModel, RGBDFrame
from components.transforms import T_to_json, invert, make_T, ov_to_matrix, rot_x, rot_z

W, H = 640, 360
MODEL = CameraModel(fx=455.0, fy=455.0, cx=320.0, cy=180.0, width=W, height=H)
TABLE_Z = 5.0

# Nominal flange->camera like the live fragment, flange->TCP like an xArm gripper.
T_FLANGE_CAM = make_T(ov_to_matrix(-0.030391, -0.003538, 0.999532, -97.731173), (83.0, -14.0, 18.0))
T_FLANGE_TCP = make_T(np.eye(3), (0.0, 0.0, 172.0))
HOME_FLANGE = make_T(ov_to_matrix(0.0, 0.0, -1.0, -26.78), (281.0, -87.0, 533.0))


def camera_pose_for_flange(T_world_flange: np.ndarray) -> np.ndarray:
    return T_world_flange @ T_FLANGE_CAM


@dataclass
class Bottle:
    x: float
    y: float
    body_r: float = 32.0
    neck_r: float = 14.0
    shoulder_h: float = 130.0
    height: float = 180.0
    obj_id: int = 1


@dataclass
class Can:
    x: float
    y: float
    r: float = 33.0
    height: float = 122.0
    obj_id: int = 2


@dataclass
class Cup:
    x: float
    y: float
    r_out: float = 42.0
    wall: float = 4.0
    height: float = 95.0
    bottom: float = 8.0
    obj_id: int = 3


@dataclass
class LyingCylinder:
    x: float
    y: float
    r: float = 32.0
    length: float = 180.0
    yaw_deg: float = 0.0
    obj_id: int = 4


@dataclass
class SynScene:
    objects: list = field(default_factory=list)
    table_z: float = TABLE_Z


def _rays(T_world_cam: np.ndarray, model: CameraModel):
    u, v = np.meshgrid(np.arange(model.width), np.arange(model.height))
    x = (u - model.cx) / model.fx
    y = (v - model.cy) / model.fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    d = d_cam @ T_world_cam[:3, :3].T      # t along d == z-depth
    o = T_world_cam[:3, 3]
    return o, d


def _cyl_side(o, d, cx, cy, r, z0, z1, inner=False):
    ox, oy = o[0] - cx, o[1] - cy
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    t = np.full(len(d), np.inf)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    root = (-b + sq) / (2 * a) if inner else (-b - sq) / (2 * a)
    z = o[2] + root * d[:, 2]
    hit = ok & (root > 1e-6) & (z >= z0) & (z <= z1)
    t[hit] = root[hit]
    return t


def _disk(o, d, cx, cy, z, r_out, r_in=0.0):
    t = np.full(len(d), np.inf)
    ok = np.abs(d[:, 2]) > 1e-12
    root = np.where(ok, (z - o[2]) / np.where(ok, d[:, 2], 1), np.inf)
    px = o[0] + root * d[:, 0] - cx
    py = o[1] + root * d[:, 1] - cy
    rr = px * px + py * py
    hit = ok & (root > 1e-6) & (rr <= r_out * r_out) & (rr >= r_in * r_in)
    t[hit] = root[hit]
    return t


def _lying(o, d, obj: LyingCylinder, table_z: float):
    # Cylinder along axis e (horizontal) through centre c at height r.
    e = np.array([math.cos(math.radians(obj.yaw_deg)), math.sin(math.radians(obj.yaw_deg)), 0.0])
    c = np.array([obj.x, obj.y, table_z + obj.r])
    oc = o - c
    dp = d - np.outer(d @ e, e)
    op = oc - (oc @ e) * e
    a = (dp ** 2).sum(1)
    b = 2 * (dp @ op)
    cc = op @ op - obj.r ** 2
    disc = b * b - 4 * a * cc
    t = np.full(len(d), np.inf)
    ok = disc >= 0
    root = (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * np.where(a > 1e-12, a, 1))
    s = (oc @ e) + root * (d @ e)
    hit = ok & (root > 1e-6) & (np.abs(s) <= obj.length / 2)
    t[hit] = root[hit]
    return t


def render(scene: SynScene, T_world_cam: np.ndarray, model: CameraModel = MODEL):
    """Returns (depth_mm float32 (H,W), id_map int (H,W)); id 0 = table/none."""
    o, d = _rays(T_world_cam, model)
    best = _disk(o, d, 0, 0, scene.table_z, 1e6)
    ids = np.zeros(len(d), dtype=int)
    tz = scene.table_z

    def take(t, oid):
        nonlocal best, ids
        m = t < best
        best = np.where(m, t, best)
        ids = np.where(m, oid, ids)

    for ob in scene.objects:
        if isinstance(ob, Bottle):
            take(_cyl_side(o, d, ob.x, ob.y, ob.body_r, tz, tz + ob.shoulder_h), ob.obj_id)
            take(_disk(o, d, ob.x, ob.y, tz + ob.shoulder_h, ob.body_r, ob.neck_r), ob.obj_id)
            take(_cyl_side(o, d, ob.x, ob.y, ob.neck_r, tz + ob.shoulder_h, tz + ob.height), ob.obj_id)
            take(_disk(o, d, ob.x, ob.y, tz + ob.height, ob.neck_r), ob.obj_id)
        elif isinstance(ob, Can):
            take(_cyl_side(o, d, ob.x, ob.y, ob.r, tz, tz + ob.height), ob.obj_id)
            take(_disk(o, d, ob.x, ob.y, tz + ob.height, ob.r), ob.obj_id)
        elif isinstance(ob, Cup):
            r_in = ob.r_out - ob.wall
            take(_cyl_side(o, d, ob.x, ob.y, ob.r_out, tz, tz + ob.height), ob.obj_id)
            take(_disk(o, d, ob.x, ob.y, tz + ob.height, ob.r_out, r_in), ob.obj_id)
            take(_cyl_side(o, d, ob.x, ob.y, r_in, tz + ob.bottom, tz + ob.height, inner=True), ob.obj_id)
            take(_disk(o, d, ob.x, ob.y, tz + ob.bottom, r_in), ob.obj_id)
        elif isinstance(ob, LyingCylinder):
            take(_lying(o, d, ob, tz), ob.obj_id)
    depth = np.where(np.isfinite(best), best, 0.0).reshape(model.height, model.width).astype(np.float32)
    return depth, ids.reshape(model.height, model.width)


def color_from_ids(ids: np.ndarray) -> np.ndarray:
    palette = np.array([[200, 200, 200], [40, 120, 200], [30, 30, 200], [60, 60, 60], [40, 180, 40], [180, 40, 180]], np.uint8)
    return palette[np.clip(ids, 0, len(palette) - 1)]


def frames_for(scene: SynScene, T_world_flange: np.ndarray = HOME_FLANGE, n: int = 3, *,
               noise_mm: float = 0.6, seed: int = 0, model: CameraModel = MODEL,
               depth_model: Optional[CameraModel] = None, depth_offset_cam: Optional[np.ndarray] = None,
               holes: Optional[dict] = None, t0: float = 1_700_000_000.0):
    """Aligned frames (or deliberately misregistered when depth_model /
    depth_offset_cam are given: depth rendered from another imager but
    labelled as if aligned)."""
    rng = np.random.default_rng(seed)
    T_wc = camera_pose_for_flange(T_world_flange)
    depth, ids = render(scene, T_wc, model)
    if depth_model is not None or depth_offset_cam is not None:
        T_wd = T_wc @ make_T(np.eye(3), depth_offset_cam if depth_offset_cam is not None else np.zeros(3))
        depth, _ = render(scene, T_wd, depth_model or model)
    frames = []
    for i in range(n):
        dn = depth + rng.normal(0, noise_mm, depth.shape).astype(np.float32)
        dn[depth <= 0] = 0
        if holes:
            for oid, frac in holes.items():
                sel = ids == oid
                drop = sel & (rng.random(ids.shape) < frac)
                dn[drop] = 0
        frames.append(
            RGBDFrame(
                color=color_from_ids(ids), depth_mm=dn, model=model, aligned_to_color=True,
                depth_units_mm=1.0, captured_at=t0 + 0.1 * i, T_world_cam=T_wc,
                T_world_flange=T_world_flange, joints_before=[0.0] * 6, joints_after=[0.0] * 6,
                frame_id=f"syn{i}",
            )
        )
    return frames, ids


def setup_doc(*, calibrated_at: Optional[str] = None, T_flange_cam=T_FLANGE_CAM, table_z: float = TABLE_Z,
              model: CameraModel = MODEL, **overrides) -> dict:
    now = calibrated_at or dt.datetime.now(dt.timezone.utc).isoformat()
    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": "calibrated",
        "setup_id": "synthetic-test",
        "calibrated_at": now,
        "max_age_days": 7,
        "machine": {"arm_name": "arm", "camera_name": "cam", "gripper_name": "gripper", "world_frame": "world"},
        "camera": {
            "serial": "SYNTHETIC",
            "mount": "eye_in_hand",
            "color_profile": {"width": model.width, "height": model.height},
            "depth_profile": {"width": model.width, "height": model.height, "units_mm_per_count": 1.0,
                              "encoding": "image/vnd.viam.dep"},
            "intrinsics": {"fx": model.fx, "fy": model.fy, "cx": model.cx, "cy": model.cy,
                           "width": model.width, "height": model.height},
            "distortion": {"model": model.dist_model, "coeffs": list(model.coeffs)},
            "reported_extrinsics": None,
            "alignment": {"method": "synthetic", "verified": True},
        },
        "transforms": {"T_flange_cam": T_to_json(T_flange_cam), "T_flange_tcp": T_to_json(T_FLANGE_TCP)},
        "gripper": {
            "closing_axis_tcp": [0.0, 1.0, 0.0],
            "pad_length_mm": 30.0,
            "pad_width_mm": 20.0,
            "finger_reach_mm": 45.0,
            "max_open_mm": 86.0,
            "jaw_mm_per_pos": 0.1,
            "jaw_mm_at_pos0": 0.0,
            "collision": {"wrist_len_mm": 60.0, "wrist_radius_mm": 45.0, "body_radius_mm": 40.0,
                          "finger_radius_mm": 5.0, "camera_radius_mm": 30.0},
        },
        "table": {"normal": [0.0, 0.0, 1.0], "d": -table_z, "method": "synthetic"},
        "obstacles": [
            {"name": "wall-front", "min": [690, -1500, -500], "max": [790, 1500, 1100]},
            {"name": "wall-side", "min": [-1500, -550, -500], "max": [1500, -450, 1100]},
            {"name": "ceiling", "min": [-1500, -1500, 1000], "max": [1500, 1500, 1100]},
        ],
        "calibration": {"method": "synthetic", "sample_count": 14, "held_out": {"p95_mm": 1.0}},
        "validation": {
            "acceptance": dict(ACCEPTANCE),
            "touchoff": {"n": 9, "p95_xy_mm": 4.0, "max_xy_mm": 6.0, "p95_z_mm": 3.0},
        },
    }
    for k, v in overrides.items():
        doc[k] = v
    doc["setup_hash"] = compute_setup_hash(doc)
    return doc


def masks_from_ids(ids: np.ndarray) -> dict:
    return {int(i): (ids == i).astype(np.uint8) for i in np.unique(ids) if i != 0}


def T_world_cam_home() -> np.ndarray:
    return camera_pose_for_flange(HOME_FLANGE)


__all__ = [
    "Bottle", "Can", "Cup", "LyingCylinder", "SynScene", "render", "frames_for", "setup_doc",
    "masks_from_ids", "MODEL", "HOME_FLANGE", "T_FLANGE_CAM", "T_FLANGE_TCP", "TABLE_Z",
    "camera_pose_for_flange", "rot_x", "rot_z", "invert", "T_world_cam_home", "color_from_ids",
]
