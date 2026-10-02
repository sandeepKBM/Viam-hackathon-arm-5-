"""Nominal (uncalibrated) pour setup, built at run time from the live machine.

Used only with ``--nominal``, at the operator's explicit request to skip the
calibration procedure. Every value here is a stand-in, so expect cm-level
error; run with an empty container first.

- camera model: live GetProperties (so the identity check matches itself)
- camera mount: the live Viam frame system (the shared fragment's nominal mount)
- table plane: RANSAC fit to the observation's own depth
- TCP: fingertip length from the taught floor (MIN_Z was taught with the
  fingertips on the table) minus half a pad; UFactory gripper values otherwise
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Optional

import numpy as np

from components.calibration import ACCEPTANCE, SCHEMA_VERSION, compute_setup_hash
from components.constants import MIN_Z
from components.transforms import T_to_json, apply, make_T

# UFactory xArm Gripper (G1) nominal geometry; collision radii padded.
NOMINAL_GRIPPER = {
    "closing_axis_tcp": [0.0, 1.0, 0.0],   # fingers separate along the camera image x = flange y
    "pad_length_mm": 30.0,
    "pad_width_mm": 22.0,
    "finger_reach_mm": 40.0,
    "max_open_mm": 84.0,
    "jaw_mm_per_pos": 86.0 / 850.0,
    "jaw_mm_at_pos0": 0.0,
    "collision": {
        "wrist_len_mm": 80.0,
        "wrist_radius_mm": 45.0,
        "body_radius_mm": 45.0,
        "finger_radius_mm": 8.0,
        "camera_radius_mm": 40.0,
    },
}
NOMINAL_TIP_MM = 172.0


def fit_table_plane(frames, *, max_points: int = 40000, iters: int = 300, tol_mm: float = 4.0,
                    seed: int = 0) -> dict:
    """Dominant near-horizontal plane in the observation (world frame)."""
    f0 = frames[0]
    d = np.asarray(f0.depth_mm, dtype=float)
    ys, xs = np.nonzero(d > 0)
    if len(xs) < 1000:
        raise ValueError("not enough depth to fit the table plane")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(xs), size=min(max_points, len(xs)), replace=False)
    P = apply(f0.T_world_cam, f0.model.deproject(xs[pick].astype(float), ys[pick].astype(float), d[ys[pick], xs[pick]]))
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        if n[2] < 0.97:                       # within ~14 deg of horizontal
            continue
        dist = np.abs((P - a) @ n)
        inl = dist < tol_mm
        if best is None or inl.sum() > best[0].sum():
            best = (inl, n)
    if best is None:
        raise ValueError("no horizontal plane found in the observation")
    Q = P[best[0]]
    c = Q.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q - c)
    n = Vt[-1] if Vt[-1][2] > 0 else -Vt[-1]
    dd = -float(n @ c)
    rms = float(np.sqrt(np.mean((Q @ n + dd) ** 2)))
    return {"normal": [float(v) for v in n], "d": dd, "rms_mm": rms,
            "inlier_frac": float(best[0].mean()), "method": "RANSAC on the observation depth (nominal)"}


def build_nominal_doc(obs, *, min_z: float = MIN_Z) -> tuple[dict, dict]:
    """Returns (setup doc with status 'nominal', report)."""
    live = obs.live
    if live is None or live.T_flange_cam_viam is None:
        raise ValueError("nominal mode needs the live camera frame from the Viam frame system")
    f0 = obs.frames[0]
    table = fit_table_plane(obs.frames)
    n = np.asarray(table["normal"])
    table_z = -table["d"] / n[2]
    tip = min_z - table_z
    tip_src = "taught floor MIN_Z minus fitted table height"
    if not (150.0 <= tip <= 200.0):
        tip, tip_src = NOMINAL_TIP_MM, f"UFactory nominal (floor-derived {min_z - table_z:.1f} mm out of range)"
    tcp_z = tip - NOMINAL_GRIPPER["pad_length_mm"] / 2.0
    m = f0.model
    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": "nominal",
        "setup_id": "armfarm5-nominal-uncalibrated",
        "calibrated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "max_age_days": 1,
        "machine": {"arm_name": "arm", "camera_name": "cam", "gripper_name": "gripper", "world_frame": "world"},
        "camera": {
            "mount": "eye_in_hand",
            "mount_source": "live Viam frame system (cam parent 'arm'); nominal, uncalibrated",
            "color_profile": {"width": m.width, "height": m.height},
            "depth_profile": {"width": f0.depth_mm.shape[1], "height": f0.depth_mm.shape[0],
                              "units_mm_per_count": 1.0, "encoding": f0.depth_encoding},
            "intrinsics": {"fx": m.fx, "fy": m.fy, "cx": m.cx, "cy": m.cy, "width": m.width, "height": m.height},
            "distortion": {"model": m.dist_model, "coeffs": list(m.coeffs)},
            "reported_extrinsics": live.reported_extrinsics,
            "alignment": {"method": "runtime per-object edge registration gate only", "verified": False},
        },
        "transforms": {
            "T_flange_cam": T_to_json(live.T_flange_cam_viam),
            "T_flange_tcp": T_to_json(make_T(np.eye(3), (0.0, 0.0, tcp_z))),
        },
        "gripper": dict(NOMINAL_GRIPPER),
        "table": table,
        "obstacles": [
            {"name": "wall-front", "min": [690, -1500, -500], "max": [790, 1500, 1100]},
            {"name": "wall-side", "min": [-1500, -550, -500], "max": [1500, -450, 1100]},
            {"name": "ceiling", "min": [-1500, -1500, 1000], "max": [1500, 1500, 1100]},
        ],
        "validation": {"acceptance": dict(ACCEPTANCE), "touchoff": None},
    }
    doc["setup_hash"] = compute_setup_hash(doc)
    report = {"table_z_mm": round(float(table_z), 2), "table_rms_mm": round(table["rms_mm"], 2),
              "fingertip_mm": round(float(tip), 1), "tip_source": tip_src, "tcp_z_mm": round(float(tcp_z), 1)}
    return doc, report


def nominal_problems(doc: dict) -> list:
    """Structural checks for a nominal doc (the calibrated-only checks are skipped)."""
    from components.calibration import parse_setup

    _, problems = parse_setup(doc, allow_nominal=True)
    return problems


def angle_from_vertical_deg(n) -> float:
    n = np.asarray(n, dtype=float)
    return math.degrees(math.acos(float(np.clip(n[2] / np.linalg.norm(n), -1, 1))))


__all__ = ["build_nominal_doc", "fit_table_plane", "NOMINAL_GRIPPER", "nominal_problems"]
