"""Calibration procedures behind scripts/calibrate_pour_setup.py.

Pure functions over recorded samples so every solve is testable offline and
re-runnable from the saved sample files. Nothing here moves the robot.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Optional, Sequence

import cv2
import numpy as np

from components.calibration import ACCEPTANCE, MIN_HAND_EYE_SAMPLES
from components.rgbd import CameraModel
from components.transforms import (
    T_from_json,
    T_to_json,
    apply,
    hand_eye_park_martin,
    invert,
    make_T,
    matrix_to_ov,
    percentile_stats,
    refine_hand_eye,
    rotation_angle_deg,
    solve_pivot_tcp,
)

DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}


# ---------------------------------------------------------------------------
# ChArUco target
# ---------------------------------------------------------------------------


def charuco_board(cols: int, rows: int, square_mm: float, marker_mm: float, dictionary: str = "5X5_100"):
    d = cv2.aruco.getPredefinedDictionary(DICTS[dictionary])
    return cv2.aruco.CharucoBoard((cols, rows), float(square_mm), float(marker_mm), d)


def detect_board_pose(bgr: np.ndarray, board, model: CameraModel, min_corners: int = 10) -> Optional[dict]:
    """T_cam_board (mm) from ChArUco corners + PnP with the camera's own
    intrinsics/distortion. Returns None when too few corners are seen."""
    det = cv2.aruco.CharucoDetector(board)
    cc, ci, _, _ = det.detectBoard(bgr)
    if cc is None or ci is None or len(cc) < min_corners:
        return None
    obj, img = board.matchImagePoints(cc, ci)
    obj = obj.reshape(-1, 3).astype(np.float64)
    img = img.reshape(-1, 2).astype(np.float64)
    if board.checkCharucoCornersCollinear(ci):
        return None  # PnP is ill-conditioned on collinear corners
    dist =np.zeros(5) if model.is_pinhole else np.array((list(model.coeffs) + [0.0] * 5)[:5])
    if model.dist_model == "inverse_brown_conrady" and not model.is_pinhole:
        # OpenCV's model is the forward (Brown-Conrady) one: undistort the
        # pixels with the RealSense inverse model and solve as pinhole.
        xn, yn = model.normalized(img[:, 0], img[:, 1])
        img = np.stack([xn * model.fx + model.cx, yn * model.fy + model.cy], axis=1)
        dist = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(obj, img, model.K(), dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, model.K(), dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    R, _ = cv2.Rodrigues(rvec)
    return {
        "T_cam_board": make_T(R, tvec.reshape(3)),
        "reproj_rms_px": float(np.sqrt(np.mean(err ** 2))),
        "reproj_max_px": float(err.max()),
        "n_corners": int(len(obj)),
        "corner_ids": ci.reshape(-1).astype(int).tolist(),
    }


def board_points(board) -> np.ndarray:
    return np.asarray(board.getChessboardCorners(), dtype=float).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Hand-eye with held-out validation
# ---------------------------------------------------------------------------


def _consensus_target(Tbf, X, Tct) -> np.ndarray:
    Ts = [F @ X @ C for F, C in zip(Tbf, Tct)]
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    # chordal mean of rotations
    M = sum(T[:3, :3] for T in Ts)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return make_T(R, t)


def _sample_errors(Tbf, X, Tct, T_bt, pts) -> list:
    """Per-sample max 3D error (mm) of the target points predicted through the
    chain vs the consensus target pose; also split into XY and Z."""
    ref = apply(T_bt, pts)
    out = []
    for F, C in zip(Tbf, Tct):
        d = apply(F @ X @ C, pts) - ref
        out.append({
            "err3d_max_mm": float(np.linalg.norm(d, axis=1).max()),
            "err_xy_max_mm": float(np.linalg.norm(d[:, :2], axis=1).max()),
            "err_z_max_mm": float(np.abs(d[:, 2]).max()),
        })
    return out


def rotation_diversity(Tbf: Sequence[np.ndarray]) -> dict:
    Rs = [T[:3, :3] for T in Tbf]
    angles = [rotation_angle_deg(Rs[i], Rs[j]) for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
    return {"max_pair_rotation_deg": float(max(angles)) if angles else 0.0,
            "median_pair_rotation_deg": float(np.median(angles)) if angles else 0.0}


def solve_hand_eye_samples(samples: Sequence[dict], board_pts: np.ndarray, *, holdout_every: int = 4) -> dict:
    """Eye-in-hand X = T_flange_cam from >= 12 samples, each
    {"T_base_flange": 4x4, "T_cam_target": 4x4, "reproj_rms_px": float}.
    Every ``holdout_every``-th sample is held out; the model (Park-Martin or
    its 3D refinement) with the lower held-out p95 error wins."""
    if len(samples) < MIN_HAND_EYE_SAMPLES:
        raise ValueError(f"need >= {MIN_HAND_EYE_SAMPLES} hand-eye samples, got {len(samples)}")
    Tbf = [T_from_json(s["T_base_flange"]) if not isinstance(s["T_base_flange"], np.ndarray) else s["T_base_flange"] for s in samples]
    Tct = [T_from_json(s["T_cam_target"]) if not isinstance(s["T_cam_target"], np.ndarray) else s["T_cam_target"] for s in samples]
    div = rotation_diversity(Tbf)
    if div["max_pair_rotation_deg"] < 30.0:
        raise ValueError(f"poses are not diverse enough (max pair rotation {div['max_pair_rotation_deg']:.1f} deg < 30)")
    idx = list(range(len(samples)))
    held = [i for i in idx if i % holdout_every == holdout_every - 1]
    train = [i for i in idx if i not in held]
    X_pm, diag = hand_eye_park_martin([Tbf[i] for i in train], [Tct[i] for i in train])
    X_ref, _ = refine_hand_eye(X_pm, [Tbf[i] for i in train], [Tct[i] for i in train], board_pts)
    results = {}
    for name, X in (("park_martin", X_pm), ("park_martin+3d_refine", X_ref)):
        T_bt = _consensus_target([Tbf[i] for i in train], X, [Tct[i] for i in train])
        tr = _sample_errors([Tbf[i] for i in train], X, [Tct[i] for i in train], T_bt, board_pts)
        ho = _sample_errors([Tbf[i] for i in held], X, [Tct[i] for i in held], T_bt, board_pts)
        results[name] = {
            "X": X, "T_base_target": T_bt, "train": tr, "held": ho,
            "held_p95_mm": percentile_stats([e["err3d_max_mm"] for e in ho])["p95"],
        }
    best = min(results, key=lambda k: results[k]["held_p95_mm"])
    r = results[best]
    per_sample = []
    for k, i in enumerate(train):
        per_sample.append({"index": i, "split": "train", "reproj_rms_px": samples[i].get("reproj_rms_px"), **r["train"][k]})
    for k, i in enumerate(held):
        per_sample.append({"index": i, "split": "held_out", "reproj_rms_px": samples[i].get("reproj_rms_px"), **r["held"][k]})
    return {
        "method": f"eye_in_hand {best} (ChArUco, OpenCV solvePnP); train={len(train)} held_out={len(held)}",
        "T_flange_cam": r["X"],
        "T_base_target": r["T_base_target"],
        "per_sample": sorted(per_sample, key=lambda e: e["index"]),
        "train": percentile_stats([e["err3d_max_mm"] for e in r["train"]]),
        "held_out": {
            **percentile_stats([e["err3d_max_mm"] for e in r["held"]]),
            "xy": percentile_stats([e["err_xy_max_mm"] for e in r["held"]]),
            "z": percentile_stats([e["err_z_max_mm"] for e in r["held"]]),
        },
        "candidates": {k: v["held_p95_mm"] for k, v in results.items()},
        "diagnostics": {**diag, **div},
    }


def table_from_board(T_base_board: np.ndarray, board_thickness_mm: float) -> dict:
    """Board lying flat on the table: table plane = board plane shifted down by
    the board thickness along the (upward) normal."""
    n = T_base_board[:3, 2].copy()
    if n[2] < 0:
        n = -n
    p = T_base_board[:3, 3] - n * board_thickness_mm
    return {"normal": [float(v) for v in n], "d": float(-n @ p), "method": "charuco board pose (hand-eye consensus)",
            "board_thickness_mm": board_thickness_mm}


# ---------------------------------------------------------------------------
# TCP, jaw, touch-off
# ---------------------------------------------------------------------------


def solve_tcp(flange_poses: Sequence[np.ndarray]) -> dict:
    c, p, res = solve_pivot_tcp(flange_poses)
    return {
        "T_flange_tcp": make_T(np.eye(3), c),
        "pivot_point_base_mm": [float(v) for v in p],
        "residuals_mm": res,
        "rms_mm": float(np.sqrt(np.mean(np.square(res)))),
        "method": f"pivot (same fixed point, {len(flange_poses)} orientations), TCP axes = flange axes",
    }


def fit_jaw(samples: Sequence[tuple]) -> dict:
    """Samples of (gauge_width_mm, gripper_pos). Linear width = a + b * pos."""
    if len(samples) < 3:
        raise ValueError("jaw fit needs >= 3 gauge widths")
    w = np.array([s[0] for s in samples], dtype=float)
    pos = np.array([s[1] for s in samples], dtype=float)
    A = np.column_stack([np.ones_like(pos), pos])
    (a, b), *_ = np.linalg.lstsq(A, w, rcond=None)
    res = w - (a + b * pos)
    return {"jaw_mm_at_pos0": float(a), "jaw_mm_per_pos": float(b), "max_residual_mm": float(np.abs(res).max()),
            "samples": [[float(x), float(y)] for x, y in samples]}


def touchoff_stats(pairs: Sequence[tuple]) -> dict:
    """pairs of (predicted_world_xyz, touched_world_xyz)."""
    d = np.array([np.asarray(p, float) - np.asarray(t, float) for p, t in pairs])
    xy = np.linalg.norm(d[:, :2], axis=1)
    z = np.abs(d[:, 2])
    out = {
        "n": int(len(d)),
        "p95_xy_mm": float(np.percentile(xy, 95)),
        "max_xy_mm": float(xy.max()),
        "p95_z_mm": float(np.percentile(z, 95)),
        "errors": [[round(float(a), 2), round(float(b), 2)] for a, b in zip(xy, z)],
    }
    out["passed"] = bool(out["n"] >= 9 and out["p95_xy_mm"] <= ACCEPTANCE["p95_xy_mm"]
                         and out["max_xy_mm"] <= ACCEPTANCE["max_xy_mm"] and out["p95_z_mm"] <= ACCEPTANCE["p95_z_mm"])
    return out


def viam_frame_config(T_flange_cam: np.ndarray, *, parent: str = "arm", camera_name: str = "cam",
                      fragment_id: str = "fd2be28c-71e5-4b8a-90a8-a514dbe75ca7") -> dict:
    ox, oy, oz, th = matrix_to_ov(T_flange_cam[:3, :3])
    frame = {
        "parent": parent,
        "translation": {k: round(float(v), 3) for k, v in zip("xyz", T_flange_cam[:3, 3])},
        "orientation": {"type": "ov_degrees", "value": {"x": round(ox, 6), "y": round(oy, 6), "z": round(oz, 6),
                                                        "th": round(th, 4)}},
    }
    return {
        "frame": frame,
        "fragment_mods": [{
            "fragment_id": fragment_id,
            "mods": [
                {"$set": {f"components.{camera_name}.frame": frame}},
                {"$set": {f"components.{camera_name}.attributes.align_color_depth": True}},
            ],
        }],
        "note": "Apply on armfarm5 only (fragment_mods), not in the shared fragment.",
    }


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def T_json(T: np.ndarray) -> list:
    return T_to_json(T)


def frame_delta(A: np.ndarray, B: np.ndarray) -> dict:
    D = invert(A) @ B
    return {"mm": float(np.linalg.norm(D[:3, 3])), "deg": rotation_angle_deg(np.eye(3), D[:3, :3])}


def ov_deg(T: np.ndarray) -> dict:
    ox, oy, oz, th = matrix_to_ov(T[:3, :3])
    return {"x": ox, "y": oy, "z": oz, "th": th}


def angle_deg(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
