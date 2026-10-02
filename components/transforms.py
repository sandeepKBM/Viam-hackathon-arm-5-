"""Rigid-transform math for the calibrated pour. Pure numpy, no robot I/O.

Conventions (the same ones Viam uses, so values round-trip through the
frame system unchanged):

- Poses are 4x4 homogeneous matrices ``T_a_b``: maps points expressed in
  frame ``b`` into frame ``a`` (``p_a = T_a_b @ p_b``). Millimetres.
- ``world`` is the arm base (the live config parents ``arm`` to ``world``
  with an identity frame).
- Viam orientation vectors (OV): ``R = Rz(lon) @ Ry(lat) @ Rz(theta)`` with
  ``(ox, oy, oz) = R[:, 2]``; ``theta`` is degrees in the Pose proto. At the
  poles (|oz| ~ 1) longitude is 0, matching rdk spatialmath.
- Camera frames follow the pinhole convention Viam's RealSense module and
  OpenCV share: +x right, +y down, +z along the optical axis.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

_POLE_EPS = 1e-4


def rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def axis_angle(axis: Sequence[float], rad: float) -> np.ndarray:
    """Rodrigues rotation about a (not necessarily unit) axis."""
    k = np.asarray(axis, dtype=float)
    n = np.linalg.norm(k)
    if n < 1e-12:
        return np.eye(3)
    k = k / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(rad) * K + (1 - math.cos(rad)) * (K @ K)


def log_rotation(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, radians) of a rotation matrix."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if math.pi - theta < 1e-6:
        # Near 180 deg: axis from the symmetric part.
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        i = int(np.argmax(axis))
        axis = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return axis / np.linalg.norm(axis) * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(theta)) * theta


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return math.degrees(np.linalg.norm(log_rotation(R_a.T @ R_b)))


def make_T(R: np.ndarray | None = None, t: Sequence[float] | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform an (N, 3) or (3,) array of points."""
    p = np.asarray(pts, dtype=float)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return p @ T[:3, :3].T + T[:3, 3]


def ov_to_matrix(ox: float, oy: float, oz: float, theta_deg: float) -> np.ndarray:
    v = np.array([ox, oy, oz], dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("orientation vector has zero length")
    v = v / n
    lat = math.acos(float(np.clip(v[2], -1.0, 1.0)))
    lon = math.atan2(v[1], v[0]) if 1.0 - abs(v[2]) > _POLE_EPS else 0.0
    return rot_z(lon) @ rot_y(lat) @ rot_z(math.radians(theta_deg))


def matrix_to_ov(R: np.ndarray) -> tuple[float, float, float, float]:
    o = R[:, 2] / np.linalg.norm(R[:, 2])
    lat = math.acos(float(np.clip(o[2], -1.0, 1.0)))
    lon = math.atan2(o[1], o[0]) if 1.0 - abs(o[2]) > _POLE_EPS else 0.0
    M = (rot_z(lon) @ rot_y(lat)).T @ R
    theta = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return float(o[0]), float(o[1]), float(o[2]), float(theta)


def pose_to_T(pose) -> np.ndarray:
    """Viam Pose proto / dict / object with x,y,z,o_x,o_y,o_z,theta -> T."""
    get = (lambda k: pose[k]) if isinstance(pose, dict) else (lambda k: getattr(pose, k))
    R = ov_to_matrix(get("o_x"), get("o_y"), get("o_z"), get("theta"))
    return make_T(R, (get("x"), get("y"), get("z")))


def T_to_pose(T: np.ndarray) -> dict:
    ox, oy, oz, theta = matrix_to_ov(T[:3, :3])
    return {
        "x": float(T[0, 3]),
        "y": float(T[1, 3]),
        "z": float(T[2, 3]),
        "o_x": ox,
        "o_y": oy,
        "o_z": oz,
        "theta": theta,
    }


def T_to_json(T: np.ndarray) -> list:
    return [[float(v) for v in row] for row in np.asarray(T)]


def T_from_json(rows) -> np.ndarray:
    T = np.asarray(rows, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got shape {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1]) or not is_rotation(T[:3, :3]):
        raise ValueError("transform is not rigid (bad rotation or last row)")
    return T


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0:
        U[:, -1] *= -1
        out = U @ Vt
    return out


def slerp_R(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    return R0 @ _rot_from_vec(log_rotation(R0.T @ R1) * s)


def _rot_from_vec(w: np.ndarray) -> np.ndarray:
    return axis_angle(w, float(np.linalg.norm(w)))


def interpolate_T(T0: np.ndarray, T1: np.ndarray, max_step_mm: float, max_step_deg: float) -> list[np.ndarray]:
    """Position + orientation interpolated together (lerp + slerp), end
    inclusive, start exclusive. Step count honours both bounds."""
    d = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    a = rotation_angle_deg(T0[:3, :3], T1[:3, :3])
    n = max(1, int(math.ceil(max(d / max_step_mm, a / max_step_deg))))
    out = []
    for i in range(1, n + 1):
        s = i / n
        out.append(make_T(slerp_R(T0[:3, :3], T1[:3, :3], s), (1 - s) * T0[:3, 3] + s * T1[:3, 3]))
    return out


# ---------------------------------------------------------------------------
# Calibration solvers
# ---------------------------------------------------------------------------


def hand_eye_park_martin(
    T_base_flange: Sequence[np.ndarray], T_cam_target: Sequence[np.ndarray]
) -> tuple[np.ndarray, dict]:
    """Eye-in-hand AX = XB (Park & Martin 1994), X = T_flange_cam.

    Uses every sample pair. Returns (X, diagnostics). Raises ValueError when
    the motion set is degenerate (too few samples or rotations about one axis
    only), which would leave part of X unobservable.
    """
    n = len(T_base_flange)
    if n != len(T_cam_target) or n < 3:
        raise ValueError("hand-eye needs >= 3 paired samples")
    alphas, betas, As, Bs = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            A = invert(T_base_flange[i]) @ T_base_flange[j]
            B = T_cam_target[i] @ invert(T_cam_target[j])
            a, b = log_rotation(A[:3, :3]), log_rotation(B[:3, :3])
            if np.linalg.norm(a) < math.radians(2.0):
                continue  # near-pure translation pairs carry no rotation info
            alphas.append(a)
            betas.append(b)
            As.append(A)
            Bs.append(B)
    if len(alphas) < 2:
        raise ValueError("hand-eye motions are degenerate: need rotations about >= 2 axes")
    Aa = np.array(alphas)
    axes = Aa / np.linalg.norm(Aa, axis=1, keepdims=True)
    sv = np.linalg.svd(axes, compute_uv=False)
    if sv.size < 2 or sv[1] / max(sv[0], 1e-12) < 0.2:
        raise ValueError(
            "hand-eye motions are degenerate: rotation axes are nearly parallel "
            f"(axis singular values {np.round(sv, 3).tolist()})"
        )
    M = np.zeros((3, 3))
    for a, b in zip(alphas, betas):
        M += np.outer(b, a)
    w, V = np.linalg.eigh(M.T @ M)
    if np.min(w) <= 1e-12:
        raise ValueError("hand-eye rotation is unobservable (singular M^T M)")
    R = V @ np.diag(1.0 / np.sqrt(w)) @ V.T @ M.T
    R = orthonormalize(R)
    C = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    d = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    t, *_ = np.linalg.lstsq(C, d, rcond=None)
    X = make_T(R, t)
    rot_res = [
        rotation_angle_deg(A[:3, :3] @ R, R @ B[:3, :3]) for A, B in zip(As, Bs)
    ]
    trans_res = [
        float(np.linalg.norm((A @ X)[:3, 3] - (X @ B)[:3, 3])) for A, B in zip(As, Bs)
    ]
    return X, {
        "pairs": len(As),
        "axis_singular_values": [float(v) for v in sv],
        "pair_rot_residual_deg_max": float(max(rot_res)),
        "pair_trans_residual_mm_max": float(max(trans_res)),
    }


def refine_hand_eye(
    X0: np.ndarray,
    T_base_flange: Sequence[np.ndarray],
    T_cam_target: Sequence[np.ndarray],
    target_pts: np.ndarray,
    iters: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton on 3D target-point consistency: jointly refine X and the
    (fixed) target pose in the base frame so every sample predicts the same
    base-frame target points. Returns (X, T_base_target)."""
    Tbt0 = T_base_flange[0] @ X0 @ T_cam_target[0]

    def unpack(p):
        X = make_T(_rot_from_vec(p[0:3]) @ X0[:3, :3], X0[:3, 3] + p[3:6])
        B = make_T(_rot_from_vec(p[6:9]) @ Tbt0[:3, :3], Tbt0[:3, 3] + p[9:12])
        return X, B

    def residual(p):
        X, B = unpack(p)
        ref = apply(B, target_pts)
        res = [apply(Tf @ X @ Tc, target_pts) - ref for Tf, Tc in zip(T_base_flange, T_cam_target)]
        return np.concatenate([r.ravel() for r in res])

    p = np.zeros(12)
    for _ in range(iters):
        r = residual(p)
        J = np.empty((r.size, 12))
        for k in range(12):
            h = 1e-6 if k % 6 < 3 else 1e-4
            dp = np.zeros(12)
            dp[k] = h
            J[:, k] = (residual(p + dp) - r) / h
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        if np.linalg.norm(step) < 1e-9:
            break
    return unpack(p)


def solve_pivot_tcp(T_base_flange: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Pivot (touch-the-same-point) TCP calibration.

    Each sample is a flange pose with the gripper-pad centre on one fixed
    point p: ``R_i c + t_i = p``. Solves for c (pad centre in the flange
    frame) and p (point in base). Returns (c, p, per-sample residual mm).
    """
    n = len(T_base_flange)
    if n < 4:
        raise ValueError("pivot TCP calibration needs >= 4 poses")
    Rs = [T[:3, :3] for T in T_base_flange]
    spread = max(rotation_angle_deg(Rs[0], R) for R in Rs[1:])
    if spread < 20.0:
        raise ValueError(f"pivot poses span only {spread:.1f} deg; need >= 20 deg of rotation")
    A = np.vstack([np.hstack([R, -np.eye(3)]) for R in Rs])
    b = -np.concatenate([T[:3, 3] for T in T_base_flange])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c, p = x[:3], x[3:]
    res = [float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - p)) for T in T_base_flange]
    return c, p, res


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares plane n.x + d = 0 with n pointing +z. Returns
    (n, d, rms residual)."""
    P = np.asarray(points, dtype=float)
    if len(P) < 3:
        raise ValueError("plane fit needs >= 3 points")
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c)
    n = Vt[-1]
    if n[2] < 0:
        n = -n
    d = -float(n @ c)
    rms = float(np.sqrt(np.mean((P @ n + d) ** 2)))
    return n, d, rms


def percentile_stats(errors: Iterable[float]) -> dict:
    e = np.asarray(list(errors), dtype=float)
    if e.size == 0:
        return {"n": 0, "p95": None, "max": None, "mean": None}
    return {
        "n": int(e.size),
        "p95": float(np.percentile(e, 95)),
        "max": float(e.max()),
        "mean": float(e.mean()),
    }
