"""Manipulation poses for the pour: upright bottle/can and cup opening.

Moondream + SAM only propose a labelled mask. Everything here is geometry
from aligned RGB-D in the world frame, and every estimate either passes its
quality gates or carries a rejection reason. There is no fallback to
whole-frame or table depth: an object without good depth of its own is
unlocalizable, and the pour does not move.

Frames: points are in ``world`` (arm base). "Height" is signed distance
above the calibrated table plane. XY circle fits run in a table-plane frame.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from components.rgbd import RGBDFrame, edge_alignment_score
from components.transforms import apply, fit_plane

SOURCE_LABELS = ("bottle", "can")
CUP_LABEL = "cup"


@dataclass
class DepthParams:
    erode_px: int = 5
    exclude_depth_edges: bool = True
    edge_jump_mm: float = 15.0
    min_frames_valid_frac: float = 0.6
    temporal_tol_mm: float = 6.0
    frame_spread_tol_mm: float = 4.0
    min_valid_ratio: float = 0.30
    min_component_ratio: float = 0.6
    min_valid_px: int = 150
    min_edge_alignment: float = 0.35
    background_height_mm: float = 12.0
    max_background_fraction: float = 0.35

    @staticmethod
    def for_label(label: str) -> "DepthParams":
        if label == CUP_LABEL:
            # A cup rim is only a few pixels wide: keep it. Mixed pixels at its
            # edges always sit *below* the rim, so the rim band rejects them.
            return DepthParams(erode_px=1, exclude_depth_edges=False)
        return DepthParams()


@dataclass
class ShapeParams:
    # bottle / can (upright source)
    min_height_mm: float = 60.0
    max_height_mm: float = 330.0
    min_diameter_mm: float = 30.0
    max_diameter_mm: float = 110.0
    min_upright_ratio: float = 1.3       # height / diameter
    top_slab_mm: float = 4.0
    min_top_points: int = 20
    max_radius_mad_mm: float = 5.0
    # cup
    rim_band_mm: float = 7.0
    min_rim_points: int = 40
    min_rim_radius_mm: float = 25.0
    max_rim_radius_mm: float = 75.0
    max_rim_residual_mm: float = 3.0
    min_arc_coverage: float = 0.75
    max_rim_tilt_deg: float = 8.0
    min_cup_height_mm: float = 40.0
    min_interior_drop_mm: float = 20.0
    min_interior_points: int = 30


@dataclass
class DepthQuality:
    core_px: int = 0
    valid_px: int = 0
    valid_ratio: float = 0.0
    component_ratio: float = 0.0
    median_mm: float = 0.0
    trimmed_median_mm: float = 0.0
    mad_mm: float = 0.0
    frame_spread_mm: float = 0.0
    n_frames: int = 0
    edge_alignment: Optional[float] = None
    background_fraction: float = 0.0
    separation_mm: Optional[float] = None
    flags: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ObjectPoseEstimate:
    label: str
    mask: np.ndarray
    calibration_id: str
    frame_ids: list
    source_frame: str = "world"
    quality: DepthQuality = field(default_factory=DepthQuality)
    quality_flags: list = field(default_factory=list)
    rejection_reason: Optional[str] = None
    # geometry (world, mm)
    position: Optional[np.ndarray] = None     # bottle/can: mouth (top centre); cup: rim centre
    base: Optional[np.ndarray] = None         # bottle/can: axis point on the table
    axis: Optional[np.ndarray] = None         # unit, up
    dims: dict = field(default_factory=dict)
    error_bounds_mm: dict = field(default_factory=dict)
    center_px: Optional[tuple] = None
    key_px: dict = field(default_factory=dict)  # mouth / rim pixels for overlays
    points_world: Optional[np.ndarray] = None
    selected_px: Optional[np.ndarray] = None   # (N, 2) u, v actually used
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.rejection_reason is None

    def summary(self) -> dict:
        return {
            "label": self.label,
            "ok": self.ok,
            "rejection_reason": self.rejection_reason,
            "position_mm": None if self.position is None else [round(float(v), 2) for v in self.position],
            "base_mm": None if self.base is None else [round(float(v), 2) for v in self.base],
            "axis": None if self.axis is None else [round(float(v), 4) for v in self.axis],
            "dims_mm": {k: round(float(v), 2) for k, v in self.dims.items()},
            "error_bounds_mm": {k: round(float(v), 2) for k, v in self.error_bounds_mm.items()},
            "center_px": self.center_px,
            "key_px": self.key_px,
            "quality": self.quality.as_dict(),
            "quality_flags": list(self.quality_flags),
            "calibration_id": self.calibration_id,
            "frame_ids": list(self.frame_ids),
            "source_frame": self.source_frame,
            "extra": self.extra,
        }


def reject(est: ObjectPoseEstimate, reason: str) -> ObjectPoseEstimate:
    if est.rejection_reason is None:
        est.rejection_reason = reason
    return est


# ---------------------------------------------------------------------------
# Segmentation quality
# ---------------------------------------------------------------------------


def segmentation_problems(mask: np.ndarray, box: Optional[tuple] = None) -> list[str]:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    H, W = m.shape
    area = int(m.sum())
    out = []
    if area < 400:
        return ["mask_too_small"]
    ys, xs = np.nonzero(m)
    if xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3:
        out.append("truncated_at_image_border")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        if largest / area < 0.9:
            out.append("mask_fragmented")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = cv2.contourArea(c) / max(cv2.contourArea(hull), 1.0)
    if solidity < 0.8:
        out.append("mask_not_solid")
    x, y, w, h = cv2.boundingRect(c)
    if area / float(w * h) > 0.97:
        out.append("mask_is_a_box")  # SAM fell back to the detector box
    if box is not None:
        bx, by, bw, bh = box
        if bw * bh > 0 and area / float(bw * bh) < 0.2:
            out.append("mask_much_smaller_than_detection")
    return out


# ---------------------------------------------------------------------------
# Robust depth under a mask (no fallback)
# ---------------------------------------------------------------------------


def masked_points(
    frames: Sequence[RGBDFrame],
    mask: np.ndarray,
    table_height_fn,
    params: DepthParams,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], DepthQuality]:
    """World points for the mask's own pixels, robust across frames.

    Returns (points_world (N,3), pixels (N,2), quality). points are None when
    the depth under the mask is unusable; quality.flags says why.
    """
    q = DepthQuality(n_frames=len(frames))
    m = (np.asarray(mask) > 0).astype(np.uint8)
    k = 2 * params.erode_px + 1
    core = cv2.erode(m, np.ones((k, k), np.uint8)) > 0
    q.core_px = int(core.sum())
    if q.core_px < params.min_valid_px:
        q.flags.append("mask_core_too_small")
        return None, None, q

    D = np.stack([np.asarray(f.depth_mm, dtype=np.float32) for f in frames])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        D_med = np.nan_to_num(np.nanmedian(np.where(D > 0, D, np.nan), axis=0), nan=0.0)
    near_edge = np.zeros(core.shape, dtype=bool)
    if params.exclude_depth_edges:
        from components.rgbd import depth_edges

        near_edge = depth_edges(D_med, params.edge_jump_mm)
        near_edge = cv2.dilate(near_edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sel = core & ~near_edge
    ys, xs = np.nonzero(sel)
    vals = D[:, ys, xs]                         # (N_frames, P)
    valid = vals > 0
    need = max(1, int(math.ceil(params.min_frames_valid_frac * len(frames))))
    enough = valid.sum(axis=0) >= need
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vm = np.where(valid, vals, np.nan)
        med = np.nanmedian(vm, axis=0)
        spread = np.nanmax(vm, axis=0) - np.nanmin(vm, axis=0)
    keep = enough & np.isfinite(med) & (np.nan_to_num(spread, nan=1e9) <= params.temporal_tol_mm)
    q.valid_px = int(keep.sum())
    q.valid_ratio = q.valid_px / float(q.core_px)
    if q.valid_px < params.min_valid_px or q.valid_ratio < params.min_valid_ratio:
        q.flags.append("insufficient_object_depth")  # e.g. reflective/clear bottle holes
        return None, None, q

    kept_img = np.zeros(core.shape, np.uint8)
    kept_img[ys[keep], xs[keep]] = 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(kept_img, connectivity=8)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1 if n > 1 else 0
    q.component_ratio = float(stats[largest, cv2.CC_STAT_AREA]) / q.valid_px if largest else 0.0
    if q.component_ratio < params.min_component_ratio:
        q.flags.append("depth_fragmented")
        return None, None, q
    in_comp = labels[ys[keep], xs[keep]] == largest
    u = xs[keep][in_comp].astype(float)
    v = ys[keep][in_comp].astype(float)
    z = med[keep][in_comp].astype(float)

    q.median_mm = float(np.median(z))
    lo, hi = np.percentile(z, [10, 90])
    zt = z[(z >= lo) & (z <= hi)]
    q.trimmed_median_mm = float(np.median(zt)) if zt.size else q.median_mm
    q.mad_mm = float(np.median(np.abs(z - q.median_mm)))
    per_frame = [float(np.median(D[i, v.astype(int), u.astype(int)][D[i, v.astype(int), u.astype(int)] > 0]))
                 for i in range(len(frames)) if np.any(D[i, v.astype(int), u.astype(int)] > 0)]
    q.frame_spread_mm = float(max(per_frame) - min(per_frame)) if len(per_frame) > 1 else 0.0
    if q.frame_spread_mm > params.frame_spread_tol_mm:
        q.flags.append("temporally_inconsistent")
        return None, None, q

    f0 = frames[0]
    P_cam = f0.model.deproject(u, v, z)
    P = apply(f0.T_world_cam, P_cam)
    h = table_height_fn(P)
    q.background_fraction = float(np.mean(h < params.background_height_mm))
    top = float(np.percentile(h, 95))
    q.separation_mm = top
    q.edge_alignment = edge_alignment_score(m, D_med, jump_mm=params.edge_jump_mm)
    return P, np.stack([u, v], axis=1), q


# ---------------------------------------------------------------------------
# Table-plane frame helpers
# ---------------------------------------------------------------------------


class TableFrame:
    def __init__(self, normal: np.ndarray, d: float):
        n = np.asarray(normal, dtype=float)
        self.n = n / np.linalg.norm(n)
        ref = np.array([1.0, 0.0, 0.0]) if abs(self.n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = ref - (ref @ self.n) * self.n
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.n, self.e1)
        self.d = float(d)
        self.p0 = -self.d * self.n

    def to_plane(self, P: np.ndarray) -> np.ndarray:
        Q = np.asarray(P, dtype=float) - self.p0
        return np.stack([Q @ self.e1, Q @ self.e2, Q @ self.n], axis=-1)

    def to_world(self, abh: np.ndarray) -> np.ndarray:
        abh = np.asarray(abh, dtype=float)
        return self.p0 + abh[..., 0:1] * self.e1 + abh[..., 1:2] * self.e2 + abh[..., 2:3] * self.n

    def height(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, dtype=float) @ self.n + self.d


def fit_circle(xy: np.ndarray, iters: int = 4) -> tuple[np.ndarray, float, np.ndarray]:
    """Algebraic (Kasa) circle fit with MAD outlier rejection + a few
    Gauss-Newton geometric refinements. Returns (centre, radius, inlier mask)."""
    P = np.asarray(xy, dtype=float)
    keep = np.ones(len(P), dtype=bool)
    c = P.mean(axis=0)
    r = 0.0
    for _ in range(iters):
        Q = P[keep]
        A = np.column_stack([2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))])
        b = (Q ** 2).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = sol[:2]
        r = math.sqrt(max(sol[2] + c @ c, 1e-9))
        for _ in range(10):  # geometric refinement
            dv = Q - c
            dist = np.linalg.norm(dv, axis=1)
            dist = np.where(dist < 1e-9, 1e-9, dist)
            res = dist - r
            J = np.column_stack([-dv[:, 0] / dist, -dv[:, 1] / dist, -np.ones(len(Q))])
            step, *_ = np.linalg.lstsq(J, -res, rcond=None)
            c = c + step[:2]
            r = r + step[2]
            if np.linalg.norm(step) < 1e-6:
                break
        res_all = np.linalg.norm(P - c, axis=1) - r
        mad = np.median(np.abs(res_all[keep] - np.median(res_all[keep]))) + 1e-6
        new_keep = np.abs(res_all) <= max(3.0 * 1.4826 * mad, 1.0)
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return c, float(r), keep


def arc_coverage(xy: np.ndarray, center: np.ndarray, bins: int = 16) -> float:
    ang = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
    hist, _ = np.histogram(ang, bins=bins, range=(-math.pi, math.pi))
    return float(np.mean(hist >= 2))


def silhouette_radii(frame: RGBDFrame, mask: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Signed distance from the object's axis line to each outline ray.

    Only the side limbs of an upright solid of revolution are true tangent
    rays (distance == radius, one limb positive, the other negative); the top
    and bottom arcs of the outline give smaller magnitudes. Uses no depth at
    the (unreliable) boundary pixels."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(0)
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    xn, yn = frame.model.normalized(c[:, 0], c[:, 1])
    rays = np.stack([xn, yn, np.ones_like(xn)], axis=1) @ frame.T_world_cam[:3, :3].T
    o = frame.T_world_cam[:3, 3]
    a = np.asarray(axis_dir, dtype=float) / np.linalg.norm(axis_dir)
    cr = np.cross(rays, a)
    n = np.linalg.norm(cr, axis=1)
    ok = n > 1e-9
    return ((o - axis_point) @ cr[ok].T) / n[ok]


def limb_radius(signed: np.ndarray) -> tuple[Optional[float], float, float]:
    """Radius from the two limbs (mean of each side's upper decile). An axis
    offset e across the view adds +e to one limb and -e to the other, so the
    mean cancels it and half the difference measures it.
    Returns (radius, spread_mm, across_view_offset_mm)."""
    pos, neg = signed[signed > 0], -signed[signed < 0]
    if len(pos) < 15 or len(neg) < 15:
        return None, 99.0, 0.0
    rp, rn = float(np.percentile(pos, 92)), float(np.percentile(neg, 92))
    tp, tn = pos[pos >= np.percentile(pos, 80)], neg[neg >= np.percentile(neg, 80)]
    spread = float(max(np.median(np.abs(tp - np.median(tp))), np.median(np.abs(tn - np.median(tn)))))
    return (rp + rn) / 2.0, spread, (rp - rn) / 2.0


def across_view_dir(frame: RGBDFrame, axis_point: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Unit vector (world) along which silhouette_radii's sign is measured."""
    o = frame.T_world_cam[:3, 3]
    a = np.asarray(axis_dir, dtype=float) / np.linalg.norm(axis_dir)
    d = axis_point - o
    cr = np.cross(d, a)
    return cr / max(np.linalg.norm(cr), 1e-9)


def _project_world(frame: RGBDFrame, P: np.ndarray) -> np.ndarray:
    from components.transforms import invert

    return frame.model.project(apply(invert(frame.T_world_cam), np.atleast_2d(P)))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _prepare(label, frames, mask, setup_hash, table: TableFrame, dparams, box=None, mask_from_depth=False):
    est = ObjectPoseEstimate(
        label=label,
        mask=(np.asarray(mask) > 0).astype(np.uint8),
        calibration_id=setup_hash,
        frame_ids=[f.frame_id for f in frames],
    )
    est.extra["mask_source"] = "depth_roi" if mask_from_depth else "segmentation"
    seg = segmentation_problems(est.mask, None if mask_from_depth else box)
    if mask_from_depth:
        seg = [s for s in seg if s in ("mask_too_small", "truncated_at_image_border")]
    est.quality_flags.extend(seg)
    hard = [s for s in seg if s in ("mask_too_small", "truncated_at_image_border", "mask_is_a_box", "mask_fragmented")]
    if hard:
        reject(est, hard[0])
        return est, None
    P, px, q = masked_points(frames, est.mask, table.height, dparams)
    est.quality = q
    est.quality_flags.extend(q.flags)
    if P is None:
        reject(est, q.flags[-1] if q.flags else "no_object_depth")
        return est, None
    # A depth-derived mask coincides with depth edges by construction, so the
    # color/depth registration gate only applies to segmentation masks.
    if not mask_from_depth and (q.edge_alignment is None or q.edge_alignment < dparams.min_edge_alignment):
        est.quality_flags.append("depth_color_misregistered")
        reject(est, "depth_color_misregistered")
        return est, None
    est.points_world = P
    est.selected_px = px
    return est, P


def estimate_upright_source(
    label: str,
    frames: Sequence[RGBDFrame],
    mask: np.ndarray,
    *,
    setup_hash: str,
    table: TableFrame,
    calib_err: dict,
    max_graspable_mm: Optional[float] = None,
    dparams: Optional[DepthParams] = None,
    sparams: Optional[ShapeParams] = None,
    box: Optional[tuple] = None,
) -> ObjectPoseEstimate:
    """Upright bottle or can: axis from the top surface, height from the
    table plane, widest radius from the visible side."""
    dparams = dparams or DepthParams.for_label(label)
    sp = sparams or ShapeParams()
    est, P = _prepare(label, frames, mask, setup_hash, table, dparams, box)
    if P is None:
        return est
    if est.quality.background_fraction > dparams.max_background_fraction:
        return reject(est, "background_heavy_depth")
    abh = table.to_plane(P)
    h = abh[:, 2]
    H = float(np.percentile(h, 98))
    if H < sp.min_height_mm:
        return reject(est, "too_short_or_lying_down")
    if H > sp.max_height_mm:
        return reject(est, "implausible_height")
    slab = h >= H - sp.top_slab_mm
    if int(slab.sum()) < sp.min_top_points:
        return reject(est, "no_top_surface_depth")
    axis_ab = np.median(abh[slab, :2], axis=0)
    r_slab = np.linalg.norm(abh[slab, :2] - axis_ab, axis=1)
    # Erosion + edge exclusion shaved the top surface; add that back in mm.
    lost_px = dparams.erode_px + (2 if dparams.exclude_depth_edges else 0)
    z_top = float(np.median(apply(np.linalg.inv(frames[0].T_world_cam), P[slab])[:, 2]))
    r_top_raw = float(np.percentile(r_slab, 98))
    r_top = r_top_raw + lost_px * z_top / frames[0].model.fx
    axis_pt = table.to_world(np.array([axis_ab[0], axis_ab[1], 0.0]))
    r_body, rad_mad, e_across = limb_radius(silhouette_radii(frames[0], est.mask, axis_pt, table.n))
    if r_body is None:
        return reject(est, "no_silhouette_limbs")
    # The limbs locate the axis across the view better than the top surface
    # (which is biased toward the camera by visible neck sides).
    n_hat = across_view_dir(frames[0], axis_pt, table.n)
    shift = -e_across * n_hat
    axis_ab = axis_ab + np.array([shift @ table.e1, shift @ table.e2])
    D = 2.0 * r_body
    r_all = np.linalg.norm(abh[:, :2] - axis_ab, axis=1)
    if float(np.percentile(r_all, 99)) > r_body + 8.0:
        est.quality_flags.append("points_outside_silhouette_radius")
    est.dims = {"height": H, "diameter": D, "top_radius": r_top}
    est.extra.update({"radius_mad_mm": rad_mad, "top_points": int(slab.sum())})
    if D < sp.min_diameter_mm or D > sp.max_diameter_mm:
        return reject(est, "implausible_diameter")
    if max_graspable_mm is not None and D > max_graspable_mm:
        return reject(est, "too_wide_for_gripper")
    if H / D < sp.min_upright_ratio:
        return reject(est, "not_upright")
    if rad_mad > sp.max_radius_mad_mm:
        return reject(est, "poor_cylinder_fit")
    if r_top_raw > r_body + 3.0:
        return reject(est, "top_wider_than_body")
    est.dims["top_radius"] = min(r_top, r_body)

    base = table.to_world(np.array([axis_ab[0], axis_ab[1], 0.0]))
    mouth = table.to_world(np.array([axis_ab[0], axis_ab[1], H]))
    est.base, est.position, est.axis = base, mouth, table.n.copy()
    slab_mad = float(np.median(np.abs(abh[slab, :2] - axis_ab)))
    fit_xy = 1.4826 * slab_mad / math.sqrt(max(1, int(slab.sum())))
    est.error_bounds_mm = {
        "xy": math.hypot(float(calib_err.get("p95_xy_mm", 0.0)), fit_xy + 1.0),
        "z": float(calib_err.get("p95_z_mm", 0.0)) + est.quality.mad_mm,
    }
    f0 = frames[0]
    mp = _project_world(f0, mouth)[0]
    bp = _project_world(f0, base)[0]
    est.key_px = {"mouth": [float(mp[0]), float(mp[1])], "base": [float(bp[0]), float(bp[1])]}
    est.center_px = (float(mp[0]), float(mp[1]))
    return est


def estimate_cup(
    frames: Sequence[RGBDFrame],
    mask: np.ndarray,
    *,
    setup_hash: str,
    table: TableFrame,
    calib_err: dict,
    dparams: Optional[DepthParams] = None,
    sparams: Optional[ShapeParams] = None,
    box: Optional[tuple] = None,
    mask_from_depth: bool = False,
    require_interior: bool = True,
) -> ObjectPoseEstimate:
    """Cup opening: rim circle + plane from the highest ring of points, with an
    interior that must sit well below the rim (so a can top is not a cup).
    ``require_interior=False`` only for re-observing a cup whose opening was
    already verified from above (an oblique view cannot see the bottom)."""
    dparams = dparams or DepthParams.for_label(CUP_LABEL)
    sp = sparams or ShapeParams()
    est, P = _prepare(CUP_LABEL, frames, mask, setup_hash, table, dparams, box, mask_from_depth)
    if P is None:
        return est
    abh = table.to_plane(P)
    h = abh[:, 2]
    h_rim = float(np.percentile(h, 98))
    if h_rim < sp.min_cup_height_mm:
        return reject(est, "too_short_for_cup")
    band = h >= h_rim - sp.rim_band_mm
    if int(band.sum()) < sp.min_rim_points:
        return reject(est, "no_rim_points")
    c_ab, R, inl = fit_circle(abh[band, :2])
    rim_ab = abh[band][inl]
    res = np.linalg.norm(rim_ab[:, :2] - c_ab, axis=1) - R
    rms = float(np.sqrt(np.mean(res ** 2)))
    cov = arc_coverage(rim_ab[:, :2], c_ab)
    n_plane, d_plane, plane_rms = fit_plane(table.to_world(rim_ab))
    tilt = math.degrees(math.acos(float(np.clip(abs(n_plane @ table.n), -1.0, 1.0))))
    r_all = np.linalg.norm(abh[:, :2] - c_ab, axis=1)
    inner = r_all < R - 10.0
    interior_h = float(np.median(h[inner])) if inner.sum() else float("nan")
    est.dims = {"rim_radius": R, "height": h_rim}
    est.extra.update({
        "rim_residual_rms_mm": rms,
        "arc_coverage": cov,
        "rim_tilt_deg": tilt,
        "rim_plane_rms_mm": plane_rms,
        "rim_points": int(len(rim_ab)),
        "interior_points": int(inner.sum()),
        "interior_median_height_mm": interior_h,
    })
    if R < sp.min_rim_radius_mm or R > sp.max_rim_radius_mm:
        return reject(est, "implausible_rim_radius")
    if rms > sp.max_rim_residual_mm:
        return reject(est, "rim_fit_residual_high")
    if cov < sp.min_arc_coverage:
        return reject(est, "rim_occluded_or_partial")
    if tilt > sp.max_rim_tilt_deg:
        return reject(est, "rim_tilted")
    if require_interior:
        if inner.sum() < sp.min_interior_points:
            return reject(est, "interior_not_observed")
        if not (h_rim - interior_h >= sp.min_interior_drop_mm):
            return reject(est, "not_an_open_container")
    else:
        est.quality_flags.append("open_container_verified_by_prior_observation")

    rim_h = float(np.median(rim_ab[:, 2]))
    center = table.to_world(np.array([c_ab[0], c_ab[1], rim_h]))
    r_out, _, _ = limb_radius(silhouette_radii(frames[0], est.mask, table.to_world(np.array([c_ab[0], c_ab[1], 0.0])), table.n))
    est.dims["outer_radius"] = r_out if r_out is not None else R + 5.0
    est.position = center
    est.base = table.to_world(np.array([c_ab[0], c_ab[1], 0.0]))
    est.axis = n_plane if n_plane @ table.n > 0 else -n_plane
    est.dims["rim_height"] = rim_h
    est.error_bounds_mm = {
        "xy": math.hypot(float(calib_err.get("p95_xy_mm", 0.0)), rms / math.sqrt(max(1, len(rim_ab))) + 1.0),
        "z": float(calib_err.get("p95_z_mm", 0.0)) + plane_rms,
        "radius": rms + 1.0,
    }
    f0 = frames[0]
    ring = np.array([[c_ab[0] + R * math.cos(t), c_ab[1] + R * math.sin(t), rim_h]
                     for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
    ring_px = _project_world(f0, table.to_world(ring))
    cp = _project_world(f0, center)[0]
    est.center_px = (float(cp[0]), float(cp[1]))
    est.key_px = {"rim_center": [float(cp[0]), float(cp[1])], "rim": ring_px.round(1).tolist()}
    ys, xs = np.nonzero(est.mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    est.extra["bbox_center_px"] = [float((x0 + x1) / 2), float((y0 + y1) / 2)]
    return est


# ---------------------------------------------------------------------------
# Scene selection (exactly one source + one real cup)
# ---------------------------------------------------------------------------


@dataclass
class PairSelection:
    source: Optional[ObjectPoseEstimate]
    cup: Optional[ObjectPoseEstimate]
    reason: Optional[str]
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.reason is None


def select_pour_pair(
    estimates: Sequence[ObjectPoseEstimate],
    requested_source: str,
    *,
    source_px: Optional[tuple] = None,
    cup_px: Optional[tuple] = None,
) -> PairSelection:
    """``requested_source`` is "bottle", "can", or "any" (thirsty / unspecified).
    An explicit pixel pick (user selection) disambiguates; otherwise the scene
    must contain exactly one valid source of the requested kind and exactly one
    valid cup. A can is never treated as a cup."""
    if requested_source not in ("bottle", "can", "any"):
        return PairSelection(None, None, f"invalid_source_request:{requested_source}")
    want = SOURCE_LABELS if requested_source == "any" else (requested_source,)

    def pick(cands, px):
        if px is None:
            return cands
        u, v = int(round(px[0])), int(round(px[1]))
        return [c for c in cands if 0 <= v < c.mask.shape[0] and 0 <= u < c.mask.shape[1] and c.mask[v, u]]

    src_all = [e for e in estimates if e.label in want]
    cup_all = [e for e in estimates if e.label == CUP_LABEL]
    srcs = pick([e for e in src_all if e.ok], source_px)
    cups = pick([e for e in cup_all if e.ok], cup_px)
    detail = {
        "source_candidates": [e.summary() for e in src_all],
        "cup_candidates": [e.summary() for e in cup_all],
    }
    if not cups:
        reason = "no_cup" if not cup_all else "cup_rejected:" + ",".join(e.rejection_reason or "?" for e in cup_all)
        if not cup_all and any(e.label in SOURCE_LABELS for e in estimates) and len({e.label for e in estimates if e.label in SOURCE_LABELS}) > 1:
            reason = "bottle_and_can_without_cup"
        return PairSelection(None, None, reason, detail)
    if len(cups) > 1:
        return PairSelection(None, None, "multiple_cups", detail)
    if not srcs:
        reason = "no_source" if not src_all else "source_rejected:" + ",".join(e.rejection_reason or "?" for e in src_all)
        return PairSelection(None, cups[0], reason, detail)
    if len(srcs) > 1:
        return PairSelection(None, cups[0], "multiple_sources_need_selection", detail)
    return PairSelection(srcs[0], cups[0], None, detail)


def cup_mask_from_prior(
    frames: Sequence[RGBDFrame],
    prior: ObjectPoseEstimate,
    table: TableFrame,
    occluders,
    *,
    min_height_mm: float = 15.0,
    roi_pad_mm: float = 15.0,
    occluder_pad_px: int = 8,
) -> tuple[np.ndarray, dict]:
    """Re-observation mask without a detector call: project the prior cup's
    bounding cylinder into this view as an ROI, drop pixels covered by the held
    source / gripper (``occluders``: object with ``.C`` world centres and
    ``.R`` radii), and keep raised pixels (height > min_height_mm) connected
    to the projected rim centre."""
    from components.transforms import invert

    f0 = frames[0]
    H, W = f0.depth_mm.shape[:2]
    c = prior.position
    R_out = float(prior.dims.get("outer_radius", float(prior.dims["rim_radius"]) + 5.0)) + roi_pad_mm
    z_base = float(-(table.d + table.n[0] * c[0] + table.n[1] * c[1]) / table.n[2])
    ring = []
    for z in (z_base, float(c[2]) + 10.0):
        for t in np.linspace(0, 2 * math.pi, 36, endpoint=False):
            ring.append([c[0] + R_out * math.cos(t), c[1] + R_out * math.sin(t), z])
    T_cw = invert(f0.T_world_cam)
    ring_cam = apply(T_cw, np.array(ring))
    diag: dict = {}
    roi = np.zeros((H, W), np.uint8)
    if np.any(ring_cam[:, 2] <= 50):
        diag["roi_behind_camera"] = True
        return roi, diag
    px = f0.model.project(ring_cam).astype(np.int32)
    cv2.fillConvexPoly(roi, cv2.convexHull(px), 1)
    occ = np.zeros((H, W), np.uint8)
    C = np.asarray(occluders.C, dtype=float)
    Rr = np.asarray(occluders.R, dtype=float)
    Cc = apply(T_cw, C)
    front = Cc[:, 2] > 20.0
    if front.any():
        pp = f0.model.project(Cc[front])
        rad = f0.model.fx * Rr[front] / Cc[front, 2] + occluder_pad_px
        for (u, v), r in zip(pp, rad):
            if -r <= u <= W + r and -r <= v <= H + r:
                cv2.circle(occ, (int(round(u)), int(round(v))), int(math.ceil(r)), 1, -1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        D = np.stack([np.asarray(f.depth_mm, np.float32) for f in frames])
        Dm = np.nan_to_num(np.nanmedian(np.where(D > 0, D, np.nan), axis=0), nan=0.0)
    ys, xs = np.nonzero((roi > 0) & (Dm > 0))
    high = np.zeros((H, W), np.uint8)
    if len(xs):
        P = apply(f0.T_world_cam, f0.model.deproject(xs.astype(float), ys.astype(float), Dm[ys, xs]))
        h = table.height(P)
        # 3D ROI, not just 2D: anything standing between the camera and the cup
        # projects into the image ROI but lies outside the cup's cylinder.
        rxy = np.linalg.norm(P[:, :2] - c[:2], axis=1)
        rim_h = float(table.height(c[None, :])[0])
        r_keep = float(prior.dims.get("outer_radius", float(prior.dims["rim_radius"]) + 5.0)) + 6.0
        # Inside the prior cup cylinder every height is cup (walls, rim, and the
        # interior bottom that proves it is an open container); just outside,
        # only raised points (the table ring is dropped).
        keep = (h < rim_h + 25.0) & (
            ((rxy <= r_keep) & (h > -10.0)) | ((rxy <= R_out) & (h > min_height_mm)))
        high[ys[keep], xs[keep]] = 1
    cand = (high > 0) & (occ == 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    cp = f0.model.project(apply(T_cw, c[None, :]))[0]
    u0, v0 = int(round(cp[0])), int(round(cp[1]))
    pick = 0
    if 0 <= v0 < H and 0 <= u0 < W and labels[v0, u0] > 0:
        pick = int(labels[v0, u0])
    elif n > 1:
        pick = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    mask = (labels == pick).astype(np.uint8) if pick else np.zeros((H, W), np.uint8)
    roi_px = int(roi.sum())
    diag.update({
        "roi_px": roi_px,
        "occluded_roi_frac": float(((roi > 0) & (occ > 0)).sum()) / max(1, roi_px),
        "mask_px": int(mask.sum()),
    })
    return mask, diag
