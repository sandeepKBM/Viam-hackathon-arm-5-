"""Pure geometry for the calibrated pour: envelopes, grasp, pour path, views.

Nothing here talks to the robot. Every pose is a TCP (gripper-pad centre)
pose in ``world``; the controller converts to flange commands once, with the
calibrated ``T_flange_tcp``. The tool frame is: +z = approach (out of the
flange), ``closing_axis_tcp`` = the direction the fingers close along.

Key fixes over the old standalone script:
- Side-grasp height is the TCP height above the *table plane*. The old code
  used MIN_Z (a flange floor for a downward tool, which already contains the
  ~170 mm tool length) + 1.5 in, so the pads were ~200 mm above the table.
- The flange stand-off comes from the calibrated TCP, not a guessed 90 mm.
- The bottle mouth, not the flange, is steered over the cup: the tilt is a
  rotation about a horizontal axis through the mouth, and the mouth height at
  every tilt angle comes from the collision constraints.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from components.calibration import PourSetup
from components.object_pose import ObjectPoseEstimate, TableFrame
from components.safety import in_workspace
from components.transforms import apply, axis_angle, interpolate_T, invert, make_T, rot_z

UP = np.array([0.0, 0.0, 1.0])


class PlanningError(RuntimeError):
    def __init__(self, reason: str, detail: Optional[dict] = None):
        self.reason = reason
        self.detail = detail or {}
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Parameters (bounded)
# ---------------------------------------------------------------------------

PARAM_BOUNDS = {
    "clearance_mm": (15.0, 100.0),
    "table_clearance_mm": (5.0, 50.0),
    "mouth_inset_mm": (5.0, 40.0),
    "mouth_height_min_mm": (20.0, 120.0),
    "max_tilt_deg": (10.0, 120.0),
    "tilt_step_deg": (1.0, 10.0),
    "max_step_mm": (2.0, 20.0),
    "max_step_deg": (1.0, 10.0),
    "hold_s": (0.0, 10.0),
    "step_dwell_s": (0.0, 2.0),
    "grasp_height_frac": (0.15, 0.6),
    "pregrasp_clear_mm": (15.0, 80.0),
    "transit_clear_mm": (30.0, 200.0),
    "approach_step_deg": (5.0, 45.0),
    "jaw_tolerance_mm": (3.0, 20.0),
    "shoulder_frac": (0.5, 0.9),
    "reobserve_max_tilt_deg": (0.0, 60.0),
    "reobserve_max_tilt_liquid_deg": (0.0, 35.0),
    "reobserve_tol_xy_mm": (2.0, 15.0),
    "reobserve_tol_r_mm": (2.0, 10.0),
    "reobserve_tol_z_mm": (2.0, 15.0),
    "max_perception_age_s": (5.0, 300.0),
    "tracking_tol_mm": (0.5, 10.0),
    "tracking_tol_deg": (0.2, 5.0),
    "reach_max_mm": (300.0, 820.0),
    "min_flange_radius_mm": (100.0, 300.0),
    "liquid_max_hold_s": (0.0, 3.0),
    "hover_height_mm": (30.0, 150.0),
}


@dataclass
class PourParams:
    clearance_mm: float = 15.0
    table_clearance_mm: float = 10.0
    mouth_inset_mm: float = 10.0
    mouth_height_min_mm: float = 35.0
    max_tilt_deg: float = 95.0
    tilt_step_deg: float = 5.0
    max_step_mm: float = 10.0
    max_step_deg: float = 5.0
    hold_s: float = 1.0
    step_dwell_s: float = 0.1
    grasp_height_frac: float = 0.4
    pregrasp_clear_mm: float = 25.0
    transit_clear_mm: float = 50.0
    approach_step_deg: float = 15.0
    jaw_tolerance_mm: float = 12.0
    shoulder_frac: float = 0.75
    reobserve_max_tilt_deg: float = 45.0
    reobserve_max_tilt_liquid_deg: float = 25.0
    reobserve_tol_xy_mm: float = 10.0
    reobserve_tol_r_mm: float = 6.0
    reobserve_tol_z_mm: float = 10.0
    max_perception_age_s: float = 120.0
    tracking_tol_mm: float = 3.0
    tracking_tol_deg: float = 1.5
    reach_max_mm: float = 780.0
    min_flange_radius_mm: float = 150.0
    liquid_max_hold_s: float = 3.0
    hover_height_mm: float = 30.0

    def validate(self) -> "PourParams":
        for name, (lo, hi) in PARAM_BOUNDS.items():
            v = float(getattr(self, name))
            if not (lo <= v <= hi) or not math.isfinite(v):
                raise ValueError(f"pour parameter {name}={v} outside [{lo}, {hi}]")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Obstacles and envelopes
# ---------------------------------------------------------------------------


@dataclass
class Cylinder:
    """Vertical cylinder obstacle (cups, other objects, conservative footprints)."""

    name: str
    center_xy: np.ndarray
    radius: float
    z_min: float
    z_max: float


@dataclass
class Box:
    name: str
    lo: np.ndarray
    hi: np.ndarray


@dataclass
class Scene:
    table: TableFrame
    cup: Optional[Cylinder] = None
    others: list = field(default_factory=list)       # Cylinders
    boxes: list = field(default_factory=list)        # static Viam obstacles (walls, ceiling)
    workspace: Optional[Sequence] = None             # polygon for TCP/flange XY (None = default)

    def cylinders(self) -> list:
        return ([self.cup] if self.cup is not None else []) + list(self.others)


def boxes_from_setup(setup: PourSetup) -> list:
    out = []
    for ob in (setup.doc.get("obstacles") or []):
        if ob.get("name") == "table":
            continue  # the calibrated table plane is authoritative for the table
        out.append(Box(ob["name"], np.asarray(ob["min"], dtype=float), np.asarray(ob["max"], dtype=float)))
    return out


def cylinder_for(est: ObjectPoseEstimate, table: TableFrame, pad_mm: float = 5.0) -> Optional[Cylinder]:
    """Bounding vertical cylinder of an estimate's own 3D points (any label)."""
    if est.position is not None and est.label == "cup":
        R = float(est.dims.get("outer_radius", float(est.dims.get("rim_radius", 40.0)) + 5.0)) + pad_mm
        top = float(est.position[2])
        return Cylinder("cup", est.position[:2].copy(), R, table_z_under(table, est.position[:2]), top)
    P = est.points_world
    if P is None or len(P) < 10:
        return None
    c = np.median(P[:, :2], axis=0)
    r = float(np.percentile(np.linalg.norm(P[:, :2] - c, axis=1), 99)) + pad_mm
    return Cylinder(est.label, c, r, table_z_under(table, c), float(np.percentile(P[:, 2], 99.5)))


def table_z_under(table: TableFrame, xy) -> float:
    """World z of the table plane at (x, y)."""
    n, d = table.n, table.d
    return float(-(d + n[0] * xy[0] + n[1] * xy[1]) / n[2])


@dataclass
class Spheres:
    """Vectorized sphere set: centres (N, 3), radii (N,), tags (N,)."""

    C: np.ndarray
    R: np.ndarray
    tags: np.ndarray

    @staticmethod
    def empty() -> "Spheres":
        return Spheres(np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=object))

    def __add__(self, other: "Spheres") -> "Spheres":
        return Spheres(np.vstack([self.C, other.C]), np.concatenate([self.R, other.R]),
                       np.concatenate([self.tags, other.tags]))

    def __len__(self) -> int:
        return len(self.R)

    def transformed(self, T: np.ndarray) -> "Spheres":
        return Spheres(apply(T, self.C), self.R, self.tags)

    def shifted(self, dz: float) -> "Spheres":
        return Spheres(self.C + np.array([0.0, 0.0, dz]), self.R, self.tags)


def _capsule(p0: np.ndarray, p1: np.ndarray, r: float, tag: str, step: float = 8.0) -> Spheres:
    L = float(np.linalg.norm(p1 - p0))
    n = max(1, int(math.ceil(L / step)))
    s = np.linspace(0.0, 1.0, n + 1)[:, None]
    C = p0[None, :] + (p1 - p0)[None, :] * s
    return Spheres(C, np.full(len(C), float(r)), np.full(len(C), tag, dtype=object))


def gripper_spheres(setup: PourSetup, T_world_tcp: np.ndarray, jaw_open_mm: float) -> Spheres:
    """Collision envelope of fingers, gripper body, wrist and camera (capsule
    chains; radii from the calibrated collision envelope)."""
    g = setup.gripper
    T_tcp_flange = invert(setup.T_flange_tcp)
    z = T_world_tcp[:3, 2]
    c_axis = T_world_tcp[:3, :3] @ g.closing_axis_tcp
    tcp = T_world_tcp[:3, 3]
    flange = apply(T_world_tcp @ T_tcp_flange, np.zeros(3))
    palm = tcp - z * g.finger_reach_mm
    tip = tcp + z * (g.pad_length_mm / 2.0)
    half = max(jaw_open_mm, 0.0) / 2.0 + g.finger_radius_mm
    out = Spheres.empty()
    for s in (-1.0, 1.0):
        out = out + _capsule(palm + s * half * c_axis, tip + s * half * c_axis, g.finger_radius_mm, "finger")
    out = out + _capsule(flange, palm, g.body_radius_mm, "gripper_body")
    out = out + _capsule(flange, flange - z * g.wrist_len_mm, g.wrist_radius_mm, "wrist")
    if setup.T_flange_cam is not None:
        cam = apply(T_world_tcp @ T_tcp_flange @ setup.T_flange_cam, np.zeros(3))
        out = out + Spheres(cam[None, :], np.array([g.camera_radius_mm]), np.array(["camera"], dtype=object))
    return out


@dataclass
class BottleModel:
    """Held source in its own frame B: origin at the mouth centre, +z along the
    body axis toward the mouth. Surface sampled as rings of small spheres (a
    flat base must not bulge below itself the way a capsule end would)."""

    height: float
    body_radius: float
    top_radius: float
    shoulder_frac: float = 0.75
    _local: Optional[Spheres] = None

    def local(self) -> Spheres:
        if self._local is None:
            pts = []
            H, rb, rt = self.height, self.body_radius, self.top_radius
            zs_body = np.arange(-H, -H * (1.0 - self.shoulder_frac) + 1e-9, 8.0)
            zs_neck = np.arange(-H * (1.0 - self.shoulder_frac), 1e-9, 8.0)
            ang = np.linspace(0, 2 * math.pi, 20, endpoint=False)
            for z in zs_body:
                pts += [(rb * math.cos(t), rb * math.sin(t), z) for t in ang]
            for z in list(zs_neck) + [0.0]:
                pts += [(rt * math.cos(t), rt * math.sin(t), z) for t in ang]
            for r in (0.0, rb / 2.0):
                pts += [(r * math.cos(t), r * math.sin(t), -H) for t in ang]
            for r in (0.0, rt / 2.0):
                pts += [(r * math.cos(t), r * math.sin(t), 0.0) for t in ang]
            C = np.array(pts, dtype=float)
            self._local = Spheres(C, np.full(len(C), 1.5), np.full(len(C), "bottle", dtype=object))
        return self._local

    def spheres(self, T_world_B: np.ndarray) -> Spheres:
        return self.local().transformed(T_world_B)


def clearance_report(sph: Spheres, scene: Scene, *, ignore: Sequence[str] = (),
                     table_skip_tags: Sequence[str] = ()) -> dict:
    """Minimum surface clearance of every sphere to every obstacle (mm)."""
    worst: dict = {"table": None, "cylinders": None, "boxes": None}
    who: dict = {}
    if len(sph) == 0:
        return {"min": worst, "who": who}
    C, R = sph.C, sph.R
    tmask = ~np.isin(sph.tags, list(table_skip_tags)) if table_skip_tags else np.ones(len(R), bool)
    if tmask.any():
        h = scene.table.height(C[tmask]) - R[tmask]
        i = int(np.argmin(h))
        worst["table"], who["table"] = round(float(h[i]), 2), str(sph.tags[tmask][i])
    best = math.inf
    for cyl in scene.cylinders():
        if cyl.name in ignore:
            continue
        dxy = np.linalg.norm(C[:, :2] - cyl.center_xy, axis=1) - cyl.radius
        above = C[:, 2] - cyl.z_max
        below = cyl.z_min - C[:, 2]
        inside = (dxy <= 0) & (above <= 0) & (below <= 0)
        outside = np.hypot(np.maximum(dxy, 0.0), np.maximum(np.maximum(above, below), 0.0))
        d = np.where(inside, np.maximum(np.maximum(dxy, above), below), outside) - R
        i = int(np.argmin(d))
        if d[i] < best:
            best = float(d[i])
            who["cylinders"] = f"{sph.tags[i]}->{cyl.name}"
    if math.isfinite(best):
        worst["cylinders"] = round(best, 2)
    best = math.inf
    for b in scene.boxes:
        q = np.maximum(np.maximum(b.lo - C, 0.0), C - b.hi)
        d = np.linalg.norm(q, axis=1) - R
        i = int(np.argmin(d))
        if d[i] < best:
            best = float(d[i])
            who["boxes"] = f"{sph.tags[i]}->{b.name}"
    if math.isfinite(best):
        worst["boxes"] = round(best, 2)
    return {"min": worst, "who": who}


def check_pose(setup: PourSetup, scene: Scene, params: PourParams, T_world_tcp: np.ndarray,
               jaw_open_mm: float, *, bottle: Optional[BottleModel] = None,
               T_tcp_B: Optional[np.ndarray] = None, ignore: Sequence[str] = (),
               bottle_on_table_ok: bool = False, label: str = "") -> dict:
    """Raise PlanningError if a TCP pose (with optional held bottle) is unsafe.
    ``bottle_on_table_ok`` is only for the vertical lift-off / set-down, where
    the held bottle is resting on the table by definition. Returns the
    clearance report."""
    T_world_flange = T_world_tcp @ invert(setup.T_flange_tcp)
    tcp, flange = T_world_tcp[:3, 3], T_world_flange[:3, 3]
    # The taught polygon bounds where the tool works (it was taught as TCP
    # positions); the flange/wrist sits behind the TCP in a side grasp and is
    # bounded by reach, a keep-out around the base, and the collision envelope.
    if not in_workspace(float(tcp[0]), float(tcp[1]), scene.workspace):
        raise PlanningError("outside_workspace", {"pose": label, "point": "tcp", "xy": tcp[:2].round(1).tolist()})
    if float(np.linalg.norm(flange)) > params.reach_max_mm:
        raise PlanningError("beyond_reach", {"pose": label, "flange_dist_mm": round(float(np.linalg.norm(flange)), 1)})
    if float(np.linalg.norm(flange[:2])) < params.min_flange_radius_mm:
        raise PlanningError("too_close_to_base", {"pose": label, "flange_xy": flange[:2].round(1).tolist()})
    sph = gripper_spheres(setup, T_world_tcp, jaw_open_mm)
    if bottle is not None and T_tcp_B is not None:
        sph = sph + bottle.spheres(T_world_tcp @ T_tcp_B)
    rep = clearance_report(sph, scene, ignore=ignore, table_skip_tags=("bottle",) if bottle_on_table_ok else ())
    m = rep["min"]
    if m["table"] is not None and m["table"] < params.table_clearance_mm:
        raise PlanningError("table_clearance", {"pose": label, **rep})
    for key in ("cylinders", "boxes"):
        if m[key] is not None and m[key] < params.clearance_mm:
            raise PlanningError(f"{key}_clearance", {"pose": label, **rep})
    return rep


# ---------------------------------------------------------------------------
# Grasp
# ---------------------------------------------------------------------------


def tool_rotation(approach: np.ndarray, body_up_sign: float, closing_axis_tcp: np.ndarray) -> np.ndarray:
    """R_world_tcp for a horizontal side grasp: tool +z -> approach, closing
    axis horizontal, the remaining tool axis along +/- world up (the held
    object's axis)."""
    a = approach / np.linalg.norm(approach)
    c = closing_axis_tcp / np.linalg.norm(closing_axis_tcp)
    zt = np.array([0.0, 0.0, 1.0])
    b = np.cross(zt, c)                        # (c, b, z) right-handed in TCP frame
    b_w = body_up_sign * UP
    c_w = np.cross(b_w, a)
    Mw = np.column_stack([c_w, b_w, a])
    Mt = np.column_stack([c, b, zt])
    return Mw @ Mt.T


@dataclass
class GraspPlan:
    approach: np.ndarray
    body_up_sign: float
    T_grasp: np.ndarray
    T_pregrasp: np.ndarray
    T_lift: np.ndarray
    T_tcp_B: np.ndarray
    bottle: BottleModel
    grasp_height_mm: float
    expected_jaw_mm: float
    shallow_offset_mm: float
    approach_path: list
    lift_path: list
    retreat_path: list
    transit_mouth_z: float
    min_clearance: dict
    rejected: dict

    def summary(self) -> dict:
        return {
            "approach": self.approach.round(4).tolist(),
            "body_up_sign": self.body_up_sign,
            "grasp_tcp_mm": self.T_grasp[:3, 3].round(2).tolist(),
            "pregrasp_tcp_mm": self.T_pregrasp[:3, 3].round(2).tolist(),
            "lift_tcp_mm": self.T_lift[:3, 3].round(2).tolist(),
            "grasp_height_above_table_mm": round(self.grasp_height_mm, 2),
            "expected_jaw_mm": round(self.expected_jaw_mm, 2),
            "shallow_offset_mm": round(self.shallow_offset_mm, 2),
            "transit_mouth_z_mm": round(self.transit_mouth_z, 2),
            "min_clearance": self.min_clearance,
            "rejected_candidates": self.rejected,
        }


def transit_mouth_height(scene: Scene, bottle: BottleModel, params: PourParams, *,
                         ignore: Sequence[str] = ()) -> float:
    """Mouth height at which an upright held bottle clears every known object
    top (any XY), the cup included."""
    tops = [c.z_max for c in scene.cylinders() if c.name not in ignore]
    base_min = max(tops) if tops else table_z_under(scene.table, np.zeros(2))
    return base_min + params.transit_clear_mm + bottle.height


def plan_side_grasp(
    setup: PourSetup,
    scene: Scene,
    source: ObjectPoseEstimate,
    params: PourParams,
) -> GraspPlan:
    g = setup.gripper
    H = float(source.dims["height"])
    D = float(source.dims["diameter"])
    r_body = D / 2.0
    r_top = float(source.dims.get("top_radius", r_body))
    bottle = BottleModel(H, r_body, min(r_top, r_body), params.shoulder_frac)
    if g.max_open_mm / 2.0 - g.finger_radius_mm < r_body + 3.0:
        raise PlanningError("too_wide_to_straddle", {"diameter_mm": D, "max_open_mm": g.max_open_mm})
    shallow = max(0.0, r_body + 5.0 - g.finger_reach_mm)
    if shallow > 0.35 * r_body:
        raise PlanningError("fingers_too_short_for_diameter", {"diameter_mm": D, "finger_reach_mm": g.finger_reach_mm})
    z_min = g.finger_radius_mm + g.pad_width_mm / 2.0 + params.table_clearance_mm
    heights = []
    for frac in (params.grasp_height_frac, 0.5, 0.55, 0.3):
        z = max(frac * H, z_min)
        if z <= 0.6 * H and all(abs(z - h) > 3.0 for h in heights):
            heights.append(z)
    if not heights:
        raise PlanningError("no_grasp_band", {"height_mm": H})
    mouth = source.position
    transit_z = transit_mouth_height(scene, bottle, params, ignore=(source.label,))
    rejected: dict = {}
    for z_g in heights:
        plan = _side_grasp_at(setup, scene, source, params, bottle, D, r_body, shallow, z_g, mouth,
                              transit_z, rejected)
        if plan is not None:
            return plan
    raise PlanningError("no_feasible_side_grasp", {"rejected": rejected, "heights_tried_mm": heights})


def _side_grasp_at(setup, scene, source, params, bottle, D, r_body, shallow, z_g, mouth, transit_z, rejected):
    g = setup.gripper
    grasp_pt = source.base + setup.table_normal * z_g
    best = None
    base_dir = grasp_pt[:2] / max(np.linalg.norm(grasp_pt[:2]), 1e-6)
    n_ang = int(round(360.0 / params.approach_step_deg))
    for k in range(n_ang):
        phi = 2 * math.pi * k / n_ang
        a = np.array([math.cos(phi), math.sin(phi), 0.0])
        for sign in (1.0, -1.0):
            R = tool_rotation(a, sign, g.closing_axis_tcp)
            T_grasp = make_T(R, grasp_pt - a * shallow)
            stand = r_body + params.pregrasp_clear_mm + g.pad_length_mm / 2.0 + g.finger_radius_mm
            T_pre = make_T(R, grasp_pt - a * stand)
            lift_dz = transit_z - float(mouth[2])
            T_lift = make_T(R, T_grasp[:3, 3] + UP * max(lift_dz, params.transit_clear_mm))
            R_B = np.column_stack([a, np.cross(UP, a), UP])
            T_world_B = make_T(R_B, mouth)
            T_tcp_B = invert(T_grasp) @ T_world_B
            try:
                reports = []
                approach = [T_pre] + interpolate_T(T_pre, T_grasp, params.max_step_mm, params.max_step_deg)
                for i, T in enumerate(approach):
                    reports.append(check_pose(setup, scene, params, T, g.max_open_mm, ignore=(source.label,),
                                              label=f"approach{i}"))
                lift = interpolate_T(T_grasp, T_lift, params.max_step_mm, params.max_step_deg)
                for i, T in enumerate(lift):
                    reports.append(check_pose(setup, scene, params, T, D, bottle=bottle, T_tcp_B=T_tcp_B,
                                              ignore=(source.label,), bottle_on_table_ok=True,
                                              label=f"lift{i}"))
                retreat = interpolate_T(T_grasp, T_pre, params.max_step_mm, params.max_step_deg)
            except PlanningError as exc:
                rejected[exc.reason] = rejected.get(exc.reason, 0) + 1
                continue
            cam_up = 0.0
            if setup.T_flange_cam is not None:
                cam = apply(T_grasp @ invert(setup.T_flange_tcp) @ setup.T_flange_cam, np.zeros(3))
                cam_up = float(cam[2] - T_grasp[2, 3])
            min_c = min(min(v for v in r["min"].values() if v is not None) for r in reports)
            score = 2.0 * float(a[:2] @ base_dir) + 0.01 * min_c + (0.5 if cam_up > 0 else 0.0)
            if best is None or score > best[0]:
                best = (score, a, sign, T_grasp, T_pre, T_lift, T_tcp_B, approach, lift, retreat, min_c)
    if best is None:
        return None
    _, a, sign, T_grasp, T_pre, T_lift, T_tcp_B, approach, lift, retreat, min_c = best
    return GraspPlan(
        approach=a, body_up_sign=sign, T_grasp=T_grasp, T_pregrasp=T_pre, T_lift=T_lift,
        T_tcp_B=T_tcp_B, bottle=bottle, grasp_height_mm=z_g, expected_jaw_mm=D,
        shallow_offset_mm=shallow, approach_path=approach, lift_path=lift, retreat_path=retreat,
        transit_mouth_z=transit_z, min_clearance={"min_mm": round(min_c, 2)}, rejected=rejected,
    )


# ---------------------------------------------------------------------------
# Pour
# ---------------------------------------------------------------------------


@dataclass
class PourPlan:
    pour_dir: np.ndarray                 # horizontal, source side -> cup centre
    tilt_axis: np.ndarray
    yaw_deg: float
    mouth_xy: np.ndarray
    tilt_deg: list
    mouth_z: list
    T_tcp: list                          # TCP poses per tilt waypoint (index 0 = pre-pour, upright)
    transit_path: list                   # lift pose -> pre-pour
    retreat_path: list                   # pre-pour -> transit
    return_path: list                    # transit -> above the original spot -> grasp pose
    mouth_positions: list
    lip_positions: list
    min_clearance_mm: float
    mouth_margin_mm: float
    notes: list

    def summary(self) -> dict:
        return {
            "pour_dir": self.pour_dir.round(4).tolist(),
            "tilt_axis": self.tilt_axis.round(4).tolist(),
            "yaw_deg": round(self.yaw_deg, 2),
            "mouth_xy_mm": self.mouth_xy.round(2).tolist(),
            "tilt_deg": [round(a, 2) for a in self.tilt_deg],
            "mouth_z_mm": [round(z, 2) for z in self.mouth_z],
            "tcp_mm": [T[:3, 3].round(2).tolist() for T in self.T_tcp],
            "lip_mm": [p.round(2).tolist() for p in self.lip_positions],
            "min_clearance_mm": round(self.min_clearance_mm, 2),
            "mouth_margin_inside_rim_mm": round(self.mouth_margin_mm, 2),
            "waypoint_counts": {
                "transit": len(self.transit_path), "tilt": len(self.T_tcp),
                "retreat": len(self.retreat_path), "return": len(self.return_path),
            },
            "notes": self.notes,
        }


def _B_pose(R_B0: np.ndarray, u: np.ndarray, alpha_deg: float, mouth: np.ndarray) -> np.ndarray:
    return make_T(axis_angle(u, math.radians(alpha_deg)) @ R_B0, mouth)


def _mouth_z_lower_bound(setup: PourSetup, scene: Scene, params: PourParams, grasp: GraspPlan,
                         R_B0: np.ndarray, u: np.ndarray, alpha: float, mouth_xy: np.ndarray,
                         ignore: Sequence[str]) -> float:
    """Lowest mouth height at tilt ``alpha`` that keeps the held bottle and the
    gripper clear of the table and of every cylinder obstacle."""
    T_B = _B_pose(R_B0, u, alpha, np.array([mouth_xy[0], mouth_xy[1], 0.0]))
    T_tcp = T_B @ invert(grasp.T_tcp_B)
    sph = grasp.bottle.spheres(T_B) + gripper_spheres(setup, T_tcp, grasp.expected_jaw_mm)
    C, R = sph.C, sph.R
    n = scene.table.n
    tz = -(scene.table.d + C[:, 0] * n[0] + C[:, 1] * n[1]) / n[2]
    lb = float(np.max(tz + params.table_clearance_mm + R - C[:, 2]))
    for cyl in scene.cylinders():
        if cyl.name in ignore:
            continue
        near = np.linalg.norm(C[:, :2] - cyl.center_xy, axis=1) <= cyl.radius + R + params.clearance_mm
        if near.any():
            lb = max(lb, float(np.max(cyl.z_max + params.clearance_mm + R[near] - C[near, 2])))
    return lb


def plan_pour(
    setup: PourSetup,
    scene: Scene,
    grasp: GraspPlan,
    cup: ObjectPoseEstimate,
    params: PourParams,
    *,
    source_label: str,
) -> PourPlan:
    if not cup.ok or cup.position is None:
        raise PlanningError("cup_not_localized")
    c = cup.position
    R_rim = float(cup.dims["rim_radius"])
    rim_z = float(c[2])
    r_top = grasp.bottle.top_radius
    start_xy = grasp.T_grasp[:3, 3][:2]
    d = c[:2] - start_xy
    if np.linalg.norm(d) < 1e-3:
        raise PlanningError("source_on_top_of_cup")
    d = np.array([d[0], d[1], 0.0]) / np.linalg.norm(d)
    reach = R_rim - params.mouth_inset_mm - r_top
    if reach < 0:
        raise PlanningError("cup_opening_too_small_for_source", {"rim_radius_mm": R_rim, "top_radius_mm": r_top})
    mouth_xy = c[:2] - d[:2] * reach
    u = np.cross(UP, d)
    a_grasp = grasp.approach
    ignore = (source_label,)
    best = None
    notes = []
    for a_pour in (u, -u):
        yaw = math.atan2(a_grasp[0] * a_pour[1] - a_grasp[1] * a_pour[0], float(a_grasp @ a_pour))
        R_B0 = rot_z(yaw) @ (grasp.T_grasp @ grasp.T_tcp_B)[:3, :3]
        try:
            fine = np.arange(0.0, params.max_tilt_deg + 1e-9, 1.0)
            lbs = [_mouth_z_lower_bound(setup, scene, params, grasp, R_B0, u, float(al), mouth_xy, ignore)
                   for al in fine]
            n_steps = int(math.ceil(params.max_tilt_deg / params.tilt_step_deg))
            alphas = [min(params.max_tilt_deg, i * params.tilt_step_deg) for i in range(n_steps + 1)]
            zs = []
            for i, al in enumerate(alphas):
                lo = alphas[i - 1] if i else 0.0
                window = [lb for f, lb in zip(fine, lbs) if lo - 1e-9 <= f <= al + 1e-9]
                zs.append(max(max(window), rim_z + params.mouth_height_min_mm))
            T_tcp = []
            mouths, lips = [], []
            for al, z in zip(alphas, zs):
                mouth = np.array([mouth_xy[0], mouth_xy[1], z])
                T_B = _B_pose(R_B0, u, al, mouth)
                T = T_B @ invert(grasp.T_tcp_B)
                check_pose(setup, scene, params, T, grasp.expected_jaw_mm, bottle=grasp.bottle,
                           T_tcp_B=grasp.T_tcp_B, ignore=ignore, label=f"tilt{al:.0f}")
                T_tcp.append(T)
                mouths.append(mouth)
                ar = math.radians(al)
                lips.append(mouth + r_top * (math.cos(ar) * d - math.sin(ar) * UP))
            # Transit: lift pose -> yaw in place -> above pre-pour -> descend.
            zT = max(grasp.transit_mouth_z, zs[0])
            T_lift = grasp.T_lift
            lift_B = T_lift @ grasp.T_tcp_B
            up_dz = zT - float(lift_B[2, 3])
            T_up = make_T(T_lift[:3, :3], T_lift[:3, 3] + UP * max(up_dz, 0.0))
            yawed_B = make_T(rot_z(yaw) @ (T_up @ grasp.T_tcp_B)[:3, :3], (T_up @ grasp.T_tcp_B)[:3, 3])
            T_yawed = yawed_B @ invert(grasp.T_tcp_B)
            above_B = make_T(yawed_B[:3, :3], np.array([mouth_xy[0], mouth_xy[1], zT]))
            T_above = above_B @ invert(grasp.T_tcp_B)
            transit = []
            prev = T_lift
            for tgt in (T_up, T_yawed, T_above, T_tcp[0]):
                seg = interpolate_T(prev, tgt, params.max_step_mm, params.max_step_deg) if not np.allclose(prev, tgt) else []
                transit += seg
                prev = tgt
            for i, T in enumerate(transit):
                check_pose(setup, scene, params, T, grasp.expected_jaw_mm, bottle=grasp.bottle,
                           T_tcp_B=grasp.T_tcp_B, ignore=ignore, label=f"transit{i}")
            retreat = interpolate_T(T_tcp[0], T_above, params.max_step_mm, params.max_step_deg)
            # Return: undo the yaw above the original spot, then descend to the grasp pose.
            ret = []
            prev = T_above
            T_home_above = make_T(T_up[:3, :3], T_up[:3, 3])
            for tgt in (T_yawed, T_home_above, grasp.T_lift):
                ret += interpolate_T(prev, tgt, params.max_step_mm, params.max_step_deg) if not np.allclose(prev, tgt) else []
                prev = tgt
            for i, T in enumerate(ret):
                check_pose(setup, scene, params, T, grasp.expected_jaw_mm, bottle=grasp.bottle,
                           T_tcp_B=grasp.T_tcp_B, ignore=ignore, label=f"return{i}")
            # Final set-down is the checked lift path reversed.
            ret += list(reversed(grasp.lift_path[:-1])) + [grasp.T_grasp]
        except PlanningError as exc:
            notes.append(f"yaw option rejected: {exc.reason}")
            continue
        clear = []
        for T in T_tcp + transit:
            rep = clearance_report(gripper_spheres(setup, T, grasp.expected_jaw_mm)
                                   + grasp.bottle.spheres(T @ grasp.T_tcp_B), scene, ignore=ignore)
            clear.append(min(v for v in rep["min"].values() if v is not None))
        margin = min(R_rim - (float(np.linalg.norm(m[:2] - c[:2])) + r_top) for m in mouths)
        lip_margin = min(R_rim - float(np.linalg.norm(p[:2] - c[:2])) for p in lips)
        cand = PourPlan(
            pour_dir=d, tilt_axis=u, yaw_deg=math.degrees(yaw), mouth_xy=mouth_xy,
            tilt_deg=alphas, mouth_z=zs, T_tcp=T_tcp, transit_path=transit, retreat_path=retreat,
            return_path=ret, mouth_positions=mouths, lip_positions=lips,
            min_clearance_mm=float(min(clear)), mouth_margin_mm=float(min(margin, lip_margin)),
            notes=list(notes),
        )
        if best is None or abs(cand.yaw_deg) < abs(best.yaw_deg):
            best = cand
    if best is None:
        raise PlanningError("no_feasible_pour_path", {"notes": notes})
    if best.mouth_margin_mm < params.mouth_inset_mm - 1e-6:
        raise PlanningError("mouth_leaves_cup_interior", {"margin_mm": best.mouth_margin_mm})
    return best


# ---------------------------------------------------------------------------
# Re-observation view (held bottle occludes part of the wrist camera image)
# ---------------------------------------------------------------------------


@dataclass
class ViewPlan:
    T_tcp: np.ndarray
    path_in: list
    path_out: list
    bottle_tilt_deg: float
    rim_px: np.ndarray
    occluder_px: np.ndarray
    min_margin_px: float


def plan_reobserve_view(
    setup: PourSetup,
    scene: Scene,
    grasp: GraspPlan,
    cup: ObjectPoseEstimate,
    params: PourParams,
    *,
    liquid: bool,
    camera_model,
    source_label: str,
) -> ViewPlan:
    """Pick a pose from which the wrist camera sees the whole prior rim ring
    outside the image region the rigidly held bottle/gripper cover. The bottle
    tilts by the camera pitch, so the pitch is bounded (tighter with liquid)."""
    max_tilt = params.reobserve_max_tilt_liquid_deg if liquid else params.reobserve_max_tilt_deg
    if setup.T_flange_cam is None:
        raise PlanningError("reobserve_needs_eye_in_hand")
    T_tcp_cam = invert(setup.T_flange_tcp) @ setup.T_flange_cam
    c = cup.position
    R_rim = float(cup.dims["rim_radius"])
    ring = np.array([[c[0] + R_rim * math.cos(t), c[1] + R_rim * math.sin(t), c[2]]
                     for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
    W, H = camera_model.width, camera_model.height
    occ = grasp.bottle.spheres(grasp.T_tcp_B) + gripper_spheres(setup, np.eye(4), grasp.expected_jaw_mm)
    T_cam_tcp = invert(T_tcp_cam)
    occ_cam = apply(T_cam_tcp, occ.C)
    occ_r = occ.R
    front = occ_cam[:, 2] > 20.0
    occ_px = camera_model.project(occ_cam[front]) if front.any() else np.zeros((0, 2))
    occ_rad_px = camera_model.fx * occ_r[front] / occ_cam[front, 2] if front.any() else np.zeros(0)
    closing_w = grasp.T_grasp[:3, :3] @ setup.gripper.closing_axis_tcp
    best = None
    for tilt in np.arange(0.0, max_tilt + 1e-9, 5.0):
        for yaw in np.arange(-180.0, 180.0, 30.0):
            for s in (1.0, -1.0):
                R = rot_z(math.radians(yaw)) @ axis_angle(closing_w, s * math.radians(tilt)) @ grasp.T_grasp[:3, :3]
                R_wc = R @ T_tcp_cam[:3, :3]
                cam_z = R_wc[:, 2]
                if cam_z[2] > -0.2:
                    continue  # camera must look down at the table
                for dist in np.linspace(280.0, 420.0, 4):
                    for (tu, tv) in ((0.5, 0.3), (0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.5), (0.7, 0.5)):
                        u_t, v_t = tu * W, tv * H
                        xn, yn = camera_model.normalized(np.array([u_t]), np.array([v_t]))
                        ray = R_wc @ np.array([float(xn[0]), float(yn[0]), 1.0])
                        ray /= np.linalg.norm(ray)
                        cam_pos = c - dist * ray
                        T_wc = make_T(R_wc, cam_pos)
                        T = T_wc @ invert(T_tcp_cam)
                        ring_cam = apply(invert(T_wc), ring)
                        if np.any(ring_cam[:, 2] < 100):
                            continue
                        rp = camera_model.project(ring_cam)
                        if (rp[:, 0].min() < 20 or rp[:, 1].min() < 20 or rp[:, 0].max() > W - 20
                                or rp[:, 1].max() > H - 20):
                            continue
                        if len(occ_px):
                            dd = np.linalg.norm(rp[:, None, :] - occ_px[None, :, :], axis=2) - occ_rad_px[None, :]
                            margin = float(dd.min())
                        else:
                            margin = float(min(W, H))
                        if margin < 15.0:
                            continue
                        try:
                            check_pose(setup, scene, params, T, grasp.expected_jaw_mm, bottle=grasp.bottle,
                                       T_tcp_B=grasp.T_tcp_B, ignore=(source_label,), label="reobserve")
                            path_in = interpolate_T(grasp.T_lift, T, params.max_step_mm, params.max_step_deg)
                            for i, P in enumerate(path_in):
                                check_pose(setup, scene, params, P, grasp.expected_jaw_mm, bottle=grasp.bottle,
                                           T_tcp_B=grasp.T_tcp_B, ignore=(source_label,), label=f"reobs_in{i}")
                        except PlanningError:
                            continue
                        # Steeper views fit the rim better; the tilt bound already
                        # caps how far the held bottle may lean.
                        score = min(margin, 60.0) + tilt
                        if best is None or score > best[0]:
                            path_out = interpolate_T(T, grasp.T_lift, params.max_step_mm, params.max_step_deg)
                            best = (score, ViewPlan(T, path_in, path_out, float(tilt), rp, occ_px, margin))
    if best is None:
        raise PlanningError("reobserve_view_unavailable", {"max_bottle_tilt_deg": max_tilt, "liquid": liquid})
    return best[1]


def hover_targets(setup: PourSetup, scene: Scene, source: ObjectPoseEstimate, cup: ObjectPoseEstimate,
                  params: PourParams, top_down_R: np.ndarray) -> list:
    """Stage D: TCP poses with a downward tool whose fingertips sit
    ``hover_height_mm`` above the bottle mouth and the cup rim centre."""
    out = []
    tip = setup.gripper.pad_length_mm / 2.0
    for name, est in (("source_mouth", source), ("cup_rim_center", cup)):
        p = est.position + UP * (params.hover_height_mm + tip)
        T = make_T(top_down_R, p)
        check_pose(setup, scene, params, T, setup.gripper.max_open_mm, label=f"hover_{name}")
        out.append((name, T, est.position.copy()))
    return out
