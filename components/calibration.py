"""Calibrated setup for the pour (config/calibration/pour_setup.json).

The file is a hard prerequisite: every pour path (dry-run included) loads it
through ``check_setup`` and refuses to plan motion when anything is missing,
stale, hand-edited, or disagrees with the live hardware. Acceptance limits
live here in code; a file that carries different limits is rejected rather
than trusted, so limits cannot be loosened silently.

Nothing in here talks to the robot. The live comparison takes a
``LiveIdentity`` gathered by the caller (components/rgbd.py).
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from components.transforms import T_from_json, invert, rotation_angle_deg

ROOT = Path(__file__).resolve().parent.parent
SETUP_PATH = ROOT / "config" / "calibration" / "pour_setup.json"
TRIALS_PATH = ROOT / "config" / "calibration" / "pour_trials.jsonl"

SCHEMA_VERSION = 1

# Localization acceptance (initial). Report failures; never loosen here
# without a measured justification recorded in the README.
ACCEPTANCE = {"p95_xy_mm": 10.0, "max_xy_mm": 15.0, "p95_z_mm": 10.0}
MIN_HAND_EYE_SAMPLES = 12
MIN_TOUCHOFF_POINTS = 9
# Live-vs-calibrated tolerances.
INTRINSICS_TOL_PX = 0.5
FRAME_TOL_MM = 1.0
FRAME_TOL_DEG = 0.3
DEFAULT_MAX_AGE_DAYS = 7.0

MOUNT_TYPES = ("eye_in_hand", "eye_to_hand")


class CalibrationError(RuntimeError):
    """The setup is missing, stale, or does not match the live hardware."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("calibration not usable: " + "; ".join(self.problems))


@dataclass
class LiveIdentity:
    """What the running machine reports right now (see rgbd.read_live_identity)."""

    color_size: tuple[int, int]            # (width, height) of the decoded color image
    depth_size: tuple[int, int]            # (width, height) of the decoded depth image
    intrinsics: dict                       # fx, fy, cx, cy, width, height (GetProperties)
    distortion: dict                       # model, coeffs (GetProperties)
    reported_extrinsics: Optional[dict]    # GetProperties extrinsic_parameters, if any
    depth_encoding: str                    # mime type of the depth image
    T_flange_cam_viam: Optional[np.ndarray] = None  # from the Viam frame system
    cam_parent_is_arm: Optional[bool] = None


# ---------------------------------------------------------------------------
# Hash / IO
# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def compute_setup_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "setup_hash"}
    raw = json.dumps(_canonical(body), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def load_setup_doc(path: Path | str | None = None) -> dict:
    p = Path(path) if path else SETUP_PATH
    if not p.is_file():
        raise CalibrationError([f"calibration file missing: {p}"])
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationError([f"calibration file is not valid JSON: {exc}"]) from exc


def write_setup_doc(doc: dict, path: Path | str | None = None) -> str:
    p = Path(path) if path else SETUP_PATH
    doc = copy.deepcopy(doc)
    doc["setup_hash"] = compute_setup_hash(doc)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return doc["setup_hash"]


# ---------------------------------------------------------------------------
# Typed view
# ---------------------------------------------------------------------------


@dataclass
class GripperGeometry:
    closing_axis_tcp: np.ndarray   # unit vector, fingers close along this axis (TCP frame)
    pad_length_mm: float           # along the approach axis
    pad_width_mm: float            # across (along the object's axis when side-grasping)
    finger_reach_mm: float         # palm face -> pad centre, along approach
    max_open_mm: float
    jaw_mm_per_pos: float          # jaw opening = jaw_mm_at_pos0 + pos * jaw_mm_per_pos
    jaw_mm_at_pos0: float
    wrist_len_mm: float            # capsule behind the flange along -approach
    wrist_radius_mm: float
    body_radius_mm: float          # gripper body capsule radius (flange -> palm)
    finger_radius_mm: float
    camera_radius_mm: float

    def jaw_width_mm(self, pos: float) -> float:
        return self.jaw_mm_at_pos0 + pos * self.jaw_mm_per_pos


@dataclass
class PourSetup:
    doc: dict
    setup_hash: str
    mount: str
    T_flange_cam: Optional[np.ndarray]
    T_world_cam_fixed: Optional[np.ndarray]
    T_flange_tcp: np.ndarray
    table_normal: np.ndarray
    table_d: float
    intrinsics: dict
    distortion: dict
    depth_mm_per_count: float
    gripper: GripperGeometry
    validation: dict = field(default_factory=dict)

    @property
    def setup_id(self) -> str:
        return str(self.doc.get("setup_id") or self.setup_hash[:19])

    def table_height(self, pts: np.ndarray) -> np.ndarray:
        """Signed height above the table plane (mm)."""
        return np.asarray(pts, dtype=float) @ self.table_normal + self.table_d

    def T_world_cam(self, T_world_flange: Optional[np.ndarray]) -> np.ndarray:
        if self.mount == "eye_in_hand":
            if T_world_flange is None:
                raise CalibrationError(["eye-in-hand camera needs the flange pose at capture"])
            return T_world_flange @ self.T_flange_cam
        return self.T_world_cam_fixed


def _require(doc: dict, dotted: str, problems: list[str]) -> Any:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or cur.get(part) is None:
            problems.append(f"missing {dotted}")
            return None
        cur = cur[part]
    return cur


def parse_setup(doc: dict, *, now: Optional[dt.datetime] = None,
                allow_nominal: bool = False) -> tuple[Optional[PourSetup], list[str]]:
    """Validate the document on its own (no live hardware). Returns the typed
    setup (or None) and every problem found; callers treat any problem as
    fail-closed. ``allow_nominal`` accepts a run-time 'nominal' doc
    (components/pour_nominal.py, operator skipped calibration): structure is
    still checked, the calibration-evidence checks are not."""
    problems: list[str] = []
    nominal = allow_nominal and doc.get("status") == "nominal"
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if doc.get("status") != "calibrated" and not nominal:
        problems.append(f"status is {doc.get('status')!r}, not 'calibrated'")
    stored = doc.get("setup_hash")
    if not stored:
        problems.append("no setup_hash (only `calibrate_pour_setup.py finalize` writes a usable file)")
    elif stored != compute_setup_hash(doc):
        problems.append("setup_hash does not match file contents (edited after calibration?)")

    now = now or dt.datetime.now(dt.timezone.utc)
    when = doc.get("calibrated_at")
    if not when:
        problems.append("missing calibrated_at")
    else:
        try:
            t = dt.datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            age = (now - t).total_seconds() / 86400.0
            limit = float(doc.get("max_age_days") or DEFAULT_MAX_AGE_DAYS)
            if age > limit:
                problems.append(f"calibration is stale ({age:.1f} d > {limit:.1f} d)")
            if age < -0.01:
                problems.append("calibrated_at is in the future")
        except ValueError:
            problems.append(f"calibrated_at is not ISO-8601: {when!r}")

    mount = (doc.get("camera") or {}).get("mount")
    if mount not in MOUNT_TYPES:
        problems.append(f"camera.mount must be declared as one of {MOUNT_TYPES} (never inferred)")

    tr = doc.get("transforms") or {}

    def tf(name: str, needed: bool) -> Optional[np.ndarray]:
        raw = tr.get(name)
        if raw is None:
            if needed:
                problems.append(f"missing transforms.{name}")
            return None
        try:
            return T_from_json(raw)
        except ValueError as exc:
            problems.append(f"transforms.{name}: {exc}")
            return None

    T_fc = tf("T_flange_cam", mount == "eye_in_hand")
    T_wc = tf("T_world_cam", mount == "eye_to_hand")
    T_ft = tf("T_flange_tcp", True)

    intr = _require(doc, "camera.intrinsics", problems) or {}
    for k in ("fx", "fy", "cx", "cy", "width", "height"):
        if intr and k not in intr:
            problems.append(f"missing camera.intrinsics.{k}")
    dist = _require(doc, "camera.distortion", problems) or {}
    scale = _require(doc, "camera.depth_profile.units_mm_per_count", problems)
    align = (doc.get("camera") or {}).get("alignment") or {}
    if not align.get("verified") and not nominal:
        problems.append("depth-to-color alignment not verified (camera.alignment.verified)")

    table = doc.get("table") or {}
    n = table.get("normal")
    d = table.get("d")
    if n is None or d is None:
        problems.append("missing table.normal / table.d")
    else:
        n = np.asarray(n, dtype=float)
        if abs(np.linalg.norm(n) - 1.0) > 1e-3 or n[2] < 0.95:
            problems.append("table.normal must be a unit vector within ~18 deg of +z")

    g = doc.get("gripper") or {}
    gkeys = (
        "closing_axis_tcp", "pad_length_mm", "pad_width_mm", "finger_reach_mm",
        "max_open_mm", "jaw_mm_per_pos", "jaw_mm_at_pos0",
    )
    for k in gkeys:
        if g.get(k) is None:
            problems.append(f"missing gripper.{k} (measure it; no defaults are assumed)")
    coll = g.get("collision") or {}
    for k in ("wrist_len_mm", "wrist_radius_mm", "body_radius_mm", "finger_radius_mm", "camera_radius_mm"):
        if coll.get(k) is None:
            problems.append(f"missing gripper.collision.{k}")

    val = doc.get("validation") or {}
    acc = val.get("acceptance") or {}
    for k, v in ACCEPTANCE.items():
        if acc.get(k) != v:
            problems.append(f"validation.acceptance.{k}={acc.get(k)!r} differs from code limit {v}")
    cal = doc.get("calibration") or {}
    if mount == "eye_in_hand" and not nominal:
        if int(cal.get("sample_count") or 0) < MIN_HAND_EYE_SAMPLES:
            problems.append(f"hand-eye used {cal.get('sample_count')} samples (< {MIN_HAND_EYE_SAMPLES})")
        if not cal.get("held_out"):
            problems.append("hand-eye has no held-out validation")
    touch = val.get("touchoff") or {}
    if nominal:
        pass
    elif int(touch.get("n") or 0) < MIN_TOUCHOFF_POINTS:
        problems.append(f"touch-off validation has {touch.get('n')} points (< {MIN_TOUCHOFF_POINTS})")
    else:
        for key, lim in (("p95_xy_mm", "p95_xy_mm"), ("max_xy_mm", "max_xy_mm"), ("p95_z_mm", "p95_z_mm")):
            got = touch.get(key)
            if got is None or got > ACCEPTANCE[lim]:
                problems.append(f"touch-off {key}={got} exceeds {ACCEPTANCE[lim]} mm")

    if problems:
        return None, problems

    geom = GripperGeometry(
        closing_axis_tcp=np.asarray(g["closing_axis_tcp"], dtype=float) / np.linalg.norm(g["closing_axis_tcp"]),
        pad_length_mm=float(g["pad_length_mm"]),
        pad_width_mm=float(g["pad_width_mm"]),
        finger_reach_mm=float(g["finger_reach_mm"]),
        max_open_mm=float(g["max_open_mm"]),
        jaw_mm_per_pos=float(g["jaw_mm_per_pos"]),
        jaw_mm_at_pos0=float(g["jaw_mm_at_pos0"]),
        wrist_len_mm=float(coll["wrist_len_mm"]),
        wrist_radius_mm=float(coll["wrist_radius_mm"]),
        body_radius_mm=float(coll["body_radius_mm"]),
        finger_radius_mm=float(coll["finger_radius_mm"]),
        camera_radius_mm=float(coll["camera_radius_mm"]),
    )
    if abs(float(geom.closing_axis_tcp[2])) > 0.05:
        return None, ["gripper.closing_axis_tcp must be perpendicular to the approach (+z)"]
    setup = PourSetup(
        doc=doc,
        setup_hash=stored,
        mount=mount,
        T_flange_cam=T_fc,
        T_world_cam_fixed=T_wc,
        T_flange_tcp=T_ft,
        table_normal=np.asarray(table["normal"], dtype=float),
        table_d=float(table["d"]),
        intrinsics={k: float(intr[k]) for k in ("fx", "fy", "cx", "cy", "width", "height")},
        distortion={"model": str(dist.get("model", "")), "coeffs": [float(c) for c in dist.get("coeffs") or []]},
        depth_mm_per_count=float(scale),
        gripper=geom,
        validation=val,
    )
    return setup, []


def compare_live(setup: PourSetup, live: LiveIdentity) -> list[str]:
    """Fail-closed identity/profile comparison against the running machine."""
    problems: list[str] = []
    intr = setup.intrinsics
    w, h = int(intr["width"]), int(intr["height"])
    if tuple(live.color_size) != (w, h):
        problems.append(f"color image {live.color_size} != calibrated {(w, h)}")
    if tuple(live.depth_size) != (w, h):
        problems.append(
            f"depth image {live.depth_size} != color {(w, h)}: depth is not aligned to color"
        )
    li = live.intrinsics or {}
    for k in ("fx", "fy", "cx", "cy"):
        if k not in li or abs(float(li[k]) - intr[k]) > INTRINSICS_TOL_PX:
            problems.append(f"live intrinsics {k}={li.get(k)} != calibrated {intr[k]:.3f}")
    for k in ("width", "height"):
        if k in li and int(li[k]) != int(intr[k]):
            problems.append(f"live intrinsics {k}={li[k]} != calibrated {int(intr[k])}")
    ld = live.distortion or {}
    if str(ld.get("model", "")) != setup.distortion["model"]:
        problems.append(f"distortion model {ld.get('model')!r} != calibrated {setup.distortion['model']!r}")
    lc = [float(c) for c in ld.get("coeffs") or []]
    cc = setup.distortion["coeffs"]
    if len(lc) != len(cc) or any(abs(a - b) > 1e-4 for a, b in zip(lc, cc)):
        problems.append("distortion coefficients differ from calibration")
    expected_ext = (setup.doc.get("camera") or {}).get("reported_extrinsics")
    if expected_ext is not None or live.reported_extrinsics is not None:
        a = np.asarray((expected_ext or {}).get("translation_mm") or [0, 0, 0], dtype=float)
        b = np.asarray((live.reported_extrinsics or {}).get("translation_mm") or [0, 0, 0], dtype=float)
        if np.linalg.norm(a - b) > 0.5:
            problems.append(
                f"camera reported extrinsics {b.tolist()} != calibrated {a.tolist()} "
                "(module version / reference frame changed)"
            )
    enc = (setup.doc.get("camera") or {}).get("depth_profile", {}).get("encoding")
    if enc and live.depth_encoding and live.depth_encoding != enc:
        problems.append(f"depth encoding {live.depth_encoding!r} != calibrated {enc!r}")
    if setup.mount == "eye_in_hand":
        if live.cam_parent_is_arm is False:
            problems.append("camera is calibrated eye-in-hand but the live frame is not parented to the arm")
        if live.T_flange_cam_viam is None:
            problems.append("could not read the camera frame from the Viam frame system")
        else:
            D = invert(setup.T_flange_cam) @ live.T_flange_cam_viam
            dmm = float(np.linalg.norm(D[:3, 3]))
            ddeg = rotation_angle_deg(np.eye(3), D[:3, :3])
            if dmm > FRAME_TOL_MM or ddeg > FRAME_TOL_DEG:
                problems.append(
                    f"Viam frame system cam frame differs from calibrated T_flange_cam by "
                    f"{dmm:.1f} mm / {ddeg:.2f} deg; apply viam_frame_config from the calibration "
                    "(the frame system is authoritative and must carry the calibrated mount)"
                )
    return problems


def check_setup(
    doc_or_path: dict | Path | str | None = None,
    *,
    live: Optional[LiveIdentity] = None,
    now: Optional[dt.datetime] = None,
    allow_nominal: bool = False,
) -> PourSetup:
    doc = doc_or_path if isinstance(doc_or_path, dict) else load_setup_doc(doc_or_path)
    setup, problems = parse_setup(doc, now=now, allow_nominal=allow_nominal)
    if setup is not None and live is not None:
        problems = compare_live(setup, live)
    if problems:
        raise CalibrationError(problems)
    return setup


# ---------------------------------------------------------------------------
# Staged hardware validation bookkeeping (config/calibration/pour_trials.jsonl)
# ---------------------------------------------------------------------------

STAGES = {
    "B": {"desc": "calibration validation (arm disabled / teach mode)", "min_trials": 1, "min_success": 1},
    "C": {"desc": "localization, 9 positions x 3 arrangements", "min_trials": 27, "min_success": 27},
    "D": {"desc": "hover >= 30 mm above predicted grasp + rim", "min_trials": 9, "min_success": 9},
    "E": {"desc": "empty-container side grasp", "min_trials": 10, "min_success": 9},
    "F": {"desc": "empty-container pour trajectory", "min_trials": 10, "min_success": 10},
    "G": {"desc": "water with tray, human on e-stop, mentor approval", "min_trials": 10, "min_success": 9},
}
STAGE_ORDER = ("B", "C", "D", "E", "F", "G")


def load_trials(path: Path | str | None = None) -> list[dict]:
    p = Path(path) if path else TRIALS_PATH
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_trial(record: dict, path: Path | str | None = None) -> None:
    p = Path(path) if path else TRIALS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def stage_status(stage: str, setup_hash: str, trials: list[dict]) -> dict:
    """Pass/fail for one stage from recorded trials bound to this setup_hash.
    Any collision/contact in a stage fails it outright (E and F require zero
    contact; G requires no collision)."""
    spec = STAGES[stage]
    mine = [t for t in trials if t.get("stage") == stage and t.get("setup_hash") == setup_hash]
    ok = [t for t in mine if t.get("success")]
    contact = [t for t in mine if t.get("collision") or t.get("contact")]
    extra_ok = True
    if stage == "G":
        extra_ok = all(t.get("mentor_approved") and t.get("estop_attended") and t.get("tray") for t in mine)
    passed = (
        len(mine) >= spec["min_trials"]
        and len(ok) >= spec["min_success"]
        and not contact
        and extra_ok
    )
    return {
        "stage": stage,
        "trials": len(mine),
        "successes": len(ok),
        "contacts": len(contact),
        "required": spec,
        "passed": bool(passed),
    }


def stages_passed(required: tuple[str, ...], setup_hash: str, trials: Optional[list[dict]] = None) -> list[str]:
    """Missing stages (empty list means all required stages passed)."""
    trials = load_trials() if trials is None else trials
    return [s for s in required if not stage_status(s, setup_hash, trials)["passed"]]


def depth_units_ok(scale: float) -> bool:
    return math.isfinite(scale) and 0.05 <= scale <= 10.0
