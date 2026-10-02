"""Evidence for every pour run: event log, commanded vs reported poses,
masks, estimates, overlays, and a plan you can inspect without the robot.

Layout of a run directory (``out/pour_runs/<stamp>/``, gitignored):
    events.jsonl        state transitions, timings, abort reason
    commands.jsonl      every commanded flange/TCP pose and the reported pose
    obs_XX_*.png/json   observation RGB-D frames + metadata (replayable)
    candidates.npz      detector masks/labels/boxes (replayable)
    estimates.json      ObjectPoseEstimate summaries (incl. rejections)
    overlay.png         masks, selected depth pixels, mouth/rim, planned paths
    plan.json           grasp + pour plan (TCP, flange, mouth, lip, clearances)
    plan_top.png / plan_side.png   2D views of the 3D plan
    summary.json        final PourResult
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from components.transforms import T_to_pose, apply, invert

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "out" / "pour_runs"


def new_run_dir(base: Optional[Path] = None, tag: str = "") -> Path:
    base = Path(base) if base else RUNS_DIR
    stamp = time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    d = base / stamp
    i = 1
    while d.exists():
        d = base / f"{stamp}-{i}"
        i += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


class PourLog:
    def __init__(self, run_dir: Optional[Path], clock=time.monotonic):
        self.run_dir = run_dir
        self.clock = clock
        self.t0 = clock()
        self.events: list[dict] = []
        self.state_times: dict = {}
        self._state_t: Optional[float] = None
        self._state: Optional[str] = None

    def _write(self, name: str, rec: dict) -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / name).open("a") as fh:
            fh.write(json.dumps(rec, default=_jsonable) + "\n")

    def event(self, kind: str, **data) -> dict:
        rec = {"t": round(self.clock() - self.t0, 4), "wall": time.time(), "event": kind, **data}
        self.events.append(rec)
        self._write("events.jsonl", rec)
        return rec

    def state(self, state: str, **data) -> None:
        now = self.clock()
        if self._state is not None and self._state_t is not None:
            self.state_times[self._state] = round(self.state_times.get(self._state, 0.0) + now - self._state_t, 4)
        self._state, self._state_t = state, now
        self.event("state", state=state, **data)

    def close_state(self) -> None:
        """Account the time spent in the final state (no new transition)."""
        now = self.clock()
        if self._state is not None and self._state_t is not None:
            self.state_times[self._state] = round(self.state_times.get(self._state, 0.0) + now - self._state_t, 4)
            self._state_t = now

    def command(self, tag: str, T_tcp: np.ndarray, T_flange_cmd: np.ndarray,
                T_flange_actual: Optional[np.ndarray], err_mm: Optional[float], err_deg: Optional[float]) -> None:
        self._write("commands.jsonl", {
            "t": round(self.clock() - self.t0, 4), "tag": tag,
            "tcp_cmd": T_to_pose(T_tcp), "flange_cmd": T_to_pose(T_flange_cmd),
            "flange_actual": None if T_flange_actual is None else T_to_pose(T_flange_actual),
            "tracking_err_mm": err_mm, "tracking_err_deg": err_deg,
        })


def save_candidates(run_dir: Path, candidates: Sequence[dict], prefix: str = "candidates") -> None:
    if not candidates:
        np.savez_compressed(run_dir / f"{prefix}.npz", labels=np.array([]), masks=np.zeros((0, 1, 1), np.uint8),
                            boxes=np.zeros((0, 4)), scores=np.zeros(0))
        return
    np.savez_compressed(
        run_dir / f"{prefix}.npz",
        labels=np.array([c["label"] for c in candidates]),
        masks=np.stack([np.asarray(c["mask"], np.uint8) for c in candidates]),
        boxes=np.array([c.get("box") or (0, 0, 0, 0) for c in candidates], dtype=float),
        scores=np.array([c.get("score") if c.get("score") is not None else np.nan for c in candidates], dtype=float),
    )


def load_candidates(run_dir: Path, prefix: str = "candidates") -> list[dict]:
    z = np.load(Path(run_dir) / f"{prefix}.npz", allow_pickle=False)
    out = []
    for i in range(len(z["labels"])):
        box = tuple(int(v) for v in z["boxes"][i])
        out.append({
            "label": str(z["labels"][i]),
            "mask": z["masks"][i],
            "box": box if any(box) else None,
            "score": None if np.isnan(z["scores"][i]) else float(z["scores"][i]),
        })
    return out


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------

_COLORS = {"bottle": (255, 160, 0), "can": (0, 200, 255), "cup": (255, 0, 200)}


def _proj(frame, P: np.ndarray) -> np.ndarray:
    return frame.model.project(apply(invert(frame.T_world_cam), np.atleast_2d(P)))


def draw_overlay(frame, estimates: Sequence, grasp=None, pour=None, setup=None, selection=None) -> np.ndarray:
    vis = frame.color.copy()
    layer = np.zeros_like(vis)
    for est in estimates:
        col = _COLORS.get(est.label, (200, 200, 200))
        layer[est.mask > 0] = col
    vis = cv2.addWeighted(vis, 1.0, layer, 0.35, 0)
    for est in estimates:
        col = _COLORS.get(est.label, (200, 200, 200))
        if est.selected_px is not None and len(est.selected_px):
            step = max(1, len(est.selected_px) // 400)
            for u, v in est.selected_px[::step]:
                vis[int(v), int(u)] = (0, 255, 0)
        ys, xs = np.nonzero(est.mask)
        if len(xs):
            x0, y0 = int(xs.min()), int(ys.min())
            text = f"{est.label}: " + ("OK" if est.ok else f"REJECT {est.rejection_reason}")
            cv2.putText(vis, text, (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if est.ok and est.label == "cup" and "rim" in est.key_px:
            ring = np.array(est.key_px["rim"], dtype=np.int32)
            cv2.polylines(vis, [ring], True, (255, 255, 255), 1, cv2.LINE_AA)
            cx, cy = (int(v) for v in est.key_px["rim_center"])
            cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
            bb = est.extra.get("bbox_center_px")
            if bb:
                cv2.drawMarker(vis, (int(bb[0]), int(bb[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
        if est.ok and est.label in ("bottle", "can"):
            mx, my = (int(v) for v in est.key_px["mouth"])
            bx, by = (int(v) for v in est.key_px["base"])
            cv2.line(vis, (bx, by), (mx, my), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, (mx, my), 4, (255, 255, 255), -1)
    if grasp is not None:
        pts = _proj(frame, np.array([T[:3, 3] for T in grasp.approach_path]))
        cv2.polylines(vis, [pts.astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
        g = _proj(frame, grasp.T_grasp[:3, 3])[0]
        cv2.circle(vis, (int(g[0]), int(g[1])), 5, (0, 255, 255), 2)
    if pour is not None:
        mp = _proj(frame, np.array(pour.mouth_positions))
        cv2.polylines(vis, [mp.astype(np.int32)], False, (0, 128, 255), 2, cv2.LINE_AA)
        lp = _proj(frame, np.array(pour.lip_positions))
        for p in lp:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
    if selection is not None and not selection.ok:
        cv2.putText(vis, f"NO POUR: {selection.reason}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def draw_plan_views(grasp, pour, cup, source, size: int = 600, span_mm: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
    """Top (XY) and side (horizontal-distance-along-pour vs Z) schematic."""
    top = np.full((size, size, 3), 255, np.uint8)
    side = np.full((size, size, 3), 255, np.uint8)
    c = cup.position if cup is not None and cup.position is not None else np.zeros(3)
    s = size / span_mm

    def to_top(p):
        return int(size / 2 + (p[0] - c[0]) * s), int(size / 2 - (p[1] - c[1]) * s)

    if cup is not None and cup.position is not None:
        R = float(cup.dims.get("rim_radius", 40))
        cv2.circle(top, to_top(c), int(R * s), (200, 0, 200), 2)
        cv2.circle(top, to_top(c), int(float(cup.dims.get("outer_radius", R)) * s), (230, 150, 230), 1)
    if source is not None and source.base is not None:
        cv2.circle(top, to_top(source.base), int(source.dims["diameter"] / 2 * s), (0, 140, 255), 2)
    if grasp is not None:
        for T in grasp.approach_path + grasp.lift_path:
            cv2.circle(top, to_top(T[:3, 3]), 2, (0, 180, 180), -1)
    if pour is not None:
        for T in pour.transit_path:
            cv2.circle(top, to_top(T[:3, 3]), 1, (160, 160, 160), -1)
        for T, m, lip in zip(pour.T_tcp, pour.mouth_positions, pour.lip_positions):
            cv2.circle(top, to_top(T[:3, 3]), 3, (0, 160, 0), -1)
            cv2.circle(top, to_top(m), 3, (0, 100, 255), -1)
            cv2.circle(top, to_top(lip), 2, (0, 0, 255), -1)
        d = pour.pour_dir

        def to_side(p):
            h = float((p[:2] - c[:2]) @ d[:2])
            return int(size / 2 + h * s), int(size - 40 - (p[2] - (c[2] - float(cup.dims.get("height", 90)))) * s)

        if cup is not None:
            R = float(cup.dims.get("rim_radius", 40))
            base_z = c[2] - float(cup.dims.get("height", 90))
            cv2.rectangle(side, to_side(np.array([c[0] - d[0] * R, c[1] - d[1] * R, base_z])),
                          to_side(np.array([c[0] + d[0] * R, c[1] + d[1] * R, c[2]])), (200, 0, 200), 2)
        for T, m, lip in zip(pour.T_tcp, pour.mouth_positions, pour.lip_positions):
            cv2.circle(side, to_side(T[:3, 3]), 3, (0, 160, 0), -1)
            cv2.circle(side, to_side(m), 3, (0, 100, 255), -1)
            cv2.circle(side, to_side(lip), 2, (0, 0, 255), -1)
    for img, title in ((top, "top view (XY): cup rim, source, TCP green, mouth orange, lip red"),
                       (side, "side view along pour direction: z vs distance")):
        cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return top, side


def plan_json(setup, grasp, pour, selection, params) -> dict:
    T_ft_inv = invert(setup.T_flange_tcp)

    def poses(Ts):
        return [{"tcp": T_to_pose(T), "flange": T_to_pose(T @ T_ft_inv)} for T in Ts]

    out = {
        "calibration_id": setup.setup_hash,
        "params": params.as_dict(),
        "selection": None if selection is None else {
            "ok": selection.ok, "reason": selection.reason,
            "source": None if selection.source is None else selection.source.summary(),
            "cup": None if selection.cup is None else selection.cup.summary(),
        },
        "tcp_definition": "gripper pad centre; +z = approach; flange = tcp @ inv(T_flange_tcp)",
    }
    if grasp is not None:
        out["grasp"] = {**grasp.summary(), "approach_poses": poses(grasp.approach_path),
                        "lift_poses": poses(grasp.lift_path)}
    if pour is not None:
        out["pour"] = {**pour.summary(), "tilt_poses": poses(pour.T_tcp),
                       "transit_poses": poses(pour.transit_path), "return_poses": poses(pour.return_path),
                       "mouth_mm": [m.round(2).tolist() for m in pour.mouth_positions]}
    return out
