"""Calibrate the pour setup, step by step, into config/calibration/.

Works on a draft (pour_setup.draft.json) and only writes pour_setup.json on
``finalize`` when every field is measured and validation passes. Nothing is
inferred: the camera mount type must be declared, gripper geometry measured.
Live steps need VIAM_ALLOW_LIVE=1 (and move nothing unless stated: the
operator positions the arm in teach mode / by jogging).

  python scripts/calibrate_pour_setup.py status
  python scripts/calibrate_pour_setup.py preflight                   # live, read-only, no motion
  python scripts/calibrate_pour_setup.py init --mount eye_in_hand
  python scripts/calibrate_pour_setup.py camera                      # live: intrinsics/profile
  python scripts/calibrate_pour_setup.py alignment --tag-mm 40       # live: tag on a >=50 mm block
  python scripts/calibrate_pour_setup.py hand-eye-capture --n 16     # live: ChArUco, >=12 diverse poses
  python scripts/calibrate_pour_setup.py hand-eye-solve --samples config/calibration/samples/<stamp>
  python scripts/calibrate_pour_setup.py tcp-capture --n 5           # live: pivot on a fixed point
  python scripts/calibrate_pour_setup.py gripper --closing-axis y --pad-length-mm .. (measured)
  python scripts/calibrate_pour_setup.py jaw-capture --widths 20,40,60,80   # live: gauge blocks
  python scripts/calibrate_pour_setup.py touchoff --n 9 --tag-mm 30  # live, stage B
  python scripts/calibrate_pour_setup.py finalize
  python scripts/calibrate_pour_setup.py objects --trials 27         # live, stage C
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import boot  # noqa: F401
import cv2
import numpy as np

from components import calibration as calib
from components import pour_calibration as pc
from components.rgbd import CameraModel
from components.transforms import T_from_json, T_to_json, apply, pose_to_T

CAL_DIR = calib.SETUP_PATH.parent
DRAFT = CAL_DIR / "pour_setup.draft.json"
SAMPLES = CAL_DIR / "samples"


# ---------------------------------------------------------------------------
# draft helpers
# ---------------------------------------------------------------------------


def load_draft() -> dict:
    if not DRAFT.is_file():
        sys.exit(f"no draft at {DRAFT}; run `init --mount ...` first")
    return json.loads(DRAFT.read_text())


def save_draft(doc: dict) -> None:
    doc = copy.deepcopy(doc)
    doc["status"] = "draft"
    doc["setup_hash"] = None
    DRAFT.write_text(json.dumps(doc, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print(f"draft updated: {DRAFT}")


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


# ---------------------------------------------------------------------------
# live helpers
# ---------------------------------------------------------------------------


async def _connect():
    from components.connection import connect_machine

    return await connect_machine()


async def _camera_arm(machine, doc):
    from viam.components.arm import Arm
    from viam.components.camera import Camera

    m = doc.get("machine") or {}
    return Camera.from_robot(machine, m.get("camera_name", "cam")), Arm.from_robot(machine, m.get("arm_name", "arm"))


async def _stationary_flange(arm, tries: int = 3):
    """Flange pose with the arm verified stationary (joints unchanged)."""
    for _ in range(tries):
        j0 = list((await arm.get_joint_positions()).values)
        T = pose_to_T(await arm.get_end_position())
        await asyncio.sleep(0.3)
        j1 = list((await arm.get_joint_positions()).values)
        if max(abs(a - b) for a, b in zip(j0, j1)) < 0.05:
            return T, j1
    raise RuntimeError("arm is moving; hold still (teach mode released?) and retry")


async def _color(cam):
    from components.rgbd import _split

    images, _ = await cam.get_images()
    color, depth, _, _, denc = _split(images)
    return color, depth, denc


def _model_from_doc(doc) -> CameraModel:
    i = doc["camera"]["intrinsics"]
    d = doc["camera"]["distortion"] or {"model": "", "coeffs": []}
    return CameraModel(i["fx"], i["fy"], i["cx"], i["cy"], int(i["width"]), int(i["height"]),
                       str(d.get("model", "")), tuple(d.get("coeffs") or ()))


def _tag_pose(bgr, model: CameraModel, tag_mm: float, dictionary: str):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(pc.DICTS[dictionary]))
    corners, ids, _ = det.detectMarkers(bgr)
    if ids is None or len(ids) != 1:
        return None
    c = corners[0].reshape(4, 2).astype(np.float64)
    h = tag_mm / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    xn, yn = model.normalized(c[:, 0], c[:, 1])
    img = np.stack([xn * model.fx + model.cx, yn * model.fy + model.cy], axis=1)
    ok, rvec, tvec = cv2.solvePnP(obj, img, model.K(), np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    from components.transforms import make_T

    return {"T_cam_tag": make_T(R, tvec.reshape(3)), "corners_px": c}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


async def cmd_preflight(args) -> None:
    """Read-only live checks that need no calibration. Sends no motion
    commands and changes no config: arm pose, frame chain, camera profile,
    and a per-object depth/color registration measurement on real frames."""
    from viam.components.arm import Arm
    from viam.components.camera import Camera

    from components.constants import HOME_JOINTS
    from components.pour_evidence import dump_json, new_run_dir, save_candidates
    from components.rgbd import (
        camera_mount_from_frame_system,
        capture_rgbd_set,
        edge_alignment_score,
        reported_extrinsics,
        save_frames,
    )
    from components.transforms import T_to_pose, invert, rotation_angle_deg

    run = new_run_dir(tag="preflight")
    rep: dict = {"read_only": True}
    machine = await _connect()
    try:
        arm = Arm.from_robot(machine, "arm")
        joints = list((await arm.get_joint_positions()).values)
        T_end = pose_to_T(await arm.get_end_position())
        rep["arm"] = {"joints_deg": [round(j, 2) for j in joints],
                      "home_offset_deg": round(max(abs(a - b) for a, b in zip(joints, HOME_JOINTS)), 2),
                      "flange_pose": T_to_pose(T_end)}
        T_pc, parent = await camera_mount_from_frame_system(machine, "cam", "arm")
        rep["cam_frame"] = {"parent": parent, "pose_in_parent": None if T_pc is None else T_to_pose(T_pc)}
        cam = Camera.from_robot(machine, "cam")
        props = await cam.get_properties()
        m = CameraModel.from_props(props)
        rep["camera_properties"] = {"intrinsics": m.as_dict(), "reported_extrinsics": reported_extrinsics(props),
                                    "mime_types": list(getattr(props, "mime_types", []) or [])}
        frames, _ = await capture_rgbd_set(machine, camera_name="cam", arm_name="arm", n_frames=3,
                                           aligned_to_color=False)
        f0 = frames[0]
        D = invert(T_end) @ f0.T_world_flange
        rep["frame_system_vs_driver_flange"] = {"mm": round(float(np.linalg.norm(D[:3, 3])), 3),
                                                "deg": round(rotation_angle_deg(np.eye(3), D[:3, :3]), 4)}
        rep["frames"] = [{"color_hw": list(f.color.shape[:2]), "depth_hw": list(f.depth_mm.shape[:2]),
                          "depth_encoding": f.depth_encoding, "captured_at": f.captured_at,
                          "depth_valid_frac": round(float((f.depth_mm > 0).mean()), 3),
                          "depth_median_mm": float(np.median(f.depth_mm[f.depth_mm > 0])) if (f.depth_mm > 0).any() else None}
                         for f in frames]
        save_frames(run, frames, "obs")
        cands = []
        if not args.no_detect:
            from components.shapes import find_pick_objects

            live_path = run / "moondream_input.png"
            cv2.imwrite(str(live_path), f0.color)
            shapes = find_pick_objects(f0.color, live_path, objects=("bottle", "can", "cup"))
            cands = [{"label": s.color, "mask": s.mask, "box": tuple(int(v) for v in s.box), "score": s.score}
                     for s in shapes if s.mask is not None]
            save_candidates(run, cands)
        with np.errstate(invalid="ignore"):
            Dm = np.nanmedian(np.where(np.stack([f.depth_mm for f in frames]) > 0,
                                       np.stack([f.depth_mm for f in frames]), np.nan), axis=0)
        Dm = np.nan_to_num(Dm, nan=0.0)
        objs = []
        vis = f0.color.copy()
        for c in cands:
            mask = (np.asarray(c["mask"]) > 0).astype(np.uint8)
            base = edge_alignment_score(mask, Dm)
            best = (base or 0.0, 0, 0)
            for dy in range(-40, 41, 4):
                for dx in range(-60, 61, 4):
                    sh = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                    s = edge_alignment_score(sh, Dm)
                    if s is not None and s > best[0]:
                        best = (s, dx, dy)
            objs.append({"label": c["label"], "box": c["box"], "mask_px": int(mask.sum()),
                         "edge_alignment_as_is": None if base is None else round(base, 3),
                         "best_edge_alignment": round(best[0], 3), "best_shift_px": [best[1], best[2]],
                         "depth_valid_frac_in_mask": round(float((Dm[mask > 0] > 0).mean()), 3)})
            ys, xs = np.nonzero(mask)
            cv2.rectangle(vis, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 2)
            cv2.putText(vis, f"{c['label']} edge={base if base is None else round(base, 2)} best@{best[1]},{best[2]}",
                        (int(xs.min()), max(14, int(ys.min()) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        edges = cv2.dilate(__import__("components.rgbd", fromlist=["depth_edges"]).depth_edges(Dm).astype(np.uint8),
                           np.ones((3, 3), np.uint8)) > 0
        vis[edges] = (0, 0, 255)
        cv2.imwrite(str(run / "preflight_overlay.png"), vis)
        rep["objects"] = objs
    finally:
        await machine.close()
    dump_json(run / "preflight.json", rep)
    print(json.dumps(rep, indent=2, default=str))
    print(f"evidence: {run}")


def cmd_status(args) -> None:
    try:
        setup = calib.check_setup()
        print(f"pour_setup.json OK: {setup.setup_id} {setup.setup_hash}")
        h = setup.setup_hash
    except calib.CalibrationError as exc:
        print("pour_setup.json NOT usable (fail-closed):")
        for p in exc.problems:
            print(f"  - {p}")
        h = None
    trials = calib.load_trials()
    for st in calib.STAGE_ORDER:
        if h is None:
            print(f"  stage {st}: blocked (no valid calibration)")
            continue
        s = calib.stage_status(st, h, trials)
        print(f"  stage {st}: {'PASSED' if s['passed'] else 'not passed'} "
              f"({s['successes']}/{s['trials']} ok, {s['contacts']} contact; need {s['required']['min_success']}"
              f"/{s['required']['min_trials']}) - {calib.STAGES[st]['desc']}")


def cmd_init(args) -> None:
    if args.mount not in calib.MOUNT_TYPES:
        sys.exit(f"--mount must be one of {calib.MOUNT_TYPES}")
    doc = json.loads(calib.SETUP_PATH.read_text()) if calib.SETUP_PATH.is_file() else {}
    doc = copy.deepcopy(doc)
    doc.setdefault("camera", {})["mount"] = args.mount
    doc["camera"]["mount_declared_by"] = args.declared_by or "operator"
    save_draft(doc)


async def cmd_camera(args) -> None:
    doc = load_draft()
    machine = await _connect()
    try:
        cam, _ = await _camera_arm(machine, doc)
        props = await cam.get_properties()
        from components.rgbd import reported_extrinsics

        model = CameraModel.from_props(props)
        color, depth, denc = await _color(cam)
        c = doc["camera"]
        c["intrinsics"] = {"fx": model.fx, "fy": model.fy, "cx": model.cx, "cy": model.cy,
                           "width": model.width, "height": model.height}
        c["distortion"] = {"model": model.dist_model, "coeffs": list(model.coeffs)}
        c["reported_extrinsics"] = reported_extrinsics(props)
        c["color_profile"] = {**(c.get("color_profile") or {}), "width": int(color.shape[1]), "height": int(color.shape[0])}
        c["depth_profile"] = {**(c.get("depth_profile") or {}), "width": int(depth.shape[1]),
                              "height": int(depth.shape[0]), "encoding": denc}
        print(json.dumps({k: c[k] for k in ("intrinsics", "distortion", "reported_extrinsics", "color_profile", "depth_profile")}, indent=2))
        if depth.shape[:2] != color.shape[:2]:
            print("WARNING: depth and color sizes differ -> depth is not aligned to color")
        ext = (c["reported_extrinsics"] or {}).get("translation_mm") or [0, 0, 0]
        if np.linalg.norm(ext) > 0.5:
            print(f"WARNING: reported extrinsic translation {ext} mm: camera reference frame is not the color sensor")
        save_draft(doc)
    finally:
        await machine.close()


async def cmd_alignment(args) -> None:
    """Depth-to-color registration check: an ArUco tag on top of a raised
    block (>= 50 mm). Tag pose from the color image (PnP) predicts the depth
    of every tag pixel; aligned depth must match it. Misregistered depth
    samples the table beside the block instead."""
    doc = load_draft()
    model = _model_from_doc(doc)
    machine = await _connect()
    try:
        cam, arm = await _camera_arm(machine, doc)
        res = []
        for i in range(args.frames):
            color, depth, _ = await _color(cam)
            tp = _tag_pose(color, model, args.tag_mm, args.dict)
            if tp is None:
                sys.exit("need exactly one visible tag")
            mask = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(mask, tp["corners_px"].astype(np.int32), 1)
            mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
            ys, xs = np.nonzero(mask & (depth > 0))
            T = tp["T_cam_tag"]
            n, p0 = T[:3, 2], T[:3, 3]
            xn, yn = model.normalized(xs.astype(float), ys.astype(float))
            rays = np.stack([xn, yn, np.ones_like(xn)], axis=1)
            z_pred = (n @ p0) / (rays @ n)
            r = depth[ys, xs] * float(doc["camera"]["depth_profile"].get("units_mm_per_count", 1.0)) - z_pred
            res.append({"median_mm": float(np.median(r)), "mad_mm": float(np.median(np.abs(r - np.median(r)))),
                        "valid_frac": float(len(xs) / max(1, int(mask.sum())))})
            await asyncio.sleep(0.2)
        med = float(np.median([x["median_mm"] for x in res]))
        mad = float(np.median([x["mad_mm"] for x in res]))
        verified = abs(med) <= args.max_bias_mm and mad <= args.max_mad_mm
        doc["camera"]["alignment"] = {**(doc["camera"].get("alignment") or {}), "verified": bool(verified),
                                      "metrics": {"frames": res, "median_residual_mm": med, "mad_mm": mad,
                                                  "tag_mm": args.tag_mm, "limits": [args.max_bias_mm, args.max_mad_mm]},
                                      "checked_at": pc.now_iso()}
        print(f"depth vs color-PnP residual: median {med:.2f} mm, MAD {mad:.2f} mm -> {'VERIFIED' if verified else 'FAILED'}")
        if not verified:
            print("Set align_color_depth: true on the camera (see viam_frame_config fragment_mods) and re-run.")
        save_draft(doc)
    finally:
        await machine.close()


async def cmd_hand_eye_capture(args) -> None:
    doc = load_draft()
    if doc["camera"].get("mount") != "eye_in_hand":
        sys.exit("hand-eye capture is for a declared eye_in_hand mount")
    model = _model_from_doc(doc)
    board = pc.charuco_board(args.cols, args.rows, args.square_mm, args.marker_mm, args.dict)
    out = SAMPLES / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "board.json").write_text(json.dumps(vars(args), default=str, indent=2))
    machine = await _connect()
    try:
        cam, arm = await _camera_arm(machine, doc)
        i = 0
        while i < args.n:
            if ask(f"[{i + 1}/{args.n}] Put the arm in a NEW pose (tilt 15-40 deg, vary yaw), board in view, "
                   "then press Enter (q to stop): ").lower() == "q":
                break
            T, joints = await _stationary_flange(arm)
            color, _, _ = await _color(cam)
            det = pc.detect_board_pose(color, board, model)
            if det is None:
                print("  board not detected (need >= 10 corners, not collinear); move and retry")
                continue
            cv2.imwrite(str(out / f"sample_{i:02d}.png"), color)
            rec = {"T_base_flange": T_to_json(T), "joints_deg": joints, "T_cam_target": T_to_json(det["T_cam_board"]),
                   "reproj_rms_px": det["reproj_rms_px"], "reproj_max_px": det["reproj_max_px"],
                   "n_corners": det["n_corners"]}
            (out / f"sample_{i:02d}.json").write_text(json.dumps(rec, indent=2))
            print(f"  saved: {det['n_corners']} corners, reprojection {det['reproj_rms_px']:.2f} px rms")
            i += 1
        print(f"samples in {out}; next: hand-eye-solve --samples {out}")
    finally:
        await machine.close()


def cmd_hand_eye_solve(args) -> None:
    doc = load_draft()
    d = Path(args.samples)
    meta = json.loads((d / "board.json").read_text())
    board = pc.charuco_board(int(meta["cols"]), int(meta["rows"]), float(meta["square_mm"]),
                             float(meta["marker_mm"]), meta["dict"])
    samples = [json.loads(p.read_text()) for p in sorted(d.glob("sample_*.json"))]
    res = pc.solve_hand_eye_samples(samples, pc.board_points(board))
    X = res["T_flange_cam"]
    ho = res["held_out"]
    print(json.dumps({"method": res["method"], "held_out": ho, "train": res["train"], "candidates": res["candidates"],
                      "diagnostics": res["diagnostics"]}, indent=2, default=float))
    doc["transforms"]["T_flange_cam"] = T_to_json(X)
    doc["calibration"] = {
        "method": res["method"], "date": pc.now_iso(), "sample_count": len(samples),
        "samples_dir": str(d), "samples": res["per_sample"], "residuals": res["train"], "held_out": ho,
        "diagnostics": res["diagnostics"], "board": meta,
    }
    doc["table"] = pc.table_from_board(res["T_base_target"], args.board_thickness_mm)
    doc["viam_frame_config"] = pc.viam_frame_config(X, camera_name=doc["machine"].get("camera_name", "cam"))
    save_draft(doc)
    print("Apply viam_frame_config.fragment_mods to armfarm5 (the frame system must carry the calibrated mount),")
    print("then run touchoff. The runtime refuses to pour while the live cam frame differs by > 1 mm / 0.3 deg.")


async def cmd_tcp_capture(args) -> None:
    doc = load_draft()
    machine = await _connect()
    poses = []
    try:
        _, arm = await _camera_arm(machine, doc)
        print("Fix a sharp point on the table. In teach mode, touch the CENTRE between the closed finger pads")
        print("to it from different wrist orientations (>= 20 deg apart).")
        for i in range(args.n):
            if ask(f"[{i + 1}/{args.n}] touching the point? Enter to record: ").lower() == "q":
                break
            T, _ = await _stationary_flange(arm)
            poses.append(T)
    finally:
        await machine.close()
    res = pc.solve_tcp(poses)
    print(json.dumps({k: v for k, v in res.items() if k != "T_flange_tcp"}, indent=2, default=float))
    if res["rms_mm"] > args.max_rms_mm:
        sys.exit(f"pivot residual {res['rms_mm']:.2f} mm > {args.max_rms_mm} mm; redo with more careful touches")
    doc["transforms"]["T_flange_tcp"] = T_to_json(res["T_flange_tcp"])
    doc.setdefault("tcp_calibration", {}).update({k: v for k, v in res.items() if k != "T_flange_tcp"})
    doc["tcp_calibration"]["date"] = pc.now_iso()
    save_draft(doc)


def cmd_gripper(args) -> None:
    doc = load_draft()
    axis = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "-x": [-1.0, 0.0, 0.0], "-y": [0.0, -1.0, 0.0]}[args.closing_axis]
    g = doc.setdefault("gripper", {})
    g.update({"closing_axis_tcp": axis, "pad_length_mm": args.pad_length_mm, "pad_width_mm": args.pad_width_mm,
              "finger_reach_mm": args.finger_reach_mm, "max_open_mm": args.max_open_mm,
              "measured_at": pc.now_iso(), "measured_by": args.measured_by})
    g["collision"] = {"wrist_len_mm": args.wrist_len_mm, "wrist_radius_mm": args.wrist_radius_mm,
                      "body_radius_mm": args.body_radius_mm, "finger_radius_mm": args.finger_radius_mm,
                      "camera_radius_mm": args.camera_radius_mm}
    save_draft(doc)


async def cmd_jaw_capture(args) -> None:
    from components.gripper import GripperComponent

    doc = load_draft()
    widths = [float(w) for w in args.widths.split(",")]
    machine = await _connect()
    samples = []
    try:
        gr = GripperComponent(machine, doc["machine"].get("gripper_name", "gripper"))
        for w in widths:
            await gr.open_full()
            ask(f"Place the {w:.1f} mm gauge between the pads, Enter to close: ")
            g = await gr.grab()
            samples.append((w, float(g.pos)))
            print(f"  {w:.1f} mm -> pos {g.pos:.0f}")
        await gr.open_full()
    finally:
        await machine.close()
    fit = pc.fit_jaw(samples)
    print(json.dumps(fit, indent=2))
    if fit["max_residual_mm"] > 2.0:
        sys.exit("jaw mapping residual > 2 mm; check the gauges and repeat")
    doc["gripper"].update({"jaw_mm_per_pos": fit["jaw_mm_per_pos"], "jaw_mm_at_pos0": fit["jaw_mm_at_pos0"],
                           "jaw_samples": fit["samples"]})
    save_draft(doc)


async def cmd_touchoff(args) -> None:
    """Stage B: vision predicts a tag centre in world through the live Viam
    frame system; the operator touches it with the TCP in teach mode."""
    from components.rgbd import camera_mount_from_frame_system, frame_system_T

    doc = load_draft()
    model = _model_from_doc(doc)
    X = T_from_json(doc["transforms"]["T_flange_cam"])
    T_ft = T_from_json(doc["transforms"]["T_flange_tcp"])
    machine = await _connect()
    pairs = []
    try:
        cam, arm = await _camera_arm(machine, doc)
        name = doc["machine"].get("camera_name", "cam")
        T_live, parent = await camera_mount_from_frame_system(machine, name, doc["machine"].get("arm_name", "arm"))
        delta = pc.frame_delta(X, T_live) if T_live is not None else None
        if delta is None or delta["mm"] > calib.FRAME_TOL_MM or delta["deg"] > calib.FRAME_TOL_DEG:
            sys.exit(f"live cam frame (parent {parent}) differs from the calibrated one: {delta}. "
                     "Apply viam_frame_config first; the frame system is authoritative.")
        for i in range(args.n):
            if ask(f"[{i + 1}/{args.n}] Arm at the observation pose, tag at a new spot. Enter to observe: ").lower() == "q":
                break
            _, _ = await _stationary_flange(arm)
            T_wc = await frame_system_T(machine, name)
            color, _, _ = await _color(cam)
            tp = _tag_pose(color, model, args.tag_mm, args.dict)
            if tp is None:
                print("  tag not found; retry")
                continue
            pred = apply(T_wc @ tp["T_cam_tag"], np.zeros(3))
            print(f"  predicted tag centre (world) {pred.round(1).tolist()}")
            ask("  Teach mode: put the TCP (pad centre) on the tag centre, hold still, Enter: ")
            T, _ = await _stationary_flange(arm)
            touched = apply(T @ T_ft, np.zeros(3))
            err = pred - touched
            print(f"  touched {touched.round(1).tolist()}  error xy {np.linalg.norm(err[:2]):.1f} mm, z {abs(err[2]):.1f} mm")
            pairs.append((pred.tolist(), touched.tolist()))
            ask("  Move the arm back to the observation pose, Enter: ")
    finally:
        await machine.close()
    st = pc.touchoff_stats(pairs)
    print(json.dumps({k: v for k, v in st.items() if k != "errors"}, indent=2))
    doc["validation"]["touchoff"] = {**st, "pairs": pairs, "date": pc.now_iso(), "method": "aruco tag, teach-mode touch-off"}
    doc["validation"]["passed"] = st["passed"]
    save_draft(doc)
    if not st["passed"]:
        print(f"FAILED acceptance {calib.ACCEPTANCE}; report the measured errors, fix the calibration, re-run.")


def cmd_finalize(args) -> None:
    doc = load_draft()
    doc["status"] = "calibrated"
    doc["calibrated_at"] = pc.now_iso()
    doc["setup_hash"] = calib.compute_setup_hash(doc)
    _, problems = calib.parse_setup(doc)
    if problems:
        print("not finalized; still missing / failing:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    h = calib.write_setup_doc(doc)
    print(f"wrote {calib.SETUP_PATH} ({h})")
    tp = (doc["validation"] or {}).get("touchoff") or {}
    calib.append_trial({"stage": "B", "setup_hash": h, "success": bool(tp.get("passed")), "collision": False,
                        "t": pc.now_iso(), "metrics": {k: tp.get(k) for k in ("n", "p95_xy_mm", "max_xy_mm", "p95_z_mm")}})
    print("recorded stage B from the touch-off validation")


async def cmd_objects(args) -> None:
    """Stage C: localization accuracy on real objects vs teach-mode touch-off."""
    from components.pouring import PourController, PourMode, PourRequest, _live_ports

    setup = calib.check_setup()
    machine = await _connect()
    try:
        robot, perception = await _live_ports(machine)
        for i in range(args.trials):
            if ask(f"[{i + 1}/{args.trials}] Arrange bottle + cup (position/arrangement per the plan), arm at home, Enter: ").lower() == "q":
                break
            ctl = PourController(robot, perception)
            res = await ctl.run(PourRequest(source=args.source, mode=PourMode.PLAN))
            if ctl.selection is None or not ctl.selection.ok:
                print(f"  localization rejected: {res.reason}")
                calib.append_trial({"stage": "C", "setup_hash": setup.setup_hash, "success": False,
                                    "reason": res.reason, "run_dir": res.run_dir, "t": pc.now_iso()})
                continue
            errs = {}
            for name, est in (("source_mouth", ctl.selection.source), ("cup_rim_center", ctl.cup)):
                print(f"  predicted {name}: {est.position.round(1).tolist()}")
                ask(f"  Teach mode: put the TCP on the {name.replace('_', ' ')}, Enter: ")
                T = await robot.flange_pose()
                touched = apply(T @ setup.T_flange_tcp, np.zeros(3))
                e = est.position - touched
                errs[name] = {"xy_mm": float(np.linalg.norm(e[:2])), "z_mm": float(abs(e[2]))}
                print(f"  error xy {errs[name]['xy_mm']:.1f} mm z {errs[name]['z_mm']:.1f} mm")
                ask("  Back to the observation pose, Enter: ")
            ok = all(v["xy_mm"] <= calib.ACCEPTANCE["max_xy_mm"] and v["z_mm"] <= calib.ACCEPTANCE["p95_z_mm"]
                     for v in errs.values())
            calib.append_trial({"stage": "C", "setup_hash": setup.setup_hash, "success": ok, "errors": errs,
                                "run_dir": res.run_dir, "t": pc.now_iso()})
    finally:
        await machine.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("preflight", help="read-only live checks (no motion, no config change)")
    p.add_argument("--no-detect", action="store_true")
    p = sub.add_parser("init")
    p.add_argument("--mount", required=True, choices=calib.MOUNT_TYPES)
    p.add_argument("--declared-by", default="")
    sub.add_parser("camera")
    p = sub.add_parser("alignment")
    p.add_argument("--tag-mm", type=float, required=True)
    p.add_argument("--dict", default="4X4_50", choices=list(pc.DICTS))
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--max-bias-mm", type=float, default=3.0)
    p.add_argument("--max-mad-mm", type=float, default=3.0)
    p = sub.add_parser("hand-eye-capture")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--square-mm", type=float, required=True, help="measured on the printed board")
    p.add_argument("--marker-mm", type=float, required=True, help="measured on the printed board")
    p.add_argument("--dict", default="5X5_100", choices=list(pc.DICTS))
    p = sub.add_parser("hand-eye-solve")
    p.add_argument("--samples", required=True)
    p.add_argument("--board-thickness-mm", type=float, required=True)
    p = sub.add_parser("tcp-capture")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-rms-mm", type=float, default=1.0)
    p = sub.add_parser("gripper")
    p.add_argument("--closing-axis", required=True, choices=["x", "y", "-x", "-y"])
    for k in ("pad-length-mm", "pad-width-mm", "finger-reach-mm", "max-open-mm", "wrist-len-mm",
              "wrist-radius-mm", "body-radius-mm", "finger-radius-mm", "camera-radius-mm"):
        p.add_argument(f"--{k}", type=float, required=True)
    p.add_argument("--measured-by", default="")
    p = sub.add_parser("jaw-capture")
    p.add_argument("--widths", required=True)
    p = sub.add_parser("touchoff")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--tag-mm", type=float, required=True)
    p.add_argument("--dict", default="4X4_50", choices=list(pc.DICTS))
    sub.add_parser("finalize")
    p = sub.add_parser("objects")
    p.add_argument("--trials", type=int, default=27)
    p.add_argument("--source", default="bottle", choices=["bottle", "can"])
    args = ap.parse_args()
    fn = {
        "status": cmd_status, "preflight": cmd_preflight, "init": cmd_init, "camera": cmd_camera, "alignment": cmd_alignment,
        "hand-eye-capture": cmd_hand_eye_capture, "hand-eye-solve": cmd_hand_eye_solve,
        "tcp-capture": cmd_tcp_capture, "gripper": cmd_gripper, "jaw-capture": cmd_jaw_capture,
        "touchoff": cmd_touchoff, "finalize": cmd_finalize, "objects": cmd_objects,
    }[args.cmd]
    if asyncio.iscoroutinefunction(fn):
        asyncio.run(fn(args))
    else:
        fn(args)


if __name__ == "__main__":
    main()
