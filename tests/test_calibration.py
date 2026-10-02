import copy
import datetime as dt
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from components import calibration as calib
from components import pour_calibration as pc
from components.rgbd import CameraModel
from components.transforms import apply, invert, make_T, ov_to_matrix, rot_x, rot_y, rot_z
from tests.fakes import live_identity
from tests.synthetic import T_FLANGE_CAM, setup_doc

REPO_SETUP = Path(__file__).resolve().parent.parent / "config" / "calibration" / "pour_setup.json"


class SetupValidationTests(unittest.TestCase):
    def test_committed_setup_fails_closed_until_calibrated(self):
        with self.assertRaises(calib.CalibrationError) as ctx:
            calib.check_setup(REPO_SETUP)
        joined = " ".join(ctx.exception.problems)
        self.assertIn("not 'calibrated'", joined)
        self.assertIn("camera.mount", joined)   # never inferred
        self.assertIn("T_flange_tcp", joined)

    def test_missing_file_fails_closed(self):
        with self.assertRaises(calib.CalibrationError):
            calib.check_setup(Path(tempfile.gettempdir()) / "does_not_exist_pour_setup.json")

    def test_valid_synthetic_setup_passes(self):
        setup = calib.check_setup(setup_doc(), live=live_identity())
        self.assertEqual(setup.mount, "eye_in_hand")

    def test_stale_calibration_is_rejected(self):
        old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat()
        with self.assertRaises(calib.CalibrationError) as ctx:
            calib.check_setup(setup_doc(calibrated_at=old))
        self.assertTrue(any("stale" in p for p in ctx.exception.problems))

    def test_hand_edit_breaks_hash(self):
        doc = setup_doc()
        doc["table"]["d"] = -20.0  # edited after calibration
        with self.assertRaises(calib.CalibrationError) as ctx:
            calib.check_setup(doc)
        self.assertTrue(any("setup_hash" in p for p in ctx.exception.problems))

    def test_loosened_acceptance_is_rejected(self):
        doc = setup_doc()
        doc["validation"]["acceptance"]["p95_xy_mm"] = 25.0
        doc["setup_hash"] = calib.compute_setup_hash(doc)
        with self.assertRaises(calib.CalibrationError) as ctx:
            calib.check_setup(doc)
        self.assertTrue(any("differs from code limit" in p for p in ctx.exception.problems))

    def test_touchoff_over_limit_is_rejected(self):
        doc = setup_doc()
        doc["validation"]["touchoff"]["max_xy_mm"] = 18.0
        doc["setup_hash"] = calib.compute_setup_hash(doc)
        with self.assertRaises(calib.CalibrationError):
            calib.check_setup(doc)

    def test_live_profile_changes_fail_closed(self):
        doc = setup_doc()
        cases = {
            "resolution": dict(color_size=(1280, 720)),
            "unaligned depth grid": dict(depth_size=(848, 480)),
            "intrinsics": dict(intrinsics={"fx": 460.0, "fy": 455.0, "cx": 320.0, "cy": 180.0, "width": 640, "height": 360}),
            "distortion": dict(distortion={"model": "brown_conrady", "coeffs": [0.1, 0, 0, 0, 0]}),
            "module frame origin": dict(reported_extrinsics={"translation_mm": [-14.7, 0, 0]}),
            "not on arm": dict(cam_parent_is_arm=False),
        }
        for name, change in cases.items():
            live = live_identity()
            for k, v in change.items():
                setattr(live, k, v)
            with self.subTest(name), self.assertRaises(calib.CalibrationError):
                calib.check_setup(doc, live=live)

    def test_viam_frame_system_must_carry_calibrated_mount(self):
        doc = setup_doc()
        live = live_identity()
        live.T_flange_cam_viam = make_T(T_FLANGE_CAM[:3, :3], T_FLANGE_CAM[:3, 3] + np.array([3.0, 0, 0]))
        with self.assertRaises(calib.CalibrationError) as ctx:
            calib.check_setup(doc, live=live)
        self.assertTrue(any("frame system" in p for p in ctx.exception.problems))

    def test_mount_type_is_never_inferred(self):
        doc = setup_doc()
        doc["camera"]["mount"] = None
        doc["setup_hash"] = calib.compute_setup_hash(doc)
        with self.assertRaises(calib.CalibrationError):
            calib.check_setup(doc)


class StageBookkeepingTests(unittest.TestCase):
    def test_stage_needs_enough_clean_trials_bound_to_hash(self):
        h = "sha256:abc"
        trials = [{"stage": "E", "setup_hash": h, "success": True} for _ in range(9)]
        trials.append({"stage": "E", "setup_hash": h, "success": False})
        self.assertTrue(calib.stage_status("E", h, trials)["passed"])
        self.assertFalse(calib.stage_status("E", "sha256:other", trials)["passed"])
        trials.append({"stage": "E", "setup_hash": h, "success": True, "contact": True})
        self.assertFalse(calib.stage_status("E", h, trials)["passed"])  # zero contact required
        g = [{"stage": "G", "setup_hash": h, "success": True, "mentor_approved": True, "estop_attended": True,
              "tray": True} for _ in range(10)]
        self.assertTrue(calib.stage_status("G", h, g)["passed"])
        g[0]["mentor_approved"] = False
        self.assertFalse(calib.stage_status("G", h, g)["passed"])


class CalibrationProcedureTests(unittest.TestCase):
    """Board rendered into synthetic wrist-camera views -> ChArUco -> PnP ->
    hand-eye with held-out validation, end to end."""

    def test_charuco_hand_eye_end_to_end(self):
        model = CameraModel(fx=910.0, fy=910.0, cx=640.0, cy=360.0, width=1280, height=720)
        board = pc.charuco_board(7, 5, 30.0, 22.0)
        scale = 4.0
        W, H = int(7 * 30 * scale), int(5 * 30 * scale)
        tex = board.generateImage((W, H), marginSize=0)
        rng = np.random.default_rng(1)
        X = T_FLANGE_CAM
        # OpenCV board frame: +z into the board, so it faces down onto the table.
        T_bt = make_T(rot_z(0.2) @ rot_x(math.pi), (300.0, -100.0, 2.0))
        center = apply(T_bt, np.array([105.0, 75.0, 0.0]))
        samples = []
        for i in range(16):
            R = rot_z(rng.uniform(-1.0, 1.0)) @ rot_x(math.pi + rng.uniform(-0.45, 0.45)) @ rot_y(rng.uniform(-0.45, 0.45))
            Rc = R @ X[:3, :3]
            p_c = center - rng.uniform(330, 450) * Rc[:, 2]
            F = make_T(R, p_c - R @ X[:3, 3])
            C = invert(F @ X) @ T_bt
            corners_tex = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
            img_pts = model.project(apply(C, np.c_[corners_tex / scale, np.zeros(4)]))
            Hm = cv2.getPerspectiveTransform(corners_tex.astype(np.float32), img_pts.astype(np.float32))
            img = cv2.cvtColor(cv2.warpPerspective(tex, Hm, (model.width, model.height), flags=cv2.INTER_AREA,
                                                   borderValue=255), cv2.COLOR_GRAY2BGR)
            img = np.clip(img + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
            det = pc.detect_board_pose(img, board, model)
            self.assertIsNotNone(det, f"board not detected in sample {i}")
            samples.append({"T_base_flange": F, "T_cam_target": det["T_cam_board"], "reproj_rms_px": det["reproj_rms_px"]})
        res = pc.solve_hand_eye_samples(samples, pc.board_points(board))
        d = pc.frame_delta(X, res["T_flange_cam"])
        self.assertLess(d["mm"], 1.0)
        self.assertLess(d["deg"], 0.3)
        self.assertEqual(res["held_out"]["n"], 4)
        self.assertLess(res["held_out"]["p95"], 2.0)
        self.assertEqual(len(res["per_sample"]), 16)
        table = pc.table_from_board(res["T_base_target"], 2.0)
        self.assertAlmostEqual(-table["d"], 0.0, delta=1.0)
        frame = pc.viam_frame_config(res["T_flange_cam"])
        self.assertEqual(frame["frame"]["parent"], "arm")
        self.assertIn("align_color_depth", json.dumps(frame["fragment_mods"]))

    def test_too_few_samples_rejected(self):
        with self.assertRaises(ValueError):
            pc.solve_hand_eye_samples([{"T_base_flange": np.eye(4), "T_cam_target": np.eye(4)}] * 5, np.zeros((4, 3)))

    def test_touchoff_acceptance(self):
        good = [((i, 0, 0), (i + 3.0, 0, 1.0)) for i in range(9)]
        self.assertTrue(pc.touchoff_stats(good)["passed"])
        bad = good[:-1] + [((0, 0, 0), (16.0, 0, 0))]
        self.assertFalse(pc.touchoff_stats(bad)["passed"])
        self.assertFalse(pc.touchoff_stats(good[:5])["passed"])

    def test_jaw_fit(self):
        fit = pc.fit_jaw([(20, 200), (40, 400), (60, 600), (80, 800)])
        self.assertAlmostEqual(fit["jaw_mm_per_pos"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
