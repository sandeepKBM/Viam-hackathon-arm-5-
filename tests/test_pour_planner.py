import math
import unittest

import numpy as np

from components.calibration import check_setup
from components.object_pose import TableFrame, estimate_cup, estimate_upright_source
from components.pour_planner import (
    PlanningError,
    PourParams,
    Scene,
    boxes_from_setup,
    check_pose,
    clearance_report,
    cylinder_for,
    gripper_spheres,
    plan_pour,
    plan_reobserve_view,
    plan_side_grasp,
)
from components.transforms import apply, invert
from tests.synthetic import MODEL, TABLE_Z, Bottle, Cup, SynScene, frames_for, masks_from_ids, setup_doc

CAL = {"p95_xy_mm": 4.0, "p95_z_mm": 3.0}


def build(bxy=(300.0, -60.0), cxy=(180.0, -200.0), doc=None):
    setup = check_setup(doc or setup_doc())
    table = TableFrame(setup.table_normal, setup.table_d)
    frames, ids = frames_for(SynScene([Bottle(*bxy), Cup(*cxy)]))
    m = masks_from_ids(ids)
    b = estimate_upright_source("bottle", frames, m[1], setup_hash=setup.setup_hash, table=table, calib_err=CAL)
    c = estimate_cup(frames, m[3], setup_hash=setup.setup_hash, table=table, calib_err=CAL)
    scene = Scene(table=table, cup=cylinder_for(c, table), others=[], boxes=boxes_from_setup(setup))
    return setup, scene, b, c


class GraspTests(unittest.TestCase):
    def test_side_grasp_height_is_above_the_table_not_min_z(self):
        setup, scene, b, _ = build()
        g = plan_side_grasp(setup, scene, b, PourParams())
        pad_z = g.T_grasp[2, 3]
        self.assertGreater(pad_z, TABLE_Z + 0.2 * 180.0)
        self.assertLess(pad_z, TABLE_Z + 0.6 * 180.0)
        self.assertLess(abs(g.T_grasp[:3, 2] @ np.array([0, 0, 1.0])), 1e-9)   # horizontal approach
        self.assertAlmostEqual(np.linalg.norm(g.T_grasp[:2, 3] - b.base[:2]), g.shallow_offset_mm, delta=1e-6)

    def test_flange_standoff_comes_from_calibrated_tcp(self):
        setup, scene, b, _ = build()
        g = plan_side_grasp(setup, scene, b, PourParams())
        flange = (g.T_grasp @ invert(setup.T_flange_tcp))[:3, 3]
        standoff = np.linalg.norm(flange[:2] - b.base[:2])
        self.assertAlmostEqual(standoff, np.linalg.norm(setup.T_flange_tcp[:3, 3]) + g.shallow_offset_mm, delta=0.5)
        # The old guessed 90 mm stand-off would put the pad centre far past the
        # bottle axis and the palm inside the bottle.
        old_pad_past_axis = np.linalg.norm(setup.T_flange_tcp[:3, 3]) - 90.0
        self.assertGreater(old_pad_past_axis, b.dims["diameter"] / 2.0)

    def test_too_wide_source_is_refused(self):
        setup, scene, b, _ = build()
        b.dims = dict(b.dims, diameter=84.0)
        with self.assertRaises(PlanningError) as ctx:
            plan_side_grasp(setup, scene, b, PourParams())
        self.assertEqual(ctx.exception.reason, "too_wide_to_straddle")

    def test_no_nudging_outside_workspace(self):
        setup, scene, b, _ = build()
        b.base = b.base + np.array([900.0, 0.0, 0.0])
        b.position = b.position + np.array([900.0, 0.0, 0.0])
        with self.assertRaises(PlanningError):
            plan_side_grasp(setup, scene, b, PourParams())


class PourPathTests(unittest.TestCase):
    def setUp(self):
        self.setup, self.scene, self.b, self.c = build()
        self.params = PourParams()
        self.g = plan_side_grasp(self.setup, self.scene, self.b, self.params)
        self.p = plan_pour(self.setup, self.scene, self.g, self.c, self.params, source_label="bottle")

    def test_mouth_and_lip_stay_over_cup_interior_with_clearance(self):
        c = self.c.position
        R = self.c.dims["rim_radius"]
        r_top = self.g.bottle.top_radius
        for a, m, lip, T in zip(self.p.tilt_deg, self.p.mouth_positions, self.p.lip_positions, self.p.T_tcp):
            self.assertLessEqual(np.linalg.norm(m[:2] - c[:2]) + r_top, R - self.params.mouth_inset_mm + 1e-6)
            self.assertLessEqual(np.linalg.norm(lip[:2] - c[:2]), R - self.params.mouth_inset_mm + 1e-6)
            self.assertGreaterEqual(m[2], c[2] + self.params.mouth_height_min_mm - 1e-6)
            rep = clearance_report(gripper_spheres(self.setup, T, self.g.expected_jaw_mm)
                                   + self.g.bottle.spheres(T @ self.g.T_tcp_B), self.scene)
            self.assertGreaterEqual(rep["min"]["cylinders"], self.params.clearance_mm - 1e-6)
            self.assertGreaterEqual(rep["min"]["table"], self.params.table_clearance_mm - 1e-6)
        self.assertGreaterEqual(self.p.min_clearance_mm, self.params.clearance_mm - 1e-6)

    def test_mouth_follows_path_not_fixed_tcp(self):
        """The TCP moves while tilting so the mouth stays on its path (rotation
        about the mouth, not about the TCP)."""
        tcps = np.array([T[:3, 3] for T in self.p.T_tcp])
        self.assertGreater(np.linalg.norm(tcps[-1] - tcps[0]), 50.0)
        mouths = np.array(self.p.mouth_positions)
        self.assertLess(np.ptp(mouths[:, 0]), 1e-6)
        self.assertLess(np.ptp(mouths[:, 1]), 1e-6)
        for T, m in zip(self.p.T_tcp, self.p.mouth_positions):
            self.assertLess(np.linalg.norm((T @ self.g.T_tcp_B)[:3, 3] - m), 1e-6)

    def test_tilt_actually_tilts_the_bottle_toward_the_cup(self):
        axis0 = (self.p.T_tcp[0] @ self.g.T_tcp_B)[:3, 2]
        axisN = (self.p.T_tcp[-1] @ self.g.T_tcp_B)[:3, 2]
        self.assertAlmostEqual(math.degrees(math.acos(np.clip(axis0 @ axisN, -1, 1))), self.p.tilt_deg[-1], delta=0.5)
        self.assertGreater(axisN[:2] @ self.p.pour_dir[:2], 0.9)

    def test_every_commanded_pose_is_checked(self):
        for T in self.p.T_tcp + self.p.transit_path:
            check_pose(self.setup, self.scene, self.params, T, self.g.expected_jaw_mm,
                       bottle=self.g.bottle, T_tcp_B=self.g.T_tcp_B)

    def test_parameters_are_bounded(self):
        for bad in (dict(max_tilt_deg=170.0), dict(clearance_mm=5.0), dict(hold_s=60.0), dict(tilt_step_deg=30.0)):
            with self.subTest(bad), self.assertRaises(ValueError):
                PourParams(**bad).validate()

    def test_small_cup_opening_refused(self):
        self.c.dims = dict(self.c.dims, rim_radius=20.0)
        with self.assertRaises(PlanningError):
            plan_pour(self.setup, self.scene, self.g, self.c, self.params, source_label="bottle")

    def test_reobserve_view_bounds_bottle_tilt(self):
        v = plan_reobserve_view(self.setup, self.scene, self.g, self.c, self.params, liquid=False,
                                camera_model=MODEL, source_label="bottle")
        self.assertLessEqual(v.bottle_tilt_deg, self.params.reobserve_max_tilt_deg)
        self.assertGreaterEqual(v.min_margin_px, 15.0)
        p = PourParams(reobserve_max_tilt_liquid_deg=0.0)
        with self.assertRaises(PlanningError):
            plan_reobserve_view(self.setup, self.scene, self.g, self.c, p, liquid=True,
                                camera_model=MODEL, source_label="bottle")


if __name__ == "__main__":
    unittest.main()
