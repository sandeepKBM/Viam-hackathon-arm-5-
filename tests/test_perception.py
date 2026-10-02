import dataclasses
import unittest

import numpy as np

from components.object_pose import (
    TableFrame,
    cup_mask_from_prior,
    estimate_cup,
    estimate_upright_source,
    select_pour_pair,
)
from components.rgbd import CameraModel, validate_frames
from components.transforms import apply, invert, make_T, rot_x, rot_y
from tests.synthetic import (
    HOME_FLANGE,
    MODEL,
    TABLE_Z,
    Bottle,
    Can,
    Cup,
    LyingCylinder,
    SynScene,
    camera_pose_for_flange,
    frames_for,
    masks_from_ids,
)

TABLE = TableFrame(np.array([0.0, 0.0, 1.0]), -TABLE_Z)
CAL = {"p95_xy_mm": 4.0, "p95_z_mm": 3.0}
BOTTLE_XY = np.array([300.0, -60.0])
CUP_XY = np.array([180.0, -200.0])


def scene(*extra):
    return SynScene([Bottle(*BOTTLE_XY), Cup(*CUP_XY)] + list(extra))


def bottle(frames, mask, label="bottle"):
    return estimate_upright_source(label, frames, mask, setup_hash="h", table=TABLE, calib_err=CAL)


def cup(frames, mask, **kw):
    return estimate_cup(frames, mask, setup_hash="h", table=TABLE, calib_err=CAL, **kw)


class FrameValidationTests(unittest.TestCase):
    def setUp(self):
        self.frames, self.ids = frames_for(scene())

    def test_aligned_frames_pass(self):
        self.assertEqual(validate_frames(self.frames, MODEL), [])

    def test_unaligned_flag_rejected(self):
        f = [dataclasses.replace(x, aligned_to_color=False) for x in self.frames]
        self.assertTrue(any("not aligned" in p for p in validate_frames(f, MODEL)))

    def test_wrong_resolution_rejected(self):
        f = [dataclasses.replace(x, depth_mm=x.depth_mm[:, :600]) for x in self.frames]
        self.assertTrue(validate_frames(f, MODEL))

    def test_intrinsics_for_other_resolution_rejected(self):
        other = dataclasses.replace(MODEL, width=1280, height=720)
        f = [dataclasses.replace(x, model=other) for x in self.frames]
        self.assertTrue(any("intrinsics are for" in p for p in validate_frames(f)))
        self.assertTrue(validate_frames(self.frames, other))

    def test_wrong_units_rejected(self):
        f = [dataclasses.replace(x, depth_units_mm=1000.0) for x in self.frames]
        self.assertTrue(any("units" in p for p in validate_frames(f, MODEL)))

    def test_moving_arm_rejected(self):
        f = list(self.frames)
        f[1] = dataclasses.replace(f[1], joints_after=[0.3] * 6)
        self.assertTrue(any("moved" in p for p in validate_frames(f, MODEL)))
        f = list(self.frames)
        f[2] = dataclasses.replace(f[2], T_world_cam=f[2].T_world_cam @ make_T(t=(2.0, 0, 0)))
        self.assertTrue(any("not stationary" in p for p in validate_frames(f, MODEL)))

    def test_unknown_distortion_model_rejected(self):
        m = dataclasses.replace(MODEL, dist_model="kannala_brandt", coeffs=(0.1,))
        f = [dataclasses.replace(x, model=m) for x in self.frames]
        self.assertTrue(validate_frames(f))

    def test_inverse_brown_conrady_round_trip(self):
        m = CameraModel(fx=910, fy=910, cx=640, cy=360, width=1280, height=720,
                        dist_model="inverse_brown_conrady", coeffs=(0.05, -0.02, 0.001, -0.001, 0.0))
        P = np.array([[50.0, -30.0, 500.0], [-200.0, 100.0, 450.0]])
        uv = m.project(P)
        back = m.deproject(uv[:, 0], uv[:, 1], P[:, 2])
        self.assertLess(np.abs(back - P).max(), 0.05)


class TransformDirectionTests(unittest.TestCase):
    def test_camera_to_world_direction(self):
        # Tilted view: a straight-down camera is ~a 180 deg rotation, which is
        # its own transpose, so T and T^-1 can nearly agree there by accident.
        frames, ids = frames_for(scene(), HOME_FLANGE @ make_T(rot_y(np.radians(-8.0))))
        f0 = frames[0]
        vs, us = np.nonzero((ids == 0) & (f0.depth_mm > 0))
        pick = np.linspace(0, len(us) - 1, 200).astype(int)
        p_cam = f0.model.deproject(us[pick].astype(float), vs[pick].astype(float), f0.depth_mm[vs[pick], us[pick]])
        right = apply(f0.T_world_cam, p_cam)
        wrong = apply(invert(f0.T_world_cam), p_cam)
        self.assertLess(np.abs(right[:, 2] - TABLE_Z).max(), 3.0)       # table pixels land on the table
        self.assertGreater(float(np.median(np.linalg.norm(wrong - right, axis=1))), 50.0)

    def test_wrist_camera_transform_changes_with_arm_pose(self):
        """Eye-in-hand: the same bottle seen from two arm poses localizes to the
        same place only with the camera pose of each capture; reusing the
        first pose for the second image is wrong by the arm motion."""
        sc = scene()
        pose_b = HOME_FLANGE @ make_T(rot_y(np.radians(-8.0)))
        fa, ia = frames_for(sc, HOME_FLANGE)
        fb, ib = frames_for(sc, pose_b)
        a = bottle(fa, masks_from_ids(ia)[1])
        b = bottle(fb, masks_from_ids(ib)[1])
        self.assertTrue(a.ok and b.ok)
        self.assertLess(np.linalg.norm(a.base[:2] - b.base[:2]), 6.0)
        stale = [dataclasses.replace(x, T_world_cam=fa[0].T_world_cam) for x in fb]
        s = bottle(stale, masks_from_ids(ib)[1])
        err = np.inf if s.base is None else np.linalg.norm(s.base[:2] - BOTTLE_XY)
        self.assertTrue((not s.ok) or err > 25.0)


class EstimatorAccuracyTests(unittest.TestCase):
    def test_bottle_axis_height_diameter_mouth(self):
        for bxy, cxy in (((300, -60), (180, -200)), ((200, -60), (300, -210)), ((150, -150), (300, -100))):
            sc = SynScene([Bottle(*bxy), Cup(*cxy)])
            frames, ids = frames_for(sc)
            e = bottle(frames, masks_from_ids(ids)[1])
            with self.subTest(bxy=bxy):
                self.assertTrue(e.ok, e.rejection_reason)
                self.assertLess(np.linalg.norm(e.base[:2] - np.array(bxy)), 5.0)
                self.assertLess(abs(e.dims["height"] - 180.0), 3.0)
                self.assertLess(abs(e.dims["diameter"] - 64.0), 4.0)
                self.assertLess(np.linalg.norm(e.position - np.array([bxy[0], bxy[1], TABLE_Z + 180.0])), 6.0)
                self.assertGreaterEqual(e.dims["top_radius"], 14.0 - 1.0)  # never under-reported by much

    def test_cup_rim_center_radius_plane(self):
        frames, ids = frames_for(scene())
        e = cup(frames, masks_from_ids(ids)[3])
        self.assertTrue(e.ok, e.rejection_reason)
        self.assertLess(np.linalg.norm(e.position[:2] - CUP_XY), 2.0)
        self.assertLess(abs(e.dims["rim_radius"] - 40.0), 2.0)       # mid-wall of 42/38
        self.assertLess(abs(e.position[2] - (TABLE_Z + 95.0)), 2.0)
        self.assertLess(e.extra["rim_tilt_deg"], 3.0)
        self.assertLess(abs(e.dims["outer_radius"] - 42.0), 2.5)

    def test_cup_box_center_regression(self):
        """The old path deprojected the mask centroid / box centre at one depth;
        from the wrist camera that is 10+ mm off the opening."""
        frames, ids = frames_for(scene())
        f0 = frames[0]
        m = masks_from_ids(ids)[3]
        ys, xs = np.nonzero(m & (f0.depth_mm > 0))
        vals, cnt = np.unique(np.rint(f0.depth_mm[ys, xs]).astype(int), return_counts=True)
        z_mode = float(vals[np.argmax(cnt)])
        old_centroid = apply(f0.T_world_cam, f0.model.deproject(np.array([xs.mean()]), np.array([ys.mean()]), np.array([z_mode])))[0]
        box_c = apply(f0.T_world_cam, f0.model.deproject(np.array([(xs.min() + xs.max()) / 2]),
                                                         np.array([(ys.min() + ys.max()) / 2]), np.array([z_mode])))[0]
        new = cup(frames, m)
        self.assertGreater(np.linalg.norm(old_centroid[:2] - CUP_XY), 10.0)
        self.assertGreater(np.linalg.norm(box_c[:2] - CUP_XY), 10.0)
        self.assertLess(np.linalg.norm(new.position[:2] - CUP_XY), 2.0)


class DepthPathologyTests(unittest.TestCase):
    def test_deliberately_misregistered_depth_rejected(self):
        """Depth from the wider-FOV depth imager 15 mm to the side, labelled
        aligned (what the unaligned RealSense stream looks like)."""
        dm = CameraModel(fx=455 * 0.7, fy=455 * 0.7, cx=320, cy=180, width=640, height=360)
        frames, ids = frames_for(scene(), depth_model=dm, depth_offset_cam=np.array([-15.0, 0, 0]))
        m = masks_from_ids(ids)
        self.assertEqual(bottle(frames, m[1]).rejection_reason, "depth_color_misregistered")
        self.assertEqual(cup(frames, m[3]).rejection_reason, "depth_color_misregistered")

    def test_baseline_only_misregistration_rejected(self):
        frames, ids = frames_for(scene(), depth_offset_cam=np.array([-15.0, 0, 0]))
        self.assertEqual(bottle(frames, masks_from_ids(ids)[1]).rejection_reason, "depth_color_misregistered")

    def test_zero_depth_under_mask_rejected_no_fallback(self):
        frames, ids = frames_for(scene(), holes={1: 1.0})
        e = bottle(frames, masks_from_ids(ids)[1])
        self.assertFalse(e.ok)
        self.assertIsNone(e.position)
        self.assertEqual(e.rejection_reason, "insufficient_object_depth")

    def test_reflective_bottle_holes(self):
        frames, ids = frames_for(scene(), holes={1: 0.85})
        self.assertEqual(bottle(frames, masks_from_ids(ids)[1]).rejection_reason, "insufficient_object_depth")
        frames, ids = frames_for(scene(), holes={1: 0.5})
        e = bottle(frames, masks_from_ids(ids)[1])
        self.assertTrue(e.ok, e.rejection_reason)
        self.assertLess(np.linalg.norm(e.base[:2] - BOTTLE_XY), 6.0)

    def test_background_heavy_mask_rejected(self):
        frames, ids = frames_for(scene())
        import cv2

        fat = cv2.dilate(masks_from_ids(ids)[1], np.ones((61, 61), np.uint8))
        e = bottle(frames, fat)
        self.assertFalse(e.ok)

    def test_box_shaped_mask_rejected(self):
        frames, ids = frames_for(scene())
        ys, xs = np.nonzero(masks_from_ids(ids)[1])
        box = np.zeros_like(ids, dtype=np.uint8)
        box[ys.min():ys.max(), xs.min():xs.max()] = 1
        self.assertEqual(bottle(frames, box).rejection_reason, "mask_is_a_box")

    def test_truncated_at_edge_rejected(self):
        sc = SynScene([Bottle(-60.0, 60.0), Cup(*CUP_XY)])
        frames, ids = frames_for(sc)
        m = masks_from_ids(ids)
        if 1 in m:
            self.assertFalse(bottle(frames, m[1]).ok)

    def test_multimodal_temporal_depth_rejected(self):
        frames, ids = frames_for(scene())
        f = list(frames)
        f[1] = dataclasses.replace(f[1], depth_mm=f[1].depth_mm + 25.0)
        f[2] = dataclasses.replace(f[2], depth_mm=f[2].depth_mm + 50.0)
        self.assertFalse(bottle(f, masks_from_ids(ids)[1]).ok)

    def test_lying_bottle_rejected(self):
        frames, ids = frames_for(SynScene([Bottle(*BOTTLE_XY), Cup(*CUP_XY), LyingCylinder(120, -40, yaw_deg=30)]))
        self.assertFalse(bottle(frames, masks_from_ids(ids)[4]).ok)

    def test_can_is_never_a_cup(self):
        frames, ids = frames_for(SynScene([Bottle(*BOTTLE_XY), Can(330, -250)]))
        self.assertFalse(cup(frames, masks_from_ids(ids)[2]).ok)


class NoWholeFrameFallbackTests(unittest.TestCase):
    def test_sorter_path_skips_pourables_without_depth(self):
        """components.shapes no longer localizes a bottle/can/cup at the
        whole-frame median (table) depth when its mask has no depth."""
        import asyncio

        from components import shapes

        m = np.zeros((40, 40), np.uint8)
        m[10:30, 10:30] = 255
        depth = np.zeros((40, 40), np.float32)
        depth[:, :] = 500.0
        depth[m > 0] = 0.0      # object has no depth of its own

        class Intr:
            focal_x_px = focal_y_px = 400.0
            center_x_px = center_y_px = 20.0

        async def fake_pixel_to_world(*a, **k):
            raise AssertionError("must not localize an object without depth")

        def shp(label):
            return shapes.DetectedShape(label=label, cx=20, cy=20, area=400.0, vertices=4, aspect_ratio=1.0,
                                        box=(10, 10, 20, 20), color=label, mask=m.copy())

        orig = shapes._pixel_to_world
        shapes._pixel_to_world = fake_pixel_to_world
        try:
            for label in shapes.NO_TABLE_DEPTH_FALLBACK:
                out = asyncio.run(shapes._locate_region_shapes(None, "cam", [shp(label)], depth, Intr(), "world",
                                                               table_depth=500.0))
                self.assertEqual(out, [], label)
        finally:
            shapes._pixel_to_world = orig


class SceneSelectionTests(unittest.TestCase):
    def _est(self, sc, labels):
        frames, ids = frames_for(sc)
        m = masks_from_ids(ids)
        out = []
        for oid, label in labels.items():
            if oid not in m:
                continue
            out.append(cup(frames, m[oid]) if label == "cup" else bottle(frames, m[oid], label))
        return out

    def test_cases(self):
        cases = [
            ("ok", SynScene([Bottle(*BOTTLE_XY), Cup(*CUP_XY)]), {1: "bottle", 3: "cup"}, "bottle", None),
            ("no cup", SynScene([Bottle(*BOTTLE_XY)]), {1: "bottle"}, "bottle", "no_cup"),
            ("cup only", SynScene([Cup(*CUP_XY)]), {3: "cup"}, "bottle", "no_source"),
            ("bottle+can no cup", SynScene([Bottle(*BOTTLE_XY), Can(330, -250)]), {1: "bottle", 2: "can"}, "any",
             "bottle_and_can_without_cup"),
            ("multiple cups", SynScene([Bottle(*BOTTLE_XY), Cup(*CUP_XY), Cup(330, -250, obj_id=5)]),
             {1: "bottle", 3: "cup", 5: "cup"}, "bottle", "multiple_cups"),
            ("two sources", SynScene([Bottle(*BOTTLE_XY), Cup(*CUP_XY), Can(330, -250)]),
             {1: "bottle", 2: "can", 3: "cup"}, "any", "multiple_sources_need_selection"),
            ("requested can, have bottle", SynScene([Bottle(*BOTTLE_XY), Cup(*CUP_XY)]),
             {1: "bottle", 3: "cup"}, "can", "no_source"),
        ]
        for name, sc, labels, req, reason in cases:
            with self.subTest(name):
                sel = select_pour_pair(self._est(sc, labels), req)
                self.assertEqual(sel.reason, reason)

    def test_explicit_selection_disambiguates(self):
        sc = SynScene([Bottle(*BOTTLE_XY), Cup(*CUP_XY), Can(330, -250)])
        frames, ids = frames_for(sc)
        est = self._est(sc, {1: "bottle", 2: "can", 3: "cup"})
        v, u = np.argwhere(ids == 2)[0]
        sel = select_pour_pair(est, "any", source_px=(u, v))
        self.assertTrue(sel.ok)
        self.assertEqual(sel.source.label, "can")


class ReobservationMaskTests(unittest.TestCase):
    def test_depth_roi_mask_ignores_objects_in_front(self):
        sc = scene()
        frames, ids = frames_for(sc)
        prior = cup(frames, masks_from_ids(ids)[3])
        view = HOME_FLANGE @ make_T(rot_x(np.radians(20.0)), (-60.0, 60.0, -80.0))
        f2, ids2 = frames_for(sc, view)

        class NoOcc:
            C = np.zeros((0, 3))
            R = np.zeros(0)

        mask, diag = cup_mask_from_prior(f2, prior, TABLE, NoOcc())
        # only cup pixels, never the bottle standing elsewhere
        self.assertEqual(int((mask & (ids2 == 1)).sum()), 0)
        fresh = cup(f2, mask, mask_from_depth=True, require_interior=False)
        self.assertTrue(fresh.ok, fresh.rejection_reason)
        self.assertLess(np.linalg.norm(fresh.position[:2] - prior.position[:2]), 5.0)


if __name__ == "__main__":
    unittest.main()
