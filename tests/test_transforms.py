import math
import unittest

import numpy as np

from components.transforms import (
    T_to_pose,
    axis_angle,
    hand_eye_park_martin,
    interpolate_T,
    invert,
    make_T,
    matrix_to_ov,
    ov_to_matrix,
    pose_to_T,
    rot_x,
    rot_y,
    rot_z,
    rotation_angle_deg,
    solve_pivot_tcp,
)


class OrientationVectorTests(unittest.TestCase):
    def test_round_trip_including_poles(self):
        rng = np.random.default_rng(0)
        cases = [tuple(rng.normal(size=3)) + (float(rng.uniform(-180, 180)),) for _ in range(500)]
        cases += [(0, 0, 1, 30.0), (0, 0, -1, -26.78), (1e-9, 0, -1, 90.0), (1, 0, 0, -177.5)]
        for ox, oy, oz, th in cases:
            R = ov_to_matrix(ox, oy, oz, th)
            R2 = ov_to_matrix(*matrix_to_ov(R))
            self.assertLess(np.abs(R - R2).max(), 1e-9)

    def test_matches_viam_convention(self):
        # rdk spatialmath: R = Rz(lon) Ry(lat) Rz(theta); (0,0,1,th) is a pure yaw.
        self.assertTrue(np.allclose(ov_to_matrix(0, 0, 1, 30), rot_z(math.radians(30))))
        R = ov_to_matrix(1, 0, 0, 0)
        self.assertTrue(np.allclose(R[:, 2], [1, 0, 0]))

    def test_pose_round_trip(self):
        pose = {"x": 281.2, "y": -86.9, "z": 532.6, "o_x": -0.0319, "o_y": 0.0173, "o_z": -0.9993, "theta": -26.78}
        T = pose_to_T(pose)
        back = T_to_pose(T)
        self.assertTrue(np.allclose(pose_to_T(back), T, atol=1e-9))


class SolverTests(unittest.TestCase):
    X = make_T(ov_to_matrix(-0.030391, -0.003538, 0.999532, -97.731173), (83.0, -14.0, 18.0))

    def _samples(self, n, rng, degenerate=False):
        T_bt = make_T(rot_z(0.3), (350.0, 50.0, 0.0))
        Tbf, Tct = [], []
        for i in range(n):
            if degenerate:
                R = rot_z(-1 + 2 * i / max(1, n - 1)) @ rot_x(math.pi)
            else:
                R = rot_z(rng.uniform(-1, 1)) @ rot_x(math.pi + rng.uniform(-0.5, 0.5)) @ rot_y(rng.uniform(-0.5, 0.5))
            F = make_T(R, np.array([350.0, 50.0, 450.0]) + rng.uniform(-80, 80, 3))
            C = invert(F @ self.X) @ T_bt
            C = make_T(C[:3, :3] @ axis_angle(rng.normal(size=3), math.radians(0.05)), C[:3, 3] + rng.normal(0, 0.2, 3))
            Tbf.append(F)
            Tct.append(C)
        return Tbf, Tct

    def test_park_martin_recovers_mount(self):
        rng = np.random.default_rng(1)
        Tbf, Tct = self._samples(14, rng)
        Xe, diag = hand_eye_park_martin(Tbf, Tct)
        self.assertLess(np.linalg.norm(Xe[:3, 3] - self.X[:3, 3]), 1.0)
        self.assertLess(rotation_angle_deg(Xe[:3, :3], self.X[:3, :3]), 0.2)

    def test_degenerate_motion_is_rejected(self):
        rng = np.random.default_rng(2)
        Tbf, Tct = self._samples(12, rng, degenerate=True)
        with self.assertRaises(ValueError):
            hand_eye_park_martin(Tbf, Tct)

    def test_pivot_tcp(self):
        rng = np.random.default_rng(3)
        c, p = np.array([0.0, 0.0, 172.0]), np.array([400.0, 0.0, 20.0])
        Fs = []
        for _ in range(6):
            R = rot_z(rng.uniform(-1, 1)) @ rot_x(math.pi + rng.uniform(-0.6, 0.6)) @ rot_y(rng.uniform(-0.6, 0.6))
            Fs.append(make_T(R, p - R @ c))
        ce, pe, res = solve_pivot_tcp(Fs)
        self.assertLess(np.linalg.norm(ce - c), 1e-6)
        self.assertLess(max(res), 1e-6)

    def test_pivot_needs_rotation_spread(self):
        Fs = [make_T(rot_x(math.pi), (400.0, 0.0, 200.0 + i)) for i in range(5)]
        with self.assertRaises(ValueError):
            solve_pivot_tcp(Fs)


class InterpolationTests(unittest.TestCase):
    def test_steps_honour_both_bounds(self):
        T0 = make_T(np.eye(3), (0, 0, 0))
        T1 = make_T(rot_x(math.radians(90)), (100, 0, 0))
        seq = interpolate_T(T0, T1, 10.0, 5.0)
        self.assertTrue(np.allclose(seq[-1], T1))
        prev = T0
        for T in seq:
            self.assertLessEqual(np.linalg.norm(T[:3, 3] - prev[:3, 3]), 10.0 + 1e-9)
            self.assertLessEqual(rotation_angle_deg(prev[:3, :3], T[:3, :3]), 5.0 + 1e-9)
            prev = T


if __name__ == "__main__":
    unittest.main()
