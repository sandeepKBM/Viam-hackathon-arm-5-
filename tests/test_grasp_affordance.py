"""Offline verification of components/grasp_affordance.py against SYNTHETIC
point clouds -- no robot, no camera, no live Viam.

Two families of synthetic clouds are used deliberately:

- Deterministic structured grids (Cartesian for the cube/box/solid cylinder,
  a dense polar grid for the hollow tube) so grid-based occupancy features
  are gap-free and those tests are not flaky.
- Genuinely RANDOM polar-sampled clouds (uniform in r/theta/z, matching what
  a polar-sampled synthetic cloud -- or a real RealSense cloud, which is
  naturally denser near its own optical axis -- looks like): uniform-in-r
  sampling is intentionally sparse near the sampling origin, which is
  exactly the failure mode a prior version of this classifier had (see
  `test_polar_solid_cylinder_is_side_not_inside_outside`: a SOLID polar
  cylinder's top-slice occupancy grid can under-sample its own center badly
  enough to look like an annulus). These tests exist specifically to catch
  that regression -- do not "fix" a failure here by making the cloud more
  uniform; the point is that the CLASSIFIER must handle non-uniform sampling
  via the 3D interior-column check in `compute_features`
  (`interior_column_empty`), not that the test data must be friendly.

Also exercises the numba-absent path directly: this repo's .venv has no
`numba` installed (confirmed before writing this test), so
`components.grasp_affordance._HAS_NUMBA` is False here and every test below
runs the pure-Python `njit` no-op fallback. If numba IS installed elsewhere,
the same tests pass via the JIT path too -- the decision logic doesn't
depend on which one is active.
"""

from __future__ import annotations

import numpy as np
import pytest

from components.grasp_affordance import (
    DEFAULT_GRIPPER_MAX_WIDTH_MM,
    GraspType,
    _HAS_NUMBA,
    classify_grasp,
    compute_features,
    points_from_pcd,
)


def test_numba_flag_is_bool() -> None:
    # Documents which path (JIT vs. pure-python no-op fallback) this test
    # run is exercising; either way the module must import and run.
    assert isinstance(_HAS_NUMBA, bool)


def _solid_cube(side_xy: float, height: float, n_per_axis: int = 14) -> np.ndarray:
    """A dense, solid cube -> compact, no cavity -> expect TOP_DOWN."""
    xs = np.linspace(0.0, side_xy, n_per_axis)
    ys = np.linspace(0.0, side_xy, n_per_axis)
    zs = np.linspace(0.0, height, n_per_axis)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _solid_cylinder(radius: float, height: float, n_xy: int = 81, n_z: int = 30) -> np.ndarray:
    """A dense, solid (filled-disk cross-section) cylinder -> tall, no
    cavity (top slice is a filled disk) -> expect SIDE.

    Sampled as a Cartesian XY grid (spacing well under the classifier's
    default occupancy-grid cell size) clipped to the disk, rather than polar
    coordinates -- a polar grid's radial spacing leaves a small unsampled
    gap near the center (between r=0 and the first nonzero radius ring),
    which the occupancy grid can misread as a false interior cavity.
    """
    xs = np.linspace(-radius, radius, n_xy)
    ys = np.linspace(-radius, radius, n_xy)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    mask = X**2 + Y**2 <= radius**2
    xy = np.stack([X[mask], Y[mask]], axis=1)
    zs = np.linspace(0.0, height, n_z)
    pts = np.empty((xy.shape[0] * n_z, 3), dtype=np.float64)
    for i, z in enumerate(zs):
        pts[i * xy.shape[0] : (i + 1) * xy.shape[0], :2] = xy
        pts[i * xy.shape[0] : (i + 1) * xy.shape[0], 2] = z
    return pts


def _hollow_tube(
    inner_r: float,
    outer_r: float,
    height: float,
    n_theta: int = 180,
    n_radii: int = 6,
    n_z: int = 40,
) -> np.ndarray:
    """An open cylindrical shell (cup wall): annulus cross-section at every
    z layer, open top AND bottom -> empty interior ringed by the wall in
    the top-slice occupancy grid -> expect INSIDE_OUTSIDE."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radii = np.linspace(inner_r, outer_r, n_radii)
    zs = np.linspace(0.0, height, n_z)
    pts = []
    for z in zs:
        for r in radii:
            for t in thetas:
                pts.append((r * np.cos(t), r * np.sin(t), z))
    return np.asarray(pts, dtype=np.float64)


def _solid_box(x_mm: float, y_mm: float, height: float, n_per_axis: int = 10) -> np.ndarray:
    xs = np.linspace(0.0, x_mm, n_per_axis)
    ys = np.linspace(0.0, y_mm, n_per_axis)
    zs = np.linspace(0.0, height, max(4, n_per_axis // 2))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


# --------------------------------------------------------------------------
# RANDOM polar-sampled clouds -- the regression-repro shapes. Sampling r
# uniformly (rather than uniformly in AREA, i.e. r = R*sqrt(u)) is
# deliberately the "wrong"/naive way to sample a disk: it packs points
# increasingly sparsely as r -> 0, exactly like a real depth-camera cloud
# (denser near the sensor's optical axis / near edges, sparser elsewhere)
# or any other polar-parameterized synthetic generator. A solid cylinder
# built this way can fool a purely-2D top-slice occupancy check into seeing
# a "hole" at the center that isn't really there.
# --------------------------------------------------------------------------

_POLAR_SEED = 20240917


def _polar_solid_cylinder(radius: float, height: float, n: int = 20000) -> np.ndarray:
    """A SOLID cylinder, sampled uniformly in (r, theta, z) -- sparse near
    the axis by construction. Must still classify SIDE, never INSIDE_OUTSIDE."""
    rng = np.random.default_rng(_POLAR_SEED)
    r = rng.uniform(0.0, radius, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(0.0, height, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def _polar_hollow_cup(inner_r: float, outer_r: float, height: float, n: int = 20000) -> np.ndarray:
    """A genuinely open cup wall, sampled uniformly in (r, theta, z) with r
    restricted to the wall annulus. Must still classify INSIDE_OUTSIDE."""
    rng = np.random.default_rng(_POLAR_SEED + 1)
    r = rng.uniform(inner_r, outer_r, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(0.0, height, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def _random_solid_cube(side_xy: float, height: float, n: int = 20000) -> np.ndarray:
    """A solid cube/box sampled with plain random (non-grid) points, as an
    extra non-uniform-sampling sanity check alongside the polar ones above."""
    rng = np.random.default_rng(_POLAR_SEED + 2)
    x = rng.uniform(0.0, side_xy, n)
    y = rng.uniform(0.0, side_xy, n)
    z = rng.uniform(0.0, height, n)
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------
# compute_features sanity
# --------------------------------------------------------------------------


def test_compute_features_cube_dimensions() -> None:
    cloud = _solid_cube(side_xy=70.0, height=50.0)
    feats = compute_features(cloud)
    assert feats.height_mm == pytest.approx(50.0, abs=1.0)
    assert feats.major_length_mm == pytest.approx(70.0, abs=1.0)
    assert feats.minor_length_mm == pytest.approx(70.0, abs=1.0)
    assert not feats.has_cavity


def test_compute_features_hollow_tube_has_cavity() -> None:
    cloud = _hollow_tube(inner_r=25.0, outer_r=35.0, height=70.0)
    feats = compute_features(cloud)
    assert feats.has_cavity
    assert feats.inner_diameter_mm == pytest.approx(50.0, abs=10.0)
    assert feats.wall_thickness_mm == pytest.approx(10.0, abs=8.0)
    assert feats.rim_z == pytest.approx(70.0, abs=1.0)


# --------------------------------------------------------------------------
# classify_grasp per-type behavior
# --------------------------------------------------------------------------


def test_cube_is_top_down() -> None:
    cloud = _solid_cube(side_xy=70.0, height=50.0)
    g = classify_grasp(cloud)
    assert g.grasp_type is GraspType.TOP_DOWN
    assert g.approach == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert g.width_mm == pytest.approx(70.0, abs=2.0)
    assert g.confidence > 0.3
    assert g.note == ""


def test_tall_cylinder_is_side() -> None:
    # radius 25mm -> diameter 50mm, fits the default 80mm gripper; height
    # 130mm is well over 1.3x the 50mm footprint -> tall.
    cloud = _solid_cylinder(radius=25.0, height=130.0)
    g = classify_grasp(cloud)
    assert g.grasp_type is GraspType.SIDE
    # Approach must be horizontal (side grasp), not top-down.
    assert g.approach[2] == pytest.approx(0.0, abs=1e-6)
    assert g.grasp_axis[2] == pytest.approx(0.0, abs=1e-6)
    assert g.width_mm == pytest.approx(50.0, abs=2.0)
    assert g.confidence > 0.3


def test_hollow_cup_is_inside_outside() -> None:
    # outer radius 35 / inner radius 25 -> wall thickness 10mm, inner
    # opening 50mm: both comfortably within the classifier's default
    # thresholds (gripper_max_width_mm=80, min_finger_clearance_mm=15).
    cloud = _hollow_tube(inner_r=25.0, outer_r=35.0, height=70.0)
    g = classify_grasp(cloud)
    assert g.grasp_type is GraspType.INSIDE_OUTSIDE
    assert g.approach == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert g.rim_z == pytest.approx(70.0, abs=1.0)
    assert g.wall_thickness_mm is not None and g.wall_thickness_mm == pytest.approx(10.0, abs=8.0)
    assert g.inner_diameter_mm is not None and g.inner_diameter_mm == pytest.approx(50.0, abs=10.0)
    assert g.width_mm == pytest.approx(g.wall_thickness_mm, abs=1e-6)
    assert g.confidence > 0.0


def test_too_wide_box_gets_low_confidence_note() -> None:
    # 150mm x 150mm footprint is wider than the default 80mm gripper in
    # every branch (not tall enough for SIDE either) -> fallback.
    cloud = _solid_box(x_mm=150.0, y_mm=150.0, height=20.0)
    g = classify_grasp(cloud)
    assert g.width_mm > DEFAULT_GRIPPER_MAX_WIDTH_MM
    assert g.confidence < 0.4
    assert "too wide" in g.note


# --------------------------------------------------------------------------
# Regression coverage: non-uniform (polar) sampling must NOT false-positive
# a solid object as an open container. See the module docstring.
# --------------------------------------------------------------------------


def test_polar_solid_cylinder_is_side_not_inside_outside() -> None:
    # Same radius/height as test_tall_cylinder_is_side, but sampled uniformly
    # in (r, theta, z) -- sparse near the axis. Before the interior-column
    # 3D check was added, this misclassified as INSIDE_OUTSIDE because the
    # top-slice occupancy grid under-sampled the disk's center into a
    # false-looking hole.
    cloud = _polar_solid_cylinder(radius=25.0, height=130.0)
    g = classify_grasp(cloud)
    assert g.grasp_type is GraspType.SIDE
    assert g.grasp_type is not GraspType.INSIDE_OUTSIDE
    assert g.approach[2] == pytest.approx(0.0, abs=1e-6)
    assert g.width_mm == pytest.approx(50.0, rel=0.1)


def test_polar_hollow_cup_is_inside_outside() -> None:
    # A genuinely open cup wall, sampled the same non-uniform way -- must
    # still be recognized as hollow (the interior-column check must not
    # over-correct into rejecting REAL cavities).
    cloud = _polar_hollow_cup(inner_r=25.0, outer_r=35.0, height=70.0)
    g = classify_grasp(cloud)
    assert g.grasp_type is GraspType.INSIDE_OUTSIDE
    assert g.approach == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert g.inner_diameter_mm is not None and g.inner_diameter_mm == pytest.approx(50.0, rel=0.25)
    assert g.wall_thickness_mm is not None and g.wall_thickness_mm > 0.0


def test_polar_cube_is_top_down() -> None:
    # Non-grid (random) sampling of a compact solid -- another sanity check
    # that the fix didn't over-correct into rejecting ordinary solids.
    cloud = _random_solid_cube(side_xy=70.0, height=50.0)
    g = classify_grasp(cloud)
    assert g.grasp_type is GraspType.TOP_DOWN
    assert g.approach == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert g.width_mm == pytest.approx(70.0, rel=0.1)


def test_classify_grasp_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        classify_grasp(np.zeros((3, 3)))  # too few points
    with pytest.raises(ValueError):
        classify_grasp(np.zeros((20, 2)))  # wrong shape


# --------------------------------------------------------------------------
# points_from_pcd (ASCII path only; documents the binary-PCD limitation)
# --------------------------------------------------------------------------


def test_points_from_pcd_ascii() -> None:
    pcd_text = (
        "# .PCD v0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "POINTS 2\n"
        "DATA ascii\n"
        "1.0 2.0 3.0\n"
        "4.0 5.0 6.0\n"
    )
    arr = points_from_pcd(pcd_text.encode("ascii"))
    assert arr.shape == (2, 3)
    assert arr[0].tolist() == [1.0, 2.0, 3.0]
    assert arr[1].tolist() == [4.0, 5.0, 6.0]


def test_points_from_pcd_binary_not_implemented() -> None:
    fake_binary = b"# .PCD v0.7\nFIELDS x y z\nDATA binary\n" + b"\x00" * 16
    with pytest.raises(NotImplementedError):
        points_from_pcd(fake_binary)
