"""Grasp-affordance classifier: per-object 3D point cloud -> grasp TYPE + params.

WHERE THE INPUT COMES FROM
---------------------------
Viam's vision service exposes `get_object_point_clouds()`, which segments a
scene into per-object point clouds (each a `viam.services.vision.PointCloudObject`
carrying `.point_cloud` -- PCD-format bytes -- and `.geometries`). This module
does NOT call Viam or touch a camera: `classify_grasp()` takes a plain
``(N, 3)`` float ``numpy`` array of XYZ points in the arm/world frame (z-up),
which is exactly what you get after parsing one object's `.point_cloud`. That
keeps the classifier fully offline-testable with synthetic clouds; wiring it
to a live robot is just "parse PCD bytes -> numpy array -> classify_grasp".
`points_from_pcd()` below is a tiny, best-effort stub documenting that parsing
step -- it is NOT required by (or exercised in) the geometry/decision logic.

WHAT THIS DOES NOT DO -- READ BEFORE WIRING TO A REAL PICK
------------------------------------------------------------
This classifies what the OBJECT affords geometrically -- it has no idea what
the arm can actually reach. In particular, the xArm5 used elsewhere in this
repo (see components/ik.py, components/fast_planner.py) is a 5-DOF arm: it
can independently satisfy a 3-DOF position target plus point its wrist axis
in an arbitrary direction (2 more DOF), but it cannot in general realize an
arbitrary full 6-DOF end-effector pose. A SIDE grasp this module proposes may
need a wrist orientation the 5-DOF chain simply cannot reach at that
position; an INSIDE_OUTSIDE grasp needs one pad placed precisely inside a
rim and the other outside it, which is a much tighter position/orientation
tolerance than a symmetric top-down pinch. So: downstream code MUST run the
proposed `approach`/`grasp_axis`/`center` through a reachability check (e.g.
`components.ik.XArm5IK` / `components.fast_planner`) before committing to a
grasp -- this module only answers "what grasp does the geometry afford",
never "can this arm execute it".

NUMBA
-----
The per-point geometry (PCA accumulation/projection, top-slice rasterization,
flood-fill cavity detection) is written as explicit loops over numpy arrays
so it can be JIT-compiled with `numba.njit`. `numba` is NOT added to
requirements.txt (another agent may be editing it) and is imported lazily,
guarded by a try/except: if it's missing, `njit` falls back to a transparent
no-op decorator (supports both bare `@njit` and parameterized `@njit(cache=True)`
use) so every function below runs as plain, unaccelerated Python/numpy.
Correctness is identical either way; only speed differs. `_HAS_NUMBA` records
which path is active.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Lazy numba import with a no-op `njit` fallback.
# --------------------------------------------------------------------------
try:
    from numba import njit as _real_njit  # noqa: WPS433 (lazy, optional)

    njit = _real_njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised whenever numba isn't installed
    _HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        """No-op stand-in for numba.njit so this module (and every function
        decorated with it) works identically, just unaccelerated, when numba
        isn't installed. Supports both bare `@njit` and `@njit(cache=True)`
        style usage."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator


Tuple3 = Tuple[float, float, float]

# Typical UFactory xArm gripper opening; NOT sourced from components.constants
# (no such constant exists there today -- checked before adding this) so this
# is documented here as a default/assumption. Override per-call or via the
# GRIPPER_MAX_WIDTH_MM env var if the real gripper's max opening differs.
import os  # noqa: E402

DEFAULT_GRIPPER_MAX_WIDTH_MM = float(os.environ.get("GRIPPER_MAX_WIDTH_MM", "80.0"))


class GraspType(Enum):
    TOP_DOWN = "top_down"
    SIDE = "side"
    INSIDE_OUTSIDE = "inside_outside"


@dataclass
class GraspAffordance:
    """A geometric grasp proposal for one object's point cloud. See the
    module docstring's "WHAT THIS DOES NOT DO" section -- executability
    (can the 5-DOF arm actually reach this approach/orientation) is NOT
    checked here."""

    grasp_type: GraspType
    approach: Tuple3  # unit vector; direction the gripper travels INTO the grasp
    grasp_axis: Tuple3  # unit vector; the line the two pads close along
    width_mm: float  # object dimension the pads must span
    center: Tuple3  # grasp point (xyz) in the same frame as the input cloud
    confidence: float  # 0..1 heuristic confidence
    note: str = ""
    # INSIDE_OUTSIDE-specific (None for TOP_DOWN / SIDE):
    rim_z: Optional[float] = None
    wall_thickness_mm: Optional[float] = None
    inner_diameter_mm: Optional[float] = None


@dataclass
class GraspFeatures:
    """Intermediate geometry computed from a point cloud -- exposed mainly so
    tests (and callers who want to log/debug) can inspect it directly."""

    z_min: float
    z_max: float
    height_mm: float
    centroid: Tuple3
    footprint_center_xy: Tuple[float, float]
    major_axis_xy: Tuple[float, float]  # unit vector, longer footprint axis
    minor_axis_xy: Tuple[float, float]  # unit vector, shorter footprint axis
    major_length_mm: float
    minor_length_mm: float
    solidity: float  # occupied / bounding-area ratio of the top-slice grid
    has_cavity: bool
    inner_diameter_mm: float
    wall_thickness_mm: float
    rim_z: float
    # True iff either there's no top-slice cavity to check, or there IS one
    # and its interior column was independently verified empty in 3D (see
    # `_count_points_in_boxes` / the "interior column" check in
    # `compute_features`). False means the "cavity" is actually just a
    # sparse-sampled SOLID interior -- reject it.
    interior_column_empty: bool = True


# --------------------------------------------------------------------------
# numba-accelerated per-point geometry kernels (plain-Python-safe: numpy
# arrays + explicit loops only, no python objects, so they JIT under numba
# and run correctly, just slower, under the no-op fallback above).
# --------------------------------------------------------------------------


@njit(cache=True)
def _z_extent(z: np.ndarray) -> Tuple[float, float]:
    zmin = z[0]
    zmax = z[0]
    for k in range(z.shape[0]):
        v = z[k]
        if v < zmin:
            zmin = v
        if v > zmax:
            zmax = v
    return zmin, zmax


@njit(cache=True)
def _pca_footprint(xy: np.ndarray):
    """Mean-center the XY footprint, find its principal axes via a
    closed-form 2x2 eigendecomposition (no LAPACK dependency, numba-safe),
    then project every point onto both axes to get the true PCA-oriented
    bounding-box extents (major_len >= minor_len by convention)."""
    npts = xy.shape[0]
    mx = 0.0
    my = 0.0
    for k in range(npts):
        mx += xy[k, 0]
        my += xy[k, 1]
    mx /= npts
    my /= npts

    cxx = 0.0
    cyy = 0.0
    cxy = 0.0
    for k in range(npts):
        dx = xy[k, 0] - mx
        dy = xy[k, 1] - my
        cxx += dx * dx
        cyy += dy * dy
        cxy += dx * dy
    cxx /= npts
    cyy /= npts
    cxy /= npts

    trace = cxx + cyy
    disc = trace * trace / 4.0 - (cxx * cyy - cxy * cxy)
    if disc < 0.0:
        disc = 0.0
    disc = math.sqrt(disc)
    lam1 = trace / 2.0 + disc

    # The eigenvalue gap is 2*disc. When it's tiny relative to the overall
    # variance scale (trace), the footprint is near-isotropic (a square or
    # circle, e.g. under 4-fold+ symmetry the true covariance is EXACTLY
    # isotropic in every orientation) -- the "principal" direction is
    # ill-conditioned, and with real (non-grid) sampling noise, a computed
    # eigenvector can land anywhere, including near a square's diagonal.
    # Projecting onto a diagonal instead of an edge-parallel direction
    # inflates the measured extent by up to sqrt(2)x for a square. Snap to a
    # canonical axis-aligned direction in that regime instead -- it's no
    # more "correct" for an arbitrarily-rotated symmetric footprint (there's
    # no well-defined principal axis to recover), but it's deterministic and
    # matches the common case (axis-aligned objects) instead of amplifying
    # sampling noise into the reported width.
    if disc > 0.03 * max(trace, 1e-9) and abs(cxy) > 1e-12:
        vx = lam1 - cyy
        vy = cxy
    elif cxx >= cyy:
        vx = 1.0
        vy = 0.0
    else:
        vx = 0.0
        vy = 1.0
    norm = math.sqrt(vx * vx + vy * vy)
    if norm < 1e-12:
        vx = 1.0
        vy = 0.0
        norm = 1.0
    vx /= norm
    vy /= norm
    # Minor axis is perpendicular to the major one in 2D.
    ux = -vy
    uy = vx

    maj_min = 1.0e18
    maj_max = -1.0e18
    min_min = 1.0e18
    min_max = -1.0e18
    for k in range(npts):
        dx = xy[k, 0] - mx
        dy = xy[k, 1] - my
        pmaj = dx * vx + dy * vy
        pmin = dx * ux + dy * uy
        if pmaj < maj_min:
            maj_min = pmaj
        if pmaj > maj_max:
            maj_max = pmaj
        if pmin < min_min:
            min_min = pmin
        if pmin > min_max:
            min_max = pmin

    major_len = maj_max - maj_min
    minor_len = min_max - min_min
    if minor_len > major_len:
        # Keep the convention major_len >= minor_len regardless of which
        # eigenvalue came out larger.
        vx, vy, ux, uy = ux, uy, vx, vy
        major_len, minor_len = minor_len, major_len

    mean = np.empty(2, dtype=np.float64)
    mean[0] = mx
    mean[1] = my
    major_axis = np.empty(2, dtype=np.float64)
    major_axis[0] = vx
    major_axis[1] = vy
    minor_axis = np.empty(2, dtype=np.float64)
    minor_axis[0] = ux
    minor_axis[1] = uy

    return mean, major_axis, minor_axis, major_len, minor_len


@njit(cache=True)
def _rasterize_top_slice(
    xy: np.ndarray, minx: float, miny: float, cellw: float, cellh: float, n: int
) -> np.ndarray:
    """Project `xy` (already filtered to the top ~20% of the cloud) onto an
    n x n occupancy grid over its own XY bounding box."""
    occ = np.zeros((n, n), dtype=np.uint8)
    for k in range(xy.shape[0]):
        col = int((xy[k, 0] - minx) / cellw)
        row = int((xy[k, 1] - miny) / cellh)
        if col < 0:
            col = 0
        elif col >= n:
            col = n - 1
        if row < 0:
            row = 0
        elif row >= n:
            row = n - 1
        occ[row, col] = 1
    return occ


@njit(cache=True)
def _flood_fill_exterior(occ: np.ndarray, n: int) -> np.ndarray:
    """4-connected flood fill of EMPTY cells starting from the grid border.
    Any empty cell NOT reached is enclosed by occupied cells -- i.e. an
    interior cavity (open-container annulus). Iterative (explicit array
    stack, bounded at n*n) so it's numba-nopython-safe."""
    visited = np.zeros((n, n), dtype=np.uint8)
    stack_r = np.empty(n * n, dtype=np.int64)
    stack_c = np.empty(n * n, dtype=np.int64)
    sp = 0
    for i in range(n):
        for j in range(n):
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                if occ[i, j] == 0 and visited[i, j] == 0:
                    visited[i, j] = 1
                    stack_r[sp] = i
                    stack_c[sp] = j
                    sp += 1
    while sp > 0:
        sp -= 1
        r = stack_r[sp]
        c = stack_c[sp]
        if r > 0 and occ[r - 1, c] == 0 and visited[r - 1, c] == 0:
            visited[r - 1, c] = 1
            stack_r[sp] = r - 1
            stack_c[sp] = c
            sp += 1
        if r < n - 1 and occ[r + 1, c] == 0 and visited[r + 1, c] == 0:
            visited[r + 1, c] = 1
            stack_r[sp] = r + 1
            stack_c[sp] = c
            sp += 1
        if c > 0 and occ[r, c - 1] == 0 and visited[r, c - 1] == 0:
            visited[r, c - 1] = 1
            stack_r[sp] = r
            stack_c[sp] = c - 1
            sp += 1
        if c < n - 1 and occ[r, c + 1] == 0 and visited[r, c + 1] == 0:
            visited[r, c + 1] = 1
            stack_r[sp] = r
            stack_c[sp] = c + 1
            sp += 1
    return visited


@njit(cache=True)
def _count_points_in_cells(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_lo: float,
    z_hi: float,
    minx: float,
    miny: float,
    cellw: float,
    cellh: float,
    n: int,
    interior_mask: np.ndarray,
    occ_mask: np.ndarray,
):
    """Count FULL-cloud points, restricted to the z-band [z_lo, z_hi] (the
    object's body BELOW the top slice), that land -- via the SAME grid
    transform used for the top-slice occupancy grid -- in an `interior_mask`
    cell (the candidate cavity) vs. an `occ_mask` cell (the wall).

    Deliberately reuses the actual per-cell masks rather than an
    axis-aligned bounding box: a circular annulus's inner hole is a disk,
    and a square bbox around that disk is a poor approximation -- wall
    points near the bbox's diagonal corners (outside the disk, still part of
    the ring) would be miscounted as "interior" by a bbox test, understating
    how empty the true cavity is. Per-cell membership avoids that.

    This is the 3D check that catches a sparse-sampled SOLID object: a real
    open container is empty through its height, so its interior cells have
    ~no points at ANY z, not just a sparsely-sampled top slice. A solid
    object (even if its top-slice occupancy grid under-samples the center
    into a false-looking hole) still has points filling that same XY region
    lower down -- so `interior_count` comes back comparable in density to
    the wall (`wall_count`), not near-empty."""
    interior_count = 0
    wall_count = 0
    for k in range(x.shape[0]):
        zk = z[k]
        if zk < z_lo or zk > z_hi:
            continue
        col = int((x[k] - minx) / cellw)
        row = int((y[k] - miny) / cellh)
        if col < 0:
            col = 0
        elif col >= n:
            col = n - 1
        if row < 0:
            row = 0
        elif row >= n:
            row = n - 1
        if interior_mask[row, col] != 0:
            interior_count += 1
        elif occ_mask[row, col] != 0:
            wall_count += 1
    return interior_count, wall_count


# --------------------------------------------------------------------------
# Feature extraction (plain Python glue around the numba kernels above).
# --------------------------------------------------------------------------


def compute_features(
    points: np.ndarray,
    *,
    top_slice_frac: float = 0.2,
    grid_cells: int = 40,
    min_cavity_cells: int = 4,
    interior_column_density_ratio: float = 0.3,
) -> GraspFeatures:
    """Compute height, PCA footprint, and top-slice occupancy/cavity
    features from an (N, 3) point cloud.

    - height (H): z_max - z_min.
    - footprint: PCA over ALL points' XY -> major/minor axes + extents.
    - top-slice occupancy grid: the top `top_slice_frac` of the cloud (by z)
      rasterized to XY on a `grid_cells` x `grid_cells` grid over its own
      bounding box; `solidity` = occupied / total cells.
    - cavity: an occupancy-grid region unreachable by flood fill from the
      grid border through empty cells -- i.e. empty space fully ringed by
      occupied cells (an open container's rim, seen from above). Its
      bounding box gives `inner_diameter_mm` (conservatively, the SMALLER of
      its row/col spans, since a finger needs to clear the tightest
      dimension); the occupied region's bounding box gives the outer
      footprint of the rim, and `wall_thickness_mm` is estimated as
      (outer_span - inner_diameter) / 2.
    - interior-column check (`interior_column_empty`): a candidate cavity
      found in the 2D top-slice grid is NOT trusted on its own -- a solid
      object whose point cloud is non-uniformly/polar sampled (sparse near
      its own center, exactly like a real RealSense cloud or any
      polar-coordinate synthetic cloud) can look like an annulus in that one
      slice even though it's genuinely solid. So the candidate cavity's XY
      footprint is checked against the FULL cloud's points in the
      mid/lower-height band BELOW the top slice: if that column has a point
      density comparable to the surrounding wall (>=
      `interior_column_density_ratio` of it), it's a sparse solid, not a
      real opening, and `interior_column_empty` is False.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if pts.shape[0] < 4:
        raise ValueError("need at least 4 points to compute footprint/height")

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    z_min, z_max = _z_extent(z)
    height_mm = float(z_max - z_min)
    centroid = pts.mean(axis=0)

    xy = pts[:, :2]
    mean_xy, major_axis, minor_axis, major_len, minor_len = _pca_footprint(xy)

    # Top slice: the top `top_slice_frac` of the z range, used to look for a
    # rim/annulus (an open container's opening is only visible from above).
    thresh_z = z_max - top_slice_frac * max(height_mm, 1e-6)
    top_mask = z >= thresh_z
    xy_top = xy[top_mask]
    if xy_top.shape[0] < 4:
        xy_top = xy  # degenerate (very thin slice / few points): use all points

    minx = float(xy_top[:, 0].min())
    miny = float(xy_top[:, 1].min())
    maxx = float(xy_top[:, 0].max())
    maxy = float(xy_top[:, 1].max())
    span_x = max(maxx - minx, 1e-6)
    span_y = max(maxy - miny, 1e-6)
    cellw = span_x / grid_cells
    cellh = span_y / grid_cells

    occ = _rasterize_top_slice(np.ascontiguousarray(xy_top), minx, miny, cellw, cellh, grid_cells)
    visited = _flood_fill_exterior(occ, grid_cells)

    occupied_cells = int(occ.sum())
    solidity = occupied_cells / float(grid_cells * grid_cells)

    interior = (occ == 0) & (visited == 0)
    interior_count = int(interior.sum())
    has_cavity = interior_count >= min_cavity_cells

    inner_diameter_mm = 0.0
    wall_thickness_mm = 0.0
    interior_column_empty = True
    if has_cavity:
        rows, cols = np.nonzero(interior)
        inner_rows_span = (int(rows.max()) - int(rows.min()) + 1) * cellh
        inner_cols_span = (int(cols.max()) - int(cols.min()) + 1) * cellw
        inner_diameter_mm = float(min(inner_rows_span, inner_cols_span))

        orows, ocols = np.nonzero(occ)
        outer_rows_span = (int(orows.max()) - int(orows.min()) + 1) * cellh
        outer_cols_span = (int(ocols.max()) - int(ocols.min()) + 1) * cellw
        outer_span = float(min(outer_rows_span, outer_cols_span))
        wall_thickness_mm = max(0.0, (outer_span - inner_diameter_mm) / 2.0)

        # 3D interior-column check: is the candidate cavity actually empty
        # BELOW the top slice too, or does the object's body fill back in
        # (a sparse-sampled solid)? Band = everything below the top slice.
        # Uses the actual interior/occupied CELL masks (not a bounding box)
        # so a circular annulus's disk-shaped hole is tested correctly.
        z_lo = float(z_min)
        z_hi = float(thresh_z)
        if z_hi - z_lo > 1e-6:
            interior_mask = interior.astype(np.uint8)
            interior_pts, wall_pts = _count_points_in_cells(
                x, y, z, z_lo, z_hi, minx, miny, cellw, cellh, grid_cells, interior_mask, occ
            )
            if wall_pts <= 0:
                # No wall points in this band at all (e.g. a rim that only
                # exists right at the top, tapered walls, etc.) -- nothing
                # to compare against, so don't second-guess the 2D result.
                interior_column_empty = True
            else:
                # Per-cell (not per-area) density: same cell size for both
                # masks, so comparing counts-per-cell is equivalent to
                # comparing counts-per-area, without needing bbox areas.
                interior_density = interior_pts / float(interior_count)
                wall_density = wall_pts / float(occupied_cells)
                interior_column_empty = interior_density <= interior_column_density_ratio * wall_density
        # else: the top slice covers (almost) the whole object height -- no
        # lower band exists to cross-check against, so trust the 2D result.

    return GraspFeatures(
        z_min=float(z_min),
        z_max=float(z_max),
        height_mm=height_mm,
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        footprint_center_xy=(float(mean_xy[0]), float(mean_xy[1])),
        major_axis_xy=(float(major_axis[0]), float(major_axis[1])),
        minor_axis_xy=(float(minor_axis[0]), float(minor_axis[1])),
        major_length_mm=float(major_len),
        minor_length_mm=float(minor_len),
        solidity=float(solidity),
        has_cavity=bool(has_cavity),
        inner_diameter_mm=inner_diameter_mm,
        wall_thickness_mm=wall_thickness_mm,
        rim_z=float(z_max),
        interior_column_empty=bool(interior_column_empty),
    )


def _fit_confidence(value: float, limit: float) -> float:
    """Heuristic 0.3..0.95 confidence: comfortably under `limit` -> high,
    close to `limit` -> lower. Not a calibrated probability, just a
    monotonic "how much margin did we have" signal."""
    if limit <= 0:
        return 0.3
    ratio = value / limit
    conf = 1.0 - 0.6 * max(0.0, (ratio - 0.3)) / 0.7
    return float(min(0.95, max(0.3, conf)))


def classify_grasp(
    points: np.ndarray,
    *,
    gripper_max_width_mm: float = DEFAULT_GRIPPER_MAX_WIDTH_MM,
    top_slice_frac: float = 0.2,
    grid_cells: int = 40,
    min_cavity_cells: int = 4,
    min_finger_clearance_mm: float = 15.0,
    tall_ratio: float = 1.3,
    max_hollow_solidity: float = 0.7,
    min_cavity_diameter_frac: float = 0.35,
    interior_column_density_ratio: float = 0.3,
) -> GraspAffordance:
    """Classify the grasp TYPE + parameters a single object's point cloud
    affords. See the module docstring for the input contract and the
    "does not check executability" caveat.

    Decision heuristic (in priority order; thresholds are the keyword params
    above, all overridable per call):

    1. INSIDE_OUTSIDE -- an open-container annulus is found in the top-slice
       occupancy grid (`has_cavity`), its inner opening is wide enough for a
       finger (`inner_diameter_mm >= min_finger_clearance_mm`), and the rim's
       wall fits between the pads (`0 < wall_thickness_mm <= gripper_max_width_mm`).
       Guarded by three robustness gates against a sparse-sampled SOLID
       object (e.g. polar-sampled point clouds, which are naturally sparse
       near their own center) reading as a false annulus: top-slice solidity
       must be low (`solidity <= max_hollow_solidity`), the opening must
       span a real fraction of the footprint (`inner_diameter_mm >=
       min_cavity_diameter_frac * minor_length_mm`), AND -- the real 3D
       check -- the candidate cavity's XY column must be genuinely empty
       BELOW the top slice too, not just sparsely sampled there
       (`interior_column_empty`, from `compute_features`'s point-density
       comparison against the surrounding wall; threshold
       `interior_column_density_ratio`). Only if all hold: one pad goes
       inside the rim, one outside; the pads span the wall thickness,
       closing along a horizontal radial direction, approaching from above
       (you lower the inner pad down into the opening).

    2. SIDE -- else if the object is tall relative to its footprint
       (`height_mm > tall_ratio * minor_length_mm`) and its minor footprint
       width fits the gripper (`minor_length_mm <= gripper_max_width_mm`):
       approach horizontally (along the major axis), pinch across the minor
       axis at mid-height.

    3. TOP_DOWN -- else if the minor footprint width fits the gripper: a
       flat/compact object, approach straight down (-Z) at the centroid,
       pads close along the minor axis (yaw aligned to it).

    4. Fallback -- the minor width exceeds `gripper_max_width_mm` in every
       branch above: return the best-guess type (SIDE if tall, else
       TOP_DOWN) with LOW confidence and `note="too wide for this gripper"`.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 8:
        raise ValueError("classify_grasp needs an (N, 3) point cloud with N >= 8 points")

    feats = compute_features(
        pts,
        top_slice_frac=top_slice_frac,
        grid_cells=grid_cells,
        min_cavity_cells=min_cavity_cells,
        interior_column_density_ratio=interior_column_density_ratio,
    )

    minor = feats.minor_length_mm
    height = feats.height_mm
    cx, cy, cz = feats.centroid
    fx, fy = feats.footprint_center_xy
    maj_ax = feats.major_axis_xy
    min_ax = feats.minor_axis_xy

    fits_gripper = minor <= gripper_max_width_mm
    is_tall = (height > tall_ratio * minor) if minor > 1e-6 else False

    # --- 1. INSIDE_OUTSIDE: open-container annulus ---
    # Robustness gates so a solid object with a sparse-sampled centre (common
    # with polar/real depth clouds) is NOT misread as hollow:
    #  - low top-slice solidity (a real rim is a thin ring, not a filled disk),
    #  - the opening spans a real fraction of the footprint, not a tiny gap.
    genuinely_hollow = feats.solidity <= max_hollow_solidity
    opening_is_large = (
        feats.inner_diameter_mm >= min_cavity_diameter_frac * feats.minor_length_mm
    )
    if (
        feats.has_cavity
        and genuinely_hollow
        and opening_is_large
        and feats.interior_column_empty
        and feats.inner_diameter_mm >= min_finger_clearance_mm
        and 0.0 < feats.wall_thickness_mm <= gripper_max_width_mm
    ):
        reach = feats.inner_diameter_mm / 2.0 + feats.wall_thickness_mm / 2.0
        center = (fx + min_ax[0] * reach, fy + min_ax[1] * reach, feats.rim_z)
        confidence = _fit_confidence(feats.wall_thickness_mm, gripper_max_width_mm)
        return GraspAffordance(
            grasp_type=GraspType.INSIDE_OUTSIDE,
            approach=(0.0, 0.0, -1.0),
            grasp_axis=(min_ax[0], min_ax[1], 0.0),
            width_mm=feats.wall_thickness_mm,
            center=center,
            confidence=confidence,
            rim_z=feats.rim_z,
            wall_thickness_mm=feats.wall_thickness_mm,
            inner_diameter_mm=feats.inner_diameter_mm,
        )

    # --- 2. SIDE: tall and fits between the pads ---
    if is_tall and fits_gripper:
        center = (cx, cy, (feats.z_min + feats.z_max) / 2.0)
        confidence = _fit_confidence(minor, gripper_max_width_mm)
        return GraspAffordance(
            grasp_type=GraspType.SIDE,
            approach=(maj_ax[0], maj_ax[1], 0.0),
            grasp_axis=(min_ax[0], min_ax[1], 0.0),
            width_mm=minor,
            center=center,
            confidence=confidence,
        )

    # --- 3. TOP_DOWN: flat/compact and fits between the pads ---
    if fits_gripper:
        confidence = _fit_confidence(minor, gripper_max_width_mm)
        return GraspAffordance(
            grasp_type=GraspType.TOP_DOWN,
            approach=(0.0, 0.0, -1.0),
            grasp_axis=(min_ax[0], min_ax[1], 0.0),
            width_mm=minor,
            center=(cx, cy, cz),
            confidence=confidence,
        )

    # --- 4. Fallback: too wide for this gripper in every branch above ---
    fallback_type = GraspType.SIDE if is_tall else GraspType.TOP_DOWN
    if fallback_type is GraspType.SIDE:
        approach = (maj_ax[0], maj_ax[1], 0.0)
        center = (cx, cy, (feats.z_min + feats.z_max) / 2.0)
    else:
        approach = (0.0, 0.0, -1.0)
        center = (cx, cy, cz)
    return GraspAffordance(
        grasp_type=fallback_type,
        approach=approach,
        grasp_axis=(min_ax[0], min_ax[1], 0.0),
        width_mm=minor,
        center=center,
        confidence=0.15,
        note="too wide for this gripper",
    )


# --------------------------------------------------------------------------
# PCD parsing stub -- documents the Viam -> numpy step; NOT used by, or
# required for, classify_grasp()/compute_features() above.
# --------------------------------------------------------------------------


def points_from_pcd(pcd: Any) -> np.ndarray:
    """Best-effort helper turning one `get_object_point_clouds()` result's
    point cloud into an (N, 3) numpy array.

    Viam's vision service returns, per detected object, a
    `viam.services.vision.PointCloudObject` whose `.point_cloud` is raw PCD
    (Point Cloud Data) format bytes -- typically `DATA ascii` for small debug
    clouds or `DATA binary`/`binary_compressed` for real sensor output, with
    a `FIELDS x y z ...` header (often `rgb`/`rgba` too).

    This stub accepts either the `PointCloudObject` itself (via its
    `.point_cloud` attribute) or raw `bytes`/`bytearray`, and implements only
    the simple ASCII case directly (no extra dependency needed) since that's
    enough to unit-test the parsing contract. For `binary`/`binary_compressed`
    PCD (what a real depth camera will actually emit), parse with a proper
    library instead -- e.g. `open3d.io.read_point_cloud` or `pypcd4` -- and
    feed the resulting `(N, 3)` array straight into `classify_grasp()`.
    """
    raw = getattr(pcd, "point_cloud", pcd)
    if isinstance(raw, np.ndarray):
        return np.asarray(raw, dtype=np.float64)
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("points_from_pcd expects PCD bytes, a PointCloudObject, or an (N,3) array")

    text_head = raw[:512].decode("ascii", errors="ignore")
    if "DATA ascii" not in text_head:
        raise NotImplementedError(
            "points_from_pcd only implements the ASCII PCD path; for binary/"
            "binary_compressed PCD (what a real camera driver emits), parse "
            "with open3d.io.read_point_cloud or pypcd4 and pass the resulting "
            "(N, 3) array to classify_grasp() directly."
        )

    text = raw.decode("ascii", errors="ignore")
    lines = text.splitlines()
    fields: list[str] = []
    data_idx = None
    for i, line in enumerate(lines):
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("DATA"):
            data_idx = i + 1
            break
    if data_idx is None or not fields:
        raise ValueError("malformed ASCII PCD: missing FIELDS/DATA header")

    xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    rows = []
    for line in lines[data_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rows.append((float(parts[xi]), float(parts[yi]), float(parts[zi])))
    return np.asarray(rows, dtype=np.float64)
