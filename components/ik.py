"""Local numerical IK for the xArm5 -- the "real IK source" `fast_planner`
needs.

THE PROBLEM
-----------
`components/fast_planner.py` needs an `ik_fn(pose) -> joints` to turn each
deterministic Cartesian waypoint (lift / over / descend) into a joint
target. Its `make_ik_fn(arm)` wraps Viam's `compute_inverse_kinematics`, but
the `viam-sdk` pinned in this repo's `.venv` doesn't expose that RPC on the
`Arm` component (only `get_kinematics`, and `motion` only exposes
`get_pose`) -- so the fast path had no real IK source, only a mock for
tests.

This module fills that gap with a local, offline, numerical IK solver built
on `ikpy` (https://github.com/Phylliade/ikpy) against the bundled xArm5
URDF (`assets/xarm5.urdf` -- an already xacro-expanded copy of the xArm5 +
gripper description also used by the `xarm5_sim` MuJoCo stack). No robot,
no network call, no RRT: same "pure function over a fixed kinematic model"
shape as the rest of `fast_planner`.

`ikpy` is an OPTIONAL dependency (listed in requirements.txt, but imported
lazily here) so the rest of the repo -- including `components/fast_planner`
itself -- still imports fine in an environment that hasn't installed it;
only actually building/using a local IK solver requires it.

KINEMATIC CHAIN
----------------
`assets/xarm5.urdf` describes: `link_base -> (joint1..joint5, revolute) ->
link5 -> (joint_eef, fixed) -> link_eef -> (gripper_fix, fixed) ->
xarm_gripper_base_link -> (joint_tcp, fixed, +172mm along local Z) ->
link_tcp`. `_CHAIN_BASE_ELEMENTS` below walks exactly that path (ikpy's
`Chain.from_urdf_file` wants the alternating link/joint name list from base
to tip, not just link names), so the solved chain's tip IS the gripper's
TCP, not just the bare wrist flange. The gripper's own finger joints
(`drive_joint`, `left_finger_joint`, ...) are a separate branch off
`xarm_gripper_base_link` and never appear on this base->tip path, so they
never enter the chain at all.

ORIENTATION
-----------
Every pose `fast_planner` solves for uses the SAME fixed top-down grasp
orientation (`components.constants.PICK_ORIENTATION`), expressed the way
Viam's `Pose` proto does: a unit vector `(o_x, o_y, o_z)` -- the direction
the end effector's own +Z axis should point, in the base/world frame -- plus
a `theta` (rotation about that vector). A 5-DOF arm has 5 joints for (in
general) 6 pose DOF (3 position + 3 orientation), so it cannot satisfy an
arbitrary full orientation target; but it CAN independently point its wrist
axis in an arbitrary direction (2 orientation DOF) while satisfying a 3-DOF
position target -- exactly what a symmetric two-finger top-down grasp
needs (the roll, `theta`, doesn't matter for a symmetric gripper closing
around a block, and isn't solvable with only 5 joints anyway). So this
module targets the TCP's local +Z axis at `(o_x, o_y, o_z)` via ikpy's
`orientation_mode="Z"` (constrains only that axis, not the full frame) and
does not attempt to match `theta`.

TOLERANCE / UNREACHABLE POSES
------------------------------
`ikpy`'s optimizer always returns SOME joint solution, even for a pose that
can't actually be reached -- it just won't be close. `XArm5IK.solve` always
re-derives forward kinematics from its own answer and measures the residual
position error (mm) and wrist-axis angular error (degrees); if either
exceeds `position_tolerance_mm` / `orientation_tolerance_deg` it retries
once from a fixed "home" seed (a different starting branch can converge
much better -- see the module test/verification notes), and if that still
misses, raises `UnreachablePoseError` with the measured residuals rather
than silently handing back a bad joint target.
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "xarm5.urdf"

# Alternating link/joint names from the URDF's `link_base` to its gripper
# TCP (`link_tcp`) -- see the "KINEMATIC CHAIN" docstring section above.
_CHAIN_BASE_ELEMENTS = [
    "link_base",
    "joint1",
    "link1",
    "joint2",
    "link2",
    "joint3",
    "link3",
    "joint4",
    "link4",
    "joint5",
    "link5",
    "joint_eef",
    "link_eef",
    "gripper_fix",
    "xarm_gripper_base_link",
    "joint_tcp",
    "link_tcp",
]

# How many *active* (revolute) joints the chain above resolves -- joint1..5.
NUM_JOINTS = 5

POSITION_TOLERANCE_MM = float(os.environ.get("IK_POSITION_TOLERANCE_MM", 5.0))
ORIENTATION_TOLERANCE_DEG = float(os.environ.get("IK_ORIENTATION_TOLERANCE_DEG", 5.0))

# A reasonable default optimizer seed (degrees, joint1..5) -- the taught
# HOME_JOINTS from components.constants, duplicated here (rather than
# imported) so this module has no hard dependency on components.constants
# and stays usable completely standalone (e.g. a future non-tabletop rig).
_HOME_SEED_JOINTS_DEG = [
    -17.287645921409155,
    21.317320927933277,
    -35.8769517287989,
    0.467687971989073,
    -59.234583350503804,
]

Pose = Dict[str, float]
Joints = List[float]
IkFn = Callable[[Pose], Sequence[float]]


class UnreachablePoseError(RuntimeError):
    """Raised by `XArm5IK.solve` when no joint solution was found that
    reaches `pose` within tolerance (see module docstring)."""


def _lazy_ikpy_chain():
    try:
        from ikpy.chain import Chain
    except ImportError as exc:  # pragma: no cover - exercised only w/o ikpy
        raise ImportError(
            "components.ik requires the optional 'ikpy' package for local IK "
            "(pip install ikpy, or `pip install -r requirements.txt`). The "
            "rest of this repo imports fine without it -- only building a "
            "local IK solver (components.ik.load_chain / make_local_ik_fn) "
            "needs it."
        ) from exc
    return Chain


def load_chain(urdf_path: Optional[str] = None):
    """Build an `ikpy.chain.Chain` for the xArm5 from a URDF file (default:
    the bundled `assets/xarm5.urdf`). The chain's `active_links_mask` is set
    so only the 5 revolute arm joints are solved for -- every fixed link on
    the base->TCP path (see module docstring) is inactive, exactly matching
    the fact those links contribute no DOF.
    """
    import numpy as np

    Chain = _lazy_ikpy_chain()
    path = Path(urdf_path) if urdf_path else DEFAULT_URDF_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"components.ik.load_chain: no URDF at {path} -- expected the "
            "bundled assets/xarm5.urdf (copied from xarm5_sim's "
            "xarm5_gripper_raw.urdf), or pass an explicit urdf_path."
        )
    with warnings.catch_warnings():
        # ikpy warns "fixed link set as active" for every fixed link in
        # base_elements before we get a chance to fix active_links_mask
        # below -- expected and harmless here, so silence it.
        warnings.simplefilter("ignore")
        chain = Chain.from_urdf_file(str(path), base_elements=_CHAIN_BASE_ELEMENTS)
    chain.active_links_mask = np.array([link.joint_type != "fixed" for link in chain.links])
    return chain


def load_chain_from_urdf_bytes(urdf_bytes: bytes):
    """Same as `load_chain`, from in-memory URDF bytes (e.g. fetched at
    runtime from `arm.get_kinematics()` -- see `urdf_bytes_from_arm`) rather
    than a path on disk."""
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".urdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(urdf_bytes)
        return load_chain(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def urdf_bytes_from_arm(arm: Any) -> Optional[bytes]:
    """OPTIONAL runtime fallback: try to fetch a URDF straight from the
    connected arm's own `get_kinematics()` (Viam's Arm component does
    expose this, per the task brief -- unlike `compute_inverse_kinematics`),
    for use instead of the bundled `assets/xarm5.urdf`.

    Returns the URDF bytes if the arm both exposes `get_kinematics()` AND
    reports its format as URDF; returns `None` (not an exception) for any
    other case -- no kinematics endpoint, an RPC error, or a non-URDF
    format (Viam's Spatial-Vector-Algebra format isn't something this
    ikpy-based module knows how to parse) -- so a caller can always fall
    back to the bundled URDF without special-casing failures. The bundled
    URDF is the primary, offline-verifiable path; this is a best-effort
    extra for a live robot whose kinematics might differ from the bundled
    file (e.g. a different xArm5 mounting/tool).
    """
    get_kinematics = getattr(arm, "get_kinematics", None)
    if get_kinematics is None:
        raw = getattr(arm, "_arm", None)
        get_kinematics = getattr(raw, "get_kinematics", None) if raw is not None else None
    if get_kinematics is None:
        return None
    try:
        from viam.components.arm import KinematicsFileFormat

        result = await get_kinematics()
        fmt, data = result[0], result[1]
        if fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF:
            return bytes(data)
        return None
    except Exception:
        return None


@dataclass
class SolveDiagnostics:
    position_error_mm: float
    orientation_error_deg: float
    seed_used: str  # "continuity" or "home"


class XArm5IK:
    """Stateful local IK solver wrapping one `ikpy` chain.

    Not thread-hostile but not carefully lock-free either: a single
    `threading.Lock` serializes `solve()` calls, since ikpy/scipy's
    optimizer isn't documented as reentrant-safe and this repo's own usage
    is single-flight per task anyway (see `fast_planner`'s sequential
    `solve_ik`/`solve_ik_async`).

    Seeding: each `solve()` first seeds the optimizer from the PREVIOUS
    solution (nearby Cartesian waypoints -- e.g. `fast_planner`'s
    lift/over/descend triple -- tend to have nearby joint solutions, so
    this both converges faster and keeps the resulting joint path smooth
    instead of jumping between arbitrary IK branches). If that doesn't
    reach the target within tolerance, it retries once from a fixed
    "home" seed (a different starting branch can succeed where continuity
    seeding from a far-away previous pose gets stuck in a bad local
    optimum -- verified empirically against this repo's taught workspace,
    see tests/test_ik.py) before giving up with `UnreachablePoseError`.
    """

    def __init__(
        self,
        chain: Optional[Any] = None,
        *,
        urdf_path: Optional[str] = None,
        position_tolerance_mm: float = POSITION_TOLERANCE_MM,
        orientation_tolerance_deg: float = ORIENTATION_TOLERANCE_DEG,
        seed_joints_deg: Optional[Sequence[float]] = None,
    ) -> None:
        self.chain = chain if chain is not None else load_chain(urdf_path)
        self.position_tolerance_mm = position_tolerance_mm
        self.orientation_tolerance_deg = orientation_tolerance_deg
        self._active_indices = [i for i, active in enumerate(self.chain.active_links_mask) if active]
        if len(self._active_indices) != NUM_JOINTS:
            raise ValueError(
                f"components.ik.XArm5IK: expected {NUM_JOINTS} active joints in the "
                f"chain, found {len(self._active_indices)} -- wrong URDF/chain?"
            )
        self._home_seed = self._full_seed(seed_joints_deg or _HOME_SEED_JOINTS_DEG)
        self._last_seed = list(self._home_seed)
        self._lock = threading.Lock()
        self.last_diagnostics: Optional[SolveDiagnostics] = None

    # -- seeding / joint-vector plumbing --------------------------------

    def _full_seed(self, joints5_deg: Sequence[float]) -> List[float]:
        import numpy as np

        if len(joints5_deg) != NUM_JOINTS:
            raise ValueError(f"expected {NUM_JOINTS} joint angles, got {len(joints5_deg)}")
        full = [0.0] * len(self.chain.links)
        for idx, deg in zip(self._active_indices, joints5_deg):
            full[idx] = float(np.radians(deg))
        return full

    def _extract_active_deg(self, full_solution: Sequence[float]) -> List[float]:
        import numpy as np

        return [float(np.degrees(full_solution[i])) for i in self._active_indices]

    # -- core solve -------------------------------------------------------

    def _fk_error(self, full_solution, target_m, target_z_unit):
        import numpy as np

        fk = self.chain.forward_kinematics(full_solution)
        pos_err_mm = float(np.linalg.norm(fk[:3, 3] * 1000.0 - target_m * 1000.0))
        z_dot = float(np.clip(np.dot(fk[:3, 2], target_z_unit), -1.0, 1.0))
        ang_err_deg = float(np.degrees(np.arccos(z_dot)))
        return pos_err_mm, ang_err_deg

    def _solve_from_seed(self, target_m, target_z_unit, seed):
        sol = self.chain.inverse_kinematics(
            target_position=target_m,
            target_orientation=target_z_unit,
            orientation_mode="Z",
            initial_position=seed,
        )
        pos_err_mm, ang_err_deg = self._fk_error(sol, target_m, target_z_unit)
        return sol, pos_err_mm, ang_err_deg

    def solve(self, pose: Pose) -> Joints:
        """`ik_fn`-shaped entry point: pose dict (mm + Viam orientation
        vector) -> 5 joint angles (degrees, joint1..joint5 order -- what
        `components.arm.ArmComponent.move_to_joints` /
        `fast_planner.fast_move` expect).

        Raises `UnreachablePoseError` if no joint solution reaches `pose`
        within `position_tolerance_mm` / `orientation_tolerance_deg` (see
        class docstring for the two-seed retry this tries first).
        """
        import numpy as np

        x, y, z = float(pose["x"]), float(pose["y"]), float(pose["z"])
        o = np.array(
            [
                float(pose.get("o_x", 0.0)),
                float(pose.get("o_y", 0.0)),
                float(pose.get("o_z", -1.0)),
            ]
        )
        norm = float(np.linalg.norm(o))
        if norm < 1e-9:
            raise ValueError(f"components.ik.solve: degenerate orientation vector in pose {pose!r}")
        o_unit = o / norm
        target_m = np.array([x, y, z]) / 1000.0

        with self._lock:
            sol, pos_err, ang_err = self._solve_from_seed(target_m, o_unit, self._last_seed)
            seed_used = "continuity"
            if pos_err > self.position_tolerance_mm or ang_err > self.orientation_tolerance_deg:
                sol, pos_err, ang_err = self._solve_from_seed(target_m, o_unit, self._home_seed)
                seed_used = "home"
            if pos_err > self.position_tolerance_mm or ang_err > self.orientation_tolerance_deg:
                raise UnreachablePoseError(
                    f"pose {pose!r} appears unreachable: best residual after "
                    f"continuity+home-seed retry was position_error={pos_err:.2f}mm "
                    f"(tolerance {self.position_tolerance_mm}mm), "
                    f"orientation_error={ang_err:.2f}deg "
                    f"(tolerance {self.orientation_tolerance_deg}deg)"
                )
            self._last_seed = list(sol)
            self.last_diagnostics = SolveDiagnostics(pos_err, ang_err, seed_used)

        return self._extract_active_deg(sol)

    def forward(self, joints_deg: Sequence[float]) -> Pose:
        """Forward kinematics: 5 joint angles (degrees) -> the TCP pose
        `{x, y, z, o_x, o_y, o_z}` (mm + unit wrist-axis vector). Used by
        tests to verify `FK(IK(pose)) ~= pose`; `theta` is not returned
        (this module never solves for it -- see module docstring)."""
        full = self._full_seed(joints_deg)
        fk = self.chain.forward_kinematics(full)
        pos_mm = fk[:3, 3] * 1000.0
        z_axis = fk[:3, 2]
        return {
            "x": float(pos_mm[0]),
            "y": float(pos_mm[1]),
            "z": float(pos_mm[2]),
            "o_x": float(z_axis[0]),
            "o_y": float(z_axis[1]),
            "o_z": float(z_axis[2]),
        }

    def reset_seed(self) -> None:
        """Drop any continuity state and go back to seeding from home --
        call this between unrelated tasks/plans if desired (not required:
        `solve()` already falls back to the home seed on its own whenever
        continuity fails)."""
        self._last_seed = list(self._home_seed)


def make_local_ik_fn(
    urdf_path: Optional[str] = None,
    *,
    chain: Optional[Any] = None,
    **solver_kwargs: Any,
) -> IkFn:
    """Build a synchronous `ik_fn(pose) -> joints` (the exact shape
    `components.fast_planner.plan_pick_waypoints` / `plan_task_waypoints`
    expect) backed by this module's local `ikpy` solver against the bundled
    (or explicitly-given) xArm5 URDF. This is the primary, OFFLINE-usable
    real IK source for the fast planner -- see `fast_planner.make_local_ik_fn`,
    which just forwards here.
    """
    solver = XArm5IK(chain=chain, urdf_path=urdf_path, **solver_kwargs)

    def _ik(pose: Pose) -> Joints:
        return solver.solve(pose)

    _ik.solver = solver  # type: ignore[attr-defined]  # introspection/tests
    return _ik


async def make_local_ik_fn_from_arm(
    arm: Any,
    *,
    fallback_urdf_path: Optional[str] = None,
    **solver_kwargs: Any,
) -> IkFn:
    """Best-effort variant of `make_local_ik_fn` that first tries to source
    the URDF live from `arm.get_kinematics()` (`urdf_bytes_from_arm`) --
    e.g. so a real, individually-calibrated arm's own reported kinematics
    are used instead of the bundled file -- and falls back to the bundled
    `assets/xarm5.urdf` (or `fallback_urdf_path`) whenever that isn't
    available. Async because fetching kinematics from a live arm is a
    network RPC; `make_local_ik_fn` itself stays synchronous for the fully
    offline/bundled-URDF path.
    """
    urdf_bytes = await urdf_bytes_from_arm(arm)
    if urdf_bytes is not None:
        chain = load_chain_from_urdf_bytes(urdf_bytes)
        return make_local_ik_fn(chain=chain, **solver_kwargs)
    return make_local_ik_fn(urdf_path=fallback_urdf_path, **solver_kwargs)
