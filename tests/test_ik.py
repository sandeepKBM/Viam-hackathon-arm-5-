"""Offline verification of components/ik.py -- the fast planner's real,
local IK source (see components/ik.py's module docstring for why: the
`viam-sdk` pinned in this repo's `.venv` exposes no
`compute_inverse_kinematics`, so `fast_planner.make_ik_fn(arm)` alone
always raises; `components.ik` + `fast_planner.make_local_ik_fn` fill that
gap with a numerical solver against the bundled xArm5 URDF).

`ikpy` is an optional dependency (lazy-imported inside components/ik.py);
`pytest.importorskip` below makes this whole module skip cleanly if it
isn't installed, matching every other optional-dependency test in this
repo's style, but it IS listed in requirements.txt and IS installed in
this repo's .venv, so in the normal case these tests run and must pass.
"""

from __future__ import annotations

import pytest

ikpy = pytest.importorskip("ikpy", reason="components.ik's local IK needs the optional ikpy package")

from components import fast_planner  # noqa: E402
from components.constants import MIN_Z, PICK_ORIENTATION, TRAVEL_Z, WORKSPACE_CORNERS  # noqa: E402
from components.ik import (  # noqa: E402
    NUM_JOINTS,
    ORIENTATION_TOLERANCE_DEG,
    POSITION_TOLERANCE_MM,
    UnreachablePoseError,
    XArm5IK,
    load_chain,
    make_local_ik_fn,
)

# A handful of reachable poses spanning the taught tabletop workspace, at
# both the pick floor and travel height, all using the one fixed top-down
# grasp orientation the whole fast planner ever asks for.
_CENTROID_X = sum(c[0] for c in WORKSPACE_CORNERS) / len(WORKSPACE_CORNERS)
_CENTROID_Y = sum(c[1] for c in WORKSPACE_CORNERS) / len(WORKSPACE_CORNERS)


def _inward(corner, pull=0.2):
    """Nudge a workspace corner toward the centroid so the sample point is
    safely interior (a literal corner can sit right at the reach limit)."""
    cx, cy = corner
    return (
        cx * (1 - pull) + _CENTROID_X * pull,
        cy * (1 - pull) + _CENTROID_Y * pull,
    )


def _pose(x: float, y: float, z: float) -> dict:
    pose = {"x": x, "y": y, "z": z}
    pose.update(PICK_ORIENTATION)
    return pose


REACHABLE_POSES = [
    _pose(_CENTROID_X, _CENTROID_Y, MIN_Z),
    _pose(_CENTROID_X, _CENTROID_Y, TRAVEL_Z),
] + [_pose(*_inward(c), MIN_Z) for c in WORKSPACE_CORNERS]


@pytest.fixture(scope="module")
def solver() -> XArm5IK:
    return XArm5IK()


# ---------------------------------------------------------------------------
# Chain loading
# ---------------------------------------------------------------------------


def test_load_chain_has_exactly_five_active_joints():
    chain = load_chain()
    active = [link.joint_type != "fixed" for link in chain.links]
    assert sum(active) == NUM_JOINTS == 5


def test_load_chain_missing_urdf_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chain(str(tmp_path / "does_not_exist.urdf"))


# ---------------------------------------------------------------------------
# FK(IK(pose)) ~= pose, within tolerance, for reachable poses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pose", REACHABLE_POSES)
def test_fk_of_ik_matches_target_pose_within_tolerance(solver: XArm5IK, pose: dict):
    joints_deg = solver.solve(pose)
    assert len(joints_deg) == NUM_JOINTS

    fk_pose = solver.forward(joints_deg)
    pos_err_mm = (
        (fk_pose["x"] - pose["x"]) ** 2
        + (fk_pose["y"] - pose["y"]) ** 2
        + (fk_pose["z"] - pose["z"]) ** 2
    ) ** 0.5
    assert pos_err_mm <= POSITION_TOLERANCE_MM

    # Wrist (+Z) axis alignment, in degrees -- theta/roll is deliberately
    # not solved for (see components/ik.py's module docstring).
    import numpy as np

    target_z = np.array([pose["o_x"], pose["o_y"], pose["o_z"]])
    target_z = target_z / np.linalg.norm(target_z)
    fk_z = np.array([fk_pose["o_x"], fk_pose["o_y"], fk_pose["o_z"]])
    fk_z = fk_z / np.linalg.norm(fk_z)
    ang_err_deg = np.degrees(np.arccos(np.clip(np.dot(target_z, fk_z), -1.0, 1.0)))
    assert ang_err_deg <= ORIENTATION_TOLERANCE_DEG

    # solve() itself measured (and self-consistently agreed on) the same
    # residuals when it accepted this solution.
    assert solver.last_diagnostics is not None
    assert solver.last_diagnostics.position_error_mm <= POSITION_TOLERANCE_MM
    assert solver.last_diagnostics.orientation_error_deg <= ORIENTATION_TOLERANCE_DEG


def test_solve_is_deterministic_given_the_same_seed(solver: XArm5IK):
    solver.reset_seed()
    a = solver.solve(REACHABLE_POSES[0])
    solver.reset_seed()
    b = solver.solve(REACHABLE_POSES[0])
    assert a == pytest.approx(b, abs=1e-6)


# ---------------------------------------------------------------------------
# Unreachable poses raise a clear, typed error -- never a silently-bad joint
# target.
# ---------------------------------------------------------------------------


def test_unreachable_pose_raises_clearly(solver: XArm5IK):
    far_away_pose = _pose(5000.0, 5000.0, 500.0)  # 5m out -- nowhere near a ~700mm-reach arm
    with pytest.raises(UnreachablePoseError):
        solver.solve(far_away_pose)


def test_pick_at_travel_height_near_corner_can_be_unreachable(solver: XArm5IK):
    """Not a solver bug: holding a literal far workspace corner's XY at the
    full TRAVEL_Z height (home pose height) while pointing straight down
    genuinely exceeds a 5-DOF xArm5's reach envelope -- horizontal reach and
    lift height aren't independent. `plan_pick_waypoints`'s "lift/move at
    travel_z, descend to pick_z" shape means a target only needs to be
    reachable at pick_z; run_pick_and_place.py's summary surfaces this
    error per-object rather than crashing the whole run (see RetryController
    -> pipeline.run_task's per-step try/except)."""
    solver.reset_seed()
    corner_x, corner_y = _inward(WORKSPACE_CORNERS[1], pull=0.2)
    pose = _pose(corner_x, corner_y, TRAVEL_Z)
    with pytest.raises(UnreachablePoseError):
        solver.solve(pose)


def test_degenerate_orientation_vector_raises_value_error(solver: XArm5IK):
    pose = {"x": _CENTROID_X, "y": _CENTROID_Y, "z": MIN_Z, "o_x": 0.0, "o_y": 0.0, "o_z": 0.0, "theta": 0.0}
    with pytest.raises(ValueError):
        solver.solve(pose)


# ---------------------------------------------------------------------------
# fast_planner integration: make_local_ik_fn is a drop-in real ik_fn
# ---------------------------------------------------------------------------


def test_make_local_ik_fn_matches_ik_fn_shape():
    ik_fn = fast_planner.make_local_ik_fn()
    joints = ik_fn(REACHABLE_POSES[0])
    assert len(joints) == NUM_JOINTS
    assert all(isinstance(j, float) for j in joints)


def test_make_local_ik_fn_drives_plan_pick_waypoints():
    ik_fn = fast_planner.make_local_ik_fn()
    start_xy = (_CENTROID_X, _CENTROID_Y)
    target_x, target_y = _inward(WORKSPACE_CORNERS[0])
    waypoints = fast_planner.plan_pick_waypoints([], start_xy, (target_x, target_y), ik_fn)
    assert len(waypoints) == 3
    for wp in waypoints:
        assert len(wp) == NUM_JOINTS


def test_make_local_ik_fn_drives_a_full_task_plan():
    from components.skills import PickParams, SkillCall

    class FakeObj:
        def __init__(self, x, y, color):
            self.x, self.y, self.color = x, y, color

    ik_fn = fast_planner.make_local_ik_fn()
    # A pick's waypoints include a lift/move at TRAVEL_Z, not just the
    # pick_z descent -- pull well inward of the corner (unlike the
    # MIN_Z-only REACHABLE_POSES above) so this target is reachable at
    # BOTH heights (a literal workspace corner, held up at TRAVEL_Z with
    # the fixed top-down orientation, is a genuine reach-limit case for a
    # 5-DOF arm -- see test_pick_at_travel_height_near_corner_can_be_unreachable).
    tx, ty = _inward(WORKSPACE_CORNERS[1], pull=0.6)
    seq = [SkillCall(skill="pick", params=PickParams(object=FakeObj(tx, ty, "red")))]
    steps = fast_planner.plan_task_waypoints(seq, [], (_CENTROID_X, _CENTROID_Y), ik_fn)
    assert len(steps) == 1
    # up/over/down/lift-after-grab (4 IK'd waypoints) + the taught bin1 drop
    assert len(steps[0].waypoints) == 5
    for wp in steps[0].waypoints[:4]:
        assert len(wp) == NUM_JOINTS
