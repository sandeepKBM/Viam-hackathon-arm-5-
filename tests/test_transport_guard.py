"""Offline unit tests for components/transport_guard.py.

Purely synthetic ObjectBox extents -- no robot, no camera, no point cloud.
"""

import numpy as np
import pytest

from components.transport_guard import (
    ObjectBox,
    TransportBlockedError,
    box_from_points,
    plan_transport,
    safe_transport_height,
    transport_path_clear,
)

# A "table" object footprint far from the transport corridor used below.
OFF_CORRIDOR = ObjectBox(cx=1000.0, cy=1000.0, half_x=20.0, half_y=20.0, z_bottom=0.0, z_top=200.0)


def make_held(height: float, half: float = 25.0) -> ObjectBox:
    """A held object resting (pre-grasp) with its bottom at z=0."""
    return ObjectBox(cx=0.0, cy=0.0, half_x=half, half_y=half, z_bottom=0.0, z_top=height)


def make_obstacle(cx: float, cy: float, top: float, half: float = 25.0) -> ObjectBox:
    return ObjectBox(cx=cx, cy=cy, half_x=half, half_y=half, z_bottom=0.0, z_top=top)


# ---------------------------------------------------------------------------
# box_from_points
# ---------------------------------------------------------------------------


def test_box_from_points_axis_aligned_extents():
    pts = np.array(
        [
            [10.0, -5.0, 0.0],
            [30.0, 15.0, 40.0],
            [20.0, 5.0, 20.0],
        ]
    )
    box = box_from_points(pts)
    assert box.cx == pytest.approx(20.0)
    assert box.cy == pytest.approx(5.0)
    assert box.half_x == pytest.approx(10.0)
    assert box.half_y == pytest.approx(10.0)
    assert box.z_bottom == pytest.approx(0.0)
    assert box.z_top == pytest.approx(40.0)
    assert box.height == pytest.approx(40.0)


def test_box_from_points_rejects_empty():
    with pytest.raises(ValueError):
        box_from_points(np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# safe_transport_height
# ---------------------------------------------------------------------------


def test_safe_height_no_obstacles_is_pick_height():
    held = make_held(height=50.0)
    z = safe_transport_height(held, [], grip_z=200.0, margin_mm=20.0)
    assert z == pytest.approx(200.0)


def test_safe_height_clears_tallest_obstacle_with_margin():
    held = make_held(height=50.0)
    obstacles = [make_obstacle(100.0, 0.0, top=80.0), make_obstacle(200.0, 0.0, top=120.0)]
    z = safe_transport_height(held, obstacles, grip_z=90.0, margin_mm=20.0)
    # required = tallest_top + margin + held.height = 120 + 20 + 50
    assert z == pytest.approx(190.0)
    # the held object's bottom at this TCP height clears the tallest obstacle
    held_bottom = z - held.height
    assert held_bottom - 120.0 == pytest.approx(20.0)


def test_safe_height_never_below_pick_height():
    held = make_held(height=10.0)
    obstacles = [make_obstacle(0.0, 0.0, top=5.0)]
    z = safe_transport_height(held, obstacles, grip_z=500.0, margin_mm=20.0)
    assert z == pytest.approx(500.0)


def test_taller_held_object_needs_higher_tcp_for_same_obstacle():
    """A tall held object needs a higher TCP than a short one to clear the
    same obstacle -- the held object's own height enters additively."""
    obstacles = [make_obstacle(0.0, 0.0, top=100.0)]
    short_held = make_held(height=20.0)
    tall_held = make_held(height=80.0)
    z_short = safe_transport_height(short_held, obstacles, grip_z=50.0, margin_mm=15.0)
    z_tall = safe_transport_height(tall_held, obstacles, grip_z=50.0, margin_mm=15.0)
    assert z_tall > z_short
    assert (z_tall - z_short) == pytest.approx(tall_held.height - short_held.height)


# ---------------------------------------------------------------------------
# transport_path_clear
# ---------------------------------------------------------------------------


def test_path_clear_over_short_obstacle_at_safe_height():
    held = make_held(height=50.0)
    short_obstacle = make_obstacle(150.0, 0.0, top=60.0)  # sits ON the corridor, but short
    obstacles = [short_obstacle]
    z = safe_transport_height(held, obstacles, grip_z=90.0, margin_mm=20.0)
    clear, colliding = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        obstacles,
        transport_z=z,
        tcp_to_object_bottom=held.height,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert clear
    assert colliding == []


def test_tall_obstacle_in_corridor_flagged_at_naive_low_height():
    held = make_held(height=50.0)
    tall_obstacle = make_obstacle(150.0, 0.0, top=300.0)  # squarely in the corridor, tall
    obstacles = [tall_obstacle]
    # A height that would be fine for an EMPTY gripper (well above the
    # obstacle's footprint at the gripper's own z) but not for the held
    # object hanging below it.
    naive_height = 120.0
    clear, colliding = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        obstacles,
        transport_z=naive_height,
        tcp_to_object_bottom=held.height,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert not clear
    assert tall_obstacle in colliding


def test_tall_obstacle_in_corridor_is_clear_once_lifted_high_enough():
    held = make_held(height=50.0)
    tall_obstacle = make_obstacle(150.0, 0.0, top=300.0)
    obstacles = [tall_obstacle]
    z = safe_transport_height(held, obstacles, grip_z=90.0, margin_mm=20.0)
    clear, colliding = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        obstacles,
        transport_z=z,
        tcp_to_object_bottom=held.height,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert clear
    assert colliding == []


def test_obstacle_off_corridor_is_ignored():
    held = make_held(height=50.0)
    obstacles = [OFF_CORRIDOR]
    clear, colliding = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        obstacles,
        transport_z=90.0,  # low height that WOULD collide if it were in the corridor
        tcp_to_object_bottom=held.height,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert clear
    assert colliding == []


def test_short_obstacle_under_corridor_but_below_held_object_is_ignored():
    held = make_held(height=50.0)
    # In the corridor's XY footprint, but its top sits well below the held
    # object's swept underside even without extra lift.
    short_obstacle = make_obstacle(150.0, 0.0, top=10.0)
    clear, colliding = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        [short_obstacle],
        transport_z=90.0,
        tcp_to_object_bottom=held.height,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert clear
    assert colliding == []


# ---------------------------------------------------------------------------
# plan_transport
# ---------------------------------------------------------------------------


def test_plan_transport_returns_lift_move_lower_waypoints():
    held = make_held(height=50.0)
    obstacles = [make_obstacle(150.0, 0.0, top=60.0)]
    waypoints = plan_transport(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        obstacles,
        pick_z=90.0,
        gripper_half_width_mm=20.0,
    )
    assert len(waypoints) == 3
    lift, move, lower = waypoints
    assert lift.pose["x"] == pytest.approx(0.0) and lift.pose["y"] == pytest.approx(0.0)
    assert move.pose["x"] == pytest.approx(300.0) and move.pose["y"] == pytest.approx(0.0)
    assert lift.pose["z"] == pytest.approx(move.pose["z"])
    assert lower.pose["z"] == pytest.approx(90.0)
    # transport height clears the obstacle
    assert lift.pose["z"] > obstacles[0].z_top
    for wp in waypoints:
        assert wp.joints is None


def test_plan_transport_raises_height_past_naive_safe_height_for_corridor_obstacle():
    held = make_held(height=50.0)
    # Tall obstacle exactly in the corridor: still must be cleared even
    # from the naive `safe_transport_height` (which is already sufficient
    # under its own top-down-grasp assumption -- see the next test for
    # the case where the REAL grasp offset is larger than that
    # assumption and the naive height alone falls short).
    tall_obstacle = make_obstacle(150.0, 0.0, top=300.0)
    waypoints = plan_transport(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        [tall_obstacle],
        pick_z=90.0,
        gripper_half_width_mm=20.0,
    )
    transport_z = waypoints[1].pose["z"]
    clear, colliding = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        [tall_obstacle],
        transport_z=transport_z,
        tcp_to_object_bottom=held.height,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert clear
    assert colliding == []


def test_plan_transport_raises_past_naive_height_when_real_grasp_offset_is_larger():
    """If the object actually hangs lower below the TCP than
    `safe_transport_height`'s top-down assumption (`held.height`) -- e.g.
    it was grasped partway down rather than at its very top -- the naive
    safe height alone can be insufficient for a tall obstacle in the
    corridor; `plan_transport` must detect that via `transport_path_clear`
    and raise the transport height further.
    """
    held = make_held(height=50.0)
    tall_obstacle = make_obstacle(150.0, 0.0, top=300.0)
    real_offset = held.height + 80.0  # object hangs 80mm lower than assumed

    naive_z = safe_transport_height(held, [tall_obstacle], grip_z=90.0, margin_mm=20.0)
    naive_clear, _ = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        [tall_obstacle],
        transport_z=naive_z,
        tcp_to_object_bottom=real_offset,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert not naive_clear  # confirms the naive height is genuinely insufficient here

    waypoints = plan_transport(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        [tall_obstacle],
        pick_z=90.0,
        gripper_half_width_mm=20.0,
        tcp_to_object_bottom=real_offset,
    )
    transport_z = waypoints[1].pose["z"]
    assert transport_z > naive_z

    clear, colliding = transport_path_clear(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        [tall_obstacle],
        transport_z=transport_z,
        tcp_to_object_bottom=real_offset,
        gripper_half_width_mm=20.0,
        margin_mm=10.0,
    )
    assert clear
    assert colliding == []


def test_plan_transport_uses_ik_fn_when_given():
    held = make_held(height=50.0)
    calls = []

    def fake_ik(pose):
        calls.append(pose)
        return [pose["x"], pose["y"], pose["z"], 0.0, 0.0, 0.0]

    waypoints = plan_transport(
        (0.0, 0.0),
        (300.0, 0.0),
        held,
        [],
        pick_z=90.0,
        gripper_half_width_mm=20.0,
        ik_fn=fake_ik,
    )
    assert len(calls) == 3
    for wp in waypoints:
        assert wp.joints is not None
        assert len(wp.joints) == 6


def test_plan_transport_blocked_raises_documented_failure():
    held = make_held(height=50.0)
    tall_obstacle = make_obstacle(150.0, 0.0, top=300.0)
    # A real grasp offset far larger than the naive top-down assumption,
    # combined with a tight height ceiling, so no reachable height (within
    # the attempt/height budget given) clears the corridor -- the
    # documented "clear FAILURE" case.
    with pytest.raises(TransportBlockedError):
        plan_transport(
            (0.0, 0.0),
            (300.0, 0.0),
            held,
            [tall_obstacle],
            pick_z=90.0,
            gripper_half_width_mm=20.0,
            tcp_to_object_bottom=held.height + 10_000.0,
            max_transport_z=200.0,
            max_raise_attempts=3,
            raise_step_mm=25.0,
        )
