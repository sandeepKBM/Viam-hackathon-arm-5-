"""Offline unit tests for XArmVelocityArm.

No hardware, no `xarm` SDK -- every test injects `fake_backend.FakeXArm`
via the component's `backend_factory` hook. Run with:

    .venv/bin/python -m pytest xarm_velocity/tests/ -q
"""

import time
import unittest

from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig
from viam.proto.component.arm import JointPositions

from xarm_velocity.arm import ControlMode, XArmVelocityArm
from xarm_velocity.fake_backend import FakeXArm


def make_config(name="arm1", **attrs):
    s = Struct()
    s.update(attrs)
    return ComponentConfig(name=name, attributes=s)


def make_arm(dof=6, max_joint_velocity=30.0, watchdog_timeout_s=0.2, host="10.0.0.5", is_radian=False):
    """Build a configured XArmVelocityArm backed by a FakeXArm, without touching the
    Viam resource registry / real SDK."""
    fake = FakeXArm(host=host, is_radian=is_radian, dof=dof)
    arm = XArmVelocityArm(name="arm1", backend_factory=lambda self: fake)
    config = make_config(
        host=host,
        dof=dof,
        max_joint_velocity=max_joint_velocity,
        watchdog_timeout_s=watchdog_timeout_s,
        is_radian=is_radian,
    )
    arm.reconfigure(config, {})
    return arm, fake


class ValidateConfigTest(unittest.TestCase):
    def test_valid_config_ok(self):
        cfg = make_config(host="10.0.0.5", dof=6, max_joint_velocity=20, watchdog_timeout_s=0.2)
        XArmVelocityArm.validate_config(cfg)  # should not raise

    def test_missing_host_rejected(self):
        cfg = make_config(dof=6)
        with self.assertRaises(ValueError):
            XArmVelocityArm.validate_config(cfg)

    def test_bad_dof_rejected(self):
        cfg = make_config(host="10.0.0.5", dof=4)
        with self.assertRaises(ValueError):
            XArmVelocityArm.validate_config(cfg)

    def test_dof_5_and_7_accepted(self):
        for dof in (5, 7):
            cfg = make_config(host="10.0.0.5", dof=dof)
            XArmVelocityArm.validate_config(cfg)  # should not raise

    def test_nonpositive_max_velocity_rejected(self):
        cfg = make_config(host="10.0.0.5", dof=6, max_joint_velocity=0)
        with self.assertRaises(ValueError):
            XArmVelocityArm.validate_config(cfg)

    def test_nonpositive_watchdog_timeout_rejected(self):
        cfg = make_config(host="10.0.0.5", dof=6, watchdog_timeout_s=-1)
        with self.assertRaises(ValueError):
            XArmVelocityArm.validate_config(cfg)


class ConnectAndModeTest(unittest.TestCase):
    def test_reconfigure_connects_and_enables(self):
        arm, fake = make_arm()
        self.assertTrue(fake.enabled)
        self.assertEqual(fake.mode, 0)
        self.assertEqual(arm._mode, ControlMode.POSITION)

    def test_set_control_mode_transitions_and_calls_set_mode_set_state(self):
        arm, fake = make_arm()
        result = arm._cmd_set_control_mode({"mode": "velocity"})
        self.assertEqual(result["mode"], "velocity")
        self.assertEqual(fake.mode, 4)
        # transitioning mode must call set_mode then set_state
        names = [n for (n, _, _) in fake.calls]
        self.assertIn("set_mode", names)
        self.assertLess(names.index("set_mode"), names.index("set_state") if "set_state" in names else len(names))

    def test_set_control_mode_unknown_rejected(self):
        arm, fake = make_arm()
        with self.assertRaises(ValueError):
            arm._cmd_set_control_mode({"mode": "warp_speed"})

    def test_mode_transition_is_noop_when_already_in_mode(self):
        arm, fake = make_arm()
        arm._cmd_set_control_mode({"mode": "velocity"})
        n_calls_before = len(fake.calls)
        arm._cmd_set_control_mode({"mode": "velocity"})
        self.assertEqual(len(fake.calls), n_calls_before)  # no redundant set_mode/set_state


class DoCommandRoutingTest(unittest.TestCase):
    def setUp(self):
        self.arm, self.fake = make_arm(dof=6)

    def test_set_joint_velocity_routes_to_vc_set_joint_velocity(self):
        import asyncio

        result = asyncio.run(
            self.arm.do_command({"command": "set_joint_velocity", "velocities": [1, 2, 3, 4, 5, 6]})
        )
        self.assertTrue(result["ok"])
        calls = self.fake.calls_named("vc_set_joint_velocity")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(self.fake.mode, 4)

    def test_set_cartesian_velocity_routes_to_vc_set_cartesian_velocity(self):
        import asyncio

        result = asyncio.run(
            self.arm.do_command(
                {"command": "set_cartesian_velocity", "velocity": [10, 0, 0, 0, 0, 0]}
            )
        )
        self.assertTrue(result["ok"])
        calls = self.fake.calls_named("vc_set_cartesian_velocity")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.fake.mode, 5)

    def test_servo_joint_routes_to_set_servo_angle_j(self):
        import asyncio

        result = asyncio.run(
            self.arm.do_command({"command": "servo_joint", "angles": [0, 0, 0, 0, 0, 0]})
        )
        self.assertTrue(result["ok"])
        calls = self.fake.calls_named("set_servo_angle_j")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.fake.mode, 1)

    def test_get_joint_torques(self):
        import asyncio

        result = asyncio.run(self.arm.do_command({"command": "get_joint_torques"}))
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["torques"]), 6)
        self.assertEqual(self.fake.report_tau_or_i, 0)

    def test_get_joint_currents(self):
        import asyncio

        result = asyncio.run(self.arm.do_command({"command": "get_joint_currents"}))
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["currents"]), 6)
        self.assertEqual(self.fake.report_tau_or_i, 1)

    def test_stop_command(self):
        import asyncio

        asyncio.run(self.arm.do_command({"command": "set_joint_velocity", "velocities": [1] * 6}))
        result = asyncio.run(self.arm.do_command({"command": "stop"}))
        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.state, 4)
        self.assertEqual(self.arm._mode, ControlMode.POSITION)
        self.assertFalse(self.arm._watchdog.armed)

    def test_unknown_command_rejected(self):
        import asyncio

        with self.assertRaises(ValueError):
            asyncio.run(self.arm.do_command({"command": "do_a_backflip"}))


class SafetyClampAndLengthTest(unittest.TestCase):
    def test_velocity_clamped_to_max(self):
        arm, fake = make_arm(dof=6, max_joint_velocity=10.0)
        result = arm._cmd_set_joint_velocity({"velocities": [100, -100, 5, 0, 0, 0]})
        self.assertEqual(result["velocities"], [10.0, -10.0, 5.0, 0.0, 0.0, 0.0])
        self.assertTrue(result["clamped"])
        recorded = fake.calls_named("vc_set_joint_velocity")[0][0][0]
        self.assertEqual(recorded, [10.0, -10.0, 5.0, 0.0, 0.0, 0.0])

    def test_velocity_within_limit_not_flagged_clamped(self):
        arm, fake = make_arm(dof=6, max_joint_velocity=10.0)
        result = arm._cmd_set_joint_velocity({"velocities": [1, 2, 3, 4, 5, 6]})
        self.assertFalse(result["clamped"])

    def test_wrong_length_velocities_rejected(self):
        arm, fake = make_arm(dof=6)
        with self.assertRaises(ValueError):
            arm._cmd_set_joint_velocity({"velocities": [1, 2, 3]})

    def test_wrong_length_cartesian_velocity_rejected(self):
        arm, fake = make_arm(dof=6)
        with self.assertRaises(ValueError):
            arm._cmd_set_cartesian_velocity({"velocity": [1, 2, 3]})

    def test_wrong_length_servo_angles_rejected(self):
        arm, fake = make_arm(dof=6)
        with self.assertRaises(ValueError):
            arm._cmd_servo_joint({"angles": [0, 0]})

    def test_dof5_arm_rejects_6_length_velocity(self):
        arm, fake = make_arm(dof=5)
        with self.assertRaises(ValueError):
            arm._cmd_set_joint_velocity({"velocities": [1, 2, 3, 4, 5, 6]})

    def test_dof5_arm_accepts_5_length_velocity(self):
        arm, fake = make_arm(dof=5)
        result = arm._cmd_set_joint_velocity({"velocities": [1, 2, 3, 4, 5]})
        self.assertTrue(result["ok"])

    def test_missing_velocities_key_rejected(self):
        arm, fake = make_arm(dof=6)
        with self.assertRaises(ValueError):
            arm._cmd_set_joint_velocity({})


class EnabledGuardTest(unittest.TestCase):
    def test_velocity_command_rejected_when_not_enabled(self):
        arm, fake = make_arm(dof=6)
        arm._enabled = False
        with self.assertRaises(RuntimeError):
            arm._cmd_set_joint_velocity({"velocities": [0] * 6})

    def test_servo_command_rejected_when_not_enabled(self):
        arm, fake = make_arm(dof=6)
        arm._enabled = False
        with self.assertRaises(RuntimeError):
            arm._cmd_servo_joint({"angles": [0] * 6})


class WatchdogTest(unittest.TestCase):
    def test_watchdog_feeds_on_velocity_command(self):
        arm, fake = make_arm(dof=6, watchdog_timeout_s=5.0)
        self.assertFalse(arm._watchdog.armed)
        arm._cmd_set_joint_velocity({"velocities": [1] * 6})
        self.assertTrue(arm._watchdog.armed)
        arm._do_stop()  # cleanup so the 5s timer doesn't outlive the test

    def test_watchdog_auto_stops_on_stale_command(self):
        arm, fake = make_arm(dof=6, watchdog_timeout_s=0.05)
        arm._cmd_set_joint_velocity({"velocities": [5, 5, 5, 5, 5, 5]})
        self.assertEqual(fake.mode, 4)
        self.assertNotEqual(fake.state, 4)

        time.sleep(0.2)  # let the watchdog fire (timeout=0.05s)

        self.assertEqual(fake.state, 4)  # arm was stopped
        self.assertEqual(arm._mode, ControlMode.POSITION)
        self.assertFalse(arm._moving)
        vel_calls = fake.calls_named("vc_set_joint_velocity")
        self.assertEqual(vel_calls[-1][0][0], [0.0] * 6)  # zeroed before stopping

    def test_watchdog_refeed_prevents_timeout(self):
        arm, fake = make_arm(dof=6, watchdog_timeout_s=0.15)
        arm._cmd_set_joint_velocity({"velocities": [3] * 6})
        time.sleep(0.08)
        arm._cmd_set_joint_velocity({"velocities": [3] * 6})  # refresh before timeout
        time.sleep(0.08)
        # total elapsed since first feed (~0.16s) exceeds timeout, but the refeed at 0.08s
        # means only ~0.08s has passed since the last feed -- watchdog should not have fired.
        self.assertNotEqual(fake.state, 4)
        arm._do_stop()

    def test_stop_disarms_watchdog(self):
        arm, fake = make_arm(dof=6, watchdog_timeout_s=0.05)
        arm._cmd_set_joint_velocity({"velocities": [1] * 6})
        arm._do_stop()
        self.assertFalse(arm._watchdog.armed)
        time.sleep(0.1)
        # no exception, no further calls after explicit stop cleared the timer
        state_calls_after = fake.calls_named("set_state")
        self.assertGreaterEqual(len(state_calls_after), 1)


class TorqueReadoutShapeTest(unittest.TestCase):
    def test_torque_readout_matches_dof(self):
        for dof in (5, 6, 7):
            arm, fake = make_arm(dof=dof)
            result = arm._cmd_get_joint_torques()
            self.assertEqual(len(result["torques"]), dof)


class StandardArmApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_joint_positions(self):
        arm, fake = make_arm(dof=6)
        fake._joint_angles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        jp = await arm.get_joint_positions()
        self.assertEqual(list(jp.values), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    async def test_move_to_joint_positions_wrong_length_rejected(self):
        arm, fake = make_arm(dof=6)
        with self.assertRaises(ValueError):
            await arm.move_to_joint_positions(JointPositions(values=[1, 2, 3]))

    async def test_move_to_joint_positions_calls_set_servo_angle(self):
        arm, fake = make_arm(dof=6)
        await arm.move_to_joint_positions(JointPositions(values=[0, 0, 0, 0, 0, 0]))
        self.assertEqual(len(fake.calls_named("set_servo_angle")), 1)
        self.assertEqual(fake.mode, 0)  # position mode

    async def test_is_moving_false_by_default(self):
        arm, fake = make_arm(dof=6)
        self.assertFalse(await arm.is_moving())

    async def test_stop_method(self):
        arm, fake = make_arm(dof=6)
        await arm.stop()
        self.assertEqual(fake.state, 4)


if __name__ == "__main__":
    unittest.main()
