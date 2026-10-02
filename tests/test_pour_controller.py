import asyncio
import datetime as dt
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from components import pour_planner
from components.pouring import (
    S,
    PourController,
    PourMode,
    PourRequest,
    ReplayPerception,
)
from tests.fakes import FakeOperator, FakeRobot, HOME_JOINTS_DEG, SyntheticPerception, passing_trials
from tests.synthetic import Bottle, Can, Cup, SynScene, setup_doc

ENV = {"ENABLE_CALIBRATED_POUR": "1", "POUR_ALLOW_LIQUID": "1"}
LABELS = {1: "bottle", 3: "cup"}
MOTION_STATES = ["PREGRASP", "GRASP", "LIFT", "REOBSERVE_CUP", "PREPOUR", "TILT_INCREMENTS", "UNTILT", "RETREAT",
                 "SAFE_PLACE"]


async def _noop(_):
    return None


def make(mode="pour_dry", *, scene=None, robot=None, doc=None, env=ENV, trials=None, labels=LABELS,
         perception_kw=None, save=False, run_dir=None, clock=time.monotonic, sleep=_noop, params=None):
    doc = doc or setup_doc()
    sc = scene or SynScene([Bottle(300, -60), Cup(180, -200)])
    robot = robot or FakeRobot(jaw_mm_when_grasping=62.0)
    per = SyntheticPerception(sc, robot, labels, **(perception_kw or {}))
    ctl = PourController(robot, per, setup_doc=doc,
                         trials=passing_trials(doc["setup_hash"]) if trials is None else trials,
                         env=env, save_evidence=save, run_dir=run_dir, sleep=sleep, clock=clock,
                         operator=FakeOperator(), observation_joints=HOME_JOINTS_DEG, params=params)
    return ctl, robot, per


def run(ctl, mode="pour_dry", **req):
    return asyncio.run(ctl.run(PourRequest(mode=PourMode(mode), source=req.pop("source", "bottle"), **req)))


class ModeTests(unittest.TestCase):
    def test_plan_mode_never_moves(self):
        ctl, robot, _ = make()
        res = run(ctl, "plan")
        self.assertTrue(res.success)
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.events, [])

    def test_hover_stays_above_targets(self):
        ctl, robot, _ = make()
        res = run(ctl, "hover")
        self.assertTrue(res.success, res.reason)
        self.assertNotIn("grab", robot.events)
        tip = ctl.setup.gripper.pad_length_mm / 2.0
        for T in robot.commands:
            tcp = (T @ ctl.setup.T_flange_tcp)[:3, 3]
            self.assertGreater(tcp[2] - tip, ctl.selection.cup.position[2] + 29.0)  # fingertips >= 30 mm above the rim
        self.assertEqual(ctl.operator.calls, ["source_mouth", "cup_rim_center"])

    def test_grasp_mode_returns_and_releases(self):
        ctl, robot, _ = make()
        res = run(ctl, "grasp")
        self.assertTrue(res.success, res.reason)
        self.assertEqual(robot.events.count("grab"), 1)
        self.assertEqual(robot.events[-1], "go_home")
        self.assertIn("open", robot.events[robot.events.index("grab"):])
        self.assertNotIn("TILT_INCREMENTS", res.transitions)

    def test_full_dry_pour_reaches_done_in_order(self):
        ctl, robot, per = make()
        res = run(ctl, "pour_dry")
        self.assertTrue(res.success, (res.reason, res.detail))
        order = ["IDLE", "OBSERVE", "VALIDATE_CALIBRATION", "LOCALIZE_SOURCE_AND_CUP", "PLAN_GRASP", "PREGRASP",
                 "GRASP", "VERIFY_GRASP", "LIFT", "REOBSERVE_CUP", "PLAN_POUR", "PREPOUR", "TILT_INCREMENTS",
                 "HOLD", "UNTILT", "RETREAT", "SAFE_PLACE", "DONE"]
        self.assertEqual(res.transitions, order)
        self.assertEqual(per.capture_calls, 1)          # the cup was re-observed after lifting
        self.assertEqual(robot.events[-1], "go_home")

    def test_evidence_and_timings_written(self):
        with tempfile.TemporaryDirectory() as d:
            ctl, robot, _ = make(save=True, run_dir=Path(d))
            res = run(ctl, "pour_dry")
            self.assertTrue(res.success)
            names = {p.name for p in Path(d).iterdir()}
            for f in ("events.jsonl", "commands.jsonl", "overlay.png", "plan.json", "plan_top.png", "plan_side.png",
                      "estimates.json", "summary.json", "candidates.npz", "obs_00_color.png", "obs_00_depth_mm.png",
                      "reobs_00_meta.json", "reobserved_cup.json"):
                self.assertIn(f, names)
            plan = json.loads((Path(d) / "plan.json").read_text())
            self.assertIn("mouth_mm", plan["pour"])
            self.assertIn("flange", plan["pour"]["tilt_poses"][0])
            self.assertEqual(plan["calibration_id"], ctl.setup.setup_hash)
            self.assertIn("TILT_INCREMENTS", res.state_times_s)
            cmds = [json.loads(line) for line in (Path(d) / "commands.jsonl").read_text().splitlines()]
            self.assertTrue(all("flange_actual" in c and "tracking_err_mm" in c for c in cmds))

    def test_replay_reproduces_plan_offline(self):
        with tempfile.TemporaryDirectory() as d:
            doc = setup_doc()
            ctl, robot, _ = make(doc=doc, save=True, run_dir=Path(d))
            res = run(ctl, "plan")
            self.assertTrue(res.success)
            first = ctl.grasp.T_grasp.copy()
            rep = PourController(None, ReplayPerception(Path(d)), setup_doc=doc, save_evidence=False)
            res2 = asyncio.run(rep.run(PourRequest(mode=PourMode.PLAN, source="bottle")))
            self.assertTrue(res2.success, res2.reason)
            self.assertLess(abs(rep.grasp.T_grasp - first).max(), 1.0)  # depth PNG is mm-quantized


class GateTests(unittest.TestCase):
    def test_disabled_flag_blocks_everything_live(self):
        for mode in ("plan", "pour_dry"):
            ctl, robot, per = make(env={})
            res = run(ctl, mode)
            self.assertEqual(res.reason, "pour_disabled")
            self.assertEqual(robot.commands, [])
            self.assertEqual(per.observe_calls, 0)

    def test_stale_calibration_no_motion(self):
        old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat()
        ctl, robot, _ = make(doc=setup_doc(calibrated_at=old))
        res = run(ctl, "pour_dry")
        self.assertEqual(res.reason, "not_ready")
        self.assertTrue(any("stale" in p for p in res.detail["problems"]))
        self.assertEqual(robot.commands, [])

    def test_missing_stages_no_motion(self):
        doc = setup_doc()
        ctl, robot, _ = make(doc=doc, trials=passing_trials(doc["setup_hash"], upto="D"))
        res = run(ctl, "pour_dry")
        self.assertEqual(res.reason, "not_ready")
        self.assertEqual(robot.commands, [])

    def test_liquid_requires_flag_and_confirmations(self):
        ctl, robot, _ = make(env={"ENABLE_CALIBRATED_POUR": "1"})
        self.assertEqual(run(ctl, "pour_liquid").reason, "liquid_disabled")
        ctl, robot, _ = make()
        res = run(ctl, "pour_liquid")
        self.assertEqual(res.reason, "not_ready")
        self.assertTrue(any("confirmations" in p for p in res.detail["problems"]))
        self.assertEqual(robot.commands, [])

    def test_liquid_hold_is_bounded(self):
        from components.pour_planner import PourParams

        ctl, robot, _ = make(params=PourParams(hold_s=5.0))
        res = run(ctl, "pour_liquid", confirmations={"tray": 1, "estop_attended": 1, "mentor_approved": 1,
                                                     "bounded_amount": 1})
        self.assertEqual(res.reason, "hold_too_long")
        self.assertEqual(robot.commands, [])

    def test_poor_localization_no_motion(self):
        ctl, robot, _ = make(perception_kw={"frames_kwargs": {"holes": {1: 0.9}}})
        res = run(ctl, "pour_dry")
        self.assertFalse(res.success)
        self.assertTrue(res.reason.startswith("source_rejected"), res.reason)
        self.assertEqual(robot.commands, [])

    def test_scene_rules_no_motion(self):
        cases = {
            "no_cup": (SynScene([Bottle(300, -60)]), {1: "bottle"}, "bottle"),
            "bottle_and_can_without_cup": (SynScene([Bottle(300, -60), Can(330, -250)]), {1: "bottle", 2: "can"}, "any"),
            "multiple_sources_need_selection": (SynScene([Bottle(300, -60), Cup(180, -200), Can(330, -250)]),
                                                {1: "bottle", 2: "can", 3: "cup"}, "any"),
        }
        for reason, (sc, labels, src) in cases.items():
            with self.subTest(reason):
                ctl, robot, _ = make(scene=sc, labels=labels)
                res = run(ctl, "pour_dry", source=src)
                self.assertEqual(res.reason, reason)
                self.assertEqual(robot.commands, [])

    def test_stale_perception_no_motion(self):
        ctl, robot, _ = make(clock=lambda: time.monotonic() + 10_000.0)
        res = run(ctl, "pour_dry")
        self.assertEqual(res.reason, "stale_perception")
        self.assertEqual(robot.commands, [])

    def test_invalid_target_refused(self):
        ctl, robot, _ = make()
        res = asyncio.run(ctl.run(PourRequest(mode=PourMode.POUR_DRY, source="bottle", target="can")))
        self.assertEqual(res.reason, "invalid_target")


class AbortRecoveryTests(unittest.TestCase):
    def _state_of_calls(self):
        """Map each move_flange call index of a clean run to the state it ran in."""
        ctl, robot, _ = make()
        states = []
        orig = robot.move_flange

        async def spy(T, timeout):
            states.append(ctl.state.value)
            await orig(T, timeout)

        robot.move_flange = spy
        res = run(ctl, "pour_dry")
        self.assertTrue(res.success)
        return states

    def test_move_failure_in_every_motion_state_recovers(self):
        states = self._state_of_calls()
        for st in MOTION_STATES:
            with self.subTest(st):
                first = states.index(st) + 1
                ctl, robot, _ = make(robot=FakeRobot(jaw_mm_when_grasping=62.0, fail_on_call=first))
                res = run(ctl, "pour_dry")
                self.assertEqual(res.state, "ABORTED")
                self.assertEqual(res.detail["failed_in"], st)
                self.assertFalse(ctl.holding, "source must be released at its pick location")
                self.assertFalse(res.help_required, res.detail)
                self.assertEqual(robot.events[-1], "go_home")
                if "grab" in robot.events:
                    self.assertIn("open", robot.events[robot.events.index("grab"):])
                self.assertNotIn("HOLD", res.transitions[res.transitions.index(st) + 1:])

    def test_failure_in_non_motion_states(self):
        # OBSERVE: perception error
        ctl, robot, per = make()

        async def boom(*a, **k):
            raise RuntimeError("camera offline")

        per.observe = boom
        res = run(ctl)
        self.assertEqual(res.detail["failed_in"], "OBSERVE")
        self.assertEqual(robot.commands, [])
        # PLAN_GRASP: infeasible grasp
        ctl, robot, _ = make()
        with mock.patch("components.pouring.plan_side_grasp",
                        side_effect=pour_planner.PlanningError("no_feasible_side_grasp")):
            res = run(ctl)
        self.assertEqual((res.detail["failed_in"], res.reason), ("PLAN_GRASP", "no_feasible_side_grasp"))
        self.assertEqual(robot.commands, [])
        # PLAN_POUR: infeasible after re-observation -> bottle returned
        ctl, robot, _ = make()
        real = pour_planner.plan_pour
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] > 1:
                raise pour_planner.PlanningError("no_feasible_pour_path")
            return real(*a, **k)

        with mock.patch("components.pouring.plan_pour", side_effect=flaky):
            res = run(ctl)
        self.assertEqual(res.detail["failed_in"], "PLAN_POUR")
        self.assertFalse(ctl.holding)
        self.assertEqual(robot.events[-1], "go_home")

    def test_abort_during_hold_untilts_first(self):
        async def sleep(s):
            if ctl.state == S.HOLD:
                raise RuntimeError("operator stop")

        ctl, robot, _ = make(sleep=sleep)
        res = run(ctl)
        self.assertEqual(res.detail["failed_in"], "HOLD")
        # the rewind passes back through the tilt waypoints to upright
        tilt_poses = [T for T in ctl.pour.T_tcp]
        upright_flange = ctl.pour.T_tcp[0] @ __import__("numpy").linalg.inv(ctl.setup.T_flange_tcp)
        import numpy as np

        self.assertTrue(any(np.allclose(c, upright_flange) for c in robot.commands[-400:]))
        self.assertFalse(ctl.holding)
        self.assertEqual(robot.events[-1], "go_home")

    def test_failed_grasp_releases_and_backs_out(self):
        ctl, robot, _ = make(robot=FakeRobot(jaw_mm_when_grasping=5.0))   # closed on nothing
        res = run(ctl)
        self.assertEqual((res.detail["failed_in"], res.reason), ("VERIFY_GRASP", "grasp_failed"))
        self.assertEqual(robot.events, ["open", "grab", "open", "go_home"])
        ctl, robot, _ = make(robot=FakeRobot(jaw_mm_when_grasping=62.0, holding=False))
        self.assertEqual(run(ctl).reason, "grasp_failed")

    def test_slip_after_lift_puts_back(self):
        ctl, robot, _ = make(robot=FakeRobot(jaw_mm_when_grasping=62.0, slip_after_lift_mm=10.0))
        res = run(ctl)
        self.assertEqual((res.detail["failed_in"], res.reason), ("LIFT", "slip_detected"))
        self.assertFalse(ctl.holding)

    def test_robot_fault_stops_without_moving(self):
        robot = FakeRobot(jaw_mm_when_grasping=62.0)
        ctl, robot, _ = make(robot=robot)
        orig = robot.gripper_grab

        async def grab_then_fault():
            r = await orig()
            robot._fault = "collision detected"
            return r

        robot.gripper_grab = grab_then_fault
        res = run(ctl)
        self.assertEqual(res.reason, "robot_fault")
        self.assertTrue(res.help_required)
        self.assertTrue(robot.stopped)
        self.assertEqual(res.detail["failed_in"], "LIFT")        # next phase guard caught it
        self.assertNotIn("go_home", robot.events)                # no recovery motion on a fault

    def test_tracking_error_stops_and_asks_for_help(self):
        ctl, robot, _ = make(robot=FakeRobot(jaw_mm_when_grasping=62.0, tracking_offset_mm=8.0))
        res = run(ctl)
        self.assertEqual(res.reason, "tracking_error")
        self.assertTrue(res.help_required)
        self.assertTrue(robot.stopped)
        self.assertEqual(len(robot.commands), 1)   # stopped after the first deviating move

    def test_reobservation_disagreement_returns_bottle_no_pour(self):
        # 14 mm move: re-fit succeeds but disagrees with the first observation.
        # 29 mm move: the cup left the prior ROI, the re-fit itself fails.
        for moved_to, reason in (((192.0, -207.0), "reobservation_disagreement"),
                                 ((205.0, -215.0), "reobservation_failed")):
            with self.subTest(reason):
                ctl, robot, _ = make(perception_kw={"move_cup_before_reobserve": moved_to})
                res = run(ctl)
                self.assertEqual(res.reason, reason)
                self.assertNotIn("TILT_INCREMENTS", res.transitions)
                self.assertFalse(ctl.holding)
                self.assertEqual(robot.events[-1], "go_home")


if __name__ == "__main__":
    unittest.main()


class NominalModeTests(unittest.TestCase):
    """Operator skipped calibration: setup built from the live machine."""

    def _make(self, **kw):
        sc = kw.pop("scene", None) or SynScene([Bottle(300, -60), Cup(180, -200)])
        robot = FakeRobot(jaw_mm_when_grasping=62.0)
        per = SyntheticPerception(sc, robot, kw.pop("labels", LABELS), **kw.pop("perception_kw", {}))
        ctl = PourController(robot, per, setup_doc=None, trials=[], env=ENV, save_evidence=False, sleep=_noop,
                             observation_joints=HOME_JOINTS_DEG, nominal=True, **kw)
        return ctl, robot, per

    def test_nominal_dry_pour_without_calibration(self):
        ctl, robot, per = self._make()
        res = run(ctl, "pour_dry")
        self.assertTrue(res.success, (res.reason, res.detail))
        self.assertEqual(ctl.setup.doc["status"], "nominal")
        self.assertIn("NOMINAL (uncalibrated) setup: expect cm-level error", res.notes)
        self.assertAlmostEqual(-ctl.setup.table_d, 5.0, delta=2.0)     # table fitted from depth
        self.assertEqual(per.capture_calls, 1)

    def test_nominal_still_needs_enable_flag(self):
        ctl, robot, per = self._make()
        ctl.env = {}
        self.assertEqual(run(ctl, "pour_dry").reason, "pour_disabled")
        self.assertEqual(robot.commands, [])

    def test_nominal_misregistered_depth_refuses_with_fix(self):
        from components.rgbd import CameraModel

        dm = CameraModel(fx=455 * 0.7, fy=455 * 0.7, cx=320, cy=180, width=640, height=360)
        ctl, robot, _ = self._make(perception_kw={"frames_kwargs": {"depth_model": dm,
                                                                   "depth_offset_cam": __import__("numpy").array([-15.0, 0, 0])}})
        res = run(ctl, "pour_dry")
        self.assertFalse(res.success)
        self.assertIn("align_color_depth", res.detail.get("fix", ""))
        self.assertEqual(robot.commands, [])

    def test_cup_labelled_can_is_used_as_cup_by_geometry(self):
        ctl, robot, _ = self._make(labels={1: "bottle", 3: "can"})
        res = run(ctl, "plan")
        self.assertTrue(res.success, (res.reason, res.detail))
        self.assertIn("detector_said_can_geometry_says_cup", ctl.selection.cup.quality_flags)

    def test_real_can_never_becomes_the_cup(self):
        ctl, robot, _ = self._make(scene=SynScene([Bottle(300, -60), Can(180, -200, obj_id=3)]),
                                   labels={1: "bottle", 3: "can"})
        res = run(ctl, "plan")
        self.assertEqual(res.reason, "no_cup")

    def test_skip_reobserve_is_nominal_only(self):
        ctl, robot, per = self._make(skip_reobserve=True)
        res = run(ctl, "pour_dry")
        self.assertTrue(res.success, res.reason)
        self.assertEqual(per.capture_calls, 0)
        with self.assertRaises(ValueError):
            PourController(robot, per, skip_reobserve=True)
