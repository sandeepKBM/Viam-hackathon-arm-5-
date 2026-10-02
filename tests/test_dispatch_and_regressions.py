"""Voice / orchestrator dispatch of the pour, plus the named hardware regressions:

- "the bottle was getting missed by a lot, LIKE THE ARM WAS ABOVE THE ACTUAL
  BOTTLE WHEN IT WAS TRYING TO PICK IT UP"   -> test_arm_was_above_actual_bottle
- "the pour action is not there at all I guess"   -> test_pour_action_is_dispatchable
- "the cup finding with the correct offset for the camera needs to be
  implemented"   -> test_cup_camera_offset_comes_from_calibration_not_magic_constant
- "the action of pouring never succeeded"   -> test_pouring_reaches_done_only_after_full_trajectory
"""

import asyncio
import importlib.util
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import components.pouring as pouring
from components.calibration import check_setup
from components.object_pose import TableFrame, estimate_cup
from components.pour_planner import PourParams, Scene, boxes_from_setup, cylinder_for, plan_side_grasp
from components.transforms import make_T
from tests.fakes import FakeRobot, SyntheticPerception, live_identity, passing_trials
from tests.synthetic import (
    HOME_FLANGE,
    TABLE_Z,
    T_FLANGE_CAM,
    Bottle,
    Cup,
    SynScene,
    frames_for,
    masks_from_ids,
    setup_doc,
)

ROOT = Path(__file__).resolve().parent.parent


def _load_voice_app():
    spec = importlib.util.spec_from_file_location("voice_app_under_test", ROOT / "voice" / "voice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.speak = lambda text: mod._spoken.append(text)
    mod._spoken = []
    return mod


class _Done:
    """Stands in for PourController.run and counts invocations."""

    calls = []


async def _done_run(self, req):
    _Done.calls.append(req)
    return pouring.PourResult(state="DONE", success=True, mode=pouring.PourMode(req.mode).value)


async def _fake_ports(machine=None, arm=None, gripper=None):
    return FakeRobot(), object()


class DispatchTests(unittest.TestCase):
    def setUp(self):
        _Done.calls = []
        self.patches = [
            mock.patch.object(pouring, "PORTS_FACTORY", _fake_ports),
            mock.patch.object(pouring, "readiness_problems", lambda *a, **k: (None, [])),
            mock.patch.object(pouring.PourController, "run", _done_run),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_voice_branch_calls_the_primitive_once(self):
        mod = _load_voice_app()
        mapped = {"task": "pour", "say": "ok", "moves": [],
                  "pour": {"source": "bottle", "target": "cup", "implicit": False, "clarify": None}}
        with mock.patch.dict(os.environ, {"USE_ORCHESTRATOR": ""}):
            asyncio.run(mod.run_task("pour", mapped))
        self.assertEqual(len(_Done.calls), 1)
        self.assertEqual((_Done.calls[0].source, _Done.calls[0].target, _Done.calls[0].mode),
                         ("bottle", "cup", pouring.PourMode.POUR))
        self.assertIn("Pour finished", mod._spoken[-1])

    def test_orchestrator_calls_the_same_primitive_once(self):
        import services.orchestrator_service as orch

        class FakeMachine:
            async def close(self):
                pass

        async def fake_connect():
            return FakeMachine()

        class FakeArm:
            def __init__(self, machine):
                self.machine = machine
                self.name = "arm"

        class FakeGripper:
            def __init__(self, machine):
                pass

        with mock.patch.dict(os.environ, {"VIAM_ALLOW_LIVE": "1"}), \
                mock.patch("components.connection.connect_machine", fake_connect), \
                mock.patch("components.arm.ArmComponent", FakeArm), \
                mock.patch("components.gripper.GripperComponent", FakeGripper):
            out = asyncio.run(orch._handle_run({"task": "pour", "pour": {"source": "can", "target": "cup"}}))
        self.assertEqual(len(_Done.calls), 1)
        self.assertEqual(_Done.calls[0].source, "can")
        self.assertTrue(out["execution"]["ok"])
        self.assertEqual(out["plan"], [{"skill": "pour", "params": {"source": "can", "target": "cup"}}])

    def test_orchestrator_dry_mode_never_connects(self):
        import services.orchestrator_service as orch

        async def must_not(*a, **k):
            raise AssertionError("connected without VIAM_ALLOW_LIVE")

        with mock.patch.dict(os.environ, {"VIAM_ALLOW_LIVE": ""}), \
                mock.patch("components.connection.connect_machine", must_not):
            out = asyncio.run(orch._handle_run({"task": "pour", "pour": {"source": "bottle", "target": "cup"}}))
        self.assertIsNone(out["execution"])
        self.assertEqual(_Done.calls, [])

    def test_clarification_never_runs(self):
        mod = _load_voice_app()
        asyncio.run(mod.run_task("pour", {"task": "pour", "pour": {"clarify": "Which one?"}}))
        self.assertEqual(_Done.calls, [])
        self.assertEqual(mod._spoken, ["Which one?"])


class IntentTests(unittest.TestCase):
    def test_resolve_pour_intent(self):
        from components.voice import resolve_pour_intent

        self.assertEqual(resolve_pour_intent({"source": "water bottle", "target": "mug"})["source"], "bottle")
        self.assertIsNone(resolve_pour_intent({"source": "soda", "target": "cup"})["clarify"])
        self.assertIsNotNone(resolve_pour_intent({"source": None, "target": "cup"})["clarify"])          # ambiguous
        self.assertIsNotNone(resolve_pour_intent({"source": "bottle", "target": "can"})["clarify"])      # can != cup
        self.assertIsNotNone(resolve_pour_intent({"source": "cup", "target": "cup"})["clarify"])
        thirsty = resolve_pour_intent({"source": None, "target": None, "implicit": True})
        self.assertEqual((thirsty["source"], thirsty["clarify"]), ("any", None))

    def test_map_task_uses_the_single_existing_llm_call(self):
        from components import voice

        replies = {
            "pour the bottle into the cup": {"task": "pour", "say": "Pouring.", "moves": [],
                                             "pour": {"source": "bottle", "target": "cup", "implicit": False}},
            "i am thirsty": {"task": "pour", "say": "Let me pour.", "moves": [],
                             "pour": {"source": None, "target": None, "implicit": True}},
            "pour it": {"task": "pour", "say": "ok", "moves": [], "pour": {"source": None, "target": None}},
        }
        calls = []

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        calls.append(kw)
                        text = kw["messages"][-1]["content"]

                        class M:
                            content = json.dumps(replies[text])

                        class C:
                            message = M

                        class R:
                            choices = [C]

                        return R

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}), mock.patch("openai.OpenAI", lambda: FakeClient):
            a = voice.map_task("pour the bottle into the cup")
            b = voice.map_task("i am thirsty")
            c = voice.map_task("pour it")
        self.assertEqual(len(calls), 3)   # one model call per utterance, no extra
        self.assertEqual((a["task"], a["pour"]["source"], a["pour"]["clarify"]), ("pour", "bottle", None))
        self.assertEqual((b["pour"]["source"], b["pour"]["implicit"]), ("any", True))
        self.assertIsNotNone(c["pour"]["clarify"])
        self.assertEqual(c["say"], c["pour"]["clarify"])

    def test_thirsty_without_readiness_says_unavailable(self):
        mod = _load_voice_app()
        with mock.patch.object(pouring.PourController, "run", side_effect=AssertionError("must not run")), \
                mock.patch.dict(os.environ, {"ENABLE_CALIBRATED_POUR": "", "USE_ORCHESTRATOR": ""}):
            asyncio.run(mod.run_task("pour", {"task": "pour", "pour": {"source": "any", "target": "cup",
                                                                         "implicit": True, "clarify": None}}))
        self.assertIn("unavailable", mod._spoken[-1])
        self.assertIn("hand me the bottle", mod._spoken[-1])


class NamedRegressions(unittest.TestCase):
    def test_arm_was_above_actual_bottle(self):
        """Old: side grasp commanded the flange to MIN_Z + 1.5 in (218 mm). With a
        horizontal tool the pads sit at the flange height, but MIN_Z is the
        flange floor for a *downward* tool (it already contains the ~170 mm
        tool length), so the pads were above a 180 mm bottle. New: pad centre
        in the bottle's grasp band, measured from the calibrated table plane."""
        setup = check_setup(setup_doc())
        table = TableFrame(setup.table_normal, setup.table_d)
        from components.object_pose import estimate_upright_source

        frames, ids = frames_for(SynScene([Bottle(300, -60), Cup(180, -200)]))
        m = masks_from_ids(ids)
        b = estimate_upright_source("bottle", frames, m[1], setup_hash="h", table=table,
                                    calib_err={"p95_xy_mm": 4, "p95_z_mm": 3})
        c = estimate_cup(frames, m[3], setup_hash="h", table=table, calib_err={"p95_xy_mm": 4, "p95_z_mm": 3})
        scene = Scene(table=table, cup=cylinder_for(c, table), boxes=boxes_from_setup(setup))
        g = plan_side_grasp(setup, scene, b, PourParams())
        bottle_top = TABLE_Z + 180.0
        old_pad_z = 179.75673 + 1.5 * 25.4            # flange z == pad z for a horizontal tool
        self.assertGreater(old_pad_z, bottle_top)      # the reported failure, reproduced
        pad_z = g.T_grasp[2, 3]
        self.assertTrue(TABLE_Z + 0.2 * 180 < pad_z < TABLE_Z + 0.6 * 180, pad_z)
        self.assertLess(np.linalg.norm(g.T_grasp[:2, 3] - np.array([300.0, -60.0])), 6.0 + g.shallow_offset_mm)
        src = (ROOT / "scripts" / "pour_can.py").read_text()
        for gone in ("POUR_STICKOUT_MM", "POUR_PICK_HEIGHT_IN", "MIN_Z + mm(", "_nudge_in_workspace"):
            self.assertNotIn(gone, src)

    def test_pour_action_is_dispatchable(self):
        from components import voice
        from components.skills import SKILL_REGISTRY, make_skill_call

        self.assertIn("pour", voice.TASKS)
        self.assertIn("pour", SKILL_REGISTRY)
        make_skill_call("pour", source="bottle", target="cup")
        vsrc = (ROOT / "voice" / "voice.py").read_text()
        self.assertEqual(len(re.findall(r'if name == "pour":', vsrc)), 1)
        self.assertNotIn("subprocess", vsrc.split('if name == "pour":')[1].split("if name ==")[0])
        osrc = (ROOT / "services" / "orchestrator_service.py").read_text()
        self.assertIn('if task == "pour":', osrc)

    def test_cup_camera_offset_comes_from_calibration_not_magic_constant(self):
        """The cup position follows the calibrated camera mount (through the
        frame system), so a mount error shows up as the same localization
        error, and a frame system that disagrees with the calibration blocks
        the pour. There is no CAMERA_OFFSET-style constant anywhere."""
        sc = SynScene([Bottle(300, -60), Cup(180, -200)])
        frames, ids = frames_for(sc)
        table = TableFrame(np.array([0.0, 0.0, 1.0]), -TABLE_Z)
        cal = {"p95_xy_mm": 4, "p95_z_mm": 3}
        good = estimate_cup(frames, masks_from_ids(ids)[3], setup_hash="h", table=table, calib_err=cal)
        # Same images, camera pose computed through a mount that is 12 mm off.
        bad_mount = T_FLANGE_CAM @ make_T(t=(12.0, 0.0, 0.0))
        import dataclasses

        shifted = [dataclasses.replace(f, T_world_cam=HOME_FLANGE @ bad_mount) for f in frames]
        bad = estimate_cup(shifted, masks_from_ids(ids)[3], setup_hash="h", table=table, calib_err=cal)
        self.assertLess(np.linalg.norm(good.position[:2] - np.array([180.0, -200.0])), 2.0)
        self.assertTrue((not bad.ok) or np.linalg.norm(bad.position[:2] - good.position[:2]) > 8.0)
        live = live_identity()
        live.T_flange_cam_viam = bad_mount
        from components.calibration import CalibrationError

        with self.assertRaises(CalibrationError):
            check_setup(setup_doc(), live=live)
        for path in ("components/pouring.py", "components/pour_planner.py", "components/object_pose.py",
                     "components/rgbd.py", "scripts/pour_can.py"):
            text = (ROOT / path).read_text()
            for magic in ("CAMERA_OFFSET", "POUR_X_OFFSET", "CAM_OFFSET", "STICKOUT_MM"):
                self.assertNotIn(magic, text, f"{magic} in {path}")

    def test_pouring_reaches_done_only_after_full_trajectory(self):
        doc = setup_doc()
        robot = FakeRobot(jaw_mm_when_grasping=62.0)
        per = SyntheticPerception(SynScene([Bottle(300, -60), Cup(180, -200)]), robot, {1: "bottle", 3: "cup"})

        async def noop(_):
            return None

        ctl = pouring.PourController(robot, per, setup_doc=doc, trials=passing_trials(doc["setup_hash"]),
                                     env={"ENABLE_CALIBRATED_POUR": "1"}, save_evidence=False, sleep=noop,
                                     observation_joints=[-17.29, 21.32, -35.88, 0.47, -59.23, -15.89])
        res = asyncio.run(ctl.run(pouring.PourRequest(mode=pouring.PourMode.POUR_DRY, source="bottle")))
        self.assertTrue(res.success)
        self.assertEqual(res.transitions[-1], "DONE")
        for st in ("TILT_INCREMENTS", "HOLD", "UNTILT"):
            self.assertIn(st, res.transitions)
        self.assertGreaterEqual(max(ctl.pour.tilt_deg), 90.0)
        self.assertIn("check the cup", pouring.describe_result(res))   # honest: open-loop


if __name__ == "__main__":
    unittest.main()
