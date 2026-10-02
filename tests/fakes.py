"""Fake robot + synthetic perception ports for running PourController offline."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from components.calibration import LiveIdentity, STAGES
from components.pouring import Observation
from tests.synthetic import HOME_FLANGE, MODEL, T_FLANGE_CAM, frames_for

HOME_JOINTS_DEG = [-17.29, 21.32, -35.88, 0.47, -59.23, -15.89]


class RobotFailure(RuntimeError):
    pass


class FakeRobot:
    """Moves instantly to every commanded flange pose. Failure injection:
    ``fail_on_tag``/``fail_on_call`` raise on a move, ``fault`` reports a
    fault, ``tracking_offset_mm`` makes reported poses disagree."""

    def __init__(self, *, jaw_mm_when_grasping: Optional[float] = 64.0, mm_per_pos: float = 0.1,
                 holding: Optional[bool] = True, fail_on_call: Optional[int] = None,
                 fault: Optional[str] = None, tracking_offset_mm: float = 0.0,
                 slip_after_lift_mm: float = 0.0):
        self.T = HOME_FLANGE.copy()
        self.joint_values = list(HOME_JOINTS_DEG)
        self.commands: list = []
        self.events: list = []
        self.jaw_mm_when_grasping = jaw_mm_when_grasping
        self.mm_per_pos = mm_per_pos
        self.holding_flag = holding
        self.fail_on_call = fail_on_call
        self._fault = fault
        self.tracking_offset_mm = tracking_offset_mm
        self.slip_after_lift_mm = slip_after_lift_mm
        self.pos = 850.0
        self.stopped = False
        self._grabbed = False

    async def flange_pose(self):
        T = self.T.copy()
        T[0, 3] += self.tracking_offset_mm
        return T

    async def joints(self):
        return list(self.joint_values)

    async def move_flange(self, T, timeout):
        self.commands.append(T.copy())
        if self.fail_on_call is not None and len(self.commands) == self.fail_on_call:
            raise RobotFailure(f"injected move failure on call {self.fail_on_call}")
        self.T = T.copy()
        self.joint_values = [99.0] * 6  # anywhere but home

    async def go_home(self):
        self.events.append("go_home")
        self.T = HOME_FLANGE.copy()
        self.joint_values = list(HOME_JOINTS_DEG)

    async def gripper_open(self):
        self.events.append("open")
        self.pos = 850.0
        self._grabbed = False

    async def gripper_grab(self):
        self.events.append("grab")
        self._grabbed = True
        w = self.jaw_mm_when_grasping if self.jaw_mm_when_grasping is not None else 0.0
        self.pos = w / self.mm_per_pos
        return self.holding_flag, self.pos

    async def gripper_pos(self):
        if self._grabbed and self.slip_after_lift_mm:
            return self.pos - self.slip_after_lift_mm / self.mm_per_pos
        return self.pos

    async def fault(self):
        return self._fault

    async def stop(self):
        self.stopped = True
        self.events.append("stop")


def live_identity(model=MODEL, T_flange_cam=T_FLANGE_CAM) -> LiveIdentity:
    return LiveIdentity(
        color_size=(model.width, model.height),
        depth_size=(model.width, model.height),
        intrinsics={"fx": model.fx, "fy": model.fy, "cx": model.cx, "cy": model.cy,
                    "width": model.width, "height": model.height},
        distortion={"model": model.dist_model, "coeffs": list(model.coeffs)},
        reported_extrinsics=None,
        depth_encoding="",
        T_flange_cam_viam=T_flange_cam,
        cam_parent_is_arm=True,
    )


class SyntheticPerception:
    """Renders the synthetic scene from wherever the fake arm is."""

    def __init__(self, scene, robot: FakeRobot, labels: dict, *, live: Optional[LiveIdentity] = None,
                 move_cup_before_reobserve: Optional[tuple] = None, frames_kwargs: Optional[dict] = None,
                 picked_obj_id: Optional[int] = 1):
        self.scene = scene
        self.picked_obj_id = picked_obj_id
        self.robot = robot
        self.labels = labels          # obj_id -> label ("bottle", "cup", ...)
        self.live = live or live_identity()
        self.move_cup = move_cup_before_reobserve
        self.kw = frames_kwargs or {}
        self.observe_calls = 0
        self.capture_calls = 0

    async def observe(self, n_frames, *, aligned):
        self.observe_calls += 1
        frames, ids = frames_for(self.scene, self.robot.T, n=n_frames, **self.kw)
        cands = []
        for oid, label in self.labels.items():
            m = (ids == oid).astype(np.uint8)
            if m.sum() == 0:
                continue
            ys, xs = np.nonzero(m)
            cands.append({"label": label, "mask": m,
                          "box": (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())),
                          "score": 0.9})
        return Observation(frames=frames, candidates=cands, live=self.live, t_mono=time.monotonic())

    async def capture(self, n_frames, *, aligned):
        self.capture_calls += 1
        if self.move_cup is not None:
            for ob in self.scene.objects:
                if type(ob).__name__ == "Cup":
                    ob.x, ob.y = self.move_cup
        scene = self.scene
        if self.robot._grabbed and self.picked_obj_id is not None:
            # The picked source is in the gripper now, not on the table.
            from tests.synthetic import SynScene

            scene = SynScene([o for o in self.scene.objects if o.obj_id != self.picked_obj_id], self.scene.table_z)
        frames, _ = frames_for(scene, self.robot.T, n=n_frames, **self.kw)
        return Observation(frames=frames, candidates=[], live=self.live, t_mono=time.monotonic())


def passing_trials(setup_hash: str, upto: str = "G") -> list:
    out = []
    for stage, spec in STAGES.items():
        if stage > upto:
            continue
        for i in range(spec["min_trials"]):
            out.append({"stage": stage, "setup_hash": setup_hash, "success": True, "collision": False,
                        "mentor_approved": True, "estop_attended": True, "tray": True})
    return out


class FakeOperator:
    def __init__(self, error_mm: float = 3.0):
        self.error_mm = error_mm
        self.calls = []

    async def measure_hover(self, name, *, predicted, fingertip):
        self.calls.append(name)
        return self.error_mm
