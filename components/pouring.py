"""Calibrated bottle/can -> cup pour: one state machine, shared by the CLI,
voice, and the orchestrator.

    IDLE -> OBSERVE -> VALIDATE_CALIBRATION -> LOCALIZE_SOURCE_AND_CUP
    -> PLAN_GRASP -> PREGRASP -> GRASP -> VERIFY_GRASP -> LIFT
    -> REOBSERVE_CUP -> PLAN_POUR -> PREPOUR -> TILT_INCREMENTS -> HOLD
    -> UNTILT -> RETREAT -> SAFE_PLACE -> DONE
    any state -> ABORTED (after RECOVER)

Gates, all fail-closed:
- ``ENABLE_CALIBRATED_POUR=1`` for anything that talks to the robot.
- config/calibration/pour_setup.json must be valid, fresh, unedited, and
  match the live camera + Viam frame system (components/calibration.py).
- Each mode requires the earlier hardware stages to have passed
  (config/calibration/pour_trials.jsonl, bound to the setup hash).
- Real liquid additionally needs POUR_ALLOW_LIQUID=1 and operator
  confirmations. Pouring is duration-based (open-loop, no fill sensing).

Recovery rewinds the executed trajectory (every pose in it was validated):
untilt, return the held source to where it was picked, release, back out.
If a move fails or the robot reports a fault, it stops and asks for help.
"""

from __future__ import annotations

import asyncio
import enum
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from components import calibration as calib
from components.object_pose import (
    CUP_LABEL,
    SOURCE_LABELS,
    ObjectPoseEstimate,
    PairSelection,
    TableFrame,
    estimate_cup,
    estimate_upright_source,
    select_pour_pair,
)
from components.pour_evidence import (
    PourLog,
    draw_overlay,
    draw_plan_views,
    dump_json,
    new_run_dir,
    plan_json,
    save_candidates,
)
from components.pour_planner import (
    GraspPlan,
    PlanningError,
    PourParams,
    PourPlan,
    Scene,
    boxes_from_setup,
    check_pose,
    cylinder_for,
    gripper_spheres,
    hover_targets,
    plan_pour,
    plan_reobserve_view,
    plan_side_grasp,
)
from components.rgbd import CameraModel, RGBDFrame, save_frames, validate_frames
from components.transforms import (
    apply,
    interpolate_T,
    invert,
    make_T,
    ov_to_matrix,
    rotation_angle_deg,
)

ENABLE_ENV = "ENABLE_CALIBRATED_POUR"
LIQUID_ENV = "POUR_ALLOW_LIQUID"


def _flag(name: str, env=None) -> bool:
    return str((env or os.environ).get(name, "")).strip().lower() in ("1", "true", "yes")


def pour_enabled(env=None) -> bool:
    return _flag(ENABLE_ENV, env)


class S(str, enum.Enum):
    IDLE = "IDLE"
    OBSERVE = "OBSERVE"
    VALIDATE_CALIBRATION = "VALIDATE_CALIBRATION"
    LOCALIZE_SOURCE_AND_CUP = "LOCALIZE_SOURCE_AND_CUP"
    PLAN_GRASP = "PLAN_GRASP"
    HOVER = "HOVER"
    PREGRASP = "PREGRASP"
    GRASP = "GRASP"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    REOBSERVE_CUP = "REOBSERVE_CUP"
    PLAN_POUR = "PLAN_POUR"
    PREPOUR = "PREPOUR"
    TILT_INCREMENTS = "TILT_INCREMENTS"
    HOLD = "HOLD"
    UNTILT = "UNTILT"
    RETREAT = "RETREAT"
    SAFE_PLACE = "SAFE_PLACE"
    DONE = "DONE"
    RECOVER = "RECOVER"
    ABORTED = "ABORTED"


class PourMode(str, enum.Enum):
    PLAN = "plan"                # observe + localize + plan, never moves (dry-run)
    HOVER = "hover"              # stage D: hover >= 30 mm above predicted mouth / rim
    GRASP = "grasp"              # stage E: side grasp, lift, put back
    POUR_DRY = "pour_dry"        # stage F: full trajectory, empty container
    POUR_LIQUID = "pour_liquid"  # stage G: water trials (tray, e-stop, mentor)
    POUR = "pour"                # production (voice / orchestrator)


MODE_STAGES = {
    PourMode.PLAN: (),
    PourMode.HOVER: ("B", "C"),
    PourMode.GRASP: ("B", "C", "D"),
    PourMode.POUR_DRY: ("B", "C", "D", "E"),
    PourMode.POUR_LIQUID: ("B", "C", "D", "E", "F"),
    PourMode.POUR: ("B", "C", "D", "E", "F", "G"),
}
LIQUID_MODES = (PourMode.POUR_LIQUID, PourMode.POUR)
LIQUID_CONFIRMATIONS = ("tray", "estop_attended", "mentor_approved", "bounded_amount")


@dataclass
class PourRequest:
    source: str = "any"                   # bottle | can | any
    target: str = "cup"
    mode: PourMode = PourMode.PLAN
    source_px: Optional[tuple] = None     # explicit user selection (pixel in the observation)
    cup_px: Optional[tuple] = None
    confirmations: dict = field(default_factory=dict)
    n_frames: int = 3


@dataclass
class PourResult:
    state: str
    success: bool
    reason: Optional[str] = None
    detail: dict = field(default_factory=dict)
    help_required: bool = False
    mode: str = ""
    run_dir: Optional[str] = None
    transitions: list = field(default_factory=list)
    state_times_s: dict = field(default_factory=dict)
    selection: Optional[dict] = None
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class PourAbort(Exception):
    def __init__(self, reason: str, detail: Optional[dict] = None, *, stop_motion: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}
        self.stop_motion = stop_motion   # True = do not try to rewind (possible collision / fault)


# ---------------------------------------------------------------------------
# Ports (live implementations below; tests inject fakes)
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    frames: list
    candidates: list          # [{"label", "mask", "box", "score"}]
    live: Optional[calib.LiveIdentity]
    t_mono: float
    notes: list = field(default_factory=list)


class RobotPort:  # pragma: no cover - interface
    async def flange_pose(self) -> np.ndarray: ...
    async def joints(self) -> list: ...
    async def move_flange(self, T_world_flange: np.ndarray, timeout: float) -> None: ...
    async def go_home(self) -> None: ...
    async def gripper_open(self) -> None: ...
    async def gripper_grab(self) -> tuple[Optional[bool], float]: ...
    async def gripper_pos(self) -> float: ...
    async def fault(self) -> Optional[str]: ...
    async def stop(self) -> None: ...


class PerceptionPort:  # pragma: no cover - interface
    async def observe(self, n_frames: int, *, aligned: bool) -> Observation: ...
    async def capture(self, n_frames: int, *, aligned: bool) -> Observation: ...


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def readiness_problems(mode: PourMode, *, setup_doc=None, trials=None, env=None,
                       confirmations: Optional[dict] = None, live=None) -> tuple[Optional[calib.PourSetup], list]:
    """Everything that blocks ``mode`` (empty list = ready). Offline when
    ``live`` is None (voice uses this before connecting)."""
    env = env if env is not None else os.environ
    problems = []
    if mode != PourMode.PLAN or live is not None:
        if not pour_enabled(env):
            problems.append(f"{ENABLE_ENV}=1 is not set")
    try:
        setup = calib.check_setup(setup_doc, live=live)
    except calib.CalibrationError as exc:
        return None, problems + exc.problems
    missing = calib.stages_passed(MODE_STAGES[mode], setup.setup_hash, trials)
    if missing:
        problems.append("hardware stages not passed for this setup: " + ",".join(missing))
    if mode in LIQUID_MODES:
        if not _flag(LIQUID_ENV, env):
            problems.append(f"{LIQUID_ENV}=1 is not set (real liquid disabled)")
        if mode == PourMode.POUR_LIQUID:
            conf = confirmations or {}
            lacking = [k for k in LIQUID_CONFIRMATIONS if not conf.get(k)]
            if lacking:
                problems.append("operator confirmations missing: " + ",".join(lacking))
    return setup, problems


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class _Step:
    tag: str
    T: np.ndarray
    jaw: float
    holding: bool
    table_ok: bool = False


HOME_TOOL_R = ov_to_matrix(-0.03189537706111025, 0.01726517715312809, -0.999342082862521, -26.77995529670437)


class PourController:
    def __init__(
        self,
        robot: Optional[RobotPort],
        perception: PerceptionPort,
        *,
        params: Optional[PourParams] = None,
        setup_doc=None,
        trials: Optional[list] = None,
        operator: Any = None,
        run_dir: Optional[Path] = None,
        save_evidence: bool = True,
        env=None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable = asyncio.sleep,
        observation_joints: Optional[list] = None,
        nominal: bool = False,
        skip_reobserve: bool = False,
    ):
        if skip_reobserve and not nominal:
            raise ValueError("skip_reobserve is only available in nominal (uncalibrated) mode")
        self.nominal = nominal
        self.skip_reobserve = skip_reobserve
        if nominal:
            params = params or PourParams()
            # The camera mount is not calibrated: two viewpoints disagree more.
            params.reobserve_tol_xy_mm = max(params.reobserve_tol_xy_mm, 15.0)
            params.reobserve_tol_r_mm = max(params.reobserve_tol_r_mm, 8.0)
            params.reobserve_tol_z_mm = max(params.reobserve_tol_z_mm, 15.0)
        self.robot = robot
        self.perception = perception
        self.params = (params or PourParams()).validate()
        self.setup_doc = setup_doc
        self.trials = trials
        self.operator = operator
        self.env = env if env is not None else os.environ
        self.clock = clock
        self.sleep = sleep
        self.save_evidence = save_evidence
        self.run_dir = run_dir
        self.observation_joints = observation_joints
        self._reset()

    # -- bookkeeping --------------------------------------------------------

    def _reset(self) -> None:
        self.state = S.IDLE
        self.log: Optional[PourLog] = None
        self.setup: Optional[calib.PourSetup] = None
        self.obs: Optional[Observation] = None
        self.estimates: list = []
        self.selection: Optional[PairSelection] = None
        self.scene: Optional[Scene] = None
        self.grasp: Optional[GraspPlan] = None
        self.pour: Optional[PourPlan] = None
        self.cup: Optional[ObjectPoseEstimate] = None
        self.t_obs = None
        self.t_cup = None
        self.history: list[_Step] = []
        self.holding = False
        self.moved = False
        self.notes: list = []
        self.req: Optional[PourRequest] = None

    def _enter(self, st: S, **data) -> None:
        self.state = st
        if self.log:
            self.log.state(st.value, **data)

    # -- entry --------------------------------------------------------------

    async def run(self, req: PourRequest) -> PourResult:
        self._reset()
        self.req = req
        mode = PourMode(req.mode)
        if self.save_evidence and self.run_dir is None:
            self.run_dir = new_run_dir(tag=mode.value)
        self.log = PourLog(self.run_dir if self.save_evidence else None, clock=self.clock)
        self._enter(S.IDLE, request={
            "source": req.source, "target": req.target, "mode": mode.value,
            "source_px": req.source_px, "cup_px": req.cup_px,
        })
        try:
            self._check_request(req, mode)
            await self._observe(req, mode)
            self._validate_calibration(req, mode)
            self._localize(req)
            self._plan_grasp()
            if mode == PourMode.PLAN:
                return self._finish(True)
            if mode == PourMode.HOVER:
                await self._hover()
                return self._finish(True)
            await self._pregrasp()
            await self._grasp()
            await self._verify_grasp()
            await self._lift()
            if mode == PourMode.GRASP:
                await self._safe_place(from_lift=True)
                return self._finish(True)
            await self._reobserve_cup(mode)
            self._plan_pour()
            await self._prepour()
            await self._tilt()
            await self._hold(mode)
            await self._untilt()
            await self._retreat()
            await self._safe_place(from_lift=False)
            return self._finish(True)
        except PourAbort as exc:
            return await self._abort(exc)
        except PlanningError as exc:
            return await self._abort(PourAbort(exc.reason, exc.detail))
        except calib.CalibrationError as exc:
            return await self._abort(PourAbort("calibration_invalid", {"problems": exc.problems}))
        except Exception as exc:  # noqa: BLE001 - every failure must end in recovery
            return await self._abort(PourAbort("unexpected_error", {
                "error": repr(exc), "trace": traceback.format_exc(limit=6)}))

    def _finish(self, ok: bool) -> PourResult:
        self._enter(S.DONE)
        self.log.close_state()
        res = PourResult(
            state=S.DONE.value, success=ok, mode=PourMode(self.req.mode).value,
            run_dir=str(self.run_dir) if self.run_dir else None,
            transitions=[e["state"] for e in self.log.events if e["event"] == "state"],
            state_times_s=dict(self.log.state_times),
            selection=None if self.selection is None else {"ok": self.selection.ok, "reason": self.selection.reason},
            notes=list(self.notes),
        )
        self._write_summary(res)
        return res

    async def _abort(self, exc: PourAbort) -> PourResult:
        failed_in = self.state.value
        self.log.event("abort", reason=exc.reason, detail=exc.detail, state=failed_in)
        help_required = await self._recover(exc)
        self._enter(S.ABORTED, reason=exc.reason, failed_in=failed_in, help_required=help_required)
        self.log.close_state()
        res = PourResult(
            state=S.ABORTED.value, success=False, reason=exc.reason,
            detail={"failed_in": failed_in, **exc.detail}, help_required=help_required,
            mode=PourMode(self.req.mode).value if self.req else "",
            run_dir=str(self.run_dir) if self.run_dir else None,
            transitions=[e["state"] for e in self.log.events if e["event"] == "state"],
            state_times_s=dict(self.log.state_times),
            selection=None if self.selection is None else {"ok": self.selection.ok, "reason": self.selection.reason},
            notes=list(self.notes),
        )
        self._write_summary(res)
        return res

    def _write_summary(self, res: PourResult) -> None:
        if self.run_dir and self.save_evidence:
            dump_json(Path(self.run_dir) / "summary.json", res.as_dict())

    # -- gates --------------------------------------------------------------

    def _check_request(self, req: PourRequest, mode: PourMode) -> None:
        if req.target != CUP_LABEL:
            raise PourAbort("invalid_target", {"target": req.target, "allowed": [CUP_LABEL]})
        if req.source not in SOURCE_LABELS + ("any",):
            raise PourAbort("invalid_source", {"source": req.source})
        if mode != PourMode.PLAN and self.robot is None:
            raise PourAbort("no_robot_for_motion_mode")
        if (mode != PourMode.PLAN or self.robot is not None) and not pour_enabled(self.env):
            raise PourAbort("pour_disabled", {"hint": f"set {ENABLE_ENV}=1"})
        if mode in LIQUID_MODES and not _flag(LIQUID_ENV, self.env):
            raise PourAbort("liquid_disabled", {"hint": f"set {LIQUID_ENV}=1 only after stage F passes"})
        if mode in LIQUID_MODES and self.params.hold_s > self.params.liquid_max_hold_s:
            # Open-loop pour: duration is the only thing bounding the amount.
            raise PourAbort("hold_too_long", {"hold_s": self.params.hold_s, "max_s": self.params.liquid_max_hold_s})

    async def _guard(self, *, needs: tuple = ()) -> None:
        """Checks before every phase: robot fault, calibration file unchanged,
        perception freshness."""
        if self.robot is not None:
            fault = await self.robot.fault()
            if fault:
                raise PourAbort("robot_fault", {"fault": fault}, stop_motion=True)
        if self.setup is not None and self.setup_doc is None:
            try:
                doc = calib.load_setup_doc()
                if calib.compute_setup_hash(doc) != self.setup.setup_hash:
                    raise PourAbort("calibration_changed_during_run")
            except calib.CalibrationError as exc:
                raise PourAbort("calibration_changed_during_run", {"problems": exc.problems})
        now = self.clock()
        if "source" in needs and (self.t_obs is None or now - self.t_obs > self.params.max_perception_age_s):
            raise PourAbort("stale_perception", {"what": "source", "age_s": None if self.t_obs is None else now - self.t_obs})
        if "cup" in needs and (self.t_cup is None or now - self.t_cup > self.params.max_perception_age_s):
            raise PourAbort("stale_perception", {"what": "cup", "age_s": None if self.t_cup is None else now - self.t_cup})

    # -- OBSERVE / VALIDATE / LOCALIZE / PLAN --------------------------------

    def _declared_alignment(self) -> bool:
        if self.nominal:
            # No calibration record declares it; the per-object color/depth
            # registration gate in LOCALIZE decides (misregistered -> no pour).
            return True
        try:
            doc = self.setup_doc if isinstance(self.setup_doc, dict) else calib.load_setup_doc(self.setup_doc)
            return bool(((doc.get("camera") or {}).get("alignment") or {}).get("verified"))
        except calib.CalibrationError:
            return False

    async def _observe(self, req: PourRequest, mode: PourMode) -> None:
        self._enter(S.OBSERVE)
        if self.robot is not None and self.observation_joints is not None:
            j = await self.robot.joints()
            off = max(abs(a - b) for a, b in zip(j, self.observation_joints))
            if off > 2.0:
                if mode == PourMode.PLAN:
                    raise PourAbort("not_at_observation_pose", {"joint_offset_deg": round(off, 2),
                                                                  "hint": "run scripts/go_home.py first"})
                T = await self.robot.flange_pose()
                if T[2, 3] < 300.0:
                    raise PourAbort("unsafe_to_home_from_here", {"flange_z_mm": round(float(T[2, 3]), 1),
                                                                  "hint": "jog the arm up, then retry"})
                await self.robot.go_home()
                self.moved = True
        self.obs = await self.perception.observe(req.n_frames, aligned=self._declared_alignment())
        self.t_obs = self.obs.t_mono
        self.notes += self.obs.notes
        if self.run_dir and self.save_evidence:
            save_frames(Path(self.run_dir), self.obs.frames, "obs")
            save_candidates(Path(self.run_dir), self.obs.candidates)
        self.log.event("observed", frames=len(self.obs.frames), candidates=[c["label"] for c in self.obs.candidates])

    def _validate_calibration(self, req: PourRequest, mode: PourMode) -> None:
        self._enter(S.VALIDATE_CALIBRATION, nominal=self.nominal)
        if self.nominal:
            self._validate_nominal(req, mode)
            return
        setup, problems = readiness_problems(
            mode, setup_doc=self.setup_doc, trials=self.trials, env=self.env,
            confirmations=req.confirmations, live=self.obs.live,
        )
        if setup is not None:
            fp = validate_frames(self.obs.frames, CameraModel.from_setup(setup))
            problems += fp
        if problems or setup is None:
            raise PourAbort("not_ready", {"problems": problems})
        self.setup = setup
        self.log.event("calibration_ok", setup_id=setup.setup_id, setup_hash=setup.setup_hash)

    def _validate_nominal(self, req: PourRequest, mode: PourMode) -> None:
        """Operator skipped calibration: build the setup from the live machine
        (components/pour_nominal.py). Stage gates are skipped; the enable and
        liquid flags, frame checks, and every geometric check still apply."""
        from components.pour_nominal import build_nominal_doc

        try:
            doc, rep = build_nominal_doc(self.obs)
        except ValueError as exc:
            raise PourAbort("nominal_setup_failed", {"error": str(exc)})
        self.setup_doc = doc
        if self.run_dir and self.save_evidence:
            dump_json(Path(self.run_dir) / "nominal_setup.json", {"report": rep, "doc": doc})
        problems = []
        if mode == PourMode.POUR_LIQUID:
            lacking = [k for k in LIQUID_CONFIRMATIONS if not (req.confirmations or {}).get(k)]
            if lacking:
                problems.append("operator confirmations missing: " + ",".join(lacking))
        try:
            setup = calib.check_setup(doc, live=self.obs.live, allow_nominal=True)
        except calib.CalibrationError as exc:
            raise PourAbort("not_ready", {"problems": problems + exc.problems})
        problems += validate_frames(self.obs.frames, CameraModel.from_setup(setup))
        if problems:
            raise PourAbort("not_ready", {"problems": problems})
        self.setup = setup
        self.notes.append("NOMINAL (uncalibrated) setup: expect cm-level error")
        self.log.event("nominal_setup", **rep)

    def _localize(self, req: PourRequest) -> None:
        self._enter(S.LOCALIZE_SOURCE_AND_CUP)
        setup = self.setup
        table = TableFrame(setup.table_normal, setup.table_d)
        cal_err = setup.validation.get("touchoff") or {}
        est = []
        for i, c in enumerate(self.obs.candidates):
            label = c["label"]
            max_grasp = setup.gripper.max_open_mm - 2 * setup.gripper.finger_radius_mm - 6.0
            if label in ("can", CUP_LABEL):
                # The detector confuses dark cups and cans; the geometry decides:
                # an open container (interior well below the rim) is a cup,
                # anything else is at most an upright can.
                e = estimate_cup(self.obs.frames, c["mask"], setup_hash=setup.setup_hash, table=table,
                                 calib_err=cal_err, box=c.get("box"))
                if not e.ok and label == "can":
                    e = estimate_upright_source("can", self.obs.frames, c["mask"], setup_hash=setup.setup_hash,
                                                table=table, calib_err=cal_err, box=c.get("box"),
                                                max_graspable_mm=max_grasp)
                elif e.ok and label == "can":
                    e.quality_flags.append("detector_said_can_geometry_says_cup")
            elif label in SOURCE_LABELS:
                e = estimate_upright_source(
                    label, self.obs.frames, c["mask"], setup_hash=setup.setup_hash, table=table,
                    calib_err=cal_err, box=c.get("box"), max_graspable_mm=max_grasp,
                )
            else:
                continue
            e.extra["candidate_index"] = i
            est.append(e)
        self.estimates = est
        self.selection = select_pour_pair(est, req.source, source_px=req.source_px, cup_px=req.cup_px)
        if self.run_dir and self.save_evidence:
            dump_json(Path(self.run_dir) / "estimates.json", [e.summary() for e in est])
        self.log.event("localized", selection={"ok": self.selection.ok, "reason": self.selection.reason},
                       estimates=[{"label": e.label, "ok": e.ok, "reason": e.rejection_reason} for e in est])
        if not self.selection.ok:
            self._write_overlay()
            detail = {"estimates": [e.summary() for e in est]}
            if est and all(e.rejection_reason == "depth_color_misregistered" for e in est):
                detail["fix"] = ("depth is not aligned to color: set align_color_depth: true on the cam "
                                 "(fragment_mods on this machine), then retry")
            raise PourAbort(self.selection.reason, detail)
        self.cup = self.selection.cup
        others = []
        for i, e in enumerate(est):
            if e is self.selection.source or e is self.selection.cup:
                continue
            cyl = cylinder_for(e, table)
            if cyl is None:
                raise PourAbort("unmodelled_obstacle", {"label": e.label, "reason": e.rejection_reason,
                                                        "hint": "remove it or make its depth observable"})
            cyl.name = f"obj{i}:{e.label}"
            others.append(cyl)
        self.scene = Scene(table=table, cup=cylinder_for(self.cup, table), others=others,
                           boxes=boxes_from_setup(setup))

    def _plan_grasp(self) -> None:
        self._enter(S.PLAN_GRASP)
        src = self.selection.source
        self.grasp = plan_side_grasp(self.setup, self.scene, src, self.params)
        provisional = None
        try:
            provisional = plan_pour(self.setup, self.scene, self.grasp, self.cup, self.params, source_label=src.label)
        except PlanningError as exc:
            self.notes.append(f"provisional pour plan infeasible: {exc.reason}")
        self.pour = provisional
        self._write_plan(provisional=True)
        self._write_overlay()
        self.log.event("grasp_planned", grasp=self.grasp.summary(),
                       provisional_pour=None if provisional is None else provisional.summary())
        if provisional is None:
            raise PourAbort("no_feasible_pour_path", {"notes": self.notes})

    def _write_plan(self, provisional: bool) -> None:
        if not (self.run_dir and self.save_evidence):
            return
        pj = plan_json(self.setup, self.grasp, self.pour, self.selection, self.params)
        pj["provisional"] = provisional
        dump_json(Path(self.run_dir) / ("plan.json" if not provisional else "plan_provisional.json"), pj)
        import cv2

        top, side = draw_plan_views(self.grasp, self.pour, self.cup, self.selection.source if self.selection else None)
        cv2.imwrite(str(Path(self.run_dir) / "plan_top.png"), top)
        cv2.imwrite(str(Path(self.run_dir) / "plan_side.png"), side)

    def _write_overlay(self) -> None:
        if not (self.run_dir and self.save_evidence and self.obs and self.obs.frames):
            return
        import cv2

        vis = draw_overlay(self.obs.frames[0], self.estimates, self.grasp, self.pour, self.setup, self.selection)
        cv2.imwrite(str(Path(self.run_dir) / "overlay.png"), vis)

    # -- motion primitives --------------------------------------------------

    def _validate_step(self, st: _Step) -> None:
        g = self.grasp
        check_pose(
            self.setup, self.scene, self.params, st.T, st.jaw,
            bottle=g.bottle if (st.holding and g is not None) else None,
            T_tcp_B=g.T_tcp_B if (st.holding and g is not None) else None,
            bottle_on_table_ok=st.table_ok, label=st.tag,
        )

    async def _move(self, st: _Step, *, record: bool = True, validate: bool = True) -> None:
        if validate:
            self._validate_step(st)
        T_flange = st.T @ invert(self.setup.T_flange_tcp)
        self.moved = True  # a command that raises may still have started moving
        await self.robot.move_flange(T_flange, timeout=30.0)
        actual = await self.robot.flange_pose()
        err_mm = float(np.linalg.norm(actual[:3, 3] - T_flange[:3, 3]))
        err_deg = rotation_angle_deg(actual[:3, :3], T_flange[:3, :3])
        self.log.command(st.tag, st.T, T_flange, actual, round(err_mm, 3), round(err_deg, 3))
        if record:
            self.history.append(st)
        if err_mm > self.params.tracking_tol_mm or err_deg > self.params.tracking_tol_deg:
            raise PourAbort("tracking_error", {"tag": st.tag, "err_mm": err_mm, "err_deg": err_deg},
                            stop_motion=True)
        if self.params.step_dwell_s:
            await self.sleep(self.params.step_dwell_s)

    async def _path(self, steps: list) -> None:
        for st in steps:
            await self._move(st)

    def _steps(self, Ts, tag, jaw, holding, table_ok=False) -> list:
        return [_Step(f"{tag}{i}", T, jaw, holding, table_ok) for i, T in enumerate(Ts)]

    # -- phases -------------------------------------------------------------

    async def _hover(self) -> None:
        self._enter(S.HOVER)
        await self._guard(needs=("source",))
        start = await self.robot.flange_pose()
        T_tcp0 = start @ self.setup.T_flange_tcp
        R = T_tcp0[:3, :3]
        if rotation_angle_deg(R, HOME_TOOL_R) > 10.0:
            R = HOME_TOOL_R
        targets = hover_targets(self.setup, self.scene, self.selection.source, self.cup, self.params, R)
        results = []
        for name, T, truth in targets:
            path = interpolate_T(T_tcp0, T, self.params.max_step_mm, self.params.max_step_deg)
            await self._path(self._steps(path, f"hover_{name}_", self.setup.gripper.max_open_mm, False))
            tip = T[:3, 3] - np.array([0, 0, self.setup.gripper.pad_length_mm / 2.0])
            measured = None
            if self.operator is not None:
                measured = await self.operator.measure_hover(name, predicted=truth, fingertip=tip)
            results.append({"target": name, "predicted_mm": truth.round(2).tolist(),
                            "fingertip_mm": tip.round(2).tolist(), "operator_measured_error_mm": measured})
            back = interpolate_T(T, T_tcp0, self.params.max_step_mm, self.params.max_step_deg)
            await self._path(self._steps(back, f"hover_{name}_back", self.setup.gripper.max_open_mm, False))
        self.log.event("hover_results", results=results)

    async def _pregrasp(self) -> None:
        self._enter(S.PREGRASP)
        await self._guard(needs=("source",))
        await self.robot.gripper_open()
        g = self.grasp
        # From the observation pose to above the pregrasp, then down to it.
        start = (await self.robot.flange_pose()) @ self.setup.T_flange_tcp
        above = make_T(g.T_pregrasp[:3, :3], g.T_pregrasp[:3, 3] + np.array([0, 0, g.transit_mouth_z - g.T_pregrasp[2, 3]]))
        path = interpolate_T(start, above, self.params.max_step_mm, self.params.max_step_deg)
        path += interpolate_T(above, g.T_pregrasp, self.params.max_step_mm, self.params.max_step_deg)
        await self._path(self._steps(path, "to_pregrasp", self.setup.gripper.max_open_mm, False))

    async def _grasp(self) -> None:
        self._enter(S.GRASP)
        await self._guard(needs=("source",))
        steps = self._steps(self.grasp.approach_path[1:], "approach", self.setup.gripper.max_open_mm, False)
        await self._path(steps)
        self.history.append(_Step("grasp", self.grasp.T_grasp, self.setup.gripper.max_open_mm, False))
        holding, pos = await self.robot.gripper_grab()
        self.holding = True  # jaws are closed on (or near) the object from here on
        self._grab = (holding, pos)

    async def _verify_grasp(self) -> None:
        self._enter(S.VERIFY_GRASP)
        holding, pos = self._grab
        width = self.setup.gripper.jaw_width_mm(pos)
        exp = self.grasp.expected_jaw_mm
        self.log.event("grasp_result", holding=holding, pos=pos, jaw_mm=width, expected_mm=exp)
        if holding is False or abs(width - exp) > self.params.jaw_tolerance_mm:
            raise PourAbort("grasp_failed", {"holding": holding, "jaw_mm": width, "expected_mm": exp})
        self._jaw_after_grasp = width

    async def _lift(self) -> None:
        self._enter(S.LIFT)
        await self._guard()
        D = self.grasp.expected_jaw_mm
        await self._path(self._steps(self.grasp.lift_path, "lift", D, True, table_ok=True))
        pos = await self.robot.gripper_pos()
        width = self.setup.gripper.jaw_width_mm(pos)
        if abs(width - self._jaw_after_grasp) > 3.0:
            raise PourAbort("slip_detected", {"jaw_before_mm": self._jaw_after_grasp, "jaw_after_mm": width})

    async def _reobserve_cup(self, mode: PourMode) -> None:
        self._enter(S.REOBSERVE_CUP)
        await self._guard()
        if self.skip_reobserve:
            self.log.event("reobserve_skipped", reason="operator flag (nominal mode)")
            self.notes.append("cup NOT re-observed after lift (operator skipped it)")
            self.t_cup = self.t_obs
            return
        liquid = mode in LIQUID_MODES
        model = CameraModel.from_setup(self.setup)
        view = plan_reobserve_view(self.setup, self.scene, self.grasp, self.cup, self.params, liquid=liquid,
                                   camera_model=model, source_label=self.selection.source.label)
        D = self.grasp.expected_jaw_mm
        await self._path(self._steps(view.path_in, "reobs_in", D, True))
        obs = await self.perception.capture(self.req.n_frames, aligned=self._declared_alignment())
        fp = validate_frames(obs.frames, model)
        if fp:
            raise PourAbort("reobservation_frames_invalid", {"problems": fp})
        if self.run_dir and self.save_evidence:
            save_frames(Path(self.run_dir), obs.frames, "reobs")
        from components.object_pose import cup_mask_from_prior

        table = self.scene.table
        T_world_tcp = obs.frames[0].T_world_flange @ self.setup.T_flange_tcp
        occl = gripper_spheres(self.setup, T_world_tcp, D) + self.grasp.bottle.spheres(T_world_tcp @ self.grasp.T_tcp_B)
        mask, diag = cup_mask_from_prior(obs.frames, self.cup, table, occl)
        fresh = estimate_cup(obs.frames, mask, setup_hash=self.setup.setup_hash, table=table,
                             calib_err=self.setup.validation.get("touchoff") or {}, mask_from_depth=True,
                             require_interior=False)
        fresh.extra.update(diag)
        if self.run_dir and self.save_evidence:
            dump_json(Path(self.run_dir) / "reobserved_cup.json", fresh.summary())
        if not fresh.ok:
            raise PourAbort("reobservation_failed", {"reason": fresh.rejection_reason, **diag})
        prior = self.cup
        dxy = float(np.linalg.norm(fresh.position[:2] - prior.position[:2]))
        dr = abs(float(fresh.dims["rim_radius"]) - float(prior.dims["rim_radius"]))
        dz = abs(float(fresh.position[2]) - float(prior.position[2]))
        agree = {"dxy_mm": round(dxy, 2), "dr_mm": round(dr, 2), "dz_mm": round(dz, 2)}
        self.log.event("reobserved_cup", **agree, view_tilt_deg=view.bottle_tilt_deg)
        if dxy > self.params.reobserve_tol_xy_mm or dr > self.params.reobserve_tol_r_mm or dz > self.params.reobserve_tol_z_mm:
            raise PourAbort("reobservation_disagreement", agree)
        self.cup = fresh
        self.t_cup = obs.t_mono
        self.scene.cup = cylinder_for(fresh, table)
        await self._path(self._steps(view.path_out, "reobs_out", D, True))

    def _plan_pour(self) -> None:
        self._enter(S.PLAN_POUR)
        self.pour = plan_pour(self.setup, self.scene, self.grasp, self.cup, self.params,
                              source_label=self.selection.source.label)
        self._write_plan(provisional=False)
        self._write_overlay()
        self.log.event("pour_planned", pour=self.pour.summary())

    async def _prepour(self) -> None:
        self._enter(S.PREPOUR)
        await self._guard(needs=("cup",))
        D = self.grasp.expected_jaw_mm
        await self._path(self._steps(self.pour.transit_path, "transit", D, True))

    async def _tilt(self) -> None:
        self._enter(S.TILT_INCREMENTS)
        await self._guard(needs=("cup",))
        D = self.grasp.expected_jaw_mm
        for i, T in enumerate(self.pour.T_tcp[1:], start=1):
            await self._guard(needs=("cup",))
            await self._move(_Step(f"tilt{self.pour.tilt_deg[i]:.0f}", T, D, True))

    async def _hold(self, mode: PourMode) -> None:
        self._enter(S.HOLD, open_loop=True, hold_s=self.params.hold_s, liquid=mode in LIQUID_MODES)
        await self.sleep(self.params.hold_s)

    async def _untilt(self) -> None:
        self._enter(S.UNTILT)
        D = self.grasp.expected_jaw_mm
        back = list(reversed(self.pour.T_tcp[:-1]))
        await self._path(self._steps(back, "untilt", D, True))

    async def _retreat(self) -> None:
        self._enter(S.RETREAT)
        D = self.grasp.expected_jaw_mm
        await self._path(self._steps(self.pour.retreat_path, "retreat", D, True))

    async def _safe_place(self, *, from_lift: bool) -> None:
        self._enter(S.SAFE_PLACE)
        D = self.grasp.expected_jaw_mm
        if from_lift:
            down = list(reversed(self.grasp.lift_path[:-1])) + [self.grasp.T_grasp]
            await self._path(self._steps(down, "set_down", D, True, table_ok=True))
        else:
            n_down = len(self.grasp.lift_path)
            ret = self.pour.return_path
            head, tail = ret[: len(ret) - n_down], ret[len(ret) - n_down:]
            await self._path(self._steps(head, "return", D, True))
            await self._path(self._steps(tail, "set_down", D, True, table_ok=True))
        await self.robot.gripper_open()
        self.holding = False
        back = self._steps(self.grasp.retreat_path, "back_off", self.setup.gripper.max_open_mm, False)
        await self._path(back)
        await self._clear_up()

    async def _clear_up(self) -> None:
        """Straight up to transit height (validated), then the taught joint
        move home. The joint move is the existing sorter's home move."""
        T = (await self.robot.flange_pose()) @ self.setup.T_flange_tcp
        z_up = max(self.grasp.transit_mouth_z, float(T[2, 3]))
        up = make_T(T[:3, :3], np.array([T[0, 3], T[1, 3], z_up]))
        if not np.allclose(up, T):
            await self._path(self._steps(interpolate_T(T, up, self.params.max_step_mm, self.params.max_step_deg),
                                         "clear_up", self.setup.gripper.max_open_mm, False))
        await self.robot.go_home()

    # -- recovery -----------------------------------------------------------

    async def _recover(self, exc: PourAbort) -> bool:
        """Returns help_required."""
        self._enter(S.RECOVER, reason=exc.reason)
        if self.robot is None or not self.moved:
            return False
        if exc.stop_motion:
            try:
                await self.robot.stop()
            finally:
                self.log.event("recover_stopped", reason="possible collision or fault; not moving")
            return True
        try:
            fault = await self.robot.fault()
            if fault:
                await self.robot.stop()
                self.log.event("recover_stopped", reason=f"robot fault: {fault}")
                return True
            # Rewind every executed pose (all were validated when commanded).
            steps = list(self.history)
            grasp_idx = next((i for i, s in enumerate(steps) if s.tag == "grasp"), None)
            for i in range(len(steps) - 1, -1, -1):
                st = steps[i]
                if grasp_idx is not None and i == grasp_idx and self.holding:
                    await self._move(_Step("rewind_grasp", st.T, st.jaw, True, True), record=False, validate=False)
                    await self.robot.gripper_open()
                    self.holding = False
                    self.log.event("recover_released_at_pick_location")
                    continue
                await self._move(_Step(f"rewind_{st.tag}", st.T, st.jaw, st.holding and self.holding, st.table_ok),
                                 record=False, validate=False)
            if self.holding:
                await self.robot.gripper_open()
                self.holding = False
            await self.robot.go_home()
            self.log.event("recover_done")
            return False
        except Exception as rexc:  # noqa: BLE001
            try:
                await self.robot.stop()
            except Exception:  # noqa: BLE001
                pass
            self.log.event("recover_failed", error=repr(rexc))
            return True


# ---------------------------------------------------------------------------
# Live ports (Viam)
# ---------------------------------------------------------------------------


class LiveRobot(RobotPort):
    def __init__(self, arm, gripper):
        self.arm = arm          # components.arm.ArmComponent
        self.gripper = gripper  # components.gripper.GripperComponent

    async def flange_pose(self) -> np.ndarray:
        from components.transforms import pose_to_T

        return pose_to_T(await self.arm.get_end_position())

    async def joints(self) -> list:
        return await self.arm.get_joint_positions()

    async def move_flange(self, T_world_flange: np.ndarray, timeout: float) -> None:
        from components.transforms import T_to_pose

        p = T_to_pose(T_world_flange)
        # floor=None: no silent Z clamp. The planner validated the whole
        # collision envelope against the calibrated table plane instead.
        await self.arm.move_to_position(p["x"], p["y"], p["z"], o_x=p["o_x"], o_y=p["o_y"], o_z=p["o_z"],
                                        theta=p["theta"], timeout=timeout, check_workspace=False, floor=None)

    async def go_home(self) -> None:
        await self.arm.go_home()

    async def gripper_open(self) -> None:
        await self.gripper.open_full()

    async def gripper_grab(self) -> tuple[Optional[bool], float]:
        g = await self.gripper.grab()
        return g.holding, float(g.pos)

    async def gripper_pos(self) -> float:
        return float(await self.gripper.get_pos())

    async def fault(self) -> Optional[str]:
        # The Viam arm API exposes no fault / e-stop state for this driver;
        # tracking error after every move is the runtime proxy. A human on
        # the e-stop is required for every motion stage.
        return None

    async def stop(self) -> None:
        try:
            await self.arm._arm.stop()
        except Exception:  # noqa: BLE001
            pass


class LivePerception(PerceptionPort):
    def __init__(self, machine, *, camera_name: str, arm_name: str, detect_labels=("bottle", "can", "cup")):
        self.machine = machine
        self.camera_name = camera_name
        self.arm_name = arm_name
        self.detect_labels = tuple(detect_labels)

    async def _capture(self, n_frames: int, aligned: bool):
        from components.rgbd import (
            camera_mount_from_frame_system,
            capture_rgbd_set,
            reported_extrinsics,
        )
        from components.transforms import pose_to_T

        frames, props = await capture_rgbd_set(
            self.machine, camera_name=self.camera_name, arm_name=self.arm_name,
            n_frames=n_frames, aligned_to_color=aligned,
        )
        T_parent_cam, parent = await camera_mount_from_frame_system(self.machine, self.camera_name, self.arm_name)
        notes = []
        # Cross-check the frame system's flange with the arm driver's end pose.
        from viam.components.arm import Arm

        T_end = pose_to_T(await Arm.from_robot(self.machine, self.arm_name).get_end_position())
        T_fs = frames[0].T_world_flange
        d = float(np.linalg.norm(T_end[:3, 3] - T_fs[:3, 3]))
        if d > 2.0 or rotation_angle_deg(T_end[:3, :3], T_fs[:3, :3]) > 0.5:
            # Perception goes through the frame system, motion through the
            # driver: they must describe the same flange or nothing lines up.
            raise PourAbort("frame_system_disagrees_with_arm_driver", {"flange_delta_mm": round(d, 2)})
        notes.append(f"frame-system flange vs arm end pose: {d:.2f} mm")
        f0 = frames[0]
        live = calib.LiveIdentity(
            color_size=(f0.color.shape[1], f0.color.shape[0]),
            depth_size=(f0.depth_mm.shape[1], f0.depth_mm.shape[0]),
            intrinsics={"fx": f0.model.fx, "fy": f0.model.fy, "cx": f0.model.cx, "cy": f0.model.cy,
                        "width": f0.model.width, "height": f0.model.height},
            distortion={"model": f0.model.dist_model, "coeffs": list(f0.model.coeffs)},
            reported_extrinsics=reported_extrinsics(props),
            depth_encoding=f0.depth_encoding,
            T_flange_cam_viam=T_parent_cam if parent == self.arm_name else None,
            cam_parent_is_arm=(parent == self.arm_name) if parent else None,
        )
        return frames, live, notes

    async def observe(self, n_frames: int, *, aligned: bool) -> Observation:
        import cv2

        from components.shapes import find_pick_objects

        frames, live, notes = await self._capture(n_frames, aligned)
        t = time.monotonic()
        out_dir = Path(os.environ.get("MOONDREAM_LIVE", "out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        live_path = out_dir / "moondream_live.png"
        cv2.imwrite(str(live_path), frames[0].color)
        shapes = find_pick_objects(frames[0].color, live_path, objects=self.detect_labels)
        cands = [{"label": s.color, "mask": s.mask, "box": tuple(int(v) for v in s.box), "score": s.score}
                 for s in shapes if s.mask is not None]
        return Observation(frames=frames, candidates=cands, live=live, t_mono=t, notes=notes)

    async def capture(self, n_frames: int, *, aligned: bool) -> Observation:
        frames, live, notes = await self._capture(n_frames, aligned)
        return Observation(frames=frames, candidates=[], live=live, t_mono=time.monotonic(), notes=notes)


class ReplayPerception(PerceptionPort):
    """Offline replay of a saved run directory (PLAN mode only)."""

    def __init__(self, run_dir: Path, live: Optional[calib.LiveIdentity] = None):
        self.run_dir = Path(run_dir)
        self.live = live

    async def observe(self, n_frames: int, *, aligned: bool) -> Observation:
        from components.pour_evidence import load_candidates
        from components.rgbd import load_frames

        frames = load_frames(self.run_dir, "obs")
        return Observation(frames=frames, candidates=load_candidates(self.run_dir), live=self.live,
                           t_mono=time.monotonic(), notes=[f"replay of {self.run_dir}"])

    async def capture(self, n_frames: int, *, aligned: bool) -> Observation:
        raise PourAbort("replay_has_no_reobservation")


# ---------------------------------------------------------------------------
# Shared entry point for voice / orchestrator / CLI
# ---------------------------------------------------------------------------

# Tests replace this to inject fake ports; the default builds live Viam ports.
PORTS_FACTORY: Optional[Callable] = None
# The most recent PourResult (the orchestrator reports it after execute_call).
LAST_RESULT: Optional[PourResult] = None


async def _live_ports(machine, arm=None, gripper=None):
    from components.arm import ArmComponent
    from components.gripper import GripperComponent

    arm = arm or ArmComponent(machine)
    gripper = gripper or GripperComponent(machine)
    perception = LivePerception(machine, camera_name=os.environ.get("CAMERA_NAME", "cam"),
                                arm_name=arm.name)
    return LiveRobot(arm, gripper), perception


async def run_pour_request(
    source: str = "any",
    target: str = "cup",
    *,
    mode: PourMode | str = PourMode.POUR,
    machine=None,
    arm=None,
    gripper=None,
    params: Optional[PourParams] = None,
    confirmations: Optional[dict] = None,
    source_px=None,
    cup_px=None,
    nominal: Optional[bool] = None,
) -> PourResult:
    """The one pour primitive every caller uses. Checks readiness offline
    before touching the robot, then runs the PourController. ``nominal``
    (default: env POUR_NOMINAL=1) skips calibration at the operator's request."""
    global LAST_RESULT
    mode = PourMode(mode)
    nominal = _flag("POUR_NOMINAL") if nominal is None else nominal
    if nominal:
        problems = [] if pour_enabled() else [f"{ENABLE_ENV}=1 is not set"]
        if mode in LIQUID_MODES and not _flag(LIQUID_ENV):
            problems.append(f"{LIQUID_ENV}=1 is not set (real liquid disabled)")
    else:
        _, problems = readiness_problems(mode, confirmations=confirmations)
    if problems:
        LAST_RESULT = PourResult(state=S.ABORTED.value, success=False, reason="not_ready",
                                 detail={"problems": problems}, mode=mode.value)
        return LAST_RESULT
    own_machine = False
    if PORTS_FACTORY is not None:
        robot, perception = await PORTS_FACTORY(machine=machine, arm=arm, gripper=gripper)
    else:
        if machine is None:
            from components.connection import connect_machine

            machine = await connect_machine()
            own_machine = True
        robot, perception = await _live_ports(machine, arm, gripper)
    try:
        from components.constants import HOME_JOINTS

        ctl = PourController(robot, perception, params=params, observation_joints=HOME_JOINTS, nominal=nominal)
        LAST_RESULT = await ctl.run(PourRequest(source=source, target=target, mode=mode,
                                                confirmations=confirmations or {}, source_px=source_px,
                                                cup_px=cup_px))
        return LAST_RESULT
    finally:
        if own_machine:
            await machine.close()


def describe_result(res: PourResult) -> str:
    """Short spoken/printed outcome. Success only after DONE."""
    if res.success and res.state == S.DONE.value:
        if res.mode in (PourMode.POUR.value, PourMode.POUR_LIQUID.value, PourMode.POUR_DRY.value):
            return "Pour finished. Pouring is open-loop, so please check the cup."
        return f"{res.mode} finished."
    if res.reason == "not_ready":
        return "Pouring is unavailable: the calibrated pour is not enabled or not validated yet."
    if res.help_required:
        return f"I stopped because of {res.reason}. Please check the arm before continuing."
    return f"I did not pour: {res.reason}."
