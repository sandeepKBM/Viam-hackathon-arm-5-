"""Calibrated bottle/can -> cup pour: dry-run, replay, and staged hardware tests.

The motion logic lives in components/pouring.py (the same PourController voice
and the orchestrator use). This CLI only picks the mode and records the
operator's trial outcomes for the staged validation.

  python scripts/pour_can.py --readiness                  # calibration + stage status, offline
  python scripts/pour_can.py --replay out/pour_runs/<run> # offline replay -> overlays + plan.json
  python scripts/pour_can.py                              # dry-run: observe, localize, plan; NO motion
  python scripts/pour_can.py --stage hover                # D: hover >= 30 mm over mouth + rim
  python scripts/pour_can.py --stage grasp                # E: empty-container side grasp, put back
  python scripts/pour_can.py --stage pour-dry             # F: full trajectory, EMPTY container
  python scripts/pour_can.py --stage pour-liquid --tray --estop --mentor --bounded   # G: water

  --nominal  skip calibration (operator's call): camera mount from the live Viam
             frame system, table fitted from depth, nominal gripper. Expect
             cm-level error; stage gates are skipped, trials are not recorded.

Every live mode needs VIAM_ALLOW_LIVE=1 and ENABLE_CALIBRATED_POUR=1; motion
stages also need the earlier stages recorded as passed for this calibration.
Liquid needs POUR_ALLOW_LIQUID=1 on top. Pouring is duration-based and
open-loop (no fill sensing).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import boot  # noqa: F401
from components import calibration as calib
from components.pour_planner import PourParams
from components.pouring import (
    PourController,
    PourMode,
    PourRequest,
    ReplayPerception,
    describe_result,
    readiness_problems,
)

STAGE_MODES = {
    "hover": (PourMode.HOVER, "D"),
    "grasp": (PourMode.GRASP, "E"),
    "pour-dry": (PourMode.POUR_DRY, "F"),
    "pour-liquid": (PourMode.POUR_LIQUID, "G"),
}


def _px(v):
    if not v:
        return None
    u, w = (float(x) for x in v.split(","))
    return (u, w)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""


class CliOperator:
    def __init__(self):
        self.errors = {}

    async def measure_hover(self, name, *, predicted, fingertip):
        print(f"  hovering over {name}: predicted {predicted.round(1).tolist()} mm, fingertip {fingertip.round(1).tolist()}")
        raw = _ask(f"  measured horizontal offset fingertip-centre -> {name} (mm, blank = skip): ")
        try:
            v = float(raw)
        except ValueError:
            v = None
        self.errors[name] = v
        return v


def readiness() -> None:
    for mode in PourMode:
        _, problems = readiness_problems(mode)
        print(f"{mode.value:12s} {'READY' if not problems else 'blocked'}"
              + ("" if not problems else "\n    - " + "\n    - ".join(problems)))


async def replay(run_dir: str, args) -> None:
    ctl = PourController(None, ReplayPerception(run_dir), params=_params(args),
                         run_dir=None, save_evidence=True)
    from components.pour_evidence import new_run_dir

    ctl.run_dir = new_run_dir(tag="replay")
    res = await ctl.run(PourRequest(source=args.source, mode=PourMode.PLAN,
                                    source_px=_px(args.source_px), cup_px=_px(args.cup_px)))
    print(json.dumps(res.as_dict(), indent=2, default=str))
    print(f"replay evidence: {ctl.run_dir} (overlay.png, plan_top.png, plan_side.png, plan_provisional.json)")
    print("note: replay cannot check the live camera identity / Viam frame system; the live run does.")


def _params(args) -> PourParams:
    p = PourParams()
    if args.tilt is not None:
        p.max_tilt_deg = args.tilt
    if args.hold is not None:
        p.hold_s = args.hold
    return p.validate()


def _record(stage: str, res, extra: dict) -> None:
    try:
        setup = calib.check_setup()
    except calib.CalibrationError:
        return
    rec = {"stage": stage, "setup_hash": setup.setup_hash, "run_dir": res.run_dir, "state": res.state,
           "reason": res.reason, "t": __import__("time").time(), **extra}
    calib.append_trial(rec)
    print(f"recorded stage {stage} trial: success={rec.get('success')} contact={rec.get('contact')}")


async def live(args) -> None:
    from components.connection import connect_machine
    from components.constants import HOME_JOINTS
    from components.pouring import _live_ports

    if args.stage:
        mode, stage = STAGE_MODES[args.stage]
    else:
        mode, stage = PourMode.PLAN, None
    confirmations = {"tray": args.tray, "estop_attended": args.estop, "mentor_approved": args.mentor,
                     "bounded_amount": args.bounded}
    if args.nominal:
        from components.pouring import ENABLE_ENV, LIQUID_ENV, LIQUID_MODES, _flag, pour_enabled

        problems = [] if pour_enabled() else [f"{ENABLE_ENV}=1 is not set"]
        if mode in LIQUID_MODES and not _flag(LIQUID_ENV):
            problems.append(f"{LIQUID_ENV}=1 is not set")
        print("NOMINAL mode: calibration skipped, expect cm-level error. Empty container first.")
    else:
        _, problems = readiness_problems(mode, confirmations=confirmations)
    if problems:
        print(f"{mode.value} is blocked (no motion):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)
    if mode in (PourMode.POUR_DRY, PourMode.GRASP, PourMode.HOVER, PourMode.POUR_LIQUID):
        if _ask("Container EMPTY, human on the e-stop, area clear? [yes/no]: ") != "yes":
            sys.exit("aborted by operator")
    machine = await connect_machine()
    try:
        robot, perception = await _live_ports(machine)
        operator = CliOperator()
        ctl = PourController(robot, perception, params=_params(args), operator=operator,
                             observation_joints=HOME_JOINTS, nominal=args.nominal,
                             skip_reobserve=args.skip_reobserve)
        res = await ctl.run(PourRequest(source=args.source, mode=mode, source_px=_px(args.source_px),
                                        cup_px=_px(args.cup_px), confirmations=confirmations))
    finally:
        await machine.close()
    print(json.dumps({k: v for k, v in res.as_dict().items() if k != "transitions"}, indent=2, default=str))
    print(describe_result(res))
    print(f"evidence: {res.run_dir}")
    if res.reason and res.detail.get("fix"):
        print(f"FIX: {res.detail['fix']}")
    if stage and args.record and not args.nominal:
        extra = {"success": False, "contact": False}
        if stage == "D":
            errs = [v for v in operator.errors.values() if v is not None]
            extra["hover_errors_mm"] = operator.errors
            extra["success"] = bool(res.success and len(errs) == 2 and max(errs) <= calib.ACCEPTANCE["p95_xy_mm"])
        else:
            extra["success"] = res.success and _ask("Did the trial succeed as intended? [yes/no]: ") == "yes"
            extra["contact"] = _ask("Any unintended contact/collision? [yes/no]: ") == "yes"
            if stage == "G":
                extra.update({"spill": _ask("Spill outside the cup? [none/small/large]: "),
                              "tray": args.tray, "estop_attended": args.estop, "mentor_approved": args.mentor})
        extra["collision"] = extra.get("contact", False)
        _record(stage, res, extra)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readiness", action="store_true", help="print readiness for every mode (offline)")
    ap.add_argument("--replay", metavar="RUN_DIR", help="offline replay of a saved run (no robot)")
    ap.add_argument("--stage", choices=list(STAGE_MODES), help="staged hardware validation mode (moves the arm)")
    ap.add_argument("--source", default="bottle", choices=["bottle", "can", "any"])
    ap.add_argument("--source-px", help="u,v pixel inside the intended source (explicit selection)")
    ap.add_argument("--cup-px", help="u,v pixel inside the intended cup (explicit selection)")
    ap.add_argument("--tilt", type=float, help="max tilt deg (bounded)")
    ap.add_argument("--hold", type=float, help="hold seconds (bounded; open-loop)")
    ap.add_argument("--no-record", dest="record", action="store_false", help="do not append the trial outcome")
    ap.add_argument("--nominal", action="store_true", help="skip calibration: nominal, uncalibrated setup")
    ap.add_argument("--skip-reobserve", action="store_true", help="(nominal only) do not re-check the cup after lifting")
    ap.add_argument("--tray", action="store_true")
    ap.add_argument("--estop", action="store_true")
    ap.add_argument("--mentor", action="store_true")
    ap.add_argument("--bounded", action="store_true", help="liquid amount measured and bounded")
    ap.add_argument("--go", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--release", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.go or args.release:
        sys.exit("--go/--release were the uncalibrated experiment and are gone. Use --stage "
                 f"{'/'.join(STAGE_MODES)} after calibration (see README 'Calibrated pour').")
    if args.skip_reobserve and not args.nominal:
        sys.exit("--skip-reobserve is only allowed with --nominal")
    if args.readiness:
        readiness()
        return
    if args.replay:
        asyncio.run(replay(args.replay, args))
        return
    asyncio.run(live(args))


if __name__ == "__main__":
    main()
