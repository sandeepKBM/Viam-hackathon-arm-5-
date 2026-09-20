"""Sensor & pipeline timing budget harness (W6).

MEASURES (does not assume) the timing the rest of the perception/pick
pipeline needs to budget against:

  - per-model inference time (OWLv2 via components.zeroshot, Moondream via
    components.vlm) on a saved frame, and xN (to price the UQ
    augmentation-consistency passes W3 wants to run),
  - camera color+depth rate + latency as delivered through Viam
    (``Camera.get_image`` round-trip),
  - arm command latency (round-trip of a no-op / small move) -- optional,
    guarded, off by default.

Design: every stage is timed through ``time_callable(fn, n)``, which takes an
arbitrary zero-arg callable and returns mean/p95/min/max/stdev over N calls.
The real detectors / ``camera.get_image`` / arm move are just callables that
get injected -- this module never hard-imports torch/transformers/viam at
module scope, so it stays runnable with only the stdlib.

Anchors (comments only -- NOT written to timing.json, which holds MEASURED
numbers):
  - RealSense D435: ~30 fps RGB, up to ~90 fps depth (nominal, USB3).
  - Orbbec Astra 2: ~30 fps RGB+depth (nominal).

Usage:
    # Offline self-test (default): times mock/stub callables, no deps,
    # no hardware, no model download. Verifies the harness end-to-end.
    .venv/bin/python scripts/measure_timing.py --self-test
    .venv/bin/python scripts/measure_timing.py            # same as above

    # Real model timing on the rig (needs requirements-zeroshot.txt /
    # requirements-vlm.txt installed and a saved frame, e.g. out/frame.png
    # from capture_image.py). Guarded: only attempted if --real-models is
    # passed AND the deps + frame are present; otherwise skipped with a note.
    .venv/bin/python scripts/measure_timing.py --real-models --frame out/frame.png

    # Real camera timing on the rig (needs a live Viam machine -- .env
    # populated per .env.example, see components/connection.py). Guarded
    # the same way.
    .venv/bin/python scripts/measure_timing.py --real-camera

    # Real arm command latency on the rig (moves the arm a tiny amount --
    # off unless explicitly requested, since it touches hardware).
    .venv/bin/python scripts/measure_timing.py --real-arm

Any combination of --self-test / --real-models / --real-camera / --real-arm
may be passed together; results merge into one data/timing.json. With none
given, --self-test is the default so the harness is always verifiable
offline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "timing.json"
DEFAULT_FRAME = REPO_ROOT / "out" / "frame.png"

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Core measurement primitive
# ---------------------------------------------------------------------------


@dataclass
class StageTiming:
    """Timing stats for one measured stage.

    All durations are in milliseconds. ``n`` is how many calls were actually
    timed (may be < requested N if a call raised and was skipped -- see
    ``errors``).
    """

    stage: str
    n: int
    mean_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    total_ms: float
    ok: bool = True
    errors: int = 0
    notes: str = ""


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Nearest-rank percentile, no numpy dependency."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def time_callable(
    fn: Callable[[], Any],
    n: int = 10,
    stage: str = "unnamed",
    warmup: int = 0,
    notes: str = "",
) -> StageTiming:
    """Call ``fn()`` ``n`` times, timing each call in milliseconds.

    ``warmup`` calls are made first and discarded (e.g. to absorb one-time
    model/JIT warmup so the measured stats reflect steady-state latency).

    A call that raises is counted in ``errors`` and excluded from the stats;
    if every call raises, returns a zeroed, ``ok=False`` result rather than
    raising, so a harness run degrades gracefully instead of crashing.
    """
    for _ in range(max(warmup, 0)):
        try:
            fn()
        except Exception:
            pass

    durations_ms: List[float] = []
    errors = 0
    for _ in range(max(n, 0)):
        start = time.perf_counter()
        try:
            fn()
        except Exception:
            errors += 1
            continue
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
        durations_ms.append(elapsed)

    if not durations_ms:
        return StageTiming(
            stage=stage,
            n=0,
            mean_ms=0.0,
            p95_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            stdev_ms=0.0,
            total_ms=0.0,
            ok=False,
            errors=errors,
            notes=notes or "all calls failed",
        )

    ordered = sorted(durations_ms)
    return StageTiming(
        stage=stage,
        n=len(durations_ms),
        mean_ms=statistics.fmean(durations_ms),
        p95_ms=_percentile(ordered, 95),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        stdev_ms=statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0,
        total_ms=sum(durations_ms),
        ok=True,
        errors=errors,
        notes=notes,
    )


async def time_async_callable(
    fn: Callable[[], Any],
    n: int = 10,
    stage: str = "unnamed",
    warmup: int = 0,
    notes: str = "",
) -> StageTiming:
    """Async analogue of ``time_callable`` for coroutine-returning callables.

    ``fn()`` must return an awaitable (e.g. ``lambda: camera.get_image()``).
    Used for the real camera/arm paths, which are Viam async APIs.
    """
    for _ in range(max(warmup, 0)):
        try:
            await fn()
        except Exception:
            pass

    durations_ms: List[float] = []
    errors = 0
    for _ in range(max(n, 0)):
        start = time.perf_counter()
        try:
            await fn()
        except Exception:
            errors += 1
            continue
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
        durations_ms.append(elapsed)

    if not durations_ms:
        return StageTiming(
            stage=stage,
            n=0,
            mean_ms=0.0,
            p95_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            stdev_ms=0.0,
            total_ms=0.0,
            ok=False,
            errors=errors,
            notes=notes or "all calls failed",
        )

    ordered = sorted(durations_ms)
    return StageTiming(
        stage=stage,
        n=len(durations_ms),
        mean_ms=statistics.fmean(durations_ms),
        p95_ms=_percentile(ordered, 95),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        stdev_ms=statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0,
        total_ms=sum(durations_ms),
        ok=True,
        errors=errors,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Mock callables (self-test path -- no deps, no hardware)
# ---------------------------------------------------------------------------


def _mock_owlv2_call(sleep_s: float = 0.03) -> Callable[[], None]:
    """Stand-in for one OWLv2 ``pipe(...)`` call (components/zeroshot.py)."""

    def _call() -> None:
        time.sleep(sleep_s)

    return _call


def _mock_moondream_call(sleep_s: float = 0.08) -> Callable[[], None]:
    """Stand-in for one Moondream ``model.detect(...)`` call (components/vlm.py)."""

    def _call() -> None:
        time.sleep(sleep_s)

    return _call


def _mock_camera_get_image(sleep_s: float = 0.01) -> Callable[[], None]:
    """Stand-in for one ``camera.get_image()`` round-trip."""

    def _call() -> None:
        time.sleep(sleep_s)

    return _call


def _mock_arm_command(sleep_s: float = 0.05) -> Callable[[], None]:
    """Stand-in for one arm no-op / small-move command round-trip."""

    def _call() -> None:
        time.sleep(sleep_s)

    return _call


# ---------------------------------------------------------------------------
# Result assembly / JSON I/O
# ---------------------------------------------------------------------------


@dataclass
class TimingReport:
    schema_version: int
    timestamp: str
    mode: List[str]  # which sources contributed: "self-test", "real-models", ...
    stages: Dict[str, dict] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def add(self, timing: StageTiming) -> None:
        self.stages[timing.stage] = asdict(timing)

    def to_dict(self) -> dict:
        return asdict(self)


def write_report(report: TimingReport, output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return output


def read_report(path: Path = DEFAULT_OUTPUT) -> dict:
    with Path(path).open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Self-test (mock) measurement pass -- always runnable
# ---------------------------------------------------------------------------


def run_self_test(n: int = 10, uq_passes: int = 5) -> List[StageTiming]:
    """Time mock/stub callables standing in for each real stage.

    Verifies the harness (time_callable stats, JSON schema, N recorded)
    without any heavy deps, model weights, or hardware.
    """
    results: List[StageTiming] = []

    results.append(
        time_callable(
            _mock_owlv2_call(),
            n=n,
            stage="owlv2_inference_mock",
            notes="mock stand-in (sleep-based stub); see --real-models for measured",
        )
    )
    results.append(
        time_callable(
            _mock_moondream_call(),
            n=n,
            stage="moondream_inference_mock",
            notes="mock stand-in (sleep-based stub); see --real-models for measured",
        )
    )
    # xN pass pricing: UQ augmentation-consistency (W3) reruns detection N
    # times under jitter; price that as N sequential calls of the same stub.
    owl_call = _mock_owlv2_call()

    def _owlv2_x_n() -> None:
        for _ in range(uq_passes):
            owl_call()

    results.append(
        time_callable(
            _owlv2_x_n,
            n=max(n // 2, 3),
            stage=f"owlv2_inference_mock_x{uq_passes}",
            notes=f"prices UQ augmentation-consistency ({uq_passes} passes) using mock stub",
        )
    )
    results.append(
        time_callable(
            _mock_camera_get_image(),
            n=n,
            stage="camera_get_image_mock",
            notes="mock stand-in for Camera.get_image() round-trip; see --real-camera for measured",
        )
    )
    results.append(
        time_callable(
            _mock_arm_command(),
            n=n,
            stage="arm_command_mock",
            notes="mock stand-in for arm no-op/small-move round-trip; see --real-arm for measured",
        )
    )
    return results


# ---------------------------------------------------------------------------
# Real-model timing (guarded: needs torch/transformers + a saved frame)
# ---------------------------------------------------------------------------


def _load_frame(frame_path: Path):
    """Load a saved BGR frame with cv2. Returns None (and prints why) on failure."""
    try:
        import cv2  # noqa: WPS433
    except Exception as exc:
        print(f"[timing] real-models: cv2 not available ({exc}); skipping")
        return None
    if not frame_path.exists():
        print(f"[timing] real-models: no saved frame at {frame_path}; skipping")
        return None
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"[timing] real-models: failed to read {frame_path}; skipping")
        return None
    return img


def run_real_models(
    frame_path: Path = DEFAULT_FRAME, n: int = 5, uq_passes: int = 5
) -> List[StageTiming]:
    """Time real OWLv2 / Moondream inference on a saved frame.

    Guarded: only runs if the frame exists AND the heavy deps import
    successfully (torch/transformers via components.zeroshot / vlm). On any
    missing piece, returns a single 'skipped' StageTiming per stage with a
    note explaining why, rather than raising -- so this never breaks a run
    on a machine without the model deps or a captured frame.

    On the rig: run inside the venv with requirements-zeroshot.txt /
    requirements-vlm.txt installed, after capturing a frame with
    capture_image.py (saves out/frame.png by default), e.g.:

        .venv/bin/python capture_image.py
        .venv/bin/python scripts/measure_timing.py --real-models --frame out/frame.png
    """
    results: List[StageTiming] = []
    bgr = _load_frame(frame_path)
    if bgr is None:
        results.append(
            StageTiming(
                stage="owlv2_inference",
                n=0,
                mean_ms=0.0,
                p95_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                stdev_ms=0.0,
                total_ms=0.0,
                ok=False,
                errors=0,
                notes=f"skipped: no usable frame at {frame_path} (run capture_image.py on the rig)",
            )
        )
        results.append(
            StageTiming(
                stage="moondream_inference",
                n=0,
                mean_ms=0.0,
                p95_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                stdev_ms=0.0,
                total_ms=0.0,
                ok=False,
                errors=0,
                notes=f"skipped: no usable frame at {frame_path} (run capture_image.py on the rig)",
            )
        )
        return results

    # OWLv2 (components/zeroshot.py)
    try:
        from components.zeroshot import get_detector

        detector = get_detector()
        detector.detect(bgr)  # warm the lazy pipeline once before timing
        results.append(
            time_callable(
                lambda: detector.detect(bgr),
                n=n,
                stage="owlv2_inference",
                notes=f"measured on {frame_path.name}, model={detector.model}",
            )
        )

        def _owlv2_x_n() -> None:
            for _ in range(uq_passes):
                detector.detect(bgr)

        results.append(
            time_callable(
                _owlv2_x_n,
                n=max(n // 2, 1),
                stage=f"owlv2_inference_x{uq_passes}",
                notes=f"UQ augmentation-consistency pricing ({uq_passes} passes), measured",
            )
        )
    except Exception as exc:
        results.append(
            StageTiming(
                stage="owlv2_inference",
                n=0,
                mean_ms=0.0,
                p95_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                stdev_ms=0.0,
                total_ms=0.0,
                ok=False,
                errors=1,
                notes=f"skipped: {exc!r} (install requirements-zeroshot.txt on the rig)",
            )
        )

    # Moondream (components/vlm.py)
    try:
        from components.vlm import get_vlm

        vlm = get_vlm()
        vlm.detect(bgr, "block")  # warm
        results.append(
            time_callable(
                lambda: vlm.detect(bgr, "block"),
                n=n,
                stage="moondream_inference",
                notes=f"measured on {frame_path.name}, model={vlm.model_id}",
            )
        )
    except Exception as exc:
        results.append(
            StageTiming(
                stage="moondream_inference",
                n=0,
                mean_ms=0.0,
                p95_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                stdev_ms=0.0,
                total_ms=0.0,
                ok=False,
                errors=1,
                notes=f"skipped: {exc!r} (install requirements-vlm.txt on the rig)",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Real camera timing (guarded: needs a live Viam machine, see .env.example)
# ---------------------------------------------------------------------------


async def _run_real_camera_async(n: int) -> List[StageTiming]:
    from components.connection import connect_machine
    from components.shapes import CAMERA_NAME
    from viam.components.camera import Camera

    machine = await connect_machine()
    try:
        cam = Camera.from_robot(machine, CAMERA_NAME)
        # viam-sdk 0.80.0 Camera has no get_image(); use get_images() (returns
        # (images, metadata)). We only need the round-trip, so ignore the payload.
        await cam.get_images()  # warm
        timing = await time_async_callable(
            lambda: cam.get_images(),
            n=n,
            stage="camera_get_image",
            notes=f"measured Camera.get_images() round-trip via Viam, camera={CAMERA_NAME}",
        )
        return [timing]
    finally:
        await machine.close()


def run_real_camera(n: int = 20) -> List[StageTiming]:
    """Time ``camera.get_image()`` round-trip latency through Viam.

    Guarded: requires a live Viam machine reachable via the .env credentials
    (see .env.example / components/connection.py). Rate (fps) can be derived
    from 1000 / mean_ms; this does not separately probe max sustained rate,
    only per-call round-trip latency as delivered through Viam.

    On the rig: with .env populated and the machine online,

        .venv/bin/python scripts/measure_timing.py --real-camera
    """
    try:
        import asyncio

        return asyncio.run(_run_real_camera_async(n))
    except Exception as exc:
        return [
            StageTiming(
                stage="camera_get_image",
                n=0,
                mean_ms=0.0,
                p95_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                stdev_ms=0.0,
                total_ms=0.0,
                ok=False,
                errors=1,
                notes=f"skipped: {exc!r} (needs .env + reachable machine, see .env.example)",
            )
        ]


# ---------------------------------------------------------------------------
# Real arm command latency (guarded: touches hardware, off unless requested)
# ---------------------------------------------------------------------------


async def _run_real_arm_async(n: int) -> List[StageTiming]:
    from components.connection import connect_machine
    from viam.components.arm import Arm

    arm_name = os.environ.get("ARM_NAME", "arm")
    machine = await connect_machine()
    try:
        arm = Arm.from_robot(machine, arm_name)
        # No-op round-trip: read-only status call, never commands a move.
        # Kept intentionally non-actuating so this is safe to run without a
        # human watching the cell; a real "small move" latency probe should
        # be done deliberately on the rig with someone present.
        await arm.is_moving()  # warm
        timing = await time_async_callable(
            lambda: arm.is_moving(),
            n=n,
            stage="arm_command_latency",
            notes=(
                f"measured no-op status round-trip (arm.is_moving()) via Viam, arm={arm_name}; "
                "NOT an actuating move -- safe default. For actual small-move latency, "
                "extend with a deliberate bounded move on the rig with a human present."
            ),
        )
        return [timing]
    finally:
        await machine.close()


def run_real_arm(n: int = 20) -> List[StageTiming]:
    """Time arm command round-trip latency through Viam.

    Guarded and off by default: only runs with --real-arm, and even then
    only issues a read-only status round-trip (``arm.is_moving()``), never
    an actuating move, so it's safe to run unattended. Requires a live Viam
    machine (.env credentials, see .env.example).

    On the rig:

        .venv/bin/python scripts/measure_timing.py --real-arm
    """
    try:
        import asyncio

        return asyncio.run(_run_real_arm_async(n))
    except Exception as exc:
        return [
            StageTiming(
                stage="arm_command_latency",
                n=0,
                mean_ms=0.0,
                p95_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                stdev_ms=0.0,
                total_ms=0.0,
                ok=False,
                errors=1,
                notes=f"skipped: {exc!r} (needs .env + reachable machine, see .env.example)",
            )
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_report(
    self_test: bool,
    real_models: bool,
    real_camera: bool,
    real_arm: bool,
    n: int,
    uq_passes: int,
    frame_path: Path,
) -> TimingReport:
    modes: List[str] = []
    report = TimingReport(
        schema_version=SCHEMA_VERSION,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        mode=modes,
    )

    if self_test:
        modes.append("self-test")
        for t in run_self_test(n=n, uq_passes=uq_passes):
            report.add(t)

    if real_models:
        modes.append("real-models")
        for t in run_real_models(frame_path=frame_path, n=n, uq_passes=uq_passes):
            report.add(t)

    if real_camera:
        modes.append("real-camera")
        for t in run_real_camera(n=n):
            report.add(t)

    if real_arm:
        modes.append("real-arm")
        for t in run_real_arm(n=n):
            report.add(t)

    if not report.stages:
        report.notes.append("no stages measured -- check flags")

    for stage_name, stage in report.stages.items():
        if not stage.get("ok", False):
            report.notes.append(f"{stage_name}: {stage.get('notes', 'skipped')}")

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure sensor & pipeline timing budget (W6)."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Time mock/stub callables (default if no --real-* flag is given).",
    )
    parser.add_argument(
        "--real-models",
        action="store_true",
        help="Time real OWLv2/Moondream inference on a saved frame (needs deps + frame).",
    )
    parser.add_argument(
        "--real-camera",
        action="store_true",
        help="Time real Camera.get_image() round-trip via Viam (needs live machine).",
    )
    parser.add_argument(
        "--real-arm",
        action="store_true",
        help="Time real arm status round-trip via Viam (needs live machine; read-only, no move).",
    )
    parser.add_argument(
        "--frame",
        type=Path,
        default=DEFAULT_FRAME,
        help=f"Saved frame for --real-models (default: {DEFAULT_FRAME}).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of timed calls per stage (default: 10).",
    )
    parser.add_argument(
        "--uq-passes",
        type=int,
        default=5,
        help="N for the UQ augmentation-consistency xN pricing stage (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    any_real = args.real_models or args.real_camera or args.real_arm
    self_test = args.self_test or not any_real

    report = build_report(
        self_test=self_test,
        real_models=args.real_models,
        real_camera=args.real_camera,
        real_arm=args.real_arm,
        n=args.n,
        uq_passes=args.uq_passes,
        frame_path=args.frame,
    )

    out_path = write_report(report, output=args.output)

    print(f"[timing] mode: {', '.join(report.mode) or '(none)'}")
    for stage_name, stage in sorted(report.stages.items()):
        status = "ok" if stage.get("ok") else "SKIPPED"
        print(
            f"[timing] {stage_name:32s} {status:8s} "
            f"n={stage.get('n', 0):3d} mean={stage.get('mean_ms', 0):8.2f}ms "
            f"p95={stage.get('p95_ms', 0):8.2f}ms"
        )
    if report.notes:
        print("[timing] notes:")
        for note in report.notes:
            print(f"  - {note}")
    print(f"[timing] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
