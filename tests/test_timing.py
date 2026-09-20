"""Offline unit tests for scripts/measure_timing.py.

No Viam connection, no robot, no camera, no model weights -- exercises
``time_callable`` against known stub callables and checks the JSON
read/write round-trip and schema.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_timing.py"

# scripts/ has no __init__.py, so import the module directly from its path
# rather than relying on package discovery.
_spec = importlib.util.spec_from_file_location("measure_timing", SCRIPT_PATH)
measure_timing = importlib.util.module_from_spec(_spec)
sys.modules["measure_timing"] = measure_timing
_spec.loader.exec_module(measure_timing)


def test_time_callable_fixed_sleep_yields_mean_near_sleep():
    sleep_s = 0.02
    timing = measure_timing.time_callable(
        lambda: time.sleep(sleep_s), n=8, stage="fixed_sleep"
    )

    assert timing.ok is True
    assert timing.n == 8
    assert timing.stage == "fixed_sleep"
    # Generous tolerance: scheduler jitter, but mean should track the sleep,
    # never something wildly different (e.g. off by 10x).
    expected_ms = sleep_s * 1000.0
    assert expected_ms * 0.5 <= timing.mean_ms <= expected_ms * 4.0
    assert timing.min_ms <= timing.mean_ms <= timing.max_ms
    assert timing.p95_ms >= timing.min_ms
    assert timing.stdev_ms >= 0.0
    assert timing.total_ms > 0.0


def test_time_callable_zero_sleep_is_fast_and_valid():
    timing = measure_timing.time_callable(lambda: None, n=20, stage="noop")
    assert timing.ok is True
    assert timing.n == 20
    assert timing.mean_ms >= 0.0
    assert timing.mean_ms < 50.0  # should be sub-millisecond in practice


def test_time_callable_records_requested_n():
    for n in (1, 5, 13):
        timing = measure_timing.time_callable(lambda: None, n=n, stage=f"n{n}")
        assert timing.n == n


def test_time_callable_all_failures_reports_not_ok_without_raising():
    def _boom():
        raise RuntimeError("simulated failure")

    timing = measure_timing.time_callable(_boom, n=5, stage="always_fails")
    assert timing.ok is False
    assert timing.n == 0
    assert timing.errors == 5
    assert timing.notes  # explains why


def test_time_callable_partial_failures_counted_and_excluded():
    calls = {"i": 0}

    def _sometimes_fails():
        calls["i"] += 1
        if calls["i"] % 2 == 0:
            raise ValueError("every other call fails")

    timing = measure_timing.time_callable(_sometimes_fails, n=6, stage="flaky")
    assert timing.ok is True
    assert timing.errors == 3
    assert timing.n == 3  # only the successful calls are timed


def test_time_callable_warmup_calls_are_not_counted():
    calls = {"i": 0}

    def _counter():
        calls["i"] += 1

    timing = measure_timing.time_callable(_counter, n=4, stage="warmup", warmup=3)
    assert calls["i"] == 7  # 3 warmup + 4 timed
    assert timing.n == 4


def test_percentile_helper_matches_known_values():
    ordered = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert measure_timing._percentile(ordered, 0) == 1.0
    assert measure_timing._percentile(ordered, 100) == 5.0
    assert measure_timing._percentile([], 95) == 0.0
    assert measure_timing._percentile([42.0], 95) == 42.0


def test_self_test_produces_expected_mock_stages():
    results = measure_timing.run_self_test(n=3, uq_passes=2)
    stage_names = {r.stage for r in results}
    assert "owlv2_inference_mock" in stage_names
    assert "moondream_inference_mock" in stage_names
    assert "owlv2_inference_mock_x2" in stage_names
    assert "camera_get_image_mock" in stage_names
    assert "arm_command_mock" in stage_names
    for r in results:
        assert r.ok is True
        assert r.n > 0


def test_build_report_self_test_only_has_no_real_stages():
    report = measure_timing.build_report(
        self_test=True,
        real_models=False,
        real_camera=False,
        real_arm=False,
        n=3,
        uq_passes=2,
        frame_path=measure_timing.DEFAULT_FRAME,
    )
    assert report.mode == ["self-test"]
    assert "owlv2_inference_mock" in report.stages
    assert "owlv2_inference" not in report.stages  # real-model stage name, absent
    assert report.schema_version == measure_timing.SCHEMA_VERSION
    assert report.timestamp


def test_write_and_read_report_round_trip(tmp_path):
    report = measure_timing.build_report(
        self_test=True,
        real_models=False,
        real_camera=False,
        real_arm=False,
        n=4,
        uq_passes=2,
        frame_path=measure_timing.DEFAULT_FRAME,
    )
    out_path = tmp_path / "timing.json"
    written = measure_timing.write_report(report, output=out_path)
    assert written == out_path
    assert out_path.exists()

    loaded = measure_timing.read_report(out_path)
    assert loaded["schema_version"] == measure_timing.SCHEMA_VERSION
    assert "timestamp" in loaded
    assert "mode" in loaded
    assert loaded["mode"] == ["self-test"]
    assert "stages" in loaded

    owl = loaded["stages"]["owlv2_inference_mock"]
    for key in (
        "stage",
        "n",
        "mean_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "stdev_ms",
        "total_ms",
        "ok",
        "errors",
        "notes",
    ):
        assert key in owl
    assert owl["n"] == 4


def test_write_report_default_output_is_gitignored_data_dir():
    # data/ is listed in .gitignore -- confirm the default output path lives
    # there so a normal run never risks a git-tracked artifact.
    assert measure_timing.DEFAULT_OUTPUT.parent.name == "data"
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "data/" in gitignore


def test_real_models_skips_gracefully_without_frame(tmp_path):
    missing_frame = tmp_path / "does_not_exist.png"
    results = measure_timing.run_real_models(frame_path=missing_frame, n=2, uq_passes=2)
    stage_names = {r.stage for r in results}
    assert "owlv2_inference" in stage_names
    assert "moondream_inference" in stage_names
    for r in results:
        assert r.ok is False
        assert r.n == 0
        assert "skipped" in r.notes.lower()


def test_real_camera_skips_gracefully_without_machine():
    # No .env / live machine in this offline test environment -- must
    # degrade to a skipped stage, never raise.
    results = measure_timing.run_real_camera(n=2)
    assert len(results) == 1
    assert results[0].stage == "camera_get_image"
    assert results[0].ok is False
    assert "skipped" in results[0].notes.lower()


def test_real_arm_skips_gracefully_without_machine():
    results = measure_timing.run_real_arm(n=2)
    assert len(results) == 1
    assert results[0].stage == "arm_command_latency"
    assert results[0].ok is False
    assert "skipped" in results[0].notes.lower()


def test_cli_self_test_writes_valid_json(tmp_path, capsys):
    out_path = tmp_path / "timing.json"
    rc = measure_timing.main(
        ["--self-test", "--n", "3", "--uq-passes", "2", "--output", str(out_path)]
    )
    assert rc == 0
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert data["mode"] == ["self-test"]
    assert data["schema_version"] == measure_timing.SCHEMA_VERSION
    assert all(v["n"] == 3 or "_x2" in k for k, v in data["stages"].items())

    captured = capsys.readouterr()
    assert "wrote" in captured.out


def test_cli_defaults_to_self_test_when_no_real_flag(tmp_path):
    out_path = tmp_path / "timing.json"
    rc = measure_timing.main(["--n", "2", "--output", str(out_path)])
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["mode"] == ["self-test"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
