"""Offline unit tests for components/experience_store.py.

No Viam connection, no robot, no camera -- pure JSON I/O against a tmp
path (pytest's `tmp_path` fixture), so nothing here touches the real
gitignored data/experience.json.
"""

from dataclasses import dataclass

from components.experience_store import (
    DEFAULT_RETRY_BUDGET,
    MAX_RETRY_BUDGET,
    MIN_RETRY_BUDGET,
    ExperienceStore,
    resolve_key,
)


@dataclass
class FakeShape:
    """Minimal LocatedShape-like stand-in (mirrors tests/test_declutter.py's
    FakeObj pattern) so these tests don't need to construct a real
    LocatedShape (and don't pull in cv2/numpy via components.shapes)."""

    x: float
    y: float
    color: str = ""
    label: str = ""
    canonical_label: str = ""
    history: object = None


# ---------------------------------------------------------------------------
# write / read / append round-trip
# ---------------------------------------------------------------------------


def test_fresh_store_has_no_history_and_default_calibration(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    assert store.get_history("red cube") is None
    assert store.get_calibration("red cube") == {
        "pick_z_offset": 0.0,
        "xy_offset": [0.0, 0.0],
        "grip_params": {},
        "retry_budget": DEFAULT_RETRY_BUDGET,
    }


def test_record_attempt_persists_to_disk_and_round_trips(tmp_path):
    path = tmp_path / "experience.json"
    store = ExperienceStore(path)
    store.record_attempt(
        "red cube",
        xy=(10.0, 20.0),
        grasp_success=True,
        placement_success=True,
        plan_params={"pick_z_offset": 0.0, "xy_offset": [0.0, 0.0]},
    )
    assert path.exists()

    # A brand-new store instance pointed at the same path sees the write.
    reloaded = ExperienceStore(path)
    history = reloaded.get_history("red cube")
    assert history is not None
    assert len(history["attempts"]) == 1
    attempt = history["attempts"][0]
    assert attempt["xy"] == [10.0, 20.0]
    assert attempt["grasp_success"] is True
    assert attempt["placement_success"] is True
    assert "timestamp" in attempt and attempt["timestamp"]


def test_append_adds_to_existing_history_without_clobbering(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    store.record_attempt("yellow cube", xy=(0.0, 0.0), grasp_success=False)
    store.record_attempt("yellow cube", xy=(1.0, 1.0), grasp_success=True)

    history = store.get_history("yellow cube")
    assert len(history["attempts"]) == 2
    assert [a["grasp_success"] for a in history["attempts"]] == [False, True]

    # A different label gets its own independent history.
    store.record_attempt("red cube", xy=(5.0, 5.0), grasp_success=True)
    assert len(store.get_history("yellow cube")["attempts"]) == 2
    assert len(store.get_history("red cube")["attempts"]) == 1


# ---------------------------------------------------------------------------
# calibration rollup
# ---------------------------------------------------------------------------


def test_calibration_rollup_biases_toward_successful_offsets(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    # Two failed attempts with a bad offset, then several successes with a
    # consistent good offset -- the success-weighted rollup should land
    # close to the good offset, not the bad one.
    store.record_attempt(
        "red cube",
        xy=(0, 0),
        grasp_success=False,
        plan_params={"pick_z_offset": -10.0, "xy_offset": [-10.0, -10.0]},
    )
    store.record_attempt(
        "red cube",
        xy=(0, 0),
        grasp_success=False,
        plan_params={"pick_z_offset": -10.0, "xy_offset": [-10.0, -10.0]},
    )
    for _ in range(3):
        calib = store.record_attempt(
            "red cube",
            xy=(0, 0),
            grasp_success=True,
            placement_success=True,
            plan_params={"pick_z_offset": 2.0, "xy_offset": [1.0, -1.0]},
        )

    # The success-weighted mean lands closer to the good (successful)
    # offset than to the bad (failed) one, even though it doesn't fully
    # discount the failures (FAILURE_WEIGHT > 0).
    assert calib["pick_z_offset"] > 0.0
    assert abs(calib["pick_z_offset"] - 2.0) < abs(calib["pick_z_offset"] - (-10.0))
    assert abs(calib["xy_offset"][0] - 1.0) < abs(calib["xy_offset"][0] - (-10.0))
    assert abs(calib["xy_offset"][1] - (-1.0)) < abs(calib["xy_offset"][1] - (-10.0))


def test_calibration_rollup_grip_params_success_weighted_mean(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    store.record_attempt(
        "red cube",
        xy=(0, 0),
        grasp_success=True,
        placement_success=True,
        plan_params={"grip_params": {"force": 40.0}},
    )
    calib = store.record_attempt(
        "red cube",
        xy=(0, 0),
        grasp_success=True,
        placement_success=True,
        plan_params={"grip_params": {"force": 60.0}},
    )
    assert calib["grip_params"]["force"] == 50.0


def test_retry_budget_rises_with_recent_failure_rate(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    for _ in range(5):
        calib = store.record_attempt("triangle", xy=(0, 0), grasp_success=True, placement_success=True)
    assert calib["retry_budget"] == MIN_RETRY_BUDGET

    store2 = ExperienceStore(tmp_path / "experience2.json")
    for _ in range(5):
        calib2 = store2.record_attempt("triangle", xy=(0, 0), grasp_success=False)
    assert calib2["retry_budget"] == MAX_RETRY_BUDGET


def test_get_calibration_matches_history_calibrated_plan(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    calib = store.record_attempt("red cube", xy=(0, 0), grasp_success=True, placement_success=True)
    assert store.get_calibration("red cube") == calib


# ---------------------------------------------------------------------------
# robustness: missing / corrupt file
# ---------------------------------------------------------------------------


def test_missing_file_creates_fresh_store(tmp_path):
    path = tmp_path / "does_not_exist" / "experience.json"
    store = ExperienceStore(path)
    assert store.get_history("anything") is None
    # Writing should create the parent dir and the file.
    store.record_attempt("red cube", xy=(0, 0), grasp_success=True)
    assert path.exists()


def test_corrupt_json_file_recovers_to_fresh_store(tmp_path):
    path = tmp_path / "experience.json"
    path.write_text("{not valid json::::")
    store = ExperienceStore(path)
    assert store.get_history("red cube") is None
    # Store still works after recovering -- write overwrites the garbage.
    store.record_attempt("red cube", xy=(0, 0), grasp_success=True, placement_success=True)
    reloaded = ExperienceStore(path)
    assert reloaded.get_history("red cube") is not None


def test_malformed_top_level_shape_recovers_to_fresh_store(tmp_path):
    path = tmp_path / "experience.json"
    path.write_text("[1, 2, 3]")  # valid JSON, wrong shape (not a dict)
    store = ExperienceStore(path)
    assert store.get_history("red cube") is None


def test_entries_missing_required_keys_are_dropped_not_fatal(tmp_path):
    path = tmp_path / "experience.json"
    path.write_text('{"red cube": {"unexpected": true}, "yellow cube": {"attempts": [], "calibrated_plan": {}}}')
    store = ExperienceStore(path)
    assert store.get_history("red cube") is None
    assert store.get_history("yellow cube") is not None


# ---------------------------------------------------------------------------
# key resolution + seed()
# ---------------------------------------------------------------------------


def test_resolve_key_prefers_canonical_label():
    shape = FakeShape(x=0, y=0, color="red", label="cube", canonical_label="red cube")
    assert resolve_key(shape) == "red cube"


def test_resolve_key_falls_back_to_color_and_label_when_canonical_missing():
    shape = FakeShape(x=0, y=0, color="red", label="cube", canonical_label="")
    assert resolve_key(shape) == "red:cube"


def test_seed_attaches_history_for_known_object(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    store.record_attempt("red cube", xy=(1, 2), grasp_success=True, placement_success=True)

    shape = FakeShape(x=1, y=2, color="red", label="cube", canonical_label="red cube")
    seeded = store.seed(shape)
    assert seeded is shape
    assert shape.history is not None
    assert len(shape.history["attempts"]) == 1


def test_seed_leaves_history_none_for_unknown_object(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    shape = FakeShape(x=1, y=2, color="green", label="triangle", canonical_label="green triangle")
    store.seed(shape)
    assert shape.history is None


def test_seed_uses_fallback_key_when_canonical_label_missing(tmp_path):
    store = ExperienceStore(tmp_path / "experience.json")
    store.record_attempt("blue:cuboid", xy=(0, 0), grasp_success=True, placement_success=True)

    shape = FakeShape(x=0, y=0, color="blue", label="cuboid", canonical_label="")
    store.seed(shape)
    assert shape.history is not None
