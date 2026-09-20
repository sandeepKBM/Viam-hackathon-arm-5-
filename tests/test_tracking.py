"""Offline verification of components/tracking.py: the object-tracking
consumer (`get_tracked_objects`) and frame-to-frame identity matcher
(`match_to_known`) -- no robot, no camera, no live Viam. The tracker vision
service is a mock (`VisionClient` monkeypatched, same pattern as
tests/test_perception3d.py) run across two synthetic frames.
"""

from __future__ import annotations

import asyncio
import functools
from types import SimpleNamespace
from typing import List

import pytest

import components.tracking as tracking
from components.shapes import LocatedShape
from components.tracking import get_tracked_objects, match_to_known, tracked_shapes_from_detections


def async_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _det(class_name, x_min, y_min, x_max, y_max, confidence=0.9, **extra_attrs):
    return SimpleNamespace(
        class_name=class_name,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        confidence=confidence,
        **extra_attrs,
    )


# ---------------------------------------------------------------------------
# tracked_shapes_from_detections / get_tracked_objects: track_id extraction
# ---------------------------------------------------------------------------


def test_track_id_from_duck_typed_attribute():
    det = _det("cup", 10, 10, 30, 30, track_id="t-1")
    [shape] = tracked_shapes_from_detections([det])
    assert shape.track_id == "t-1"
    assert shape.label == "cup"
    assert shape.x == pytest.approx(20.0)
    assert shape.y == pytest.approx(20.0)


def test_track_id_from_extra_mapping():
    det = _det("can", 0, 0, 10, 10, extra={"track_id": "t-2"})
    [shape] = tracked_shapes_from_detections([det])
    assert shape.track_id == "t-2"


def test_track_id_embedded_in_class_name():
    det = _det("cup#42", 0, 0, 10, 10)
    [shape] = tracked_shapes_from_detections([det])
    assert shape.track_id == "42"
    assert shape.label == "cup"


def test_no_track_id_available():
    det = _det("person", 0, 0, 10, 10)
    [shape] = tracked_shapes_from_detections([det])
    assert shape.track_id is None
    assert shape.label == "person"


class _FakeTrackerClient:
    def __init__(self, dets_by_call):
        self._dets_by_call = list(dets_by_call)

    async def get_detections_from_camera(self, camera_name: str):
        assert camera_name == "cam"
        return self._dets_by_call.pop(0)


class _FakeVisionClient:
    last_args = None

    @classmethod
    def from_robot(cls, machine, name):
        cls.last_args = (machine, name)
        return machine.tracker_client


@async_test
async def test_get_tracked_objects_two_frames_attach_track_id(monkeypatch):
    frame1 = [_det("block#1", 0, 0, 20, 20), _det("block#2", 100, 100, 120, 120)]
    frame2 = [_det("block#1", 2, 2, 22, 22), _det("block#3", 200, 200, 220, 220)]
    fake_client = _FakeTrackerClient([frame1, frame2])
    machine = SimpleNamespace(tracker_client=fake_client)
    monkeypatch.setattr(tracking, "VisionClient", _FakeVisionClient)

    shapes1 = await get_tracked_objects(machine, "tracker-1", "cam")
    shapes2 = await get_tracked_objects(machine, "tracker-1", "cam")

    assert _FakeVisionClient.last_args == (machine, "tracker-1")
    assert [s.track_id for s in shapes1] == ["1", "2"]
    assert [s.track_id for s in shapes2] == ["1", "3"]


# ---------------------------------------------------------------------------
# match_to_known
# ---------------------------------------------------------------------------


def _shape(label, x, y, track_id=None):
    return LocatedShape(label=label, x=x, y=y, z=0.0, track_id=track_id)


def test_match_by_track_id_marks_same_object():
    known = [_shape("block", 10, 10, track_id="t-1"), _shape("block", 500, 500, track_id="t-2")]
    new = [_shape("block", 11, 11, track_id="t-1"), _shape("block", 501, 501, track_id="t-2")]

    result = match_to_known(new, known)

    assert result["matched"] == {0: 0, 1: 1}
    assert result["new"] == []
    assert result["skip_replan"] == [0, 1]
    assert result["method"] == {0: "track_id", 1: "track_id"}


def test_new_track_id_with_no_match_is_new_not_xy_fallback():
    # Even though object 0's position is close to the known object, its
    # track_id doesn't match any known id -> must be reported as new, NOT
    # XY-fallback-matched onto the nearby known object.
    known = [_shape("block", 10, 10, track_id="t-1")]
    new = [_shape("block", 11, 11, track_id="t-99")]

    result = match_to_known(new, known)

    assert result["matched"] == {}
    assert result["new"] == [0]
    assert result["skip_replan"] == []


def test_nearest_xy_fallback_when_no_track_id():
    known = [_shape("cup", 100.0, 100.0), _shape("cup", 400.0, 400.0)]
    new = [_shape("cup", 108.0, 95.0), _shape("cup", 900.0, 900.0)]

    result = match_to_known(new, known, xy_tolerance=20.0)

    assert result["matched"] == {0: 0}
    assert result["method"] == {0: "nearest_xy"}
    assert result["new"] == [1]
    assert result["skip_replan"] == [0]


def test_nearest_xy_respects_tolerance():
    known = [_shape("cup", 0.0, 0.0)]
    new = [_shape("cup", 1000.0, 1000.0)]  # far outside any reasonable tolerance

    result = match_to_known(new, known, xy_tolerance=20.0)

    assert result["matched"] == {}
    assert result["new"] == [0]


def test_each_known_object_claimed_at_most_once():
    known = [_shape("cup", 50.0, 50.0)]
    new = [_shape("cup", 51.0, 51.0), _shape("cup", 52.0, 52.0)]

    result = match_to_known(new, known, xy_tolerance=20.0)

    # Only one new object can claim the single known object; the other is new.
    assert len(result["matched"]) == 1
    assert len(result["new"]) == 1
    assert sorted(result["matched"].keys()) + result["new"] == [0, 1]


def test_empty_known_means_everything_is_new():
    new = [_shape("cup", 1.0, 1.0, track_id="t-1"), _shape("block", 2.0, 2.0)]
    result = match_to_known(new, [])
    assert result["matched"] == {}
    assert result["new"] == [0, 1]
    assert result["skip_replan"] == []


def test_empty_new_returns_all_empty():
    known = [_shape("cup", 1.0, 1.0, track_id="t-1")]
    result = match_to_known([], known)
    assert result == {"matched": {}, "new": [], "skip_replan": [], "method": {}}
