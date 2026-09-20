"""Offline unit tests for GraspPerceptionService.

No live machine, no network, no torch/CUDA -- every test builds a `dependencies`
mapping of fake camera/vision handles (via `fake_backend.build_dependencies`)
and drives the service's real `reconfigure` + `do_command` path. Run with:

    cd /common/users/ss5772/viam_5
    python -m pytest grasp_perception_module/tests -q
"""

from __future__ import annotations

import asyncio
import functools

from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig

from components.grasp_affordance import GraspType
from grasp_perception_module.fake_backend import (
    FakeCamera,
    FakeDetector,
    FakeSegmenter,
    build_dependencies,
    default_scene,
    make_point_cloud_object,
    solid_cube,
)
from grasp_perception_module.service import GraspPerceptionService


def async_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def make_config(name="grasp1", **attrs):
    s = Struct()
    s.update(attrs)
    return ComponentConfig(name=name, attributes=s)


def make_service(
    camera_name="cam",
    segmenter_name="vision-segment",
    detector_name="shape-detector",
    objects=None,
    floor_z=None,
):
    """Build a configured GraspPerceptionService backed by fakes, without the
    Viam resource registry or a live machine."""
    cam = FakeCamera(name=camera_name, objects=objects)
    seg = FakeSegmenter(name=segmenter_name, objects=objects)
    det = FakeDetector(name=detector_name)
    deps = build_dependencies(cam, seg, det)

    attrs = dict(camera=camera_name, segmenter=segmenter_name, detector=detector_name)
    if floor_z is not None:
        attrs["floor_z"] = floor_z
    cfg = make_config(**attrs)

    svc = GraspPerceptionService(name="grasp1")
    svc.reconfigure(cfg, deps)
    return svc, cam, seg, det


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_returns_three_dep_names(self):
        cfg = make_config(camera="cam", segmenter="seg", detector="det")
        deps = GraspPerceptionService.validate_config(cfg)
        assert list(deps) == ["cam", "seg", "det"]

    def test_missing_camera_rejected(self):
        cfg = make_config(segmenter="seg", detector="det")
        try:
            GraspPerceptionService.validate_config(cfg)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_missing_segmenter_rejected(self):
        cfg = make_config(camera="cam", detector="det")
        try:
            GraspPerceptionService.validate_config(cfg)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_missing_detector_rejected(self):
        cfg = make_config(camera="cam", segmenter="seg")
        try:
            GraspPerceptionService.validate_config(cfg)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_bad_floor_z_rejected(self):
        cfg = make_config(camera="cam", segmenter="seg", detector="det", floor_z="not-a-number")
        try:
            GraspPerceptionService.validate_config(cfg)
            assert False, "expected ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# reconfigure grabs LOCAL handles
# ---------------------------------------------------------------------------


class TestReconfigure:
    def test_reconfigure_grabs_local_handles(self):
        svc, cam, seg, det = make_service()
        assert svc._camera is cam
        assert svc._segmenter is seg
        assert svc._detector is det

    def test_floor_z_from_config(self):
        svc, *_ = make_service(floor_z=123.45)
        assert svc.floor_z == 123.45

    def test_floor_z_defaults_from_constants(self):
        from components.constants import FLOOR_Z

        svc, *_ = make_service()
        assert svc.floor_z == float(FLOOR_Z)


# ---------------------------------------------------------------------------
# do_command: health
# ---------------------------------------------------------------------------


class TestHealth:
    @async_test
    async def test_health_reports_names(self):
        svc, *_ = make_service(
            camera_name="cam", segmenter_name="vision-segment", detector_name="shape-detector"
        )
        res = await svc.do_command({"cmd": "health"})
        assert res == {
            "ok": True,
            "camera": "cam",
            "segmenter": "vision-segment",
            "detector": "shape-detector",
        }


# ---------------------------------------------------------------------------
# do_command: detections
# ---------------------------------------------------------------------------


class TestDetections:
    @async_test
    async def test_detections_returns_small_objects(self):
        svc, cam, seg, det = make_service()
        res = await svc.do_command({"cmd": "detections"})
        assert res["ok"] is True
        objects = res["objects"]
        assert len(objects) == 3
        labels = {o["label"] for o in objects}
        assert labels == {"block", "bottle", "cup"}
        for o in objects:
            assert set(o.keys()) == {"label", "world_xyz_mm", "score"}
            assert len(o["world_xyz_mm"]) == 3
            assert all(isinstance(v, float) for v in o["world_xyz_mm"])
            assert isinstance(o["score"], float)

    @async_test
    async def test_detections_reads_local_segmenter_with_camera_name(self):
        svc, cam, seg, det = make_service(camera_name="cam")
        await svc.do_command({"cmd": "detections"})
        assert seg.calls == [("get_object_point_clouds", ("cam",))]

    @async_test
    async def test_detections_world_xyz_uses_geometry_center_mm(self):
        # center passed to the fake PCO must surface unchanged (mm, no scaling).
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "detections"})
        by_label = {o["label"]: o["world_xyz_mm"] for o in res["objects"]}
        assert by_label["block"] == [120.0, -40.0, 25.0]
        assert by_label["bottle"] == [300.0, 60.0, 65.0]

    @async_test
    async def test_detections_never_returns_raw_cloud(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "detections"})
        # No bytes/PCD anywhere in the response.
        for o in res["objects"]:
            for v in o.values():
                assert not isinstance(v, (bytes, bytearray))


# ---------------------------------------------------------------------------
# do_command: localize
# ---------------------------------------------------------------------------


class TestLocalize:
    @async_test
    async def test_localize_block_is_top_down(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "localize", "object": "block"})
        assert res["ok"] is True
        assert res["label"] == "block"
        assert res["grasp_type"] == GraspType.TOP_DOWN.value
        assert res["world_xyz_mm"] == [120.0, -40.0, 25.0]
        assert len(res["approach_vec"]) == 3
        assert isinstance(res["yaw_deg"], float)
        assert res["n_points"] > 0

    @async_test
    async def test_localize_bottle_is_side(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "localize", "object": "bottle"})
        assert res["ok"] is True
        assert res["grasp_type"] == GraspType.SIDE.value

    @async_test
    async def test_localize_cup_is_inside_outside(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "localize", "object": "cup"})
        assert res["ok"] is True
        assert res["grasp_type"] == GraspType.INSIDE_OUTSIDE.value

    @async_test
    async def test_localize_case_insensitive_substring(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "localize", "object": "BLOCK"})
        assert res["ok"] is True
        assert res["label"] == "block"

    @async_test
    async def test_localize_missing_object_field(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "localize"})
        assert res["ok"] is False
        assert "object" in res["error"]

    @async_test
    async def test_localize_unknown_object(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "localize", "object": "banana"})
        assert res["ok"] is False
        assert "not found" in res["error"]
        assert "block" in res["error"]  # lists what it did see

    @async_test
    async def test_localize_hint_xy_picks_nearest(self):
        # Two blocks; hint_xy should pick the one closest in XY.
        objs = [
            make_point_cloud_object(solid_cube(), label="block", center=(100.0, 0.0, 25.0)),
            make_point_cloud_object(solid_cube(), label="block", center=(400.0, 0.0, 25.0)),
        ]
        svc, *_ = make_service(objects=objs)
        near = await svc.do_command({"cmd": "localize", "object": "block", "hint_xy": [390.0, 5.0]})
        assert near["ok"] is True
        assert near["world_xyz_mm"] == [400.0, 0.0, 25.0]
        far = await svc.do_command({"cmd": "localize", "object": "block", "hint_xy": [90.0, -5.0]})
        assert far["world_xyz_mm"] == [100.0, 0.0, 25.0]

    @async_test
    async def test_localize_bad_hint_xy_rejected(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "localize", "object": "block", "hint_xy": "nope"})
        assert res["ok"] is False
        assert "hint_xy" in res["error"]


# ---------------------------------------------------------------------------
# do_command: dispatch + JSON-serializability
# ---------------------------------------------------------------------------


class TestDispatch:
    @async_test
    async def test_unknown_cmd_returns_error_not_raise(self):
        svc, *_ = make_service()
        res = await svc.do_command({"cmd": "do_a_backflip"})
        assert res["ok"] is False
        assert "unknown cmd" in res["error"]

    @async_test
    async def test_missing_cmd_returns_error(self):
        svc, *_ = make_service()
        res = await svc.do_command({})
        assert res["ok"] is False

    @async_test
    async def test_all_responses_are_json_serializable(self):
        import json

        svc, *_ = make_service()
        for command in (
            {"cmd": "health"},
            {"cmd": "detections"},
            {"cmd": "localize", "object": "block"},
            {"cmd": "localize", "object": "banana"},
            {"cmd": "bogus"},
        ):
            res = await svc.do_command(command)
            json.dumps(res)  # must not raise


class TestSegmenterFailureIsContained:
    @async_test
    async def test_detection_error_becomes_ok_false(self):
        svc, cam, seg, det = make_service()

        async def boom(camera_name, **kwargs):
            raise RuntimeError("camera exploded")

        seg.get_object_point_clouds = boom
        res = await svc.do_command({"cmd": "detections"})
        assert res["ok"] is False
        assert "camera exploded" in res["error"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
