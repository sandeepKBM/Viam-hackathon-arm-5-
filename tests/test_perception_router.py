"""Offline tests for components/perception_router.py -- no network, no vision deps."""

import asyncio

import pytest

from components.perception_router import PerceptionRouter


class FakeService:
    def __init__(self, dets=None, grasp=None, fail=False):
        self._dets = dets or []
        self._grasp = grasp or {}
        self._fail = fail
        self.detections_calls = 0
        self.localize_calls = 0

    async def detections(self):
        self.detections_calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return self._dets

    async def localize(self, obj, hint_xy=None):
        self.localize_calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return {**self._grasp, "label": obj}


IDENTITY = lambda o: o  # skip the LocatedShape conversion in routing tests


def run(coro):
    return asyncio.run(coro)


def test_disabled_is_pure_local_passthrough():
    svc = FakeService(dets=[{"label": "x", "world_xyz_mm": [1, 2, 3]}])
    r = PerceptionRouter(svc, enabled=False, to_located_shape=IDENTITY)
    assert r.using_service is False
    called = {"n": 0}

    async def local():
        called["n"] += 1
        return ["LOCAL"]

    assert run(r.detections(local)) == ["LOCAL"]
    assert called["n"] == 1
    assert svc.detections_calls == 0  # service never touched


def test_no_service_even_if_enabled_uses_local():
    r = PerceptionRouter(None, enabled=True, to_located_shape=IDENTITY)
    assert r.using_service is False

    async def local():
        return ["LOCAL"]

    assert run(r.detections(local)) == ["LOCAL"]


def test_enabled_service_path_used_and_converted():
    dets = [{"label": "cup", "world_xyz_mm": [10, 20, 30], "score": 0.9}]
    svc = FakeService(dets=dets)
    r = PerceptionRouter(svc, enabled=True, to_located_shape=IDENTITY)
    assert r.using_service is True

    async def local():
        raise AssertionError("local should not be called")

    out = run(r.detections(local))
    assert out == dets
    assert svc.detections_calls == 1


def test_service_failure_falls_back_to_local():
    svc = FakeService(fail=True)
    r = PerceptionRouter(svc, enabled=True, to_located_shape=IDENTITY)
    called = {"n": 0}

    async def local():
        called["n"] += 1
        return ["LOCAL"]

    assert run(r.detections(local)) == ["LOCAL"]
    assert svc.detections_calls == 1 and called["n"] == 1  # tried service, then local


def test_localize_service_path_and_fallback():
    svc = FakeService(grasp={"world_xyz_mm": [1, 2, 3], "grasp_type": "top_down"})
    r = PerceptionRouter(svc, enabled=True, to_located_shape=IDENTITY)
    res = run(r.localize("can"))
    assert res["label"] == "can" and res["grasp_type"] == "top_down"

    # failure + fallback
    svc2 = FakeService(fail=True)
    r2 = PerceptionRouter(svc2, enabled=True, to_located_shape=IDENTITY)

    async def local():
        return {"world_xyz_mm": [0, 0, 0], "grasp_type": "local"}

    assert run(r2.localize("can", local_fallback=local))["grasp_type"] == "local"

    # failure + no fallback -> raises
    with pytest.raises(RuntimeError):
        run(r2.localize("can"))


def test_detection_to_located_shape_builds_real_shape():
    # Exercises the real converter against components.shapes.LocatedShape.
    from components.perception_router import detection_to_located_shape

    ls = detection_to_located_shape({"label": "cup", "world_xyz_mm": [1.0, 2.0, 3.0], "score": 0.7})
    assert ls.label == "cup" and (ls.x, ls.y, ls.z) == (1.0, 2.0, 3.0)
    assert ls.score == 0.7 and ls.canonical_label == "cup"


# --- build_router factory + env gating ---------------------------------------

def test_router_enabled_from_env(monkeypatch):
    from components.perception_router import router_enabled_from_env

    monkeypatch.delenv("ON_MACHINE_PERCEPTION", raising=False)
    assert router_enabled_from_env() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ON_MACHINE_PERCEPTION", truthy)
        assert router_enabled_from_env() is True
    monkeypatch.setenv("ON_MACHINE_PERCEPTION", "0")
    assert router_enabled_from_env() is False


def test_build_router_disabled_is_passthrough():
    from components.perception_router import build_router

    r = build_router(object(), enabled=False, verbose=False)
    assert r.using_service is False  # machine never touched


def test_build_router_unreachable_service_falls_back_to_local():
    # enabled=True but a bogus "machine" -> GraspServiceClient construction fails
    # -> build_router degrades to a disabled (local) router instead of raising.
    from components.perception_router import build_router

    r = build_router(object(), enabled=True, verbose=False)
    assert r.using_service is False
