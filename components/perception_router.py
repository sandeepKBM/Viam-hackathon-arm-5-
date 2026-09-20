"""Route perception to the on-machine grasp-service when available, else local.

WHY (measured over the Viam cloud relay from westeros)
------------------------------------------------------
Perception payloads are the slow part: ``camera.get_images()`` ~2 s,
``camera.get_point_cloud()`` ~12.5 s (14.7 MB). Control calls are ~36 ms. When
the on-machine module (``grasp_perception_module/``, model
``hackathons:grasp-perception:grasp-service``) is deployed, its ``detections()`` /
``localize()`` run ON the arm and return small results (~36 ms).

This router prefers that fast path and **falls back to the existing local
callable on ANY failure** (service not configured, not deployed, or a runtime
error), so behavior is safe whether or not the module is live. It is opt-in:
constructed with ``enabled=False`` (or no service) it is a transparent
pass-through to the local path, i.e. exactly today's behavior.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, List, Optional, Sequence

log = logging.getLogger(__name__)


def router_enabled_from_env() -> bool:
    """True when ON_MACHINE_PERCEPTION is set truthy."""
    return os.environ.get("ON_MACHINE_PERCEPTION", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def build_router(machine: Any, *, enabled: Optional[bool] = None, verbose: bool = True) -> "PerceptionRouter":
    """Construct a PerceptionRouter for ``machine``, honoring ON_MACHINE_PERCEPTION.

    Single place every live entrypoint builds the router, so the opt-in + graceful
    "service not deployed -> local fallback" behavior stays consistent. When
    ``enabled`` is None it is read from the env. If the on-machine service can't
    be reached, returns a disabled (pass-through) router.
    """
    if enabled is None:
        enabled = router_enabled_from_env()
    service = None
    if enabled:
        try:
            from components.grasp_service_client import GraspServiceClient
            service = GraspServiceClient(machine)
        except Exception as exc:  # noqa: BLE001 -- absent/undeployed service -> local
            if verbose:
                print(f"[perception] on-machine service unavailable ({exc}); using local detector")
            enabled = False
    router = PerceptionRouter(service, enabled=enabled)
    if verbose and router.using_service:
        print("[perception] using ON-MACHINE grasp-service (fast path)")
    return router

# () -> List[LocatedShape]
LocalDetectFn = Callable[[], Awaitable[List[Any]]]


def detection_to_located_shape(obj: dict) -> Any:
    """Convert an on-machine ``detections()`` dict into a ``LocatedShape``.

    Service shape: ``{"label": str, "world_xyz_mm": [x, y, z], "score": float}``.
    Imported lazily so this module stays importable without the vision deps.
    """
    from components.shapes import LocatedShape  # lazy: shapes pulls numpy/cv2

    x, y, z = obj["world_xyz_mm"]
    label = obj.get("label", "")
    return LocatedShape(
        label=label,
        x=float(x),
        y=float(y),
        z=float(z),
        score=obj.get("score"),
        canonical_label=label,
    )


class PerceptionRouter:
    """Prefer the on-machine grasp-service; fall back to the local path.

    Parameters
    ----------
    service:
        A ``GraspServiceClient`` (or ``None``). When ``None``, the router is a
        pass-through to the local fallback.
    enabled:
        Master switch. The fast path is used only when ``enabled`` is true AND a
        service was provided. Default ``False`` keeps current behavior.
    to_located_shape:
        Injectable converter (defaults to :func:`detection_to_located_shape`),
        so tests can avoid importing the heavy vision stack.
    """

    def __init__(
        self,
        service: Any = None,
        *,
        enabled: bool = False,
        to_located_shape: Callable[[dict], Any] = detection_to_located_shape,
    ) -> None:
        self._svc = service
        self._enabled = bool(enabled) and service is not None
        self._to_located_shape = to_located_shape

    @property
    def using_service(self) -> bool:
        """True when calls will try the on-machine service first."""
        return self._enabled

    async def detections(self, local_fallback: LocalDetectFn) -> List[Any]:
        """Return ``List[LocatedShape]`` -- service path when enabled, else local.

        On any service error, logs and falls back to ``local_fallback`` so the
        run never breaks because the module is missing or flaky.
        """
        if self._enabled:
            try:
                objs = await self._svc.detections()
                return [self._to_located_shape(o) for o in objs]
            except Exception as exc:  # noqa: BLE001 -- fallback is the whole point
                log.warning(
                    "on-machine detections failed (%s); falling back to local perception",
                    exc,
                )
        return await local_fallback()

    async def localize(
        self,
        obj: str,
        hint_xy: Optional[Sequence[float]] = None,
        *,
        local_fallback: Optional[Callable[[], Awaitable[dict]]] = None,
    ) -> dict:
        """Localize a grasp on the machine; fall back to ``local_fallback`` if given.

        Returns the small grasp dict from the service. If the service path fails
        and no ``local_fallback`` is supplied, the original error propagates.
        """
        if self._enabled:
            try:
                return await self._svc.localize(obj, hint_xy)
            except Exception as exc:  # noqa: BLE001
                if local_fallback is None:
                    raise
                log.warning(
                    "on-machine localize(%r) failed (%s); falling back to local", obj, exc
                )
        if local_fallback is None:
            raise RuntimeError(
                "PerceptionRouter.localize called with no service and no local_fallback"
            )
        return await local_fallback()
