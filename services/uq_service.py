"""UQ (uncertainty / difficulty) service -- FastAPI wrapper around
``components/uq.py`` (W3), following the ``services.base.make_service`` /
``services.base.call_service`` pattern used by the other voice-uq services.

Run
---
::

    cd /common/users/ss5772/viam_5-voice-uq
    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m uvicorn \\
        services.uq_service:app --host 127.0.0.1 --port 8802

(port comes from ``services/services.yaml``'s ``uq`` entry; override with
``--port`` if you've repointed it via ``VOICEUQ_UQ_PORT``.) Or just::

    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m services.uq_service

which reads the same config and runs uvicorn directly.

``POST /score``
----------------
multipart/form-data:

  ``image``        (required) the color frame the detections came from
                    (jpg/png bytes) -- also what gets jittered for
                    augmentation-consistency.
  ``detections``   (optional) JSON-encoded list of detection dicts, the
                    shape the perception service's ``/detect`` is expected
                    to return for a single frame, e.g.::

                        [{"label": "red_cube", "box": [120, 80, 40, 38],
                          "score": 0.83, "aspect_ratio": 1.05,
                          "area": 1520.0}, ...]

                    ``box`` is ``(x, y, w, h)`` in pixel coordinates
                    matching ``image``; extra/missing keys are fine
                    (duck-typed, same as ``components.uq``). If omitted,
                    this service calls perception's own ``POST /detect``
                    ONCE on the raw image to get a starting detection list
                    -- if that also fails, ``/score`` still responds 200
                    with ``{"ok": false, ...}`` rather than raising.
  ``n``            (optional, default ``components.uq.DEFAULT_N`` == 5)
                    augmentation passes PER DETECTION. ``n<=0`` skips
                    augmentation-consistency entirely (fast score+geometry
                    path). This is a timing-budget knob (W6): every unit of
                    ``n`` is one more POST to perception per detection
                    (mitigated somewhat by the per-request cache below), so
                    callers under a tight cycle-time budget should pass a
                    small ``n`` or ``0``.
  ``seed``         (optional) RNG seed forwarded to
                    ``components.uq.augmentation_consistency``. Fixing it
                    also makes the jitter sequence identical across
                    detections in the same request, so the per-request
                    perception-response cache (keyed by the exact jittered
                    JPEG bytes) collapses ``n * len(detections)`` calls down
                    towards ``n`` real HTTP round-trips.

Returns (always HTTP 200 -- this service does not raise 5xx to the caller;
see "Degrade gracefully" below)::

    {
      "ok": true,
      "n_requested": 5,
      "n_used": 5,                 // 0 if augmentation-consistency was skipped/degraded
      "augmented": true,           // whether consistency was actually computed
      "degraded_reason": null,     // why augmented is false, when it is
      "from_perception": false,    // true if `detections` had to be fetched from perception
      "detections": [
        {..original fields.., "score": 0.83, "difficulty": 0.21,
         "consistency": 0.95, "consistency_detail": {...}}
        , ...
      ]
    }

Augmentation-consistency
-------------------------
For each detection, ``components.uq.augmentation_consistency`` jitters
``image`` ``n`` times (small translate + crop/zoom + brightness) and, for
each jittered copy, calls the injected ``detector_fn``. Here that
``detector_fn`` is ``_cached_perception_detector``: it POSTs the jittered
frame to the perception service via
``services.base.call_service("perception", "/detect", files={"image":
(...)})`` and parses the response back into detection dicts (accepts a bare
JSON list, or ``{"detections": [...]}`` / ``{"objects": [...]}`` /
``{"shapes": [...]}``). Per detector call is cached in-request by the exact
jittered frame's encoded bytes, so a fixed ``seed`` re-uses one perception
call across every detection's pass instead of paying for each one
separately.

Each detection's best-IoU match across its ``n`` jittered detections feeds
``components.uq.difficulty(score=..., consistency=..., aspect_ratio=...,
area=...)`` to fuse confidence + augmentation-consistency + geometry into a
single per-object ``difficulty in [0, 1]``.

Degrade gracefully
-------------------
- ``n <= 0`` -> skip augmentation-consistency outright; every detection is
  scored on score+geometry only (``consistency=None``).
- perception service unreachable when probed for augmentation (i.e. the
  first ``call_service("perception", ...)`` raises) -> caught, recorded in
  ``degraded_reason``, and every detection falls back to score+geometry
  instead of the request failing.
- no ``detections`` given AND perception is unreachable -> there is nothing
  to score; responds ``{"ok": false, "error": ..., "detections": []}``
  (HTTP 200) -- the service itself stays up, it's the caller's job to check
  ``ok``.

cv2/numpy are imported lazily (inside functions) so this module -- and
``services.base``/``services.config`` re-exports -- stay importable even
without an image-processing stack installed.

Smoke check (NOT a test file -- just an inline sanity path)
-------------------------------------------------------------
::

    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m services.uq_service --smoke

Fuses two synthetic stand-in detectors (no perception service, no network)
-- one that reports a stable box/label/score every pass, one that wanders
the box, sometimes mislabels, sometimes misses -- through the exact same
``augmentation_consistency`` + ``difficulty`` calls ``/score`` uses, and
asserts the jittery/ambiguous object comes out with a higher difficulty.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from components import uq as uq_lib
from services.base import call_service, make_service

app = make_service("uq")


# --------------------------------------------------------------------------
# image <-> bytes helpers (lazy cv2/numpy)
# --------------------------------------------------------------------------


def _decode_image(raw: bytes):
    import cv2
    import numpy as np

    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image bytes")
    return bgr


def _encode_image(bgr) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("could not encode frame to jpeg")
    return buf.tobytes()


# --------------------------------------------------------------------------
# perception client: parse its response shape + call it with jittered frames
# --------------------------------------------------------------------------


def _parse_perception_response(resp: Any) -> List[Dict[str, Any]]:
    """Accept a bare list, or a dict with detections under a few likely keys."""
    if isinstance(resp, list):
        return [d for d in resp if isinstance(d, dict)]
    if isinstance(resp, dict):
        for key in ("detections", "objects", "shapes"):
            val = resp.get(key)
            if isinstance(val, list):
                return [d for d in val if isinstance(d, dict)]
    return []


def _perception_detect(bgr, timeout: float = 15.0) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """POST ``bgr`` to the perception service's ``/detect``. Never raises."""
    try:
        jpg = _encode_image(bgr)
    except Exception as exc:
        return False, [], f"could not encode image: {exc}"
    try:
        resp = call_service(
            "perception",
            "/detect",
            files={"image": ("frame.jpg", jpg, "image/jpeg")},
            timeout=timeout,
        )
    except Exception as exc:  # call_service raises RuntimeError when unreachable
        return False, [], str(exc)
    return True, _parse_perception_response(resp), None


def _cached_perception_detector(cache: Dict[bytes, List[Dict[str, Any]]], timeout: float = 15.0):
    """Build the ``detector_fn`` injected into
    ``components.uq.augmentation_consistency``: POST the jittered frame to
    perception and parse detections, memoized per exact jittered-frame bytes
    so a fixed ``seed`` de-duplicates calls across detections in one request.
    """

    def _detector(aug_bgr):
        try:
            jpg = _encode_image(aug_bgr)
        except Exception:
            return []
        cached = cache.get(jpg)
        if cached is not None:
            return cached
        ok, dets, _err = _perception_detect(aug_bgr, timeout=timeout)
        result = dets if ok else []
        cache[jpg] = result
        return result

    return _detector


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------


# FastAPI/its File/Form markers are only resolved lazily, at request time,
# by Starlette's routing -- but importing `fastapi` itself happens here at
# module load. That's fine: FastAPI is a hard dependency of this service
# (see requirements.txt). ``services.base`` only imports it lazily to keep
# the *client* side (``call_service``) usable without FastAPI installed;
# this module is the *server* side and needs it up front.
from fastapi import File, Form, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402


@app.post("/score")
async def score(
    image: UploadFile = File(...),
    detections: Optional[str] = Form(None),
    n: int = Form(uq_lib.DEFAULT_N),
    seed: Optional[int] = Form(None),
):
    raw = await image.read()
    try:
        bgr = _decode_image(raw)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"bad image: {exc}", "detections": []})

    from_perception = False
    perception_reachable: Optional[bool] = None

    if detections:
        try:
            parsed = json.loads(detections)
            if not isinstance(parsed, list):
                raise ValueError("`detections` must be a JSON list of objects")
            dets: List[Dict[str, Any]] = [dict(d) for d in parsed if isinstance(d, dict)]
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"bad detections: {exc}", "detections": []})
    else:
        from_perception = True
        ok, dets, err = _perception_detect(bgr)
        perception_reachable = ok
        if not ok:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"no detections given and perception is unreachable: {err}",
                    "detections": [],
                }
            )
        if not dets:
            return JSONResponse(
                {
                    "ok": True,
                    "n_requested": int(n),
                    "n_used": 0,
                    "augmented": False,
                    "degraded_reason": "perception returned no detections",
                    "from_perception": True,
                    "detections": [],
                }
            )

    n_requested = int(n)
    degraded_reason: Optional[str] = None
    detector_fn = None

    if n_requested > 0:
        if perception_reachable is None:
            perception_reachable, _probe_dets, probe_err = _perception_detect(bgr)
        else:
            probe_err = None
        if perception_reachable:
            cache: Dict[bytes, List[Dict[str, Any]]] = {}
            detector_fn = _cached_perception_detector(cache)
        else:
            degraded_reason = (
                f"perception unreachable -- degraded to score+geometry only: {probe_err}"
            )
    else:
        degraded_reason = "n<=0: augmentation-consistency skipped by request"

    augmented = detector_fn is not None
    out: List[Dict[str, Any]] = []

    for det in dets:
        box = det.get("box")
        label = det.get("label")
        score_val = det.get("score")
        aspect_ratio = det.get("aspect_ratio")
        area = det.get("area")

        consistency_val: Optional[float] = None
        consistency_detail: Optional[Dict[str, Any]] = None
        if detector_fn is not None and box is not None:
            agreement = uq_lib.augmentation_consistency(
                detector_fn,
                bgr,
                tuple(box),
                ref_label=label,
                n=n_requested,
                seed=seed,
            )
            consistency_val = agreement["consistency"]
            consistency_detail = agreement

        d = uq_lib.difficulty(
            score=score_val,
            consistency=consistency_val,
            aspect_ratio=aspect_ratio,
            area=area,
        )

        annotated = dict(det)
        annotated["score"] = score_val
        annotated["difficulty"] = d
        annotated["consistency"] = consistency_val
        if consistency_detail is not None:
            annotated["consistency_detail"] = consistency_detail
        out.append(annotated)

    return JSONResponse(
        {
            "ok": True,
            "n_requested": n_requested,
            "n_used": n_requested if augmented else 0,
            "augmented": augmented,
            "degraded_reason": degraded_reason,
            "from_perception": from_perception,
            "detections": out,
        }
    )


# --------------------------------------------------------------------------
# smoke check (inline sanity path -- NOT a pytest test file)
# --------------------------------------------------------------------------


def _smoke() -> None:
    """Exercise the exact fusion path ``/score`` uses --
    ``augmentation_consistency`` + ``difficulty`` -- with two synthetic
    stand-in detectors (no perception service, no network, no committed test
    file): one reports a stable box/label/score every pass, the other
    wanders the box, sometimes mislabels, sometimes misses entirely. Prints
    both and asserts the jittery/ambiguous object scores as more difficult.
    """
    import numpy as np

    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    ref_box = (80, 80, 40, 40)
    rng = np.random.default_rng(0)

    def stable_detector(_aug_bgr):
        return [{"box": ref_box, "label": "cube", "score": 0.92}]

    def jittery_detector(_aug_bgr):
        if rng.random() < 0.25:
            return []  # missed detection this pass
        dx, dy = int(rng.integers(-25, 25)), int(rng.integers(-25, 25))
        box = (80 + dx, 80 + dy, 40, 40)
        label = "cube" if rng.random() < 0.5 else "sphere"
        return [{"box": box, "label": label, "score": 0.55}]

    stable_agreement = uq_lib.augmentation_consistency(
        stable_detector, bgr, ref_box, ref_label="cube", n=8, seed=1
    )
    jittery_agreement = uq_lib.augmentation_consistency(
        jittery_detector, bgr, ref_box, ref_label="cube", n=8, seed=1
    )

    stable_d = uq_lib.difficulty(
        score=0.92,
        consistency=stable_agreement["consistency"],
        aspect_ratio=1.0,
        area=1600.0,
    )
    jittery_d = uq_lib.difficulty(
        score=0.55,
        consistency=jittery_agreement["consistency"],
        aspect_ratio=1.0,
        area=1600.0,
    )

    print(
        f"stable  consistency={stable_agreement['consistency']:.3f} "
        f"difficulty={stable_d:.3f}"
    )
    print(
        f"jittery consistency={jittery_agreement['consistency']:.3f} "
        f"difficulty={jittery_d:.3f}"
    )
    assert jittery_d > stable_d, (
        "expected the jittery/ambiguous object to be scored as more difficult "
        f"than the stable one (stable={stable_d:.3f}, jittery={jittery_d:.3f})"
    )
    print("smoke OK: jittery difficulty > stable difficulty")


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        _smoke()
    else:
        import uvicorn

        from services import config

        host, port = config.service_addr("uq")
        uvicorn.run(app, host=host, port=port)
