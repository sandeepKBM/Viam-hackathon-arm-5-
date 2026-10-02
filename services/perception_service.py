"""Real perception service: fuses Moondream + OWLv2 + SAM into one endpoint.

``POST /detect`` takes a single image (no live camera / robot involved -- this
service is offline w.r.t. the arm and only ever processes frames it's handed)
and runs up to three of color-sort's own detectors against it, then returns
one enriched, JSON-serializable list of detections:

  * **Moondream** -- named-object detection via the color-sort Moondream host
    (``components/moondream_client.py`` -> a running ``moondream/server.py``
    process). Supplies free-form object *names*; reports no confidence score.
  * **OWLv2** -- open-vocabulary detection via the ported
    ``components/zeroshot.py`` (from the source ``viam_5`` repo). Supplies
    boxes *and* a per-detection confidence ``score``.
  * **SAM** -- ``components/sam.py``'s ``refine_shape_with_sam``: segments
    each fused detection's crop and derives a gripper yaw from the mask's
    ``minAreaRect`` angle.

Detections from Moondream and OWLv2 are fused by greedy IoU clustering (see
``_fuse_detections``): a box that both detectors agree on becomes one
detection carrying OWLv2's box/score and both detectors' raw labels; a box
only one of them saw is kept as-is. Every fused detection is then run through
``components.canonicalize.canonicalize_label`` to get color-sort's pick-object
vocabulary (``red, yellow, can, cup, airpods, pen, bottle``, or the generic
"<color> <noun>" fallback), and -- if requested -- through SAM for a mask
summary + yaw.

Which detectors run is configurable per request (``run_moondream``,
``run_owlv2``, ``run_sam`` form fields) or via env vars
(``PERCEPTION_RUN_MOONDREAM`` / ``PERCEPTION_RUN_OWLV2`` / ``PERCEPTION_RUN_SAM``,
default true/true/false), so the same code can run as one unified service now
and be split later (e.g. SAM on one Mac, Moondream+OWLv2 on another) just by
running two copies of this process with different env vars / --port and
pointing ``services/services.yaml`` (or ``VOICEUQ_PERCEPTION_URL``) at
whichever one a caller needs.

Deps actually needed to RUN the models (none of these are required just to
import this module -- see the "lazy imports" note below):

  * FastAPI/uvicorn/python-multipart -- the serving stack itself (already in
    ``requirements.txt``; not a "heavy model lib", needed for any endpoint at
    all, including ``/health``).
  * OWLv2:     ``pip install torch transformers pillow accelerate``
  * SAM:       ``pip install torch transformers pillow`` (+ opencv/numpy,
               already required elsewhere) -- downloads ``facebook/sam-vit-base``
               on first use.
  * Moondream: a separate running ``python moondream/server.py`` process
               (``pip install moondream transformers`` for THAT process),
               reachable at ``MOONDREAM_URL`` (default
               ``http://127.0.0.1:8768``). This service talks to it over
               HTTP via ``components/moondream_client.py``; it does not load
               the Moondream model itself.

Lazy imports: torch/transformers/PIL and the moondream client call all
happen *inside* the request-handling functions below (``_moondream_detect``,
``_owlv2_detect``, ``_sam_refine``), not at module import time, so
``import services.perception_service`` succeeds even on a machine with none
of those installed -- only ``fastapi``/``cv2``/``numpy`` (the normal serving +
image-decoding stack) are needed up front.

Run it:

    cd <worktree>
    python -m uvicorn services.perception_service:app --host 127.0.0.1 --port 8801

Smoke-test once the deps above are installed (and, for Moondream, once
``python moondream/server.py`` is running separately):

    curl http://127.0.0.1:8801/health

    curl -s http://127.0.0.1:8801/detect \\
         -F "image=@out/frame.png" \\
         -F "run_moondream=true" -F "run_owlv2=true" -F "run_sam=true" \\
         -F "labels=red block,yellow block,cup,pen,soda can" | python -m json.tool

Or from Python, via the shared client helper (matches how ``uq``/``cognition``
will call this service):

    from services.base import call_service
    with open("out/frame.png", "rb") as f:
        result = call_service("perception", "/detect", files={"image": ("frame.png", f)})
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import File, Form, HTTPException, UploadFile

from components.canonicalize import canonicalize_label
from services.base import make_service

app = make_service("perception")


# --------------------------------------------------------------------------
# Config: which detectors run by default, and how fusion matches boxes.
# --------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")


RUN_MOONDREAM_DEFAULT = _env_bool("PERCEPTION_RUN_MOONDREAM", True)
RUN_OWLV2_DEFAULT = _env_bool("PERCEPTION_RUN_OWLV2", True)
RUN_SAM_DEFAULT = _env_bool("PERCEPTION_RUN_SAM", False)
FUSE_IOU_DEFAULT = float(os.environ.get("PERCEPTION_FUSE_IOU", "0.35"))


# --------------------------------------------------------------------------
# Image decoding (cv2/numpy only -- not "heavy" model deps, but still kept
# local to the functions that need them so a client-only import stays cheap).
# --------------------------------------------------------------------------


def _decode_image_bytes(data: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image bytes (not a recognizable image format)")
    return bgr


def _parse_labels(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    labels = [tok.strip() for tok in raw.split(",") if tok.strip()]
    return labels or None


# --------------------------------------------------------------------------
# Per-detector adapters. Each returns a list of plain dicts:
#   {label, box(x,y,w,h px), cx, cy, score(float|None), source}
# so fusion doesn't need to know each detector's native return type.
# --------------------------------------------------------------------------


def _moondream_detect(image_path: str, labels: Optional[list[str]]) -> list[dict]:
    """Named-object detection via color-sort's Moondream host.

    Talks to ``moondream/server.py`` over HTTP through
    ``components/moondream_client.py`` -- this function itself only needs
    that HTTP round trip (stdlib urllib under the hood), no model deps.
    """
    import cv2

    from components.moondream_client import detect_objects_host

    data = detect_objects_host(image_path, labels=labels)
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]

    out: list[dict] = []
    for obj in data.get("objects", []):
        x0 = float(obj["x_min"]) * w
        y0 = float(obj["y_min"]) * h
        x1 = float(obj["x_max"]) * w
        y1 = float(obj["y_max"]) * h
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        out.append(
            {
                "label": obj.get("label", ""),
                "box": (int(x0), int(y0), int(bw), int(bh)),
                "cx": int(x0 + bw / 2),
                "cy": int(y0 + bh / 2),
                "score": None,  # Moondream's detect() reports no confidence
                "source": "moondream",
            }
        )
    return out


def _owlv2_detect(bgr, prompts: Optional[list[str]], threshold: Optional[float]) -> list[dict]:
    """Open-vocab detection via the ported ``components/zeroshot.py`` (OWLv2).

    Imports torch/transformers/PIL lazily (inside ``components.zeroshot``
    itself); this function's own import of that module is also deferred to
    call time so this service's module import never needs them.
    """
    from components.zeroshot import detect_objects

    shapes = detect_objects(bgr, prompts=prompts, threshold=threshold)
    out: list[dict] = []
    for s in shapes:
        out.append(
            {
                "label": s.label,
                "box": tuple(int(v) for v in s.box),
                "cx": int(s.cx),
                "cy": int(s.cy),
                "score": s.score,
                "source": "owlv2",
            }
        )
    return out


def _box_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax0, ay0, ax1, ay1 = ax, ay, ax + aw, ay + ah
    bx0, by0, bx1, by1 = bx, by, bx + bw, by + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _fuse_detections(dets: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy cross-detector IoU clustering into one merged list.

    Detections carrying a numeric ``score`` (OWLv2) are matched first and
    become the merged box/score; a same-object detection from a *different*
    source (Moondream) that overlaps it above ``iou_threshold`` contributes
    its label + source tag but not its (coarser, unscored) box. Detections
    from the SAME source never merge with each other here (that's each
    detector's own NMS responsibility, e.g. Moondream's own ``nms`` in
    ``moondream/detect.py``).
    """
    ordered = sorted(dets, key=lambda d: (d["score"] is None, -(d["score"] or 0.0)))
    used = [False] * len(ordered)
    merged: list[dict] = []

    for i, primary in enumerate(ordered):
        if used[i]:
            continue
        used[i] = True
        raw_labels = {primary["source"]: primary["label"]}
        sources = {primary["source"]}
        for j in range(i + 1, len(ordered)):
            if used[j] or ordered[j]["source"] == primary["source"]:
                continue
            if _box_iou(primary["box"], ordered[j]["box"]) >= iou_threshold:
                used[j] = True
                sources.add(ordered[j]["source"])
                raw_labels[ordered[j]["source"]] = ordered[j]["label"]
        merged.append(
            {
                "label": primary["label"],
                "box": primary["box"],
                "cx": primary["cx"],
                "cy": primary["cy"],
                "score": primary["score"],
                "sources": sorted(sources),
                "raw_labels": raw_labels,
            }
        )
    return merged


def _mask_summary(mask) -> Optional[dict]:
    """bbox + pixel-area summary of a full-frame binary mask -- never the
    raw array (which can be megabytes for a full-resolution frame)."""
    import numpy as np

    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"bbox": [0, 0, 0, 0], "area_px": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "area_px": int((mask > 0).sum())}


def _sam_refine(bgr, det: dict, canonical: str) -> tuple[Optional[dict], Optional[float]]:
    """Segment ``det``'s box with SAM, return (mask_summary, yaw_degrees).

    ``refine_shape_with_sam`` -> ``refresh_from_mask`` rebuilds the returned
    ``DetectedShape`` from its own contour fit (see
    ``components/shapes.py::_classify_contour``), which does NOT carry the
    original detector's ``score`` forward -- only ``color``/``label`` are
    preserved. We re-apply the fused detection's score onto the refined
    shape afterward so ``DetectedShape.score`` still reflects OWLv2/Moondream
    confidence, per this service's contract (score capture into
    ``DetectedShape.score``), not SAM's own (unrelated) IoU score.
    """
    from components.sam import refine_shape_with_sam
    from components.shapes import DetectedShape

    x, y, w, h = det["box"]
    short, long_ = sorted((max(w, 1), max(h, 1)))
    shape = DetectedShape(
        label=det["label"],
        cx=det["cx"],
        cy=det["cy"],
        area=float(w * h),
        vertices=4,
        aspect_ratio=long_ / max(short, 1e-6),
        box=(x, y, w, h),
        color=canonical if canonical in ("red", "yellow") else "",
        score=det["score"],
    )
    refined = refine_shape_with_sam(bgr, shape)
    refined.score = det["score"]  # SAM/refresh_from_mask does not preserve this
    return _mask_summary(refined.mask), float(refined.angle)


# --------------------------------------------------------------------------
# The endpoint.
# --------------------------------------------------------------------------


@app.post("/detect")
async def detect(
    image: Optional[UploadFile] = File(default=None),
    image_base64: Optional[str] = Form(default=None),
    labels: Optional[str] = Form(default=None),
    moondream_labels: Optional[str] = Form(default=None),
    threshold: Optional[float] = Form(default=None),
    run_moondream: bool = Form(default=RUN_MOONDREAM_DEFAULT),
    run_owlv2: bool = Form(default=RUN_OWLV2_DEFAULT),
    run_sam: bool = Form(default=RUN_SAM_DEFAULT),
    fuse_iou: float = Form(default=FUSE_IOU_DEFAULT),
) -> dict[str, Any]:
    """Fuse Moondream + OWLv2 + SAM detections for one image.

    Request (multipart/form-data): either an ``image`` file upload or an
    ``image_base64`` field (bare base64, or a ``data:image/...;base64,``
    URI -- the prefix is stripped), plus optional form fields:

      * ``labels``            -- comma-separated prompts, used for OWLv2's
                                  candidate labels AND (unless overridden by
                                  ``moondream_labels``) Moondream's label list.
                                  Omit to use each detector's own tuned defaults.
      * ``moondream_labels``  -- comma-separated labels for Moondream only.
      * ``threshold``         -- OWLv2 score threshold (default from
                                  ``components.zeroshot.ZS_THRESHOLD``).
      * ``run_moondream`` / ``run_owlv2`` / ``run_sam`` -- booleans
                                  ("true"/"false"), default from the
                                  ``PERCEPTION_RUN_*`` env vars.
      * ``fuse_iou``          -- IoU threshold for cross-detector merging
                                  (default ``PERCEPTION_FUSE_IOU``, 0.35).

    Response JSON:

        {
          "image": {"width": W, "height": H},
          "detectors_run": {"moondream": bool, "owlv2": bool, "sam": bool},
          "detections": [
            {
              "label": "soda can",             # best raw label (OWLv2 if matched, else Moondream)
              "canonical_label": "can",        # components.canonicalize.canonicalize_label(label)
              "box": [x, y, w, h],             # pixels
              "cx": int, "cy": int,
              "score": 0.83,                   # OWLv2 confidence, or null if only Moondream saw it
              "sources": ["moondream", "owlv2"],
              "raw_labels": {"moondream": "soda can", "owlv2": "can"},
              "mask": {"bbox": [x,y,w,h], "area_px": 1234} | null,  # only when run_sam=true
              "yaw": 12.4 | null                                    # degrees, only when run_sam=true
            },
            ...
          ]
        }
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="provide an 'image' file or 'image_base64' field")

    if image is not None:
        raw = await image.read()
        suffix = Path(image.filename or "upload.png").suffix or ".png"
    else:
        b64 = image_base64.split(",", 1)[-1] if "," in image_base64 else image_base64
        try:
            raw = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid image_base64: {exc}") from exc
        suffix = ".png"

    if not raw:
        raise HTTPException(status_code=400, detail="empty image payload")

    try:
        bgr = _decode_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h, w = bgr.shape[:2]
    label_list = _parse_labels(labels)
    moondream_label_list = _parse_labels(moondream_labels) or label_list

    raw_dets: list[dict] = []
    tmp_path: Optional[str] = None
    try:
        if run_moondream:
            # Moondream's host API takes a filesystem path (it reads with
            # PIL.Image.open server-side), so the decoded bytes are written
            # to a temp file rather than re-encoded, keeping OWLv2/SAM and
            # Moondream looking at byte-identical pixels.
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                raw_dets.extend(_moondream_detect(tmp_path, moondream_label_list))
            except RuntimeError as exc:
                # Moondream host not reachable -- degrade gracefully rather
                # than failing the whole /detect call if OWLv2/SAM alone can
                # still answer.
                print(f"perception: moondream detect skipped: {exc}", flush=True)

        if run_owlv2:
            raw_dets.extend(_owlv2_detect(bgr, label_list, threshold))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    merged = _fuse_detections(raw_dets, fuse_iou)

    detections: list[dict[str, Any]] = []
    for det in merged:
        canonical = canonicalize_label(det["label"])
        entry: dict[str, Any] = {
            "label": det["label"],
            "canonical_label": canonical,
            "box": list(det["box"]),
            "cx": det["cx"],
            "cy": det["cy"],
            "score": det["score"],
            "sources": det["sources"],
            "raw_labels": det["raw_labels"],
            "mask": None,
            "yaw": None,
        }
        if run_sam:
            try:
                mask_summary, yaw = _sam_refine(bgr, det, canonical)
                entry["mask"] = mask_summary
                entry["yaw"] = yaw
            except Exception as exc:  # SAM is best-effort geometry refinement
                print(f"perception: SAM refine failed for {det['label']}: {exc}", flush=True)
        detections.append(entry)

    return {
        "image": {"width": w, "height": h},
        "detectors_run": {
            "moondream": run_moondream,
            "owlv2": run_owlv2,
            "sam": run_sam,
        },
        "detections": detections,
    }
