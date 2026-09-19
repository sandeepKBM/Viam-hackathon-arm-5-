"""Live debug snapshot for Ivo: camera frame, boxes, next pick, failed sessions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEBUG_ROOT = Path(os.environ.get("DEBUG_DIR", str(ROOT / "debug")))
LIVE_DIR = DEBUG_ROOT / "live"
FAIL_DIR = DEBUG_ROOT / "failures"

_BOX = {
    "red": (70, 70, 255),
    "yellow": (0, 210, 255),
    "can": (80, 180, 255),
    "cup": (220, 200, 80),
    "airpods": (200, 140, 255),
    "pen": (160, 220, 140),
    "bottle": (255, 160, 80),
}
_PICK = (180, 214, 255)

_state: dict[str, Any] = {
    "bgr": None,
    "predictions": [],
    "target": None,
    "context": {},
    "updated": None,
    "last_failure": None,
}


def _ensure_dirs() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)


def remember_frame(bgr: np.ndarray) -> None:
    _state["bgr"] = bgr.copy()
    _ensure_dirs()
    cv2.imwrite(str(LIVE_DIR / "frame.png"), bgr)


def _box_of(item: dict) -> Optional[tuple[int, int, int, int]]:
    box = item.get("bbox")
    if not box or len(box) != 4:
        return None
    return tuple(int(v) for v in box)


def prediction_from_block(block, dest: str | None = None, pick_z: float | None = None) -> dict:
    shape = getattr(block, "shape", None)
    return {
        "color": getattr(block, "color", "") or getattr(block, "label", ""),
        "xy_mm": [round(float(block.x), 1), round(float(block.y), 1)],
        "region_px": [round(float(getattr(block, "u", 0.0)), 1), round(float(getattr(block, "v", 0.0)), 1)],
        "depth_mm": round(float(getattr(block, "depth_mm", 0.0)), 1),
        "world_z_mm": round(float(block.z), 1),
        "pick_z_mm": None if pick_z is None else round(float(pick_z), 1),
        "bbox": list(shape.box) if shape and shape.box else None,
        "bin": dest,
        "yaw": round(float(getattr(block, "yaw", 0.0)), 1),
    }


def annotate(bgr: np.ndarray, predictions: list[dict], target: dict | None) -> np.ndarray:
    vis = bgr.copy()
    target_box = _box_of(target) if target else None
    for pred in predictions:
        box = _box_of(pred)
        if box is None:
            continue
        x, y, w, h = box
        color = _BOX.get(str(pred.get("color", "")), (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis,
            str(pred.get("color") or "obj"),
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if target_box is not None:
        x, y, w, h = target_box
        cv2.rectangle(vis, (x, y), (x + w, y + h), _PICK, 4)
        label = f"PICK {target.get('color', '')}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y0 = max(0, y - th - 14)
        cv2.rectangle(vis, (x, y0), (x + tw + 12, y0 + th + 10), _PICK, -1)
        cv2.putText(
            vis,
            label,
            (x + 6, y0 + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 16, 10),
            2,
            cv2.LINE_AA,
        )
    return vis


def publish(
    predictions: list[dict],
    target: dict | None = None,
    context: dict | None = None,
    bgr: np.ndarray | None = None,
) -> dict:
    if bgr is not None:
        remember_frame(bgr)
    frame = _state.get("bgr")
    _state["predictions"] = list(predictions)
    _state["target"] = target
    if context:
        _state["context"] = dict(context)
    _state["updated"] = datetime.now(timezone.utc).isoformat()
    _ensure_dirs()
    if frame is not None:
        boxed = annotate(frame, predictions, target)
        cv2.imwrite(str(LIVE_DIR / "annotated.png"), boxed)
    (LIVE_DIR / "state.json").write_text(
        json.dumps(snapshot(), indent=2),
        encoding="utf-8",
    )
    return snapshot()


def save_failure(target: dict | None, attempt: dict, error: str) -> Path:
    _ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    color = (target or {}).get("color") or "unknown"
    dest = FAIL_DIR / f"{stamp}_{color}"
    dest.mkdir(parents=True, exist_ok=True)
    frame = _state.get("bgr")
    if frame is not None:
        cv2.imwrite(str(dest / "frame.png"), frame)
        cv2.imwrite(
            str(dest / "annotated.png"),
            annotate(frame, _state.get("predictions") or [], target),
        )
    session = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "target": target,
        "attempt": attempt,
        "predictions": _state.get("predictions") or [],
        "context": _state.get("context") or {},
        "folder": str(dest),
    }
    (dest / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    _state["last_failure"] = {
        "folder": str(dest),
        "error": error,
        "color": color,
        "saved_at": session["saved_at"],
    }
    print(f"  debug session saved: {dest}", flush=True)
    return dest


def snapshot() -> dict:
    return {
        "updated": _state.get("updated"),
        "predictions": _state.get("predictions") or [],
        "target": _state.get("target"),
        "context": _state.get("context") or {},
        "last_failure": _state.get("last_failure"),
        "has_frame": _state.get("bgr") is not None,
        "frame_url": "/api/debug/frame.png",
    }


def annotated_path() -> Path:
    return LIVE_DIR / "annotated.png"


def raw_path() -> Path:
    return LIVE_DIR / "frame.png"
