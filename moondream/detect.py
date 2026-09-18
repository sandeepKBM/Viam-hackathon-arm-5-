"""Classify objects and draw bounding boxes with a loaded Moondream model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SEED_LABELS = [
    "red block",
    "yellow block",
    "blue sticky note",
    "pink sticky note",
    "green bin",
    "water bottle",
    "soda can",
    "juice bottle",
    "cup",
    "bowl",
    "pen",
    "marker",
    "notebook",
    "white case",
    "bottle cap",
    "gripper",
]

QUERY = (
    "List every distinct object in this photo as a comma-separated list of "
    "short labels, like: red cube, yellow block, soda can, water bottle."
)


def parse_labels(text: str) -> list[str]:
    text = text.replace("\n", ",")
    parts = re.split(r"[,;/]| and ", text)
    labels: list[str] = []
    seen: set[str] = set()
    for raw in parts:
        label = re.sub(r"[^a-z0-9 \-]", "", raw.strip().lower())
        label = re.sub(r"\s+", " ", label).strip()
        if len(label) < 3 or label in seen:
            continue
        if label in {"object", "objects", "item", "items", "photo", "image", "table"}:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["x_min"], a["y_min"], a["x_max"], a["y_max"]
    bx0, by0, bx1, by1 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def nms(boxes: list[dict], thresh: float = 0.55) -> list[dict]:
    kept: list[dict] = []
    for box in sorted(boxes, key=lambda b: b["area"], reverse=True):
        if any(_iou(box, k) > thresh for k in kept):
            continue
        kept.append(box)
    return kept


def _region(obj: dict, label: str) -> dict | None:
    keys = {str(k).replace("-", "_"): v for k, v in obj.items()}
    try:
        x_min = float(keys["x_min"])
        y_min = float(keys["y_min"])
        x_max = float(keys["x_max"])
        y_max = float(keys["y_max"])
    except (KeyError, TypeError, ValueError):
        return None
    if x_max - x_min < 0.01 or y_max - y_min < 0.01:
        return None
    if x_max - x_min > 0.95 and y_max - y_min > 0.95:
        return None
    return {
        "label": label,
        "x_min": max(0.0, min(1.0, x_min)),
        "y_min": max(0.0, min(1.0, y_min)),
        "x_max": max(0.0, min(1.0, x_max)),
        "y_max": max(0.0, min(1.0, y_max)),
        "area": max(0.0, (x_max - x_min) * (y_max - y_min)),
    }


def list_labels(model, image: Image.Image) -> list[str]:
    try:
        answer = model.query(image, QUERY).get("answer", "")
    except Exception:
        answer = ""
    labels = parse_labels(str(answer))
    seen = set(labels)
    for seed in SEED_LABELS:
        if seed not in seen:
            labels.append(seed)
            seen.add(seed)
    return labels


def detect_objects(model, image: Image.Image, labels: list[str] | None = None) -> dict:
    labels = labels or list_labels(model, image)
    boxes: list[dict] = []
    for label in labels:
        try:
            found = model.detect(image, label).get("objects") or []
        except Exception:
            found = []
        for obj in found:
            region = _region(obj, label)
            if region:
                boxes.append(region)
    boxes = nms(boxes)
    return {"labels": labels, "objects": boxes}


def draw_boxes(bgr: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    palette = [
        (80, 160, 255),
        (0, 215, 255),
        (90, 200, 90),
        (70, 70, 230),
        (210, 160, 80),
        (200, 110, 220),
        (40, 200, 200),
        (180, 180, 40),
    ]
    for i, obj in enumerate(objects):
        color = palette[i % len(palette)]
        x0, y0 = int(obj["x_min"] * w), int(obj["y_min"] * h)
        x1, y1 = int(obj["x_max"] * w), int(obj["y_max"] * h)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        caption = obj["label"]
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y0 - th - 8)
        cv2.rectangle(out, (x0, ty), (x0 + tw + 8, ty + th + 8), color, -1)
        cv2.putText(
            out,
            caption,
            (x0 + 4, ty + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 16, 12),
            1,
            cv2.LINE_AA,
        )
    return out


def annotate_path(model, image_path: Path, out_dir: Path) -> dict:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    result = detect_objects(model, image)
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    boxed = draw_boxes(bgr, result["objects"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    boxed_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    cv2.imwrite(str(boxed_path), boxed)
    payload = {
        "source": str(image_path),
        "annotated": str(boxed_path),
        "count": len(result["objects"]),
        "labels_tried": result["labels"],
        "objects": [
            {k: v for k, v in obj.items() if k != "area"} for obj in result["objects"]
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
