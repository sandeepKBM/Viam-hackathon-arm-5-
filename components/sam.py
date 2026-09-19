"""Segment Anything on a Moondream crop so gripper yaw follows the object.

Uses the public SAM weights (`facebook/sam-vit-base`) — the same family as
Viam registry `viam:sam2-detector`. Runs locally so pick does not depend on
a SAM service on the arm farm. Set SAM_VISION to a robot vision name to try
that service first (not required).
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

SAM_ID = os.environ.get("SAM_MODEL", "facebook/sam-vit-base")


@lru_cache(maxsize=1)
def load_sam():
    import torch
    from transformers import SamModel, SamProcessor

    device = os.environ.get("SAM_DEVICE") or (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"loading SAM {SAM_ID} on {device}…", flush=True)
    processor = SamProcessor.from_pretrained(SAM_ID)
    model = SamModel.from_pretrained(SAM_ID)
    model.to(device)
    model.float()
    model.eval()
    print(f"SAM ready on {device}", flush=True)
    return processor, model, device


def _model_inputs(inputs, device):
    import torch

    tensors = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            tensors[key] = value
            continue
        if key in {"original_sizes", "reshaped_input_sizes"}:
            tensors[key] = value
            continue
        if value.is_floating_point():
            value = value.to(dtype=torch.float32)
        tensors[key] = value.to(device)
    return tensors


def _predict_masks(rgb: np.ndarray, points, labels, box=None):
    import torch

    processor, model, device = load_sam()
    h, w = rgb.shape[:2]
    image = Image.fromarray(rgb)
    kwargs = {
        "input_points": [[points]],
        "input_labels": [[labels]],
        "return_tensors": "pt",
    }
    if box is not None:
        kwargs["input_boxes"] = [[box]]
    inputs = processor(image, **kwargs)
    tensors = _model_inputs(inputs, device)
    with torch.no_grad():
        outputs = model(**tensors, multimask_output=True)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().float().cpu(),
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )[0]
    scores = outputs.iou_scores[0].detach().cpu().numpy().reshape(-1)
    masks_np = masks.detach().cpu().numpy()
    while masks_np.ndim > 3:
        masks_np = masks_np[0]
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    out = []
    for i, raw in enumerate(masks_np):
        mask = (raw > 0.5).astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        score = float(scores[i]) if i < scores.size else 0.0
        out.append((score, mask))
    return out


def _mask_rank(score: float, fill: float) -> float:
    rank = score
    if 0.08 <= fill <= 0.50:
        rank += 0.12
    if fill > 0.60:
        rank -= 0.20
    return rank


def _pick_mask(candidates, h: int, w: int) -> np.ndarray | None:
    ranked = []
    for score, mask in candidates:
        fill = float(mask.mean())
        if fill < 0.04 or fill > 0.72:
            continue
        ranked.append((_mask_rank(score, fill), mask))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    mask = ranked[0][1]
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN, kernel)
    if int(mask.sum()) < 32:
        return None
    return mask


def _best_mask(rgb: np.ndarray) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    points = [
        [cx, cy],
        [2.0, 2.0],
        [float(w - 3), 2.0],
        [2.0, float(h - 3)],
        [float(w - 3), float(h - 3)],
    ]
    labels = [1, 0, 0, 0, 0]
    chosen = _pick_mask(_predict_masks(rgb, points, labels), h, w)
    if chosen is not None:
        return chosen
    inset = max(2.0, 0.08 * min(h, w))
    box = [inset, inset, float(max(inset + 1, w - inset)), float(max(inset + 1, h - inset))]
    return _pick_mask(_predict_masks(rgb, [[cx, cy]], [1], box=box), h, w)


def refine_shape_with_sam(bgr: np.ndarray, shape):
    """Segment the crop, keep the full-frame mask, drop unsegmented pixels."""
    from components.shapes import box_mask, refresh_from_mask

    x, y, bw, bh = (int(v) for v in shape.box)
    pad = max(8, int(0.12 * max(bw, bh)))
    h, w = bgr.shape[:2]
    fallback = box_mask(h, w, shape.box)
    if shape.mask is None:
        shape.mask = fallback
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return shape
    crop = bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        crop_mask = _best_mask(rgb)
    except Exception as exc:
        print(f"  SAM failed on {shape.color}: {exc}", flush=True)
        return shape
    if crop_mask is None:
        return shape
    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = crop_mask
    refined = refresh_from_mask(shape, full)
    orig_area = max(1.0, float(bw * bh))
    sam_area = float(refined.box[2] * refined.box[3])
    if sam_area < 0.35 * orig_area and refined.aspect_ratio < 2.5:
        print(
            f"  SAM {shape.color} kept box region "
            f"(partial mask area={sam_area:.0f}/{orig_area:.0f})",
            flush=True,
        )
        shape.mask = fallback
        return shape
    print(
        f"  SAM {shape.color} ar={refined.aspect_ratio:.2f} "
        f"angle={refined.angle:.1f} px=({refined.cx},{refined.cy}) box={refined.box}",
        flush=True,
    )
    return refined
