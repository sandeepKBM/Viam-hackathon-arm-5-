"""Lightweight VLM (Moondream) as the "namer" for open-world detection.

Pipeline this enables:

    VLM lists what's on the table  ->  labels  ->  OWLv2 localizes each (components.zeroshot)
                                              \\->  or Moondream detect()/point() localizes directly

Moondream 2B (`vikhyatk/moondream2`) natively supports caption / query / detect /
point, so it can both *name* objects (to feed the zero-shot detector) and localize
them itself. Output of detect() is the same ``DetectedShape`` the OpenCV / OWLv2
paths produce, so depth deprojection + the pick pipeline are unchanged.

Runs on any device via the transformers backend, auto-detected:
- CUDA (this server's RTX 2080 Ti) -> float16 (Turing has no native bf16, and no
  FlashAttention-2, which Moondream doesn't require anyway).
- Apple Silicon (MPS) -> float32 (safest for MPS op coverage).
- Plain CPU (Intel Mac) -> float32. Works but slow; for a weak CPU set
  VLM_MODEL to a smaller model (e.g. Florence-2) or use the OWLv2 path.

Heavy deps are imported lazily. Install with:
    pip install -r requirements-vlm.txt
"""

import os
import re
from typing import List, Optional, Sequence

import cv2
import numpy as np

from components.shapes import (
    CAMERA_NAME,
    DetectedShape,
    LocatedShape,
    _color_depth_intrinsics,
    _depth_at,
    _pixel_to_world,
)

VLM_MODEL = os.environ.get("VLM_MODEL", "vikhyatk/moondream2")
VLM_REVISION = os.environ.get("VLM_REVISION", "2025-06-21")
VLM_DTYPE = os.environ.get("VLM_DTYPE", "auto")  # auto: fp16 on CUDA, fp32 on MPS/CPU
VLM_DEVICE = os.environ.get("VLM_DEVICE", "")  # override: cuda | mps | cpu
# Question used to enumerate pickable objects on the table.
VLM_LIST_PROMPT = os.environ.get(
    "VLM_LIST_PROMPT",
    "List each distinct object on the table as a short noun phrase, comma-separated. "
    "Include its color, e.g. 'red block, yellow block'.",
)

_KNOWN_COLORS = ("red", "yellow", "green", "blue", "orange", "purple", "white", "black")


def _color_from_label(label: str) -> str:
    low = label.lower()
    for c in _KNOWN_COLORS:
        if c in low:
            return c
    return ""


def _bgr_to_pil(bgr: np.ndarray):
    from PIL import Image  # noqa: WPS433

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _norm_box_to_shape(box: dict, w: int, h: int, label: str) -> DetectedShape:
    """Moondream detect() boxes are normalized [0,1]; scale to pixels."""
    x0 = int(box["x_min"] * w)
    y0 = int(box["y_min"] * h)
    x1 = int(box["x_max"] * w)
    y1 = int(box["y_max"] * h)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    short, long_ = sorted((bw, bh))
    return DetectedShape(
        label=label,
        cx=(x0 + x1) // 2,
        cy=(y0 + y1) // 2,
        area=float(bw * bh),
        vertices=4,
        aspect_ratio=long_ / max(short, 1e-6),
        box=(x0, y0, bw, bh),
        color=_color_from_label(label) or label,
    )


class MoondreamVLM:
    """Lazy-loaded Moondream wrapper: caption / query / list_objects / detect / point."""

    def __init__(
        self,
        model: str = VLM_MODEL,
        revision: str = VLM_REVISION,
        dtype: str = VLM_DTYPE,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model
        self.revision = revision
        self.dtype = dtype
        self._device = device or VLM_DEVICE or None
        self._model = None

    def _resolve(self):
        import torch  # noqa: WPS433

        if self._device:
            device = self._device
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if self.dtype in ("", "auto"):
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, self.dtype)
        return device, torch_dtype

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from transformers import AutoModelForCausalLM  # noqa: WPS433

        device, torch_dtype = self._resolve()
        print(f"[vlm] loading {self.model_id}@{self.revision} on {device} ({torch_dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self._model = model.to(device)
        return self._model

    # --- language ---
    def caption(self, bgr: np.ndarray, length: str = "normal") -> str:
        model = self._ensure_model()
        return model.caption(_bgr_to_pil(bgr), length=length)["caption"]

    def query(self, bgr: np.ndarray, question: str) -> str:
        model = self._ensure_model()
        return model.query(_bgr_to_pil(bgr), question)["answer"]

    def list_objects(
        self, bgr: np.ndarray, prompt: str = VLM_LIST_PROMPT
    ) -> List[str]:
        """Ask the VLM to enumerate objects; return de-duped label strings.

        These feed straight into the zero-shot detector as candidate_labels.
        """
        answer = self.query(bgr, prompt)
        parts = re.split(r"[,\n]", answer)
        labels: List[str] = []
        seen = set()
        for p in parts:
            lab = re.sub(r"^[\s\-\*\d\.\)]+", "", p).strip().rstrip(".").lower()
            if lab and lab not in seen and len(lab) <= 40:
                seen.add(lab)
                labels.append(lab)
        return labels

    # --- localization (Moondream native) ---
    def detect(self, bgr: np.ndarray, label: str) -> List[DetectedShape]:
        model = self._ensure_model()
        h, w = bgr.shape[:2]
        result = model.detect(_bgr_to_pil(bgr), label)
        return [_norm_box_to_shape(o, w, h, label) for o in result.get("objects", [])]

    def point(self, bgr: np.ndarray, label: str) -> List[tuple]:
        """Return pick points in pixel coords [(x, y), ...]."""
        model = self._ensure_model()
        h, w = bgr.shape[:2]
        result = model.point(_bgr_to_pil(bgr), label)
        return [(int(p["x"] * w), int(p["y"] * h)) for p in result.get("points", [])]

    def detect_many(
        self, bgr: np.ndarray, labels: Sequence[str]
    ) -> List[DetectedShape]:
        out: List[DetectedShape] = []
        for lab in labels:
            out.extend(self.detect(bgr, lab))
        out.sort(key=lambda s: s.area, reverse=True)
        return out


_VLM: Optional[MoondreamVLM] = None


def get_vlm() -> MoondreamVLM:
    global _VLM
    if _VLM is None:
        _VLM = MoondreamVLM()
    return _VLM


def annotate(bgr: np.ndarray, shapes: List[DetectedShape]) -> np.ndarray:
    out = bgr.copy()
    for s in shapes:
        x, y, w, h = s.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            out,
            s.label,
            (x, y - 8 if y > 20 else y + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


async def locate_objects_vlm(
    machine,
    camera_name: str = CAMERA_NAME,
    world_frame: str = "world",
    labels: Optional[Sequence[str]] = None,
    use_zeroshot: bool = False,
) -> List[LocatedShape]:
    """VLM-driven localization → world mm, reusing the shared depth helpers.

    labels=None  -> VLM enumerates objects itself.
    use_zeroshot -> localize with OWLv2 (components.zeroshot); else Moondream detect().
    """
    from viam.components.camera import Camera

    cam = Camera.from_robot(machine, camera_name)
    bgr, depth_mm, intr = await _color_depth_intrinsics(cam)

    vlm = get_vlm()
    names = list(labels) if labels is not None else vlm.list_objects(bgr)
    if not names:
        return []

    if use_zeroshot:
        from components.zeroshot import detect_objects

        objects = detect_objects(bgr, prompts=names)
    else:
        objects = vlm.detect_many(bgr, names)
    if not objects:
        return []

    table_depth = 0.0
    if depth_mm is not None:
        valid = depth_mm[depth_mm > 0]
        if valid.size:
            table_depth = float(np.median(valid))

    located: List[LocatedShape] = []
    for s in objects:
        z = _depth_at(depth_mm, s)
        if z <= 0:
            z = table_depth
        if z <= 0:
            print(f"  skip {s.label} px=({s.cx},{s.cy}): no depth")
            continue
        p = await _pixel_to_world(machine, camera_name, s.cx, s.cy, z, intr, world_frame)
        located.append(
            LocatedShape(label=s.label, x=p.x, y=p.y, z=p.z, shape=s, color=s.color)
        )
    return located
