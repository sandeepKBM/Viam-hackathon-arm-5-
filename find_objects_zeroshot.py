"""Offline zero-shot object detection on a saved image (no robot needed).

Mirrors find_colors.py but uses a HuggingFace open-vocabulary detector, so you
can try prompts before wiring it into the arm.

    python capture_image.py                        # grab out/frame.png from cam
    python find_objects_zeroshot.py                # default prompts on out/frame.png
    python find_objects_zeroshot.py out/frame.png "red block" "yellow block" "screwdriver"

Env: ZS_MODEL, ZS_THRESHOLD, ZS_PROMPTS (see components/zeroshot.py).
"""

import json
import os
import sys
from collections import Counter

import cv2

from components.zeroshot import ZS_PROMPTS, annotate, detect_objects

IMAGE = sys.argv[1] if len(sys.argv) > 1 else os.path.join("out", "frame.png")
PROMPTS = sys.argv[2:] if len(sys.argv) > 2 else list(ZS_PROMPTS)


def main() -> None:
    bgr = cv2.imread(IMAGE)
    if bgr is None:
        raise SystemExit(f"could not read {IMAGE} (run capture_image.py first?)")

    print(f"model prompts: {PROMPTS}")
    shapes = detect_objects(bgr, prompts=PROMPTS)
    counts = Counter(s.label for s in shapes)
    payload = {
        "counts": dict(counts),
        "objects": [
            {
                "label": s.label,
                "color": s.color,
                "bbox": {"x": s.box[0], "y": s.box[1], "w": s.box[2], "h": s.box[3]},
                "center_px": [s.cx, s.cy],
            }
            for s in shapes
        ],
    }
    print(json.dumps(payload, indent=2))

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", "zeroshot.png")
    cv2.imwrite(out_path, annotate(bgr, shapes))
    print(f"annotated: {out_path}")


if __name__ == "__main__":
    main()
