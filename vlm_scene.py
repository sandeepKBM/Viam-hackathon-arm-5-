"""Offline Moondream VLM smoke test on a saved image (no robot needed).

    python capture_image.py                 # grab out/frame.png
    python vlm_scene.py                      # caption + list objects + detect them
    python vlm_scene.py out/frame.png "red block"     # detect a specific label
    python vlm_scene.py out/frame.png --ask "which block is on top?"

First run downloads Moondream weights (~4GB in fp16). Env: VLM_MODEL, VLM_REVISION,
VLM_DTYPE (float16 on this Turing GPU), VLM_LIST_PROMPT.
"""

import json
import os
import sys

import cv2

from components.vlm import annotate, get_vlm


def main() -> None:
    args = sys.argv[1:]
    image = args[0] if args and not args[0].startswith("--") else os.path.join("out", "frame.png")
    rest = [a for a in args if a != image]

    bgr = cv2.imread(image)
    if bgr is None:
        raise SystemExit(f"could not read {image} (run capture_image.py first?)")

    vlm = get_vlm()

    if "--ask" in rest:
        q = rest[rest.index("--ask") + 1]
        print(json.dumps({"question": q, "answer": vlm.query(bgr, q)}, indent=2))
        return

    explicit = [a for a in rest if not a.startswith("--")]

    print("caption:", vlm.caption(bgr))
    labels = explicit if explicit else vlm.list_objects(bgr)
    print("labels:", labels)

    shapes = vlm.detect_many(bgr, labels)
    payload = [
        {
            "label": s.label,
            "color": s.color,
            "bbox": {"x": s.box[0], "y": s.box[1], "w": s.box[2], "h": s.box[3]},
            "center_px": [s.cx, s.cy],
        }
        for s in shapes
    ]
    print(json.dumps({"count": len(payload), "objects": payload}, indent=2))

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", "vlm.png")
    cv2.imwrite(out_path, annotate(bgr, shapes))
    print(f"annotated: {out_path}")


if __name__ == "__main__":
    main()
