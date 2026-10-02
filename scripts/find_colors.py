import json
import sys
from collections import Counter

import cv2

import boot
from boot import ROOT
from components.shapes import annotate_colors, find_block_colors

IMAGE = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "out" / "frame.png")


def main() -> None:
    bgr = cv2.imread(IMAGE)
    if bgr is None:
        raise SystemExit(f"could not read {IMAGE}")
    blocks = find_block_colors(bgr, colors=("red", "yellow"))
    counts = Counter(b.color for b in blocks)
    payload = {
        "counts": {"red": counts.get("red", 0), "yellow": counts.get("yellow", 0)},
        "blocks": [
            {
                "color": b.color,
                "bbox": {"x": b.box[0], "y": b.box[1], "w": b.box[2], "h": b.box[3]},
                "center_px": [b.cx, b.cy],
                "aspect": round(b.aspect_ratio, 2),
                "long_angle_deg": round(b.angle, 1),
            }
            for b in blocks
        ],
    }
    print(json.dumps(payload, indent=2))
    out_dir = ROOT / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "colors.png"
    cv2.imwrite(out_path, annotate_colors(bgr, blocks))
    print(f"annotated: {out_path}")


if __name__ == "__main__":
    main()
