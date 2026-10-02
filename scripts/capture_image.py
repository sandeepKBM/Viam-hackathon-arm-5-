import asyncio
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np
from viam.components.camera import Camera

import boot
from boot import ROOT
from components.connection import connect_machine
from components.constants import COLOR_BINS, PICK_OBJECTS
from components.pickplace import pick_order, tcp_pick_z
from components.safety import in_workspace
from components.shapes import (
    CAMERA_NAME,
    LocatedShape,
    _color_depth_intrinsics,
    _depth_at,
    _pixel_to_world,
    _split_color_depth,
    find_pick_objects,
)

OUT_DIR = os.environ.get("OUT_DIR", str(ROOT / "out"))


def _depth_colormap(depth_mm: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_mm, dtype=np.float32)
    valid = d[d > 0]
    if valid.size == 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    u8 = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    u8[d <= 0] = 0
    return cv2.applyColorMap((u8 * 255).astype(np.uint8), cv2.COLORMAP_JET)


async def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    machine = await connect_machine()
    try:
        camera_name = os.environ.get("CAMERA_NAME", CAMERA_NAME)
        cam = Camera.from_robot(machine, camera_name)
        bgr, depth_mm, intr = await _color_depth_intrinsics(cam)
        if depth_mm is None:
            images, _ = await cam.get_images(timeout=60)
            bgr2, depth = _split_color_depth(images)
            if bgr is None:
                bgr = bgr2
            if bgr is None:
                raise RuntimeError("camera returned no color image")
            depth_mm = np.asarray(depth) if depth is not None else None

        color_path = os.path.join(OUT_DIR, f"frame_{stamp}.png")
        latest = os.path.join(OUT_DIR, "frame.png")
        cv2.imwrite(color_path, bgr)
        cv2.imwrite(latest, bgr)
        print(f"color: {bgr.shape[1]}x{bgr.shape[0]} -> {color_path}")

        depth_stats = None
        if depth_mm is None:
            print("depth: none")
        else:
            valid = depth_mm[depth_mm > 0]
            depth_stats = {
                "width": int(depth_mm.shape[1]),
                "height": int(depth_mm.shape[0]),
                "valid_pct": round(100.0 * valid.size / depth_mm.size, 1) if depth_mm.size else 0,
                "min_mm": round(float(valid.min()), 1) if valid.size else None,
                "max_mm": round(float(valid.max()), 1) if valid.size else None,
                "median_mm": round(float(np.median(valid)), 1) if valid.size else None,
            }
            raw_path = os.path.join(OUT_DIR, "depth_raw.png")
            colorized = os.path.join(OUT_DIR, "depth_color.png")
            cv2.imwrite(raw_path, depth_mm.astype(np.uint16))
            cv2.imwrite(colorized, _depth_colormap(depth_mm))
            print(f"depth: {json.dumps(depth_stats)} -> {colorized}")

        table_depth = 0.0
        if depth_mm is not None:
            valid = depth_mm[depth_mm > 0]
            if valid.size:
                table_depth = float(np.median(valid))

        located = []
        for s in find_pick_objects(bgr, color_path, objects=PICK_OBJECTS):
            z_cam = _depth_at(depth_mm, s, table_depth=table_depth) if depth_mm is not None else 0.0
            if z_cam <= 0:
                z_cam = table_depth
            if z_cam <= 0:
                print(f"  skip {s.color} px=({s.cx},{s.cy}): no depth")
                continue
            p = await _pixel_to_world(
                machine, camera_name, s.cx, s.cy, z_cam, intr, "world"
            )
            located.append(
                LocatedShape(
                    label=s.label,
                    x=p.x,
                    y=p.y,
                    z=p.z,
                    shape=s,
                    color=s.color,
                    depth_mm=z_cam,
                )
            )

        blocks = [
            {
                "color": b.color,
                "bin": COLOR_BINS.get(b.color),
                "center_px": [b.shape.cx, b.shape.cy] if b.shape else None,
                "bbox": b.shape.box if b.shape else None,
                "depth_mm": round(b.depth_mm, 1),
                "world_xyz_mm": [round(b.x, 1), round(b.y, 1), round(b.z, 1)],
                "pick_z_mm": round(tcp_pick_z(b), 1),
                "in_workspace": in_workspace(b.x, b.y),
            }
            for b in pick_order(located)
        ]
        print(json.dumps({"count": len(blocks), "blocks": blocks}, indent=2))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
