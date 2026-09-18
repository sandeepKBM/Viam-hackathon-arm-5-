import asyncio
import os
from datetime import datetime, timezone

import cv2
from dotenv import load_dotenv
from viam.components.camera import Camera
from viam.services.vision import VisionClient

from components.connection import connect_machine
from components.shapes import CAMERA_NAME, _decode_color

OUT_DIR = os.environ.get("OUT_DIR", "out")


async def main() -> None:
    load_dotenv()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    machine = await connect_machine()
    try:
        camera_name = os.environ.get("CAMERA_NAME", CAMERA_NAME)
        color = None
        try:
            vision = VisionClient.from_robot(machine, "vision-segment")
            captured = await vision.capture_all_from_camera(
                camera_name, return_image=True, timeout=30
            )
            if captured.image is not None:
                color = _decode_color(captured.image)
        except Exception as exc:
            print(f"vision-segment capture skipped: {exc}")
        if color is None:
            cam = Camera.from_robot(machine, camera_name)
            images, _ = await cam.get_images(timeout=60)
            color_img = next(
                (
                    im
                    for im in images
                    if "dep" not in (im.mime_type or "").lower()
                    and "depth" not in (getattr(im, "name", "") or "").lower()
                ),
                images[0] if images else None,
            )
            if color_img is None:
                raise RuntimeError("camera returned no color image")
            color = _decode_color(color_img)
        color_path = os.path.join(OUT_DIR, f"frame_{stamp}.png")
        latest = os.path.join(OUT_DIR, "frame.png")
        cv2.imwrite(color_path, color)
        cv2.imwrite(latest, color)
        print(f"color: {color.shape[1]}x{color.shape[0]} -> {color_path}")
        print(f"latest: {latest}")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
