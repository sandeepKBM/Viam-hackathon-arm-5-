"""Dataset collector UI: live camera stream, save paired color + depth.

  python collector/collect.py    # http://127.0.0.1:8767
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from viam.components.camera import Camera

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.connection import connect_machine
from components.shapes import CAMERA_NAME, _split_color_depth

load_dotenv(ROOT / ".env")

STATIC = HERE / "static"
HOST = os.environ.get("COLLECT_HOST", "127.0.0.1")
PORT = int(os.environ.get("COLLECT_PORT", "8767"))
DATASET = Path(os.environ.get("DATASET_DIR", str(HERE / "dataset")))
if not DATASET.is_absolute():
    DATASET = (ROOT / DATASET).resolve()


class Collector:
    def __init__(self) -> None:
        self.machine = None
        self.cam = None
        self.lock = asyncio.Lock()
        self.bgr: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.error = ""
        self.connected = False


state = Collector()


def dataset_dirs() -> dict[str, Path]:
    folders = {
        "color": DATASET / "color",
        "depth": DATASET / "depth",
        "depth_viz": DATASET / "depth_viz",
        "meta": DATASET / "meta",
    }
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return folders


def next_id() -> str:
    color_dir = dataset_dirs()["color"]
    ids = [
        int(p.stem)
        for p in color_dir.glob("*.png")
        if p.stem.isdigit()
    ]
    return f"{(max(ids) + 1) if ids else 1:06d}"


def sample_count() -> int:
    return len(list((DATASET / "color").glob("*.png"))) if (DATASET / "color").exists() else 0


def depth_colormap(depth_mm: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_mm, dtype=np.float32)
    valid = d[d > 0]
    if valid.size == 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    u8 = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    u8[d <= 0] = 0
    return cv2.applyColorMap((u8 * 255).astype(np.uint8), cv2.COLORMAP_JET)


def encode_jpeg(bgr: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("could not encode jpeg")
    return buf.tobytes()


async def grab_pair():
    images, _ = await state.cam.get_images(timeout=60)
    return _split_color_depth(images)


async def ensure_camera() -> None:
    if state.cam is not None:
        return
    camera_name = os.environ.get("CAMERA_NAME", CAMERA_NAME)
    state.machine = await connect_machine()
    state.cam = Camera.from_robot(state.machine, camera_name)
    state.connected = True
    state.error = ""


async def preview_loop() -> None:
    while True:
        try:
            await ensure_camera()
            async with state.lock:
                color, depth = await grab_pair()
            if color is None:
                raise RuntimeError("camera returned no color image")
            state.bgr = color
            state.depth = np.asarray(depth) if depth is not None else None
            state.jpeg = encode_jpeg(color)
            state.error = ""
            state.connected = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.error = str(exc)
            state.connected = state.cam is not None
        await asyncio.sleep(0.08)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_dotenv(ROOT / ".env")
    dataset_dirs()
    task = asyncio.create_task(preview_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if state.machine is not None:
            await state.machine.close()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
async def status() -> dict:
    return {
        "connected": state.connected,
        "error": state.error,
        "has_frame": state.bgr is not None,
        "has_depth": state.depth is not None,
        "count": sample_count(),
        "dataset": str(DATASET.resolve()),
    }


@app.get("/api/preview.jpg")
async def preview() -> Response:
    if not state.jpeg:
        return Response(status_code=503, content=b"no frame yet")
    return Response(
        content=state.jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    async def frames():
        while True:
            if state.jpeg:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + state.jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.08)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/capture")
async def capture() -> dict:
    async with state.lock:
        if state.cam is not None:
            try:
                color, depth = await grab_pair()
                if color is not None:
                    state.bgr = color
                    state.depth = np.asarray(depth) if depth is not None else None
                    state.jpeg = encode_jpeg(color)
            except Exception as exc:
                if state.bgr is None:
                    return {"ok": False, "error": str(exc)}

    bgr = state.bgr
    depth = state.depth
    if bgr is None:
        return {"ok": False, "error": state.error or "no color frame yet"}

    folders = dataset_dirs()
    sample_id = next_id()
    stamp = datetime.now(timezone.utc).isoformat()
    color_rel = f"color/{sample_id}.png"
    depth_rel = f"depth/{sample_id}.png"
    viz_rel = f"depth_viz/{sample_id}.png"

    cv2.imwrite(str(folders["color"] / f"{sample_id}.png"), bgr)

    depth_stats = None
    if depth is not None:
        raw = np.asarray(depth)
        if raw.dtype != np.uint16:
            raw = np.clip(raw, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(folders["depth"] / f"{sample_id}.png"), raw)
        cv2.imwrite(str(folders["depth_viz"] / f"{sample_id}.png"), depth_colormap(raw))
        valid = raw[raw > 0]
        depth_stats = {
            "encoding": "uint16_mm",
            "width": int(raw.shape[1]),
            "height": int(raw.shape[0]),
            "valid_pct": round(100.0 * valid.size / raw.size, 1) if raw.size else 0,
            "min_mm": int(valid.min()) if valid.size else None,
            "max_mm": int(valid.max()) if valid.size else None,
            "median_mm": int(np.median(valid)) if valid.size else None,
        }
    else:
        depth_rel = None
        viz_rel = None

    meta = {
        "id": sample_id,
        "captured_at": stamp,
        "camera": os.environ.get("CAMERA_NAME", CAMERA_NAME),
        "color": {
            "path": color_rel,
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
        },
        "depth": {"path": depth_rel, **(depth_stats or {"missing": True})},
        "depth_viz": viz_rel,
    }
    (folders["meta"] / f"{sample_id}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"saved dataset {sample_id}: {color_rel} {depth_rel or 'no-depth'}")
    return {"ok": True, "count": sample_count(), **meta}


@app.get("/dataset/{kind}/{name}")
async def dataset_file(kind: str, name: str):
    if kind not in {"color", "depth", "depth_viz", "meta"}:
        return Response(status_code=404)
    path = (DATASET / kind / name).resolve()
    if not str(path).startswith(str(DATASET.resolve())) or not path.is_file():
        return Response(status_code=404)
    media = "application/json" if kind == "meta" else "image/png"
    return FileResponse(path, media_type=media)


def main() -> None:
    load_dotenv(ROOT / ".env")
    import uvicorn

    print(f"Dataset collector: http://{HOST}:{PORT}")
    print(f"Saving into {DATASET.resolve()}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
