"""Keep Moondream loaded and ping it for classify + boxes.

  python moondream/server.py     # http://127.0.0.1:8768
  curl http://127.0.0.1:8768/api/health
  curl -X POST http://127.0.0.1:8768/api/detect \\
       -H 'Content-Type: application/json' \\
       -d '{"path":"collector/dataset/color/000001.png"}'
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect import annotate_path  # noqa: E402

load_dotenv(ROOT / ".env")

HOST = os.environ.get("MOONDREAM_HOST", "127.0.0.1")
PORT = int(os.environ.get("MOONDREAM_PORT", "8768"))
MODEL_ID = os.environ.get("MOONDREAM_MODEL", "moondream2")
OUT_DIR = Path(os.environ.get("MOONDREAM_OUT", str(ROOT / "collector" / "dataset" / "annotated")))
STATIC = HERE / "static"


class ModelHost:
    def __init__(self) -> None:
        self.model = None
        self.ready = False
        self.error = ""
        self.last: dict = {}


state = ModelHost()


def load_model():
    import moondream as md

    print(f"loading {MODEL_ID} once…", flush=True)
    state.model = md.photon(MODEL_ID)
    state.ready = True
    state.error = ""
    print(f"{MODEL_ID} ready", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_model()
    except Exception as exc:
        state.error = str(exc)
        state.ready = False
        print(f"moondream failed to load: {exc}", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": state.ready,
        "model": MODEL_ID,
        "error": state.error,
        "last_count": state.last.get("count"),
    }


class DetectIn(BaseModel):
    path: str = "collector/dataset/color/000001.png"


@app.post("/api/detect")
async def detect(body: DetectIn | None = None) -> JSONResponse:
    if not state.ready or state.model is None:
        return JSONResponse(
            {"ok": False, "error": state.error or "model is still loading"},
            status_code=503,
        )
    raw_path = (body.path if body else None) or "collector/dataset/color/000001.png"
    path = Path(raw_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        return JSONResponse({"ok": False, "error": f"missing image {path}"}, status_code=404)
    try:
        result = annotate_path(state.model, path, OUT_DIR)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    state.last = result
    return JSONResponse({"ok": True, **result})


@app.get("/api/last.png")
async def last_png():
    annotated = state.last.get("annotated")
    if not annotated or not Path(annotated).is_file():
        return JSONResponse({"ok": False, "error": "no annotated image yet"}, status_code=404)
    return FileResponse(annotated, media_type="image/png")


def main() -> None:
    import uvicorn

    print(f"Moondream host: http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
