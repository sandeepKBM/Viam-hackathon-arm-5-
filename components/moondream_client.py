"""Call the local Moondream host. Model stays loaded in moondream/server.py."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

MOONDREAM_URL = os.environ.get("MOONDREAM_URL", "http://127.0.0.1:8768")


def detect_objects_host(image_path: str | Path, labels: list[str] | None = None) -> dict:
    payload = {"path": str(Path(image_path).resolve())}
    if labels:
        payload["labels"] = labels
    req = urllib.request.Request(
        f"{MOONDREAM_URL.rstrip('/')}/api/detect",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # HTTPError subclasses URLError: the host is up but refused (e.g. 503
        # while the model is loading or failed to load). Surface its reason.
        try:
            detail = json.loads(exc.read().decode()).get("error") or ""
        except (ValueError, OSError):
            detail = ""
        raise RuntimeError(
            f"Moondream host at {MOONDREAM_URL} returned HTTP {exc.code}: "
            f"{detail or exc.reason}. Check {MOONDREAM_URL}/api/health"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Moondream host is not reachable at {MOONDREAM_URL}. "
            "Start it with: python moondream/server.py"
        ) from exc
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "Moondream detect failed")
    return data
