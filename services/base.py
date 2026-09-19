"""Tiny shared plumbing for the voice-uq FastAPI services.

``make_service`` builds a FastAPI app with a ``/health`` route, matching
the pattern in ``moondream/server.py`` and ``voice/voice.py``.

``call_service`` is the client side: it resolves a service's base URL via
``services.config.service_url`` and does a plain HTTP request, the same
way ``components/moondream_client.py`` talks to the moondream server --
stdlib ``urllib`` only, no extra HTTP dependency (color-sort's
requirements.txt does not pin httpx or requests, so urllib keeps this
importable everywhere, including client-only environments).

FastAPI itself is imported lazily inside ``make_service`` so this module
stays importable (and ``call_service`` usable) even in an environment that
only has the client side installed, not FastAPI/uvicorn.
"""

from __future__ import annotations

import json as _json
import mimetypes
import uuid
from typing import Any, Mapping, Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import config


def make_service(name: str):
    """Create a FastAPI app for service ``name`` with a ``GET /health`` route.

    Kept intentionally tiny: each service module adds its own routes on
    top of the returned app, e.g.::

        app = make_service("perception")

        @app.post("/detect")
        async def detect(...): ...
    """
    from fastapi import FastAPI  # lazy: keep this module import-safe without FastAPI installed

    app = FastAPI(title=f"voice-uq-{name}")

    @app.get("/health")
    async def health() -> dict:
        host, port = config.service_addr(name)
        return {"ok": True, "service": name, "host": host, "port": port}

    return app


def _encode_multipart(
    fields: Optional[Mapping[str, Any]], files: Mapping[str, Any]
) -> tuple[bytes, str]:
    """Very small multipart/form-data encoder -- just enough for sending an
    image alongside a few form fields, without pulling in ``requests``."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    for key, value in (fields or {}).items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())

    for key, value in files.items():
        filename = key
        content_type = "application/octet-stream"
        if isinstance(value, tuple):
            if len(value) == 3:
                filename, data, content_type = value
            elif len(value) == 2:
                filename, data = value
            else:
                data = value[0]
        else:
            data = value
        if hasattr(data, "read"):
            data = data.read()
        if content_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(filename)
            content_type = guessed or content_type
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        parts.append(data if isinstance(data, (bytes, bytearray)) else str(data).encode())
        parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def call_service(
    name: str,
    path: str,
    *,
    method: str = "POST",
    json: Optional[dict] = None,
    files: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    timeout: float = 30.0,
) -> Any:
    """Call another service by name and return its parsed JSON response.

    ``name`` is looked up via ``services.config.service_url`` (so it obeys
    the same services.yaml / env-var precedence as everything else --
    including being pointed at the other Mac). ``path`` is joined onto
    that base URL. Pass ``json=`` for a JSON body, or ``files=`` (plus
    optional ``json=`` for accompanying form fields) to send a
    multipart/form-data upload, e.g. an image.
    """
    base = config.service_url(name).rstrip("/")
    url = f"{base}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params)}"

    headers: dict[str, str] = {}
    data: Optional[bytes] = None

    if files:
        data, content_type = _encode_multipart(json, files)
        headers["Content-Type"] = content_type
    elif json is not None:
        data = _json.dumps(json).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as resp:
            raw = resp.read()
    except URLError as exc:
        raise RuntimeError(f"{name} service is not reachable at {base}: {exc}") from exc

    if not raw:
        return None
    try:
        return _json.loads(raw.decode("utf-8"))
    except ValueError:
        return raw
