"""Resolve where each voice-uq service lives.

Reads ``services/services.yaml`` (path overridable via the ``SERVICES_CONFIG``
env var) and layers per-service environment overrides on top, so a single
service can be pinned to another machine (e.g. the other Mac over a
Thunderbolt bridge) without editing the yaml.

Precedence, highest first:
  1. ``VOICEUQ_<NAME>_URL``            (e.g. VOICEUQ_PERCEPTION_URL=http://169.254.10.1:8801)
  2. ``VOICEUQ_<NAME>_HOST`` / ``VOICEUQ_<NAME>_PORT`` (either or both)
  3. the matching entry in services.yaml
  4. the hard-coded localhost defaults below

Missing/unreadable yaml is not an error: it just means step 3 contributes
nothing, and everything falls back to defaults (optionally still overridden
by env vars).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Hard-coded fallback -- always valid, even with no services.yaml at all.
DEFAULTS: dict[str, dict[str, Any]] = {
    "perception": {"host": "127.0.0.1", "port": 8801},
    "uq": {"host": "127.0.0.1", "port": 8802},
    "cognition": {"host": "127.0.0.1", "port": 8803},
    "orchestrator": {"host": "127.0.0.1", "port": 8804},
}

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "services.yaml"

_FLOW_MAP_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*:\s*\{(.*)\}\s*$")


def _config_path() -> Path:
    override = os.environ.get("SERVICES_CONFIG")
    return Path(override) if override else _DEFAULT_CONFIG_PATH


def _coerce(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _parse_flow_mapping_yaml(text: str) -> dict[str, dict[str, Any]]:
    """Minimal fallback parser for our fixed ``name: {host: .., port: ..}``
    format, used when PyYAML isn't installed. Only handles top-level keys
    whose value is a single-line ``{...}`` flow mapping, which is all
    services.yaml ever needs."""
    result: dict[str, dict[str, Any]] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0]
        if not line.strip():
            continue
        match = _FLOW_MAP_RE.match(line)
        if not match:
            continue
        name, body = match.group(1), match.group(2)
        entry: dict[str, Any] = {}
        for part in body.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            key, _, val = part.partition(":")
            entry[key.strip()] = _coerce(val)
        result[name] = entry
    return result


def _load_yaml_text(text: str) -> dict[str, dict[str, Any]]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
        return {}
    except ImportError:
        return _parse_flow_mapping_yaml(text)


def _load_file_config() -> dict[str, dict[str, Any]]:
    path = _config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = _load_yaml_text(text)
    except Exception:
        return {}
    return {
        name: dict(entry)
        for name, entry in data.items()
        if isinstance(entry, dict)
    }


def _merged_config() -> dict[str, dict[str, Any]]:
    """Defaults, overlaid with whatever services.yaml (or its fallback
    parse) contains. Re-read on every call so SERVICES_CONFIG / env
    overrides set mid-process (e.g. in tests) are picked up."""
    cfg = {name: dict(entry) for name, entry in DEFAULTS.items()}
    for name, entry in _load_file_config().items():
        cfg.setdefault(name, {})
        cfg[name].update(entry)
    return cfg


def _env_prefix(name: str) -> str:
    return f"VOICEUQ_{name.upper()}_"


def service_addr(name: str) -> tuple[str, int]:
    """Return the (host, port) a service should be reached at."""
    prefix = _env_prefix(name)

    url_override = os.environ.get(prefix + "URL")
    if url_override:
        parsed = urlparse(url_override)
        if parsed.hostname:
            return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

    entry = _merged_config().get(name, {})
    host = entry.get("host", "127.0.0.1")
    port = entry.get("port", DEFAULTS.get(name, {}).get("port", 8000))

    host = os.environ.get(prefix + "HOST", host)
    port = os.environ.get(prefix + "PORT", port)
    return str(host), int(port)


def service_url(name: str) -> str:
    """Return the base URL ("http://host:port") for a service.

    If VOICEUQ_<NAME>_URL is set, it is returned verbatim (minus a
    trailing slash) so a non-default scheme/path can be used; otherwise
    it's built from ``service_addr``.
    """
    prefix = _env_prefix(name)
    url_override = os.environ.get(prefix + "URL")
    if url_override:
        return url_override.rstrip("/")
    host, port = service_addr(name)
    return f"http://{host}:{port}"
