"""Deterministic goal → object table. No pouring, no extra inference."""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from components.safety import in_workspace

GOALS = {
    "hydrate": {
        "candidates": ("bottle",),
        "place": "handoff",
        "count": 1,
        "label": "bottle",
        "missing": "I cannot see a safe bottle.",
    },
    "write": {
        "candidates": ("pen",),
        "place": "handoff",
        "count": 1,
        "label": "pen",
        "missing": "I cannot see a safe pen.",
    },
}


def is_safe(block: Any) -> bool:
    depth = float(getattr(block, "depth_mm", 0.0) or 0.0)
    x = float(getattr(block, "x", 0.0) or 0.0)
    y = float(getattr(block, "y", 0.0) or 0.0)
    return depth > 0 and in_workspace(x, y)


def _area(block: Any) -> float:
    shape = getattr(block, "shape", None)
    return float(getattr(shape, "area", 0.0) or 0.0)


def resolve_goal(goal: Optional[str], detections: Iterable[Any]) -> dict:
    """Pick one visible, safe candidate. p95 target is under 10 ms."""
    started = time.perf_counter()
    spec = GOALS.get(str(goal or "").strip().lower())
    if spec is None:
        return {
            "ok": False,
            "moves": [],
            "chosen": None,
            "say": "I am not sure what you want.",
            "resolution_ms": (time.perf_counter() - started) * 1000.0,
        }
    found = [
        item
        for item in detections
        if getattr(item, "color", "") in spec["candidates"] and is_safe(item)
    ]
    if not found:
        return {
            "ok": False,
            "moves": [],
            "chosen": None,
            "say": spec["missing"],
            "resolution_ms": (time.perf_counter() - started) * 1000.0,
        }
    chosen = max(found, key=_area)
    return {
        "ok": True,
        "moves": [
            {
                "object": chosen.color,
                "place": spec["place"],
                "count": spec["count"],
            }
        ],
        "chosen": chosen,
        "say": f"Handing you the {spec['label']}.",
        "resolution_ms": (time.perf_counter() - started) * 1000.0,
    }
