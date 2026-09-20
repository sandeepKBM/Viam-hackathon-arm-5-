"""Per-object experience store (W1).

A lightweight, gitignored JSON keystore that remembers how picks went for
each object the arm has encountered before, keyed by `canonical_label`
(components.shapes.LocatedShape.canonical_label, W2). Every attempt is
appended to that object's history; a small rolled-up `calibrated_plan` is
recomputed after each write so callers can bias the *next* attempt (pick
height offset, xy offset, grip params, retry budget) without having to
replay the whole history themselves.

Schema is templated on the dataclass -> asdict() -> JSON pattern used by
`ur5e_benchmark/benchmark/episode_schema.py`: plain dataclasses, serialized
with `dataclasses.asdict`, no custom encoder/decoder needed.

On-disk shape (`data/experience.json`):

```
{
  "<canonical_label>": {
    "attempts": [
      {
        "timestamp": "2026-09-18T12:34:56.789012+00:00",
        "xy": [123.4, -56.7],
        "grasp_success": true,
        "placement_success": true,
        "failure_type": null,
        "plan_params": {"pick_z_offset": 0.0, "xy_offset": [0.0, 0.0], "grip_params": {}}
      },
      ...
    ],
    "calibrated_plan": {
      "pick_z_offset": -1.2,
      "xy_offset": [0.6, -0.3],
      "grip_params": {},
      "retry_budget": 2
    }
  },
  ...
}
```

This module is pure I/O + arithmetic: no robot, no camera, no async. It is
safe to unit test offline against a tmp path (see tests/test_experience_store.py).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Location + tunables (same style as components/constants.py: env-overridable)
# ---------------------------------------------------------------------------

DEFAULT_STORE_PATH = Path(os.environ.get("EXPERIENCE_STORE_PATH", "data/experience.json"))

# Rolling window (most-recent-N attempts) used to derive the failure rate
# that feeds the retry-budget calibration.
RECENT_WINDOW = int(os.environ.get("EXPERIENCE_RECENT_WINDOW", 5))

# success-weighted mean: a successful attempt's plan_params count MUCH more
# than a failed one's when rolling up "what offset should we use next time",
# but failures aren't ignored outright (there's still signal in "closer but
# not quite" attempts).
SUCCESS_WEIGHT = float(os.environ.get("EXPERIENCE_SUCCESS_WEIGHT", 1.0))
FAILURE_WEIGHT = float(os.environ.get("EXPERIENCE_FAILURE_WEIGHT", 0.25))

MIN_RETRY_BUDGET = int(os.environ.get("EXPERIENCE_MIN_RETRY_BUDGET", 1))
MAX_RETRY_BUDGET = int(os.environ.get("EXPERIENCE_MAX_RETRY_BUDGET", 4))
DEFAULT_RETRY_BUDGET = int(os.environ.get("EXPERIENCE_DEFAULT_RETRY_BUDGET", 2))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class AttemptRecord:
    timestamp: str
    xy: List[float]
    grasp_success: bool
    placement_success: Optional[bool] = None
    failure_type: Optional[str] = None
    plan_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibratedPlan:
    pick_z_offset: float = 0.0
    xy_offset: List[float] = field(default_factory=lambda: [0.0, 0.0])
    grip_params: Dict[str, float] = field(default_factory=dict)
    retry_budget: int = DEFAULT_RETRY_BUDGET


def _attempt_success(attempt: Dict[str, Any]) -> bool:
    """An attempt counts as a full success if the grasp succeeded and
    placement wasn't recorded as a failure (unknown/None placement is
    treated as "don't penalize" since not every caller tracks placement)."""
    if not attempt.get("grasp_success"):
        return False
    placement = attempt.get("placement_success")
    return placement is not False


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ExperienceStore:
    """JSON-backed keystore of per-object attempt history + calibration.

    Robust to a missing or corrupt file: if the file doesn't exist, isn't
    valid JSON, or isn't shaped like `{label: {...}}`, the store starts
    fresh (empty) rather than raising.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_STORE_PATH
        self._data: Dict[str, Dict[str, Any]] = self._load()

    # -- persistence ---------------------------------------------------

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw = self.path.read_text()
        except (FileNotFoundError, OSError):
            return {}
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        # Best-effort validation: drop any entry that isn't shaped like a
        # per-object record rather than failing the whole load.
        cleaned: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict) and "attempts" in value and "calibrated_plan" in value:
                cleaned[key] = value
        return cleaned

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a tmp file in the same directory, then
        # rename over the target, so a crash mid-write can't corrupt the
        # store (the next _load() would otherwise see a truncated file).
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise

    # -- writes ----------------------------------------------------------

    def record_attempt(
        self,
        canonical_label: str,
        *,
        xy: Tuple[float, float],
        grasp_success: bool,
        placement_success: Optional[bool] = None,
        failure_type: Optional[str] = None,
        plan_params: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append one attempt for `canonical_label` and recompute + persist
        its calibrated_plan. Returns the updated calibrated_plan dict."""
        label = canonical_label or "unknown"
        record = self._data.setdefault(
            label, {"attempts": [], "calibrated_plan": asdict(CalibratedPlan())}
        )
        attempt = AttemptRecord(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            xy=[float(xy[0]), float(xy[1])],
            grasp_success=bool(grasp_success),
            placement_success=placement_success,
            failure_type=failure_type,
            plan_params=dict(plan_params or {}),
        )
        record["attempts"].append(asdict(attempt))
        record["calibrated_plan"] = self._recompute_calibration(record["attempts"])
        self._save()
        return record["calibrated_plan"]

    # -- calibration rollup ------------------------------------------------

    def _recompute_calibration(self, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple, documented aggregation:

        - pick_z_offset / xy_offset: success-weighted mean of the offsets
          that were *applied* for each attempt (plan_params["pick_z_offset"],
          plan_params["xy_offset"]). Successful attempts count
          SUCCESS_WEIGHT, failed ones FAILURE_WEIGHT (failures still carry
          a little signal -- "closer but not quite" -- so they aren't
          dropped outright, just discounted).
        - grip_params: same success-weighted mean, per numeric key, over
          whichever attempts recorded that key.
        - retry_budget: derived from the failure rate over the most recent
          RECENT_WINDOW attempts. More recent failures -> bigger budget
          (up to MAX_RETRY_BUDGET); a clean recent streak -> smaller budget
          (down to MIN_RETRY_BUDGET). This is a *prior*; components.retry
          treats a live per-object difficulty score as authoritative when
          one is available and only falls back to this value otherwise.
        """
        if not attempts:
            return asdict(CalibratedPlan())

        weights = [SUCCESS_WEIGHT if _attempt_success(a) else FAILURE_WEIGHT for a in attempts]

        z_values = [float(a.get("plan_params", {}).get("pick_z_offset", 0.0)) for a in attempts]
        pick_z_offset = _weighted_mean(z_values, weights)

        xy_values = [a.get("plan_params", {}).get("xy_offset", [0.0, 0.0]) for a in attempts]
        dx = _weighted_mean([float(v[0]) for v in xy_values], weights)
        dy = _weighted_mean([float(v[1]) for v in xy_values], weights)

        grip_keys: set = set()
        for a in attempts:
            grip_keys.update((a.get("plan_params", {}).get("grip_params") or {}).keys())
        grip_params: Dict[str, float] = {}
        for key in grip_keys:
            vals, wts = [], []
            for a, w in zip(attempts, weights):
                gp = a.get("plan_params", {}).get("grip_params") or {}
                if key in gp:
                    vals.append(float(gp[key]))
                    wts.append(w)
            if vals:
                grip_params[key] = _weighted_mean(vals, wts)

        recent = attempts[-RECENT_WINDOW:]
        failures = sum(1 for a in recent if not _attempt_success(a))
        failure_rate = failures / len(recent) if recent else 0.0
        span = MAX_RETRY_BUDGET - MIN_RETRY_BUDGET
        retry_budget = MIN_RETRY_BUDGET + round(failure_rate * span)
        retry_budget = max(MIN_RETRY_BUDGET, min(MAX_RETRY_BUDGET, retry_budget))

        return asdict(
            CalibratedPlan(
                pick_z_offset=pick_z_offset,
                xy_offset=[dx, dy],
                grip_params=grip_params,
                retry_budget=retry_budget,
            )
        )

    # -- reads -------------------------------------------------------------

    def get_history(self, canonical_label: str) -> Optional[Dict[str, Any]]:
        return self._data.get(canonical_label)

    def get_calibration(self, canonical_label: str) -> Dict[str, Any]:
        record = self._data.get(canonical_label)
        if record is None:
            return asdict(CalibratedPlan())
        return record["calibrated_plan"]

    def seed(self, located_shape: Any) -> Any:
        """Attach `.history` (the full {"attempts", "calibrated_plan"}
        record, or None if this object has never been seen) to a
        LocatedShape-like object, keyed by `resolve_key(located_shape)`.
        Mutates and returns `located_shape` for convenient chaining."""
        key = resolve_key(located_shape)
        located_shape.history = self.get_history(key)
        return located_shape


def resolve_key(obj: Any) -> str:
    """Best-effort store key for a LocatedShape-like object: prefer
    `canonical_label` (W2); fall back to "<color>:<label>" when it's empty
    or missing (e.g. canonicalization hasn't run / isn't wired up yet)."""
    canonical = (getattr(obj, "canonical_label", "") or "").strip()
    if canonical:
        return canonical
    color = (getattr(obj, "color", "") or "").strip()
    label = (getattr(obj, "label", "") or "").strip()
    if color and label:
        return f"{color}:{label}"
    return label or color or "unknown"
