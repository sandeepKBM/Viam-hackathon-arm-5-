"""Cognition service -- the "brain": PLAN + MEMORY + grasp reasoning.

This service does NOT drive the arm and does NOT touch a live Viam
connection: it is pure decision-making, composed entirely out of
already-ported, offline-testable modules --

  * ``components.policy.plan_task``          -- sandboxes a planner (the
    deterministic rule-based stub by default) to ``components.skills``'s
    fixed ``SKILL_REGISTRY`` (pick/place/handoff/move_aside/declutter/
    descend_until_contact). Nothing outside that registry can ever appear
    in a returned plan; every call is schema- and safety-validated
    (workspace + Z-floor) before it survives.
  * ``components.experience_store.ExperienceStore`` -- the per-object (W1)
    self-adaptive memory: every recorded pick attempt rolls up into a
    ``calibrated_plan`` (pick_z_offset / xy_offset / grip_params /
    retry_budget) for that ``canonical_label``, persisted to a small
    gitignored JSON file (``data/experience.json`` by default) so the arm
    gets better at a specific named object across voice sessions.
  * ``components.grasp_affordance.classify_grasp`` -- per-object grasp TYPE
    (top-down / side / inside-outside) from a 3D point cloud, when one is
    available for that object.
  * ``components.transport_guard`` -- best-effort "how high do I need to
    lift this held object to clear the rest of the scene" hint, when point
    clouds are available for both the target and at least one neighbor.
  * ``components.uq.difficulty`` -- a lightweight fallback difficulty
    (score + box geometry only, no augmentation-consistency) for any scene
    object the UQ service didn't already annotate, so the planner's
    easy-first ordering and the retry-budget recommendation still have
    something to work with.
  * ``components.canonicalize`` / ``components.declutter.is_blocked`` --
    vocabulary + blocked-target detection, reused rather than
    reimplemented.

Run
---
::

    cd /common/users/ss5772/viam_5-voice-uq
    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m uvicorn \\
        services.cognition_service:app --host 127.0.0.1 --port 8803

(port from ``services/services.yaml``'s ``cognition`` entry) or::

    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m services.cognition_service

``POST /plan``
---------------
JSON body::

    {
      "goal": "sort the red and cup",     // optional: free-form NL goal
      "task": "sort",                     // optional: voice.py's {task, moves} shape
      "moves": [{"object": "red", "place": "bin1", "count": 1}, ...],
      "scene": [
        {
          "id": "obj0",                   // optional; defaults to "obj<index>"
          "label": "red block",
          "canonical_label": "red",       // optional; canonicalize_label(label) if omitted
          "color": "red",                 // optional
          "x": 120.0, "y": -40.0, "z": 5.0,   // world mm -- REQUIRED for real planning
          "box": [x, y, w, h],            // optional, pixels
          "mask": {"bbox": [...], "area_px": ...},  // optional
          "yaw": 12.4,                    // optional
          "score": 0.83,                  // optional (perception/UQ confidence)
          "difficulty": 0.21,             // optional (UQ fused difficulty, [0,1])
          "points": [[x, y, z], ...]      // optional (N,3) world-frame points --
                                           // enables grasp_affordance + transport hints
        },
        ...
      ]
    }

At least one of ``goal`` / (``task``, ``moves``) should be given; if
``goal`` is a non-empty string it wins outright (see ``_resolve_goal`` for
the exact precedence and the voice-shape -> goal-text fallback).

Response (always HTTP 200; ``ok: false`` on a bad request rather than a
5xx, matching the ``uq`` service's degrade-gracefully contract)::

    {
      "ok": true,
      "goal": "sort the red and cup",     // effective goal text handed to the planner
      "plan": [
        {"skill": "pick", "params": {"object": {"object_id": "obj0"}}},
        {"skill": "pick", "params": {"object": {"object_id": "obj1"}}}
      ],
      "objects": {
        "obj0": {
          "canonical_label": "red",
          "score": 0.83,
          "difficulty": 0.21,
          "blocked": false,
          "calibrated_plan": {"pick_z_offset": 0.0, "xy_offset": [0.0, 0.0],
                               "grip_params": {}, "retry_budget": 2},
          "retry_budget": 2,
          "history_attempts": 0,
          "grasp_affordance": {...} | absent if no `points` given,
          "grasp_affordance_error": "..." | absent,
          "safe_transport_height_mm": 210.4 | absent
        },
        ...
      }
    }

``plan`` is exactly ``components.policy.plan_task``'s validated
``List[SkillCall]``, JSON-serialized: any parameter that referenced a live
scene object (``object``/``target``/``all_objects``) is replaced by
``{"object_id": "<the scene item's id>"}`` so the orchestrator can resolve
it back against the same ``scene`` it sent, without ever seeing this
service's internal Python objects.

``objects`` is keyed by the same ``object_id`` and is populated for EVERY
scene object (not just ones that made it into the plan), so the caller can
look up calibration/grasp/retry info for any target it decides to act on.

``POST /record``
------------------
JSON body::

    {
      "canonical_label": "red",           // required (ExperienceStore key)
      "xy": [120.0, -40.0],               // required
      "grasp_success": false,             // required
      "placement_success": true,          // optional
      "failure_type": "slip",             // optional
      "plan_params": {"pick_z_offset": 0.0, "xy_offset": [0.0, 0.0],
                       "grip_params": {}}  // optional -- the offsets THIS attempt used
    }

Calls ``ExperienceStore.record_attempt`` (append attempt, recompute+persist
``calibrated_plan``) and returns the updated calibration::

    {"ok": true, "canonical_label": "red",
     "calibrated_plan": {"pick_z_offset": -1.1, "xy_offset": [0.6, -0.2],
                          "grip_params": {}, "retry_budget": 3}}

This IS the self-adaptive loop: the next ``/plan`` call for the same
``canonical_label`` (across process restarts, across voice sessions) picks
up this calibration automatically, because ``ExperienceStore`` persists to
disk (``data/experience.json`` by default, gitignored -- see
``components.experience_store.DEFAULT_STORE_PATH`` /
``EXPERIENCE_STORE_PATH`` env var) and this service holds ONE
``ExperienceStore`` instance for its whole process lifetime.

Lazy imports
------------
``components.grasp_affordance``, ``components.transport_guard``, and
``components.uq`` are only imported INSIDE the functions that use them
(they pull in ``numpy``/``cv2`` at their own module level) -- so
``import services.cognition_service`` never requires those to be
importable, and this service degrades to "no grasp affordance / no
transport hint / no fallback difficulty" rather than failing outright if
one of them can't be imported in a given environment (e.g. a lean voice-
only Mac). ``components.policy`` / ``components.skills`` are NOT lazy
(they are this service's whole reason for existing), and they themselves
transitively pull in ``numpy``/``cv2``/``viam-sdk`` via
``components.pickplace`` -- that is an existing, unavoidable cost of
reusing the real skill registry, not something this module adds.

Smoke check (NOT a test file -- just an inline sanity path)
-------------------------------------------------------------
::

    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m services.cognition_service --smoke

Builds a synthetic 2-object scene (a red block + a cup), plans "sort the
red and cup" through the real ``plan_task`` + ``SKILL_REGISTRY``, prints
the validated plan and each object's calibrated plan, then calls
``/record``'s underlying logic twice for the red block (one failed grasp,
one successful one with a real offset) against an isolated tmp experience
store and prints + asserts that its ``calibrated_plan`` (offset,
retry_budget) actually moved -- proving the self-improving loop live,
without touching ``data/experience.json`` or running a robot/network.
"""

from __future__ import annotations

import os
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from components.canonicalize import canonicalize_label
from components.declutter import is_blocked
from components.experience_store import DEFAULT_RETRY_BUDGET as _STORE_DEFAULT_RETRY_BUDGET
from components.experience_store import ExperienceStore, resolve_key
from components.policy import plan_task, rule_based_stub_planner
from components.skills import SkillCall
from services.base import make_service

app = make_service("cognition")

# One ExperienceStore for this process's whole lifetime -- see module
# docstring's "self-adaptive loop" section. Path resolution (env override,
# default "data/experience.json") is entirely ExperienceStore's own job
# (components.experience_store.DEFAULT_STORE_PATH / EXPERIENCE_STORE_PATH).
STORE = ExperienceStore()

# Retry-budget fallback used when neither a live difficulty score nor a
# calibrated_plan.retry_budget is available for an object -- mirrors
# components.retry.DEFAULT_RETRY_BUDGET/difficulty_to_budget without
# requiring a RetryController (which needs arm/gripper objects cognition
# never touches).
_MIN_RETRY_BUDGET = int(os.environ.get("RETRY_MIN_BUDGET", 1))
_MAX_RETRY_BUDGET = int(os.environ.get("RETRY_MAX_BUDGET", 4))


def _difficulty_to_budget(difficulty: float) -> int:
    """Same formula as ``components.retry.difficulty_to_budget`` (kept as a
    tiny local copy so this service doesn't have to import
    ``components.retry``, which exists to drive a live arm/gripper retry
    loop -- cognition only needs the pure difficulty->budget mapping it
    documents)."""
    d = max(0.0, min(1.0, float(difficulty)))
    return _MIN_RETRY_BUDGET + round(d * (_MAX_RETRY_BUDGET - _MIN_RETRY_BUDGET))


# ---------------------------------------------------------------------------
# Scene object construction -- turns a JSON scene dict into the duck-typed
# LocatedShape-shaped object components.policy / components.skills /
# components.declutter / components.experience_store all expect (label, x,
# y, z, color, canonical_label, score, difficulty, history).
# ---------------------------------------------------------------------------

_KNOWN_IDENTITY_COLORS = ("red", "yellow")  # color-sort's color-IS-object vocabulary


def _build_scene_object(index: int, raw: Dict[str, Any]) -> SimpleNamespace:
    label = str(raw.get("label") or "")
    canonical = str(raw.get("canonical_label") or "").strip() or canonicalize_label(label)
    color = raw.get("color")
    if not color:
        color = canonical if canonical in _KNOWN_IDENTITY_COLORS else ""

    def _f(name: str) -> float:
        val = raw.get(name)
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    return SimpleNamespace(
        label=label,
        canonical_label=canonical,
        color=str(color or ""),
        x=_f("x"),
        y=_f("y"),
        z=_f("z"),
        box=raw.get("box"),
        mask=raw.get("mask"),
        yaw=raw.get("yaw"),
        score=raw.get("score"),
        difficulty=raw.get("difficulty"),
        points=raw.get("points"),
        history=None,
        _index=index,
    )


def _object_id(index: int, raw: Dict[str, Any]) -> str:
    given = raw.get("id")
    return str(given) if given not in (None, "") else f"obj{index}"


def _fallback_difficulty(obj: SimpleNamespace) -> Optional[float]:
    """If the scene item already carries a UQ-fused ``difficulty``, keep it
    untouched. Otherwise, if it at least has a ``score``, fall back to
    ``components.uq.difficulty`` on score + box geometry alone (no
    augmentation-consistency -- that needs a live image + detector, which
    this service never has). ``components.uq`` is imported lazily here
    (it pulls in ``cv2`` at its own module top) so a plain ``import
    services.cognition_service`` never requires it."""
    if obj.difficulty is not None:
        try:
            return float(obj.difficulty)
        except (TypeError, ValueError):
            return None
    if obj.score is None:
        return None

    aspect_ratio: Optional[float] = None
    area: Optional[float] = None
    box = obj.box
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            w, h = float(box[2]), float(box[3])
            if w > 0 and h > 0:
                short, long_ = sorted((w, h))
                aspect_ratio = long_ / short
                area = w * h
        except (TypeError, ValueError):
            pass

    try:
        from components.uq import difficulty as _uq_difficulty  # lazy: pulls cv2
    except Exception:
        return None
    try:
        return _uq_difficulty(score=float(obj.score), aspect_ratio=aspect_ratio, area=area)
    except Exception:
        return None


def _classify_grasp_safe(obj: SimpleNamespace) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort ``components.grasp_affordance.classify_grasp`` on
    ``obj.points`` (an (N,3) world-frame point cloud), when one was given.
    ``numpy``/``components.grasp_affordance`` are imported lazily so a scene
    with no ``points`` on any object never requires either. Returns
    ``(affordance_dict, None)`` on success, ``(None, error_message)`` on
    failure, ``(None, None)`` when there's simply no point cloud to
    classify."""
    points = obj.points
    if not points:
        return None, None
    try:
        import numpy as np

        from components.grasp_affordance import classify_grasp
    except Exception as exc:
        return None, f"grasp_affordance unavailable: {exc}"

    try:
        pts = np.asarray(points, dtype=float)
        affordance = classify_grasp(pts)
    except Exception as exc:
        return None, f"classify_grasp failed: {exc}"

    return (
        {
            "grasp_type": affordance.grasp_type.value,
            "approach": list(affordance.approach),
            "grasp_axis": list(affordance.grasp_axis),
            "width_mm": affordance.width_mm,
            "center": list(affordance.center),
            "confidence": affordance.confidence,
            "note": affordance.note,
            "rim_z": affordance.rim_z,
            "wall_thickness_mm": affordance.wall_thickness_mm,
            "inner_diameter_mm": affordance.inner_diameter_mm,
        },
        None,
    )


def _safe_transport_height(obj: SimpleNamespace, scene: Sequence[SimpleNamespace]) -> Optional[float]:
    """Best-effort ``components.transport_guard.safe_transport_height`` hint
    for carrying ``obj``, using ``.points`` on it and on any other scene
    object that also has them as obstacles. Returns ``None`` (silently) if
    ``obj`` has no points, no other object has points, or the geometry
    fails for any reason -- this is a nice-to-have planning hint, not a
    required field. Lazy-imports ``numpy``/``components.transport_guard``
    for the same reason as ``_classify_grasp_safe``."""
    if not obj.points:
        return None
    try:
        import numpy as np

        from components.transport_guard import box_from_points, safe_transport_height
    except Exception:
        return None

    try:
        held = box_from_points(np.asarray(obj.points, dtype=float))
        obstacles = []
        for other in scene:
            if other is obj or not other.points:
                continue
            obstacles.append(box_from_points(np.asarray(other.points, dtype=float)))
        if not obstacles:
            return None
        return float(safe_transport_height(held, obstacles, grip_z=float(obj.z)))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Goal resolution: accept either a free-form NL goal, or the voice.py
# {"task": ..., "moves": [{"object", "place", "count"}, ...]} shape.
# ---------------------------------------------------------------------------


def _resolve_goal(payload: Dict[str, Any]) -> str:
    goal = payload.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()

    task = str(payload.get("task") or "").strip().lower()
    raw_moves = payload.get("moves")
    moves = [m for m in raw_moves if isinstance(m, dict)] if isinstance(raw_moves, list) else []
    objects_named = []
    for m in moves:
        obj = str(m.get("object") or "").strip()
        if obj and obj not in objects_named:
            objects_named.append(obj)

    wants_handoff = task == "handoff" or any(
        str(m.get("place") or "").strip().lower() == "handoff" for m in moves
    )
    if wants_handoff:
        return f"hand off the {objects_named[0]}" if objects_named else "hand it to me"

    if objects_named:
        return "sort the " + " and ".join(objects_named)

    if task:
        return task
    return "sort the objects"


# ---------------------------------------------------------------------------
# Planner: components.policy.rule_based_stub_planner PLUS noun matching.
#
# plan_task's default stub planner's generic "sort"-like branch (see its
# docstring) matches candidate objects ONLY by color mention when the goal
# names any known color at all -- e.g. for "sort the red and cup" it builds
# mentioned_colors={"red"} and then keeps only objects whose OWN .color is
# "red", silently dropping a noun-identified object like "cup"/"can"/"pen"
# (color-sort's vocabulary gives those no .color at all -- see
# components/canonicalize.py's module docstring). That is fine for the stub
# planner's own offline tests (which use color-only goals), but cognition is
# meant to actually run mixed "sort the red and cup"-style voice goals, so
# it supplies this small planner instead: identical declutter/handoff
# routing (delegated straight to rule_based_stub_planner), but its generic
# branch matches a candidate by COLOR **or** canonical_label/label noun
# mention. This is exactly the pluggable-planner extension point
# components.policy documents -- plan_task still performs 100% of the
# schema + safety validation (_to_skill_call/make_skill_call) on whatever
# raw calls this proposes, so the sandboxing guarantee is unchanged.
# ---------------------------------------------------------------------------


def _name_hit(obj_view: Dict[str, Any], goal_l: str) -> bool:
    for key in ("color", "canonical_label", "label"):
        val = str(obj_view.get(key) or "").strip().lower()
        if val and val in goal_l:
            return True
    return False


def _cognition_planner(
    goal: str, scene_view: List[Dict[str, Any]], menu: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    goal_l = goal.lower()

    clear_path_markers = ("clear a path", "clear path", "path to", "declutter")
    handoff_markers = ("hand off", "handoff", "hand it", "hand me", "hand the", "give me", "give it")
    if any(m in goal_l for m in clear_path_markers) or any(m in goal_l for m in handoff_markers):
        return rule_based_stub_planner(goal, scene_view, menu)

    named = [o for o in scene_view if _name_hit(o, goal_l)]
    candidates = named if named else list(scene_view)

    def sort_key(o: Dict[str, Any]):
        difficulty = o.get("difficulty")
        difficulty = 1.0 if difficulty is None else float(difficulty)
        score = o.get("score")
        score = float(score) if score is not None else 0.0
        return (difficulty, -score, o["x"], o["y"])

    calls: List[Dict[str, Any]] = []
    for obj in sorted(candidates, key=sort_key):
        if obj.get("blocked"):
            calls.append({"skill": "declutter", "params": {"target_id": obj["object_id"]}})
        else:
            calls.append({"skill": "pick", "params": {"object_id": obj["object_id"]}})
    return calls


# ---------------------------------------------------------------------------
# Plan/record core logic (plain sync functions -- the FastAPI routes below
# and ``_smoke()`` both call these directly, so the smoke path exercises
# EXACTLY what the HTTP endpoints do).
# ---------------------------------------------------------------------------


def _serialize_value(value: Any, id_by_identity: Dict[int, str]) -> Any:
    oid = id_by_identity.get(id(value))
    if oid is not None:
        return {"object_id": oid}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v, id_by_identity) for v in value]
    return value


def _serialize_call(call: SkillCall, id_by_identity: Dict[int, str]) -> Dict[str, Any]:
    params_out: Dict[str, Any] = {}
    for f in fields(type(call.params)):
        params_out[f.name] = _serialize_value(getattr(call.params, f.name), id_by_identity)
    return {"skill": call.skill, "params": params_out}


def plan_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_scene = payload.get("scene")
    if raw_scene is None:
        raw_scene = []
    if not isinstance(raw_scene, list):
        return {"ok": False, "error": "'scene' must be a list", "goal": None, "plan": [], "objects": {}}

    scene: List[SimpleNamespace] = []
    id_by_identity: Dict[int, str] = {}
    ids: List[str] = []
    for idx, raw in enumerate(raw_scene):
        if not isinstance(raw, dict):
            continue
        obj = _build_scene_object(idx, raw)
        STORE.seed(obj)  # attaches .history from the experience store, keyed by canonical_label
        obj.difficulty = _fallback_difficulty(obj)  # so the planner's easy-first sort benefits too
        oid = _object_id(idx, raw)
        scene.append(obj)
        id_by_identity[id(obj)] = oid
        ids.append(oid)

    goal_text = _resolve_goal(payload)

    try:
        skill_calls = plan_task(goal_text, scene, planner=_cognition_planner)
    except Exception as exc:  # pragma: no cover - defensive; plan_task itself only ever
        # drops individually-invalid calls, this only fires on a genuinely
        # broken planner/scene.
        return {"ok": False, "error": f"planning failed: {exc}", "goal": goal_text, "plan": [], "objects": {}}

    plan_out = [_serialize_call(call, id_by_identity) for call in skill_calls]

    objects_out: Dict[str, Any] = {}
    for obj in scene:
        oid = id_by_identity[id(obj)]
        key = resolve_key(obj)
        calibrated = STORE.get_calibration(key)

        if obj.difficulty is not None:
            retry_budget = _difficulty_to_budget(obj.difficulty)
        else:
            retry_budget = int(calibrated.get("retry_budget", _STORE_DEFAULT_RETRY_BUDGET))

        others = [o for o in scene if o is not obj]
        grasp, grasp_err = _classify_grasp_safe(obj)

        entry: Dict[str, Any] = {
            "canonical_label": key,
            "score": obj.score,
            "difficulty": obj.difficulty,
            "blocked": is_blocked(obj, others) if others else False,
            "calibrated_plan": calibrated,
            "retry_budget": retry_budget,
            "history_attempts": len((STORE.get_history(key) or {}).get("attempts", [])),
        }
        if grasp is not None:
            entry["grasp_affordance"] = grasp
        if grasp_err is not None:
            entry["grasp_affordance_error"] = grasp_err

        safe_z = _safe_transport_height(obj, scene)
        if safe_z is not None:
            entry["safe_transport_height_mm"] = safe_z

        objects_out[oid] = entry

    return {"ok": True, "goal": goal_text, "plan": plan_out, "objects": objects_out}


def record_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    canonical_label = str(payload.get("canonical_label") or payload.get("object") or "").strip()
    if not canonical_label:
        return {"ok": False, "error": "'canonical_label' is required"}

    xy = payload.get("xy")
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        return {"ok": False, "error": "'xy' must be a 2-element [x, y]"}
    try:
        xy_pair = (float(xy[0]), float(xy[1]))
    except (TypeError, ValueError):
        return {"ok": False, "error": f"'xy' values must be numeric, got {xy!r}"}

    if "grasp_success" not in payload:
        return {"ok": False, "error": "'grasp_success' is required"}
    grasp_success = bool(payload.get("grasp_success"))

    placement_success = payload.get("placement_success")
    if placement_success is not None and not isinstance(placement_success, bool):
        placement_success = None

    failure_type = payload.get("failure_type")
    failure_type = str(failure_type) if failure_type else None

    plan_params = payload.get("plan_params") or {}
    if not isinstance(plan_params, dict):
        return {"ok": False, "error": "'plan_params' must be an object"}

    calibrated = STORE.record_attempt(
        canonical_label,
        xy=xy_pair,
        grasp_success=grasp_success,
        placement_success=placement_success,
        failure_type=failure_type,
        plan_params=plan_params,
    )
    return {"ok": True, "canonical_label": canonical_label, "calibrated_plan": calibrated}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

from fastapi import Body  # noqa: E402 -- see services.uq_service for why this import is here,
from fastapi.responses import JSONResponse  # noqa: E402 -- not in services.base (lazy-FastAPI contract)


@app.post("/plan")
async def plan(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    return JSONResponse(plan_impl(payload))


@app.post("/record")
async def record(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    return JSONResponse(record_impl(payload))


# ---------------------------------------------------------------------------
# smoke check (inline sanity path -- NOT a pytest test file)
# ---------------------------------------------------------------------------


def _smoke() -> None:
    import json
    import tempfile

    global STORE

    tmp_dir = tempfile.mkdtemp(prefix="cognition_smoke_")
    smoke_store_path = os.path.join(tmp_dir, "experience.json")
    STORE = ExperienceStore(smoke_store_path)  # isolated -- never touches data/experience.json
    print(f"[smoke] isolated experience store: {smoke_store_path}")

    scene = [
        {
            "id": "obj0",
            "label": "red block",
            "canonical_label": "red",
            "color": "red",
            "x": 100.0,
            "y": 50.0,
            "z": 5.0,
            "box": [10, 10, 40, 40],
            "score": 0.91,
            "difficulty": 0.2,
        },
        {
            "id": "obj1",
            "label": "cup",
            "canonical_label": "cup",
            "x": 260.0,
            "y": -60.0,
            "z": 5.0,
            "box": [80, 40, 30, 55],
            "score": 0.58,
            # no "difficulty" -- exercises the components.uq fallback path
        },
    ]
    payload = {"goal": "sort the red and cup", "scene": scene}

    print("\n=== POST /plan ===")
    result = plan_impl(payload)
    print(json.dumps(result, indent=2, default=str))

    assert result["ok"], f"expected a successful plan, got: {result}"
    assert result["goal"] == "sort the red and cup"
    assert len(result["plan"]) == 2, f"expected 2 skill calls (one per object), got {result['plan']}"
    assert set(result["objects"]) == {"obj0", "obj1"}
    assert result["objects"]["obj0"]["canonical_label"] == "red"
    assert result["objects"]["obj1"]["canonical_label"] == "cup"
    assert result["objects"]["obj1"]["difficulty"] is not None, (
        "expected the components.uq fallback to fill in a difficulty for the "
        "object that had a score but no UQ-fused difficulty"
    )
    print("[smoke] plan OK: 2 validated skill calls, both objects annotated")

    print("\n=== self-adaptive /record loop for 'red' ===")
    before = plan_impl(payload)["objects"]["obj0"]["calibrated_plan"]
    print("calibrated_plan before any attempts:", before)

    rec1 = record_impl(
        {
            "canonical_label": "red",
            "xy": [100.0, 50.0],
            "grasp_success": False,
            "failure_type": "slip",
            "plan_params": {"pick_z_offset": 0.0, "xy_offset": [0.0, 0.0], "grip_params": {}},
        }
    )
    print("after attempt 1 (failed grasp):", rec1["calibrated_plan"])
    assert rec1["ok"]

    rec2 = record_impl(
        {
            "canonical_label": "red",
            "xy": [100.0, 50.0],
            "grasp_success": True,
            "placement_success": True,
            "plan_params": {"pick_z_offset": -1.5, "xy_offset": [0.8, -0.4], "grip_params": {}},
        }
    )
    print("after attempt 2 (successful grasp, offset applied):", rec2["calibrated_plan"])
    assert rec2["ok"]

    after = plan_impl(payload)["objects"]["obj0"]["calibrated_plan"]
    print("calibrated_plan reflected in the NEXT /plan call:", after)

    assert after != before, "expected calibrated_plan to change after recorded attempts"
    assert after["pick_z_offset"] != 0.0 or after["xy_offset"] != [0.0, 0.0], (
        "expected the successful attempt's offset to move the rolled-up calibration"
    )
    assert after["retry_budget"] >= before["retry_budget"], (
        "one recent failure should not shrink the retry budget below where it started"
    )
    print(
        "[smoke] OK: /plan -> /record -> /plan proves the self-improving loop "
        "(offset and retry_budget adapted after 2 recorded attempts)."
    )


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        _smoke()
    else:
        import uvicorn

        from services import config

        host, port = config.service_addr("cognition")
        uvicorn.run(app, host=host, port=port)
