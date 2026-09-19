"""Orchestrator service: the real end-to-end loop that ties

    voice-intent -> perception -> UQ -> cognition -> execution

together over HTTP, composing the other three voice-uq services
(``services/perception_service.py``, ``services/uq_service.py``, and the
(not-yet-built-here) ``services/cognition_service.py``) via
``services.base.call_service`` -- this module never imports those services'
code directly.

``POST /run`` takes one voice-intent in color-sort's own shape, the exact
dict ``components.voice.map_task`` / ``components/voice.py``'s ``run_task``
already produce for the "sort" task::

    {"task": "sort", "moves": [{"object": "yellow", "place": "bin2",
                                 "count": 1}, ...], "say": "..."}

and a frame to look at (``frame_path`` -- a saved image on disk, the normal
way to exercise this offline/in tests -- or ``image_base64``; a live camera
frame is only ever used when ``VIAM_ALLOW_LIVE`` is set, see below), then:

  1. resolves a frame (saved file, base64, or -- ONLY if VIAM_ALLOW_LIVE --
     the connected machine's camera);
  2. ``call_service("perception", "/detect", ...)``            -> enriched detections;
  3. ``call_service("uq", "/score", ...)``                     -> + per-detection difficulty;
  4. builds a scene (``components.shapes.LocatedShape`` per detection) and
     ``call_service("cognition", "/plan", {goal, scene})``     -> a validated
     skill sequence (falling back to ``components.policy``'s own offline
     ``rule_based_stub_planner`` if cognition is unreachable or answers with
     something that doesn't parse -- see ``_plan`` below);
  5. **only if ``VIAM_ALLOW_LIVE`` is set**: connects to the real machine
     (via ``components.connection.connect_machine``, which itself refuses to
     connect unless that same env var is set -- belt and suspenders),
     re-grounds the plan's objects in a real, depth-calibrated scene via
     ``components.vision.VisionComponent.locate_blocks``, applies each
     move's requested ``count``/``place`` (routing to the ``handoff`` skill
     when asked), and runs ``components.pipeline.execute_plan_with_memory``
     against color-sort's real ``PickPlace``/``RetryController``. With
     ``VIAM_ALLOW_LIVE`` unset (the default), this step is skipped entirely
     -- the robot is NEVER connected -- and ``/run`` returns the plan alone
     (a dry / plan-only result);
  6. after a live run, best-effort ``call_service("cognition", "/record",
     outcome)`` per step, to close the self-adaptive loop (in addition to
     the experience-store calibration ``execute_plan_with_memory`` already
     persists locally on every attempt, live or not needed).

Cognition's ``/plan`` contract (as this service composes it, so it works
standalone even before ``services/cognition_service.py`` exists -- it just
always falls back to the local planner and reports the outage)::

    request  : {"goal": "<str>", "scene": [<scene-view dict per object, the
                same shape components.policy._scene_view builds>, ...]}
    response : {"ok": true, "plan": [{"skill": "...", "params": {...}}, ...],
                "grasp_affordance": {<object_id>: {...}} | null}

``plan`` entries are raw planner calls referencing objects by the scene
view's ``object_id`` (``"obj0"``, ``"obj1"``, ...) -- exactly the shape
``components.policy.plan_task``'s pluggable ``Planner`` interface expects,
so cognition's answer is run through the SAME validation
(``components.policy.plan_task`` -> ``components.skills.make_skill_call``)
as the offline stub: every call is schema- and safety-checked before it can
appear in the plan this service returns or executes.

Degrade gracefully -- this endpoint always answers (never a bare 5xx for a
downstream outage):
  - perception unreachable  -> ``detections: []``, noted in ``errors.perception``.
  - uq unreachable          -> detections keep their perception score, no
                                ``difficulty``; noted in ``errors.uq``.
  - cognition unreachable / bad response -> plan comes from the local
                                ``rule_based_stub_planner`` instead;
                                ``planning_source: "local_fallback"``, noted
                                in ``errors.cognition``.

Per-object CALIBRATED PLAN comes from color-sort's own experience store
(``components/experience_store.py``) -- it's local and always available,
independent of cognition being reachable, exactly what
``components/pipeline.py``'s ``_run_pick`` already relies on for its
``RetryController`` bias.

Lazy imports: everything that needs cv2/numpy/viam/torch/etc. is imported
inside the functions that use it, so ``import services.orchestrator_service``
succeeds anywhere FastAPI is installed, with no robot/vision/model deps.

Run it::

    cd <worktree>
    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m uvicorn \\
        services.orchestrator_service:app --host 127.0.0.1 --port 8804

Smoke-test the DRY path (no test file -- an inline sanity check, same
pattern as ``services/uq_service.py --smoke``): mocks perception/uq/
cognition's HTTP responses (no network) and a canary on
``components.connection.connect_machine`` that raises if it's ever called,
then asserts ``VIAM_ALLOW_LIVE`` unset -> execution skipped, robot never
touched::

    PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m services.orchestrator_service --smoke
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.base import call_service, make_service

app = make_service("orchestrator")


# ---------------------------------------------------------------------------
# Live gate
# ---------------------------------------------------------------------------


def _live_allowed() -> bool:
    """Mirror ``components.connection``'s own credential-hygiene gate for
    this service's OWN control flow (whether to even attempt a connection,
    grab a live camera frame, or execute). ``components.connection.connect_machine``
    enforces the identical check independently before it will ever open a
    socket, so this is defense-in-depth, not the only gate."""
    return os.environ.get("VIAM_ALLOW_LIVE", "").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Frame acquisition
# ---------------------------------------------------------------------------


async def _resolve_frame(payload: Dict[str, Any], machine: Optional[Any]) -> Tuple[bytes, str]:
    """Return (image_bytes, source_description). Never touches the network/
    robot unless `machine` is already a connected client (only ever passed
    in when VIAM_ALLOW_LIVE is set -- see `_handle_run`)."""
    image_base64 = payload.get("image_base64")
    if image_base64:
        raw = image_base64.split(",", 1)[-1] if "," in image_base64 else image_base64
        try:
            return base64.b64decode(raw, validate=False), "image_base64"
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid image_base64: {exc}") from exc

    frame_path = payload.get("frame_path")
    if frame_path:
        data = Path(frame_path).read_bytes()
        if not data:
            raise ValueError(f"frame_path {frame_path!r} is empty")
        return data, f"frame_path:{frame_path}"

    if machine is not None:
        data = await _live_camera_frame_bytes(machine, payload.get("camera_name"))
        return data, "live_camera"

    # No frame given and not live: don't fail the whole request -- hand back
    # a tiny synthetic placeholder frame so the perception/uq/cognition
    # wiring can still be exercised end-to-end (detections will simply be
    # empty/trivial on it). Real testing should pass frame_path/image_base64.
    return _synthetic_placeholder_frame(), "synthetic_placeholder (no frame_path/image_base64 given)"


async def _live_camera_frame_bytes(machine: Any, camera_name: Optional[str]) -> bytes:
    from viam.components.camera import Camera

    name = camera_name or os.environ.get("CAMERA_NAME", "cam")
    cam = Camera.from_robot(machine, name)
    images, _ = await cam.get_images(timeout=60)
    for im in images:
        mime = (im.mime_type or "").lower()
        img_name = (getattr(im, "name", "") or "").lower()
        if "dep" in mime or "depth" in img_name:
            continue
        if im.data:
            return bytes(im.data)
    raise RuntimeError(f"camera {name!r} returned no decodable color image")


def _synthetic_placeholder_frame() -> bytes:
    import cv2
    import numpy as np

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    if not ok:  # pragma: no cover - cv2 encoding a blank frame never fails
        raise RuntimeError("could not encode synthetic placeholder frame")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Perception + UQ composition
# ---------------------------------------------------------------------------


def _labels_from_moves(moves: List[Dict[str, Any]]) -> Optional[str]:
    labels = [str(mv.get("object")) for mv in moves if mv.get("object")]
    return ",".join(dict.fromkeys(labels)) or None  # de-dup, preserve order


def _call_perception(image_bytes: bytes, labels: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    fields: Dict[str, Any] = {}
    if labels:
        fields["labels"] = labels
    try:
        resp = call_service(
            "perception",
            "/detect",
            files={"image": ("frame.png", image_bytes, "image/png")},
            json=fields or None,
            timeout=60.0,
        )
    except Exception as exc:
        return [], str(exc)
    detections = (resp or {}).get("detections") if isinstance(resp, dict) else None
    return list(detections or []), None


def _call_uq(image_bytes: bytes, detections: List[Dict[str, Any]], n: Optional[int]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    fields: Dict[str, Any] = {"detections": json.dumps(detections)}
    if n is not None:
        fields["n"] = n
    try:
        resp = call_service(
            "uq",
            "/score",
            files={"image": ("frame.png", image_bytes, "image/png")},
            json=fields,
            timeout=60.0,
        )
    except Exception as exc:
        return detections, None, str(exc)
    if isinstance(resp, dict) and resp.get("ok"):
        return list(resp.get("detections") or detections), resp, None
    error = None if not isinstance(resp, dict) else resp.get("error")
    return detections, resp if isinstance(resp, dict) else None, error


# ---------------------------------------------------------------------------
# Scene building (perception+uq detections -> components.shapes.LocatedShape)
# ---------------------------------------------------------------------------


def _detections_to_scene(detections: List[Dict[str, Any]]) -> List[Any]:
    from components.shapes import LocatedShape

    scene = []
    for det in detections:
        box = list(det.get("box") or [0, 0, 0, 0])
        box = (box + [0, 0, 0, 0])[:4]
        x, y, w, h = box
        cx = det.get("cx", x + w / 2.0)
        cy = det.get("cy", y + h / 2.0)
        canonical = (det.get("canonical_label") or "").strip()
        scene.append(
            LocatedShape(
                label=det.get("label") or canonical or "object",
                # NOTE: x/y here are PIXEL coordinates from the 2D detector,
                # not calibrated world millimeters -- fine for planning/
                # reporting (this is all `plan_task` needs), but NOT used
                # directly to drive the arm. Live execution re-grounds each
                # planned object in a real depth-calibrated scene from
                # components.vision.VisionComponent.locate_blocks before
                # anything moves -- see `_execute_live`.
                x=float(cx),
                y=float(cy),
                z=0.0,
                color=canonical,
                canonical_label=canonical,
                score=det.get("score"),
                difficulty=det.get("difficulty"),
                u=float(cx),
                v=float(cy),
            )
        )
    return scene


def _goal_text(task: Optional[str], moves: List[Dict[str, Any]], say: Optional[str]) -> str:
    parts = []
    for mv in moves or []:
        obj = mv.get("object") or mv.get("color") or ""
        if not obj:
            continue
        place = (mv.get("place") or "").strip().lower()
        count = mv.get("count", 1)
        qty = "all" if count is None else str(count)
        if place == "handoff":
            parts.append(f"hand me the {obj}")
        elif place == "dropoff":
            parts.append(f"drop off the {obj}")
        else:
            parts.append(f"sort {qty} {obj} to {place or 'its bin'}")
    if parts:
        return "; ".join(parts)
    return say or task or "sort the blocks"


# ---------------------------------------------------------------------------
# Planning: cognition service, falling back to the offline stub planner
#
# services/cognition_service.py's real ``POST /plan`` contract (see that
# module's own docstring): request ``{"goal": ..., "scene": [<raw dict per
# object: id?, label, canonical_label?, color?, x, y, z, score?,
# difficulty?, box?, ...>, ...]}``; response ``{"ok", "goal", "plan": [the
# validated components.policy.plan_task SkillCall list, JSON-serialized --
# every object/target/all_objects param replaced by {"object_id": "<id>"}],
# "objects": {<object_id>: {canonical_label, score, difficulty, blocked,
# calibrated_plan, retry_budget, history_attempts, grasp_affordance?,
# grasp_affordance_error?, safe_transport_height_mm?}}}``. This service
# never re-implements cognition's planning/validation logic: it resolves
# cognition's ``{"object_id": ...}`` refs back onto ITS OWN `scene` objects
# (same list, same order -> same "obj<i>" ids) via `components.policy.
# make_skill_call`, which re-runs the identical schema+safety validation as
# a defense-in-depth check before anything ends up in the returned/executed
# plan.
# ---------------------------------------------------------------------------


def _scene_to_cognition_payload(scene: List[Any]) -> List[Dict[str, Any]]:
    """Raw JSON scene dicts, one per `scene` object IN THE SAME ORDER, with
    an explicit ``id`` ("obj0", "obj1", ...) so this service's own `by_id`
    lookup (built the same way) is guaranteed to agree with however
    cognition numbers them -- see services/cognition_service.py's
    `_object_id` (defaults to "obj<index>" too, but we don't rely on that
    default matching by coincidence)."""
    payload = []
    for i, obj in enumerate(scene):
        payload.append(
            {
                "id": f"obj{i}",
                "label": getattr(obj, "label", "") or "",
                "canonical_label": getattr(obj, "canonical_label", "") or "",
                "color": getattr(obj, "color", "") or "",
                "x": float(getattr(obj, "x", 0.0) or 0.0),
                "y": float(getattr(obj, "y", 0.0) or 0.0),
                "z": float(getattr(obj, "z", 0.0) or 0.0),
                "score": getattr(obj, "score", None),
                "difficulty": getattr(obj, "difficulty", None),
            }
        )
    return payload


def _deref_object_ids(value: Any, by_id: Dict[str, Any]) -> Any:
    """Undo cognition's `_serialize_value`: `{"object_id": "obj0"}` ->
    `by_id["obj0"]` (a real scene object); recurses into lists (for
    `all_objects`); anything else (numbers, plain [x, y] pairs, strings) is
    passed through unchanged."""
    if isinstance(value, dict) and set(value.keys()) == {"object_id"}:
        oid = value["object_id"]
        if oid not in by_id:
            raise KeyError(f"cognition referenced unknown object_id {oid!r}")
        return by_id[oid]
    if isinstance(value, list):
        return [_deref_object_ids(v, by_id) for v in value]
    return value


def _resolve_cognition_plan(raw_plan: List[Dict[str, Any]], scene: List[Any]) -> List[Any]:
    from components.policy import make_skill_call
    from components.skills import SkillValidationError

    by_id = {f"obj{i}": obj for i, obj in enumerate(scene)}
    calls = []
    for raw in raw_plan:
        if not isinstance(raw, dict):
            continue
        skill = raw.get("skill")
        raw_params = raw.get("params") or {}
        try:
            kwargs = {k: _deref_object_ids(v, by_id) for k, v in raw_params.items()}
            calls.append(make_skill_call(skill, **kwargs))
        except (SkillValidationError, KeyError, TypeError) as exc:
            print(f"orchestrator: dropped cognition plan call {raw!r}: {exc}")
    return calls


def _plan(goal: str, scene: List[Any]) -> Tuple[List[Any], str, Optional[str], Optional[Dict[str, Any]]]:
    """Returns (skill_calls, planning_source, cognition_error, cognition_objects).
    `cognition_objects` is cognition's per-object_id `objects` dict (calibrated
    plan + grasp affordance + retry budget), or None when cognition wasn't
    used (unreachable / bad response)."""
    from components.policy import plan_task

    scene_payload = _scene_to_cognition_payload(scene)
    cognition_error: Optional[str] = None
    resolved_plan: Optional[List[Any]] = None
    cognition_objects: Optional[Dict[str, Any]] = None

    try:
        resp = call_service("cognition", "/plan", json={"goal": goal, "scene": scene_payload}, timeout=30.0)
        if isinstance(resp, dict) and resp.get("ok") and isinstance(resp.get("plan"), list):
            resolved_plan = _resolve_cognition_plan(resp["plan"], scene)
            cognition_objects = resp.get("objects") if isinstance(resp.get("objects"), dict) else {}
        else:
            cognition_error = f"cognition returned an unexpected response: {resp!r}"[:500]
    except Exception as exc:
        cognition_error = str(exc)

    if resolved_plan is not None:
        return resolved_plan, "cognition", cognition_error, cognition_objects

    # local_fallback: components.policy.plan_task's own default
    # rule_based_stub_planner -- fully offline, no robot/network needed.
    plan = plan_task(goal, scene)
    return plan, "local_fallback", cognition_error, None


def _calibrated_plans(scene: List[Any], cognition_objects: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-object calibrated pick params. When cognition answered, use ITS
    `objects[*].calibrated_plan` (keyed by object_id -- precise even when two
    objects share a canonical_label). Otherwise fall back to querying
    color-sort's own experience store directly (local, always available,
    keyed by canonical_label -- the same source components/pipeline.py's
    `_run_pick` biases RetryController with)."""
    if cognition_objects:
        return {
            oid: entry.get("calibrated_plan")
            for oid, entry in cognition_objects.items()
            if isinstance(entry, dict)
        }
    from components.experience_store import ExperienceStore, resolve_key

    store = ExperienceStore()
    return {resolve_key(obj): store.get_calibration(resolve_key(obj)) for obj in scene}


def _grasp_affordances(cognition_objects: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Per-object grasp affordance -- ONLY available via cognition
    (components.grasp_affordance needs a 3D point cloud this service has no
    other source for); None when cognition wasn't used."""
    if not cognition_objects:
        return None
    out = {
        oid: entry["grasp_affordance"]
        for oid, entry in cognition_objects.items()
        if isinstance(entry, dict) and "grasp_affordance" in entry
    }
    return out or None


def _plan_to_json(calls: List[Any]) -> List[Dict[str, Any]]:
    def _obj_view(o: Any) -> Dict[str, Any]:
        return {
            "canonical_label": getattr(o, "canonical_label", None),
            "color": getattr(o, "color", None),
            "x": getattr(o, "x", None),
            "y": getattr(o, "y", None),
            "difficulty": getattr(o, "difficulty", None),
        }

    out = []
    for call in calls:
        params: Dict[str, Any] = {}
        for f in dataclasses.fields(call.params):
            value = getattr(call.params, f.name)
            if hasattr(value, "x") and hasattr(value, "canonical_label"):
                params[f.name] = _obj_view(value)
            elif isinstance(value, (list, tuple)) and value and hasattr(value[0], "x"):
                params[f.name] = [_obj_view(v) for v in value]
            else:
                params[f.name] = value
        out.append({"skill": call.skill, "params": params})
    return out


# ---------------------------------------------------------------------------
# Move constraints: per-move `count` limit + `place` routing (handoff)
# ---------------------------------------------------------------------------


def _object_key_of(call: Any) -> Tuple[Optional[str], Optional[Any]]:
    for field_name in ("object", "target"):
        target = getattr(call.params, field_name, None)
        if target is not None:
            key = (getattr(target, "canonical_label", "") or getattr(target, "color", "") or "").strip().lower()
            return key or None, target
    return None, None


def _apply_move_constraints(calls: List[Any], moves: List[Dict[str, Any]]) -> List[Any]:
    """Enforce each move's requested `count` (drop extra calls for that
    object beyond the requested amount) and `place` (route a `pick` to the
    `handoff` skill when the move asked for "handoff"; "dropoff"/"bin1"/
    "bin2" are left as the object's own default-color-bin `pick`/`place` --
    this codebase has no place-only primitive that reaches an arbitrary bin,
    see components/skills.py's PlaceParams docstring)."""
    from components.constants import normalize_object
    from components.policy import make_skill_call
    from components.skills import SkillValidationError

    by_object: Dict[str, Dict[str, Any]] = {}
    for mv in moves or []:
        key = normalize_object(mv.get("object") or mv.get("color") or "")
        if key:
            by_object[key] = mv

    used: Dict[str, int] = {}
    out: List[Any] = []
    for call in calls:
        key, target = _object_key_of(call)
        move = by_object.get(key) if key else None
        if move is not None:
            limit = move.get("count", 1)
            if limit is not None:
                n_used = used.get(key, 0)
                if n_used >= int(limit):
                    continue  # move's count already satisfied -- drop this extra call
                used[key] = n_used + 1
            place = (move.get("place") or "").strip().lower()
            if place == "handoff" and call.skill == "pick":
                try:
                    call = make_skill_call("handoff", object=target)
                except SkillValidationError:
                    pass  # handoff not valid here (e.g. out of workspace) -- keep the plain pick
        out.append(call)
    return out


# ---------------------------------------------------------------------------
# Live execution (ONLY reached when VIAM_ALLOW_LIVE is set -- see _handle_run)
# ---------------------------------------------------------------------------


def _retarget_calls_to_real_scene(calls: List[Any], exec_scene: List[Any]) -> List[Any]:
    """Re-point each planned call's object at the matching REAL,
    depth-calibrated object in `exec_scene` (from
    components.vision.VisionComponent.locate_blocks), by canonical_label/
    color, greedily consuming matches so the same physical object is never
    targeted twice. A call whose target has no remaining match in
    `exec_scene` is dropped -- this service refuses to drive the arm to a
    pixel-space location that the depth-grounded re-scan didn't confirm.
    Every retargeted call is re-validated (via make_skill_call, workspace +
    Z-floor) against the REAL coordinates before it survives into the
    result."""
    from components.policy import make_skill_call
    from components.skills import SkillValidationError

    remaining = list(exec_scene)
    out: List[Any] = []
    for call in calls:
        key, _old_target = _object_key_of(call)
        if key is None:
            out.append(call)
            continue
        match = None
        for cand in remaining:
            cand_key = (getattr(cand, "canonical_label", "") or getattr(cand, "color", "") or "").strip().lower()
            if cand_key == key:
                match = cand
                break
        if match is None:
            continue
        remaining.remove(match)
        try:
            if call.skill == "declutter":
                out.append(make_skill_call("declutter", target=match, all_objects=exec_scene))
            elif hasattr(call.params, "object"):
                kwargs = {
                    f.name: getattr(call.params, f.name)
                    for f in dataclasses.fields(call.params)
                    if f.name != "object"
                }
                kwargs["object"] = match
                out.append(make_skill_call(call.skill, **kwargs))
            else:
                out.append(call)
        except SkillValidationError:
            continue  # real coordinates failed safety validation -- drop
    return out


def _seed_difficulty_from_plan_scene(exec_scene: List[Any], plan_scene: List[Any]) -> None:
    """Best-effort: carry each planned object's uq-derived score/difficulty
    over onto the matching REAL object (by canonical_label), so the live
    RetryController's difficulty-derived retry budget still reflects what
    perception+uq actually saw. Objects with no match keep difficulty=None
    (RetryController falls back to DEFAULT_RETRY_BUDGET for those)."""
    pool: Dict[str, List[Any]] = {}
    for obj in plan_scene:
        key = (getattr(obj, "canonical_label", "") or getattr(obj, "color", "") or "").strip().lower()
        pool.setdefault(key, []).append(obj)
    for obj in exec_scene:
        key = (getattr(obj, "canonical_label", "") or getattr(obj, "color", "") or "").strip().lower()
        bucket = pool.get(key)
        if bucket:
            src = bucket.pop(0)
            obj.score = getattr(src, "score", None)
            obj.difficulty = getattr(src, "difficulty", None)


async def _execute_live(
    machine: Any,
    calls: List[Any],
    plan_scene: List[Any],
    moves: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from components.arm import ArmComponent
    from components.constants import PICK_OBJECTS
    from components.experience_store import ExperienceStore
    from components.gripper import GripperComponent
    from components.pickplace import PickPlace
    from components.pipeline import TaskContext, execute_plan_with_memory
    from components.retry import RetryController
    from components.vision import VisionComponent

    arm = ArmComponent(machine)
    gripper = GripperComponent(machine)
    pickplace = PickPlace(arm, gripper)
    vision = VisionComponent(machine)
    store = ExperienceStore()

    exec_scene = await vision.locate_blocks(colors=PICK_OBJECTS)
    _seed_difficulty_from_plan_scene(exec_scene, plan_scene)
    for obj in exec_scene:
        store.seed(obj)

    retargeted = _retarget_calls_to_real_scene(calls, exec_scene)
    final_calls = _apply_move_constraints(retargeted, moves)

    retry = RetryController(arm, gripper)
    ctx = TaskContext(arm=arm, gripper=gripper, pickplace=pickplace, store=store, retry=retry)

    # execute_plan_with_memory appends exactly one StepResult per call, in
    # order, so `results[i]` <-> `final_calls[i]`.
    results = await execute_plan_with_memory(final_calls, exec_scene, ctx)

    record_report = _record_outcomes_with_cognition(results, final_calls)

    return {
        "ok": True,
        "scene_size": len(exec_scene),
        "planned_calls": len(calls),
        "executed_calls": len(final_calls),
        "results": [dataclasses.asdict(r) for r in results],
        "cognition_record": record_report,
    }


def _record_outcomes_with_cognition(results: List[Any], calls: List[Any]) -> Dict[str, Any]:
    """Best-effort POST of each step's outcome to cognition's real
    ``POST /record`` contract (``{canonical_label, xy, grasp_success,
    failure_type?, plan_params?}`` -- see services/cognition_service.py),
    to close the self-adaptive loop per the orchestrator spec (step 6).
    components/pipeline.py's execute_plan_with_memory already persisted
    these outcomes locally via the experience store's own record_attempt
    (calibration keeps working even if this HTTP call fails -- both it and
    cognition's own ExperienceStore read/write the same default
    data/experience.json unless overridden)."""
    errors: List[str] = []
    reported = 0
    for r, call in zip(results, calls):
        target = getattr(call.params, "object", None) or getattr(call.params, "target", None)
        if target is None:
            continue
        outcome = {
            "canonical_label": r.key,
            "xy": [float(getattr(target, "x", 0.0)), float(getattr(target, "y", 0.0))],
            "grasp_success": r.success,
            "failure_type": r.failure_type,
            "plan_params": {"skill": r.skill, "attempts": r.attempts, "escalations": list(r.escalations)},
        }
        try:
            call_service("cognition", "/record", json=outcome, timeout=10.0)
            reported += 1
        except Exception as exc:
            errors.append(str(exc))
    return {"reported": reported, "total": len(results), "errors": errors[:3]}


# ---------------------------------------------------------------------------
# The core pipeline -- shared by the FastAPI route and the --smoke check
# ---------------------------------------------------------------------------


async def _handle_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    task = payload.get("task")
    moves = list(payload.get("moves") or [])
    say = payload.get("say")
    live = _live_allowed()
    errors: Dict[str, Optional[str]] = {"perception": None, "uq": None, "cognition": None}

    machine = None
    try:
        if live:
            from components.connection import connect_machine

            machine = await connect_machine()

        try:
            image_bytes, frame_source = await _resolve_frame(payload, machine)
        except Exception as exc:
            return {
                "ok": False,
                "mode": "live" if live else "dry",
                "error": f"could not obtain a frame: {exc}",
                "errors": errors,
            }

        labels = payload.get("labels") or _labels_from_moves(moves)
        detections, perception_error = _call_perception(image_bytes, labels)
        errors["perception"] = perception_error

        scored, uq_resp, uq_error = _call_uq(image_bytes, detections, payload.get("n"))
        errors["uq"] = uq_error

        scene = _detections_to_scene(scored)
        goal = _goal_text(task, moves, say)

        plan_calls, planning_source, cognition_error, cognition_objects = _plan(goal, scene)
        errors["cognition"] = cognition_error

        result: Dict[str, Any] = {
            "ok": True,
            "mode": "live" if live else "dry",
            "frame_source": frame_source,
            "goal": goal,
            "detections": scored,
            "uq": uq_resp,
            "plan": _plan_to_json(plan_calls),
            "planning_source": planning_source,
            "calibrated_plan": _calibrated_plans(scene, cognition_objects),
            "grasp_affordance": _grasp_affordances(cognition_objects),
            "errors": errors,
            "say": say,
        }

        if not live:
            result["execution"] = None
            result["execution_note"] = (
                "execution skipped: VIAM_ALLOW_LIVE not set -- plan-only/dry mode, "
                "the robot was never connected"
            )
            return result

        # Live: apply move constraints to the (pixel-space) planned calls'
        # object *selection* isn't meaningful yet -- constraints are applied
        # again after retargeting in _execute_live, against the real scene;
        # this call here only decides ordering/limits don't matter twice
        # since count-limiting is idempotent given the same call order.
        try:
            result["execution"] = await _execute_live(machine, plan_calls, scene, moves)
        except Exception as exc:
            result["execution"] = {"ok": False, "error": str(exc)}
        return result
    finally:
        if machine is not None:
            await machine.close()


@app.post("/run")
async def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST a voice-intent dict ``{task, moves, say}`` (color-sort's
    ``components.voice.map_task`` shape), plus optionally ``frame_path`` or
    ``image_base64`` (a saved test frame -- used whenever VIAM_ALLOW_LIVE is
    unset) and ``n`` (UQ augmentation-consistency sample count). See the
    module docstring for the full response shape and the DRY vs. live gate.
    """
    return await _handle_run(payload)


# ---------------------------------------------------------------------------
# Smoke check (inline sanity path -- NOT a pytest test file)
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """Exercise the DRY path with mocked perception/uq/cognition responses --
    no network, no robot. Confirms:
      * with VIAM_ALLOW_LIVE unset, `/run`'s execution step is skipped and
        the result says so;
      * `components.connection.connect_machine` is never even imported/called
        (a canary replaces it with something that raises if invoked);
      * the composed plan/detections/difficulty/calibrated_plan all come
        back populated from the mocked perception/uq/cognition responses.
    """
    import asyncio
    import sys
    import types

    import services.base as base

    os.environ.pop("VIAM_ALLOW_LIVE", None)  # make sure we're testing the DRY default

    # -- canary: connect_machine must NEVER be called in the DRY path -------
    fake_connection = types.ModuleType("components.connection")

    async def _must_not_connect(*_a, **_kw):  # pragma: no cover - should never run
        raise AssertionError("connect_machine was called in the DRY (no VIAM_ALLOW_LIVE) path!")

    fake_connection.connect_machine = _must_not_connect
    sys.modules["components.connection"] = fake_connection

    # -- mock the 3 downstream services (no HTTP, no processes running) -----
    # cx/cy chosen to fall inside components.constants.WORKSPACE_CORNERS (a
    # real-world-mm polygon), so components.policy.plan_task's safety
    # validation (in_workspace) accepts both -- these detections' pixel
    # coordinates are used as a plan-only placeholder for "world xy" in the
    # DRY path (see `_detections_to_scene`'s docstring note); this is purely
    # to exercise the real validation path in the smoke check, not a claim
    # that pixels equal millimeters.
    mock_detections = [
        {
            "label": "yellow block",
            "canonical_label": "yellow",
            "box": [190, 90, 20, 20],
            "cx": 200,
            "cy": 100,
            "score": 0.91,
            "sources": ["owlv2"],
            "raw_labels": {"owlv2": "yellow block"},
            "mask": None,
            "yaw": None,
        },
        {
            "label": "red block",
            "canonical_label": "red",
            "box": [240, 140, 20, 20],
            "cx": 250,
            "cy": 150,
            "score": 0.77,
            "sources": ["owlv2"],
            "raw_labels": {"owlv2": "red block"},
            "mask": None,
            "yaw": None,
        },
    ]

    def _fake_call_service(name, path, **kwargs):
        if name == "perception" and path == "/detect":
            return {"image": {"width": 64, "height": 64}, "detections": mock_detections}
        if name == "uq" and path == "/score":
            dets = json.loads(kwargs["json"]["detections"])
            for d in dets:
                d["difficulty"] = 0.1 if d["canonical_label"] == "yellow" else 0.6
                d["consistency"] = 0.95
            return {"ok": True, "n_requested": 5, "n_used": 5, "augmented": True, "degraded_reason": None, "from_perception": False, "detections": dets}
        if name == "cognition" and path == "/plan":
            # Real services/cognition_service.py response shape: `plan`
            # entries are already-validated SkillCalls, JSON-serialized
            # with object refs as {"object_id": "objN"} nested under each
            # param name (see that module's `_serialize_call`); `objects`
            # carries per-object_id calibrated_plan/grasp_affordance/etc.
            return {
                "ok": True,
                "goal": kwargs["json"]["goal"],
                "plan": [
                    {"skill": "pick", "params": {"object": {"object_id": "obj0"}}},
                    {"skill": "pick", "params": {"object": {"object_id": "obj1"}}},
                ],
                "objects": {
                    "obj0": {
                        "canonical_label": "yellow",
                        "calibrated_plan": {"pick_z_offset": 0.0, "xy_offset": [0.0, 0.0], "grip_params": {}, "retry_budget": 1},
                        "grasp_affordance": {"grasp_type": "top_down"},
                    },
                    "obj1": {
                        "canonical_label": "red",
                        "calibrated_plan": {"pick_z_offset": 0.0, "xy_offset": [0.0, 0.0], "grip_params": {}, "retry_budget": 3},
                    },
                },
            }
        raise AssertionError(f"unexpected call_service({name!r}, {path!r}) during dry smoke")

    base.call_service = _fake_call_service  # this module imported `call_service` by reference

    import services.orchestrator_service as orch

    orch.call_service = _fake_call_service

    payload = {
        "task": "sort",
        "moves": [
            {"object": "yellow", "place": "bin2", "count": 1},
            {"object": "red", "place": "bin1", "count": 1},
        ],
        "say": "Sorting one yellow and one red block.",
        "image_base64": base64.b64encode(_synthetic_placeholder_frame()).decode("ascii"),
    }

    result = asyncio.run(orch._handle_run(payload))

    assert result["ok"] is True, result
    assert result["mode"] == "dry", result
    assert result["execution"] is None, result
    assert "execution skipped" in result["execution_note"], result
    assert result["planning_source"] == "cognition", result
    assert len(result["plan"]) == 2, result
    assert result["errors"]["cognition"] is None, result
    assert any(d["canonical_label"] == "yellow" and d.get("difficulty") == 0.1 for d in result["detections"]), result
    assert result["calibrated_plan"]["obj0"]["retry_budget"] == 1, result
    assert result["grasp_affordance"]["obj0"] == {"grasp_type": "top_down"}, result
    assert "obj1" not in (result["grasp_affordance"] or {}), result

    print("DRY /run result:")
    print(json.dumps({k: v for k, v in result.items() if k != "uq"}, indent=2, default=str))
    print("smoke OK: DRY path returns the plan, execution skipped, robot never connected")


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        _smoke()
    else:
        import uvicorn

        from services import config

        host, port = config.service_addr("orchestrator")
        uvicorn.run(app, host=host, port=port)
