"""Stage A code-as-policy: the constrained skill-sequencer.

`plan_task(goal, scene, planner=...)` hands a PLANNER three things:

  1. the skill menu (name + param schema + docstring, from
     components.skills.skill_menu()) -- the ONLY actions it may choose;
  2. a JSON-shape-friendly view of the scene, including each object's
     difficulty/history and a precomputed `blocked` flag; and
  3. the natural-language goal.

The planner returns an ORDERED list of raw calls, each shaped
``{"skill": <name>, "params": {...}}`` where any object reference is by
``object_id``/``target_id`` (a string key into the scene view) rather than a
live Python object -- exactly what a real LLM (JSON tool output) or a Viam
MCP tool-call loop would produce. `plan_task` never trusts this output: every
call is resolved back to a real scene object and pushed through
`components.skills.make_skill_call`, which re-checks (a) the skill is in the
menu, (b) params type-check against that skill's dataclass, and (c) the
skill's own safety validation (workspace + Z-floor). Calls that fail any of
that are REJECTED (dropped, with a printed reason) rather than aborting the
whole plan. `plan_task` does not execute anything -- it returns the
validated `List[SkillCall]`; see `execute()` (thin, not required for tests)
to actually run one via color-sort's PickPlace.

Pluggable planner interface
----------------------------
``Planner = Callable[[str, List[dict], List[dict]], List[dict]]``
(goal, scene_view, skill_menu) -> raw_calls

The default is `rule_based_stub_planner`, a deterministic, offline,
zero-dependency stub (see its docstring for the exact rule) so this whole
module runs with no network/API access.

A REAL planner slots in behind the identical signature:
  - **Claude via the Anthropic API**: send `goal` + `scene_view` + `menu` in
    the prompt (or as a tool-use/JSON-mode schema built from `menu`), ask
    for a JSON array of ``{"skill", "params"}`` in the same shape the stub
    emits, `json.loads` the response, return it. No other code changes --
    `plan_task`'s validation is planner-agnostic.
  - **Viam MCP**: expose each `SKILL_REGISTRY` entry as an MCP tool (name +
    JSON schema from its dataclass fields + its docstring as the tool
    description); an MCP-driving agent's sequence of tool calls IS the raw
    call list, collected and returned once the agent finishes planning
    (planning and execution stay separate steps -- the agent proposes, this
    module validates, `execute()` -- or a real MCP tool executor -- performs).
  - Either way the planner is untrusted input. `plan_task` is the sandbox:
    nothing outside `SKILL_REGISTRY`'s adapters can ever run, and every
    param is safety-checked before the call survives into the returned plan.

Stage B (later, not implemented here): full codegen, where the LLM writes
Python composing these same skills, run in a locked sandbox instead of
picking from a fixed menu.

ADAPTATION NOTE (color-sort's PickPlace)
------------------------------------------
This module's shape is UNCHANGED from the source it was ported from -- it
only ever talks to the skill registry (components/skills.py), never
PickPlace directly, so the adaptation to color-sort's manipulation API
(pick_and_place/hand_to_human/sort_blocks vs. the source's
pick_and_place/pick_with_declutter) is entirely contained in
components/skills.py's adapters. The one addition here is the `handoff`
skill: it is wired into `_OBJECT_REF_FIELDS` (so a planner-emitted
``{"skill": "handoff", "params": {"object_id": ...}}`` resolves the same way
`pick` does) and into `rule_based_stub_planner` (goal phrases like "hand it
to me" / "give me the ..." / "hand off the ..." emit a `handoff` call
instead of a plain `pick`, mirroring the existing "clear a path to ..." ->
`declutter` special case).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from components.declutter import is_blocked
from components.skills import (
    SKILL_REGISTRY,
    PickPlace,
    SkillCall,
    SkillValidationError,
    execute_call,
    make_skill_call,
    skill_menu,
)

RawCall = Dict[str, Any]
SceneView = List[Dict[str, Any]]
Planner = Callable[[str, SceneView, List[Dict[str, Any]]], List[RawCall]]


# ---------------------------------------------------------------------------
# Scene view: the JSON-safe projection of `scene` handed to any planner.
# ---------------------------------------------------------------------------


def _scene_view(scene: Sequence[Any]) -> Tuple[SceneView, Dict[str, Any]]:
    """Build the planner-facing scene view and an object_id -> real object
    lookup used to resolve the planner's output back to live objects.
    `object_id` is just a stable per-call index ("obj0", "obj1", ...); it is
    NOT persisted across calls to plan_task, matching the fact a fresh
    scene is (re)perceived each time.
    """

    view: SceneView = []
    by_id: Dict[str, Any] = {}
    scene = list(scene)
    for i, obj in enumerate(scene):
        oid = f"obj{i}"
        by_id[oid] = obj
        others = [o for o in scene if o is not obj]
        view.append(
            {
                "object_id": oid,
                "label": getattr(obj, "label", "") or "",
                "canonical_label": getattr(obj, "canonical_label", "") or "",
                "color": getattr(obj, "color", "") or "",
                "x": float(getattr(obj, "x")),
                "y": float(getattr(obj, "y")),
                "z": float(getattr(obj, "z", 0.0) or 0.0),
                "difficulty": getattr(obj, "difficulty", None),
                "score": getattr(obj, "score", None),
                "history": getattr(obj, "history", None),
                "blocked": is_blocked(obj, others) if others else False,
            }
        )
    return view, by_id


# ---------------------------------------------------------------------------
# Rule-based stub planner (default; fully offline, deterministic)
# ---------------------------------------------------------------------------


def rule_based_stub_planner(
    goal: str, scene_view: SceneView, menu: List[Dict[str, Any]]
) -> List[RawCall]:
    """Deterministic default planner -- no network, no model, no randomness.

    Rule:
      - If the goal reads like a decluttering request ("clear a path to
        ...", "clear path", "path to ..."): pick the single best-matching
        object (by color mention + noun mention in the goal, falling back
        to the first mentioned-color object, then the first object overall)
        and emit exactly one `declutter(target=<match>)` call.
      - If the goal reads like a handoff request ("hand off ...", "hand it
        to me", "hand me the ...", "give me the ...", "give it to ..."):
        same best-match rule as declutter, and emit exactly one
        `handoff(object=<match>)` call.
      - Otherwise (e.g. "sort the red and yellow blocks"): take every
        object whose color is named in the goal (or ALL objects if no
        known color is named), and emit one call per object, EASY-DIFFICULTY
        FIRST (ascending `difficulty`; unknown difficulty sorts last, as if
        hardest; ties broken by higher detector `score` then position for
        determinism). For each: if the scene view marks it `blocked`, emit
        `declutter(target=<it>)` (which itself clears blockers then picks);
        otherwise emit a plain `pick(object=<it>)`.
    """

    goal_l = goal.lower()
    scene_colors = {o["color"] for o in scene_view if o.get("color")}
    mentioned_colors = {c for c in scene_colors if c and c in goal_l}

    clear_path_markers = ("clear a path", "clear path", "path to", "declutter")
    if any(marker in goal_l for marker in clear_path_markers):
        target = _best_goal_match(goal_l, scene_view, mentioned_colors)
        if target is None:
            return []
        return [{"skill": "declutter", "params": {"target_id": target["object_id"]}}]

    handoff_markers = (
        "hand off",
        "handoff",
        "hand it",
        "hand me",
        "hand the",
        "give me",
        "give it",
    )
    if any(marker in goal_l for marker in handoff_markers):
        target = _best_goal_match(goal_l, scene_view, mentioned_colors)
        if target is None:
            return []
        return [{"skill": "handoff", "params": {"object_id": target["object_id"]}}]

    if mentioned_colors:
        candidates = [o for o in scene_view if o.get("color") in mentioned_colors]
    else:
        candidates = list(scene_view)

    def sort_key(o: Dict[str, Any]):
        difficulty = o.get("difficulty")
        difficulty = 1.0 if difficulty is None else float(difficulty)
        score = o.get("score")
        score = float(score) if score is not None else 0.0
        return (difficulty, -score, o["x"], o["y"])

    calls: List[RawCall] = []
    for obj in sorted(candidates, key=sort_key):
        if obj.get("blocked"):
            calls.append({"skill": "declutter", "params": {"target_id": obj["object_id"]}})
        else:
            calls.append({"skill": "pick", "params": {"object_id": obj["object_id"]}})
    return calls


def _best_goal_match(
    goal_l: str, scene_view: SceneView, mentioned_colors: Any
) -> Optional[Dict[str, Any]]:
    pool = [o for o in scene_view if o.get("color") in mentioned_colors] if mentioned_colors else list(scene_view)
    if not pool:
        pool = list(scene_view)
    if not pool:
        return None

    def noun_hit(o: Dict[str, Any]) -> bool:
        for key in ("canonical_label", "label"):
            val = (o.get(key) or "").strip()
            if val and val.lower() in goal_l:
                return True
        return False

    for o in pool:
        if noun_hit(o):
            return o
    return pool[0]


# ---------------------------------------------------------------------------
# Planner-output -> validated SkillCall
# ---------------------------------------------------------------------------

_OBJECT_REF_FIELDS = {
    "pick": ("object_id", "object"),
    "place": ("object_id", "object"),
    "handoff": ("object_id", "object"),
    "move_aside": ("object_id", "object"),
    "declutter": ("target_id", "target"),
}


def _resolve_object(raw_params: Dict[str, Any], key: str, by_id: Dict[str, Any]) -> Any:
    if key not in raw_params:
        raise SkillValidationError(f"missing required {key!r} in params")
    oid = raw_params[key]
    if oid not in by_id:
        raise SkillValidationError(f"unknown object_id/target_id {oid!r}")
    return by_id[oid]


def _to_skill_call(raw: Any, by_id: Dict[str, Any], scene: Sequence[Any]) -> SkillCall:
    if not isinstance(raw, dict):
        raise SkillValidationError(f"malformed planner output (not an object): {raw!r}")
    name = raw.get("skill")
    if not isinstance(name, str) or name not in SKILL_REGISTRY:
        raise SkillValidationError(
            f"{name!r} is not in the skill menu; expected one of {list(SKILL_REGISTRY)}"
        )
    raw_params = dict(raw.get("params") or {})
    kwargs: Dict[str, Any] = {}

    if name in _OBJECT_REF_FIELDS:
        id_key, param_key = _OBJECT_REF_FIELDS[name]
        kwargs[param_key] = _resolve_object(raw_params, id_key, by_id)

    if name == "place":
        kwargs["bin"] = raw_params.get("bin")
    elif name == "move_aside":
        to_xy = raw_params.get("to_xy")
        if to_xy is None or len(to_xy) != 2:
            raise SkillValidationError(f"move_aside requires a 2-element to_xy, got {to_xy!r}")
        kwargs["to_xy"] = (float(to_xy[0]), float(to_xy[1]))
    elif name == "declutter":
        # all_objects is always the full current scene -- the planner
        # references only the target; it never has to (and cannot) hand
        # back a list of live objects itself.
        kwargs["all_objects"] = list(scene)
    elif name == "descend_until_contact":
        for field in ("x", "y", "z_target", "force_threshold_n"):
            if field in raw_params and raw_params[field] is not None:
                kwargs[field] = raw_params[field]

    return make_skill_call(name, **kwargs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan_task(
    goal: str,
    scene: List[Any],
    planner: Optional[Planner] = None,
) -> List[SkillCall]:
    """Plan a validated, ordered sequence of skill calls for `goal` over
    `scene` (a list of LocatedShape-like objects). Does NOT execute
    anything. `planner` defaults to `rule_based_stub_planner` (offline); any
    callable matching the `Planner` signature can be injected instead (see
    module docstring for how a real LLM/MCP planner slots in).

    Every call the planner proposes is resolved + validated via
    components.skills.make_skill_call before it can appear in the returned
    list; calls that reference an unknown skill, fail param type-checking,
    or fail safety validation (out-of-workspace / below the Z-floor / an
    unreachable bin / etc.) are REJECTED -- dropped with a printed reason --
    rather than raising, so one bad call doesn't abort an otherwise-valid
    plan. Order among the accepted calls is preserved.
    """

    chosen_planner = planner or rule_based_stub_planner
    menu = skill_menu()
    scene_view, by_id = _scene_view(scene)

    raw_calls = chosen_planner(goal, scene_view, menu)

    plan: List[SkillCall] = []
    for raw in raw_calls:
        try:
            plan.append(_to_skill_call(raw, by_id, scene))
        except SkillValidationError as exc:
            print(f"  [policy] rejected planner call {raw!r}: {exc}")
            continue
    return plan


async def execute(
    plan: List[SkillCall],
    pickplace: PickPlace,
) -> List[Any]:
    """Thin executor: run each already-validated call via its skill
    adapter, in order. Not required for/used by the offline tests -- real
    execution needs a connected ArmComponent/GripperComponent behind
    `pickplace`. Re-validates each call defensively (see
    components.skills.execute_call) in case the scene has gone stale
    between planning and execution."""

    results = []
    for call in plan:
        results.append(await execute_call(call, pickplace))
    return results
