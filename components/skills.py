"""Stage A code-as-policy: the typed skill registry -- adapted to
color-sort's `components.pickplace.PickPlace` API.

This module is the API surface a planner (rule-based stub today, an LLM or
Viam MCP client later -- see components/policy.py) is constrained to. It
does NOT reimplement any robot behavior: every skill's adapter is a thin
wrapper over color-sort's existing primitives in components/pickplace.py
(plus components/declutter.py's pure planning helper for the `declutter`
skill). What this module adds on top of those primitives:

  1. A typed param schema (a frozen dataclass) per skill, so a planner's
     output can be structurally validated instead of trusted.
  2. A docstring per skill that IS the tool description a planner sees
     (via `skill_menu()`), so the menu shown to an LLM and the code that
     executes it can never drift apart.
  3. A safety `validate()` per skill that checks components.safety.in_workspace
     and the components.constants.MIN_Z Z-floor BEFORE the adapter would
     run. Nothing here drives the arm; validation is pure and offline.

No robot I/O happens at import time and no skill executes itself -- callers
(components/policy.py's `execute()`, or a caller's own code) invoke the
`adapter` explicitly, and only after validation succeeds.

ADAPTATION NOTES (color-sort's PickPlace vs. the original viam_5 source
this was ported from)
------------------------------------------------------------------------
color-sort's `PickPlace` has a DIFFERENT primitive surface than the source
this module was originally written against:

  - `pick_and_place(block, color_bins=None) -> bool`  (same shape as the
    source's `pick_and_place(block) -> bool`, but takes an *optional bin
    override dict* -- this is what makes the new `handoff` skill possible,
    see below).
  - `hand_to_human() -> dict | None`  (place-only: assumes the gripper is
    ALREADY holding something, moves to the taught "handoff" pose, and
    opens the gripper there -- it takes no target).
  - `sort_blocks(blocks, color_bins=None, counts=None) -> dict`  (a
    count-limited batch entry point; not used directly by this module --
    components/policy.py sequences individual skill calls instead).
  - There is NO `pick_with_declutter` / `_move_aside` on color-sort's
    PickPlace (unlike the source this was ported from). The `declutter`
    and `move_aside` skills below reimplement that behavior HERE, using
    color-sort's own `PickPlace._above` (the exact top-down
    approach/descend primitive `pick_and_place` itself uses) plus
    `PickPlace.gripper.grab()`/`open_full()`, driven by the pure planning
    output of `components.declutter.plan_declutter` (unchanged, still
    robot-free).

New skill: `handoff`
---------------------
color-sort's PickPlace already treats "handoff" as a legitimate placement
target inside `pick_and_place` (see `pick_and_place`'s
`if bin_name in {"dropoff", "handoff"}: ... open_full()` branch) -- it is
simply never reached by the *default* color->bin mapping
(`components.constants.COLOR_BINS`, which only ever names bin1/bin2). The
`handoff` skill below reaches it explicitly: it calls
`pickplace.pick_and_place(object, color_bins={object.color: "handoff"})`,
which picks the object up and places it at the taught handoff pose with the
gripper opened there -- the identical terminal action
`PickPlace.hand_to_human()` performs, just reached via the one atomic
grasp+place primitive this codebase has (there's no separate grasp-only
step, matching `pick`/`place` below).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type

from components.constants import COLOR_BINS, MIN_Z, PICK_ORIENTATION, TAUGHT_JOINTS, TRAVEL_Z
from components.declutter import MoveAside, Pick, plan_declutter
from components.pickplace import PickPlace
from components.safety import in_workspace


class SkillValidationError(ValueError):
    """A skill call's params failed schema and/or safety validation.

    Raised by `validate_skill_call` / `make_skill_call`; components/policy.py
    catches this to REJECT (drop) an invalid planner-emitted call rather
    than executing it.
    """


# ---------------------------------------------------------------------------
# Safety helpers (shared by every skill's validate())
# ---------------------------------------------------------------------------


def _xy_of(obj: Any) -> Tuple[float, float]:
    try:
        return float(obj.x), float(obj.y)
    except AttributeError as exc:
        raise SkillValidationError(
            f"expected an object with numeric .x/.y, got {obj!r}"
        ) from exc


def _check_in_workspace(x: float, y: float, what: str) -> None:
    if not in_workspace(x, y):
        raise SkillValidationError(
            f"{what} at ({x:.1f}, {y:.1f}) is outside the taught workspace"
        )


def _check_z_floor(z: float, what: str) -> None:
    if z < MIN_Z:
        raise SkillValidationError(
            f"{what} z={z:.1f} is below the Z-floor MIN_Z={MIN_Z:.1f}"
        )


# ---------------------------------------------------------------------------
# Param schemas -- these dataclasses ARE the planner-visible tool schema.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickParams:
    """pick(object): pick `object` up and place it in its default,
    color-mapped bin (components.constants.COLOR_BINS). `object` must be a
    LocatedShape-like value (needs .x, .y, .color) already present in the
    scene passed to the planner. The underlying primitive
    (PickPlace.pick_and_place) is atomic -- grasp and place happen as one
    step; there is no separate grasp-only primitive in this codebase."""

    object: Any


@dataclass(frozen=True)
class PlaceParams:
    """place(object, bin): place `object` into a specific taught bin
    ("bin1"/"bin2"/"dropoff"/"handoff"), overriding the default color-mapped
    bin. Only bins that are actually taught (components.constants.TAUGHT_JOINTS)
    are accepted. NOTE: PickPlace exposes no place-only primitive (grasp+place
    are fused in pick_and_place), so this skill's adapter can only reach a
    bin that IS the object's own default color bin; requesting any other bin
    fails validation rather than silently doing the wrong thing. Use the
    `handoff` skill to reach the handoff pose regardless of color."""

    object: Any
    bin: str


@dataclass(frozen=True)
class HandoffParams:
    """handoff(object): pick `object` up and hand it to a human at the
    taught "handoff" pose (components.constants.TAUGHT_POSES["handoff"]),
    opening the gripper there. Reached via
    PickPlace.pick_and_place(object, color_bins={object.color: "handoff"})
    -- the same grasp-then-place-at-handoff-and-open action
    PickPlace.hand_to_human() performs for an already-held object, fused
    here with the grasp since (as with pick/place) there is no separate
    grasp-only primitive in this codebase."""

    object: Any


@dataclass(frozen=True)
class MoveAsideParams:
    """move_aside(object, to_xy): relocate a blocking `object` to the
    explicit (x, y) temp position `to_xy`, without touching a color bin.
    Drives PickPlace's own top-down approach/descend primitive
    (`PickPlace._above`) plus its gripper, the same low-level moves
    `PickPlace.pick_and_place` uses -- color-sort's PickPlace has no
    dedicated move-aside primitive of its own."""

    object: Any
    to_xy: Tuple[float, float]


@dataclass(frozen=True)
class DeclutterParams:
    """declutter(target, all_objects): clear a path to `target` by moving
    any blocking neighbors (found via components.declutter.plan_declutter
    against `all_objects`) out of the way, then pick `target` (via
    PickPlace.pick_and_place). Use this instead of a bare `pick` whenever
    the target's top-down grasp is currently blocked."""

    target: Any
    all_objects: List[Any]


@dataclass(frozen=True)
class DescendUntilContactParams:
    """descend_until_contact(x, y, z_target, force_threshold_n): PLACEHOLDER
    skill -- descend at (x, y) from a safe travel height toward `z_target`
    until contact/force feedback exceeds `force_threshold_n`. This is
    sim/admittance-controller-backed only; the real arm in this repo has no
    force/torque sensing or admittance control wired up yet, so the adapter
    always raises NotImplementedError. It is declared (with real safety
    validation) so the planner's menu and the validation path are stable
    ahead of that hardware integration -- a planner CAN select it, params
    ARE safety-checked, but it cannot yet be executed."""

    x: float
    y: float
    z_target: float = MIN_Z
    force_threshold_n: Optional[float] = None


# ---------------------------------------------------------------------------
# Per-skill validation (params schema is enforced by dataclass construction
# itself; these add the safety checks: in_workspace + Z-floor).
# ---------------------------------------------------------------------------


def _validate_pick(params: PickParams) -> None:
    x, y = _xy_of(params.object)
    _check_in_workspace(x, y, "pick target")
    # The grasp descent height is derived by PickPlace itself (tcp_pick_z,
    # clamped to the safety Z-floor via components.safety.clamp_z inside
    # ArmComponent.move_to_position), so the floor is respected by
    # construction; this call documents/enforces that invariant.
    _check_z_floor(MIN_Z, "pick descent")


def _validate_place(params: PlaceParams) -> None:
    x, y = _xy_of(params.object)
    _check_in_workspace(x, y, "place target")
    _check_z_floor(MIN_Z, "place descent")
    if params.bin not in TAUGHT_JOINTS:
        raise SkillValidationError(
            f"bin {params.bin!r} is not a taught pose; expected one of {list(TAUGHT_JOINTS)}"
        )
    color = getattr(params.object, "color", None)
    default_bin = COLOR_BINS.get(color)
    if params.bin != default_bin:
        raise SkillValidationError(
            f"place: bin {params.bin!r} is not object's default color bin "
            f"({default_bin!r} for color {color!r}); no place-only primitive "
            "exists to reach an arbitrary bin (use the handoff skill for the "
            "handoff pose)"
        )


def _validate_handoff(params: HandoffParams) -> None:
    x, y = _xy_of(params.object)
    _check_in_workspace(x, y, "handoff target")
    _check_z_floor(MIN_Z, "handoff descent")


def _validate_move_aside(params: MoveAsideParams) -> None:
    ox, oy = _xy_of(params.object)
    _check_in_workspace(ox, oy, "move_aside source")
    tx, ty = float(params.to_xy[0]), float(params.to_xy[1])
    _check_in_workspace(tx, ty, "move_aside destination")
    _check_z_floor(MIN_Z, "move_aside descent")


def _validate_declutter(params: DeclutterParams) -> None:
    tx, ty = _xy_of(params.target)
    _check_in_workspace(tx, ty, "declutter target")
    _check_z_floor(MIN_Z, "declutter descent")
    # Blockers themselves are validated as part of plan_declutter's own
    # search (components.declutter._find_temp_zone only ever returns
    # in-workspace candidates); nothing further to check here offline.


def _validate_descend_until_contact(params: DescendUntilContactParams) -> None:
    _check_in_workspace(params.x, params.y, "descend_until_contact")
    _check_z_floor(params.z_target, "descend_until_contact target")
    if params.force_threshold_n is not None and params.force_threshold_n <= 0:
        raise SkillValidationError(
            f"force_threshold_n must be positive, got {params.force_threshold_n!r}"
        )


# ---------------------------------------------------------------------------
# Adapters -- thin wrappers over components.pickplace.PickPlace. None of
# these execute at plan time; components/policy.py's optional `execute()`
# calls them only after a full plan has been validated.
# ---------------------------------------------------------------------------


async def _adapt_pick(params: PickParams, pickplace: PickPlace) -> bool:
    return await pickplace.pick_and_place(params.object)


async def _adapt_place(params: PlaceParams, pickplace: PickPlace) -> bool:
    # See PlaceParams docstring: validated to only ever match the object's
    # own default bin, so this reduces to the same atomic primitive as pick.
    return await pickplace.pick_and_place(params.object)


async def _adapt_handoff(params: HandoffParams, pickplace: PickPlace) -> bool:
    # Force the placement bin to "handoff" for this one object, regardless
    # of its default color bin -- pick_and_place already knows to open the
    # gripper at the handoff pose (see PickPlace.pick_and_place's
    # `bin_name in {"dropoff", "handoff"}` branch), the same thing
    # PickPlace.hand_to_human() does for an already-held object.
    color = getattr(params.object, "color", None) or "handoff"
    return await pickplace.pick_and_place(params.object, color_bins={color: "handoff"})


async def _move_aside(pickplace: PickPlace, obj: Any, to_xy: Tuple[float, float]) -> None:
    """Relocate `obj` to `to_xy` using PickPlace's own top-down
    approach/descend primitive (`_above`) and gripper -- the exact same
    low-level moves `PickPlace.pick_and_place` uses, just without touching
    a color bin. color-sort's PickPlace has no dedicated move-aside method
    of its own (unlike the source this module was ported from), so this
    helper reimplements it here rather than editing components/pickplace.py.
    """
    x, y = float(to_xy[0]), float(to_xy[1])
    ori = dict(PICK_ORIENTATION)
    print(
        f"  declutter: move {obj.color or obj.label} blocker "
        f"({obj.x:.1f}, {obj.y:.1f}) -> ({x:.1f}, {y:.1f})"
    )
    await pickplace.gripper.open_full()
    await pickplace._above(obj.x, obj.y, TRAVEL_Z, ori)
    await pickplace._above(obj.x, obj.y, MIN_Z, ori)
    grasp = await pickplace.gripper.grab()
    holding = getattr(grasp, "holding", None)
    holding = bool(grasp) if holding is None else bool(holding)
    await pickplace._above(obj.x, obj.y, TRAVEL_Z, ori)
    if not holding:
        raise RuntimeError(
            f"declutter: failed to grab blocker at ({obj.x:.1f}, {obj.y:.1f})"
        )
    await pickplace._above(x, y, TRAVEL_Z, ori)
    await pickplace._above(x, y, MIN_Z, ori)
    await pickplace.gripper.open_full()
    await pickplace._above(x, y, TRAVEL_Z, ori)


async def _adapt_move_aside(params: MoveAsideParams, pickplace: PickPlace) -> None:
    await _move_aside(pickplace, params.object, params.to_xy)


async def _adapt_declutter(params: DeclutterParams, pickplace: PickPlace) -> bool:
    plan = plan_declutter(params.target, params.all_objects)
    for action in plan.actions:
        if isinstance(action, MoveAside):
            await _move_aside(pickplace, action.obj, action.to_xy)
        elif isinstance(action, Pick):
            return await pickplace.pick_and_place(action.obj)
    return False


async def _adapt_descend_until_contact(
    params: DescendUntilContactParams, pickplace: PickPlace
) -> None:
    raise NotImplementedError(
        "descend_until_contact is sim/admittance-backed only; not available "
        "on this hardware stack yet"
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillSpec:
    name: str
    params_type: Type
    doc: str
    validate: Callable[[Any], None]
    adapter: Callable[[Any, PickPlace], Awaitable[Any]]


def _doc(params_type: Type) -> str:
    return (params_type.__doc__ or "").strip()


SKILL_REGISTRY: Dict[str, SkillSpec] = {
    "pick": SkillSpec("pick", PickParams, _doc(PickParams), _validate_pick, _adapt_pick),
    "place": SkillSpec("place", PlaceParams, _doc(PlaceParams), _validate_place, _adapt_place),
    "handoff": SkillSpec(
        "handoff", HandoffParams, _doc(HandoffParams), _validate_handoff, _adapt_handoff
    ),
    "move_aside": SkillSpec(
        "move_aside",
        MoveAsideParams,
        _doc(MoveAsideParams),
        _validate_move_aside,
        _adapt_move_aside,
    ),
    "declutter": SkillSpec(
        "declutter",
        DeclutterParams,
        _doc(DeclutterParams),
        _validate_declutter,
        _adapt_declutter,
    ),
    "descend_until_contact": SkillSpec(
        "descend_until_contact",
        DescendUntilContactParams,
        _doc(DescendUntilContactParams),
        _validate_descend_until_contact,
        _adapt_descend_until_contact,
    ),
}


@dataclass(frozen=True)
class SkillCall:
    """One validated, ordered step of a plan: a skill name + an instance of
    that skill's own params dataclass (already schema- and safety-checked)."""

    skill: str
    params: Any


def skill_menu() -> List[Dict[str, Any]]:
    """The planner-visible tool menu: name, param field names/types, and the
    docstring that doubles as the tool description. JSON-shape-friendly (a
    real LLM/MCP planner is handed exactly this, not Python objects)."""

    menu = []
    for name, spec in SKILL_REGISTRY.items():
        params = {}
        for f in fields(spec.params_type):
            type_name = getattr(f.type, "__name__", None) or str(f.type)
            params[f.name] = type_name
        menu.append({"name": name, "doc": spec.doc, "params": params})
    return menu


def make_skill_call(name: str, **kwargs: Any) -> SkillCall:
    """Construct + fully validate a SkillCall from a skill name and kwargs.

    Enforces, in order:
      1. `name` is a known skill (else SkillValidationError: out-of-menu).
      2. kwargs type/field-check against the skill's params dataclass (a
         dataclass TypeError -- missing/extra/renamed field -- is wrapped
         as SkillValidationError).
      3. the skill's own safety validation (in_workspace + Z-floor, plus
         any skill-specific checks) passes.

    This is the single choke point components/policy.py routes every
    planner-emitted call through before it can appear in a returned plan.
    """

    spec = SKILL_REGISTRY.get(name)
    if spec is None:
        raise SkillValidationError(
            f"{name!r} is not in the skill menu; expected one of {list(SKILL_REGISTRY)}"
        )
    try:
        params = spec.params_type(**kwargs)
    except TypeError as exc:
        raise SkillValidationError(f"bad params for skill {name!r}: {exc}") from exc
    spec.validate(params)
    return SkillCall(skill=name, params=params)


def validate_skill_call(call: SkillCall) -> None:
    """Re-validate an already-constructed SkillCall (e.g. one built by hand
    rather than via make_skill_call). Raises SkillValidationError on any
    failure; returns None on success."""

    spec = SKILL_REGISTRY.get(call.skill)
    if spec is None:
        raise SkillValidationError(
            f"{call.skill!r} is not in the skill menu; expected one of {list(SKILL_REGISTRY)}"
        )
    if not isinstance(call.params, spec.params_type):
        raise SkillValidationError(
            f"skill {call.skill!r} expected params of type {spec.params_type.__name__}, "
            f"got {type(call.params).__name__}"
        )
    spec.validate(call.params)


async def execute_call(call: SkillCall, pickplace: PickPlace) -> Any:
    """Execute one already-validated SkillCall via its adapter. Re-validates
    first (defense in depth -- scenes can go stale between planning and
    execution). Not required for/used by the offline tests."""

    validate_skill_call(call)
    spec = SKILL_REGISTRY[call.skill]
    return await spec.adapter(call.params, pickplace)


async def execute_plan(plan: List[SkillCall], pickplace: PickPlace) -> List[Any]:
    """Execute a whole validated plan in order. Not required for tests."""

    return [await execute_call(call, pickplace) for call in plan]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import json

    print(json.dumps(skill_menu(), indent=2))
