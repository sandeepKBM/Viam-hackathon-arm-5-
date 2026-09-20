"""Stage A code-as-policy: the typed skill registry.

This module is the API surface a planner (rule-based stub today, an LLM or
Viam MCP client later -- see components/policy.py) is constrained to. It
does NOT reimplement any robot behavior: every skill's adapter is a thin
wrapper over the existing primitives in components/pickplace.py and
components/declutter.py. What this module adds on top of those primitives:

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
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type

from components.constants import COLOR_BINS, MIN_Z, TAUGHT_JOINTS
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
    ("bin1"/"bin2"), overriding the default color-mapped bin. Only bins that
    are actually taught (components.constants.TAUGHT_JOINTS) are accepted.
    NOTE: PickPlace exposes no place-only primitive (grasp+place are fused
    in pick_and_place), so this skill's adapter can only reach a bin that
    IS the object's own default color bin; requesting any other bin fails
    validation rather than silently doing the wrong thing. Decomposing
    PickPlace into separate grasp/release primitives would require editing
    components/pickplace.py, which is out of scope for this stage."""

    object: Any
    bin: str


@dataclass(frozen=True)
class MoveAsideParams:
    """move_aside(object, to_xy): relocate a blocking `object` to the
    explicit (x, y) temp position `to_xy`, without touching a color bin.
    Wraps the same primitive components.pickplace.PickPlace uses internally
    to execute a components.declutter.MoveAside action."""

    object: Any
    to_xy: Tuple[float, float]


@dataclass(frozen=True)
class DeclutterParams:
    """declutter(target, all_objects): clear a path to `target` by moving
    any blocking neighbors (found via components.declutter.plan_declutter
    against `all_objects`) out of the way, then pick `target`. Use this
    instead of a bare `pick` whenever the target's top-down grasp is
    currently blocked."""

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
    # The grasp descent height is hardcoded to MIN_Z by PickPlace itself
    # (see PickPlace._above / pick_and_place), so the Z-floor is respected
    # by construction; this call documents/enforces that invariant.
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
            "exists to reach an arbitrary bin"
        )


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


async def _adapt_move_aside(params: MoveAsideParams, pickplace: PickPlace) -> None:
    # Reuses the exact primitive PickPlace.pick_with_declutter uses to
    # execute a declutter.MoveAside action.
    await pickplace._move_aside(params.object, params.to_xy)


async def _adapt_declutter(params: DeclutterParams, pickplace: PickPlace) -> bool:
    return await pickplace.pick_with_declutter(params.target, params.all_objects)


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
