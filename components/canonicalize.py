"""Canonical vocabulary reconciliation (W2) for color-sort's pick-object vocabulary.

RECONCILIATION NOTE
--------------------
The source repo's ``canonicalize.py`` (viam_5, zero-shot-detection branch)
produced a small, generic ``"<color> <noun>"`` vocabulary (e.g. ``"red
block"``, ``"yellow block"``, ``"blue triangle"``) for its zero-shot / VLM
detection pipeline. color-sort's own vocabulary
(``components.constants.PICK_OBJECTS`` / ``OBJECT_ALIASES`` /
``normalize_object``) is narrower and OBJECT-IDENTITY based:
``{red, yellow, can, cup, airpods, pen, bottle}``. In color-sort, "red" and
"yellow" ARE the pick-object identities (the color IS the object -- there is
no separate "block" noun in their vocabulary), while can/cup/airpods/pen/
bottle are picked by noun regardless of color.

Rather than keep a second, parallel synonym table, this module DELEGATES to
color-sort's own ``components.constants.normalize_object`` (and its
``OBJECT_ALIASES`` map) wherever it can, so THEIR vocabulary wins and there
is exactly one alias table to maintain going forward
(``components.constants.OBJECT_ALIASES``). This module only contributes the
free-form NLP preprocessing (article/punctuation/plural stripping, hyphen
splitting, and folding a handful of shape synonyms like "cube"/"box"/
"brick" -> "block" or "mug" -> "cup") needed to turn a noisy VLM/LLM phrase
like ``"a red cube"`` or ``"Yellow Blocks!"`` into something
``normalize_object`` can match, then normalizes again.

Examples (color-sort vocabulary out):

    canonicalize_label("soda can")   -> "can"     (OBJECT_ALIASES["sodacan"])
    canonicalize_label("red block")  -> "red"     (OBJECT_ALIASES["redblock"])
    canonicalize_label("mug")        -> "cup"     (OBJECT_ALIASES["mug"])
    canonicalize_label("a red cube") -> "red"     (cube -> block synonym,
                                                     then "redblock" -> "red")
    canonicalize_label("Yellow Blocks!") -> "yellow"
    canonicalize_label("earbuds")    -> "airpods"

A label that isn't part of color-sort's pick-object vocabulary at all (e.g.
a shape-only label like "triangle"/"ball"/"cylinder", or a color outside
{red, yellow} such as "green"/"blue") falls back to the original
``"<color> <noun>"`` scheme so components.shapes' HSV shape classifier
labels still canonicalize sanely (e.g. ``"red triangle"`` stays
``"red triangle"``, ``"green box"`` -> ``"green block"``).
"""

import re
from typing import Optional

from components.constants import normalize_object

# Reuse the same color vocabulary as zeroshot.py / vlm.py (kept in sync
# manually since this module must not import either -- both of those import
# from components.shapes / are heavier; canonicalize.py stays dependency-
# light, only pulling in components.constants for the shared alias table).
_KNOWN_COLORS = (
    "red",
    "yellow",
    "green",
    "blue",
    "orange",
    "purple",
    "white",
    "black",
)

# --- EDITABLE synonym map (fallback path only) ----------------------------
# Maps a raw (singularized) noun -> the canonical noun used by the fallback
# "<color> <noun>" scheme for labels outside color-sort's pick-object
# vocabulary. `block` is the canonical choice for the cube family; change
# the value side to repoint everything at once (e.g. to "cube"). Also used
# as an intermediate step so "a red cube" resolves to "redblock" before
# being handed to `normalize_object`.
_NOUN_SYNONYMS = {
    "cube": "block",
    "box": "block",
    "brick": "block",
    "square": "block",
    "cuboid": "block",
    "rectangle": "block",
    "triangle": "triangle",
    "pyramid": "triangle",
    "sphere": "ball",
    "circle": "ball",
    "cylinder": "cylinder",
    # components.constants.OBJECT_ALIASES already folds "mug" -> "cup"; kept
    # here too so the fallback "<color> <noun>" path (labels that don't
    # resolve through normalize_object) stays consistent if a color is
    # ever paired with "mug" (e.g. a hypothetical "blue mug").
    "mug": "cup",
}

# Articles / filler words stripped before tokenizing.
_ARTICLES = {"a", "an", "the", "some", "each", "distinct", "object", "objects"}

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


def _strip_punct(text: str) -> str:
    # Hyphens/underscores act as word separators ("orange-brick" -> "orange brick").
    text = text.replace("-", " ").replace("_", " ")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _singularize(word: str) -> str:
    """Minimal, deterministic plural stripper for the small vocab we see.

    Not a general English singularizer -- just enough for block/cube/box/
    brick/triangle/sphere/etc plurals, without false-singularizing words
    that already end in a "natural" s (e.g. leaves this alone if stripping
    would produce an empty or single-char string).
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes") or word.endswith("shes") or word.endswith("ches"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _color_from_label(label: str) -> str:
    """Same idea as zeroshot._color_from_label / vlm._color_from_label."""
    low = label.lower()
    for c in _KNOWN_COLORS:
        if c in low:
            return c
    return ""


def canonicalize_label(raw: Optional[str]) -> str:
    """Normalize a free-form detector/VLM label to color-sort's pick-object
    vocabulary (``red, yellow, can, cup, airpods, pen, bottle``) wherever
    possible, falling back to the generic ``"<color> <noun>"`` scheme for
    labels outside that vocabulary.

    Deterministic, offline, no network/model dependency. Empty/None input
    returns "".
    """
    if not raw:
        return ""

    cleaned = _strip_punct(str(raw).lower())
    if not cleaned:
        return ""

    # 1) Fast path: color-sort's own alias table already normalizes many
    #    whole phrases (normalize_object alnum-compacts internally), e.g.
    #    "soda can" -> "can", "Yellow Blocks!" -> "yellow", "mug" -> "cup",
    #    "red" -> "red".
    direct = normalize_object(cleaned)
    if direct:
        return direct

    # 2) Strip articles/filler, then retry the alias table on the
    #    remaining compacted phrase (handles "a red block" -> "redblock",
    #    "a distinct red object" -> "red").
    tokens = [t for t in cleaned.split(" ") if t and t not in _ARTICLES]
    if tokens:
        retry = normalize_object("".join(tokens))
        if retry:
            return retry

    # 3) Split color from noun, singularize + fold noun synonyms (block
    #    family, mug -> cup, etc.), then try the alias table again on
    #    "<color><noun>" (covers "a red cube": color=red, noun=cube->block
    #    -> "redblock" -> "red") and, if there's no noun at all, on the
    #    color alone.
    color = _color_from_label(cleaned)
    noun_tokens = [t for t in tokens if t not in _KNOWN_COLORS]
    noun = ""
    for tok in reversed(noun_tokens):  # last remaining word is usually the noun
        singular = _singularize(tok)
        noun = _NOUN_SYNONYMS.get(singular, singular)
        break

    if color and noun:
        retry2 = normalize_object(f"{color}{noun}")
        if retry2:
            return retry2

    if noun:
        retry3 = normalize_object(noun)
        if retry3:
            return retry3
    elif color:
        retry4 = normalize_object(color)
        if retry4:
            return retry4

    # 4) Not part of color-sort's pick vocabulary (e.g. "triangle", "ball",
    #    "cylinder", or a non-red/yellow color) -- fall back to the
    #    original "<color> <noun>" scheme so shape-only labels (from
    #    components.shapes' HSV classifier) still canonicalize sanely.
    if not noun:
        return color
    return f"{color} {noun}".strip()


def canonical_key(located_shape) -> str:
    """Set ``.canonical_label`` on a LocatedShape-like object and return it.

    Prefers the shape's own ``label``; falls back to ``color`` if the label
    carries no usable noun. Mutates and returns the canonical label so
    callers can do ``label = canonical_key(obj)``.
    """
    raw = getattr(located_shape, "label", "") or ""
    canonical = canonicalize_label(raw)
    if not canonical:
        color = getattr(located_shape, "color", "") or ""
        canonical = canonicalize_label(color)
    located_shape.canonical_label = canonical
    return canonical
