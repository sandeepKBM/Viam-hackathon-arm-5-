"""Canonical VLM vocabulary (W2).

Free-form labels from the VLM ("a red cube", "Yellow Blocks!", "orange-brick")
or the zero-shot detector need to collapse onto a small, stable vocabulary
before they can key the experience store or be compared across detectors.
This module is pure and deterministic: no network calls, no model
dependency, just string normalization + a small editable synonym table.

Output shape is always ``"<color> <noun>"`` (color omitted if unknown), e.g.:

    "a red cube"      -> "red cube"
    "Yellow Blocks!"   -> "yellow block"
    "boxes"            -> "box"
    "the blue brick"   -> "blue block"   (brick is a synonym of block, see
                                            _NOUN_SYNONYMS)
"""

import re
from typing import Optional

# Reuse the same color vocabulary as zeroshot.py / vlm.py (kept in sync
# manually since this module must not import either -- both of those import
# from components.shapes / are heavier; canonicalize.py stays dependency-free).
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

# --- EDITABLE synonym map -------------------------------------------------
# Maps a raw (singularized) noun -> the canonical noun the rest of the
# system should use. Add new nouns here as the VLM's vocabulary drifts.
# `block` is the canonical choice for the cube family; change the value
# side to repoint everything at once (e.g. to "cube").
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
    # components/prompts.py's OBJECT_VOCAB uses "cup" as the canonical noun
    # for the cup/mug tabletop object; fold "mug" onto it so both detector
    # phrasings key the same components.experience_store bucket.
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
    """Normalize a free-form detector/VLM label to ``"<color> <noun>"``.

    Deterministic, offline, no network/model dependency. Empty/None input
    returns "".
    """
    if not raw:
        return ""

    cleaned = _strip_punct(raw.lower())
    if not cleaned:
        return ""

    color = _color_from_label(cleaned)

    tokens = [t for t in cleaned.split(" ") if t and t not in _ARTICLES]
    # Drop the color word(s) themselves from the noun search -- they've
    # already been captured in `color`.
    noun_tokens = [t for t in tokens if t not in _KNOWN_COLORS]

    noun = ""
    for tok in reversed(noun_tokens):  # last remaining word is usually the noun
        singular = _singularize(tok)
        noun = _NOUN_SYNONYMS.get(singular, singular)
        break

    if not noun:
        # Nothing left after stripping color/articles (e.g. raw == "red").
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
