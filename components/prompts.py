"""Curated object vocabulary + tuned detection prompts (W2/W3 follow-on).

`vlm.py` (Moondream "namer") and `zeroshot.py` (OWLv2 localizer) each shipped
with a generic default prompt/candidate-label set tuned only for the two
sorting blocks. This module is the single source of truth for the REAL
tabletop set this stack targets -- red block, yellow block, cup/mug, pen,
soda can -- so both detectors agree on the same vocabulary and their raw
labels collapse cleanly onto `components.canonicalize.canonicalize_label`'s
output space (which is what keys `components.experience_store` and drives
`components.constants.COLOR_BINS` routing).

Pure, dependency-free (no cv2/torch/network) so importing it never pulls in
the heavy VLM/zero-shot stacks -- safe for both `vlm.py`/`zeroshot.py` (lazy
heavy deps) and offline tests to import unconditionally.

Design notes
------------
- `OBJECT_VOCAB` is the canonical target vocabulary: every entry is a FIXED
  POINT of `canonicalize_label` (`canonicalize_label(v) == v`), verified by
  `tests/test_prompts.py`. That's what lets a raw VLM/OWLv2 label collapse
  onto exactly these keys instead of drifting.
- `cup`/`mug` need to key the SAME experience-store bucket, so
  `components/canonicalize.py`'s `_NOUN_SYNONYMS` maps `"mug" -> "cup"`
  (edited alongside this module). `"soda can"` already collapses to `"can"`
  under the existing noun-extraction rule (last non-color token wins), so no
  further synonym edit was needed there.
- `VLM_LIST_PROMPT` explicitly anchors Moondream's free-form listing to this
  noun set so its output canonicalizes cleanly instead of drifting into
  synonyms canonicalize.py doesn't know about.
- `ZS_PROMPTS` is deliberately a bit wider than `OBJECT_VOCAB` (includes
  "mug" and "soda can" alongside "cup" and "can") -- OWLv2 is a zero-shot
  detector, so offering a couple of natural phrasings per object improves
  recall; canonicalize.py folds them back onto the same vocabulary either
  way.

Both `components/vlm.py` and `components/zeroshot.py` import their defaults
from here and still honor their own env-var overrides
(`VLM_LIST_PROMPT`, `ZS_PROMPTS`) unchanged.
"""

from typing import Tuple

# --- canonical target vocabulary -------------------------------------------
# The real tabletop set: two sorting blocks + three "clutter" objects the
# zero-shot/VLM path should also be able to name and localize. Every entry
# must equal canonicalize_label(entry) -- see tests/test_prompts.py.
OBJECT_VOCAB: Tuple[str, ...] = (
    "red block",
    "yellow block",
    "cup",
    "pen",
    "can",
)

# --- Moondream (VLM) object-listing prompt ----------------------------------
# Asks for exactly the "<color> <noun>" shape canonicalize.py expects, and
# anchors the noun choices to OBJECT_VOCAB so free-form answers don't drift.
VLM_LIST_PROMPT: str = (
    "List each distinct object on the table as a short phrase in the form "
    "'<color> <noun>'. Use only these nouns: block, cup, pen, can. Include "
    "the color for block; for cup/pen/can only include a color if it's "
    "obvious. For example: 'red block, yellow block, cup, pen, can'. "
    "Comma-separated, no numbering, no extra words."
)

# --- OWLv2 (zero-shot) candidate labels -------------------------------------
# A couple of natural phrasings per object (e.g. both "cup" and "mug") to
# help OWLv2's recall; canonicalize.py folds every variant back onto
# OBJECT_VOCAB (see module docstring).
ZS_PROMPTS: Tuple[str, ...] = (
    "red block",
    "yellow block",
    "cup",
    "mug",
    "pen",
    "soda can",
    "can",
)
