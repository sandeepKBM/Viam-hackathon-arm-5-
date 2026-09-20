"""Claude integration -- STUB (final-stage only; deliberately NOT wired now).

Claude is the "final"/wow-demo upgrade, not the runtime default (the local
rule-based planner + local VLM carry the bulk; Claude is gated to the HARD
cases for cost/latency). This module is the seam so a future agent can fill in
the real Anthropic API / Viam MCP calls WITHOUT changing any call site.

Two seams, both mirroring the AudioIO stub pattern (subclass + fill one method):

1. ``ClaudePlanner`` -- a ``components.policy.Planner``:
   ``(goal, scene_view, menu) -> [{"skill", "params"}, ...]``. Drops straight
   into ``policy.plan_task(planner=ClaudePlanner())``. Fill ``_call_claude`` with
   an Anthropic API call (send goal + scene_view + menu; ask for a JSON array of
   ``{"skill","params"}`` in the shape the rule-based stub emits) or route through
   Viam MCP. ``policy.plan_task`` already validates/sandboxes whatever it returns.

2. ``ClaudeOracle.verify_label`` -- adjudicate an UNCERTAIN detection (one the UQ
   layer flagged and the local VLM couldn't resolve) by asking Claude vision to
   name it. Fill ``_call_claude_vision``.

No ``anthropic`` dependency is imported; this file imports fine offline. Until the
two ``_call_*`` methods are filled, they raise ``NotImplementedError`` with wiring
instructions -- an explicit seam, never a silent wrong answer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# The model id to use when wired. Latest capable Claude (see the Claude API docs).
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"


class ClaudePlanner:
    """Stub Claude planner conforming to ``components.policy.Planner``.

    Subclass and override ``_call_claude`` to enable it; the call site
    (``plan_task(planner=...)``) never changes."""

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL) -> None:
        self.model = model

    def _call_claude(self, prompt: str) -> str:
        """Return Claude's raw text (expected: a JSON array of {"skill","params"}).

        TODO(final): wire the Anthropic API here, e.g.
            from anthropic import Anthropic
            msg = Anthropic().messages.create(model=self.model, max_tokens=1024,
                      messages=[{"role":"user","content": prompt}])
            return msg.content[0].text
        or route the same prompt through Viam MCP tool calls.
        """
        raise NotImplementedError(
            "ClaudePlanner is a stub -- fill _call_claude with the Anthropic API "
            "or Viam MCP. Claude is the final-stage planner; the rule-based "
            "planner is the runtime default."
        )

    def _build_prompt(self, goal: str, scene_view: List[dict], menu: List[dict]) -> str:
        return (
            "You control a tabletop arm. Choose ONLY from the skill menu.\n"
            f"GOAL: {goal}\n"
            f"SKILLS: {json.dumps(menu)}\n"
            f"SCENE: {json.dumps(scene_view)}\n"
            'Reply with ONLY a JSON array of {"skill": <name>, "params": {...}} '
            "referencing objects by their object_id/target_id."
        )

    def __call__(
        self, goal: str, scene_view: List[dict], menu: List[dict]
    ) -> List[Dict[str, Any]]:
        raw = self._call_claude(self._build_prompt(goal, scene_view, menu))
        try:
            calls = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        return calls if isinstance(calls, list) else []


class ClaudeOracle:
    """Stub 'ask Claude to name the hard one' oracle for UQ-flagged detections.

    Subclass + override ``_call_claude_vision``; keep it UQ-gated so it's only
    the uncertain few, never every frame."""

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL) -> None:
        self.model = model

    def _call_claude_vision(self, image: Any, box: Any, candidate_labels: Any) -> str:
        """Return Claude's chosen label for the object in ``box`` of ``image``.

        TODO(final): Anthropic vision call with the cropped/annotated image and the
        candidate labels; return the single best label string.
        """
        raise NotImplementedError(
            "ClaudeOracle is a stub -- fill _call_claude_vision with an Anthropic "
            "vision call. Use it only for UQ-flagged uncertain detections."
        )

    def verify_label(
        self, image: Any, box: Any, candidate_labels: Optional[List[str]] = None
    ) -> str:
        return self._call_claude_vision(image, box, candidate_labels or [])
