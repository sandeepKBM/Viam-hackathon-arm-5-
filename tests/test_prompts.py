"""Offline tests for components/prompts.py (tuned detection prompts, W2/W3
follow-on) and for vlm.py / zeroshot.py picking up its defaults.

Pure string checks + module-attribute checks -- no cv2/torch/network
required for the prompts.py checks themselves; importing vlm.py/zeroshot.py
still only touches their lazy-loaded heavy deps if a detector is actually
instantiated (not done here).
"""

import importlib

from components.canonicalize import canonicalize_label
from components.prompts import OBJECT_VOCAB, VLM_LIST_PROMPT, ZS_PROMPTS


class TestObjectVocab:
    def test_every_entry_round_trips_through_canonicalize(self):
        # Every OBJECT_VOCAB label must already be in canonical form -- a
        # detector/VLM label that lands on one of these should key the
        # experience store the same way every time.
        for label in OBJECT_VOCAB:
            assert canonicalize_label(label) == label

    def test_covers_the_real_tabletop_set(self):
        assert set(OBJECT_VOCAB) == {"red block", "yellow block", "cup", "pen", "can"}


class TestZsPromptsCanonicalizeCleanly:
    def test_every_candidate_label_collapses_onto_the_vocab(self):
        # ZS_PROMPTS is intentionally a bit wider than OBJECT_VOCAB (extra
        # phrasings like "mug"/"soda can" to help OWLv2 recall); every one
        # of them must still canonicalize onto a vocab entry.
        for prompt in ZS_PROMPTS:
            assert canonicalize_label(prompt) in OBJECT_VOCAB

    def test_mug_and_soda_can_fold_onto_cup_and_can(self):
        assert canonicalize_label("mug") == "cup"
        assert canonicalize_label("soda can") == "can"


class TestVlmListPromptShape:
    def test_mentions_every_vocab_noun(self):
        low = VLM_LIST_PROMPT.lower()
        for noun in ("block", "cup", "pen", "can"):
            assert noun in low

    def test_asks_for_comma_separated_color_noun_phrases(self):
        low = VLM_LIST_PROMPT.lower()
        assert "comma" in low
        assert "<color>" in low or "color" in low


class TestModulesPickUpNewDefaults:
    def test_vlm_module_default_matches_prompts(self):
        from components import vlm

        assert vlm.VLM_LIST_PROMPT == VLM_LIST_PROMPT

    def test_zeroshot_module_default_matches_prompts(self):
        from components import zeroshot

        assert tuple(zeroshot.ZS_PROMPTS) == tuple(ZS_PROMPTS)

    def test_zeroshot_default_prompts_are_object_vocab_superset(self):
        from components import zeroshot

        for prompt in zeroshot.ZS_PROMPTS:
            assert canonicalize_label(prompt) in OBJECT_VOCAB

    def test_env_override_still_wins_for_zeroshot(self, monkeypatch):
        monkeypatch.setenv("ZS_PROMPTS", "green block,blue block")
        from components import zeroshot

        importlib.reload(zeroshot)
        try:
            assert zeroshot.ZS_PROMPTS == ("green block", "blue block")
        finally:
            monkeypatch.delenv("ZS_PROMPTS", raising=False)
            importlib.reload(zeroshot)  # restore the module-level default for later tests

    def test_env_override_still_wins_for_vlm(self, monkeypatch):
        monkeypatch.setenv("VLM_LIST_PROMPT", "custom prompt")
        from components import vlm

        importlib.reload(vlm)
        try:
            assert vlm.VLM_LIST_PROMPT == "custom prompt"
        finally:
            monkeypatch.delenv("VLM_LIST_PROMPT", raising=False)
            importlib.reload(vlm)  # restore the module-level default for later tests
