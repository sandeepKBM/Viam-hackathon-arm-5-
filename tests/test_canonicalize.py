"""Offline unit tests for components/canonicalize.py (W2).

Pure string normalization -- no cv2/numpy/model dependency required, but we
also exercise ``canonical_key`` against a tiny stand-in object so we don't
need to import components.shapes (which pulls in cv2).
"""

from dataclasses import dataclass

from components.canonicalize import canonical_key, canonicalize_label


class TestCanonicalizeLabel:
    def test_simple_color_noun(self):
        assert canonicalize_label("red block") == "red block"

    def test_article_stripped(self):
        assert canonicalize_label("a red cube") == "red block"

    def test_leading_article_the(self):
        assert canonicalize_label("the blue brick") == "blue block"

    def test_punctuation_and_case(self):
        assert canonicalize_label("Yellow Blocks!") == "yellow block"

    def test_hyphenated(self):
        assert canonicalize_label("orange-brick") == "orange block"

    def test_plural_cubes(self):
        assert canonicalize_label("cubes") == "block"

    def test_plural_boxes(self):
        assert canonicalize_label("boxes") == "block"

    def test_synonym_box_to_block(self):
        assert canonicalize_label("green box") == "green block"

    def test_synonym_cube_to_block(self):
        assert canonicalize_label("purple cube") == "purple block"

    def test_triangle_passthrough(self):
        assert canonicalize_label("red triangle") == "red triangle"

    def test_pyramid_synonym(self):
        assert canonicalize_label("blue pyramid") == "blue triangle"

    def test_sphere_synonym(self):
        assert canonicalize_label("white sphere") == "white ball"

    def test_unknown_color_defaults_empty_prefix(self):
        # No known color word -> just the normalized noun (no leading space).
        assert canonicalize_label("block") == "block"

    def test_color_only_input(self):
        assert canonicalize_label("red") == "red"

    def test_empty_and_none(self):
        assert canonicalize_label("") == ""
        assert canonicalize_label(None) == ""

    def test_idempotent(self):
        once = canonicalize_label("a red cube")
        twice = canonicalize_label(once)
        assert once == twice == "red block"

    def test_whitespace_and_multi_word_noise(self):
        assert canonicalize_label("  the   Yellow   Block  ") == "yellow block"

    def test_distinct_object_filler(self):
        assert canonicalize_label("a distinct red object") == "red"


class TestCanonicalKey:
    @dataclass
    class FakeLocated:
        label: str
        color: str = ""
        canonical_label: str = ""

    def test_sets_canonical_label_from_label(self):
        obj = self.FakeLocated(label="a red cube")
        result = canonical_key(obj)
        assert result == "red block"
        assert obj.canonical_label == "red block"

    def test_falls_back_to_color_when_label_has_no_noun(self):
        obj = self.FakeLocated(label="", color="red")
        result = canonical_key(obj)
        assert result == "red"
        assert obj.canonical_label == "red"

    def test_zeroshot_style_label(self):
        # zeroshot.py / vlm.py set DetectedShape.label from the raw prompt.
        obj = self.FakeLocated(label="yellow block", color="yellow")
        assert canonical_key(obj) == "yellow block"
