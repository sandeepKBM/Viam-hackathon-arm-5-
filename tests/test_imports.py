"""Prove the arm5 package skeleton is import-clean.

This intentionally does not test behavior (most modules are stubs) -- it
just imports every submodule so a broken import path fails fast in CI.
"""

import importlib

import pytest

MODULES = [
    "arm5",
    "arm5.vision",
    "arm5.vision.camera",
    "arm5.vision.detectors",
    "arm5.vision.perception",
    "arm5.hardware",
    "arm5.hardware.robot",
    "arm5.hardware.arm",
    "arm5.hardware.gripper",
    "arm5.controls",
    "arm5.controls.controllers",
    "arm5.controls.safety",
    "arm5.planning",
    "arm5.planning.base",
    "arm5.planning.classical",
    "arm5.planning.classical.motion",
    "arm5.planning.learned",
    "arm5.planning.learned.policy",
    "arm5.planning.soft_logic",
    "arm5.planning.soft_logic.rules",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Each listed module should import without raising."""
    importlib.import_module(module_name)
