"""
Type annotation integrity.

A string annotation that names something never imported is invisible at runtime:
the module loads, the tests pass, and only a tool that resolves types fails. That
is how `tuple[np.ndarray, "Skeleton"]` survived in features.py with no import
behind it - lint caught it on CI, but nothing caught it locally.

These tests fail if an annotation cannot be resolved, so the same defect cannot
return silently.
"""

from __future__ import annotations

import importlib
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# Modules whose annotations are part of their interface.
MODULES_WITH_ANNOTATED_FUNCTIONS = (
    "features",
    "skeleton",
    "mocap",
    "physics",
    "evaluate",
    "tracker",
    "filter",
    "predictor",
    "actions",
    "pose",
    "blazepose",
    "schema",
)


@pytest.mark.parametrize("module_name", MODULES_WITH_ANNOTATED_FUNCTIONS)
def test_module_annotations_resolve(module_name: str) -> None:
    """
    Every annotation in the module must resolve to a real object.

    typing.get_type_hints() raises NameError for an annotation naming something that
    was never imported, which is exactly the failure this guards.
    """
    module = importlib.import_module(module_name)

    for name in dir(module):
        attribute = getattr(module, name)
        if not (hasattr(attribute, "__annotations__") and callable(attribute)):
            continue
        if getattr(attribute, "__module__", None) != module_name:
            continue
        # A failure here names the function and the unresolved type.
        typing.get_type_hints(attribute)


def test_load_joint_positions_return_type_is_resolved() -> None:
    """The specific defect that reached CI: Skeleton was never imported."""
    import features

    hints = typing.get_type_hints(features.load_joint_positions)
    assert "Skeleton" in str(hints["return"])
