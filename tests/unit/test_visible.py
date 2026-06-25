"""Unit tests for `visible` field coercion (core._coerce_visible)."""

import pytest

from odl_renderer.core import _coerce_visible


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Booleans pass through.
        (True, True),
        (False, False),
        # Templated string forms — the motivating case.
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        # Empty / whitespace-only strings hide (preserves the HA workaround).
        ("", False),
        ("   ", False),
        ("\n  \t", False),
        # Surrounding whitespace is stripped before comparison.
        ("  false  ", False),
        ("  true  ", True),
        # Narrowed set: only "false"/empty are falsy. Other words stay truthy
        # so a value that happens to be "no"/"0"/"none" is not silently hidden.
        ("no", True),
        ("off", True),
        ("0", True),
        ("none", True),
        ("null", True),
        ("1", True),
        ("yes", True),
        ("anything", True),
        # Numbers via bool().
        (0, False),
        (1, True),
        (2, True),
        (0.0, False),
        (2.5, True),
        # None hides.
        (None, False),
    ],
)
def test_coerce_visible(value, expected):
    assert _coerce_visible(value) is expected
