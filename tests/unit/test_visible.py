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
        # Common falsy string forms hide too — a template rendering {{ 0 }} yields
        # "0", and "no"/"off"/"none" read as hidden rather than silently shown.
        ("no", False),
        ("off", False),
        ("0", False),
        ("none", False),
        ("No", False),
        ("OFF", False),
        # Other words stay truthy.
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
