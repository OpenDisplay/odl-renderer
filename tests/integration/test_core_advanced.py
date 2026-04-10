"""Tests for core.py branches not covered by other tests."""

import pytest

from drawcustom import generate_image
from drawcustom.core import should_show_element


@pytest.mark.asyncio
class TestVisibility:
    async def test_hidden_element_skipped(self):
        """Element with visible=False is skipped without error."""
        image = await generate_image(
            width=100,
            height=100,
            elements=[
                {"type": "text", "x": 10, "y": 10, "value": "visible", "font": "ppb", "size": 16},
                {"type": "text", "x": 10, "y": 30, "value": "hidden", "font": "ppb", "size": 16, "visible": False},
            ],
        )
        assert image.size == (100, 100)

    async def test_visible_true_explicit(self):
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "text", "x": 10, "y": 10, "value": "shown", "font": "ppb", "size": 16, "visible": True}],
        )
        assert image.size == (100, 100)


class TestShouldShowElement:
    def test_default_visible(self):
        assert should_show_element({}) is True

    def test_visible_true(self):
        assert should_show_element({"visible": True}) is True

    def test_visible_false(self):
        assert should_show_element({"visible": False}) is False


@pytest.mark.asyncio
class TestErrorHandling:
    async def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing required"):
            await generate_image(width=100, height=100, elements=[{"x": 10, "y": 10}])

    async def test_handler_generic_exception_wrapped(self):
        """A handler raising a non-ValueError/KeyError is caught and re-raised as ValueError."""
        with pytest.raises(ValueError, match="Element 1"):
            # dlimg with an invalid (relative) url will trigger a generic error path
            await generate_image(
                width=100,
                height=100,
                elements=[{"type": "dlimg", "x": 0, "y": 0, "url": "relative/bad.png", "xsize": 10, "ysize": 10}],
            )
