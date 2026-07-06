"""Regression tests for finding A8: numeric element fields must tolerate string
input (Home Assistant templates render everything to strings) and, where
sensible, percentage strings."""

import pytest

from odl_renderer import generate_image


@pytest.mark.asyncio
class TestNumericStringCoercion:
    async def test_progress_bar_string_progress(self):
        """progress_bar.progress as a string does not crash and clamps correctly."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "progress_bar",
                    "x_start": 10,
                    "y_start": 10,
                    "x_end": 190,
                    "y_end": 40,
                    "progress": "50",
                    "show_percentage": True,
                }
            ],
        )
        assert image.size == (200, 100)

    async def test_multiline_non_string_value(self):
        """multiline.value given a number is coerced to str instead of crashing."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "multiline",
                    "x": 10,
                    "y": 10,
                    "value": 123,
                    "delimiter": "|",
                    "offset_y": "20",
                    "font": "ppb",
                    "size": "16",
                }
            ],
        )
        assert image.size == (200, 100)

    async def test_icon_percentage_size(self):
        """icon.size as a percentage string resolves instead of crashing."""
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "icon", "value": "home", "x": 50, "y": 50, "size": "50%"}],
        )
        assert image.size == (100, 100)

    async def test_rectangle_pattern_percentage_start(self):
        """rectangle_pattern geometry accepts percentage/number strings."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "rectangle_pattern",
                    "x_start": "10%",
                    "y_start": "10%",
                    "x_size": "10",
                    "y_size": "10",
                    "x_offset": "5",
                    "y_offset": "5",
                    "x_repeat": "3",
                    "y_repeat": "3",
                    "fill": "black",
                }
            ],
        )
        assert image.size == (200, 100)

    async def test_circle_percentage_radius(self):
        """circle.radius accepts a percentage string (parity with arc.radius)."""
        image = await generate_image(
            width=200,
            height=200,
            elements=[{"type": "circle", "x": 100, "y": 100, "radius": "10%", "fill": "black"}],
        )
        assert image.size == (200, 200)

    async def test_line_string_width(self):
        """line.width as a string does not crash."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[{"type": "line", "x_start": 0, "y_start": 50, "x_end": 200, "width": "2"}],
        )
        assert image.size == (200, 100)

    async def test_text_string_dimensions(self):
        """text stroke_width / y_padding / max_width as strings do not crash."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "text",
                    "x": 10,
                    "value": "Hello world this is a fairly long line",
                    "font": "ppb",
                    "size": "16",
                    "y_padding": "10",
                    "stroke_width": "1",
                    "max_width": "150",
                }
            ],
        )
        assert image.size == (200, 100)

    async def test_dlimg_string_size(self):
        """dlimg xsize/ysize as strings are parsed (URL load failure is separate)."""
        # A bad URL raises during load; we only assert the size parsing path does
        # not raise a TypeError first. Use a data: URL that decodes to a 1x1 image.
        one_by_one_png = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "dlimg", "x": 10, "y": 10, "url": one_by_one_png, "xsize": "50", "ysize": "50"}],
        )
        assert image.size == (100, 100)
