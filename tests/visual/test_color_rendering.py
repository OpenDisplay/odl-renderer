from io import BytesIO

import pytest

from odl_renderer import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestGrayColorVisualRegression:
    """Visual regression tests for the grayscale ramp and gray text."""

    async def test_gray_ramp_named(self, snapshot_png):
        """Named grayscale ramp: black -> dkgray -> gray -> ltgray -> white."""
        levels = ["black", "dkgray", "gray", "ltgray", "white"]
        image = await generate_image(
            width=300,
            height=80,
            background="red",
            elements=[
                E.rectangle(
                    x_start=10 + i * 58,
                    y_start=10,
                    x_end=58 + i * 58,
                    y_end=70,
                    fill=level,
                    outline="black",
                    width=1,
                )
                for i, level in enumerate(levels)
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_gray_ramp_hex(self, snapshot_png):
        """16-gray addressed via hex grayscale values."""
        swatches = ["#000000", "#404040", "#808080", "#c0c0c0", "#ffffff"]
        image = await generate_image(
            width=300,
            height=80,
            background="red",
            elements=[
                E.rectangle(
                    x_start=10 + i * 58,
                    y_start=10,
                    x_end=58 + i * 58,
                    y_end=70,
                    fill=swatch,
                    outline="black",
                    width=1,
                )
                for i, swatch in enumerate(swatches)
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_gray_text_inline_markup(self, snapshot_png):
        """Inline [dkgray]/[ltgray] markup tags render with their gray levels."""
        image = await generate_image(
            width=300,
            height=80,
            background="white",
            elements=[
                E.text(
                    value="[dkgray]dark[/dkgray] [gray]mid[/gray] [ltgray]light[/ltgray]",
                    x=10,
                    y=20,
                    size=28,
                    parse_colors=True,
                ),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png
