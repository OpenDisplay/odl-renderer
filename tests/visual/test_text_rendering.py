from io import BytesIO

import pytest

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestTextVisualRegression:
    """Visual regression tests for text rendering."""

    async def test_basic_text(self, snapshot_png, ppb_font):
        """Test basic text rendering matches snapshot."""
        image = await generate_image(
            width=200, height=100, elements=[E.text("Hello, World!", x=10, y=50, font=ppb_font, size=24, color="black")]
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_text_colors(self, snapshot_png, ppb_font):
        """Test text with various colors."""
        image = await generate_image(
            width=300,
            height=200,
            elements=[
                E.text("Red", x=10, y=20, font=ppb_font, size=20, color="red"),
                E.text("Blue", x=10, y=60, font=ppb_font, size=20, color="blue"),
                E.text("Green", x=10, y=100, font=ppb_font, size=20, color="green"),
                E.text("Black", x=10, y=140, font=ppb_font, size=20, color="black"),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    @pytest.mark.parametrize("size", [12, 16, 24, 32])
    async def test_text_sizes(self, snapshot_png, ppb_font, size):
        """Test text at various sizes."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[E.text(f"Size {size}", x=10, y=50, font=ppb_font, size=size, color="black")],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png
