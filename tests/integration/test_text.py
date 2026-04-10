import pytest

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestTextRendering:
    """Test text element rendering."""

    async def test_basic_text(self, ppb_font):
        """Test basic text rendering."""
        image = await generate_image(
            width=200, height=100, elements=[E.text("Hello World", x=10, y=10, font=ppb_font, size=16)]
        )

        assert image.size == (200, 100)

    @pytest.mark.parametrize(
        "text_value",
        [
            "Simple",
            "With Numbers 123",
            "Special!@#$%",
            "émojis and ñ",
        ],
    )
    async def test_various_text_content(self, ppb_font, text_value):
        """Test text with various content."""
        image = await generate_image(
            width=200, height=100, elements=[E.text(text_value, x=10, y=10, font=ppb_font, size=16)]
        )

        assert image.size == (200, 100)

    @pytest.mark.parametrize("size", [8, 12, 16, 24, 32, 48])
    async def test_various_text_sizes(self, ppb_font, size):
        """Test text at various sizes."""
        image = await generate_image(
            width=300, height=200, elements=[E.text("Test", x=10, y=10, font=ppb_font, size=size)]
        )

        assert image.size == (300, 200)

    @pytest.mark.parametrize("color", ["black", "red", "blue", "#FF0000"])
    async def test_text_colors(self, ppb_font, color):
        """Test text with various colors."""
        image = await generate_image(
            width=200, height=100, elements=[E.text("Test", x=10, y=10, font=ppb_font, size=16, color=color)]
        )

        assert image.size == (200, 100)

    async def test_empty_text(self, ppb_font):
        """Test rendering empty text doesn't crash."""
        image = await generate_image(width=200, height=100, elements=[E.text("", x=10, y=10, font=ppb_font, size=16)])

        assert image.size == (200, 100)
