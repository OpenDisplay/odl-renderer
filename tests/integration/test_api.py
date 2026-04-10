from io import BytesIO

import pytest
from PIL import Image

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestGenerateImageAPI:
    """Test the main generate_image() API."""

    async def test_basic_image_generation(self):
        """Test basic image generation returns PIL Image."""
        image = await generate_image(
            width=100, height=100, elements=[E.rectangle(x_start=10, y_start=10, x_end=90, y_end=90, fill="red")]
        )

        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)
        assert image.mode in ("RGB", "RGBA")

    async def test_empty_elements_list(self):
        """Test generation with no elements creates blank canvas."""
        image = await generate_image(width=200, height=100, elements=[])

        assert image.size == (200, 100)
        # Should be all white (default background)
        pixel = image.getpixel((100, 50))
        assert pixel[:3] == (255, 255, 255)

    @pytest.mark.parametrize(
        "width,height",
        [
            (100, 100),
            (296, 128),
            (400, 300),
            (800, 600),
        ],
    )
    async def test_various_dimensions(self, width, height):
        """Test generation with various canvas dimensions."""
        image = await generate_image(width=width, height=height, elements=[])
        assert image.size == (width, height)

    @pytest.mark.parametrize(
        "invalid_width,invalid_height",
        [
            (0, 100),
            (100, 0),
            (-10, 100),
            (100, -10),
            (0, 0),
        ],
    )
    async def test_invalid_dimensions_raise_error(self, invalid_width, invalid_height):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Invalid canvas dimensions|width|height"):
            await generate_image(width=invalid_width, height=invalid_height, elements=[])

    @pytest.mark.parametrize(
        "background_color",
        [
            "white",
            "black",
            "red",
            "#FF0000",
            "#000",
        ],
    )
    async def test_background_colors(self, background_color):
        """Test various background color formats."""
        image = await generate_image(width=100, height=100, background=background_color, elements=[])

        assert image.size == (100, 100)
        # Just verify it doesn't crash - pixel comparison depends on color resolution

    @pytest.mark.parametrize("accent_color", ["red", "yellow", "black"])
    async def test_accent_color_parameter(self, accent_color):
        """Test accent_color parameter is accepted."""
        image = await generate_image(
            width=100, height=100, accent_color=accent_color, elements=[E.rectangle(fill="accent")]
        )

        assert image.size == (100, 100)

    async def test_missing_element_type_raises_error(self):
        """Test element without 'type' field raises ValueError."""
        with pytest.raises(ValueError, match="type|element"):
            await generate_image(
                width=100,
                height=100,
                elements=[{"x": 10, "y": 10}],  # Missing "type"
            )

    async def test_unknown_element_type_raises_error(self):
        """Test unknown element type raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid ElementType|Element"):
            await generate_image(width=100, height=100, elements=[{"type": "nonexistent_element"}])

    async def test_multiple_elements(self):
        """Test rendering multiple elements in one image."""
        image = await generate_image(
            width=300,
            height=200,
            elements=[
                E.rectangle(x_start=10, y_start=10, x_end=90, y_end=90, fill="red"),
                E.text("Test", x=150, y=50, font="ppb", size=16),
                E.qrcode("test", x=250, y=50, size=40),
            ],
        )

        assert image.size == (300, 200)

    async def test_image_can_be_saved(self):
        """Test generated image can be saved to bytes."""
        image = await generate_image(width=100, height=100, elements=[E.text("Test")])

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert len(buffer.getvalue()) > 0
