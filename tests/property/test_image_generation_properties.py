import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from drawcustom import generate_image


class TestImageGenerationProperties:
    """Property-based tests for image generation."""

    @given(width=st.integers(min_value=1, max_value=1000), height=st.integers(min_value=1, max_value=1000))
    @pytest.mark.asyncio
    async def test_generated_image_matches_dimensions(self, width, height):
        """Generated image always matches requested dimensions."""
        image = await generate_image(width=width, height=height, elements=[])

        assert isinstance(image, Image.Image)
        assert image.size == (width, height)
        assert image.width == width
        assert image.height == height

    @given(
        width=st.integers(min_value=1, max_value=500),
        height=st.integers(min_value=1, max_value=500),
        background=st.sampled_from(["white", "black", "red", "blue", "green", "#FF0000", "#000000"]),
    )
    @pytest.mark.asyncio
    async def test_background_color_accepted(self, width, height, background):
        """Any valid background color is accepted."""
        image = await generate_image(width=width, height=height, background=background, elements=[])

        assert isinstance(image, Image.Image)
        assert image.size == (width, height)

    @given(accent_color=st.sampled_from(["red", "yellow", "black"]))
    @pytest.mark.asyncio
    async def test_accent_color_parameter_accepted(self, accent_color):
        """Any valid accent color is accepted."""
        image = await generate_image(width=100, height=100, accent_color=accent_color, elements=[])

        assert isinstance(image, Image.Image)

    @given(
        width=st.integers(min_value=1, max_value=500),
        height=st.integers(min_value=1, max_value=500),
        num_elements=st.integers(min_value=0, max_value=20),
    )
    @pytest.mark.asyncio
    async def test_multiple_rectangles_render(self, width, height, num_elements):
        """Image generation works with varying numbers of elements."""
        elements = [
            {
                "type": "rectangle",
                "x_start": 0,
                "y_start": 0,
                "x_end": min(50, width),
                "y_end": min(50, height),
                "fill": "red",
            }
            for _ in range(num_elements)
        ]

        image = await generate_image(width=width, height=height, elements=elements)

        assert isinstance(image, Image.Image)
        assert image.size == (width, height)

    @given(width=st.integers(max_value=0), height=st.integers(min_value=1, max_value=100))
    @pytest.mark.asyncio
    async def test_invalid_width_raises_error(self, width, height):
        """Invalid width (<=0) always raises ValueError."""
        with pytest.raises(ValueError, match="Invalid canvas dimensions|width"):
            await generate_image(width=width, height=height, elements=[])

    @given(width=st.integers(min_value=1, max_value=100), height=st.integers(max_value=0))
    @pytest.mark.asyncio
    async def test_invalid_height_raises_error(self, width, height):
        """Invalid height (<=0) always raises ValueError."""
        with pytest.raises(ValueError, match="Invalid canvas dimensions|height"):
            await generate_image(width=width, height=height, elements=[])
