import pytest

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestRectangleRendering:
    """Test rectangle element rendering."""

    @pytest.mark.parametrize(
        "color,expected_rgb",
        [
            ("black", (0, 0, 0)),
            ("white", (255, 255, 255)),
            ("red", (255, 0, 0)),
            ("yellow", (255, 255, 0)),
            ("#FF0000", (255, 0, 0)),
            ("#00FF00", (0, 255, 0)),
            ("#0000FF", (0, 0, 255)),
            ("#F00", (255, 0, 0)),
        ],
    )
    async def test_filled_rectangle_colors(self, color, expected_rgb):
        """Test filled rectangle with various color formats."""
        image = await generate_image(
            width=100, height=100, elements=[E.rectangle(x_start=0, y_start=0, x_end=100, y_end=100, fill=color)]
        )

        # Sample center pixel
        pixel = image.getpixel((50, 50))
        assert pixel[:3] == expected_rgb

    async def test_outlined_rectangle(self):
        """Test rectangle with outline but no fill."""
        image = await generate_image(
            width=100,
            height=100,
            background="white",
            elements=[E.rectangle(x_start=20, y_start=20, x_end=80, y_end=80, outline="black", width=2)],
        )

        # Outline pixel should be black
        outline_pixel = image.getpixel((20, 20))
        assert outline_pixel[:3] == (0, 0, 0)

        # Inside should be white (background)
        inside_pixel = image.getpixel((50, 50))
        assert inside_pixel[:3] == (255, 255, 255)

    @pytest.mark.parametrize(
        "x_start,y_start,x_end,y_end",
        [
            (0, 0, 50, 50),
            (25, 25, 75, 75),
            (10, 20, 90, 80),
        ],
    )
    async def test_rectangle_positions(self, x_start, y_start, x_end, y_end):
        """Test rectangles at various positions."""
        image = await generate_image(
            width=100,
            height=100,
            elements=[E.rectangle(x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end, fill="black")],
        )

        # Center of rectangle should be black
        center_x = (x_start + x_end) // 2
        center_y = (y_start + y_end) // 2
        pixel = image.getpixel((center_x, center_y))
        assert pixel[:3] == (0, 0, 0)

    async def test_overlapping_rectangles(self):
        """Test multiple overlapping rectangles render in order."""
        image = await generate_image(
            width=200,
            height=200,
            elements=[
                E.rectangle(x_start=0, y_start=0, x_end=100, y_end=100, fill="red"),
                E.rectangle(x_start=50, y_start=50, x_end=150, y_end=150, fill="blue"),
            ],
        )

        # First rectangle area (not overlapped) should be red
        assert image.getpixel((25, 25))[:3] == (255, 0, 0)

        # Overlapped area should be blue (drawn second)
        assert image.getpixel((75, 75))[:3] == (0, 0, 255)

        # Second rectangle only area should be blue
        assert image.getpixel((125, 125))[:3] == (0, 0, 255)

    async def test_rectangle_debug_multiple(self):
        """Debug test to see if multiple rectangles render."""
        import logging

        logging.basicConfig(level=logging.DEBUG)

        image = await generate_image(
            width=200,
            height=200,
            background="white",
            elements=[
                E.rectangle(x_start=0, y_start=0, x_end=100, y_end=100, fill="red"),
                E.rectangle(x_start=50, y_start=50, x_end=150, y_end=150, fill="blue"),
            ],
        )

        # Save for visual inspection
        # image.save("debug_multi_rect.png")

        # Sample multiple points
        print(f"Point (25, 25) - should be red: {image.getpixel((25, 25))}")
        print(f"Point (75, 75) - should be blue: {image.getpixel((75, 75))}")
        print(f"Point (125, 125) - should be blue: {image.getpixel((125, 125))}")

        assert image.size == (200, 200)
