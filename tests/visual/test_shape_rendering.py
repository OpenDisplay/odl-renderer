from io import BytesIO

import pytest

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestShapeVisualRegression:
    """Visual regression tests for shape rendering."""

    async def test_rectangles_filled(self, snapshot_png):
        """Test filled rectangles."""
        image = await generate_image(
            width=300,
            height=200,
            elements=[
                E.rectangle(x_start=10, y_start=10, x_end=90, y_end=90, fill="red"),
                E.rectangle(x_start=110, y_start=10, x_end=190, y_end=90, fill="blue"),
                E.rectangle(x_start=210, y_start=10, x_end=290, y_end=90, fill="green"),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_rectangles_outlined(self, snapshot_png):
        """Test outlined rectangles."""
        image = await generate_image(
            width=300,
            height=200,
            background="white",
            elements=[
                E.rectangle(x_start=10, y_start=10, x_end=90, y_end=90, outline="black", width=2),
                E.rectangle(x_start=110, y_start=10, x_end=190, y_end=90, outline="red", width=3),
                E.rectangle(x_start=210, y_start=10, x_end=290, y_end=90, outline="blue", width=4),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_circles(self, snapshot_png):
        """Test circle rendering."""
        image = await generate_image(
            width=300,
            height=200,
            elements=[
                E.circle(x=60, y=60, radius=40, fill="red"),
                E.circle(x=150, y=60, radius=40, fill="blue"),
                E.circle(x=240, y=60, radius=40, fill="green"),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_lines(self, snapshot_png):
        """Test line rendering."""
        image = await generate_image(
            width=200,
            height=200,
            background="white",
            elements=[
                E.line(x_start=10, y_start=10, x_end=190, y_end=10, color="black", width=1),
                E.line(x_start=10, y_start=50, x_end=190, y_end=50, color="red", width=2),
                E.line(x_start=10, y_start=100, x_end=190, y_end=100, color="blue", width=4),
                E.line(x_start=10, y_start=150, x_end=190, y_end=150, color="green", width=8),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png
