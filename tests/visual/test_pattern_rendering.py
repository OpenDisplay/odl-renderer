from io import BytesIO

import pytest

from odl_renderer import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestPatternVisualRegression:
    """Visual regression tests for rectangle patterns (no text, so no font dependence)."""

    async def test_pattern_types(self, snapshot_png):
        """Hatch at several angles, dots, a grid and tones, each in its own rectangle."""
        patterns = [
            "hatch",
            {"type": "hatch", "angle": 135},
            {"type": "hatch", "angle": 0, "spacing": 5},
            {"type": "hatch", "angle": 90, "spacing": 5},
            {"type": "hatch", "angle": 30, "spacing": 7},
            {"type": "dots", "spacing": 5, "width": 2},
            {"type": "grid", "spacing": 8, "color": "red"},
            {"type": "tone", "level": 25},
            {"type": "tone", "level": 50},
            {"type": "tone", "level": 75},
        ]
        elements = [
            E.rectangle(
                10 + (index % 5) * 58,
                10 + (index // 5) * 58,
                58 + (index % 5) * 58 - 6,
                58 + (index // 5) * 58 - 6,
                pattern=pattern,
            )
            for index, pattern in enumerate(patterns)
        ]
        image = await generate_image(width=300, height=130, background="white", elements=elements)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_pattern_with_fill_outline_and_rounded_corners(self, snapshot_png):
        """The fill shows between the lines, the pattern stays inside the corners, the outline is on top."""
        image = await generate_image(
            width=300,
            height=100,
            background="white",
            elements=[
                E.rectangle(
                    10,
                    10,
                    140,
                    90,
                    fill="yellow",
                    outline="black",
                    width=3,
                    radius=20,
                    corners="all",
                    pattern={"type": "hatch", "color": "red", "width": 2, "spacing": 8},
                ),
                E.rectangle(
                    160,
                    10,
                    290,
                    90,
                    outline=None,
                    radius=30,
                    corners="top_left,bottom_right",
                    pattern={"type": "tone", "level": 40},
                ),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png
