"""Visual regression tests for rotation/mirror transforms.

Local-only (excluded from CI due to cross-platform font rendering). These lock
in the visual correctness of the geometric transforms, complementing the
behavioral assertions in tests/integration/test_transforms.py.
"""

from io import BytesIO

import pytest

from odl_renderer import generate_image


def _png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
class TestTransformVisualRegression:
    async def test_rotated_text(self, snapshot_png, ppb_font):
        image = await generate_image(
            width=120,
            height=120,
            background="white",
            elements=[
                {"type": "text", "value": "Rotate", "x": 60, "y": 60, "font": ppb_font, "size": 28, "rotation": 90},
            ],
        )
        assert _png_bytes(image) == snapshot_png

    async def test_rotated_text_pivot_top_left(self, snapshot_png, ppb_font):
        image = await generate_image(
            width=120,
            height=120,
            background="white",
            elements=[
                {
                    "type": "text",
                    "value": "Pivot",
                    "x": 60,
                    "y": 60,
                    "font": ppb_font,
                    "size": 28,
                    "rotation": 45,
                    "pivot": "tl",
                },
            ],
        )
        assert _png_bytes(image) == snapshot_png

    async def test_mirrored_polygon(self, snapshot_png):
        # Asymmetric L-shape makes the horizontal flip visually obvious.
        image = await generate_image(
            width=120,
            height=120,
            background="white",
            elements=[
                {
                    "type": "polygon",
                    "points": [[20, 20], [40, 20], [40, 80], [90, 80], [90, 100], [20, 100]],
                    "fill": "red",
                    "mirror": "h",
                },
            ],
        )
        assert _png_bytes(image) == snapshot_png
