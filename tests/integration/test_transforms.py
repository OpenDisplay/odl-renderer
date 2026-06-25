"""End-to-end tests for generic rotation/mirror transforms via generate_image."""

import pytest
from PIL import ImageChops

from odl_renderer import generate_image


def _ink_bbox(image):
    """Bounding box of non-background (non-white) pixels."""
    from PIL import Image

    bg = Image.new("RGB", image.size, "white")
    return ImageChops.difference(image.convert("RGB"), bg).getbbox()


async def _render(element, size=(120, 120)):
    return await generate_image(size[0], size[1], [element], background="white")


@pytest.mark.asyncio
class TestVisibleStrings:
    async def test_string_false_hides(self):
        shown = await _render({"type": "text", "value": "Hi", "x": 10, "y": 10, "size": 24})
        hidden = await _render({"type": "text", "value": "Hi", "x": 10, "y": 10, "size": 24, "visible": "false"})
        assert _ink_bbox(shown) is not None
        assert _ink_bbox(hidden) is None

    async def test_string_true_shows(self):
        image = await _render({"type": "text", "value": "Hi", "x": 10, "y": 10, "size": 24, "visible": "true"})
        assert _ink_bbox(image) is not None


@pytest.mark.asyncio
class TestRotation:
    async def test_rotation_changes_layout(self):
        base = {"type": "text", "value": "Fg", "x": 40, "y": 45, "size": 40}
        plain = await _render(base)
        rotated = await _render({**base, "rotation": 90})
        assert _ink_bbox(plain) != _ink_bbox(rotated)

    async def test_rotation_zero_is_noop(self):
        base = {"type": "text", "value": "Fg", "x": 40, "y": 45, "size": 40}
        plain = await _render(base)
        rot0 = await _render({**base, "rotation": 0})
        assert ImageChops.difference(plain.convert("RGB"), rot0.convert("RGB")).getbbox() is None

    async def test_pivot_changes_result(self):
        base = {"type": "text", "value": "Fg", "x": 40, "y": 45, "size": 40, "rotation": 90}
        center = await _render(base)
        corner = await _render({**base, "pivot": "lt"})
        assert ImageChops.difference(center.convert("RGB"), corner.convert("RGB")).getbbox() is not None

    async def test_coordinate_pivot_renders(self):
        image = await _render(
            {"type": "text", "value": "Fg", "x": 40, "y": 45, "size": 40, "rotation": 90, "pivot": ["50%", "50%"]}
        )
        assert image.size == (120, 120)


@pytest.mark.asyncio
class TestMirror:
    async def test_horizontal_mirror_flips_content(self):
        base = {"type": "text", "value": "Rq", "x": 40, "y": 45, "size": 40}
        plain = await _render(base)
        mirrored = await _render({**base, "mirror": "h"})
        # Bounding box is preserved (flip about its own center) but content differs.
        assert ImageChops.difference(plain.convert("RGB"), mirrored.convert("RGB")).getbbox() is not None

    async def test_unknown_mirror_is_noop(self):
        base = {"type": "text", "value": "Rq", "x": 40, "y": 45, "size": 40}
        plain = await _render(base)
        weird = await _render({**base, "mirror": "diagonal"})
        assert ImageChops.difference(plain.convert("RGB"), weird.convert("RGB")).getbbox() is None
