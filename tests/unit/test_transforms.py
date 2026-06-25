"""Unit tests for the generic transform module."""

from PIL import Image, ImageChops

from odl_renderer.coordinates import CoordinateParser
from odl_renderer.transforms import (
    _anchor_point,
    _resolve_pivot,
    apply_transform,
    has_transform,
)


def _layer_with_box(size=(100, 100), box=(10, 10, 40, 30), color=(0, 0, 0, 255)):
    """Transparent RGBA layer with one opaque rectangle drawn on it."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(layer).rectangle(box, fill=color)
    return layer


class TestHasTransform:
    def test_none(self):
        assert has_transform({"type": "text"}) is False

    def test_zero_rotation_is_noop(self):
        assert has_transform({"rotation": 0}) is False

    def test_rotation(self):
        assert has_transform({"rotation": 90}) is True

    def test_mirror(self):
        assert has_transform({"mirror": "h"}) is True


class TestAnchorPoint:
    # Pillow text format: first char horizontal (l/m/r), second vertical (t/m/b).
    def test_center_default(self):
        assert _anchor_point("mm", (10, 20, 30, 40)) == (20.0, 30.0)

    def test_top_left(self):
        assert _anchor_point("lt", (10, 20, 30, 40)) == (10.0, 20.0)

    def test_bottom_right(self):
        assert _anchor_point("rb", (10, 20, 30, 40)) == (30.0, 40.0)

    def test_middle_top(self):
        # Regression: the 'm' in "mt" must not collapse to center.
        assert _anchor_point("mt", (10, 20, 30, 40)) == (20.0, 20.0)

    def test_middle_bottom(self):
        assert _anchor_point("mb", (10, 20, 30, 40)) == (20.0, 40.0)

    def test_left_middle(self):
        assert _anchor_point("lm", (10, 20, 30, 40)) == (10.0, 30.0)

    def test_empty_and_invalid_fall_back_to_center(self):
        assert _anchor_point("", (10, 20, 30, 40)) == (20.0, 30.0)
        assert _anchor_point("??", (10, 20, 30, 40)) == (20.0, 30.0)


class TestResolvePivot:
    def test_default_is_bbox_center(self):
        assert _resolve_pivot(None, (0, 0, 100, 50), None) == (50.0, 25.0)

    def test_anchor_keyword(self):
        assert _resolve_pivot("lt", (0, 0, 100, 50), None) == (0.0, 0.0)

    def test_coord_pair_pixels(self):
        coords = CoordinateParser(200, 100)
        assert _resolve_pivot([20, 30], (0, 0, 10, 10), coords) == (20.0, 30.0)

    def test_coord_pair_percent(self):
        coords = CoordinateParser(200, 100)
        assert _resolve_pivot(["50%", "50%"], (0, 0, 10, 10), coords) == (100.0, 50.0)


class TestApplyTransform:
    def test_empty_layer_unchanged(self):
        layer = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
        out = apply_transform(layer, rotation=45)
        assert out.getbbox() is None

    def test_rotation_changes_pixels(self):
        layer = _layer_with_box(box=(10, 10, 60, 20))  # wide bar
        rotated = apply_transform(layer, rotation=90)
        assert ImageChops.difference(layer, rotated).getbbox() is not None

    def test_180_rotation_is_involutive(self):
        layer = _layer_with_box(box=(10, 10, 40, 20))
        once = apply_transform(layer, rotation=180)
        twice = apply_transform(once, rotation=180)
        # Two 180° turns about the same bbox center return to the original.
        assert ImageChops.difference(layer, twice).getbbox() is None

    def test_horizontal_mirror_is_involutive(self):
        layer = _layer_with_box(box=(10, 10, 40, 20))
        once = apply_transform(layer, mirror="h")
        twice = apply_transform(once, mirror="h")
        assert ImageChops.difference(layer, twice).getbbox() is None

    def test_mirror_moves_asymmetric_content(self):
        # An L-shape is asymmetric about its bbox center, so a horizontal flip
        # must actually move pixels (a single rectangle would be a no-op).
        from PIL import ImageDraw

        layer = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rectangle((10, 10, 20, 60), fill=(0, 0, 0, 255))  # vertical stem
        draw.rectangle((10, 50, 60, 60), fill=(0, 0, 0, 255))  # bottom foot to the right
        mirrored = apply_transform(layer, mirror="h")
        assert ImageChops.difference(layer, mirrored).getbbox() is not None

    def test_unknown_mirror_code_is_noop(self):
        layer = _layer_with_box()
        out = apply_transform(layer, mirror="x")
        assert ImageChops.difference(layer, out).getbbox() is None
