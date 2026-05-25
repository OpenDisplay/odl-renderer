import pytest

from odl_renderer import generate_image

MDI_ICON = "home"  # stable icon present in all MDI releases


@pytest.mark.asyncio
class TestIconElement:
    async def test_basic_icon(self):
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "icon", "value": MDI_ICON, "x": 50, "y": 50, "size": 24}],
        )
        assert image.size == (100, 100)

    async def test_icon_color(self):
        for color in ["black", "red", "#0000FF"]:
            image = await generate_image(
                width=100,
                height=100,
                elements=[{"type": "icon", "value": MDI_ICON, "x": 50, "y": 50, "size": 24, "color": color}],
            )
            assert image.size == (100, 100)

    async def test_icon_fill_alias(self):
        """fill= is an alias for color=."""
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "icon", "value": MDI_ICON, "x": 50, "y": 50, "size": 24, "fill": "black"}],
        )
        assert image.size == (100, 100)

    @pytest.mark.parametrize("anchor", ["la", "mm", "lt", "rt", "lb", "rb", "mt", "mb", "lm", "rm"])
    async def test_icon_anchors(self, anchor):
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "icon", "value": MDI_ICON, "x": 50, "y": 50, "size": 24, "anchor": anchor}],
        )
        assert image.size == (100, 100)

    async def test_icon_invalid_anchor_raises(self):
        """Unknown anchor strings are rejected by Pillow."""
        with pytest.raises(Exception):
            await generate_image(
                width=100,
                height=100,
                elements=[{"type": "icon", "value": MDI_ICON, "x": 50, "y": 50, "size": 24, "anchor": "unknown"}],
            )

    async def test_icon_invalid_name_raises(self):
        with pytest.raises(ValueError):
            await generate_image(
                width=100,
                height=100,
                elements=[{"type": "icon", "value": "nonexistent_icon_xyz", "x": 50, "y": 50, "size": 24}],
            )


@pytest.mark.asyncio
class TestIconSequenceElement:
    async def test_basic_icon_sequence(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "icon_sequence",
                    "x": 10,
                    "y": 50,
                    "size": 24,
                    "icons": [MDI_ICON, MDI_ICON],
                }
            ],
        )
        assert image.size == (200, 100)

    async def test_icon_sequence_with_spacing(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "icon_sequence",
                    "x": 10,
                    "y": 50,
                    "size": 24,
                    "spacing": 5,
                    "icons": [MDI_ICON, MDI_ICON],
                    "color": "black",
                }
            ],
        )
        assert image.size == (200, 100)

    @pytest.mark.parametrize("direction", ["right", "left", "down", "up"])
    async def test_icon_sequence_directions(self, direction):
        image = await generate_image(
            width=200,
            height=200,
            elements=[
                {
                    "type": "icon_sequence",
                    "x": 100,
                    "y": 100,
                    "size": 24,
                    "icons": [MDI_ICON, MDI_ICON],
                    "direction": direction,
                }
            ],
        )
        assert image.size == (200, 200)
