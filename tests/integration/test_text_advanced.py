import pytest
from PIL import Image, ImageChops

from odl_renderer import generate_image


def text(**kwargs):
    return {"type": "text", "x": 10, "y": 10, "value": "Hello", "font": "ppb", "size": 16, **kwargs}


def content_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of the non-white (drawn) pixels of a white-background image."""
    return ImageChops.invert(image.convert("L")).getbbox()


def multiline(**kwargs):
    return {
        "type": "multiline",
        "x": 10,
        "value": "Line one|Line two|Line three",
        "delimiter": "|",
        "offset_y": 20,
        "font": "ppb",
        "size": 16,
        **kwargs,
    }


@pytest.mark.asyncio
class TestTextPositioning:
    async def test_text_without_y_uses_pos_y(self):
        """Text without y falls back to pos_y + y_padding."""
        image = await generate_image(
            width=200, height=100, elements=[{"type": "text", "x": 10, "value": "Test", "font": "ppb", "size": 16}]
        )
        assert image.size == (200, 100)

    async def test_text_y_padding(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[{"type": "text", "x": 10, "value": "Test", "font": "ppb", "size": 16, "y_padding": 20}],
        )
        assert image.size == (200, 100)

    async def test_text_with_anchor(self):
        for anchor in ["lt", "mm", "rb", "la"]:
            image = await generate_image(width=200, height=100, elements=[text(anchor=anchor)])
            assert image.size == (200, 100)

    async def test_text_with_stroke(self):
        image = await generate_image(width=200, height=100, elements=[text(stroke_width=2, stroke_fill="white")])
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestTextWrapping:
    async def test_text_wrapping(self):
        """max_width without truncate wraps text into multiple lines."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="This is a long sentence that should wrap", max_width=100)],
        )
        assert image.size == (200, 100)

    async def test_text_truncation(self):
        """max_width with truncate=True adds ellipsis."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="This is a long sentence that should be truncated", max_width=80, truncate=True)],
        )
        assert image.size == (200, 100)

    async def test_text_truncation_short_text_unchanged(self):
        """Text shorter than max_width is not truncated."""
        image = await generate_image(width=200, height=100, elements=[text(value="Hi", max_width=200, truncate=True)])
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestColoredText:
    async def test_parse_colors_single_segment(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="[red]Hello[/red] world", parse_colors=True)],
        )
        assert image.size == (200, 100)

    async def test_parse_colors_multiple_segments(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="[red]Red[/red] and [black]black[/black]", parse_colors=True)],
        )
        assert image.size == (200, 100)

    async def test_parse_colors_with_newlines(self):
        """parse_colors=True with newlines in text uses multiline colored path."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="[red]Line one[/red]\n[black]Line two[/black]", parse_colors=True)],
        )
        assert image.size == (200, 100)

    async def test_parse_colors_center_align(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="[red]Centered[/red]", parse_colors=True, align="center", x=100)],
        )
        assert image.size == (200, 100)

    async def test_parse_colors_right_align(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="[red]Right[/red]", parse_colors=True, align="right", x=190)],
        )
        assert image.size == (200, 100)

    async def test_parse_colors_hex_color(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="[#FF0000]Red hex[/#FF0000]", parse_colors=True)],
        )
        assert image.size == (200, 100)

    async def test_parse_colors_anchor_middle(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="[red]Middle[/red]", parse_colors=True, anchor="mm", x=100, y=50)],
        )
        assert image.size == (200, 100)

    async def test_parse_colors_uses_element_color(self):
        """Unmarked text on the parse_colors path uses the element color, not black (A6)."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[text(value="plain text", parse_colors=True, color="red")],
        )
        colors = {c for _, c in image.convert("RGB").getcolors(maxcolors=100000)}
        assert (255, 0, 0) in colors, "element color 'red' was ignored on the parse_colors path"

    async def test_parse_colors_honors_vertical_anchor(self):
        """A 'mm' anchor must position single-line parse_colors text like the plain path (A7)."""
        with_parse = await generate_image(
            width=200,
            height=100,
            elements=[text(value="Middle", parse_colors=True, anchor="mm", x=100, y=50, color="black")],
        )
        plain = await generate_image(
            width=200,
            height=100,
            elements=[text(value="Middle", anchor="mm", x=100, y=50, color="black")],
        )
        parse_box = content_bbox(with_parse)
        plain_box = content_bbox(plain)
        assert parse_box is not None and plain_box is not None
        # Vertical top should match within a couple of pixels, not be shifted down by
        # half the text height as it was when the anchor was dropped.
        assert abs(parse_box[1] - plain_box[1]) <= 2

    async def test_parse_colors_anchor_bottom(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                text(value="[red]Line1[/red]\n[black]Line2[/black]", parse_colors=True, anchor="mb", x=100, y=80)
            ],
        )
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestMultilineElement:
    async def test_basic_multiline(self):
        image = await generate_image(width=200, height=100, elements=[multiline(y=10)])
        assert image.size == (200, 100)

    async def test_multiline_with_start_y(self):
        """Legacy start_y field is supported."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "multiline",
                    "x": 10,
                    "start_y": 10,
                    "value": "A|B",
                    "delimiter": "|",
                    "offset_y": 20,
                    "font": "ppb",
                    "size": 16,
                }
            ],
        )
        assert image.size == (200, 100)

    async def test_multiline_without_y_uses_pos_y(self):
        """Multiline without y or start_y falls back to pos_y."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "multiline",
                    "x": 10,
                    "value": "A|B",
                    "delimiter": "|",
                    "offset_y": 20,
                    "font": "ppb",
                    "size": 16,
                }
            ],
        )
        assert image.size == (200, 100)

    async def test_multiline_parse_colors(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[multiline(y=10, value="[red]Red[/red]|[black]Black[/black]", parse_colors=True)],
        )
        assert image.size == (200, 100)

    async def test_multiline_alignment(self):
        for align in ["left", "center", "right"]:
            image = await generate_image(width=200, height=100, elements=[multiline(y=10, align=align)])
            assert image.size == (200, 100)
