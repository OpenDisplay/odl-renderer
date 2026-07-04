import pytest

from odl_renderer import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestLineElement:
    async def test_basic_line(self):
        image = await generate_image(
            width=200, height=100, elements=[E.line(x_start=0, y_start=50, x_end=200, y_end=50)]
        )
        assert image.size == (200, 100)

    async def test_line_without_y_uses_pos_y(self):
        """Line without y_start falls back to pos_y."""
        image = await generate_image(width=200, height=100, elements=[{"type": "line", "x_start": 0, "x_end": 200}])
        assert image.size == (200, 100)

    async def test_dashed_line(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[E.line(x_start=0, y_start=50, x_end=200, y_end=50, dashed=True, dash_length=4, space_length=4)],
        )
        assert image.size == (200, 100)

    async def test_zero_length_dashed_line_does_not_crash(self):
        """A dashed line with start == end must not divide by zero (finding A4)."""
        image = await generate_image(
            width=200,
            height=100,
            elements=[E.line(x_start=50, y_start=50, x_end=50, y_end=50, dashed=True)],
        )
        assert image.size == (200, 100)

    async def test_line_colors(self):
        for color in ["black", "red", "#FF0000"]:
            image = await generate_image(
                width=200,
                height=100,
                elements=[E.line(x_start=0, y_start=50, x_end=200, y_end=50, fill=color)],
            )
            assert image.size == (200, 100)


@pytest.mark.asyncio
class TestRectangleElement:
    async def test_filled_rectangle(self):
        image = await generate_image(
            width=200, height=100, elements=[E.rectangle(x_start=10, y_start=10, x_end=90, y_end=50, fill="black")]
        )
        assert image.size == (200, 100)

    async def test_outlined_rectangle(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[E.rectangle(x_start=10, y_start=10, x_end=90, y_end=50, outline="black", fill=None)],
        )
        assert image.size == (200, 100)

    async def test_rounded_rectangle(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[E.rectangle(x_start=10, y_start=10, x_end=90, y_end=50, fill="black", radius=8)],
        )
        assert image.size == (200, 100)

    async def test_rounded_corners_selection(self):
        for corners in ["top_left", "top_right", "bottom_left", "bottom_right", "top", "bottom", "all", ""]:
            image = await generate_image(
                width=200,
                height=100,
                elements=[E.rectangle(x_start=10, y_start=10, x_end=90, y_end=50, radius=8, corners=corners)],
            )
            assert image.size == (200, 100)


@pytest.mark.asyncio
class TestRectanglePatternElement:
    async def test_basic_pattern(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "rectangle_pattern",
                    "x_start": 0,
                    "y_start": 0,
                    "x_size": 10,
                    "y_size": 10,
                    "x_offset": 5,
                    "y_offset": 5,
                    "x_repeat": 3,
                    "y_repeat": 3,
                    "fill": "black",
                }
            ],
        )
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestPolygonElement:
    async def test_basic_polygon(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[{"type": "polygon", "points": [(10, 10), (100, 10), (55, 80)], "fill": "black"}],
        )
        assert image.size == (200, 100)

    async def test_polygon_outline_only(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[{"type": "polygon", "points": [(10, 10), (100, 10), (55, 80)], "outline": "black"}],
        )
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestCircleElement:
    async def test_basic_circle(self):
        image = await generate_image(width=200, height=100, elements=[E.circle(x=100, y=50, radius=30, fill="black")])
        assert image.size == (200, 100)

    async def test_circle_outline(self):
        image = await generate_image(
            width=200, height=100, elements=[E.circle(x=100, y=50, radius=30, outline="black")]
        )
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestEllipseElement:
    async def test_basic_ellipse(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[{"type": "ellipse", "x_start": 10, "y_start": 10, "x_end": 90, "y_end": 50, "fill": "black"}],
        )
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestArcElement:
    async def test_basic_arc(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[{"type": "arc", "x": 100, "y": 50, "radius": 30, "start_angle": 0, "end_angle": 180}],
        )
        assert image.size == (200, 100)

    async def test_arc_with_color(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "arc",
                    "x": 100,
                    "y": 50,
                    "radius": 30,
                    "start_angle": 0,
                    "end_angle": 270,
                    "fill": "black",
                    "width": 2,
                }
            ],
        )
        assert image.size == (200, 100)
