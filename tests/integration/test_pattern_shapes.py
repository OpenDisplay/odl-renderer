"""The `pattern` field on every shape: circle, ellipse, polygon, arc, bars and series."""

import pytest
from PIL import ImageChops

from odl_renderer import generate_image
from tests.builders import ElementBuilder as E
from tests.integration.test_plot import MockDataProvider, make_states

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
# Horizontal stripes on every even row: easy to tell from a fill and from the background.
STRIPES = {"type": "hatch", "angle": 0, "spacing": 2}


async def render(*elements, width=100, height=100, **options):
    image = await generate_image(width=width, height=height, elements=list(elements), background="white", **options)
    return image.convert("RGB")


def shape(kind, **fields):
    return {"type": kind, **fields}


def same(first, second):
    return ImageChops.difference(first, second).getbbox() is None


def pixels(image, color):
    return {(x, y) for x in range(image.width) for y in range(image.height) if image.getpixel((x, y)) == color}


@pytest.mark.asyncio
class TestCircleAndEllipse:
    async def test_a_circle_is_striped_only_inside_its_outline(self):
        image = await render(E.circle(50, 50, 30, outline=None, pattern=STRIPES))

        assert image.getpixel((50, 50)) == BLACK
        assert image.getpixel((50, 51)) == WHITE
        assert image.getpixel((22, 22)) == WHITE  # a corner of the bounding box
        assert all((x - 50) ** 2 + (y - 50) ** 2 <= 31**2 for x, y in pixels(image, BLACK))

    async def test_the_pattern_sits_between_the_fill_and_the_outline(self):
        pattern = {**STRIPES, "color": "red"}
        image = await render(E.circle(50, 50, 30, fill="yellow", outline="black", width=3, pattern=pattern))

        assert image.getpixel((50, 50)) == RED
        assert image.getpixel((50, 51)) == YELLOW
        assert image.getpixel((20, 50)) == BLACK  # the outline covers the stripe

    async def test_an_ellipse_is_striped_inside_its_outline(self):
        image = await render(
            shape("ellipse", x_start=10, y_start=30, x_end=90, y_end=70, outline=None, pattern=STRIPES)
        )

        assert image.getpixel((50, 50)) == BLACK
        assert image.getpixel((12, 32)) == WHITE
        assert all(30 <= y <= 70 for _, y in pixels(image, BLACK))

    async def test_a_shape_without_a_pattern_is_unchanged(self):
        plain = await render(E.circle(50, 50, 30, fill="yellow"))
        solid = await render(E.circle(50, 50, 30, fill="yellow", pattern="solid"))

        assert same(plain, solid)

    async def test_the_pattern_is_anchored_to_the_canvas(self):
        circle = await render(E.circle(50, 50, 30, outline=None, pattern="hatch"))
        square = await render(E.rectangle(0, 0, 99, 99, outline=None, pattern="hatch"))

        assert pixels(circle, BLACK) <= pixels(square, BLACK)


@pytest.mark.asyncio
class TestPolygonAndArc:
    async def test_a_polygon_is_striped_inside_its_edges(self):
        triangle = [(10, 90), (90, 90), (50, 10)]
        image = await render(shape("polygon", points=triangle, outline=None, pattern=STRIPES))

        assert image.getpixel((50, 80)) == BLACK
        assert image.getpixel((50, 81)) == WHITE
        assert image.getpixel((15, 50)) == WHITE  # left of the left edge
        assert all(10 <= y <= 90 for _, y in pixels(image, BLACK))

    async def test_a_polygon_without_points_draws_nothing(self):
        image = await render(shape("polygon", points=[], fill="black", pattern=STRIPES))

        assert pixels(image, BLACK) == set()

    async def test_a_pattern_turns_an_arc_into_a_striped_pie_slice(self):
        image = await render(
            shape("arc", x=50, y=50, radius=40, start_angle=0, end_angle=90, outline=None, pattern=STRIPES)
        )

        assert image.getpixel((70, 60)) == BLACK  # in the lower right quarter
        assert image.getpixel((30, 60)) == WHITE  # the lower left is outside the slice

    async def test_an_arc_with_neither_fill_nor_pattern_is_still_a_line(self):
        image = await render(shape("arc", x=50, y=50, radius=40, start_angle=0, end_angle=90, width=2))

        assert image.getpixel((50, 50)) == WHITE
        assert image.getpixel((90, 50)) == BLACK


@pytest.mark.asyncio
class TestProgressBar:
    def bar(self, **kwargs):
        return shape(
            "progress_bar",
            x_start=0,
            y_start=0,
            x_end=99,
            y_end=19,
            progress=50,
            outline=None,
            fill="red",
            background="white",
            **kwargs,
        )

    async def test_the_pattern_covers_only_the_progress(self):
        pattern = {"type": "hatch", "angle": 0, "spacing": 2, "color": "black"}
        image = await render(self.bar(pattern=pattern), height=20)

        assert image.getpixel((20, 2)) == BLACK
        assert image.getpixel((20, 3)) == RED
        assert image.getpixel((80, 2)) == WHITE

    async def test_the_background_pattern_covers_the_rest(self):
        image = await render(self.bar(background_pattern=STRIPES), height=20)

        assert image.getpixel((80, 2)) == BLACK
        assert image.getpixel((80, 3)) == WHITE
        assert image.getpixel((20, 3)) == RED


@pytest.mark.asyncio
class TestDiagram:
    async def test_the_bars_can_be_patterned(self):
        bars = {"values": "a,10;b,10", "color": "red", "pattern": {**STRIPES, "color": "black"}}
        image = await render(shape("diagram", x=0, height=100, bars=bars))

        assert image.getpixel((40, 50)) == BLACK  # on a stripe row
        assert image.getpixel((40, 51)) == RED  # between the stripes the fill shows


@pytest.mark.asyncio
class TestPlot:
    def plot(self, **series):
        return {
            "type": "plot",
            "data": [{"entity": "sensor.flat", "color": "red", **series}],
            "legend": False,
        }

    async def render_plot(self, **series):
        provider = MockDataProvider({"sensor.flat": make_states(24, [20.0] * 24)})
        return await render(self.plot(**series), width=200, height=100, data_provider=provider)

    async def test_the_area_under_the_line_gets_the_pattern_in_the_series_color(self):
        plain = await self.render_plot()
        patterned = await self.render_plot(pattern=STRIPES)

        added = pixels(patterned, RED) - pixels(plain, RED)
        line_row = min(y for _, y in pixels(plain, RED))
        assert len(added) > 500
        assert all(y > line_row for _, y in added)

    async def test_a_series_without_a_pattern_is_unchanged(self):
        plain = await self.render_plot()
        solid = await self.render_plot(pattern="solid")

        assert same(plain, solid)
