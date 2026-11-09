import logging

import pytest
from PIL import ImageChops

from odl_renderer import generate_image
from odl_renderer.elements import patterns
from tests.builders import ElementBuilder as E

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
PALETTE = {BLACK, WHITE, RED, YELLOW, BLUE}
DIAGONAL_PERIOD = 6  # a spacing of 4 measured across the lines, 4 * sqrt(2) along the axes


async def render(*elements, width=100, height=100):
    image = await generate_image(width=width, height=height, elements=list(elements), background="white")
    return image.convert("RGB")


def pattern_rectangle(pattern, **kwargs):
    """A 100 x 100 rectangle with no outline, so every pixel shows the fill or the pattern."""
    return E.rectangle(0, 0, 99, 99, outline=None, pattern=pattern, **kwargs)


def same(first, second):
    return ImageChops.difference(first, second).getbbox() is None


def ink(image, color=BLACK):
    return {(x, y) for x in range(image.width) for y in range(image.height) if image.getpixel((x, y)) == color}


@pytest.mark.asyncio
class TestHatch:
    async def test_the_default_hatch_rises_to_the_right(self):
        image = await render(pattern_rectangle("hatch"))

        for x, y in ((10, 20), (11, 19), (12, 18), (40, 8)):
            assert (x + y) % DIAGONAL_PERIOD == 0
            assert image.getpixel((x, y)) == BLACK
        assert image.getpixel((10, 21)) == WHITE
        assert image.getpixel((11, 21)) == WHITE

    async def test_a_hatch_at_135_degrees_falls_to_the_right(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 135}))

        for x, y in ((20, 8), (21, 9), (22, 10), (60, 36)):
            assert (x - y) % DIAGONAL_PERIOD == 0
            assert image.getpixel((x, y)) == BLACK
        assert image.getpixel((20, 9)) == WHITE

    async def test_a_hatch_at_0_degrees_draws_horizontal_stripes(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 0, "spacing": 5}))

        for y in range(100):
            expected = BLACK if y % 5 == 0 else WHITE
            assert image.getpixel((37, y)) == expected
            assert image.getpixel((90, y)) == expected

    async def test_a_hatch_at_90_degrees_draws_vertical_stripes(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 90, "spacing": 5}))

        for x in range(100):
            assert image.getpixel((x, 37)) == (BLACK if x % 5 == 0 else WHITE)

    async def test_the_width_makes_the_lines_thicker(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 0, "spacing": 6, "width": 3}))

        rows = [y for y in range(12) if image.getpixel((50, y)) == BLACK]
        assert rows == [0, 1, 2, 6, 7, 8]

    async def test_other_angles_have_lines_of_an_even_thickness(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 30, "spacing": 10, "width": 2}))

        density = len(ink(image)) / (100 * 100)
        assert 0.17 < density < 0.23

    @pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 90, 120, 135, 170, 225, -45])
    async def test_every_pixel_is_a_palette_color_at_any_angle(self, angle):
        image = await render(pattern_rectangle({"type": "hatch", "angle": angle, "color": "red"}, fill="yellow"))

        assert set(image.getdata()) == {RED, YELLOW}

    async def test_an_angle_is_taken_modulo_a_half_turn(self):
        rising = await render(pattern_rectangle({"type": "hatch", "angle": 45}))

        assert same(await render(pattern_rectangle({"type": "hatch", "angle": 225})), rising)
        assert same(await render(pattern_rectangle({"type": "hatch", "angle": "45"})), rising)

    async def test_negative_angles_wrap_to_the_other_diagonal(self):
        falling = await render(pattern_rectangle({"type": "hatch", "angle": 135}))

        assert same(await render(pattern_rectangle({"type": "hatch", "angle": -45})), falling)

    @pytest.mark.parametrize("angle", [30, 45, 135])
    async def test_lines_thicker_than_their_spacing_cover_the_shape(self, angle):
        image = await render(pattern_rectangle({"type": "hatch", "angle": angle, "width": 100_000}))

        assert set(image.getdata()) == {BLACK}


@pytest.mark.asyncio
class TestDotsAndGrid:
    async def test_dots_sit_on_a_grid_of_the_spacing(self):
        image = await render(pattern_rectangle({"type": "dots", "spacing": 5}))

        for x in range(0, 100, 5):
            for y in range(0, 100, 5):
                assert image.getpixel((x, y)) == BLACK
        assert image.getpixel((2, 2)) == WHITE
        assert len(ink(image)) == 20 * 20

    async def test_the_width_of_a_dot_is_its_size(self):
        image = await render(pattern_rectangle({"type": "dots", "spacing": 10, "width": 2}))

        assert len(ink(image)) == 10 * 10 * 4

    async def test_a_dot_from_three_pixels_up_is_round(self):
        image = await render(pattern_rectangle({"type": "dots", "spacing": 12, "width": 5}))

        # The dot at (12, 12) covers 12..16 both ways: its middle is ink, its corners are not.
        assert image.getpixel((14, 14)) == BLACK
        assert image.getpixel((14, 12)) == BLACK
        assert image.getpixel((12, 14)) == BLACK
        for corner in ((12, 12), (16, 12), (12, 16), (16, 16)):
            assert image.getpixel(corner) == WHITE

    async def test_a_dot_of_two_pixels_is_square(self):
        image = await render(pattern_rectangle({"type": "dots", "spacing": 12, "width": 2}))

        for corner in ((12, 12), (13, 12), (12, 13), (13, 13)):
            assert image.getpixel(corner) == BLACK

    async def test_dots_wide_enough_to_touch_cover_the_shape(self):
        image = await render(pattern_rectangle({"type": "dots", "spacing": 1, "width": 100_000}))

        assert set(image.getdata()) == {BLACK}

    async def test_a_dot_cut_by_the_edge_of_the_shape_is_still_drawn(self):
        pattern = {"type": "dots", "spacing": 8, "width": 4}
        whole = await render(pattern_rectangle(pattern))
        part = await render(E.rectangle(2, 2, 60, 60, outline=None, pattern=pattern))

        # The dots at x = 0 and y = 0 start outside the shape but reach into it.
        assert same(part.crop((2, 2, 61, 61)), whole.crop((2, 2, 61, 61)))

    async def test_a_grid_draws_lines_both_ways(self):
        image = await render(pattern_rectangle({"type": "grid", "spacing": 10}))

        assert image.getpixel((20, 33)) == BLACK
        assert image.getpixel((33, 20)) == BLACK
        assert image.getpixel((33, 33)) == WHITE


@pytest.mark.asyncio
class TestTone:
    @pytest.mark.parametrize("level", [0, 10, 25, 50, 75, 90, 100])
    async def test_the_share_of_ink_is_the_level(self, level):
        image = await render(pattern_rectangle({"type": "tone", "level": level}), width=64, height=64)

        assert len(ink(image)) == round(level / 100 * 64) * 64

    async def test_every_pixel_is_a_palette_color(self):
        image = await render(pattern_rectangle({"type": "tone", "level": 37, "color": "red"}))

        assert set(image.getdata()) == {RED, WHITE}

    async def test_a_darker_tone_has_every_pixel_of_a_lighter_one(self):
        light = ink(await render(pattern_rectangle({"type": "tone", "level": 30}), width=32, height=32))
        dark = ink(await render(pattern_rectangle({"type": "tone", "level": 60}), width=32, height=32))

        assert light < dark

    async def test_a_tone_repeats_every_8_pixels(self):
        image = await render(pattern_rectangle({"type": "tone", "level": 40}))

        for x, y in ((0, 0), (3, 5), (7, 7), (13, 2)):
            assert image.getpixel((x, y)) == image.getpixel((x + 8, y)) == image.getpixel((x, y + 8))

    async def test_neighbouring_shapes_share_one_tone(self):
        whole = await render(E.rectangle(0, 0, 99, 49, outline=None, pattern={"type": "tone", "level": 35}))
        halves = await render(
            E.rectangle(0, 0, 36, 49, outline=None, pattern={"type": "tone", "level": 35}),
            E.rectangle(37, 0, 99, 49, outline=None, pattern={"type": "tone", "level": 35}),
        )

        assert same(whole, halves)

    async def test_a_tone_is_clipped_to_rounded_corners(self):
        image = await render(pattern_rectangle({"type": "tone", "level": 100}, radius=30, corners="all"))

        assert image.getpixel((0, 0)) == WHITE
        assert image.getpixel((50, 50)) == BLACK

    async def test_a_level_outside_the_range_is_clamped(self):
        low = await render(pattern_rectangle({"type": "tone", "level": -20}), width=16, height=16)
        high = await render(pattern_rectangle({"type": "tone", "level": 250}), width=16, height=16)
        text = await render(pattern_rectangle({"type": "tone", "level": "100"}), width=16, height=16)

        assert ink(low) == set()
        assert len(ink(high)) == 16 * 16
        assert same(high, text)

    async def test_the_level_defaults_to_half(self):
        image = await render(pattern_rectangle("tone"), width=64, height=64)

        assert len(ink(image)) == 32 * 64


@pytest.mark.asyncio
class TestLayers:
    async def test_the_fill_shows_between_the_lines(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 0, "color": "red"}, fill="yellow"))

        assert image.getpixel((50, 0)) == RED
        assert image.getpixel((50, 1)) == YELLOW

    async def test_without_a_fill_what_is_below_shows_between_the_lines(self):
        below = E.rectangle(0, 0, 99, 99, fill="blue", outline=None)

        image = await render(below, pattern_rectangle({"type": "hatch", "angle": 0}))

        assert image.getpixel((50, 0)) == BLACK
        assert image.getpixel((50, 1)) == BLUE

    async def test_the_outline_is_drawn_over_the_pattern(self):
        stripes = {"type": "hatch", "angle": 0, "spacing": 1}
        element = E.rectangle(10, 10, 90, 90, outline="blue", width=3, pattern=stripes)

        image = await render(element)

        assert image.getpixel((10, 50)) == BLUE
        assert image.getpixel((12, 50)) == BLUE
        assert image.getpixel((13, 50)) == BLACK
        assert image.getpixel((50, 12)) == BLUE
        assert image.getpixel((50, 13)) == BLACK

    async def test_the_pattern_is_clipped_to_rounded_corners(self):
        element = pattern_rectangle({"type": "hatch", "angle": 0, "spacing": 1}, radius=30, corners="all")

        image = await render(element)

        assert image.getpixel((0, 0)) == WHITE
        assert image.getpixel((1, 1)) == WHITE
        assert image.getpixel((99, 99)) == WHITE
        assert image.getpixel((50, 50)) == BLACK
        assert image.getpixel((50, 0)) == BLACK

    async def test_only_the_chosen_corners_are_clipped(self):
        element = pattern_rectangle({"type": "hatch", "angle": 0, "spacing": 1}, radius=30, corners="top_left")

        image = await render(element)

        assert image.getpixel((0, 0)) == WHITE
        assert image.getpixel((99, 0)) == BLACK
        assert image.getpixel((0, 99)) == BLACK

    async def test_neighbouring_shapes_continue_the_same_pattern(self):
        whole = await render(E.rectangle(0, 0, 99, 49, outline=None, pattern="hatch"))
        halves = await render(
            E.rectangle(0, 0, 49, 49, outline=None, pattern="hatch"),
            E.rectangle(50, 0, 99, 49, outline=None, pattern="hatch"),
        )

        assert same(whole, halves)

    async def test_the_pattern_does_not_move_with_the_shape(self):
        left = await render(E.rectangle(0, 0, 59, 59, outline=None, pattern="hatch"))
        right = await render(E.rectangle(30, 0, 89, 59, outline=None, pattern="hatch"))

        for x, y in ((40, 12), (45, 30), (59, 55)):
            assert left.getpixel((x, y)) == right.getpixel((x, y))

    async def test_a_color_with_alpha_blends_with_what_is_below(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 0, "spacing": 4, "color": "#FF000080"}))

        red, green, blue = image.getpixel((50, 0))
        assert red == 255
        assert 120 <= green <= 135
        assert 120 <= blue <= 135

    async def test_the_accent_color_works_as_the_pattern_color(self):
        image = await generate_image(
            width=20,
            height=20,
            elements=[pattern_rectangle({"type": "hatch", "angle": 0, "color": "accent"})],
            accent_color="yellow",
        )

        assert image.convert("RGB").getpixel((5, 0)) == YELLOW


@pytest.mark.asyncio
class TestFields:
    async def test_the_name_alone_is_the_same_as_the_object_with_defaults(self):
        short = await render(pattern_rectangle("dots"))
        long = await render(pattern_rectangle({"type": "dots", "color": "black", "spacing": 4, "width": 1}))

        assert same(short, long)

    @pytest.mark.parametrize("pattern", [None, False, "solid", "none", {"type": "solid"}, {}, ""])
    async def test_no_pattern_is_a_plain_fill(self, pattern):
        plain = await render(E.rectangle(0, 0, 99, 99, fill="red", outline=None))

        assert same(await render(pattern_rectangle(pattern, fill="red")), plain)

    async def test_an_unknown_type_is_ignored_and_reported_once(self, caplog):
        patterns._warned_types.clear()
        caplog.set_level(logging.WARNING)

        first = await render(pattern_rectangle("zigzag", fill="red"))
        await render(pattern_rectangle("zigzag", fill="red"))

        assert set(first.getdata()) == {RED}
        assert caplog.text.count("zigzag") == 1

    async def test_a_pattern_that_is_not_a_name_or_an_object_is_ignored(self):
        image = await render(pattern_rectangle(["hatch"], fill="red"))

        assert set(image.getdata()) == {RED}

    async def test_numbers_may_be_strings_as_templates_render_them(self):
        strings = await render(pattern_rectangle({"type": "hatch", "angle": "0", "spacing": "5", "width": "2"}))
        numbers = await render(pattern_rectangle({"type": "hatch", "angle": 0, "spacing": 5, "width": 2}))

        assert same(strings, numbers)

    @pytest.mark.parametrize("spacing", [0, -3, "x"])
    async def test_a_spacing_that_makes_no_sense_falls_back_to_something_drawable(self, spacing):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 0, "spacing": spacing}))

        assert BLACK in set(image.getdata())

    async def test_the_pattern_defaults_to_black(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 0}))

        assert image.getpixel((50, 0)) == BLACK

    async def test_an_empty_color_is_the_default_color(self):
        image = await render(pattern_rectangle({"type": "hatch", "angle": 0, "color": None}))

        assert image.getpixel((50, 0)) == BLACK

    async def test_a_hidden_rectangle_draws_no_pattern(self):
        image = await render(pattern_rectangle("hatch", visible=False))

        assert set(image.getdata()) == {WHITE}


@pytest.mark.asyncio
class TestPlacement:
    async def test_a_shape_partly_off_the_canvas_is_clipped(self):
        image = await render(E.rectangle(-30, -30, 40, 40, outline=None, pattern={"type": "hatch", "angle": 0}))

        assert image.getpixel((10, 0)) == BLACK
        assert image.getpixel((60, 0)) == WHITE

    async def test_a_shape_wholly_off_the_canvas_draws_nothing(self):
        image = await render(E.rectangle(200, 200, 300, 300, outline=None, pattern="hatch"))

        assert set(image.getdata()) == {WHITE}

    async def test_percent_coordinates_work(self):
        element = E.rectangle("0%", "0%", "50%", "50%", outline=None, pattern={"type": "hatch", "angle": 0})

        image = await render(element)

        assert image.getpixel((20, 0)) == BLACK
        assert image.getpixel((70, 0)) == WHITE

    async def test_a_rotated_rectangle_keeps_its_pattern(self):
        element = E.rectangle(10, 10, 60, 40, outline=None, pattern={"type": "hatch", "angle": 0}, rotation=90)

        image = await render(element)

        assert BLACK in set(image.getdata())
        assert set(image.getdata()) <= PALETTE

    async def test_a_pattern_in_a_thin_rectangle_does_not_fail(self):
        image = await render(E.rectangle(10, 10, 12, 80, pattern="grid"))

        assert set(image.getdata()) <= PALETTE


@pytest.mark.asyncio
class TestBackwardCompatibility:
    async def test_a_rectangle_without_a_pattern_is_drawn_as_before(self):
        image = await render(E.rectangle(10, 10, 60, 50, fill="red", outline="black", width=2, radius=8, corners="all"))

        assert image.getpixel((35, 30)) == RED
        assert image.getpixel((10, 30)) == BLACK
        assert image.getpixel((11, 30)) == BLACK
        assert image.getpixel((12, 30)) == RED
        assert image.getpixel((10, 10)) == WHITE
