"""Pattern fills for shapes: hatch, dots, grid and tone, drawn pixel-exact.

A pattern is drawn in one flat color, without anti-aliasing, so every pixel is a
palette color and survives any dithering mode. It is anchored to the canvas, not to
the shape, so shapes that touch continue the same pattern without a seam.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from odl_renderer.colors import BLACK
from odl_renderer.coordinates import coerce_number
from odl_renderer.types import DrawingContext

_LOGGER = logging.getLogger(__name__)

PATTERN_TYPES = frozenset({"hatch", "dots", "grid", "tone"})
NO_PATTERN_TYPES = frozenset({"", "solid", "none"})
DEFAULT_SPACING = 4
DEFAULT_WIDTH = 1
DEFAULT_ANGLE = 45.0
DEFAULT_LEVEL = 50.0
BAYER_ORDER = 8
DIAGONALS = (45.0, 135.0)
OPAQUE = 255
ELLIPTICAL_DOT_WIDTH = 3

# Pattern types already reported as unknown, so a payload that re-renders on every
# update logs once rather than on every frame.
_warned_types: set[str] = set()


@dataclass(frozen=True)
class Pattern:
    """A parsed ``pattern`` field."""

    type: str
    color: str
    spacing: int
    width: int
    angle: float
    level: float


def parse_pattern(spec: Any, default_color: str = "black") -> Pattern | None:
    """Parse a ``pattern`` field; None means a plain fill.

    Accepts a type name (``"hatch"``) or an object with ``type``, ``color``,
    ``spacing``, ``width``, ``angle`` and ``level``. ``solid`` and ``none`` mean no pattern,
    and an unknown type is ignored, so the shape is drawn as if it had none.
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        _LOGGER.warning("Pattern %r is neither a name nor an object; ignored", spec)
        return None
    kind = str(spec.get("type", "")).strip().lower()
    if kind in NO_PATTERN_TYPES:
        return None
    if kind not in PATTERN_TYPES:
        if kind not in _warned_types:
            _warned_types.add(kind)
            _LOGGER.warning("Unknown pattern type %r; drawing without a pattern", kind)
        return None
    return Pattern(
        type=kind,
        color=str(spec.get("color") or default_color),
        spacing=max(1, int(coerce_number(spec.get("spacing", DEFAULT_SPACING), DEFAULT_SPACING))),
        width=max(1, int(coerce_number(spec.get("width", DEFAULT_WIDTH), DEFAULT_WIDTH))),
        angle=float(coerce_number(spec.get("angle", DEFAULT_ANGLE), DEFAULT_ANGLE)) % 180,
        level=min(100.0, max(0.0, float(coerce_number(spec.get("level", DEFAULT_LEVEL), DEFAULT_LEVEL)))),
    )


@dataclass(frozen=True)
class Area:
    """The part of the canvas a pattern is drawn on: its top-left corner and size."""

    x: int
    y: int
    width: int
    height: int


def clip_area(ctx: DrawingContext, left: int, top: int, right: int, bottom: int) -> Area | None:
    """Return the part of an inclusive box that lies on the canvas, or None."""
    left, right = min(left, right), max(left, right)
    top, bottom = min(top, bottom), max(top, bottom)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(right, ctx.img.width - 1), min(bottom, ctx.img.height - 1)
    if x1 < x0 or y1 < y0:
        return None
    return Area(x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _line_positions(start: int, length: int, pattern: Pattern) -> range:
    """Return where lines begin on the axis: multiples of the spacing that reach in."""
    first = -(-(start - pattern.width + 1) // pattern.spacing) * pattern.spacing
    return range(first, start + length, pattern.spacing)


def _horizontal_lines(draw: ImageDraw.ImageDraw, area: Area, pattern: Pattern) -> None:
    for y in _line_positions(area.y, area.height, pattern):
        top = y - area.y
        draw.rectangle((0, top, area.width - 1, top + pattern.width - 1), fill=OPAQUE)


def _vertical_lines(draw: ImageDraw.ImageDraw, area: Area, pattern: Pattern) -> None:
    for x in _line_positions(area.x, area.width, pattern):
        left = x - area.x
        draw.rectangle((left, 0, left + pattern.width - 1, area.height - 1), fill=OPAQUE)


def _look_up_strip(ink: Image.Image, strip: Image.Image, coefficients: tuple[float, ...]) -> None:
    """Fill the mask by looking every pixel up in a strip.

    The coefficients are those of an affine transform: they say where in the strip each
    output pixel is read. Nothing is blended, so every pixel is either ink or not.
    """
    ink.paste(strip.transform(ink.size, Image.Transform.AFFINE, coefficients, Image.Resampling.NEAREST))


def _diagonal_lines(ink: Image.Image, area: Area, pattern: Pattern) -> None:
    """Fill the mask with 45 and 135 degree lines, exactly one pixel per row.

    A line rising to the right keeps ``x + y`` constant and one falling to the right
    keeps ``x - y`` constant. On the canvas a pixel has ink when that number, taken
    modulo the period between the lines, is less than the thickness of a line. One strip
    holds that for every value the area reaches, so the time does not grow with the
    thickness.
    """
    rising = pattern.angle == DIAGONALS[0]
    period = max(1, round(pattern.spacing * math.sqrt(2)))
    band = max(1, round(pattern.width * math.sqrt(2)))
    lowest = area.x + area.y if rising else area.x - area.y - (area.height - 1)
    length = area.width + area.height - 1
    values = bytes(OPAQUE if (lowest + index) % period < band else 0 for index in range(length))
    strip = Image.frombytes("L", (length, 1), values)
    # Output pixel (x, y) reads the strip at x + y (rising) or x - y + height - 1 (falling);
    # the half pixels are there because Pillow reads at the centre of a pixel.
    if rising:
        _look_up_strip(ink, strip, (1, 1, -0.5, 0, 0, 0.5))
    else:
        _look_up_strip(ink, strip, (1, -1, area.height - 0.5, 0, 0, 0.5))


def _slanted_lines(ink: Image.Image, area: Area, pattern: Pattern) -> None:
    """Fill the mask with lines at any other angle, all of the same thickness.

    A one pixel wide strip holds the lines of the pattern across their normal, and
    every output pixel looks up the strip at its distance from the canvas origin. That
    keeps the thickness even, which drawing slanted lines pixel by pixel does not.
    """
    radians = math.radians(pattern.angle)
    normal = (math.sin(radians), math.cos(radians))
    offsets = [
        normal[0] * (area.x + dx) + normal[1] * (area.y + dy) for dx in (0, area.width) for dy in (0, area.height)
    ]
    lowest = math.floor(min(offsets)) - 1
    strip = Image.new("L", (2, math.ceil(max(offsets)) - lowest + 2), 0)
    draw = ImageDraw.Draw(strip)
    for row in range(strip.height):
        if (lowest + row) % pattern.spacing < pattern.width:
            draw.line([(0, row), (1, row)], fill=OPAQUE)
    shift = normal[0] * area.x + normal[1] * area.y - lowest
    _look_up_strip(ink, strip, (0, 0, 1, normal[0], normal[1], shift))


def _hatch(ink: Image.Image, draw: ImageDraw.ImageDraw, area: Area, pattern: Pattern) -> None:
    if pattern.angle == 0:
        _horizontal_lines(draw, area, pattern)
    elif pattern.angle == 90:  # noqa: PLR2004 - a right angle
        _vertical_lines(draw, area, pattern)
    elif pattern.angle in DIAGONALS:
        _diagonal_lines(ink, area, pattern)
    else:
        _slanted_lines(ink, area, pattern)


def _dots(draw: ImageDraw.ImageDraw, area: Area, pattern: Pattern) -> None:
    step = pattern.spacing
    size = pattern.width
    if size >= 2 * step:
        # Dots this wide overlap until nothing is left between them.
        draw.rectangle((0, 0, area.width - 1, area.height - 1), fill=OPAQUE)
        return
    # Start with the dots that begin before the area but reach into it.
    first_x = -(-(area.x - size + 1) // step) * step
    first_y = -(-(area.y - size + 1) // step) * step
    for y in range(first_y, area.y + area.height, step):
        for x in range(first_x, area.x + area.width, step):
            box = (x - area.x, y - area.y, x - area.x + size - 1, y - area.y + size - 1)
            if size >= ELLIPTICAL_DOT_WIDTH:
                draw.ellipse(box, fill=OPAQUE)
            else:
                draw.rectangle(box, fill=OPAQUE)


def _bayer(order: int) -> list[list[int]]:
    """Return a Bayer matrix of ``order`` x ``order`` thresholds, 0 to order squared - 1."""
    matrix = [[0]]
    while len(matrix) < order:
        matrix = [[4 * value for value in row] + [4 * value + 2 for value in row] for row in matrix] + [
            [4 * value + 3 for value in row] + [4 * value + 1 for value in row] for row in matrix
        ]
    return matrix


_BAYER = _bayer(BAYER_ORDER)


def _tone(ink: Image.Image, area: Area, pattern: Pattern) -> None:
    """Fill the mask with an ordered dither whose share of ink is the ``level``.

    The Bayer matrix is tiled from the canvas origin, so a tone is the same on every
    shape and continues across shapes that touch. An 8 x 8 matrix gives 64 tones.
    """
    covered = round(pattern.level / 100 * BAYER_ORDER * BAYER_ORDER)
    tile = Image.new("L", (BAYER_ORDER, BAYER_ORDER), 0)
    tile.putdata([OPAQUE if _BAYER[y][x] < covered else 0 for y in range(BAYER_ORDER) for x in range(BAYER_ORDER)])
    left = area.x - area.x % BAYER_ORDER
    top = area.y - area.y % BAYER_ORDER
    row = Image.new("L", (area.width + 2 * BAYER_ORDER, BAYER_ORDER), 0)
    for x in range(0, row.width, BAYER_ORDER):
        row.paste(tile, (x, 0))
    for y in range(top, area.y + area.height, BAYER_ORDER):
        ink.paste(row, (left - area.x, y - area.y))


def pattern_ink(pattern: Pattern, area: Area) -> Image.Image:
    """Return a mask of the area that is 255 where the pattern has ink."""
    ink = Image.new("L", (area.width, area.height), 0)
    draw = ImageDraw.Draw(ink)
    if pattern.type == "hatch":
        _hatch(ink, draw, area, pattern)
    elif pattern.type == "grid":
        _horizontal_lines(draw, area, pattern)
        _vertical_lines(draw, area, pattern)
    elif pattern.type == "tone":
        _tone(ink, area, pattern)
    else:
        _dots(draw, area, pattern)
    return ink


def draw_pattern(ctx: DrawingContext, pattern: Pattern, area: Area, shape: Image.Image) -> None:
    """Draw a pattern on the canvas, only where the ``shape`` mask is set.

    Args:
        ctx: Drawing context
        pattern: The parsed pattern
        area: The part of the canvas to draw on
        shape: An ``L`` mask of the area, 255 inside the shape
    """
    # A parsed pattern always names a color; the fallback only satisfies the resolver's type.
    color = ctx.colors.resolve(pattern.color) or BLACK
    mask = ImageChops.multiply(pattern_ink(pattern, area), shape)
    if color[3] < OPAQUE:
        mask = mask.point(lambda value: value * color[3] // OPAQUE)
    ctx.img.paste(Image.new("RGBA", mask.size, (*color[:3], OPAQUE)), (area.x, area.y), mask)


Color = tuple[int, int, int, int]


@dataclass(frozen=True)
class ShapeStyle:
    """How a shape is painted: its fill, outline and pattern."""

    fill: Color | None
    outline: Color | None
    width: int
    pattern: Pattern | None


# Draws one shape. Called with a drawing surface and the offset to add to every coordinate, so
# the same function paints the canvas (offset 0) and the mask of the shape (offset of its area).
Shape = Callable[..., None]


def paint_shape(ctx: DrawingContext, style: ShapeStyle, box: tuple[int, int, int, int], shape: Shape) -> None:
    """Paint a shape with its fill, then its pattern, then its outline.

    This is the one place that knows about patterns, so every shape gets them the same way.
    A shape only says how to draw its silhouette:
    ``shape(draw, dx, dy, fill=..., outline=..., width=...)`` draws it with ``dx`` and ``dy``
    added to its coordinates.

    Args:
        ctx: Drawing context
        style: Fill, outline, outline width and the parsed pattern
        box: The inclusive bounding box of the shape, ``(left, top, right, bottom)``
        shape: Draws the silhouette of the shape
    """
    draw = ImageDraw.Draw(ctx.img)
    if style.pattern is None:
        shape(draw, 0, 0, fill=style.fill, outline=style.outline, width=style.width)
        return
    if style.fill is not None:
        shape(draw, 0, 0, fill=style.fill, outline=None, width=style.width)
    area = clip_area(ctx, *box)
    if area is not None:
        silhouette = Image.new("L", (area.width, area.height), 0)
        shape(ImageDraw.Draw(silhouette), -area.x, -area.y, fill=OPAQUE, outline=None, width=style.width)
        draw_pattern(ctx, style.pattern, area, silhouette)
    if style.outline is not None and style.width > 0:
        shape(draw, 0, 0, fill=None, outline=style.outline, width=style.width)


def rectangle_shape(left: float, top: float, right: float, bottom: float) -> Shape:
    """Return the shape of a square-cornered rectangle, for elements that draw bars."""

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, **paint: Any) -> None:
        draw.rectangle((left + dx, top + dy, right + dx, bottom + dy), **paint)

    return shape


def ellipse_shape(left: float, top: float, right: float, bottom: float) -> Shape:
    """Return the shape of an ellipse that fills the box, which is a circle when it is square."""

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, **paint: Any) -> None:
        draw.ellipse([(left + dx, top + dy), (right + dx, bottom + dy)], **paint)

    return shape


def polygon_shape(vertices: list[tuple[int, int]]) -> Shape:
    """Return the shape of a polygon. Its outline is one pixel wide whatever the `width`."""

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, *, fill: Any, outline: Any, width: int) -> None:
        draw.polygon([(x + dx, y + dy) for x, y in vertices], fill=fill, outline=outline)

    return shape


def bounding_box(left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
    """Round a box outwards to whole pixels, for the area a pattern may cover."""
    return math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)


def box_around(vertices: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return the bounding box of the vertices of a polygon."""
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    return bounding_box(min(xs), min(ys), max(xs), max(ys))
