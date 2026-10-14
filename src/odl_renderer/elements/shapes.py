from __future__ import annotations

import logging
from typing import Any

from PIL import ImageDraw

from odl_renderer.colors import BLACK
from odl_renderer.coordinates import coerce_number
from odl_renderer.elements.patterns import ShapeStyle, paint_shape, parse_pattern
from odl_renderer.registry import element_handler
from odl_renderer.types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)


@element_handler(ElementType.LINE, requires=["x_start", "x_end"])
def draw_line(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """
    Draw line element.

    Renders a straight line between two points, with options for color,
    thickness, and dashed style.

    Args:
        ctx: Drawing context
        element: Element dictionary with line properties
    """
    draw = ImageDraw.Draw(ctx.img)

    # Get vertical position
    if "y_start" not in element:
        y_start = ctx.pos_y + element.get("y_padding", 0)
        y_end = y_start
    else:
        y_start = ctx.coords.parse_y(element["y_start"])
        y_end = ctx.coords.parse_y(element.get("y_end", element["y_start"]))

    # Get line properties
    fill = ctx.colors.resolve(element.get("fill", "black"))
    width = int(coerce_number(element.get("width", 1), 1))
    dashed = element.get("dashed", False)
    dash_length = int(coerce_number(element.get("dash_length", 5), 5))
    space_length = int(coerce_number(element.get("space_length", 3), 3))

    x_start = ctx.coords.parse_x(element["x_start"])
    x_end = ctx.coords.parse_x(element["x_end"])

    if dashed:
        draw_dashed_line(draw, (x_start, y_start), (x_end, y_end), dash_length, space_length, fill or BLACK, width)
    else:
        draw.line([(x_start, y_start), (x_end, y_end)], fill=fill, width=width)

    ctx.pos_y = max(y_start, y_end)


def shape_style(ctx: DrawingContext, element: dict[str, Any]) -> ShapeStyle:
    """Read the fill, outline, outline width and pattern a shape element asks for."""
    return ShapeStyle(
        fill=ctx.colors.resolve(element.get("fill")),
        outline=ctx.colors.resolve(element.get("outline", "black")),
        width=int(coerce_number(element.get("width", 1), 1)),
        pattern=parse_pattern(element.get("pattern")),
    )


@element_handler(ElementType.RECTANGLE, requires=["x_start", "x_end", "y_start", "y_end"])
def draw_rectangle(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """
    Draw rectangle element.

    Renders a rectangle with options for fill, pattern, outline, and rounded corners.

    Args:
        ctx: Drawing context
        element: Element dictionary with rectangle properties
    """
    # Coordinates
    x_start = ctx.coords.parse_x(element["x_start"])
    x_end = ctx.coords.parse_x(element["x_end"])
    y_start = ctx.coords.parse_y(element["y_start"])
    y_end = ctx.coords.parse_y(element["y_end"])

    radius = int(coerce_number(element.get("radius", 10 if "corners" in element else 0)))
    corners = get_rounded_corners(element.get("corners", "all" if "radius" in element else ""))

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, **paint: Any) -> None:
        box = (x_start + dx, y_start + dy, x_end + dx, y_end + dy)
        draw.rounded_rectangle(box, radius=radius, corners=corners, **paint)

    box = (min(x_start, x_end), min(y_start, y_end), max(x_start, x_end), max(y_start, y_end))
    paint_shape(ctx, shape_style(ctx, element), box, shape)

    ctx.pos_y = y_end


@element_handler(
    ElementType.RECTANGLE_PATTERN,
    requires=["x_start", "x_size", "y_start", "y_size", "x_repeat", "y_repeat", "x_offset", "y_offset"],
)
def draw_rectangle_pattern(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """
    Draw repeated rectangle pattern.

    Renders a grid of rectangles with consistent spacing, useful for
    creating regular patterns or grids.

    Args:
        ctx: Drawing context
        element: Element dictionary with rectangle pattern properties
    """
    draw = ImageDraw.Draw(ctx.img)

    # Get pattern properties
    fill = ctx.colors.resolve(element.get("fill"))
    outline = ctx.colors.resolve(element.get("outline", "black"))
    width = int(coerce_number(element.get("width", 1), 1))
    radius = int(coerce_number(element.get("radius", 10 if "corners" in element else 0)))
    corners = get_rounded_corners(element.get("corners", "all" if "radius" in element else ""))

    # Parse geometry once (supports numeric strings and percentages from templates).
    x_start = ctx.coords.parse_x(element["x_start"])
    y_start = ctx.coords.parse_y(element["y_start"])
    x_size = ctx.coords.parse_size(element["x_size"], is_width=True)
    y_size = ctx.coords.parse_size(element["y_size"], is_width=False)
    x_offset = ctx.coords.parse_size(element["x_offset"], is_width=True)
    y_offset = ctx.coords.parse_size(element["y_offset"], is_width=False)
    x_repeat = int(coerce_number(element["x_repeat"]))
    y_repeat = int(coerce_number(element["y_repeat"]))

    max_y = y_start

    # Draw rectangle grid
    for x in range(x_repeat):
        for y in range(y_repeat):
            # Calculate rectangle position
            x_pos = x_start + x * (x_offset + x_size)
            y_pos = y_start + y * (y_offset + y_size)

            # Draw individual rectangle
            draw.rounded_rectangle(
                (x_pos, y_pos, x_pos + x_size, y_pos + y_size),
                fill=fill,
                outline=outline,
                width=width,
                radius=radius,
                corners=corners,
            )

            max_y = max(max_y, y_pos + y_size)

    ctx.pos_y = max_y


@element_handler(ElementType.POLYGON, requires=["points"])
def draw_polygon(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw a polygon.

    Renders a polygon defined by a list of vertex coordinates.

    Args:
        ctx: Drawing context
        element: Element dictionary with polygon properties
    """
    vertices = [(ctx.coords.parse_x(x), ctx.coords.parse_y(y)) for x, y in element["points"]]
    if not vertices:
        return

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, *, fill: Any, outline: Any, width: int) -> None:
        # The outline is one pixel wide whatever the `width`.
        draw.polygon([(x + dx, y + dy) for x, y in vertices], fill=fill, outline=outline)

    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    paint_shape(ctx, shape_style(ctx, element), (min(xs), min(ys), max(xs), max(ys)), shape)

    ctx.pos_y = max(ys)


@element_handler(ElementType.CIRCLE, requires=["x", "y", "radius"])
def draw_circle(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw circle element.

    Renders a circle with options for fill, pattern and outline.

    Args:
        ctx: Drawing Context
        element: Element dictionary with circle properties
    """
    x = ctx.coords.parse_x(element["x"])
    y = ctx.coords.parse_y(element["y"])
    radius = ctx.coords.parse_size(element["radius"], is_width=True)

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, **paint: Any) -> None:
        draw.ellipse([(x - radius + dx, y - radius + dy), (x + radius + dx, y + radius + dy)], **paint)

    paint_shape(ctx, shape_style(ctx, element), (x - radius, y - radius, x + radius, y + radius), shape)

    ctx.pos_y = y + radius


@element_handler(ElementType.ELLIPSE, requires=["x_start", "x_end", "y_start", "y_end"])
def draw_ellipse(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """
    Draw ellipse element.

    Renders an ellipse with options for fill, pattern and outline.

    Args:
        ctx: Drawing context
        element: Element dictionary with ellipse properties
    """
    x_start = ctx.coords.parse_x(element["x_start"])
    x_end = ctx.coords.parse_x(element["x_end"])
    y_start = ctx.coords.parse_y(element["y_start"])
    y_end = ctx.coords.parse_y(element["y_end"])

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, **paint: Any) -> None:
        draw.ellipse([(x_start + dx, y_start + dy), (x_end + dx, y_end + dy)], **paint)

    box = (min(x_start, x_end), min(y_start, y_end), max(x_start, x_end), max(y_start, y_end))
    paint_shape(ctx, shape_style(ctx, element), box, shape)

    ctx.pos_y = y_end


@element_handler(ElementType.ARC, requires=["x", "y", "radius", "start_angle", "end_angle"])
def draw_arc(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw an arc or pie slice.

    Renders an arc (outline) or pie slice (filled or patterned) based on center point,
    radius, and angle range.

    Args:
        ctx: Drawing context
        element: Element dictionary with arc properties
    """
    x = ctx.coords.parse_x(element["x"])
    y = ctx.coords.parse_y(element["y"])
    radius = ctx.coords.parse_size(element["radius"], is_width=True)
    start_angle = element["start_angle"]
    end_angle = element["end_angle"]

    style = shape_style(ctx, element)
    # A fill or a pattern makes a pie slice; with neither it is an arc line.
    pie = style.fill is not None or style.pattern is not None

    def shape(draw: ImageDraw.ImageDraw, dx: int, dy: int, *, fill: Any, outline: Any, width: int) -> None:
        bbox = [(x - radius + dx, y - radius + dy), (x + radius + dx, y + radius + dy)]
        if pie:
            draw.pieslice(bbox, start=start_angle, end=end_angle, fill=fill, outline=outline)
        else:
            draw.arc(bbox, start=start_angle, end=end_angle, fill=outline, width=width)

    paint_shape(ctx, style, (x - radius, y - radius, x + radius, y + radius), shape)

    ctx.pos_y = y + radius


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    dash_length: int,
    space_length: int,
    fill: tuple[int, int, int, int] = BLACK,
    width: int = 1,
) -> None:
    """Draw dashed line.

    Renders a dashed line between two points by drawing alternating
    segments of visible line and invisible space.

    Args:
        draw: PIL ImageDraw object to draw with
        start: Start point coordinates (x, y)
        end: End point coordinates (x, y)
        dash_length: Length of visible line segments
        space_length: Length of invisible space segments
        fill: Line color
        width: Line width
    """
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    line_length = (dx**2 + dy**2) ** 0.5

    # A zero-length line (start == end, easily produced by templated coordinates)
    # has no direction to step along and would divide by zero below. Draw the single
    # point if a dash starts there, then stop.
    if line_length == 0:
        if dash_length > 0:
            draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
        return

    step_x = dx / line_length
    step_y = dy / line_length

    current_pos = 0.0

    while True:
        # 1) Draw a dash segment
        dash_end = current_pos + dash_length

        if dash_end >= line_length:
            # A partial dash exists that ends exactly or beyond the line_end
            dash_end = line_length
            # segment_len = dash_end - current_pos

            segment_start_x = x1 + step_x * current_pos
            segment_start_y = y1 + step_y * current_pos
            segment_end_x = x1 + step_x * dash_end
            segment_end_y = y1 + step_y * dash_end

            draw.line([(segment_start_x, segment_start_y), (segment_end_x, segment_end_y)], fill=fill, width=width)
            # Process is done because the end of the line has been reached
            break
        else:
            # Normal full dash
            segment_start_x = x1 + step_x * current_pos
            segment_start_y = y1 + step_y * current_pos
            segment_end_x = x1 + step_x * dash_end
            segment_end_y = y1 + step_y * dash_end

            draw.line([(segment_start_x, segment_start_y), (segment_end_x, segment_end_y)], fill=fill, width=width)

            # 2) Move current_pos forward past this dash
            current_pos = dash_end

        # 3) Skip the space segment
        space_end = current_pos + space_length
        if space_end >= line_length:
            # The space would exceed the line's end, so processing is complete
            break
        else:
            # Jump over the space
            current_pos = space_end


def get_rounded_corners(corner_string: str) -> tuple[bool, bool, bool, bool]:
    """Get rounded corner configuration.

    Parses a string specifying which corners of a rectangle should be rounded.

    Args:
        corner_string: String specifying corners to round ("all" or comma-separated list)

    Returns:
        tuple: Boolean flags for (top_left, top_right, bottom_right, bottom_left)
    """
    if corner_string == "all":
        return True, True, True, True

    corners = corner_string.split(",")
    corner_map = {"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3}

    result = [False] * 4
    for corner in corners:
        corner = corner.strip()
        if corner in corner_map:
            result[corner_map[corner]] = True

    return result[0], result[1], result[2], result[3]
