from __future__ import annotations

import logging
import math
from typing import Any

from PIL import ImageDraw

from drawcustom.registry import element_handler
from drawcustom.types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)

# TODO: Plot element requires refactoring
# The plot element (ElementType.PLOT) has been temporarily removed because it depends on
# Home Assistant's Recorder component to fetch historical data. To include it in drawcustom,
# it needs to be refactored to accept pre-processed data arrays instead of fetching data directly.


@element_handler(ElementType.PROGRESS_BAR, requires=["x_start", "x_end", "y_start", "y_end", "progress"])
async def draw_progress_bar(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw progress bar with optional percentage text.

    Renders a progress bar to visualize a percentage value, with options
    for fill direction, colors, and text display.

    Args:
        ctx: Drawing context
        element: Element dictionary with progress bar properties
    """
    draw = ImageDraw.Draw(ctx.img)

    x_start = ctx.coords.parse_x(element["x_start"])
    y_start = ctx.coords.parse_y(element["y_start"])
    x_end = ctx.coords.parse_x(element["x_end"])
    y_end = ctx.coords.parse_y(element["y_end"])

    progress = min(100, max(0, element["progress"]))  # Clamp to 0-100
    direction = element.get("direction", "right")
    background = ctx.colors.resolve(element.get("background", "white"))
    fill = ctx.colors.resolve(element.get("fill", "red"))
    outline = ctx.colors.resolve(element.get("outline", "black"))
    width = element.get("width", 1)
    show_percentage = element.get("show_percentage", False)
    font_name = element.get("font_name", "ppb.ttf")

    # Draw background
    draw.rectangle(((x_start, y_start), (x_end, y_end)), fill=background, outline=outline, width=width)

    # Calculate progress dimensions
    if direction in ["right", "left"]:
        progress_width = int((x_end - x_start) * (progress / 100))
        progress_height = y_end - y_start
    else:  # up or down
        progress_width = x_end - x_start
        progress_height = int((y_end - y_start) * (progress / 100))

    # Draw progress
    if direction == "right":
        draw.rectangle((x_start, y_start, x_start + progress_width, y_end), fill=fill)
    elif direction == "left":
        draw.rectangle((x_end - progress_width, y_start, x_end, y_end), fill=fill)
    elif direction == "up":
        draw.rectangle((x_start, y_end - progress_height, x_end, y_end), fill=fill)
    elif direction == "down":
        draw.rectangle((x_start, y_start, x_end, y_start + progress_height), fill=fill)

    # Draw outline
    draw.rectangle((x_start, y_start, x_end, y_end), fill=None, outline=outline, width=width)

    # Add percentage text if enabled
    if show_percentage:
        # Calculate font size based on bar dimensions
        font_size = min(y_end - y_start - 4, x_end - x_start - 4, 20)
        font = ctx.fonts.get_font(font_name, font_size)

        percentage_text = f"{progress}%"

        # Get text dimensions
        text_bbox = draw.textbbox((0, 0), percentage_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center text
        text_x = (x_start + x_end - text_width) / 2
        text_y = (y_start + y_end - text_height) / 2

        # Choose text color based on position relative to progress
        if progress > 50:
            text_color = background
        else:
            text_color = fill

        draw.text((text_x, text_y), percentage_text, font=font, fill=text_color, anchor="lt")

    ctx.pos_y = y_end


@element_handler(ElementType.DIAGRAM, requires=["x", "height"])
async def draw_diagram(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw diagram with optional bars.

    Renders a basic diagram with axes and optional bar chart elements.

    Args:
        ctx: Drawing context
        element: Element dictionary with diagram properties
    """
    draw = ImageDraw.Draw(ctx.img)
    draw.fontmode = "1"

    # Get base properties
    pos_x = element["x"]
    height = element["height"]
    width = element.get("width", ctx.img.width)
    offset_lines = element.get("margin", 20)

    # Draw axes
    # X axis
    draw.line(
        [(pos_x + offset_lines, ctx.pos_y + height - offset_lines), (pos_x + width, ctx.pos_y + height - offset_lines)],
        fill=ctx.colors.resolve("black"),
        width=1,
    )
    # Y axis
    draw.line(
        [(pos_x + offset_lines, ctx.pos_y), (pos_x + offset_lines, ctx.pos_y + height - offset_lines)],
        fill=ctx.colors.resolve("black"),
        width=1,
    )

    if "bars" in element:
        bar_config = element["bars"]
        bar_margin = bar_config.get("margin", 10)
        bar_data = bar_config["values"].split(";")
        bar_count = len(bar_data)
        font_name = bar_config.get("font", "ppb.ttf")

        # Calculate bar width
        bar_width = math.floor((width - offset_lines - ((bar_count + 1) * bar_margin)) / bar_count)

        # Set up font for legends
        size = bar_config.get("legend_size", 10)
        font = ctx.fonts.get_font(font_name, size)
        legend_color = ctx.colors.resolve(bar_config.get("legend_color", "black"))

        # Find maximum value for scaling
        max_val = 0
        for bar in bar_data:
            try:
                name, value = bar.split(",", 1)
                max_val = max(max_val, int(value))
            except (ValueError, IndexError):
                continue

        if max_val == 0:
            ctx.pos_y = ctx.pos_y + height

        height_factor = (height - offset_lines) / max_val

        # Draw bars and legends
        for bar_pos, bar in enumerate(bar_data):
            try:
                name, value = bar.split(",", 1)
                value = int(value)

                # Calculate bar position
                x_pos = ((bar_margin + bar_width) * bar_pos) + offset_lines + pos_x

                # Draw legend
                draw.text(
                    (x_pos + (bar_width / 2), ctx.pos_y + height - offset_lines / 2),
                    str(name),
                    fill=legend_color,
                    font=font,
                    anchor="mm",
                )

                # Draw bar
                bar_height = height_factor * value
                draw.rectangle(
                    (
                        x_pos,
                        ctx.pos_y + height - offset_lines - bar_height,
                        x_pos + bar_width,
                        ctx.pos_y + height - offset_lines,
                    ),
                    fill=ctx.colors.resolve(bar_config["color"]),
                )

            except (ValueError, IndexError, KeyError) as e:
                raise ValueError(f"Invalid bar data: {e}") from e

    ctx.pos_y = ctx.pos_y + height
