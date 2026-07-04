from __future__ import annotations

import logging
import re
from typing import Any, List, Tuple

from PIL import ImageDraw, ImageFont

from odl_renderer.colors import COLOR_TOKEN_PATTERN
from odl_renderer.registry import element_handler
from odl_renderer.types import DrawingContext, ElementType, TextSegment

_LOGGER = logging.getLogger(__name__)


@element_handler(ElementType.TEXT, requires=["x", "value"])
async def draw_text(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw (colored) text with optional wrapping or ellipsis.

    Renders text with support for multiple formatting options:

    - Color markup with [color]text[/color] syntax
    - Text wrapping based on max_width
    - Text truncation with ellipsis
    - Multiple anchoring options
    - Font selection and sizing

    Args:
        ctx: Drawing context
        element: Element dictionary with text properties
    """
    draw = ImageDraw.Draw(ctx.img)
    draw.fontmode = "1"

    x = ctx.coords.parse_x(element["x"])
    if "y" not in element:
        y = ctx.pos_y + element.get("y_padding", 10)
    else:
        y = ctx.coords.parse_y(element["y"])
    # Get text properties
    size = ctx.coords.parse_size(element.get("size", 20), is_width=False)
    font_name = element.get("font", "ppb.ttf")
    font = ctx.fonts.get_font(font_name, size)

    # Get alignment and default color
    align = element.get("align", "left")
    default_color = ctx.colors.resolve(element.get("color", "black"))
    anchor = element.get("anchor")
    spacing = element.get("spacing", 5)
    stroke_width = element.get("stroke_width", 0)
    stroke_fill = ctx.colors.resolve(element.get("stroke_fill", "white"))

    # Process text content
    text = str(element["value"])
    max_width = element.get("max_width")

    # Handle text wrapping if max_width is specified
    final_text = text
    if max_width is not None:
        if element.get("truncate", False):
            final_text = _truncate_to_width(text, font, max_width)
        else:
            final_text = _wrap_to_width(text, font, max_width)

    # Set appropriate anchor based on line count
    if not anchor:
        anchor = "la" if "\n" in final_text else "lt"

    # Draw the text
    if element.get("parse_colors", False):
        segments = parse_colored_text(final_text, element.get("color", "black"))

        # Check if text contains newlines
        has_newlines = "\n" in final_text

        if has_newlines:
            # Split text into lines
            seg_lines = split_segments_by_newlines(segments)

            # Calculate vertical positions
            line_y_positions, total_height = calculate_multiline_positions(seg_lines, font, spacing)

            # Apply vertical anchor offset to the entire block
            adjusted_y = calculate_anchor_offset_y(y, total_height, anchor)

            # Draw each line
            for line_segments, line_y_offset in zip(seg_lines, line_y_positions):
                # Calculate horizontal positions for this line
                line_segments, line_width = calculate_segment_positions(line_segments, font, x, align, anchor)

                # Calculate absolute y position for this line
                line_y = adjusted_y + line_y_offset

                # Draw each segment in the line
                for segment in line_segments:
                    color = ctx.colors.resolve(segment.color)
                    draw.text(
                        (segment.start_x, line_y),
                        segment.text,
                        fill=color,
                        font=font,
                        anchor="lt",
                        spacing=spacing,
                        stroke_width=stroke_width,
                        stroke_fill=stroke_fill,
                    )
            ctx.pos_y = adjusted_y + total_height
        else:
            # Apply the vertical anchor component (e.g. 'mm', 'mb') to the single
            # line too — Pillow honours it on the non-parse path, so parse_colors
            # must not silently drop it.
            line_height = _line_height(font)
            adjusted_y = calculate_anchor_offset_y(y, line_height, anchor)
            segments, total_width = calculate_segment_positions(segments, font, x, align, anchor)

            for segment in segments:
                color = ctx.colors.resolve(segment.color)
                draw.text(
                    (segment.start_x, adjusted_y),
                    segment.text,
                    fill=color,
                    font=font,
                    anchor="lt",
                    spacing=spacing,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
            ctx.pos_y = adjusted_y + line_height
    else:
        bbox = draw.textbbox((x, y), final_text, font=font, anchor=anchor, spacing=spacing, align=align)
        draw.text(
            (x, y),
            final_text,
            fill=default_color,
            font=font,
            anchor=anchor,
            align=align,
            spacing=spacing,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        ctx.pos_y = int(bbox[3])


@element_handler(ElementType.MULTILINE, requires=["x", "value", "delimiter", "offset_y"])
async def draw_multiline(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw multiline text with delimiter.

    Renders multiple lines of text separated by a delimiter character.
    Supports similar formatting options as the _draw_text method.

    Args:
        ctx: Drawing context
        element: Element dictionary with multiline text properties
    """
    draw = ImageDraw.Draw(ctx.img)
    draw.fontmode = "1"

    # Get text properties
    size = element.get("size", 20)
    font_name = element.get("font", "ppb.ttf")
    font = ctx.fonts.get_font(font_name, size)
    color = ctx.colors.resolve(element.get("color", "black"))
    align = element.get("align", "left")
    anchor = element.get("anchor", "lm")
    stroke_width = element.get("stroke_width", 0)
    stroke_fill = ctx.colors.resolve(element.get("stroke_fill", "white"))

    x = ctx.coords.parse_x(element["x"])
    # Support both 'y' (standard) and 'start_y' (legacy) for backward compatibility
    if "y" in element:
        current_y = ctx.coords.parse_y(element["y"])
    elif "start_y" in element:
        current_y = ctx.coords.parse_y(element["start_y"])
    else:
        current_y = ctx.pos_y + element.get("y_padding", 10)

    # Split text using delimiter
    lines = element["value"].replace("\n", "").split(element["delimiter"])

    max_y = current_y
    for line in lines:
        if element.get("parse_colors", False):
            segments = parse_colored_text(str(line), element.get("color", "black"))
            segments, total_width = calculate_segment_positions(segments, font, x, align, anchor)

            for segment in segments:
                color = ctx.colors.resolve(segment.color)
                draw.text(
                    (segment.start_x, current_y),
                    segment.text,
                    fill=color,
                    font=font,
                    anchor="lt",
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
        else:
            draw.text(
                (x, current_y),
                str(line),
                fill=color,
                font=font,
                anchor=anchor,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        current_y += element["offset_y"]
        max_y = current_y

    ctx.pos_y = max_y


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    """Return a line height in pixels using ascender/descender-bearing sample chars."""
    bbox = font.getbbox("Ay")
    return int(bbox[3] - bbox[1])


def _truncate_to_width(text: str, font: ImageFont.FreeTypeFont, max_width: float, ellipsis: str = "...") -> str:
    """Truncate *text* with an ellipsis so it fits within *max_width* pixels.

    Binary-searches the cut point (O(n log n)) instead of trimming one character at
    a time and re-measuring the whole string on each step (O(n^2)), which could take
    seconds on the event loop for a long templated string.

    Args:
        text: The text to fit.
        font: Font used to measure widths.
        max_width: Maximum width in pixels.
        ellipsis: String appended to indicate truncation.

    Returns:
        The original text if it already fits, otherwise the longest prefix plus the
        ellipsis that fits within ``max_width``.
    """
    if font.getlength(text) <= max_width:
        return text
    # Largest prefix length whose text + ellipsis still fits (monotonic in length).
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.getlength(text[:mid] + ellipsis) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ellipsis


def _wrap_to_width(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> str:
    """Word-wrap *text* to fit *max_width*, measuring each word only once.

    Accumulates precomputed word and space widths rather than re-measuring the whole
    current line for every word (which is quadratic per line).

    Args:
        text: The text to wrap.
        font: Font used to measure widths.
        max_width: Maximum line width in pixels.

    Returns:
        The text with ``\\n`` inserted at wrap points.
    """
    words = text.split()
    if not words:
        return ""
    space_width = font.getlength(" ")
    lines: list[str] = []
    current_words: list[str] = []
    current_width = 0.0
    for word in words:
        word_width = font.getlength(word)
        if not current_words:
            current_words = [word]
            current_width = word_width
        elif current_width + space_width + word_width <= max_width:
            current_words.append(word)
            current_width += space_width + word_width
        else:
            lines.append(" ".join(current_words))
            current_words = [word]
            current_width = word_width
    if current_words:
        lines.append(" ".join(current_words))
    return "\n".join(lines)


def get_wrapped_text(text: str, font: ImageFont.ImageFont, line_length: int) -> str:
    """Wrap text to fit within a given width.

    Breaks text into multiple lines to fit within the specified width.

    Args:
        text: Text to wrap
        font: Font to measure text width with
        line_length: Maximum line length in pixels

    Returns:
        str: Text with newlines inserted for wrapping
    """
    lines = [""]
    for word in text.split():
        line = f"{lines[-1]} {word}".strip()
        if font.getlength(line) <= line_length:
            lines[-1] = line
        else:
            lines.append(word)
    return "\n".join(lines)


def parse_colored_text(text: str, default_color: str = "black") -> List[TextSegment]:
    """Parse text with color markup into text segments.

    Breaks text with color markup like "[red]text[/red]" into segments
    with associated colors.

    Args:
        text: Text with color markup
        default_color: Color applied to text outside of any ``[color]`` markup
            (i.e. the element-level color).

    Returns:
        List[TextSegment]: List of text segments with colors
    """

    segments = []
    current_pos = 0
    # Valid color tokens (names + hex) are derived centrally in colors.py,
    # so new named colors need no change here.
    pattern = r"\[(" + COLOR_TOKEN_PATTERN + r")\](.*?)\[/\1\]"

    for match in re.finditer(pattern, text, re.DOTALL):
        # Add any text before the match with default color
        if match.start() > current_pos:
            segments.append(TextSegment(text=text[current_pos : match.start()], color=default_color))
        # Add the matched text with the specified color
        segments.append(TextSegment(text=match.group(2), color=match.group(1)))
        current_pos = match.end()

    # Add any remaining text with default color
    if current_pos < len(text):
        segments.append(TextSegment(text=text[current_pos:], color=default_color))

    return segments


def calculate_segment_positions(
    segments: List[TextSegment],
    font: ImageFont.FreeTypeFont,
    start_x: int,
    alignment: str = "left",
    anchor: str | None = None,
) -> Tuple[List[TextSegment], float]:
    """Calculate x positions for each text segment based on alignment.

    Determines the starting x position for each text segment based on
    the overall alignment and font metrics.

    Args:
        segments: List of text segments
        font: Font to measure text width with
        start_x: Base starting x position
        alignment: Text alignment (left, center, right)
        anchor: Anchor point for text

    Returns:
        tuple: (modified segments with positions, total width)
    """

    total_width = sum(font.getlength(segment.text) for segment in segments)

    current_x: float = start_x
    match alignment.lower():
        case "left":
            pass  # start_x is already correct
        case "center":
            current_x -= total_width / 2
        case "right":
            current_x -= total_width
        case _:
            # Default to left alignment for unknown values
            _LOGGER.warning("Unknown alignment '%s', defaulting to left", alignment)
    # Apply anchor-based horizontal offset
    if anchor:
        anchor_horizontal = anchor[0]  # First char: l/m/r
        if anchor_horizontal == "m":  # Middle
            current_x -= total_width / 2
        elif anchor_horizontal == "r":  # Right
            current_x -= total_width
        # else: left anchor, no adjustment needed

    for segment in segments:
        segment.start_x = int(current_x)
        current_x += font.getlength(segment.text)

    return segments, total_width


def split_segments_by_newlines(segments: List[TextSegment]) -> List[List[TextSegment]]:
    """
    Split text segments by newline characters into separate lines.

    Args:
        segments: List of text segments (may contain \\n characters)

    Returns:
        List of lines, where each line is a list of TextSegment objects.
    """
    lines: list[list[TextSegment]] = [[]]

    for segment in segments:
        if "\n" not in segment.text:
            # No newlines, add to current line
            lines[-1].append(segment)
        else:
            # Split segments by newlines
            parts = segment.text.split("\n")
            for i, part in enumerate(parts):
                if part:
                    lines[-1].append(TextSegment(text=part, color=segment.color))
                if i < len(parts) - 1:
                    lines.append([])

    # Remove empty lines
    return [line for line in lines if line]


def calculate_multiline_positions(
    lines: List[List[TextSegment]], font: ImageFont.FreeTypeFont, spacing: int
) -> Tuple[List[int], int]:
    """
    Calculate y positions for each line and total height.

    Args:
        lines: List of lines, where each line is a list of TextSegment objects.
        font: Font to measure text height with
        spacing: Spacing between lines in pixels

    Returns:
        tuple: (list of y positions for each line, total block height)
    """
    # Get line height from font metrics
    bbox = font.getbbox("Ay")  # Use chars with ascenders/descenders
    line_height = int(bbox[3] - bbox[1])

    # Calculate y positions
    line_positions = []
    current_y = 0

    for i in range(len(lines)):
        line_positions.append(current_y)
        current_y += line_height + spacing

    # Total height is position of last line + line height
    total_height = line_positions[-1] + line_height if line_positions else 0

    return line_positions, total_height


def calculate_anchor_offset_y(base_y: int, total_height: int, anchor: str | None) -> int:
    """
    Calculate y offset based on the vertical anchor component.

    Args:
        base_y: Base y coordinate from element
        total_height: Total height of text block
        anchor: Anchor string (e.g. 'mm', 'lt', 'rb')
    """
    if not anchor or len(anchor) < 2:
        return base_y

    anchor_vertical = anchor[1]
    if anchor_vertical == "m":
        return base_y - total_height // 2
    elif anchor_vertical == "b":
        return base_y - total_height
    return base_y
