"""Icon element handlers using Material Design Icons.

This module provides icon rendering using the bundled MDI font.
Over 10,000 icons available at https://pictogrammers.com/library/mdi/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import ImageDraw, ImageFont

from odl_renderer.colors import BLACK
from odl_renderer.coordinates import coerce_number
from odl_renderer.registry import element_handler
from odl_renderer.types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)


def _build_mdi_index() -> dict[str, str]:
    """Load the MDI icon index from the bundled metadata file.

    The metadata is stored as a pre-flattened ``{name: codepoint}`` dict so this
    is a direct JSON load with no transformation. Called once at module import time
    so the blocking ``open()`` happens in an executor, not during an async render.
    """
    assets_dir = Path(__file__).parent.parent / "assets"
    metadata_path = assets_dir / "materialdesignicons-webfont_meta.json"

    try:
        with open(metadata_path, encoding="utf-8") as f:
            index: dict[str, str] = json.load(f)
    except Exception as err:
        raise ValueError(f"Failed to load MDI metadata: {err}") from err

    _LOGGER.debug("Loaded %d MDI icons", len(index))
    return index


# Load once at module import time — no blocking I/O during async renders.
_mdi_index: dict[str, str] = _build_mdi_index()

# Module-level font cache keyed by size — avoids reloading the TTF on every render.
_mdi_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_mdi_index() -> dict[str, str]:
    """Return the pre-loaded MDI icon index."""
    return _mdi_index


def _get_mdi_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a cached FreeType font for the given size, loading it on first use."""
    if size not in _mdi_font_cache:
        assets_dir = Path(__file__).parent.parent / "assets"
        font_path = assets_dir / "materialdesignicons-webfont.ttf"
        try:
            _mdi_font_cache[size] = ImageFont.truetype(str(font_path), size)
        except OSError as err:
            raise ValueError(f"Failed to load MDI font: {err}") from err
    return _mdi_font_cache[size]


@element_handler(ElementType.ICON, requires=["x", "y", "value", "size"])
async def draw_icon(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw Material Design Icon.

    Renders an icon from the bundled MDI font (10,000+ icons).

    Args:
        ctx: Drawing context
        element: Element dictionary with:
                - value: Icon name (e.g., "home", "cog")
                - x, y: Position (supports percentages)
                - size: Icon size in pixels
                - color or fill: Icon color (default: black)
                - anchor: Pillow text anchor (default: "la")
                - stroke_width: Stroke width in pixels (default: 0)
                - stroke_fill: Stroke color (default: "white")

    Example:
        {"type": "icon", "value": "home", "x": 50, "y": 50, "size": 48}
    """
    x = ctx.coords.parse_x(element["x"])
    y = ctx.coords.parse_y(element["y"])

    name = element["value"]
    if name.startswith("mdi:"):
        name = name[4:]

    index = _get_mdi_index()
    codepoint = index.get(name)
    if not codepoint:
        raise ValueError(f"Icon '{name}' not found. Search icons at https://pictogrammers.com/library/mdi/")
    char = chr(int(codepoint, 16))

    font = _get_mdi_font(ctx.coords.parse_size(element["size"], is_width=False))
    color = ctx.colors.resolve(element.get("color") or element.get("fill", "black")) or BLACK
    anchor = element.get("anchor", "la")
    stroke_width = int(coerce_number(element.get("stroke_width", 0), 0))
    stroke_fill = ctx.colors.resolve(element.get("stroke_fill", "white"))

    draw = ImageDraw.Draw(ctx.img)
    draw.text(
        (x, y),
        char,
        font=font,
        fill=color,
        anchor=anchor,
        fontmode="1",
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    bbox = draw.textbbox((x, y), char, font=font, anchor=anchor)
    ctx.pos_y = int(bbox[3])


@element_handler(ElementType.ICON_SEQUENCE, requires=["x", "y", "icons", "size"])
async def draw_icon_sequence(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw a sequence of MDI icons.

    Renders multiple icons in a row with consistent spacing.

    Args:
        ctx: Drawing context
        element: Element dictionary with:
                - icons: List of icon names (e.g., ["home", "cog"])
                - x, y: Starting position
                - size: Icon size in pixels
                - spacing: Space between icons (default: size/4)
                - direction: "right", "left", "up", or "down" (default: "right")
                - color or fill: Icon color (default: black)
                - anchor: Pillow text anchor (default: "la")
                - stroke_width: Stroke width in pixels (default: 0)
                - stroke_fill: Stroke color (default: "white")

    Example:
        {"type": "icon_sequence", "icons": ["home", "cog"], "x": 10, "y": 10, "size": 32}
    """
    x_start = ctx.coords.parse_x(element["x"])
    y_start = ctx.coords.parse_y(element["y"])

    size = ctx.coords.parse_size(element["size"], is_width=False)
    spacing = int(coerce_number(element.get("spacing", size // 4), size // 4))
    color = ctx.colors.resolve(element.get("color") or element.get("fill", "black")) or BLACK
    anchor = element.get("anchor", "la")
    stroke_width = int(coerce_number(element.get("stroke_width", 0), 0))
    stroke_fill = ctx.colors.resolve(element.get("stroke_fill", "white"))
    direction = element.get("direction", "right")

    font = _get_mdi_font(size)
    draw = ImageDraw.Draw(ctx.img)

    current_x, current_y = x_start, y_start
    max_x, max_y = x_start, y_start

    for name in element["icons"]:
        if name.startswith("mdi:"):
            name = name[4:]

        index = _get_mdi_index()
        codepoint = index.get(name)
        if not codepoint:
            _LOGGER.warning("Skipping unknown icon '%s'", name)
            continue
        char = chr(int(codepoint, 16))

        draw.text(
            (current_x, current_y),
            char,
            font=font,
            fill=color,
            anchor=anchor,
            fontmode="1",
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

        bbox = draw.textbbox((current_x, current_y), char, font=font, anchor=anchor)
        max_x = max(max_x, int(bbox[2]))
        max_y = max(max_y, int(bbox[3]))

        if direction == "right":
            current_x += size + spacing
        elif direction == "left":
            current_x -= size + spacing
        elif direction == "down":
            current_y += size + spacing
        elif direction == "up":
            current_y -= size + spacing

    ctx.pos_y = int(max_y)
