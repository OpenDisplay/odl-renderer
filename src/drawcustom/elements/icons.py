"""Icon element handlers using Material Design Icons.

This module provides icon rendering using the bundled MDI font.
Over 10,000 icons available at https://pictogrammers.com/library/mdi/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from drawcustom.colors import BLACK
from drawcustom.registry import element_handler
from drawcustom.types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)

# MDI icon index cache (name -> codepoint)
_mdi_index: dict[str, str] | None = None


def _get_mdi_index() -> dict[str, str]:
    """Get MDI icon index (cached).

    Returns:
        Dictionary mapping icon names to codepoints

    Raises:
        ValueError: If metadata cannot be loaded
    """
    global _mdi_index

    if _mdi_index is not None:
        return _mdi_index

    # Load metadata
    assets_dir = Path(__file__).parent.parent / "assets"
    metadata_path = assets_dir / "materialdesignicons-webfont_meta.json"

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as err:
        raise ValueError(f"Failed to load MDI metadata: {err}") from err

    # Build index
    _mdi_index = {}
    for icon in metadata:
        name = icon.get("name")
        codepoint = icon.get("codepoint")
        if name and codepoint:
            _mdi_index[name] = codepoint
            # Index aliases too
            for alias in icon.get("aliases", []):
                if alias:
                    _mdi_index[alias] = codepoint

    _LOGGER.debug(f"Loaded {len(_mdi_index)} MDI icons")
    return _mdi_index


def _render_mdi_icon(name: str, size: int, color: tuple[int, int, int, int]) -> Image.Image:
    """Render MDI icon to PIL Image.

    Args:
        name: Icon name (e.g., "home", "cog")
        size: Icon size in pixels
        color: RGBA color tuple

    Returns:
        PIL Image with rendered icon

    Raises:
        ValueError: If icon not found or cannot be rendered
    """
    # Strip mdi: prefix if present
    if name.startswith("mdi:"):
        name = name[4:]

    # Find codepoint
    index = _get_mdi_index()
    codepoint = index.get(name)
    if not codepoint:
        raise ValueError(f"Icon '{name}' not found. Search icons at https://pictogrammers.com/library/mdi/")

    # Convert hex to character
    try:
        char = chr(int(codepoint, 16))
    except ValueError as err:
        raise ValueError(f"Invalid codepoint for icon '{name}'") from err

    # Load font
    assets_dir = Path(__file__).parent.parent / "assets"
    font_path = assets_dir / "materialdesignicons-webfont.ttf"

    try:
        font = ImageFont.truetype(str(font_path), size)
    except OSError as err:
        raise ValueError(f"Failed to load MDI font: {err}") from err

    # Render icon
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((size // 2, size // 2), char, font=font, fill=color, anchor="mm", fontmode="1")

    return img


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
                - anchor: Positioning anchor (default: "mm")

    Example:
        {"type": "icon", "value": "home", "x": 50, "y": 50, "size": 48}
    """
    # Parse coordinates
    x = ctx.coords.parse_x(element["x"])
    y = ctx.coords.parse_y(element["y"])

    # Get icon properties
    name = element["value"]
    size = element["size"]
    color = ctx.colors.resolve(element.get("color") or element.get("fill", "black")) or BLACK
    anchor = element.get("anchor", "mm")

    # Render icon
    icon_img = _render_mdi_icon(name, size, color)

    # Calculate paste position based on anchor
    if anchor == "mm":
        paste_x, paste_y = x - size // 2, y - size // 2
    elif anchor == "tl":
        paste_x, paste_y = x, y
    elif anchor == "tr":
        paste_x, paste_y = x - size, y
    elif anchor == "bl":
        paste_x, paste_y = x, y - size
    elif anchor == "br":
        paste_x, paste_y = x - size, y - size
    elif anchor == "mt":
        paste_x, paste_y = x - size // 2, y
    elif anchor == "mb":
        paste_x, paste_y = x - size // 2, y - size
    elif anchor == "lm":
        paste_x, paste_y = x, y - size // 2
    elif anchor == "rm":
        paste_x, paste_y = x - size, y - size // 2
    else:
        _LOGGER.warning(f"Unknown anchor '{anchor}', using top-left")
        paste_x, paste_y = x, y

    # Paste icon
    ctx.img.paste(icon_img, (paste_x, paste_y), icon_img)
    ctx.pos_y = paste_y + size


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
                - anchor: Positioning anchor (default: "mm")

    Example:
        {"type": "icon_sequence", "icons": ["home", "cog"], "x": 10, "y": 10, "size": 32}
    """
    # Parse start position
    x_start = ctx.coords.parse_x(element["x"])
    y_start = ctx.coords.parse_y(element["y"])

    # Get properties
    size = element["size"]
    spacing = element.get("spacing", size // 4)
    color = ctx.colors.resolve(element.get("color") or element.get("fill", "black")) or BLACK
    anchor = element.get("anchor", "mm")
    direction = element.get("direction", "right")

    current_x, current_y = x_start, y_start
    max_x, max_y = x_start, y_start

    # Draw each icon
    for name in element["icons"]:
        try:
            icon_img = _render_mdi_icon(name, size, color)
        except ValueError as err:
            _LOGGER.warning(f"Skipping icon '{name}': {err}")
            continue

        # Calculate paste position
        if anchor == "mm":
            paste_x, paste_y = current_x - size // 2, current_y - size // 2
        elif anchor == "tl":
            paste_x, paste_y = current_x, current_y
        else:
            paste_x, paste_y = current_x, current_y

        # Paste icon
        ctx.img.paste(icon_img, (paste_x, paste_y), icon_img)

        # Track bounds
        max_x = max(max_x, paste_x + size)
        max_y = max(max_y, paste_y + size)

        # Move to next position
        if direction == "right":
            current_x += size + spacing
        elif direction == "left":
            current_x -= size + spacing
        elif direction == "down":
            current_y += size + spacing
        elif direction == "up":
            current_y -= size + spacing

    ctx.pos_y = max_y
