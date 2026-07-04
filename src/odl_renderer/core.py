from __future__ import annotations

import logging
from typing import Any

import aiohttp
from PIL import Image

from .colors import ColorResolver
from .coordinates import CoordinateParser

# Import handler modules to trigger decorator registration
from .elements import debug, icons, media, shapes, text, visualizations  # noqa: F401
from .fonts import FontManager
from .registry import get_all_handlers
from .transforms import apply_transform, has_transform
from .types import DataProvider, DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)


async def generate_image(
    width: int,
    height: int,
    elements: list[dict[str, Any]],
    background: str = "white",
    accent_color: str = "red",
    session: aiohttp.ClientSession | None = None,
    data_provider: DataProvider | None = None,
    font_dirs: list[str] | None = None,
) -> Image.Image:
    """Generate image from drawing instructions.

    Pure rendering function that accepts data and returns a full-color PIL Image.
    No Home Assistant dependencies, no entity resolution, no dithering.

    Args:
        width: Canvas width in pixels
        height: Canvas height in pixels
        elements: List of element configurations (dictionaries)
        background: Background color (name, hex, RGB tuple, etc.)
        accent_color: Accent color name - used when element specifies color="accent"
                     Common values: "red" (default), "yellow", "black"
                     Based on e-paper display capabilities
        session: Optional aiohttp.ClientSession for HTTP image requests
                 If provided, reuses existing session (efficient for HA integration)
                 If not provided, creates temporary session for each request
        font_dirs: Optional list of directories to search for fonts by name,
                   in priority order, before falling back to bundled fonts.
                   Useful for providing host-specific font locations (e.g. /config/www/fonts).

    Returns:
        PIL.Image.Image in RGB or RGBA mode (full color, no dithering)

    Raises:
        ValueError: If canvas dimensions are invalid or element processing fails
    """
    # Validate dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid canvas dimensions: {width}x{height} (must be positive)")

    _LOGGER.debug("Generating image: %dx%d, background=%s, accent=%s", width, height, background, accent_color)

    # Initialize components
    colors = ColorResolver(accent_color)
    fonts = FontManager(font_dirs=font_dirs)

    # Create base image
    img = Image.new("RGBA", (width, height), color=colors.resolve(background))

    # Create drawing context
    ctx = DrawingContext(
        img=img,
        colors=colors,
        coords=CoordinateParser(img.width, img.height),
        fonts=fonts,
        session=session,
        data_provider=data_provider,
        pos_y=0,
    )

    # Get all registered handlers
    draw_handlers = {element_type: handler for element_type, (handler, _) in get_all_handlers().items()}

    # Process each element
    for i, element in enumerate(elements):
        # Skip hidden elements
        if not _coerce_visible(element.get("visible", True)):
            continue

        try:
            # Get element type
            if "type" not in element:
                raise ValueError("Element missing required 'type' field")
            element_type = ElementType(element["type"])

            # Get the appropriate handler and call it
            handler = draw_handlers.get(element_type)
            if handler:
                if has_transform(element):
                    await _render_transformed(ctx, handler, element)
                else:
                    await handler(ctx, element)
            else:
                error_msg = f"No handler found for element type: {element_type}"
                _LOGGER.warning(error_msg)
                # Continue processing other elements

        except (ValueError, KeyError) as e:
            error_msg = f"Element {i + 1}: {str(e)}"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Element {i + 1} (type '{element.get('type', 'unknown')}'): {str(e)}"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg) from e

    # Return full-color PIL Image (caller handles dithering if needed)
    return img


async def _render_transformed(ctx: DrawingContext, handler: Any, element: dict[str, Any]) -> None:
    """Render an element onto its own layer, transform it, then composite back.

    The element is drawn onto a transparent full-canvas layer by temporarily
    pointing the context at it, so the handler is used unchanged. The layer is
    then rotated/mirrored and alpha-composited onto the base image. Mutating
    ``ctx.img`` in place preserves any ``ctx.pos_y`` flow updates the handler
    makes (e.g. text auto-flow).
    """
    base = ctx.img
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ctx.img = layer
    try:
        await handler(ctx, element)
    finally:
        ctx.img = base

    layer = apply_transform(
        layer,
        rotation=element.get("rotation"),
        mirror=element.get("mirror"),
        pivot=element.get("pivot"),
        coords=ctx.coords,
    )
    base.alpha_composite(layer)


_FALSY_VISIBLE_STRINGS = frozenset({"false", "", "0", "no", "off", "none"})


def _coerce_visible(value: Any) -> bool:
    """Coerce a `visible` field value to a boolean.

    Lenient by design: Home Assistant templates render to strings, so values
    like "false"/"False" must read as hidden even though any non-empty string
    is normally truthy. Whitespace-only strings fold to False, preserving the
    "render an empty string to hide" workaround.

    Args:
        value: Raw `visible` value (bool, str, number, or other).

    Returns:
        bool: True if the element should be shown, False if hidden.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSY_VISIBLE_STRINGS
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return bool(value)


def should_show_element(element: dict[str, Any]) -> bool:
    """Check if an element should be displayed.

    Elements can be hidden by setting visible=False in their definition.
    This is useful for conditional rendering.

    Args:
        element: Element dictionary

    Returns:
        bool: True if the element should be displayed, False otherwise
    """
    return _coerce_visible(element.get("visible", True))
