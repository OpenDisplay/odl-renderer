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
from .types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)


async def generate_image(
    width: int,
    height: int,
    elements: list[dict[str, Any]],
    background: str = "white",
    accent_color: str = "red",
    session: aiohttp.ClientSession | None = None,
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
    fonts = FontManager()

    # Create base image
    img = Image.new("RGBA", (width, height), color=colors.resolve(background))

    # Create drawing context
    ctx = DrawingContext(
        img=img,
        colors=colors,
        coords=CoordinateParser(img.width, img.height),
        fonts=fonts,
        session=session,  # Pass session for HTTP image loading
        pos_y=0,
    )

    # Get all registered handlers
    draw_handlers = {element_type: handler for element_type, (handler, _) in get_all_handlers().items()}

    # Process each element
    for i, element in enumerate(elements):
        # Skip hidden elements
        if not element.get("visible", True):
            continue

        try:
            # Get element type
            if "type" not in element:
                raise ValueError("Element missing required 'type' field")
            element_type = ElementType(element["type"])

            # Get the appropriate handler and call it
            handler = draw_handlers.get(element_type)
            if handler:
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


def should_show_element(element: dict[str, Any]) -> bool:
    """Check if an element should be displayed.

    Elements can be hidden by setting visible=False in their definition.
    This is useful for conditional rendering.

    Args:
        element: Element dictionary

    Returns:
        bool: True if the element should be displayed, False otherwise
    """
    return bool(element.get("visible", True))
