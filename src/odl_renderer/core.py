from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from PIL import Image

from .colors import ColorResolver
from .coordinates import CoordinateParser

# Import handler modules to trigger decorator registration
from .elements import debug, icons, media, shapes, text, visualizations  # noqa: F401
from .fonts import FontManager
from .registry import get_all_handlers, get_resolver
from .transforms import apply_transform_region, has_transform
from .types import DataProvider, DrawingContext, ElementType, SourceGuard, SourceNotAllowedError

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
    allow_source: SourceGuard | None = None,
) -> Image.Image:
    """Generate image from drawing instructions.

    Pure rendering function that accepts data and returns a full-color PIL Image.
    No Home Assistant dependencies, no entity resolution, no dithering.

    Runs in two phases so the event loop stays responsive: network and history
    access is awaited on the loop first, then all decoding and drawing runs in a
    worker thread.

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
        allow_source: Optional policy ``allow_source(kind, source) -> bool`` consulted before
                      an element loads an HTTP/HTTPS URL (kind "url") or a file given by
                      absolute path, image or font (kind "file"). Returning False refuses
                      the source. Without it every URL and readable file can be loaded, so
                      pass one whenever elements come from an untrusted author. URL checks
                      run on the event loop; file checks run in the drawing thread.

    Returns:
        PIL.Image.Image in RGB or RGBA mode (full color, no dithering)

    Raises:
        ValueError: If canvas dimensions are invalid or element processing fails
        SourceNotAllowedError: If an element names a source that allow_source refuses,
                               or a font name that leaves the font directories. A
                               ValueError subclass carrying ``kind`` and ``source``.
    """
    # Validate dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid canvas dimensions: {width}x{height} (must be positive)")

    _LOGGER.debug("Generating image: %dx%d, background=%s, accent=%s", width, height, background, accent_color)

    resolved = await _resolve_elements(elements, session, data_provider, allow_source)
    return await asyncio.to_thread(
        _render, width, height, resolved, background, accent_color, session, data_provider, font_dirs, allow_source
    )


async def _resolve_elements(
    elements: list[dict[str, Any]],
    session: aiohttp.ClientSession | None,
    data_provider: DataProvider | None,
    allow_source: SourceGuard | None,
) -> list[dict[str, Any]]:
    """Run the I/O half of every element that has one, on the event loop.

    Returns a list aligned with *elements*, where each element that needs network
    or history access is replaced by a copy carrying the fetched data. Hidden
    elements and elements with a missing or unknown type are passed through
    untouched; the latter are reported by the drawing phase.
    """
    resolved: list[dict[str, Any]] = []
    for i, element in enumerate(elements):
        resolver = None
        if _coerce_visible(element.get("visible", True)):
            try:
                resolver = get_resolver(ElementType(element["type"]))
            except (KeyError, ValueError):
                pass
        if resolver is not None:
            try:
                element = await resolver(
                    element, session=session, data_provider=data_provider, allow_source=allow_source
                )
            except SourceNotAllowedError:
                raise
            except Exception as e:
                raise _element_error(i, element, e) from e
        resolved.append(element)
    return resolved


def _element_error(index: int, element: dict[str, Any], err: Exception) -> ValueError:
    """Log a failed element and return the ValueError to raise for it."""
    if isinstance(err, (ValueError, KeyError)):
        error_msg = f"Element {index + 1}: {str(err)}"
    else:
        error_msg = f"Element {index + 1} (type '{element.get('type', 'unknown')}'): {str(err)}"
    _LOGGER.error(error_msg)
    return ValueError(error_msg)


def _render(
    width: int,
    height: int,
    elements: list[dict[str, Any]],
    background: str,
    accent_color: str,
    session: aiohttp.ClientSession | None,
    data_provider: DataProvider | None,
    font_dirs: list[str] | None,
    allow_source: SourceGuard | None,
) -> Image.Image:
    """Draw already-resolved elements onto a new canvas.

    Synchronous and CPU-bound: generate_image() runs it in a worker thread.
    """
    # Initialize components
    colors = ColorResolver(accent_color)
    fonts = FontManager(font_dirs=font_dirs, allow_source=allow_source)

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
        allow_source=allow_source,
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
                    _render_transformed(ctx, handler, element)
                else:
                    handler(ctx, element)
            else:
                error_msg = f"No handler found for element type: {element_type}"
                _LOGGER.warning(error_msg)
                # Continue processing other elements

        except SourceNotAllowedError:
            raise
        except Exception as e:
            raise _element_error(i, element, e) from e

    # Return full-color PIL Image (caller handles dithering if needed)
    return img


def _render_transformed(ctx: DrawingContext, handler: Any, element: dict[str, Any]) -> None:
    """Render an element onto its own layer, transform it, then composite back.

    The element is drawn onto a transparent full-canvas layer by temporarily
    pointing the context at it, so the handler is used unchanged. Only the region
    the element occupies is then rotated/mirrored and alpha-composited onto the
    base image (cost proportional to the element, not the whole canvas). Mutating
    ``ctx.img`` in place preserves any ``ctx.pos_y`` flow updates the handler
    makes (e.g. text auto-flow).

    The layer is kept in the context and reused. After an element only the region it
    touched is cleared, so many transformed elements on a large canvas do not each
    allocate a whole new layer.
    """
    base = ctx.img
    if ctx.transform_layer is None:
        ctx.transform_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer = ctx.transform_layer
    ctx.img = layer
    try:
        handler(ctx, element)
    finally:
        ctx.img = base

    result = apply_transform_region(
        layer,
        rotation=element.get("rotation"),
        mirror=element.get("mirror"),
        pivot=element.get("pivot"),
        coords=ctx.coords,
    )
    if result is not None:
        transformed, offset = result
        base.alpha_composite(transformed, offset)
        # The region cropped for the transform holds everything the element drew.
        left, top = offset
        layer.paste((0, 0, 0, 0), (left, top, left + transformed.width, top + transformed.height))


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
