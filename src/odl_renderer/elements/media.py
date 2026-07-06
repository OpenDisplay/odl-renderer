from __future__ import annotations

import functools
import logging
from typing import Any

import qrcode  # type: ignore[import-untyped]
from PIL import Image
from resizeimage import resizeimage  # type: ignore[import-untyped]

from odl_renderer.colors import BLACK
from odl_renderer.coordinates import coerce_number
from odl_renderer.media_loader import load_image
from odl_renderer.registry import element_handler
from odl_renderer.types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=32)
def _render_qr_image(
    data: str,
    boxsize: int,
    border: int,
    fill: tuple[int, int, int],
    back: tuple[int, int, int],
) -> Image.Image:
    """Render (and cache) a QR code image.

    Generating a QR code costs ~5 ms, almost all in the library's best-mask-pattern
    search, and e-paper dashboards re-render the same QR (Wi-Fi creds, a URL) on
    every update. Caching by the inputs that determine the pixels turns the
    steady-state cost to ~0. The returned image must not be mutated by callers
    (it is composited read-only).
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=boxsize,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color=fill, back_color=back)
    rgba: Image.Image = qr_img.convert("RGBA")
    return rgba


@element_handler(ElementType.QRCODE, requires=["x", "y", "data"])
async def draw_qrcode(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw QR code element.

    Generates and renders a QR code with the specified data and properties.

    Args:
        ctx: Drawing context
        element: Element dictionary with QR code properties:
                - x, y: Position
                - data: QR code data (URL, text, etc.)
                - color: QR code color (default: black)
                - bgcolor: Background color (default: white)
                - border: Border size in boxes (default: 1)
                - boxsize: Size of each box in pixels (default: 2)

    Raises:
        ValueError: If QR code generation fails
    """
    # Coordinates
    x = ctx.coords.parse_x(element["x"])
    y = ctx.coords.parse_y(element["y"])

    # Get QR code properties
    color = ctx.colors.resolve(element.get("color", "black")) or BLACK
    bgcolor = ctx.colors.resolve(element.get("bgcolor", "white")) or BLACK
    border = int(coerce_number(element.get("border", 1), 1))
    boxsize = int(coerce_number(element.get("boxsize", 2), 2))

    try:
        # Render (or reuse a cached) QR code image for these exact inputs.
        qr_img = _render_qr_image(str(element["data"]), boxsize, border, color[:3], bgcolor[:3])

        # Paste QR code onto main image
        ctx.img.paste(qr_img, (x, y), qr_img)

        # Update vertical position
        ctx.pos_y = y + qr_img.height

    except Exception as err:
        raise ValueError(f"Failed to generate QR code: {err}") from err


@element_handler(ElementType.DLIMG, requires=["x", "y", "url", "xsize", "ysize"])
async def draw_downloaded_image(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw downloaded or local image.

    Loads and renders an image from various sources (HTTP URL, file path,
    data URI, PIL Image, or bytes).

    Args:
        ctx: Drawing context
        element: Element dictionary with image properties:
                - x, y: Position
                - url: Image source (HTTP URL, file path, data URI, PIL Image, or bytes)
                      Note: Entity IDs are NOT supported - caller must resolve to URL/bytes
                - xsize, ysize: Target size in pixels
                - rotate: Rotation angle in degrees (default: 0)
                - resize_method: "stretch", "crop", "cover", or "contain" (default: "stretch")

    Raises:
        ValueError: If image loading or processing fails

    Note:
        This element does NOT support Home Assistant entity IDs.
        Caller (HA integration) must resolve entity IDs to URLs or bytes before calling.
    """
    try:
        # Get image properties
        pos_x = ctx.coords.parse_x(element["x"])
        pos_y = ctx.coords.parse_y(element["y"])
        target_size = (
            ctx.coords.parse_size(element["xsize"], is_width=True),
            ctx.coords.parse_size(element["ysize"], is_width=False),
        )
        rotate = element.get("rotate", 0)
        resize_method = element.get("resize_method", "stretch")

        # Load image using media_loader
        # Pass session from context if available (for HA integration efficiency)
        session = getattr(ctx, "session", None)
        source_img = await load_image(element["url"], session=session)

        # Process image
        if rotate:
            source_img = source_img.rotate(-rotate, expand=True)

        # Resize if needed
        if source_img.size != target_size:
            if resize_method in ["crop", "cover", "contain"]:
                source_img = resizeimage.resize(resize_method, source_img, target_size)
            elif resize_method != "stretch":
                _LOGGER.warning(f"Unsupported resize_method '{resize_method}', using stretch resize")

            # Final resize to ensure exact target size
            if source_img.size != target_size:
                source_img = source_img.resize(target_size)

        # Convert to RGBA and composite in place — avoids allocating and blending
        # a full-canvas temporary image three times per dlimg. Clip to the canvas
        # first so a partially off-canvas image doesn't raise (alpha_composite
        # requires the source to fit within the destination).
        source_img = source_img.convert("RGBA")
        crop_left = max(0, -pos_x)
        crop_top = max(0, -pos_y)
        crop_right = min(source_img.width, ctx.img.width - pos_x)
        crop_bottom = min(source_img.height, ctx.img.height - pos_y)
        if crop_right > crop_left and crop_bottom > crop_top:
            if (crop_left, crop_top, crop_right, crop_bottom) != (0, 0, source_img.width, source_img.height):
                source_img = source_img.crop((crop_left, crop_top, crop_right, crop_bottom))
            ctx.img.alpha_composite(source_img, (pos_x + crop_left, pos_y + crop_top))

        # Update vertical position
        ctx.pos_y = pos_y + target_size[1]

    except Exception as err:
        raise ValueError(f"Failed to process image: {err}") from err
