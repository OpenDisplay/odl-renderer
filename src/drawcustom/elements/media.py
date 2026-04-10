from __future__ import annotations

import logging
from typing import Any

import qrcode  # type: ignore[import-untyped]
from PIL import Image
from resizeimage import resizeimage  # type: ignore[import-untyped]

from drawcustom.colors import BLACK
from drawcustom.media_loader import load_image
from drawcustom.registry import element_handler
from drawcustom.types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)


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
    border = element.get("border", 1)
    boxsize = element.get("boxsize", 2)

    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=boxsize,
            border=border,
        )

        # Add data and generate QR code
        qr.add_data(element["data"])
        qr.make(fit=True)

        # Create QR code image (convert RGBA to RGB for fill/back colors)
        qr_img = qr.make_image(fill_color=color[:3], back_color=bgcolor[:3])
        qr_img = qr_img.convert("RGBA")

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
        target_size = (element["xsize"], element["ysize"])
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

        # Convert to RGBA
        source_img = source_img.convert("RGBA")

        # Create temporary image for composition
        temp_img = Image.new("RGBA", ctx.img.size)
        temp_img.paste(source_img, (pos_x, pos_y), source_img)

        # Composite images
        img_composite = Image.alpha_composite(ctx.img, temp_img)
        ctx.img.paste(img_composite, (0, 0))

        # Update vertical position
        ctx.pos_y = pos_y + target_size[1]

    except Exception as err:
        raise ValueError(f"Failed to process image: {err}") from err
