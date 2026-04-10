from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)


async def load_image(
    source: str | bytes | Image.Image,
    session: aiohttp.ClientSession | None = None,
) -> Image.Image:
    """Load an image from various sources.

    Args:
        source: Image source, can be:
                - PIL Image object (returned as-is)
                - bytes (PNG/JPEG/etc - decoded with PIL)
                - HTTP/HTTPS URL (fetched and decoded)
                - Absolute file path (loaded from disk)
                - Data URI (data:image/png;base64, ...)
        session: Optional aiohttp ClientSession for HTTP requests.
                If provided, reuses existing session.
                If not provided, creates a temporary session for each request.

    Returns:
        PIL Image object

    Raises:
        ValueError: If a source is invalid or an image cannot be loaded
    """
    # Already a PIL Image - return as-is
    if isinstance(source, Image.Image):
        return source

    # Bytes - decode with PIL
    if isinstance(source, bytes):
        try:
            return Image.open(io.BytesIO(source))
        except Exception as err:
            raise ValueError(f"Failed to decode image from bytes: {err}") from err

    # String source - determine type
    if not isinstance(source, str):
        raise ValueError(f"Invalid image source type: {type(source)}")

    # HTTP/HTTPS URL
    if source.startswith("http://") or source.startswith("https://"):
        return await _load_from_http(source, session)

    # Data URI
    if source.startswith("data:"):
        return _load_from_data_uri(source)

    # Absolute file path
    if source.startswith("/"):
        return _load_from_file(source)

    raise ValueError(
        f"Invalid image source: {source}\n"
        f"Must be one of:\n"
        f"  - PIL Image object\n"
        f"  - bytes (raw image data)\n"
        f"  - HTTP/HTTPS URL\n"
        f"  - Absolute file path (starting with /)\n"
        f"  - Data URI (data:image/...)"
    )


async def _load_from_http(
    url: str,
    session: aiohttp.ClientSession | None = None,
) -> Image.Image:
    """Load image from HTTP/HTTPS URL.

    Args:
        url: HTTP/HTTPS URL
        session: Optional aiohttp session

    Returns:
        PIL Image object

    Raises:
        ValueError: If the image cannot be fetched or decoded
    """
    try:
        # Use provided session or create temporary one
        if session:
            async with session.get(url) as response:
                response.raise_for_status()
                image_bytes = await response.read()
        else:
            import aiohttp

            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(url) as response:
                    response.raise_for_status()
                    image_bytes = await response.read()

        return Image.open(io.BytesIO(image_bytes))

    except Exception as err:
        raise ValueError(f"Failed to load image from {url}: {err}") from err


def _load_from_file(file_path: str) -> Image.Image:
    """Load an image from a file path.

    Args:
        file_path: Absolute file path

    Returns:
        PIL Image object

    Raises:
        ValueError: If the file doesn't exist or cannot be loaded
    """
    try:
        return Image.open(file_path)
    except FileNotFoundError:
        raise ValueError(f"Image file not found: {file_path}")
    except Exception as err:
        raise ValueError(f"Failed to load image from {file_path}: {err}") from err


def _load_from_data_uri(data_uri: str) -> Image.Image:
    """Load image from data URI.

    Args:
        data_uri: Data URI (data:image/png;base64, ...)

    Returns:
        PIL Image object

    Raises:
        ValueError: If data URI is invalid or cannot be decoded
    """
    try:
        # Parse data URI: data:[<mediatype>][;base64],<data>
        if ";base64," not in data_uri:
            raise ValueError("Only base64-encoded data URIs are supported")

        # Extract base64 data
        _, base64_data = data_uri.split(";base64,", 1)

        # Decode base64
        image_bytes = base64.b64decode(base64_data)

        # Load with PIL
        return Image.open(io.BytesIO(image_bytes))

    except Exception as err:
        raise ValueError(f"Failed to load image from data URI: {err}") from err
