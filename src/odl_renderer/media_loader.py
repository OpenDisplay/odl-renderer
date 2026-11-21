from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

import aiohttp
from PIL import Image, ImageCms

from .types import SourceGuard, check_source

_LOGGER = logging.getLogger(__name__)

# Warn only once about creating a throwaway HTTP session, so a dashboard that
# re-renders every update doesn't repeat the message on every frame.
_warned_sessionless = False

# Limits on an HTTP image fetch, so a slow or oversized response cannot hold a
# render (and the memory behind it) indefinitely.
FETCH_TIMEOUT_SECONDS = 30
MAX_IMAGE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class FetchedImage:
    """Raw bytes of an image fetched over HTTP, not yet decoded.

    Fetching happens on the event loop; decoding is CPU work, so it is deferred to
    :func:`decode_image`, which can run in a worker thread.
    """

    url: str
    data: bytes


def is_http_url(source: object) -> bool:
    """Return True if *source* is an HTTP/HTTPS URL string."""
    return isinstance(source, str) and source.startswith(("http://", "https://"))


async def load_image(
    source: str | bytes | Image.Image,
    session: aiohttp.ClientSession | None = None,
    allow_source: SourceGuard | None = None,
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
        allow_source: Optional policy consulted before a URL is fetched or a file is opened.

    Returns:
        PIL Image object

    Raises:
        ValueError: If a source is invalid or an image cannot be loaded
        SourceNotAllowedError: If allow_source refuses the URL or file
    """
    if is_http_url(source):
        assert isinstance(source, str)
        return decode_image(await fetch_image(source, session, allow_source))
    return decode_image(source, allow_source)


def decode_image(
    source: str | bytes | Image.Image | FetchedImage,
    allow_source: SourceGuard | None = None,
) -> Image.Image:
    """Decode an image from any source that needs no network access.

    Blocking (file I/O and image decoding): call it from a worker thread, not the
    event loop. HTTP sources must be fetched first with :func:`fetch_image`.

    Args:
        source: PIL Image, bytes, a :class:`FetchedImage`, an absolute file path or a data URI.
        allow_source: Optional policy consulted before a file is opened.

    Returns:
        PIL Image object

    Raises:
        ValueError: If a source is invalid or an image cannot be loaded
        SourceNotAllowedError: If allow_source refuses the file
    """
    # Already a PIL Image - return as-is
    if isinstance(source, Image.Image):
        return source

    # Fetched over HTTP - decode, reporting failures against the URL
    if isinstance(source, FetchedImage):
        try:
            img = Image.open(io.BytesIO(source.data))
            img.load()  # decode now, not later mid-composite
            return img
        except Exception as err:
            raise ValueError(f"Failed to load image from {source.url}: {err}") from err

    # Bytes - decode with PIL
    if isinstance(source, bytes):
        try:
            return Image.open(io.BytesIO(source))
        except Exception as err:
            raise ValueError(f"Failed to decode image from bytes: {err}") from err

    # String source - determine type
    if not isinstance(source, str):
        raise ValueError(f"Invalid image source type: {type(source)}")

    # Data URI
    if source.startswith("data:"):
        return _load_from_data_uri(source)

    # Absolute file path
    if source.startswith("/"):
        check_source(allow_source, "file", source)
        return _load_from_file(source)

    if is_http_url(source):
        raise ValueError(f"Image URL was not fetched before decoding: {source}")

    raise ValueError(
        f"Invalid image source: {source}\n"
        f"Must be one of:\n"
        f"  - PIL Image object\n"
        f"  - bytes (raw image data)\n"
        f"  - HTTP/HTTPS URL\n"
        f"  - Absolute file path (starting with /)\n"
        f"  - Data URI (data:image/...)"
    )


def convert_to_srgb(img: Image.Image) -> Image.Image:
    """Convert an image to sRGB using its embedded ICC profile.

    Blocking (image decoding and colour conversion): call it from a worker thread.

    Args:
        img: Decoded or lazily opened image. It is not modified.

    Returns:
        A converted copy, or *img* itself if it has no profile, its profile describes
        itself as sRGB, or the profile cannot be applied (which logs a warning).

    Raises:
        OSError: If the image data itself cannot be decoded
    """
    icc_profile = img.info.get("icc_profile")
    if not icc_profile:
        return img

    img.load()  # outside the try: bad image data is not a bad profile
    try:
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        if "srgb" in ImageCms.getProfileDescription(source_profile).lower():
            return img
        return _apply_profile(img, source_profile)
    except (OSError, ValueError, TypeError, ImageCms.PyCMSError) as err:
        _LOGGER.warning("Ignoring ICC profile that could not be applied: %s", err)
        return img


def _apply_profile(img: Image.Image, source_profile: ImageCms.ImageCmsProfile) -> Image.Image:
    """Transform *img* from *source_profile* to sRGB, keeping any alpha channel."""

    def transform(image: Image.Image, output_mode: str) -> Image.Image:
        srgb = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(image, source_profile, srgb, outputMode=output_mode)
        assert converted is not None  # only None when converting in place
        return converted

    if img.mode == "LA":
        # The transform does not carry alpha across for this mode, so restore it afterwards.
        converted = transform(img.getchannel("L"), "RGB")
        converted.putalpha(img.getchannel("A"))
        return converted

    if img.mode in ("P", "PA"):
        has_alpha = img.mode == "PA" or "transparency" in img.info
        img = img.convert("RGBA" if has_alpha else "RGB")

    return transform(img, "RGBA" if img.mode == "RGBA" else "RGB")


async def _read_capped(response: aiohttp.ClientResponse) -> bytes:
    """Read a response body, refusing one larger than MAX_IMAGE_BYTES."""
    too_large = ValueError(f"image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    if (response.content_length or 0) > MAX_IMAGE_BYTES:
        raise too_large
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > MAX_IMAGE_BYTES:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch_image(
    url: str,
    session: aiohttp.ClientSession | None = None,
    allow_source: SourceGuard | None = None,
) -> FetchedImage:
    """Fetch an image's bytes from an HTTP/HTTPS URL without decoding them.

    The request is bounded by FETCH_TIMEOUT_SECONDS and the body by MAX_IMAGE_BYTES.

    Args:
        url: HTTP/HTTPS URL
        session: Optional aiohttp session
        allow_source: Optional policy consulted before the URL is requested.

    Returns:
        The fetched bytes, tagged with their URL

    Raises:
        ValueError: If the image cannot be fetched
        SourceNotAllowedError: If allow_source refuses the URL
    """
    check_source(allow_source, "url", url)
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
    try:
        # Use provided session or create temporary one
        if session:
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                image_bytes = await _read_capped(response)
        else:
            global _warned_sessionless
            if not _warned_sessionless:
                _warned_sessionless = True
                _LOGGER.warning(
                    "load_image called without a session; creating a temporary aiohttp "
                    "ClientSession per request. Pass a shared session for connection "
                    "pooling and DNS caching."
                )

            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(url, timeout=timeout) as response:
                    response.raise_for_status()
                    image_bytes = await _read_capped(response)

        return FetchedImage(url, image_bytes)

    except TimeoutError as err:
        # aiohttp raises a bare TimeoutError with no message when the total timeout expires.
        raise ValueError(f"Failed to load image from {url}: timed out after {FETCH_TIMEOUT_SECONDS} s") from err
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
        img = Image.open(file_path)
        img.load()  # force decode now so the file handle is released (no ResourceWarning)
        return img
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
