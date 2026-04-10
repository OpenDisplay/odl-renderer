"""Unit tests for media_loader — covers file, bytes, data URI, and error paths."""

from __future__ import annotations

import base64
import io
import tempfile

import pytest
from PIL import Image

from drawcustom.media_loader import load_image


def _make_png_bytes() -> bytes:
    """Create minimal valid PNG bytes."""
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_data_uri() -> str:
    png = _make_png_bytes()
    b64 = base64.b64encode(png).decode()
    return f"data:image/png;base64,{b64}"


@pytest.mark.asyncio
class TestLoadImage:
    async def test_pil_image_returned_as_is(self):
        img = Image.new("RGB", (10, 10))
        result = await load_image(img)
        assert result is img

    async def test_bytes_decoded(self):
        png_bytes = _make_png_bytes()
        result = await load_image(png_bytes)
        assert isinstance(result, Image.Image)

    async def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            await load_image(b"not an image")

    async def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid image source type"):
            await load_image(12345)  # type: ignore[arg-type]

    async def test_data_uri(self):
        uri = _make_data_uri()
        result = await load_image(uri)
        assert isinstance(result, Image.Image)

    async def test_data_uri_non_base64_raises(self):
        with pytest.raises(ValueError, match="Only base64"):
            await load_image("data:image/png,rawdata")

    async def test_file_path(self):
        png_bytes = _make_png_bytes()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            path = f.name
        result = await load_image(path)
        assert isinstance(result, Image.Image)

    async def test_nonexistent_file_raises(self):
        with pytest.raises(ValueError, match="not found"):
            await load_image("/nonexistent/path/image.png")

    async def test_invalid_source_string_raises(self):
        with pytest.raises(ValueError, match="Invalid image source"):
            await load_image("relative/path/image.png")
