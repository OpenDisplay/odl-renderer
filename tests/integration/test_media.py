"""Integration tests for media elements: dlimg and qrcode error paths."""

from __future__ import annotations

import base64
import io
import tempfile

import pytest
from PIL import Image

from drawcustom import generate_image


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color="blue").save(buf, format="PNG")
    return buf.getvalue()


def _png_data_uri() -> str:
    return f"data:image/png;base64,{base64.b64encode(_png_bytes()).decode()}"


def _temp_png_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_png_bytes())
        return f.name


@pytest.mark.asyncio
class TestDownloadedImageElement:
    async def test_local_file(self):
        path = _temp_png_path()
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "dlimg", "x": 0, "y": 0, "url": path, "xsize": 20, "ysize": 20}],
        )
        assert image.size == (100, 100)

    async def test_pil_image_source(self):
        source_img = Image.new("RGB", (20, 20), color="green")
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "dlimg", "x": 0, "y": 0, "url": source_img, "xsize": 20, "ysize": 20}],
        )
        assert image.size == (100, 100)

    async def test_bytes_source(self):
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "dlimg", "x": 0, "y": 0, "url": _png_bytes(), "xsize": 20, "ysize": 20}],
        )
        assert image.size == (100, 100)

    async def test_data_uri_source(self):
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "dlimg", "x": 0, "y": 0, "url": _png_data_uri(), "xsize": 20, "ysize": 20}],
        )
        assert image.size == (100, 100)

    async def test_rotate(self):
        path = _temp_png_path()
        image = await generate_image(
            width=100,
            height=100,
            elements=[{"type": "dlimg", "x": 0, "y": 0, "url": path, "xsize": 20, "ysize": 20, "rotate": 90}],
        )
        assert image.size == (100, 100)

    @pytest.mark.parametrize("method", ["crop", "cover", "contain"])
    async def test_resize_methods(self, method):
        # Source must be larger than target for crop/cover to work
        buf = io.BytesIO()
        Image.new("RGB", (80, 80), color="blue").save(buf, format="PNG")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(buf.getvalue())
            path = f.name
        image = await generate_image(
            width=100,
            height=100,
            elements=[
                {"type": "dlimg", "x": 0, "y": 0, "url": path, "xsize": 40, "ysize": 30, "resize_method": method}
            ],
        )
        assert image.size == (100, 100)

    async def test_unknown_resize_method_falls_back(self):
        path = _temp_png_path()
        image = await generate_image(
            width=100,
            height=100,
            elements=[
                {"type": "dlimg", "x": 0, "y": 0, "url": path, "xsize": 20, "ysize": 20, "resize_method": "unknown"}
            ],
        )
        assert image.size == (100, 100)

    async def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="Failed to process image"):
            await generate_image(
                width=100,
                height=100,
                elements=[{"type": "dlimg", "x": 0, "y": 0, "url": "relative/bad.png", "xsize": 20, "ysize": 20}],
            )


@pytest.mark.asyncio
class TestQRCodeErrorPath:
    async def test_qrcode_exception_wrapped(self):
        """An invalid boxsize triggers the exception path in draw_qrcode."""
        with pytest.raises(ValueError, match="Failed to generate QR code"):
            await generate_image(
                width=100,
                height=100,
                elements=[{"type": "qrcode", "x": 0, "y": 0, "data": "test", "boxsize": -1}],
            )
