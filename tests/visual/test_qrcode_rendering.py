from io import BytesIO

import pytest

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestQRCodeVisualRegression:
    """Visual regression tests for QR code rendering."""

    async def test_qrcode_url(self, snapshot_png):
        """Test QR code with URL."""
        image = await generate_image(
            width=200, height=200, elements=[E.qrcode("https://example.com", x=100, y=100, size=150)]
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_qrcode_text(self, snapshot_png):
        """Test QR code with plain text."""
        image = await generate_image(
            width=200, height=200, elements=[E.qrcode("Hello, World!", x=100, y=100, size=150)]
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    @pytest.mark.parametrize("size", [80, 120, 160])
    async def test_qrcode_sizes(self, snapshot_png, size):
        """Test QR codes at various sizes."""
        canvas_size = size + 40
        image = await generate_image(
            width=canvas_size,
            height=canvas_size,
            elements=[E.qrcode("test", x=canvas_size // 2, y=canvas_size // 2, size=size)],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png
