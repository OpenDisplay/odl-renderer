import pytest

from odl_renderer import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestQRCodeRendering:
    """Test QR code element rendering."""

    @pytest.mark.parametrize(
        "qr_data",
        [
            "https://example.com",
            "Simple text",
            "mailto:test@example.com",
            "tel:+1234567890",
        ],
    )
    async def test_qrcode_various_data(self, qr_data):
        """Test QR code with various data types."""
        image = await generate_image(width=200, height=200, elements=[E.qrcode(qr_data, x=100, y=100, size=100)])

        assert image.size == (200, 200)

    @pytest.mark.parametrize("size", [50, 80, 100, 150])
    async def test_qrcode_various_sizes(self, size):
        """Test QR codes at various sizes."""
        canvas_size = size + 50  # Ensure QR fits
        image = await generate_image(
            width=canvas_size,
            height=canvas_size,
            elements=[E.qrcode("test", x=canvas_size // 2, y=canvas_size // 2, size=size)],
        )

        assert image.size == (canvas_size, canvas_size)

    async def test_qrcode_with_long_url(self):
        """Test QR code with long URL (high density)."""
        long_url = "https://example.com/very/long/path/with/many/segments/" + "x" * 100
        image = await generate_image(width=200, height=200, elements=[E.qrcode(long_url, x=100, y=100, size=150)])

        assert image.size == (200, 200)

    async def test_empty_qrcode_data(self):
        """Test QR code with empty data doesn't crash."""
        image = await generate_image(width=200, height=200, elements=[E.qrcode("", x=100, y=100, size=100)])

        assert image.size == (200, 200)


def test_qr_image_cache_reuses_render():
    """Identical QR inputs reuse a cached rendered image (B5)."""
    from odl_renderer.elements.media import _render_qr_image

    a = _render_qr_image("https://example.org", 2, 1, (0, 0, 0), (255, 255, 255))
    b = _render_qr_image("https://example.org", 2, 1, (0, 0, 0), (255, 255, 255))
    assert a is b  # cache hit — no regeneration

    c = _render_qr_image("https://different.org", 2, 1, (0, 0, 0), (255, 255, 255))
    assert c is not a  # different data → distinct render
