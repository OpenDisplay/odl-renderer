import pytest

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestMultiElementLayouts:
    """Test complex layouts with multiple elements."""

    async def test_dashboard_layout(self, ppb_font):
        """Test dashboard-style layout with multiple element types."""
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                # Title
                E.text("Dashboard", x="50%", y=10, font=ppb_font, size=20, anchor="mt"),
                # Status boxes
                E.rectangle(x_start=10, y_start=30, x_end=90, y_end=70, fill="red"),
                E.rectangle(x_start=100, y_start=30, x_end=180, y_end=70, fill="green"),
                E.rectangle(x_start=190, y_start=30, x_end=270, y_end=70, fill="blue"),
                # QR code
                E.qrcode("https://example.com", x=148, y=100, size=50),
            ],
        )

        assert image.size == (296, 128)

    async def test_element_layering(self):
        """Test elements render in correct order (later elements on top)."""
        image = await generate_image(
            width=200,
            height=200,
            elements=[
                E.rectangle(x_start=0, y_start=0, x_end=100, y_end=100, fill="red"),
                E.rectangle(x_start=50, y_start=50, x_end=150, y_end=150, fill="blue"),
                E.rectangle(x_start=100, y_start=100, x_end=200, y_end=200, fill="green"),
            ],
        )

        # Save for visual inspection
        # image.save("debug_multi_layer.png")

        # Check specific pixels
        assert image.getpixel((25, 25))[:3] == (255, 0, 0)  # Red only
        assert image.getpixel((76, 76))[:3] == (0, 0, 255)  # Blue over red
        assert image.getpixel((125, 125))[:3] == (0, 255, 0)  # Green over blue

    @pytest.mark.parametrize("num_elements", [1, 5, 10, 20])
    async def test_many_elements(self, num_elements):
        """Test rendering with various numbers of elements."""
        elements = [
            E.rectangle(
                x_start=i * 10,
                y_start=i * 10,
                x_end=i * 10 + 50,
                y_end=i * 10 + 50,
                fill="red" if i % 2 == 0 else "blue",
            )
            for i in range(num_elements)
        ]

        image = await generate_image(width=400, height=400, elements=elements)

        assert image.size == (400, 400)
