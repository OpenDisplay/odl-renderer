from io import BytesIO

import pytest

from drawcustom import generate_image
from tests.builders import ElementBuilder as E


@pytest.mark.asyncio
class TestLayoutVisualRegression:
    """Visual regression tests for complex layouts."""

    async def test_dashboard_layout(self, snapshot_png, ppb_font):
        """Test dashboard-style layout."""
        image = await generate_image(
            width=296,
            height=128,
            background="white",
            elements=[
                # Title
                E.text("Status Dashboard", x="50%", y=10, font=ppb_font, size=20, color="black", anchor="mt"),
                # Status indicators
                E.rectangle(x_start=10, y_start=40, x_end=70, y_end=80, fill="red"),
                E.text("Alert", x=40, y=60, font=ppb_font, size=12, color="white", anchor="mm"),
                E.rectangle(x_start=80, y_start=40, x_end=140, y_end=80, fill="green"),
                E.text("OK", x=110, y=60, font=ppb_font, size=12, color="white", anchor="mm"),
                E.rectangle(x_start=150, y_start=40, x_end=210, y_end=80, fill="blue"),
                E.text("Info", x=180, y=60, font=ppb_font, size=12, color="white", anchor="mm"),
                # QR code
                E.qrcode("https://status.example.com", x=250, y=60, size=50),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_layered_elements(self, snapshot_png):
        """Test layered overlapping elements."""
        image = await generate_image(
            width=200,
            height=200,
            background="white",
            elements=[
                E.rectangle(x_start=20, y_start=20, x_end=120, y_end=120, fill="red"),
                E.circle(x=100, y=100, radius=50, fill="blue"),
                E.rectangle(x_start=80, y_start=80, x_end=180, y_end=180, fill="green"),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png

    async def test_grid_layout(self, snapshot_png):
        """Test grid-style layout with multiple elements."""
        image = await generate_image(
            width=300,
            height=300,
            background="white",
            elements=[
                # 3x3 grid of colored squares
                E.rectangle(x_start=10, y_start=10, x_end=90, y_end=90, fill="red"),
                E.rectangle(x_start=110, y_start=10, x_end=190, y_end=90, fill="green"),
                E.rectangle(x_start=210, y_start=10, x_end=290, y_end=90, fill="blue"),
                E.rectangle(x_start=10, y_start=110, x_end=90, y_end=190, fill="yellow"),
                E.rectangle(x_start=110, y_start=110, x_end=190, y_end=190, fill="black"),
                E.rectangle(x_start=210, y_start=110, x_end=290, y_end=190, fill="white"),
                E.rectangle(x_start=10, y_start=210, x_end=90, y_end=290, fill="blue"),
                E.rectangle(x_start=110, y_start=210, x_end=190, y_end=290, fill="red"),
                E.rectangle(x_start=210, y_start=210, x_end=290, y_end=290, fill="green"),
            ],
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        assert buffer.getvalue() == snapshot_png
