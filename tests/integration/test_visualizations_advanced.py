"""Integration tests for progress_bar and diagram elements."""

import pytest

from drawcustom import generate_image


def progress_bar(**kwargs) -> dict:
    return {
        "type": "progress_bar",
        "x_start": 10,
        "y_start": 10,
        "x_end": 190,
        "y_end": 40,
        "progress": 50,
        **kwargs,
    }


@pytest.mark.asyncio
class TestProgressBarElement:
    async def test_basic_progress_bar(self):
        image = await generate_image(width=200, height=100, elements=[progress_bar()])
        assert image.size == (200, 100)

    async def test_progress_clamped_at_zero(self):
        image = await generate_image(width=200, height=100, elements=[progress_bar(progress=-10)])
        assert image.size == (200, 100)

    async def test_progress_clamped_at_100(self):
        image = await generate_image(width=200, height=100, elements=[progress_bar(progress=150)])
        assert image.size == (200, 100)

    @pytest.mark.parametrize("direction", ["right", "left", "up", "down"])
    async def test_progress_directions(self, direction):
        image = await generate_image(width=200, height=100, elements=[progress_bar(direction=direction)])
        assert image.size == (200, 100)

    async def test_show_percentage_above_50(self):
        image = await generate_image(width=200, height=100, elements=[progress_bar(progress=75, show_percentage=True)])
        assert image.size == (200, 100)

    async def test_show_percentage_below_50(self):
        image = await generate_image(width=200, height=100, elements=[progress_bar(progress=25, show_percentage=True)])
        assert image.size == (200, 100)

    async def test_custom_colors(self):
        image = await generate_image(
            width=200,
            height=100,
            elements=[progress_bar(fill="black", background="white", outline="red")],
        )
        assert image.size == (200, 100)


@pytest.mark.asyncio
class TestDiagramElement:
    async def test_basic_diagram_no_bars(self):
        image = await generate_image(
            width=200,
            height=150,
            elements=[{"type": "diagram", "x": 0, "height": 100}],
        )
        assert image.size == (200, 150)

    async def test_diagram_with_bars(self):
        image = await generate_image(
            width=200,
            height=150,
            elements=[
                {
                    "type": "diagram",
                    "x": 0,
                    "height": 100,
                    "bars": {
                        "values": "Mon,30;Tue,50;Wed,40",
                        "color": "black",
                        "margin": 5,
                        "legend_size": 8,
                    },
                }
            ],
        )
        assert image.size == (200, 150)

    async def test_diagram_custom_width(self):
        image = await generate_image(
            width=200,
            height=150,
            elements=[{"type": "diagram", "x": 0, "height": 80, "width": 150}],
        )
        assert image.size == (200, 150)
