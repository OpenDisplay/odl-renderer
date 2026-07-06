from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from odl_renderer import generate_image
from odl_renderer.elements.visualizations import _smooth_segment


def make_states(count: int, values: list[float]) -> list[dict]:
    """Build state records spread over the last 24h, ending now."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)
    step = timedelta(hours=24 / count)
    return [{"state": str(v), "last_changed": (start + step * i).isoformat()} for i, v in enumerate(values)]


class MockDataProvider:
    """Minimal DataProvider for tests — no Home Assistant required."""

    def __init__(self, data: dict[str, list[dict]]) -> None:
        self._data = data

    async def get_history(self, entity_ids, start, end) -> dict[str, list[dict]]:
        return {eid: self._data[eid] for eid in entity_ids if eid in self._data}


TEMP_STATES = make_states(
    24,
    [
        20.0,
        20.5,
        21.0,
        22.0,
        21.5,
        20.0,
        19.5,
        19.0,
        18.5,
        18.0,
        17.5,
        17.0,
        17.5,
        18.0,
        19.0,
        20.0,
        21.0,
        22.5,
        23.0,
        22.0,
        21.0,
        20.5,
        20.0,
        19.5,
    ],
)


@pytest.mark.asyncio
class TestPlotElement:
    async def test_plot_renders_image(self):
        """Basic smoke test — plot with valid data returns an image."""
        provider = MockDataProvider({"sensor.temp": TEMP_STATES})
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "color": "black"}],
                }
            ],
            data_provider=provider,
        )
        assert isinstance(image, Image.Image)
        assert image.size == (296, 128)

    async def test_plot_missing_provider_raises(self):
        """Plot without a data_provider raises ValueError."""
        with pytest.raises(ValueError, match="data_provider"):
            await generate_image(
                width=296,
                height=128,
                elements=[
                    {
                        "type": "plot",
                        "data": [{"entity": "sensor.temp", "color": "black"}],
                    }
                ],
            )

    async def test_plot_missing_entity_raises(self):
        """Plot requesting an entity not in the provider output raises ValueError."""
        provider = MockDataProvider({})  # returns empty for all entities
        with pytest.raises(ValueError, match="sensor.temp"):
            await generate_image(
                width=296,
                height=128,
                elements=[
                    {
                        "type": "plot",
                        "data": [{"entity": "sensor.temp", "color": "black"}],
                    }
                ],
                data_provider=provider,
            )

    async def test_plot_invalid_duration_raises(self):
        """Negative duration raises ValueError."""
        provider = MockDataProvider({"sensor.temp": TEMP_STATES})
        with pytest.raises(ValueError, match="duration"):
            await generate_image(
                width=296,
                height=128,
                elements=[
                    {
                        "type": "plot",
                        "data": [{"entity": "sensor.temp"}],
                        "duration": -1,
                    }
                ],
                data_provider=provider,
            )

    async def test_plot_with_axes_and_legend(self):
        """Plot with yaxis, xaxis, ylegend, xlegend config renders without error."""
        provider = MockDataProvider({"sensor.temp": TEMP_STATES})
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "color": "black", "width": 2}],
                    "ylegend": {"position": "left", "color": "black", "size": 8},
                    "yaxis": {"width": 1, "color": "black", "tick_every": 2, "grid": True, "grid_style": "dotted"},
                    "xlegend": {"format": "%H:%M", "interval": 21600, "size": 8, "position": "bottom"},
                    "xaxis": {"width": 1, "color": "black", "grid": True, "grid_style": "dotted"},
                }
            ],
            data_provider=provider,
        )
        assert isinstance(image, Image.Image)

    async def test_plot_smooth(self):
        """Smooth Catmull-Rom rendering doesn't raise."""
        provider = MockDataProvider({"sensor.temp": TEMP_STATES})
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "smooth": True, "smooth_steps": 10}],
                }
            ],
            data_provider=provider,
        )
        assert isinstance(image, Image.Image)

    async def test_plot_skipped_entity_does_not_shift_config(self):
        """An entity with no valid data must not lend its style to the next series (A1).

        sensor.a is all-unavailable and configured red; sensor.b has real data and is
        configured black. Only sensor.b should draw, and it must be black — if the
        skipped entity shifted the config/segment pairing, sensor.b would render red.
        """
        end = datetime.now(timezone.utc)
        unavailable = [
            {"state": "unavailable", "last_changed": (end - timedelta(hours=24 - i)).isoformat()} for i in range(24)
        ]
        provider = MockDataProvider({"sensor.a": unavailable, "sensor.b": TEMP_STATES})
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [
                        {"entity": "sensor.a", "color": "red"},
                        {"entity": "sensor.b", "color": "black"},
                    ],
                }
            ],
            data_provider=provider,
        )
        colors = {c for _, c in image.convert("RGB").getcolors(maxcolors=100000)}
        assert (255, 0, 0) not in colors, "sensor.b was drawn with sensor.a's red style"

    async def test_plot_huge_spread_does_not_hang(self):
        """A pathological value spread with a grid must render quickly, not hang (A2)."""
        states = make_states(4, [0.0, 1_000_000.0, 500_000.0, 250_000.0])
        provider = MockDataProvider({"sensor.counter": states})
        image = await generate_image(
            width=200,
            height=100,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.counter"}],
                    "yaxis": {"grid": True},
                    "ylegend": {},
                }
            ],
            data_provider=provider,
        )
        assert isinstance(image, Image.Image)

    async def test_plot_step_style(self):
        """Step line style renders without error."""
        provider = MockDataProvider({"sensor.temp": TEMP_STATES})
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "line_style": "step"}],
                }
            ],
            data_provider=provider,
        )
        assert isinstance(image, Image.Image)

    async def test_plot_multiple_entities(self):
        """Multiple entities on one plot render without error."""
        humid_states = make_states(24, [float(50 + i % 10) for i in range(24)])
        provider = MockDataProvider(
            {
                "sensor.temp": TEMP_STATES,
                "sensor.humidity": humid_states,
            }
        )
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [
                        {"entity": "sensor.temp", "color": "black"},
                        {"entity": "sensor.humidity", "color": "red"},
                    ],
                }
            ],
            data_provider=provider,
        )
        assert isinstance(image, Image.Image)

    async def test_plot_span_gaps_numeric(self):
        """Time-based gap detection (span_gaps as seconds) doesn't crash."""
        gapped = TEMP_STATES[:8] + TEMP_STATES[16:]  # introduce a gap in the middle
        provider = MockDataProvider({"sensor.temp": gapped})
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "span_gaps": 7200}],
                }
            ],
            data_provider=provider,
        )
        assert isinstance(image, Image.Image)


def test_smooth_segment_three_points_keeps_middle():
    """A 3-point series must curve through its middle point, not collapse (A5)."""
    result = _smooth_segment([(0, 100), (50, 0), (100, 100)], 10)
    assert len(result) > 2, "smoothing collapsed the 3-point series to a straight line"
    assert min(y for _, y in result) < 50, "the middle peak/dip was dropped"
