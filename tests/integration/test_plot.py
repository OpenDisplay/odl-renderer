from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from odl_renderer import generate_image


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

    async def test_plot_naive_timestamps(self):
        """Naive (tz-less) last_changed values are treated as UTC, not crash (A12)."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=24)
        step = timedelta(hours=2)
        naive_states = [
            # isoformat() on a naive datetime has no offset suffix.
            {"state": str(20 + i), "last_changed": (start + step * i).replace(tzinfo=None).isoformat()}
            for i in range(12)
        ]
        provider = MockDataProvider({"sensor.temp": naive_states})
        image = await generate_image(
            width=296,
            height=128,
            elements=[{"type": "plot", "data": [{"entity": "sensor.temp"}]}],
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
