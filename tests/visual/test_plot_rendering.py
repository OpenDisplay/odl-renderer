"""
Visual regression tests for the plot element.

Uses time-machine to freeze datetime.now() so the time window is deterministic,
and syrupy PNG snapshots to catch any rendering regression.

Run with --snapshot-update once to generate the reference snapshots.
"""

from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest
import time_machine

from drawcustom import generate_image

FROZEN_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _states(count: int, values: list[float]) -> list[dict]:
    """Build state records spanning the 24h window ending at FROZEN_NOW."""
    start = FROZEN_NOW - timedelta(hours=24)
    step = timedelta(hours=24 / count)
    return [{"state": str(v), "last_changed": (start + step * i).isoformat()} for i, v in enumerate(values)]


TEMP_STATES = _states(
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


class MockDataProvider:
    def __init__(self, data: dict[str, list[dict]]) -> None:
        self._data = data

    async def get_history(self, entity_ids, start, end):
        return {eid: self._data[eid] for eid in entity_ids if eid in self._data}


@pytest.mark.asyncio
class TestPlotVisualRegression:
    @time_machine.travel(FROZEN_NOW, tick=False)
    async def test_plot_basic(self, snapshot_png):
        """Basic line plot — black line, no axes or legend config."""
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "color": "black", "width": 2}],
                }
            ],
            data_provider=MockDataProvider({"sensor.temp": TEMP_STATES}),
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        assert buf.getvalue() == snapshot_png

    @time_machine.travel(FROZEN_NOW, tick=False)
    async def test_plot_with_axes_and_legend(self, snapshot_png):
        """Full config: y-legend, y-axis, x-legend, x-axis with dotted grid."""
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "color": "black", "width": 2}],
                    "ylegend": {"position": "left", "color": "black", "size": 8},
                    "yaxis": {"width": 1, "color": "black", "tick_every": 1, "grid": True, "grid_style": "dotted"},
                    "xlegend": {"format": "%H:%M", "interval": 21600, "size": 8, "position": "bottom"},
                    "xaxis": {"width": 1, "color": "black", "grid": True, "grid_style": "dotted"},
                }
            ],
            data_provider=MockDataProvider({"sensor.temp": TEMP_STATES}),
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        assert buf.getvalue() == snapshot_png

    @time_machine.travel(FROZEN_NOW, tick=False)
    async def test_plot_smooth(self, snapshot_png):
        """Catmull-Rom smoothing."""
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "color": "black", "smooth": True, "smooth_steps": 10}],
                }
            ],
            data_provider=MockDataProvider({"sensor.temp": TEMP_STATES}),
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        assert buf.getvalue() == snapshot_png

    @time_machine.travel(FROZEN_NOW, tick=False)
    async def test_plot_step_style(self, snapshot_png):
        """Step line style."""
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "color": "black", "line_style": "step"}],
                }
            ],
            data_provider=MockDataProvider({"sensor.temp": TEMP_STATES}),
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        assert buf.getvalue() == snapshot_png

    @time_machine.travel(FROZEN_NOW, tick=False)
    async def test_plot_right_legend(self, snapshot_png):
        """Y-legend on the right side."""
        image = await generate_image(
            width=296,
            height=128,
            elements=[
                {
                    "type": "plot",
                    "data": [{"entity": "sensor.temp", "color": "black"}],
                    "ylegend": {"position": "right", "color": "black", "size": 8},
                }
            ],
            data_provider=MockDataProvider({"sensor.temp": TEMP_STATES}),
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        assert buf.getvalue() == snapshot_png
