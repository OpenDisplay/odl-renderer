from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from PIL import ImageDraw

from odl_renderer.registry import element_handler
from odl_renderer.types import DrawingContext, ElementType

_LOGGER = logging.getLogger(__name__)


def _fmt_value(v: int | float) -> str:
    """Format a numeric axis label, stripping unnecessary decimal places.

    Args:
        v: The numeric value to format.

    Returns:
        Whole-number string if the value is whole (e.g. ``18.0`` → ``"18"``),
        otherwise up to 2 decimal places with trailing zeros stripped
        (e.g. ``1.50`` → ``"1.5"``).
    """
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        rounded = round(v, 2)
        return f"{rounded:.2f}".rstrip("0").rstrip(".")
    return str(v)


def _draw_grid_line(
    draw: ImageDraw.ImageDraw,
    style: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw one axis grid line in the requested style.

    Direction (horizontal vs vertical) is inferred from the coordinates —
    a horizontal line has ``y1 == y2``; a vertical line has ``x1 == x2``.

    Args:
        draw: PIL ImageDraw instance.
        style: One of ``"lines"``, ``"dashed"``, or ``"dotted"``.
        x1, y1: Start coordinates.
        x2, y2: End coordinates.
        color: PIL fill colour.
    """
    if style == "lines":
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
    elif style == "dashed":
        dash_length, gap_length = 5, 3
        if y1 == y2:  # horizontal
            pos = x1
            while pos < x2:
                draw.line([(pos, y1), (min(pos + dash_length, x2), y1)], fill=color, width=1)
                pos += dash_length + gap_length
        else:  # vertical
            pos = y1
            while pos < y2:
                draw.line([(x1, pos), (x1, min(pos + dash_length, y2))], fill=color, width=1)
                pos += dash_length + gap_length
    elif style == "dotted":
        if y1 == y2:  # horizontal
            for x in range(int(x1), int(x2), 5):
                draw.point((x, y1), fill=color)
        else:  # vertical
            for y in range(int(y1), int(y2), 5):
                draw.point((x1, y), fill=color)


def _catmull_rom_point(
    p0: tuple[int, int],
    p1: tuple[int, int],
    p2: tuple[int, int],
    p3: tuple[int, int],
    t: float,
) -> tuple[int, int]:
    """Evaluate a Catmull-Rom spline at parameter *t*.

    Args:
        p0: Control point before the segment start.
        p1: Segment start point.
        p2: Segment end point.
        p3: Control point after the segment end.
        t: Interpolation parameter in ``[0, 1]``.

    Returns:
        Interpolated ``(x, y)`` pixel coordinate.
    """
    t2 = t * t
    t3 = t2 * t
    return (
        int(
            0.5
            * (
                (-t3 + 2 * t2 - t) * p0[0]
                + (3 * t3 - 5 * t2 + 2) * p1[0]
                + (-3 * t3 + 4 * t2 + t) * p2[0]
                + (t3 - t2) * p3[0]
            )
        ),
        int(
            0.5
            * (
                (-t3 + 2 * t2 - t) * p0[1]
                + (3 * t3 - 5 * t2 + 2) * p1[1]
                + (-3 * t3 + 4 * t2 + t) * p2[1]
                + (t3 - t2) * p3[1]
            )
        ),
    )


def _smooth_segment(
    points: list[tuple[int, int]],
    steps: int,
) -> list[tuple[int, int]]:
    """Smooth a screen-space polyline using Catmull-Rom spline interpolation.

    Args:
        points: Screen-space ``(x, y)`` pixel coordinates (at least 3 points).
        steps: Number of interpolation steps per control-point pair.

    Returns:
        Densified coordinate list suitable for passing directly to
        ``ImageDraw.line()``.
    """
    smooth_coords: list[tuple[int, int]] = [points[0]]

    if len(points) > 3:
        for i in range(1, steps):
            t = i / steps
            smooth_coords.append(_catmull_rom_point(points[0], points[0], points[1], points[2], t))

    for i in range(len(points) - 3):
        p0, p1, p2, p3 = points[i], points[i + 1], points[i + 2], points[i + 3]
        for j in range(steps):
            t = j / steps
            smooth_coords.append(_catmull_rom_point(p0, p1, p2, p3, t))

    if len(points) > 3:
        for i in range(1, steps):
            t = i / steps
            smooth_coords.append(_catmull_rom_point(points[-3], points[-2], points[-1], points[-1], t))

    smooth_coords.append(points[-1])
    return smooth_coords


def _process_entity_segments(
    plot: dict[str, Any],
    states: list[dict[str, Any]],
    min_v: float | None,
    max_v: float | None,
) -> tuple[list[list[tuple[datetime, float]]], float | None, float | None]:
    """Convert raw entity states into contiguous time-series segments.

    Splits the state history into segments, breaking on gaps or invalid
    (unavailable) values according to the ``span_gaps`` policy, and updates
    the running min/max bounds across all entities.

    Args:
        plot: Per-entity plot config dict (may include ``"span_gaps"`` and
              ``"value_scale"``).
        states: Raw state records from the data provider.
        min_v: Running minimum seen so far across all entities (``None`` if none yet).
        max_v: Running maximum seen so far across all entities (``None`` if none yet).

    Returns:
        ``(segments, updated_min_v, updated_max_v)`` where *segments* is a list
        of contiguous ``(timestamp, value)`` lists (may be empty if all states
        were invalid).
    """
    segments: list[list[tuple[datetime, float]]] = []
    current_segment: list[tuple[datetime, float]] = []
    span_gaps = plot.get("span_gaps", False)
    value_scale = plot.get("value_scale", 1.0)
    prev_timestamp: datetime | None = None
    prev_was_valid = True

    for state in states:
        try:
            value = float(state["state"]) * value_scale
            timestamp = datetime.fromisoformat(state["last_changed"])

            should_break = False
            if isinstance(span_gaps, (int, float)) and span_gaps is not True and span_gaps is not False:
                if prev_timestamp and (timestamp - prev_timestamp).total_seconds() > span_gaps:
                    should_break = True
            elif span_gaps is False and not prev_was_valid:
                should_break = True

            if should_break and current_segment:
                segments.append(current_segment)
                current_segment = []

            current_segment.append((timestamp, value))
            prev_timestamp = timestamp
            prev_was_valid = True

        except (ValueError, TypeError):
            if span_gaps is False and current_segment:
                segments.append(current_segment)
                current_segment = []
            prev_was_valid = False

    if current_segment:
        segments.append(current_segment)

    if not segments:
        return segments, min_v, max_v

    all_values = [p[1] for segment in segments for p in segment]
    min_v = min(all_values) if min_v is None else min(min_v, min(all_values))
    max_v = max(all_values) if max_v is None else max(max_v, max(all_values))

    return segments, min_v, max_v


def _parse_y_legend(
    element: dict[str, Any], ctx: DrawingContext, font_name: str, min_v: float, max_v: float
) -> SimpleNamespace:
    cfg = element.get("ylegend", {})
    if not cfg:
        return SimpleNamespace(enabled=False, width=0, pos=None, color=None, font=None)
    width = cfg.get("width", -1)
    pos = cfg.get("position", "left")
    if pos not in ("left", "right", None):
        pos = "left"
    color = ctx.colors.resolve(cfg.get("color", "black"))
    font = ctx.fonts.get_font(font_name, cfg.get("size", 10))
    if width == -1:
        bb_max = font.getbbox(_fmt_value(max_v))
        bb_min = font.getbbox(_fmt_value(min_v))
        width = math.ceil(max(bb_max[2] - bb_max[0], bb_min[2] - bb_min[0]))
    return SimpleNamespace(enabled=True, width=width, pos=pos, color=color, font=font)


def _parse_y_axis(element: dict[str, Any], ctx: DrawingContext) -> SimpleNamespace:
    cfg = element.get("yaxis")
    if not cfg:
        return SimpleNamespace(enabled=False, tick_every=0)
    tick_every = float(cfg.get("tick_every", 1))
    if tick_every <= 0:
        raise ValueError("yaxis tick_every must be greater than 0")
    return SimpleNamespace(
        enabled=True,
        width=cfg.get("width", 1),
        color=ctx.colors.resolve(cfg.get("color", "black")),
        tick_length=cfg.get("tick_length", 4),
        tick_width=cfg.get("tick_width", 2),
        tick_every=tick_every,
        grid=cfg.get("grid", True),
        grid_color=ctx.colors.resolve(cfg.get("grid_color", "black")),
        grid_style=cfg.get("grid_style", "dotted"),
    )


def _parse_x_legend(
    element: dict[str, Any], ctx: DrawingContext, font_name: str, duration: timedelta
) -> SimpleNamespace:
    interval = duration.total_seconds() / 4
    cfg = element.get("xlegend", {})
    if not cfg:
        return SimpleNamespace(
            enabled=False,
            interval=interval,
            height=0,
            position=None,
            font=None,
            color=None,
            time_format="%H:%M",
            snap_to_hours=True,
        )
    raw_interval = cfg.get("interval")
    if raw_interval is not None:
        interval = float(raw_interval)
    if interval <= 0:
        raise ValueError("xlegend interval must be greater than 0")
    position = cfg.get("position", "bottom")
    if position not in ("top", "bottom", None):
        position = "bottom"
    return SimpleNamespace(
        enabled=True,
        time_format=cfg.get("format", "%H:%M"),
        interval=interval,
        font=ctx.fonts.get_font(font_name, cfg.get("size", 10)),
        color=ctx.colors.resolve(cfg.get("color", "black")),
        position=position,
        height=cfg.get("height", -1),
        snap_to_hours=cfg.get("snap_to_hours", True),
    )


def _parse_x_axis(element: dict[str, Any], ctx: DrawingContext) -> SimpleNamespace:
    cfg = element.get("xaxis", {})
    if not cfg:
        return SimpleNamespace(
            enabled=False,
            width=1,
            color=None,
            tick_length=0,
            tick_width=0,
            grid=False,
            grid_color=None,
            grid_style="dotted",
        )
    return SimpleNamespace(
        enabled=True,
        width=cfg.get("width", 1),
        color=ctx.colors.resolve(cfg.get("color", "black")),
        tick_length=cfg.get("tick_length", 4),
        tick_width=cfg.get("tick_width", 2),
        grid=cfg.get("grid", True),
        grid_color=ctx.colors.resolve(cfg.get("grid_color", "black")),
        grid_style=cfg.get("grid_style", "dotted"),
    )


def _x_label_height(xlc: SimpleNamespace, xac: SimpleNamespace) -> int:
    if not xlc.enabled or xlc.height == 0:
        return 0
    if xlc.height > 0:
        return int(xlc.height)
    return int(xlc.font.getbbox("00:00")[3]) + (xac.tick_width if xac.enabled else 0) + 2


def _calc_diagram(
    x_start: int,
    y_start: int,
    width: int,
    height: int,
    ylc: SimpleNamespace,
    xlc: SimpleNamespace,
    label_h: int,
) -> SimpleNamespace:
    diag_x = x_start + (ylc.width if ylc.pos == "left" else 0)
    diag_y = y_start + (label_h if xlc.position == "top" and xlc.height != 0 else 0)
    diag_w = width - (ylc.width if ylc.pos in ("left", "right") else 0)
    return SimpleNamespace(x=diag_x, y=diag_y, width=diag_w, height=height - label_h)


def _x_time_range(xlc: SimpleNamespace, start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if xlc.enabled and xlc.height != 0 and xlc.snap_to_hours:
        curr = start.replace(minute=0, second=0, microsecond=0)
        end_t = end.replace(minute=0, second=0, microsecond=0)
        return curr, end_t + timedelta(hours=1) if end > end_t else end_t
    return start, end


def _render_y_legend(
    draw: ImageDraw.ImageDraw,
    ylc: SimpleNamespace,
    tick_every: float,
    min_v: float,
    max_v: float,
    spread: float,
    diag: SimpleNamespace,
    x_start: int,
    x_end: int,
    top_y: int,
    bottom_y: int,
) -> None:
    if tick_every > 0:
        curr, max_drawn = min_v, False
        while curr <= max_v:
            curr_y = round(diag.y + (1 - ((curr - min_v) / spread)) * (diag.height - 1))
            if ylc.pos == "left":
                draw.text((x_start, curr_y), _fmt_value(curr), fill=ylc.color, font=ylc.font, anchor="lm")
            elif ylc.pos == "right":
                draw.text((x_end, curr_y), _fmt_value(curr), fill=ylc.color, font=ylc.font, anchor="rm")
            if abs(curr - max_v) < 0.0001:
                max_drawn = True
            curr += tick_every
        if not max_drawn and abs(max_v - min_v) > 0.0001:
            max_y = round(diag.y + (1 - ((max_v - min_v) / spread)) * (diag.height - 1))
            if ylc.pos == "left":
                draw.text((x_start, max_y), _fmt_value(max_v), fill=ylc.color, font=ylc.font, anchor="lm")
            elif ylc.pos == "right":
                draw.text((x_end, max_y), _fmt_value(max_v), fill=ylc.color, font=ylc.font, anchor="rm")
    else:
        if ylc.pos == "left":
            draw.text((x_start, top_y), _fmt_value(max_v), fill=ylc.color, font=ylc.font, anchor="lt")
            draw.text((x_start, bottom_y), _fmt_value(min_v), fill=ylc.color, font=ylc.font, anchor="ls")
        elif ylc.pos == "right":
            draw.text((x_end, top_y), _fmt_value(max_v), fill=ylc.color, font=ylc.font, anchor="rt")
            draw.text((x_end, bottom_y), _fmt_value(min_v), fill=ylc.color, font=ylc.font, anchor="rs")


def _render_y_axis(
    draw: ImageDraw.ImageDraw,
    yac: SimpleNamespace,
    diag: SimpleNamespace,
    min_v: float,
    max_v: float,
    spread: float,
) -> None:
    if yac.width > 0 and yac.color:
        draw.rectangle((diag.x, diag.y, diag.x + yac.width - 1, diag.y + diag.height - 1), fill=yac.color)
    if yac.tick_length > 0 and yac.color:
        curr = min_v
        while curr <= max_v:
            curr_y = round(diag.y + (1 - ((curr - min_v) / spread)) * (diag.height - 1))
            draw.line((diag.x, curr_y, diag.x + yac.tick_length - 1, curr_y), fill=yac.color, width=yac.tick_width)
            curr += yac.tick_every
    if yac.grid and yac.grid_color:
        curr = min_v
        while curr <= max_v:
            curr_y = round(diag.y + (1 - ((curr - min_v) / spread)) * (diag.height - 1))
            _draw_grid_line(draw, yac.grid_style, diag.x, curr_y, diag.x + diag.width, curr_y, yac.grid_color)
            curr += yac.tick_every


def _render_x_axis(
    draw: ImageDraw.ImageDraw,
    xac: SimpleNamespace,
    xlc: SimpleNamespace,
    diag: SimpleNamespace,
    start: datetime,
    duration: timedelta,
    curr_time: datetime,
    end_time: datetime,
) -> None:
    if xac.width > 0 and xac.color:
        draw.line(
            [(diag.x, diag.y + diag.height), (diag.x + diag.width, diag.y + diag.height)],
            fill=xac.color,
            width=xac.width,
        )
    if xac.tick_length > 0 and xac.color:
        curr = curr_time
        while curr <= end_time:
            rel_x = (curr - start) / duration
            x = round(diag.x + rel_x * (diag.width - 1))
            if diag.x <= x <= diag.x + diag.width:
                draw.line(
                    [(x, diag.y + diag.height), (x, diag.y + diag.height - xac.tick_length)],
                    fill=xac.color,
                    width=xac.tick_width,
                )
            curr += timedelta(seconds=xlc.interval)
    if xac.grid and xac.grid_color:
        curr = curr_time
        while curr <= end_time:
            rel_x = (curr - start) / duration
            x = round(diag.x + rel_x * (diag.width - 1))
            if diag.x <= x <= diag.x + diag.width:
                _draw_grid_line(draw, xac.grid_style, x, diag.y, x, diag.y + diag.height, xac.grid_color)
            curr += timedelta(seconds=xlc.interval)


def _render_x_labels(
    draw: ImageDraw.ImageDraw,
    xlc: SimpleNamespace,
    xac: SimpleNamespace,
    diag: SimpleNamespace,
    start: datetime,
    duration: timedelta,
    curr_time: datetime,
    end_time: datetime,
    y_start: int,
) -> None:
    while curr_time <= end_time:
        rel_x = (curr_time - start) / duration
        x = round(diag.x + rel_x * (diag.width - 1))
        if diag.x <= x <= diag.x + diag.width:
            text = curr_time.astimezone().strftime(xlc.time_format)
            if xlc.position == "bottom":
                if xac.width > 0 and xac.color:
                    draw.line(
                        [(x, diag.y + diag.height), (x, diag.y + diag.height - xac.tick_width)],
                        fill=xac.color,
                        width=xac.width,
                    )
                draw.text(
                    (x, diag.y + diag.height + xac.tick_width + 2),
                    text,
                    fill=xlc.color,
                    font=xlc.font,
                    anchor="mt",
                )
            else:  # top
                if xac.width > 0 and xac.color:
                    draw.line([(x, diag.y), (x, diag.y + xac.tick_width)], fill=xac.color, width=xac.width)
                draw.text((x, y_start), text, fill=xlc.color, font=xlc.font, anchor="mt")
        curr_time += timedelta(seconds=xlc.interval)


def _render_series(
    draw: ImageDraw.ImageDraw,
    ctx: DrawingContext,
    raw_data: list[Any],
    plot_configs: list[dict[str, Any]],
    start: datetime,
    duration: timedelta,
    diag: SimpleNamespace,
    min_v: float,
    spread: float,
) -> None:
    for plot_segments, plot_config in zip(raw_data, plot_configs):
        line_color = ctx.colors.resolve(plot_config.get("color", "black"))
        line_width = plot_config.get("width", 1)
        smooth = plot_config.get("smooth", False)
        line_style = plot_config.get("line_style", "linear")
        steps = plot_config.get("smooth_steps", 10)

        all_screen_points: list[tuple[int, int]] = []
        for segment_data in plot_segments:
            points: list[tuple[int, int]] = []
            for timestamp, value in segment_data:
                rel_time = (timestamp - start) / duration
                rel_value = (value - min_v) / spread
                x = round(diag.x + rel_time * (diag.width - 1))
                y = round(diag.y + (1 - rel_value) * (diag.height - 1))
                points.append((x, y))
                all_screen_points.append((x, y))

            if len(points) > 1:
                if line_style == "step":
                    step_points = [points[0]]
                    for i in range(1, len(points)):
                        prev_x, prev_y = points[i - 1]
                        curr_x, curr_y = points[i]
                        step_points.append((curr_x, prev_y))
                        step_points.append((curr_x, curr_y))
                    points = step_points
                if smooth and len(points) > 2 and line_style != "step":
                    draw.line(_smooth_segment(points, steps), fill=line_color, width=line_width, joint="curve")
                else:
                    draw.line(points, fill=line_color, width=line_width)

        if plot_config.get("show_points", False):
            point_size = plot_config.get("point_size", 3)
            point_color = ctx.colors.resolve(plot_config.get("point_color", "black"))
            for x, y in all_screen_points:
                draw.ellipse([(x - point_size, y - point_size), (x + point_size, y + point_size)], fill=point_color)


@element_handler(ElementType.PLOT, requires=["data"])
async def draw_plot(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw a line plot of time-series sensor data.

    Requires a DataProvider in ctx.data_provider that supplies historical state
    records for each entity referenced in element["data"].

    Args:
        ctx: Drawing context — must have data_provider set.
        element: Element dictionary with plot properties.
    Raises:
        ValueError: If data_provider is missing, config is invalid, or no data is available.
    """
    if ctx.data_provider is None:
        raise ValueError("plot element requires a data_provider in DrawingContext")

    draw = ImageDraw.Draw(ctx.img)
    x_start = element.get("x_start", 0)
    y_start = element.get("y_start", 0)
    x_end = element.get("x_end", ctx.img.width - 1 - x_start)
    y_end = element.get("y_end", ctx.img.height - 1 - y_start)
    font_name = element.get("font", "ppb.ttf")

    duration_seconds = float(element.get("duration", 60 * 60 * 24))
    if duration_seconds <= 0:
        raise ValueError("plot duration must be greater than 0")
    duration = timedelta(seconds=duration_seconds)
    end = datetime.now(timezone.utc)
    start = end - duration

    # Fetch and process data
    min_v, max_v = element.get("low"), element.get("high")
    entity_ids = [p["entity"] for p in element["data"]]
    all_states = await ctx.data_provider.get_history(entity_ids, start, end)
    raw_data = []
    for plot in element["data"]:
        if plot["entity"] not in all_states:
            raise ValueError(f"no data returned for entity: {plot['entity']}")
        segments, min_v, max_v = _process_entity_segments(plot, all_states[plot["entity"]], min_v, max_v)
        if segments:
            raw_data.append(segments)
    if not raw_data:
        raise ValueError("plot has no valid data points")
    assert min_v is not None and max_v is not None  # guaranteed by non-empty raw_data
    if element.get("round_values", False):
        max_v, min_v = math.ceil(max_v), math.floor(min_v)
    if max_v == min_v:
        min_v -= 1
    spread = max_v - min_v

    # Parse axis/legend config and compute layout
    ylc = _parse_y_legend(element, ctx, font_name, min_v, max_v)
    yac = _parse_y_axis(element, ctx)
    xlc = _parse_x_legend(element, ctx, font_name, duration)
    xac = _parse_x_axis(element, ctx)
    label_h = _x_label_height(xlc, xac)
    diag = _calc_diagram(x_start, y_start, x_end - x_start + 1, y_end - y_start + 1, ylc, xlc, label_h)

    # Debug borders
    if element.get("debug", False):
        draw.rectangle((x_start, y_start, x_end, y_end), fill=None, outline=ctx.colors.resolve("black"), width=1)
        draw.rectangle(
            (diag.x, diag.y, diag.x + diag.width - 1, diag.y + diag.height - 1),
            fill=None,
            outline=ctx.colors.resolve("red"),
            width=1,
        )

    # Y legend and axis
    if ylc.enabled:
        shift = label_h if xlc.position == "top" and xlc.height != 0 else 0
        _render_y_legend(
            draw,
            ylc,
            yac.tick_every,
            min_v,
            max_v,
            spread,
            diag,
            x_start,
            x_end,
            y_start + shift,
            y_end - label_h + shift,
        )
    if yac.enabled:
        _render_y_axis(draw, yac, diag, min_v, max_v, spread)

    # X axis and labels
    curr_time, end_time = _x_time_range(xlc, start, end)
    if xac.enabled:
        _render_x_axis(draw, xac, xlc, diag, start, duration, curr_time, end_time)
    if xlc.enabled and xlc.height != 0:
        _render_x_labels(draw, xlc, xac, diag, start, duration, curr_time, end_time, y_start)

    # Data series
    _render_series(draw, ctx, raw_data, element["data"], start, duration, diag, min_v, spread)

    ctx.pos_y = y_end


@element_handler(ElementType.PROGRESS_BAR, requires=["x_start", "x_end", "y_start", "y_end", "progress"])
async def draw_progress_bar(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw progress bar with optional percentage text.

    Renders a progress bar to visualize a percentage value, with options
    for fill direction, colors, and text display.

    Args:
        ctx: Drawing context
        element: Element dictionary with progress bar properties
    """
    draw = ImageDraw.Draw(ctx.img)

    x_start = ctx.coords.parse_x(element["x_start"])
    y_start = ctx.coords.parse_y(element["y_start"])
    x_end = ctx.coords.parse_x(element["x_end"])
    y_end = ctx.coords.parse_y(element["y_end"])

    progress = min(100, max(0, element["progress"]))  # Clamp to 0-100
    direction = element.get("direction", "right")
    background = ctx.colors.resolve(element.get("background", "white"))
    fill = ctx.colors.resolve(element.get("fill", "red"))
    outline = ctx.colors.resolve(element.get("outline", "black"))
    width = element.get("width", 1)
    show_percentage = element.get("show_percentage", False)
    font_name = element.get("font_name", "ppb.ttf")

    # Draw background
    draw.rectangle(((x_start, y_start), (x_end, y_end)), fill=background, outline=outline, width=width)

    # Calculate progress dimensions
    if direction in ["right", "left"]:
        progress_width = int((x_end - x_start) * (progress / 100))
        progress_height = y_end - y_start
    else:  # up or down
        progress_width = x_end - x_start
        progress_height = int((y_end - y_start) * (progress / 100))

    # Draw progress
    if direction == "right":
        draw.rectangle((x_start, y_start, x_start + progress_width, y_end), fill=fill)
    elif direction == "left":
        draw.rectangle((x_end - progress_width, y_start, x_end, y_end), fill=fill)
    elif direction == "up":
        draw.rectangle((x_start, y_end - progress_height, x_end, y_end), fill=fill)
    elif direction == "down":
        draw.rectangle((x_start, y_start, x_end, y_start + progress_height), fill=fill)

    # Draw outline
    draw.rectangle((x_start, y_start, x_end, y_end), fill=None, outline=outline, width=width)

    # Add percentage text if enabled
    if show_percentage:
        # Calculate font size based on bar dimensions
        font_size = min(y_end - y_start - 4, x_end - x_start - 4, 20)
        font = ctx.fonts.get_font(font_name, font_size)

        percentage_text = f"{progress}%"

        # Get text dimensions
        text_bbox = draw.textbbox((0, 0), percentage_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center text
        text_x = (x_start + x_end - text_width) / 2
        text_y = (y_start + y_end - text_height) / 2

        # Choose text color based on position relative to progress
        if progress > 50:
            text_color = background
        else:
            text_color = fill

        draw.text((text_x, text_y), percentage_text, font=font, fill=text_color, anchor="lt")

    ctx.pos_y = y_end


@element_handler(ElementType.DIAGRAM, requires=["x", "height"])
async def draw_diagram(ctx: DrawingContext, element: dict[str, Any]) -> None:
    """Draw diagram with optional bars.

    Renders a basic diagram with axes and optional bar chart elements.

    Args:
        ctx: Drawing context
        element: Element dictionary with diagram properties
    """
    draw = ImageDraw.Draw(ctx.img)
    draw.fontmode = "1"

    # Get base properties
    pos_x = element["x"]
    height = element["height"]
    width = element.get("width", ctx.img.width)
    offset_lines = element.get("margin", 20)

    # Draw axes
    # X axis
    draw.line(
        [(pos_x + offset_lines, ctx.pos_y + height - offset_lines), (pos_x + width, ctx.pos_y + height - offset_lines)],
        fill=ctx.colors.resolve("black"),
        width=1,
    )
    # Y axis
    draw.line(
        [(pos_x + offset_lines, ctx.pos_y), (pos_x + offset_lines, ctx.pos_y + height - offset_lines)],
        fill=ctx.colors.resolve("black"),
        width=1,
    )

    if "bars" in element:
        bar_config = element["bars"]
        bar_margin = bar_config.get("margin", 10)
        bar_data = bar_config["values"].split(";")
        bar_count = len(bar_data)
        font_name = bar_config.get("font", "ppb.ttf")

        # Calculate bar width
        bar_width = math.floor((width - offset_lines - ((bar_count + 1) * bar_margin)) / bar_count)

        # Set up font for legends
        size = bar_config.get("legend_size", 10)
        font = ctx.fonts.get_font(font_name, size)
        legend_color = ctx.colors.resolve(bar_config.get("legend_color", "black"))

        # Find maximum value for scaling
        max_val = 0
        for bar in bar_data:
            try:
                name, value = bar.split(",", 1)
                max_val = max(max_val, int(value))
            except (ValueError, IndexError):
                continue

        if max_val == 0:
            ctx.pos_y = ctx.pos_y + height

        height_factor = (height - offset_lines) / max_val

        # Draw bars and legends
        for bar_pos, bar in enumerate(bar_data):
            try:
                name, value = bar.split(",", 1)
                value = int(value)

                # Calculate bar position
                x_pos = ((bar_margin + bar_width) * bar_pos) + offset_lines + pos_x

                # Draw legend
                draw.text(
                    (x_pos + (bar_width / 2), ctx.pos_y + height - offset_lines / 2),
                    str(name),
                    fill=legend_color,
                    font=font,
                    anchor="mm",
                )

                # Draw bar
                bar_height = height_factor * value
                draw.rectangle(
                    (
                        x_pos,
                        ctx.pos_y + height - offset_lines - bar_height,
                        x_pos + bar_width,
                        ctx.pos_y + height - offset_lines,
                    ),
                    fill=ctx.colors.resolve(bar_config["color"]),
                )

            except (ValueError, IndexError, KeyError) as e:
                raise ValueError(f"Invalid bar data: {e}") from e

    ctx.pos_y = ctx.pos_y + height
