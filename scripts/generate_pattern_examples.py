"""Generate the example pictures for the `pattern` field.

Each picture is written to docs/screenshots/patterns/. They show what the pattern types
look like, how a tone compares with a gray under dithering, how patterns layer with fills,
outlines and rounded corners, and how every shape takes a pattern.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_screenshots import MockDataProvider

from odl_renderer import generate_image

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "screenshots" / "patterns"

Element = dict[str, Any]


def rectangle(x: int, y: int, w: int, h: int, **fields: Any) -> Element:
    """Return a rectangle given its corner and size (`width` stays the outline width)."""
    return {
        "type": "rectangle",
        "x_start": x,
        "y_start": y,
        "x_end": x + w - 1,
        "y_end": y + h - 1,
        **fields,
    }


def text(value: str, x: int, y: int, size: int = 12, **fields: Any) -> Element:
    """Return a text element anchored at its middle top."""
    return {"type": "text", "value": value, "x": x, "y": y, "size": size, "anchor": "mt", **fields}


async def render(elements: list[Element], width: int, height: int) -> Image.Image:
    """Render elements on a white canvas."""
    image: Image.Image = await generate_image(width=width, height=height, elements=elements, background="white")
    return image.convert("RGB")


def save(image: Image.Image, name: str) -> None:
    """Write a picture with a thin border, so it stands out on a white page."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ImageOps.expand(image, border=1, fill="black").save(OUTPUT_DIR / f"{name}.png")
    print(f"  OK  {name}.png")


def pattern_types() -> list[Element]:
    """Every pattern type, angle and a few parameters, each with a caption."""
    cells: list[tuple[str, dict[str, Any] | str]] = [
        ("hatch (45)", "hatch"),
        ("hatch angle 135", {"type": "hatch", "angle": 135}),
        ("hatch angle 0", {"type": "hatch", "angle": 0}),
        ("hatch angle 90", {"type": "hatch", "angle": 90}),
        ("hatch angle 30", {"type": "hatch", "angle": 30, "spacing": 7}),
        ("hatch width 3", {"type": "hatch", "width": 3, "spacing": 9}),
        ("dots", "dots"),
        ("dots width 3", {"type": "dots", "width": 3, "spacing": 9}),
        ("grid", {"type": "grid", "spacing": 8}),
        ("tone level 25", {"type": "tone", "level": 25}),
        ("tone level 50", {"type": "tone", "level": 50}),
        ("tone level 75", {"type": "tone", "level": 75}),
        ("red hatch on yellow", {"type": "hatch", "color": "red", "width": 2, "spacing": 6}),
        ("red dots", {"type": "dots", "color": "red", "width": 2, "spacing": 7}),
        ("red grid", {"type": "grid", "color": "red", "spacing": 10}),
        ("red tone", {"type": "tone", "level": 40, "color": "red"}),
    ]
    elements: list[Element] = []
    for index, (caption, pattern) in enumerate(cells):
        x = 20 + (index % 4) * 190
        y = 20 + (index // 4) * 118
        fill = "yellow" if caption == "red hatch on yellow" else None
        elements.append(rectangle(x, y, 170, 80, radius=8, corners="all", fill=fill, pattern=pattern))
        elements.append(text(caption, x + 85, y + 86, 13))
    return elements


def layers() -> list[Element]:
    """Fill, pattern and outline in order, rounded corners, alpha and shapes that touch."""
    return [
        rectangle(
            20,
            20,
            220,
            120,
            fill="yellow",
            radius=20,
            corners="all",
            outline="black",
            width=3,
            pattern={"type": "hatch", "color": "red", "width": 2, "spacing": 8},
        ),
        text("fill, pattern, then outline", 130, 148, 13),
        rectangle(
            260,
            20,
            220,
            120,
            radius=40,
            corners="top_left,bottom_right",
            width=2,
            pattern={"type": "hatch", "angle": 0, "spacing": 5},
        ),
        text("clipped to the rounded corners", 370, 148, 13),
        rectangle(
            500,
            20,
            280,
            120,
            fill="#DDDDFF",
            outline=None,
            pattern={"type": "hatch", "color": "#FF000090", "width": 5, "spacing": 14},
        ),
        text("color with alpha blends", 640, 148, 13),
        rectangle(20, 200, 105, 120, outline=None, pattern="hatch"),
        rectangle(125, 200, 105, 120, outline=None, pattern="hatch"),
        rectangle(230, 200, 105, 120, outline=None, pattern="hatch"),
        text("three shapes that touch: one pattern, no seams", 177, 328, 13),
        rectangle(
            380, 200, 400, 120, fill="white", pattern={"type": "dots", "spacing": 6, "width": 2}, outline="black"
        ),
        text("no fill: the red below shows between the white lines", 580, 328, 13),
        rectangle(400, 220, 360, 40, fill="red", outline=None),
        rectangle(
            400,
            220,
            360,
            80,
            outline=None,
            pattern={"type": "hatch", "angle": 0, "spacing": 4, "width": 2, "color": "white"},
        ),
    ]


def dithering() -> list[Element]:
    """Grays that a display dithers next to patterns that are already exact pixels."""
    elements: list[Element] = []
    for index, color in enumerate(("dark_gray", "gray", "light_gray")):
        elements.append(rectangle(20 + index * 130, 20, 110, 90, fill=color, outline="black"))
        elements.append(text(color, 75 + index * 130, 116, 13))
    for index, level in enumerate((75, 50, 25)):
        elements.append(
            rectangle(
                420 + index * 130,
                20,
                110,
                90,
                outline="black",
                pattern={"type": "tone", "level": level},
            )
        )
        elements.append(text(f"tone {level}", 475 + index * 130, 116, 13))
    elements.append(rectangle(20, 150, 360, 40, outline="black", pattern="hatch"))
    elements.append(rectangle(420, 150, 360, 40, outline="black", pattern={"type": "dots", "spacing": 5}))
    return elements


async def dithering_picture() -> Image.Image:
    """Stack the picture as rendered over the same picture after 1-bit dithering."""
    rendered = await render(dithering(), 800, 210)
    dithered = rendered.convert("1").convert("RGB")
    sheet = Image.new("RGB", (800, 470), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 2), "As rendered", fill="black")
    sheet.paste(rendered, (0, 14))
    draw.text((10, 236), "After 1-bit Floyd-Steinberg dithering: the grays change, the patterns do not", fill="black")
    sheet.paste(dithered, (0, 248))
    return sheet


def shape(kind: str, **fields: Any) -> Element:
    """Return an element of any type from its fields."""
    return {"type": kind, **fields}


def shapes_sheet() -> list[Element]:
    """Every shape that takes a pattern, each with a caption."""
    fills = {"fill": "yellow"}
    red_hatch = {"type": "hatch", "color": "red", "angle": 135, "width": 2, "spacing": 7}
    star = [(90, 20), (104, 60), (146, 60), (112, 85), (125, 128), (90, 102), (55, 128), (68, 85), (34, 60), (76, 60)]
    star = [(x + 340, y + 8) for x, y in star]
    elements: list[Element] = [
        shape("circle", x=70, y=75, radius=50, pattern="hatch"),
        text("circle: hatch", 70, 135, 13),
        shape("circle", x=190, y=75, radius=50, width=3, pattern={"type": "dots", "spacing": 5, "width": 2}),
        text("circle: dots", 190, 135, 13),
        shape("ellipse", x_start=270, y_start=30, x_end=400, y_end=120, **fills, pattern=red_hatch),
        text("ellipse: red hatch on yellow", 335, 135, 13),
        shape("polygon", points=[(x + 100, y) for x, y in star], pattern={"type": "tone", "level": 35}),
        text("polygon: tone", 530, 135, 13),
        shape("arc", x=700, y=75, radius=55, start_angle=-60, end_angle=240, pattern={"type": "grid", "spacing": 8}),
        text("arc: a pie slice with a grid", 700, 135, 13),
        shape(
            "progress_bar",
            x_start=20,
            y_start=180,
            x_end=380,
            y_end=210,
            progress=65,
            fill="black",
            pattern={"type": "dots", "color": "white", "spacing": 3},
        ),
        text("progress_bar: pattern on the progress", 200, 214, 13),
        shape(
            "progress_bar",
            x_start=20,
            y_start=250,
            x_end=380,
            y_end=280,
            progress=40,
            fill="white",
            pattern="hatch",
            background_pattern={"type": "tone", "level": 20},
        ),
        shape(
            "diagram",
            x=410,
            height=190,
            width=370,
            bars={
                "values": "Mon,10;Tue,20;Wed,15;Thu,25",
                "color": "red",
                "pattern": {"type": "hatch", "color": "black", "angle": 135, "spacing": 5},
            },
        ),
        text("progress_bar: pattern and background_pattern", 200, 284, 13),
        shape(
            "plot",
            x_start=20,
            y_start=325,
            x_end=380,
            y_end=450,
            duration=24 * 3600,
            data=[
                {
                    "entity": "falling",
                    "color": "black",
                    "width": 2,
                    "pattern": {"type": "hatch", "angle": 90, "spacing": 5},
                },
                {"entity": "rising", "color": "red", "width": 2, "pattern": {"type": "dots", "spacing": 4}},
            ],
        ),
        text("plot: pattern under each series", 200, 456, 13),
        text("diagram: bars.pattern", 600, 262, 13),
    ]
    return elements


async def main() -> None:
    """Write every picture."""
    save(await render(pattern_types(), 800, 480), "pattern_types")
    save(await render(layers(), 800, 340), "layers")
    save(await dithering_picture(), "dithering")
    shapes = await generate_image(
        width=800, height=480, elements=shapes_sheet(), background="white", data_provider=MockDataProvider()
    )
    save(shapes.convert("RGB"), "shapes")


if __name__ == "__main__":
    asyncio.run(main())
