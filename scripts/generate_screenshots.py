"""Generate README screenshots for each element type.

Reads element examples directly from README.md and renders each one
to docs/screenshots/<element_name>.png at 296x128 pixels.
"""

from __future__ import annotations

import ast
import asyncio
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
from PIL import ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from odl_renderer import generate_image

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "screenshots"
README_PATH = Path(__file__).parent.parent / "README.md"
CANVAS_WIDTH = 296
CANVAS_HEIGHT = 128

ELEMENT_HEADING_RE = re.compile(r"^### `([a-z_]+)`$")
CODE_BLOCK_RE = re.compile(r"```yaml\n(.*?)```", re.DOTALL)
DLIMG_REAL_URL = "https://picsum.photos/seed/odl/200/150"


class MockDataProvider:
    """Minimal DataProvider for plot screenshots — returns a seeded random walk."""

    async def get_history(
        self,
        entity_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, list[dict]]:
        rng = random.Random(42)
        result = {}
        for entity_id in entity_ids:
            records = []
            current = start
            value = 20.0
            while current <= end:
                records.append(
                    {
                        "state": str(round(value, 1)),
                        "last_changed": current.isoformat(),
                    }
                )
                value += (rng.random() - 0.5) * 2
                current += timedelta(hours=1)
            result[entity_id] = records
        return result


def parse_element_examples(readme: str) -> dict[str, dict]:
    """Extract element config dicts keyed by element name."""
    examples = {}
    for section in re.split(r"\n(?=### `[a-z_]+`)", readme):
        first_line = section.split("\n")[0].strip()
        heading_match = ELEMENT_HEADING_RE.match(first_line)
        if not heading_match:
            continue
        element_name = heading_match.group(1)
        code_match = CODE_BLOCK_RE.search(section)
        if not code_match:
            continue
        try:
            config = ast.literal_eval(code_match.group(1).strip())
        except (ValueError, SyntaxError) as e:
            print(f"  WARN could not parse {element_name}: {e}")
            continue
        examples[element_name] = config
    return examples


async def render_element(name: str, config: dict, session: aiohttp.ClientSession) -> None:
    """Render a single element config to a PNG file."""

    image = await generate_image(
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        elements=[config],
        background="white",
        session=session,
        data_provider=MockDataProvider(),
    )
    image = ImageOps.expand(image, border=1, fill="black")
    image.save(OUTPUT_DIR / f"{name}.png")
    print(f"  OK  {name}.png")


async def main() -> int:
    """Parse README examples and render each element to a PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readme = README_PATH.read_text()
    examples = parse_element_examples(readme)
    print(f"Found {len(examples)} element examples")

    failed = []
    async with aiohttp.ClientSession() as session:
        for name, config in examples.items():
            try:
                await render_element(name, config, session)
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed.append(name)

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(examples)} screenshots saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
