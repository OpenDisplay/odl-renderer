#!/usr/bin/env python3
"""Regenerate the bundled MDI icon index from official MDI metadata.

The index is a flat ``{name: codepoint}`` dict covering icon names and their aliases.
Some aliases of one icon are the canonical name of another (``battery`` is also an alias
of ``fuel-cell``), so canonical names are protected from being overwritten by aliases.

Use the ``meta.json`` of the ``@mdi/svg`` release whose ``@mdi/font`` TTF is the bundled
font. Metadata from any other release can point at glyphs the font does not contain.

    npm pack @mdi/svg@7.2.96
    tar -xzf mdi-svg-7.2.96.tgz package/meta.json
    python scripts/build_mdi_index.py package/meta.json

Names already in the index that the metadata does not know are kept, so regenerating
never removes a name that callers may depend on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path(__file__).parents[1] / "src" / "odl_renderer" / "assets" / "materialdesignicons-webfont_meta.json"


def build_index(metadata: Iterable[Mapping[str, Any]], previous: Mapping[str, str] | None = None) -> dict[str, str]:
    """Flatten MDI metadata into ``{name: codepoint}``.

    Args:
        metadata: Icons from ``meta.json``, each with ``name``, ``codepoint`` and ``aliases``.
        previous: An existing index whose names unknown to the metadata are carried over.

    Returns:
        The index, in which a canonical name always maps to its own icon's codepoint. An
        alias declared by several icons maps to the last of them.
    """
    icons = list(metadata)
    canonical_names = {icon["name"] for icon in icons}

    index: dict[str, str] = {}
    for icon in icons:
        index[icon["name"]] = icon["codepoint"]
        for alias in icon["aliases"]:
            if alias not in canonical_names:
                index[alias] = icon["codepoint"]

    return {**index, **{name: cp for name, cp in (previous or {}).items() if name not in index}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", type=Path, help="meta.json of the @mdi/svg release matching the bundled font")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="index file to update")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else None
    index = build_index(metadata, previous)
    args.output.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8", newline="\n")
    print(f"Wrote {len(index)} names for {len(metadata)} icons to {args.output}")


if __name__ == "__main__":
    main()
