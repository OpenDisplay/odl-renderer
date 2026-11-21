"""The bundled MDI index must agree with the official metadata of the bundled font.

``tests/data/mdi_meta_7.2.96.json`` is ``meta.json`` from ``@mdi/svg@7.2.96`` reduced to
name, codepoint and aliases. 7.2.96 is the release whose ``@mdi/font`` TTF is byte-identical
to the bundled font, which ``test_bundled_font_is_the_reference_release`` pins.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parents[2]
ASSETS_DIR = REPO_ROOT / "src" / "odl_renderer" / "assets"
FONT_PATH = ASSETS_DIR / "materialdesignicons-webfont.ttf"
INDEX_PATH = ASSETS_DIR / "materialdesignicons-webfont_meta.json"
METADATA_PATH = REPO_ROOT / "tests" / "data" / "mdi_meta_7.2.96.json"
FONT_SHA256 = "a58ecb54f45eec1afadbc21314d1f0932cf009e5cbc7f3225d7e4a4e1b71ef6b"

METADATA: list[dict] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
INDEX: dict[str, str] = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
CANONICAL = {icon["name"]: icon["codepoint"] for icon in METADATA}


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_mdi_index", REPO_ROOT / "scripts" / "build_mdi_index.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_font_is_the_reference_release():
    """Changing the font must come with new reference metadata and a regenerated index."""
    assert hashlib.sha256(FONT_PATH.read_bytes()).hexdigest() == FONT_SHA256


def test_every_canonical_name_maps_to_its_official_codepoint():
    wrong = {name: INDEX.get(name) for name, codepoint in CANONICAL.items() if INDEX.get(name) != codepoint}
    assert wrong == {}


def test_every_alias_resolves_to_an_icon_that_declares_it():
    owners: dict[str, set[str]] = {}
    for icon in METADATA:
        for alias in icon["aliases"]:
            owners.setdefault(alias, set()).add(icon["codepoint"])

    wrong = {
        alias: INDEX.get(alias)
        for alias, codepoints in owners.items()
        if alias not in CANONICAL and INDEX.get(alias) not in codepoints
    }
    assert wrong == {}


@pytest.mark.parametrize(
    ("name", "codepoint"),
    [("battery", "F0079"), ("fuel-cell", "F18B5"), ("water", "F058C"), ("waves", "F078D")],
)
def test_reported_icons_use_their_own_glyph(name: str, codepoint: str):
    assert INDEX[name] == codepoint


def test_bundled_index_is_the_generator_output():
    assert _load_generator().build_index(METADATA, INDEX) == INDEX


class TestBuildIndex:
    def test_canonical_name_beats_alias_on_a_later_icon(self):
        metadata = [
            {"name": "battery", "codepoint": "A", "aliases": []},
            {"name": "fuel-cell", "codepoint": "B", "aliases": ["battery"]},
        ]
        assert _load_generator().build_index(metadata) == {"battery": "A", "fuel-cell": "B"}

    def test_canonical_name_beats_alias_on_an_earlier_icon(self):
        metadata = [
            {"name": "fuel-cell", "codepoint": "B", "aliases": ["battery"]},
            {"name": "battery", "codepoint": "A", "aliases": []},
        ]
        assert _load_generator().build_index(metadata) == {"fuel-cell": "B", "battery": "A"}

    def test_alias_shared_by_two_icons_resolves_to_the_last(self):
        metadata = [
            {"name": "album", "codepoint": "A", "aliases": ["record"]},
            {"name": "radiobox-marked", "codepoint": "B", "aliases": ["record"]},
        ]
        assert _load_generator().build_index(metadata)["record"] == "B"

    def test_names_unknown_to_the_metadata_are_carried_over(self):
        metadata = [{"name": "home", "codepoint": "A", "aliases": []}]
        previous = {"home": "WRONG", "holiday": "B"}
        assert _load_generator().build_index(metadata, previous) == {"home": "A", "holiday": "B"}
