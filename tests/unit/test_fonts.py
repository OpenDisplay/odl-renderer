"""Unit tests for FontManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import ImageFont

from drawcustom.fonts import FontManager

ASSETS_DIR = Path(__file__).parent.parent.parent / "src" / "drawcustom" / "assets"


class TestFontManager:
    def test_freetype_object_returned_as_is(self):
        font_obj = ImageFont.truetype(str(ASSETS_DIR / "ppb.ttf"), 16)
        manager = FontManager()
        result = manager.get_font(font_obj, 16)
        assert result is font_obj

    def test_builtin_font_by_name(self):
        manager = FontManager()
        font = manager.get_font("ppb", 16)
        assert isinstance(font, ImageFont.FreeTypeFont)

    def test_builtin_font_with_extension(self):
        manager = FontManager()
        font = manager.get_font("ppb.ttf", 16)
        assert isinstance(font, ImageFont.FreeTypeFont)

    def test_font_is_cached(self):
        manager = FontManager()
        font1 = manager.get_font("ppb", 16)
        font2 = manager.get_font("ppb", 16)
        assert font1 is font2

    def test_absolute_path_loading(self):
        manager = FontManager()
        font = manager.get_font(str(ASSETS_DIR / "ppb.ttf"), 16)
        assert isinstance(font, ImageFont.FreeTypeFont)

    def test_nonexistent_absolute_path_raises(self):
        manager = FontManager()
        with pytest.raises(ValueError, match="not found"):
            manager.get_font("/nonexistent/path/font.ttf", 16)

    def test_invalid_absolute_path_file_raises(self):
        """A file that exists but is not a font raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"not a font")
            path = f.name
        manager = FontManager()
        with pytest.raises(ValueError, match="Failed to load font"):
            manager.get_font(path, 16)

    def test_unknown_builtin_name_raises(self):
        manager = FontManager()
        with pytest.raises(ValueError, match="not found"):
            manager.get_font("nonexistent_font", 16)

    def test_clear_cache(self):
        manager = FontManager()
        manager.get_font("ppb", 16)
        assert len(manager._font_cache) == 1
        manager.clear_cache()
        assert len(manager._font_cache) == 0
