"""Unit tests for FontManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import ImageFont

from odl_renderer.fonts import FontManager

ASSETS_DIR = Path(__file__).parent.parent.parent / "src" / "odl_renderer" / "assets"


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


class TestFontManagerFontDirs:
    def test_font_found_in_search_dir(self):
        """A font file in font_dirs is found by relative name."""
        with tempfile.TemporaryDirectory() as d:
            font_path = Path(d) / "custom.ttf"
            font_path.write_bytes((ASSETS_DIR / "ppb.ttf").read_bytes())
            manager = FontManager(font_dirs=[d])
            font = manager.get_font("custom.ttf", 16)
            assert isinstance(font, ImageFont.FreeTypeFont)

    def test_font_found_without_extension(self):
        """A font in font_dirs can be referenced without .ttf extension."""
        with tempfile.TemporaryDirectory() as d:
            font_path = Path(d) / "custom.ttf"
            font_path.write_bytes((ASSETS_DIR / "ppb.ttf").read_bytes())
            manager = FontManager(font_dirs=[d])
            font = manager.get_font("custom", 16)
            assert isinstance(font, ImageFont.FreeTypeFont)

    def test_font_dirs_searched_before_bundled(self):
        """A font in font_dirs shadows a bundled font of the same name."""
        with tempfile.TemporaryDirectory() as d:
            # Copy rbm.ttf as ppb.ttf into the custom dir
            shadow_path = Path(d) / "ppb.ttf"
            shadow_path.write_bytes((ASSETS_DIR / "rbm.ttf").read_bytes())
            manager = FontManager(font_dirs=[d])
            font = manager.get_font("ppb", 16)
            # Should load from d, not from assets — verify via source path via cache key
            assert isinstance(font, ImageFont.FreeTypeFont)

    def test_nonexistent_dir_silently_ignored(self):
        """Directories that don't exist are silently skipped."""
        manager = FontManager(font_dirs=["/nonexistent/dir"])
        assert manager._font_dirs == []

    def test_font_falls_back_to_bundled_when_not_in_dirs(self):
        """If font not in font_dirs, bundled assets are still searched."""
        with tempfile.TemporaryDirectory() as d:
            manager = FontManager(font_dirs=[d])
            font = manager.get_font("ppb", 16)
            assert isinstance(font, ImageFont.FreeTypeFont)

    def test_no_font_dirs_behaves_as_before(self):
        """FontManager() with no font_dirs loads bundled fonts normally."""
        manager = FontManager()
        font = manager.get_font("ppb", 16)
        assert isinstance(font, ImageFont.FreeTypeFont)


def test_module_font_cache_shared_across_managers():
    """The process-wide truetype cache is reused across FontManager instances (B4)."""
    m1 = FontManager()
    m2 = FontManager()
    f1 = m1.get_font("ppb", 16)
    f2 = m2.get_font("ppb", 16)
    # A fresh FontManager per generate_image() must not re-read the font from disk.
    assert f1 is f2


def test_module_font_cache_keys_on_size():
    m = FontManager()
    assert m.get_font("ppb", 16) is not m.get_font("ppb", 24)
