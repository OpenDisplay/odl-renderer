import pytest

from odl_renderer.colors import (
    BLACK,
    BLUE,
    DARK_GRAY,
    GREEN,
    HALF_BLACK,
    HALF_RED,
    HALF_YELLOW,
    LIGHT_GRAY,
    RED,
    WHITE,
    YELLOW,
    ColorResolver,
)


class TestColorResolver:
    """Test color resolution functionality."""

    def test_none_returns_none(self):
        """Test that the None input returns None."""

        resolver = ColorResolver()
        assert resolver.resolve(None) is None

    @pytest.mark.parametrize(
        "hex_color,expected",
        [
            # 6-digit hex
            ("#FF0000", (255, 0, 0, 255)),
            ("#00FF00", (0, 255, 0, 255)),
            ("#0000FF", (0, 0, 255, 255)),
            ("#FFFFFF", (255, 255, 255, 255)),
            ("#000000", (0, 0, 0, 255)),
            # 3-digit hex shorthand
            ("#F00", (255, 0, 0, 255)),
            ("#0F0", (0, 255, 0, 255)),
            ("#00F", (0, 0, 255, 255)),
            ("#FFF", (255, 255, 255, 255)),
            ("#000", (0, 0, 0, 255)),
            # Case insensitive
            ("#ff0000", (255, 0, 0, 255)),
            ("#FF0000", (255, 0, 0, 255)),
            ("#Ff0000", (255, 0, 0, 255)),
            # 8-digit hex with alpha
            ("#FF000080", (255, 0, 0, 128)),
            ("#FFFFFFFF", (255, 255, 255, 255)),
            ("#00000000", (0, 0, 0, 0)),
            # 4-digit hex shorthand with alpha
            ("#F008", (255, 0, 0, 136)),
            ("#FFFF", (255, 255, 255, 255)),
            # Surrounding whitespace is stripped
            ("  #FF0000  ", (255, 0, 0, 255)),
        ],
    )
    def test_hex_colors(self, hex_color, expected):
        """Test hex color parsing (3/4/6/8-digit, alpha, case, whitespace)."""
        resolver = ColorResolver()
        assert resolver.resolve(hex_color) == expected

    @pytest.mark.parametrize("invalid_hex", ["#FF", "#FFFFFFF"])
    def test_hex_invalid_length_returns_white(self, invalid_hex):
        """Test invalid hex length returns white."""
        resolver = ColorResolver()
        assert resolver.resolve(invalid_hex) == WHITE  # TODO throw exception instead?

    @pytest.mark.parametrize(
        "color_name,expected",
        [
            # Black
            ("black", BLACK),
            ("b", BLACK),
            # White
            ("white", WHITE),
            ("w", WHITE),
            # Half black / mid gray
            ("half_black", HALF_BLACK),
            ("hb", HALF_BLACK),
            ("gray", HALF_BLACK),
            ("grey", HALF_BLACK),
            # Dark gray
            ("dark_gray", DARK_GRAY),
            ("darkgray", DARK_GRAY),
            ("dkgray", DARK_GRAY),
            # Light gray (half_white fixed: was wrongly mid gray)
            ("light_gray", LIGHT_GRAY),
            ("lightgray", LIGHT_GRAY),
            ("ltgray", LIGHT_GRAY),
            ("half_white", LIGHT_GRAY),
            ("hw", LIGHT_GRAY),
            # Red
            ("red", RED),
            ("r", RED),
            # Half red
            ("half_red", HALF_RED),
            ("hr", HALF_RED),
            # Yellow
            ("yellow", YELLOW),
            ("y", YELLOW),
            # Half yellow
            ("half_yellow", HALF_YELLOW),
            ("hy", HALF_YELLOW),
            # Blue
            ("blue", BLUE),
            ("bl", BLUE),
            # Green
            ("green", GREEN),
            ("gr", GREEN),
            # Unknown falls back to white # TODO should this throw an exception?
            ("unknown_color", WHITE),
            ("invalid", WHITE),
        ],
    )
    def test_named_colors(self, color_name, expected):
        """Test named color resolution and aliases."""
        resolver = ColorResolver()
        assert resolver.resolve(color_name) == expected

    @pytest.mark.parametrize(
        "accent_color,color_alias,expected",
        [
            # Red accent
            ("red", "accent", RED),
            ("red", "a", RED),
            ("red", "half_accent", HALF_RED),
            ("red", "ha", HALF_RED),
            # Yellow accent
            ("yellow", "accent", YELLOW),
            ("yellow", "a", YELLOW),
            ("yellow", "half_accent", HALF_YELLOW),
            ("yellow", "ha", HALF_YELLOW),
        ],
    )
    def test_accent_color_resolution(self, accent_color, color_alias, expected):
        """Test accent color resolves based on accent_color parameter."""
        resolver = ColorResolver(accent_color=accent_color)
        assert resolver.resolve(color_alias) == expected

    def test_accent_color_default_red(self):
        """Test accent color defaults to red when not specified."""
        resolver = ColorResolver()
        assert resolver.resolve("accent") == RED
        assert resolver.resolve("half_accent") == HALF_RED


class TestUnresolvedColorWarnings:
    """Unknown names and malformed hex render white but should warn once (finding A11)."""

    def test_unknown_name_warns_once(self, caplog):
        import logging

        from odl_renderer import colors

        colors._warned_color_tokens.discard("balck")
        resolver = ColorResolver()
        with caplog.at_level(logging.WARNING, logger="odl_renderer.colors"):
            assert resolver.resolve("balck") == WHITE
            assert resolver.resolve("balck") == WHITE  # second call must not re-warn
        warnings = [r for r in caplog.records if "balck" in r.getMessage()]
        assert len(warnings) == 1

    def test_malformed_hex_warns_once(self, caplog):
        import logging

        from odl_renderer import colors

        colors._warned_color_tokens.discard("#12345")
        resolver = ColorResolver()
        with caplog.at_level(logging.WARNING, logger="odl_renderer.colors"):
            assert resolver.resolve("#12345") == WHITE
            assert resolver.resolve("#12345") == WHITE
        warnings = [r for r in caplog.records if "#12345" in r.getMessage()]
        assert len(warnings) == 1

    def test_known_color_does_not_warn(self, caplog):
        import logging

        resolver = ColorResolver()
        with caplog.at_level(logging.WARNING, logger="odl_renderer.colors"):
            assert resolver.resolve("red") == RED
        assert not caplog.records
