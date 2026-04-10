import pytest

from drawcustom.colors import BLACK, BLUE, GREEN, HALF_BLACK, HALF_RED, HALF_YELLOW, RED, WHITE, YELLOW, ColorResolver


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
        ],
    )
    def test_hex_colors(self, hex_color, expected):
        """Test hex color parsing (6-digit, 3-digit, case variations)."""
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
            # Half black / gray
            ("half_black", HALF_BLACK),
            ("hb", HALF_BLACK),
            ("gray", HALF_BLACK),
            ("grey", HALF_BLACK),
            ("half_white", HALF_BLACK),
            ("hw", HALF_BLACK),
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
