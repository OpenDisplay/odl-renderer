# Color constants with alpha channel
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
HALF_BLACK = (127, 127, 127, 255)
DARK_GRAY = (64, 64, 64, 255)
LIGHT_GRAY = (191, 191, 191, 255)
RED = (255, 0, 0, 255)
HALF_RED = (255, 127, 127, 255)
YELLOW = (255, 255, 0, 255)
HALF_YELLOW = (255, 255, 127, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)

# Single source of truth for statically-resolvable named colors.
# Add new named colors here only: the resolver and the inline text-markup
# allowlist are both derived from this mapping.
NAMED_COLORS: dict[str, tuple[int, int, int, int]] = {
    "black": BLACK,
    "b": BLACK,
    "white": WHITE,
    "w": WHITE,
    "dark_gray": DARK_GRAY,
    "darkgray": DARK_GRAY,
    "dkgray": DARK_GRAY,
    "half_black": HALF_BLACK,
    "hb": HALF_BLACK,
    "gray": HALF_BLACK,
    "grey": HALF_BLACK,
    "light_gray": LIGHT_GRAY,
    "lightgray": LIGHT_GRAY,
    "ltgray": LIGHT_GRAY,
    "half_white": LIGHT_GRAY,
    "hw": LIGHT_GRAY,
    "red": RED,
    "r": RED,
    "half_red": HALF_RED,
    "hr": HALF_RED,
    "yellow": YELLOW,
    "y": YELLOW,
    "half_yellow": HALF_YELLOW,
    "hy": HALF_YELLOW,
    "blue": BLUE,
    "bl": BLUE,
    "green": GREEN,
    "gr": GREEN,
    "g": GREEN,
}

# Accent aliases resolve dynamically from the configured accent color,
# so they are valid tokens but resolved separately from NAMED_COLORS.
ACCENT_ALIASES: tuple[str, ...] = ("accent", "a", "half_accent", "ha")

# Hex token forms accepted as colors (longest-first for unambiguous matching).
_HEX_TOKENS: tuple[str, ...] = (
    r"#[0-9A-Fa-f]{8}",
    r"#[0-9A-Fa-f]{6}",
    r"#[0-9A-Fa-f]{4}",
    r"#[0-9A-Fa-f]{3}",
)

# Regex alternation of every valid color token (named + accent + hex).
# Named tokens are sorted longest-first so aliases like "grey" match before
# "gr". Derived once and shared by inline text markup so adding a color in
# NAMED_COLORS automatically extends the markup allowlist too.
COLOR_TOKEN_PATTERN: str = "|".join(sorted([*NAMED_COLORS, *ACCENT_ALIASES], key=len, reverse=True) + list(_HEX_TOKENS))


class ColorResolver:
    """Resolves color inputs to RGBA tuples."""

    def __init__(self, accent_color: str = "red"):
        self.accent_color = accent_color

    def resolve(self, color: str | None) -> tuple[int, int, int, int] | None:
        """Resolve color input to RGBA tuple."""
        if color is None:
            return None

        color_str = str(color).strip().lower()

        # Hex color support: #RGB, #RGBA, #RRGGBB, or #RRGGBBAA
        if color_str.startswith("#"):
            return self._parse_hex(color_str[1:])

        return self._resolve_named(color_str)

    @staticmethod
    def _parse_hex(hex_val: str) -> tuple[int, int, int, int]:
        """Parse hex color string to RGBA tuple.

        Supports #RGB and #RGBA shorthand (each nibble doubled) and the
        full #RRGGBB and #RRGGBBAA forms. Alpha defaults to fully opaque
        when not provided. Invalid length or content falls back to white.
        """
        hex_val = hex_val.strip()
        try:
            if len(hex_val) == 3:
                r = int(hex_val[0] * 2, 16)
                g = int(hex_val[1] * 2, 16)
                b = int(hex_val[2] * 2, 16)
                a = 255
            elif len(hex_val) == 4:
                r = int(hex_val[0] * 2, 16)
                g = int(hex_val[1] * 2, 16)
                b = int(hex_val[2] * 2, 16)
                a = int(hex_val[3] * 2, 16)
            elif len(hex_val) == 6:
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                a = 255
            elif len(hex_val) == 8:
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                a = int(hex_val[6:8], 16)
            else:
                return WHITE
        except ValueError:
            return WHITE
        return r, g, b, a

    def _resolve_named(self, color_str: str) -> tuple[int, int, int, int]:
        """Resolve named color to RGBA tuple.

        Accent aliases resolve dynamically from the configured accent color;
        all other names are looked up in NAMED_COLORS. Unknown names fall
        back to white.
        """
        if color_str in ("accent", "a"):
            return YELLOW if self.accent_color == "yellow" else RED
        if color_str in ("half_accent", "ha"):
            return HALF_YELLOW if self.accent_color == "yellow" else HALF_RED
        return NAMED_COLORS.get(color_str, WHITE)
