"""Drawcustom - Pure image rendering from drawing instructions."""

from .colors import BLACK, BLUE, GREEN, HALF_BLACK, HALF_RED, HALF_YELLOW, RED, WHITE, YELLOW, ColorResolver
from .coordinates import CoordinateParser
from .core import generate_image, should_show_element
from .fonts import FontManager
from .types import DrawingContext, ElementType, TextSegment

__version__ = "0.5.1"

__all__ = [
    "generate_image",
    "should_show_element",
    "ElementType",
    "DrawingContext",
    "TextSegment",
    "ColorResolver",
    "CoordinateParser",
    "FontManager",
    "WHITE",
    "BLACK",
    "RED",
    "YELLOW",
    "HALF_BLACK",
    "HALF_RED",
    "HALF_YELLOW",
    "BLUE",
    "GREEN",
]
