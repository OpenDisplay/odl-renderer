"""Drawcustom - Pure image rendering from drawing instructions."""

from .colors import (
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
from .coordinates import CoordinateParser
from .core import generate_image, should_show_element
from .fonts import FontManager
from .types import DrawingContext, ElementType, TextSegment
from .warmup import warmup

__version__ = "0.5.10"

__all__ = [
    "warmup",
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
    "DARK_GRAY",
    "LIGHT_GRAY",
    "BLUE",
    "GREEN",
]
