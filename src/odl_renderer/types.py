from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

import aiohttp
from PIL import Image

if TYPE_CHECKING:
    from PIL import Image

    from .colors import ColorResolver
    from .coordinates import CoordinateParser
    from .fonts import FontManager


# What an element is asking to load: an HTTP/HTTPS URL, or a file on disk
# (an image or a font given by absolute path).
SourceKind = Literal["url", "file"]

# Caller-supplied policy deciding whether a source may be loaded. Returns True to
# allow. URL checks run on the event loop; file checks run in the worker thread
# that draws, so blocking work (e.g. resolving symlinks) is fine there.
SourceGuard = Callable[[SourceKind, str], bool]


class SourceNotAllowedError(ValueError):
    """An element named a URL or file that may not be loaded.

    Attributes:
        kind: "url" or "file".
        source: The refused URL or path, as written in the element.
    """

    def __init__(self, kind: SourceKind, source: str) -> None:
        super().__init__(f"{'URL' if kind == 'url' else 'File'} not allowed: {source}")
        self.kind = kind
        self.source = source


def check_source(allow_source: SourceGuard | None, kind: SourceKind, source: str) -> None:
    """Raise SourceNotAllowedError if the caller's policy refuses *source*."""
    if allow_source is not None and not allow_source(kind, source):
        raise SourceNotAllowedError(kind, source)


class DataProvider(Protocol):
    """Protocol for providing historical state data to draw handlers.

    Implement this protocol to supply time-series data (e.g. from Home Assistant's
    recorder, a database, or a mock) without coupling drawcustom to any specific
    data source.
    """

    async def get_history(
        self,
        entity_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return historical states for the given entities and time range.

        Args:
            entity_ids: List of entity identifiers to fetch history for.
            start: Start of the time range (timezone-aware).
            end: End of the time range (timezone-aware).

        Returns:
            Mapping of entity_id to a list of state records, ordered oldest-first.
            Each record is a dict with at minimum:
                - "state": str — the state value (e.g. "23.5", "on")
                - "last_changed": str — ISO 8601 timestamp. Prefer a timezone-aware
                  value (with offset); a naive timestamp is interpreted as UTC.
            Entities with no data in the range may be absent or map to an empty list.
        """
        ...


class ElementType(str, Enum):
    """Enum for supported element types.

    Defines all the drawable element types supported by the ImageGen class.
    Each type corresponds to a specific drawing method that handles the
    rendering of that element type.

    The enum values are used in the payload to identify the type of each element.
    """

    TEXT = "text"
    MULTILINE = "multiline"
    LINE = "line"
    RECTANGLE = "rectangle"
    RECTANGLE_PATTERN = "rectangle_pattern"
    POLYGON = "polygon"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    ARC = "arc"
    ICON = "icon"
    DLIMG = "dlimg"
    QRCODE = "qrcode"
    PLOT = "plot"
    PROGRESS_BAR = "progress_bar"
    DIAGRAM = "diagram"
    ICON_SEQUENCE = "icon_sequence"
    DEBUG_GRID = "debug_grid"

    def __str__(self) -> str:
        """Return the string value of the enum.

        Returns:
            str: The string value of the enum
        """

        return self.value


@dataclass
class TextSegment:
    """Represents a segment of text with its color.

    Used for handling colored text markup, where different parts of a text
    string can have different colors (e.g., "[red]Text[/red]").

    Attributes:
        text: The text content
        color: The color name for this segment
        start_x: Starting x position for rendering (calculated during layout)
    """

    text: str
    color: str
    start_x: int = 0


@dataclass
class DrawingContext:
    """Drawing context passed to element handlers."""

    img: Image.Image
    colors: "ColorResolver"
    coords: "CoordinateParser"
    fonts: "FontManager"
    session: aiohttp.ClientSession | None = None
    data_provider: "DataProvider | None" = None
    pos_y: int = 0
    allow_source: SourceGuard | None = None
    # Transparent layer that transformed elements are drawn on, kept to reuse it.
    transform_layer: Image.Image | None = None
