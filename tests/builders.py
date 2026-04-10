from typing import Any


class ElementBuilder:
    """Builder for creating test element configurations."""

    @staticmethod
    def text(value: str = "Test", x: int | str = 10, y: int | str = 10, **kwargs) -> dict[str, Any]:
        """Build a text element with defaults."""
        return {
            "type": "text",
            "value": value,
            "x": x,
            "y": y,
            "font": kwargs.get("font", "ppb"),
            "size": kwargs.get("size", 16),
            **{k: v for k, v in kwargs.items() if k not in ("font", "size")},
        }

    @staticmethod
    def rectangle(x_start: int = 0, y_start: int = 0, x_end: int = 100, y_end: int = 50, **kwargs) -> dict[str, Any]:
        """Build a rectangle element with defaults."""
        return {"type": "rectangle", "x_start": x_start, "y_start": y_start, "x_end": x_end, "y_end": y_end, **kwargs}

    @staticmethod
    def circle(x: int = 50, y: int = 50, radius: int = 25, **kwargs) -> dict[str, Any]:
        """Build a circle element with defaults."""
        return {"type": "circle", "x": x, "y": y, "radius": radius, **kwargs}

    @staticmethod
    def line(x_start: int = 0, y_start: int = 0, x_end: int = 100, y_end: int = 100, **kwargs) -> dict[str, Any]:
        """Build a line element with defaults."""
        return {"type": "line", "x_start": x_start, "y_start": y_start, "x_end": x_end, "y_end": y_end, **kwargs}

    @staticmethod
    def qrcode(data: str, **kwargs) -> dict[str, Any]:
        """Build a QR code element."""
        return {
            "type": "qrcode",
            "data": data,
            "x": kwargs.get("x", 50),
            "y": kwargs.get("y", 50),
            "size": kwargs.get("size", 100),
            **{k: v for k, v in kwargs.items() if k not in ("x", "y", "size")},
        }

    @staticmethod
    def multiline(value: str = "Test\nText", x: int = 10, y: int = 10, **kwargs) -> dict[str, Any]:
        """Build a multiline text element with defaults."""
        return {
            "type": "multiline",
            "value": value,
            "x": x,
            "y": y,
            "font": kwargs.get("font", "ppb"),
            "size": kwargs.get("size", 16),
            **{k: v for k, v in kwargs.items() if k not in ("font", "size")},
        }

    @staticmethod
    def icon(icon_name: str, **kwargs) -> dict[str, Any]:
        """Build an icon element."""
        return {
            "type": "icon",
            "icon": icon_name,
            "x": kwargs.get("x", 50),
            "y": kwargs.get("y", 50),
            "size": kwargs.get("size", 24),
            **{k: v for k, v in kwargs.items() if k not in ("x", "y", "size")},
        }

    @staticmethod
    def progress(
        value: float = 50.0, x: int = 10, y: int = 10, width: int = 100, height: int = 20, **kwargs
    ) -> dict[str, Any]:
        """Build a progress bar element."""
        return {"type": "progress", "value": value, "x": x, "y": y, "width": width, "height": height, **kwargs}
