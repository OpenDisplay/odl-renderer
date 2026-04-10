import pytest

from tests.builders import ElementBuilder as E


class TestElementBuilder:
    """Test element builder factory methods."""

    @pytest.mark.parametrize(
        "builder_method,expected_type",
        [
            (E.text, "text"),
            (E.rectangle, "rectangle"),
            (E.circle, "circle"),
            (E.line, "line"),
            (E.qrcode, "qrcode"),
            (E.multiline, "multiline"),
            (E.icon, "icon"),
            (E.progress, "progress"),
        ],
    )
    def test_builder_creates_correct_type(self, builder_method, expected_type):
        """Test each builder creates element with correct type field."""
        if builder_method == E.qrcode:
            element = builder_method("test_value")  # TODO ask why?
        elif builder_method == E.icon:
            element = builder_method("test_icon")
        else:
            element = builder_method()

        assert element["type"] == expected_type

    @pytest.mark.parametrize(
        "value,x,y,size,font,expected",
        [
            ("Hello", 50, 100, 24, "rbm", {"value": "Hello", "x": 50, "y": 100, "size": 24, "font": "rbm"}),
            ("Test", 10, 10, 16, "ppb", {"value": "Test", "x": 10, "y": 10, "size": 16, "font": "ppb"}),
        ],
    )
    def test_text_element_parameters(self, value, x, y, size, font, expected):
        """Test text element with various parameter combinations."""
        element = E.text(value, x=x, y=y, size=size, font=font)

        for key, val in expected.items():
            assert element[key] == val

    def test_text_element_defaults(self):
        """Test text element uses sensible defaults."""
        element = E.text()
        assert element["value"] == "Test"
        assert element["font"] == "ppb"
        assert element["size"] == 16

    def test_text_element_extra_kwargs(self):
        """Test text element accepts extra parameters."""
        element = E.text("Hello", color="red", anchor="mm")
        assert element["color"] == "red"
        assert element["anchor"] == "mm"

    @pytest.mark.parametrize(
        "x_start,y_start,x_end,y_end",
        [
            (0, 0, 100, 50),
            (10, 20, 200, 150),
            (50, 50, 250, 250),
        ],
    )
    def test_rectangle_coordinates(self, x_start, y_start, x_end, y_end):
        """Test rectangle with various coordinate combinations."""
        element = E.rectangle(x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end)
        assert element["x_start"] == x_start
        assert element["y_start"] == y_start
        assert element["x_end"] == x_end
        assert element["y_end"] == y_end

    def test_rectangle_extra_kwargs(self):
        """Test rectangle accepts extra parameters."""
        element = E.rectangle(fill="blue", outline="black", width=2)
        assert element["fill"] == "blue"
        assert element["outline"] == "black"
        assert element["width"] == 2

    @pytest.mark.parametrize(
        "url,x,y,size",
        [
            ("https://example.com", 50, 50, 100),
            ("test-data", 10, 20, 80),
            ("https://qr.test/abc", 100, 100, 120),
        ],
    )
    def test_qrcode_parameters(self, url, x, y, size):
        """Test QR code with various parameters."""
        element = E.qrcode(url, x=x, y=y, size=size)
        assert element["data"] == url
        assert element["x"] == x
        assert element["y"] == y
        assert element["size"] == size

    @pytest.mark.parametrize(
        "icon_name,expected_icon",
        [
            ("home", "home"),
            ("settings", "settings"),
            ("battery-50", "battery-50"),
        ],
    )
    def test_icon_names(self, icon_name, expected_icon):
        """Test icon element with various icon names."""
        element = E.icon(icon_name, x=60, y=70, size=32)
        assert element["icon"] == expected_icon
        assert element["x"] == 60
        assert element["y"] == 70
        assert element["size"] == 32

    @pytest.mark.parametrize("progress_value", [0.0, 25.5, 50.0, 75.0, 100.0])
    def test_progress_values(self, progress_value):
        """Test progress bar with various values."""
        element = E.progress(value=progress_value, x=10, y=10, width=200, height=30)
        assert element["value"] == progress_value
        assert element["width"] == 200
        assert element["height"] == 30
