"""Unit tests for CoordinateParser."""

from __future__ import annotations

from drawcustom.coordinates import CoordinateParser


class TestCoordinateParser:
    def setup_method(self):
        self.parser = CoordinateParser(canvas_width=200, canvas_height=100)

    def test_integer_coordinate(self):
        assert self.parser.parse_x(50) == 50

    def test_float_coordinate(self):
        assert self.parser.parse_x(50.5) == 50

    def test_string_integer(self):
        assert self.parser.parse_x("50") == 50

    def test_percentage_x(self):
        assert self.parser.parse_x("50%") == 100  # 50% of width=200

    def test_percentage_y(self):
        assert self.parser.parse_y("50%") == 50  # 50% of height=100

    def test_invalid_percentage_returns_zero(self):
        assert self.parser.parse_x("abc%") == 0

    def test_invalid_string_returns_zero(self):
        assert self.parser.parse_x("notanumber") == 0

    def test_parse_size_width(self):
        assert self.parser.parse_size("50%", is_width=True) == 100

    def test_parse_size_height(self):
        assert self.parser.parse_size("50%", is_width=False) == 50

    def test_parse_coordinates(self):
        element = {"x": 10, "y": 20}
        x, y = self.parser.parse_coordinates(element)
        assert x == 10
        assert y == 20

    def test_parse_coordinates_with_prefix(self):
        element = {"start_x": 10, "start_y": 20}
        x, y = self.parser.parse_coordinates(element, prefix="start_")
        assert x == 10
        assert y == 20

    def test_parse_coordinates_defaults_to_zero(self):
        x, y = self.parser.parse_coordinates({})
        assert x == 0
        assert y == 0
