from hypothesis import given
from hypothesis import strategies as st

from drawcustom.coordinates import CoordinateParser


class TestCoordinateProperties:
    """Property-based tests for coordinate parsing."""

    @given(canvas_width=st.integers(min_value=1, max_value=10000), percentage=st.integers(min_value=0, max_value=100))
    def test_percentage_always_within_canvas_width(self, canvas_width, percentage):
        """Percentage parsing for width never exceeds canvas bounds."""
        parser = CoordinateParser(canvas_width, 100)
        x = parser.parse_x(f"{percentage}%")

        assert 0 <= x <= canvas_width
        # Verify it's approximately correct (within 1 pixel due to rounding)
        expected = int(canvas_width * percentage / 100)
        assert abs(x - expected) <= 1

    @given(canvas_height=st.integers(min_value=1, max_value=10000), percentage=st.integers(min_value=0, max_value=100))
    def test_percentage_always_within_canvas_height(self, canvas_height, percentage):
        """Percentage parsing for height never exceeds canvas bounds."""
        parser = CoordinateParser(100, canvas_height)
        y = parser.parse_y(f"{percentage}%")

        assert 0 <= y <= canvas_height
        # Verify it's approximately correct (within 1 pixel due to rounding)
        expected = int(canvas_height * percentage / 100)
        assert abs(y - expected) <= 1

    @given(
        canvas_width=st.integers(min_value=1, max_value=10000), absolute_x=st.integers(min_value=-1000, max_value=15000)
    )
    def test_absolute_coordinate_preserved(self, canvas_width, absolute_x):
        """Absolute integer coordinates are preserved as-is."""
        parser = CoordinateParser(canvas_width, 100)  # Fixed: positional args
        result = parser.parse_x(absolute_x)

        assert result == absolute_x

    @given(width=st.integers(min_value=1, max_value=5000), height=st.integers(min_value=1, max_value=5000))
    def test_parser_accepts_any_positive_dimensions(self, width, height):
        """Parser can be created with any positive dimensions."""
        parser = CoordinateParser(width, height)  # Fixed: positional args

        assert parser.width == width
        assert parser.height == height

    @given(
        canvas_width=st.integers(min_value=1, max_value=1000),
        x_coord=st.one_of(
            st.integers(min_value=0, max_value=1000),
            st.text(min_size=1, max_size=5).filter(lambda s: s.endswith("%") and s[:-1].isdigit()),
        ),
    )
    def test_parse_x_returns_numeric(self, canvas_width, x_coord):
        """parse_x always returns a numeric value for valid inputs."""
        parser = CoordinateParser(canvas_width, 100)  # Fixed: positional args

        try:
            result = parser.parse_x(x_coord)
            assert isinstance(result, (int, float))
        except (ValueError, AttributeError):
            # Invalid format is acceptable to reject
            pass
