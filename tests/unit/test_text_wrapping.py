"""Unit tests for the text truncation/wrapping helpers (finding B2)."""

from odl_renderer.elements.text import _truncate_to_width, _wrap_to_width


def test_truncate_to_width_fits_and_adds_ellipsis(load_font):
    font = load_font("ppb", 16)
    text = "word " * 200  # far wider than 100px
    result = _truncate_to_width(text, font, 100)
    assert result.endswith("...")
    assert font.getlength(result) <= 100


def test_truncate_to_width_short_text_unchanged(load_font):
    font = load_font("ppb", 16)
    assert _truncate_to_width("Hi", font, 500) == "Hi"


def test_truncate_to_width_matches_linear_result(load_font):
    """Binary search must produce the same cut as trimming one char at a time."""
    font = load_font("ppb", 16)
    ellipsis = "..."
    text = "The quick brown fox jumps over the lazy dog again and again"
    max_width = 90

    expected = text
    if font.getlength(text) > max_width:
        truncated = text
        while truncated and font.getlength(truncated + ellipsis) > max_width:
            truncated = truncated[:-1]
        expected = truncated + ellipsis

    assert _truncate_to_width(text, font, max_width) == expected


def test_wrap_to_width_lines_fit(load_font):
    font = load_font("ppb", 16)
    text = "This is a long sentence that should wrap onto several lines"
    wrapped = _wrap_to_width(text, font, 100)
    assert "\n" in wrapped
    for line in wrapped.split("\n"):
        # A single word longer than max_width is allowed to overflow its own line.
        assert font.getlength(line) <= 100 or " " not in line


def test_wrap_to_width_empty_string(load_font):
    font = load_font("ppb", 16)
    assert _wrap_to_width("   ", font, 100) == ""
