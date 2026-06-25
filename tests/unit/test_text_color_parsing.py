from odl_renderer.elements.text import parse_colored_text


def test_parse_colored_text_supports_blue_and_green():
    segments = parse_colored_text("[blue]B[/blue][green]G[/green]")

    assert [segment.text for segment in segments] == ["B", "G"]
    assert [segment.color for segment in segments] == ["blue", "green"]


def test_parse_colored_text_keeps_unmarked_text_black():
    segments = parse_colored_text("Hello [blue]B[/blue]")

    assert [segment.text for segment in segments] == ["Hello ", "B"]
    assert [segment.color for segment in segments] == ["black", "blue"]


def test_parse_colored_text_supports_hex_tags():
    segments = parse_colored_text("[#0f0]G[/#0f0][#00FFAA]H[/#00FFAA]")

    assert [segment.text for segment in segments] == ["G", "H"]
    assert [segment.color for segment in segments] == ["#0f0", "#00FFAA"]


def test_parse_colored_text_supports_gray_names():
    segments = parse_colored_text("[ltgray]L[/ltgray][dkgray]D[/dkgray]")

    assert [segment.text for segment in segments] == ["L", "D"]
    assert [segment.color for segment in segments] == ["ltgray", "dkgray"]


def test_parse_colored_text_supports_8_digit_hex_tags():
    segments = parse_colored_text("[#FF000080]A[/#FF000080]")

    assert [segment.text for segment in segments] == ["A"]
    assert [segment.color for segment in segments] == ["#FF000080"]
