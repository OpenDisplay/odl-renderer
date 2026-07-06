"""Unit tests for the pixel-comparing PNG snapshot extension (finding C1)."""

from io import BytesIO

from PIL import Image

from tests.conftest import PixelPNGImageSnapshotExtension


def _png(img: Image.Image, **save_kwargs) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG", **save_kwargs)
    return buffer.getvalue()


def _matches(a: bytes, b: bytes) -> bool:
    # matches() only needs `self` for the (unused here) byte fallback.
    ext = object.__new__(PixelPNGImageSnapshotExtension)
    return ext.matches(serialized_data=a, snapshot_data=b)


def test_same_pixels_different_bytes_match():
    """Different PNG encodings of identical pixels must compare equal."""
    img = Image.new("RGBA", (20, 12), (10, 20, 30, 255))
    img.putpixel((5, 5), (200, 100, 50, 255))
    a = _png(img, compress_level=0)
    b = _png(img, compress_level=9)
    assert a != b  # the encoders really did produce different bytes
    assert _matches(a, b) is True


def test_different_pixels_do_not_match():
    a = _png(Image.new("RGBA", (20, 12), (0, 0, 0, 255)))
    b = _png(Image.new("RGBA", (20, 12), (255, 255, 255, 255)))
    assert _matches(a, b) is False


def test_different_size_does_not_match():
    a = _png(Image.new("RGBA", (20, 12), (0, 0, 0, 255)))
    b = _png(Image.new("RGBA", (24, 12), (0, 0, 0, 255)))
    assert _matches(a, b) is False


def test_non_image_falls_back_to_bytes():
    assert _matches(b"not-a-png", b"not-a-png") is True
    assert _matches(b"not-a-png", b"other") is False
