"""Minimal ICC v2 display profiles, built by hand so tests need no binary fixtures.

Pillow can only create sRGB, LAB and XYZ profiles, and wide-gamut ones are what tests need.
"""

from __future__ import annotations

import struct

D50 = (0.9642, 1.0, 0.8249)
ADOBE_RGB_GAMMA = 2.19921875
# Adobe RGB (1998) colorants, chromatically adapted to the D50 profile connection space.
ADOBE_RGB_COLORANTS = (
    (0.60974, 0.31111, 0.01947),
    (0.20528, 0.62567, 0.06087),
    (0.14919, 0.06322, 0.74457),
)


def _xyz(xyz: tuple[float, float, float]) -> bytes:
    return b"XYZ \0\0\0\0" + b"".join(struct.pack(">i", round(value * 65536)) for value in xyz)


def _gamma(gamma: float) -> bytes:
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256)) + b"\0\0"


def _description(text: str) -> bytes:
    ascii_text = text.encode("ascii") + b"\0"
    return b"desc\0\0\0\0" + struct.pack(">I", len(ascii_text)) + ascii_text + bytes(4 + 4 + 2 + 1 + 67)


def _profile(colour_space: bytes, description: str, tags: dict[bytes, bytes]) -> bytes:
    all_tags = {
        b"desc": _description(description),
        b"cprt": b"text\0\0\0\0test\0\0\0\0",
        b"wtpt": _xyz(D50),
        **tags,
    }
    table_size = 4 + 12 * len(all_tags)
    table = struct.pack(">I", len(all_tags))
    body = b""
    for signature, data in all_tags.items():
        padded = data + bytes(-len(data) % 4)
        table += signature + struct.pack(">II", 128 + table_size + len(body), len(data))
        body += padded

    header = bytearray(128)
    struct.pack_into(">I", header, 0, 128 + table_size + len(body))
    struct.pack_into(">I", header, 8, 0x02400000)
    header[12:24] = b"mntr" + colour_space + b"XYZ "
    header[36:40] = b"acsp"
    header[68:80] = _xyz(D50)[8:]
    return bytes(header) + table + body


def adobe_rgb_profile(description: str = "Adobe RGB (1998) test profile") -> bytes:
    red, green, blue = ADOBE_RGB_COLORANTS
    curve = _gamma(ADOBE_RGB_GAMMA)
    return _profile(
        b"RGB ",
        description,
        {
            b"rXYZ": _xyz(red),
            b"gXYZ": _xyz(green),
            b"bXYZ": _xyz(blue),
            b"rTRC": curve,
            b"gTRC": curve,
            b"bTRC": curve,
        },
    )


def gray_profile(gamma: float = 1.0) -> bytes:
    return _profile(b"GRAY", "Linear gray test profile", {b"kTRC": _gamma(gamma)})


def adobe_rgb_to_srgb(rgb: tuple[int, int, int]) -> tuple[int, ...]:
    """Convert an 8-bit Adobe RGB colour to sRGB by the published maths, without a CMS."""
    linear = [(channel / 255) ** ADOBE_RGB_GAMMA for channel in rgb]
    xyz = [sum(colorant[axis] * value for colorant, value in zip(ADOBE_RGB_COLORANTS, linear)) for axis in range(3)]
    # XYZ (D50) to linear sRGB, the Bradford-adapted matrix from brucelindbloom.com.
    matrix = (
        (3.1338561, -1.6168667, -0.4906146),
        (-0.9787684, 1.9161415, 0.0334540),
        (0.0719453, -0.2289914, 1.4052427),
    )
    return tuple(_encode_srgb(sum(c * v for c, v in zip(row, xyz))) for row in matrix)


def _encode_srgb(linear: float) -> int:
    clipped = min(max(linear, 0.0), 1.0)
    encoded = 12.92 * clipped if clipped <= 0.0031308 else 1.055 * clipped ** (1 / 2.4) - 0.055
    return round(encoded * 255)
