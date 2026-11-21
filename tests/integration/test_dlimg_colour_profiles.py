"""dlimg must render wide-gamut sources in sRGB, whichever way the image reaches it."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import pytest
from PIL import Image, ImageCms

from odl_renderer import generate_image
from tests.icc_profiles import adobe_rgb_profile, adobe_rgb_to_srgb

RAW_COLOUR = (64, 200, 96)
CONVERTED_COLOUR = adobe_rgb_to_srgb(RAW_COLOUR)
TOLERANCE = 2
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _png(icc_profile: bytes | None, colour: tuple[int, ...] = RAW_COLOUR, size: tuple[int, int] = (20, 20)) -> bytes:
    buf = io.BytesIO()
    image = Image.new("RGBA" if len(colour) == 4 else "RGB", size, colour)
    image.save(buf, format="PNG", **({"icc_profile": icc_profile} if icc_profile is not None else {}))
    return buf.getvalue()


def _as_bytes(data: bytes) -> bytes:
    return data


def _as_data_uri(data: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def _as_pil_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


async def _render(source: Any, **options: Any) -> Image.Image:
    return await generate_image(
        width=40,
        height=40,
        background="white",
        elements=[{"type": "dlimg", "x": 0, "y": 0, "url": source, "xsize": 20, "ysize": 20, **options}],
    )


def _assert_close(actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    assert all(abs(a - e) <= TOLERANCE for a, e in zip(actual[:3], expected[:3])), f"{actual} is not near {expected}"


def test_the_test_colour_changes_when_converted():
    """Guards the tests below: they prove nothing if conversion leaves this colour alone."""
    assert max(abs(raw - converted) for raw, converted in zip(RAW_COLOUR, CONVERTED_COLOUR)) > 10 * TOLERANCE


@pytest.mark.asyncio
class TestWideGamutSource:
    @pytest.mark.parametrize("wrap", [_as_bytes, _as_data_uri, _as_pil_image])
    async def test_adobe_rgb_is_converted(self, wrap):
        image = await _render(wrap(_png(adobe_rgb_profile())))
        _assert_close(image.getpixel((5, 5)), CONVERTED_COLOUR)

    @pytest.mark.parametrize("options", [{"resize_method": "stretch"}, {"resize_method": "cover"}, {"rotate": 90}])
    async def test_adobe_rgb_is_converted_when_resized_or_rotated(self, options):
        image = await _render(_png(adobe_rgb_profile(), size=(60, 30)), **options)
        _assert_close(image.getpixel((10, 10)), CONVERTED_COLOUR)

    async def test_alpha_is_preserved(self):
        opaque = await _render(_png(adobe_rgb_profile(), colour=(*RAW_COLOUR, 255)))
        clear = await _render(_png(adobe_rgb_profile(), colour=(*RAW_COLOUR, 0)))
        _assert_close(opaque.getpixel((5, 5)), CONVERTED_COLOUR)
        assert clear.getpixel((5, 5))[:3] == (255, 255, 255)

    async def test_caller_owned_image_is_not_modified(self):
        source = _as_pil_image(_png(adobe_rgb_profile()))
        await _render(source)
        assert source.getpixel((5, 5)) == RAW_COLOUR
        assert source.info["icc_profile"] == adobe_rgb_profile()


@pytest.mark.asyncio
class TestSourcesLeftAlone:
    @pytest.mark.parametrize("description", ["sRGB IEC61966-2.1", "SRGB display"])
    async def test_profile_described_as_srgb_is_skipped(self, description):
        """Described as sRGB, so its Adobe RGB colorants must not be applied."""
        with_profile = await _render(_png(adobe_rgb_profile(description)))
        without_profile = await _render(_png(None))
        assert with_profile.tobytes() == without_profile.tobytes()

    async def test_srgb_is_pixel_identical(self):
        with_profile = await _render(_png(SRGB_PROFILE))
        without_profile = await _render(_png(None))
        assert with_profile.tobytes() == without_profile.tobytes()
        assert with_profile.getpixel((5, 5))[:3] == RAW_COLOUR

    async def test_unreadable_profile_warns_and_keeps_raw_values(self, caplog):
        with caplog.at_level(logging.WARNING, logger="odl_renderer.media_loader"):
            image = await _render(_png(b"this is not an ICC profile"))
        assert image.getpixel((5, 5))[:3] == RAW_COLOUR
        assert any("ICC profile" in record.getMessage() for record in caplog.records)
