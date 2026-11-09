"""Unit tests for media_loader — covers file, bytes, data URI, and error paths."""

from __future__ import annotations

import base64
import io
import logging
import tempfile

import pytest
from PIL import Image, ImageCms

from odl_renderer.media_loader import convert_to_srgb, load_image
from tests.icc_profiles import adobe_rgb_profile, gray_profile


def _make_png_bytes() -> bytes:
    """Create minimal valid PNG bytes."""
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_data_uri() -> str:
    png = _make_png_bytes()
    b64 = base64.b64encode(png).decode()
    return f"data:image/png;base64,{b64}"


@pytest.mark.asyncio
class TestLoadImage:
    async def test_pil_image_returned_as_is(self):
        img = Image.new("RGB", (10, 10))
        result = await load_image(img)
        assert result is img

    async def test_bytes_decoded(self):
        png_bytes = _make_png_bytes()
        result = await load_image(png_bytes)
        assert isinstance(result, Image.Image)

    async def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            await load_image(b"not an image")

    async def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid image source type"):
            await load_image(12345)  # type: ignore[arg-type]

    async def test_data_uri(self):
        uri = _make_data_uri()
        result = await load_image(uri)
        assert isinstance(result, Image.Image)

    async def test_data_uri_non_base64_raises(self):
        with pytest.raises(ValueError, match="Only base64"):
            await load_image("data:image/png,rawdata")

    async def test_file_path(self):
        png_bytes = _make_png_bytes()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            path = f.name
        result = await load_image(path)
        assert isinstance(result, Image.Image)

    async def test_file_path_releases_handle(self):
        """The file is decoded eagerly and its handle released (finding A12)."""
        png_bytes = _make_png_bytes()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            path = f.name
        result = await load_image(path)
        # After .load(), Pillow drops the file pointer it owned.
        assert getattr(result, "fp", None) is None
        # The image is usable without touching the file again.
        assert result.convert("RGBA").size == result.size

    async def test_nonexistent_file_raises(self):
        with pytest.raises(ValueError, match="not found"):
            await load_image("/nonexistent/path/image.png")

    async def test_invalid_source_string_raises(self):
        with pytest.raises(ValueError, match="Invalid image source"):
            await load_image("relative/path/image.png")


class TestConvertToSrgb:
    def test_image_without_profile_is_returned_unchanged(self):
        image = Image.new("RGB", (2, 2), (10, 20, 30))
        assert convert_to_srgb(image) is image

    def test_srgb_image_is_returned_unchanged(self):
        image = Image.new("RGB", (2, 2), (10, 20, 30))
        image.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        assert convert_to_srgb(image) is image

    def test_converted_image_is_tagged_srgb(self):
        image = Image.new("RGB", (2, 2), (64, 200, 96))
        image.info["icc_profile"] = adobe_rgb_profile()
        converted = convert_to_srgb(image)
        profile = ImageCms.ImageCmsProfile(io.BytesIO(converted.info["icc_profile"]))
        assert "sRGB" in ImageCms.getProfileDescription(profile)

    def test_palette_image_is_converted(self):
        image = Image.new("RGB", (2, 2), (64, 200, 96)).convert("P")
        image.info["icc_profile"] = adobe_rgb_profile()
        assert convert_to_srgb(image).getpixel((0, 0)) != image.convert("RGB").getpixel((0, 0))

    def test_palette_image_keeps_transparency(self):
        image = Image.new("RGBA", (2, 2), (64, 200, 96, 0)).convert("P")
        image.info["transparency"] = image.getpixel((0, 0))
        image.info["icc_profile"] = adobe_rgb_profile()
        converted = convert_to_srgb(image)
        assert converted.mode == "RGBA"
        assert converted.getpixel((0, 0))[3] == 0

    def test_gray_image_with_gray_profile_is_converted(self):
        image = Image.new("L", (2, 2), 128)
        image.info["icc_profile"] = gray_profile(gamma=1.0)
        converted = convert_to_srgb(image).convert("RGB")
        # Linear 0.5 encodes to about 188 in sRGB.
        assert abs(converted.getpixel((0, 0))[0] - 188) <= 2

    def test_gray_image_with_alpha_keeps_alpha(self):
        image = Image.new("LA", (2, 2), (128, 77))
        image.info["icc_profile"] = gray_profile(gamma=1.0)
        red, _, _, alpha = convert_to_srgb(image).convert("RGBA").getpixel((0, 0))
        assert abs(red - 188) <= 2
        assert alpha == 77

    def test_profile_not_matching_the_image_mode_warns_and_keeps_the_image(self, caplog):
        image = Image.new("L", (2, 2), 128)
        image.info["icc_profile"] = adobe_rgb_profile()
        with caplog.at_level(logging.WARNING, logger="odl_renderer.media_loader"):
            assert convert_to_srgb(image) is image
        assert any("ICC profile" in record.getMessage() for record in caplog.records)

    def test_truncated_image_data_is_not_reported_as_a_bad_profile(self, caplog):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (64, 200, 96)).save(buf, format="PNG", icc_profile=adobe_rgb_profile())
        image = Image.open(io.BytesIO(buf.getvalue()[:-40]))
        with caplog.at_level(logging.WARNING, logger="odl_renderer.media_loader"), pytest.raises(OSError):
            convert_to_srgb(image)
        assert caplog.records == []
