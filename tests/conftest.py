from io import BytesIO
from pathlib import Path

import imagehash
import pytest
from PIL import Image, ImageChops, ImageFont, UnidentifiedImageError
from syrupy.extensions.image import PNGImageSnapshotExtension

# Get package root
PACKAGE_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PACKAGE_ROOT / "src" / "odl_renderer" / "assets"


class PixelPNGImageSnapshotExtension(PNGImageSnapshotExtension):
    """PNG snapshot extension that matches on decoded pixels, not encoded bytes.

    The default image extension compares raw PNG bytes, so a Pillow/zlib upgrade
    that re-encodes pixel-identical output into different bytes breaks every
    snapshot (observed on main under Pillow 12). Decoding both sides and comparing
    pixels keeps snapshots stable across encoder changes while still catching any
    real visual difference.
    """

    def matches(self, *, serialized_data: object, snapshot_data: object) -> bool:
        try:
            new_img = Image.open(BytesIO(bytes(serialized_data)))  # type: ignore[arg-type]
            old_img = Image.open(BytesIO(bytes(snapshot_data)))  # type: ignore[arg-type]
        except (UnidentifiedImageError, OSError, TypeError, ValueError):
            # Not decodable as images — fall back to the byte comparison.
            return bool(serialized_data == snapshot_data)
        if new_img.size != old_img.size:
            return False
        diff = ImageChops.difference(new_img.convert("RGBA"), old_img.convert("RGBA"))
        # alpha_only=False so a difference in any RGB channel counts — the default
        # (alpha-only) would ignore color changes on fully-opaque e-paper renders.
        return diff.getbbox(alpha_only=False) is None


@pytest.fixture
def ppb_font():
    """Provide path to bundled ppb font."""
    return str(ASSETS_DIR / "ppb.ttf")


@pytest.fixture
def rbm_font():
    """Provide path to bundled rbm font."""
    return str(ASSETS_DIR / "rbm.ttf")


@pytest.fixture
def load_font():
    """Factory fixture for loading fonts."""

    def _load(name: str, size: int = 16) -> ImageFont.FreeTypeFont:
        font_path = ASSETS_DIR / f"{name}.ttf"
        if not font_path.exists():
            pytest.skip(f"Font {name} not available")
        return ImageFont.truetype(str(font_path), size)

    return _load


@pytest.fixture
def assert_images_similar():
    """Fixture for fuzzy image comparison."""

    def _compare(img1: Image.Image, img2: Image.Image, threshold: int = 5):
        hash1 = imagehash.average_hash(img1)
        hash2 = imagehash.average_hash(img2)
        distance = hash1 - hash2
        assert distance <= threshold, f"Images differ by {distance} (threshold: {threshold})"

    return _compare


@pytest.fixture
def snapshot_png(snapshot):
    return snapshot.use_extension(PixelPNGImageSnapshotExtension)
