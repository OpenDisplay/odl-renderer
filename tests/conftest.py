from pathlib import Path

import imagehash
import pytest
from PIL import Image, ImageFont
from syrupy.extensions.image import PNGImageSnapshotExtension

# Get package root
PACKAGE_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PACKAGE_ROOT / "src" / "drawcustom" / "assets"


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
    return snapshot.use_extension(PNGImageSnapshotExtension)
