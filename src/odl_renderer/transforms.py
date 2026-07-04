"""Generic per-element transforms (rotation, mirror) applied centrally.

An element is rendered onto its own transparent full-canvas layer, then the
layer is transformed and composited back onto the base image. This keeps every
element handler unchanged: any element type gains `rotation`/`mirror` support
for free.

Both transforms act around a pivot point. By default the pivot is the element's
rendered visual center (its bounding box center), so the element rotates/mirrors
in place. The pivot can be overridden per element.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from .coordinates import CoordinateParser

# Extra transparent margin (px) kept around the element when cropping before a
# transform, so the BICUBIC rotation kernel (±2 px support) samples the same
# neighborhood it would on the full canvas — keeping the output pixel-identical.
_CROP_MARGIN = 2


def has_transform(element: dict[str, Any]) -> bool:
    """Return True if an element requests a rotation or mirror transform."""
    return bool(element.get("rotation")) or bool(element.get("mirror"))


def _transform_layer(
    layer: Image.Image,
    rotation: float | int | None,
    mirror: str | None,
    px: float,
    py: float,
) -> Image.Image:
    """Apply mirror then rotation to *layer* about the pivot ``(px, py)``.

    ``(px, py)`` are in *layer*-local coordinates. The output keeps the input size.
    """
    if mirror:
        layer = _apply_mirror(layer, mirror, px, py)

    if rotation:
        # PIL rotates counter-clockwise; negate so positive = clockwise.
        layer = layer.rotate(
            -float(rotation),
            resample=Image.Resampling.BICUBIC,
            center=(px, py),
            expand=False,
        )

    return layer


def apply_transform(
    layer: Image.Image,
    *,
    rotation: float | int | None = None,
    mirror: str | None = None,
    pivot: Any = None,
    coords: CoordinateParser | None = None,
) -> Image.Image:
    """Apply mirror then rotation to a transparent element layer.

    Args:
        layer: Transparent RGBA layer containing a single rendered element.
        rotation: Degrees, positive = clockwise. None/0 skips rotation.
        mirror: "h", "v", or "hv" (case-insensitive). None/"" skips mirror.
        pivot: Optional pivot. Either an anchor keyword (e.g. "tl", "mm", "br")
               resolved against the element bbox, or an [x, y] canvas-coordinate
               pair parsed via ``coords``. Defaults to the bbox center.
        coords: CoordinateParser, required only to resolve an [x, y] pivot.

    Returns:
        The transformed layer (same size as the input). If the layer is empty
        (nothing was drawn), it is returned unchanged.
    """
    bbox = layer.getbbox()
    if bbox is None:
        # Nothing was drawn — no transform to apply, nothing to composite.
        return layer

    px, py = _resolve_pivot(pivot, bbox, coords)
    return _transform_layer(layer, rotation, mirror, px, py)


def apply_transform_region(
    layer: Image.Image,
    *,
    rotation: float | int | None = None,
    mirror: str | None = None,
    pivot: Any = None,
    coords: CoordinateParser | None = None,
) -> tuple[Image.Image, tuple[int, int]] | None:
    """Transform only the region of *layer* the element occupies.

    Same result as :func:`apply_transform` followed by an ``alpha_composite`` of the
    full layer, but the mirror/rotate run on a crop around the element instead of the
    whole canvas — so the cost is proportional to the element size, not the canvas.

    Mirror and rotation are both isometries about the pivot, so every non-transparent
    pixel stays within distance ``R`` (the farthest bbox corner from the pivot) of it.
    A crop of that radius (plus a resampling margin), clipped to the canvas, therefore
    contains all source *and* transformed content, and transforming it about the
    translated pivot yields pixels identical to the full-canvas path.

    Returns:
        ``(transformed_crop, (offset_x, offset_y))`` to composite the crop at, or
        ``None`` if the layer is empty or the element lies entirely off-canvas.
    """
    bbox = layer.getbbox()
    if bbox is None:
        return None

    px, py = _resolve_pivot(pivot, bbox, coords)

    left, top, right, bottom = bbox
    radius = max(math.hypot(cx - px, cy - py) for cx in (left, right) for cy in (top, bottom))
    reach = math.ceil(radius) + _CROP_MARGIN

    ox = max(0, math.floor(px - reach))
    oy = max(0, math.floor(py - reach))
    ex = min(layer.width, math.ceil(px + reach))
    ey = min(layer.height, math.ceil(py + reach))
    if ex <= ox or ey <= oy:
        return None

    crop = layer.crop((ox, oy, ex, ey))
    transformed = _transform_layer(crop, rotation, mirror, px - ox, py - oy)
    return transformed, (ox, oy)


def _apply_mirror(layer: Image.Image, mirror: str, px: float, py: float) -> Image.Image:
    """Mirror a layer about the vertical line x=px and/or horizontal line y=py.

    Uses an affine transform so the flip happens about an arbitrary pivot axis
    rather than the image center. Unknown mirror codes are ignored (lenient).
    """
    code = mirror.strip().lower()
    flip_h = "h" in code
    flip_v = "v" in code
    if not (flip_h or flip_v):
        return layer

    # Affine maps output (x, y) -> input (a*x + b*y + c, d*x + e*y + f).
    # Horizontal flip about x=px: input_x = 2*px - x.  Vertical about y=py: input_y = 2*py - y.
    a = -1.0 if flip_h else 1.0
    c = 2.0 * px if flip_h else 0.0
    e = -1.0 if flip_v else 1.0
    f = 2.0 * py if flip_v else 0.0

    return layer.transform(
        layer.size,
        Image.Transform.AFFINE,
        (a, 0.0, c, 0.0, e, f),
        resample=Image.Resampling.NEAREST,
        fillcolor=(0, 0, 0, 0),
    )


def _resolve_pivot(
    pivot: Any,
    bbox: tuple[int, int, int, int],
    coords: CoordinateParser | None,
) -> tuple[float, float]:
    """Resolve a pivot spec to absolute canvas coordinates.

    An [x, y] pair is parsed as canvas coordinates (percentages supported); a
    string is treated as a bbox-relative anchor keyword. Anything else (or a
    missing ``coords`` for the coordinate form) falls back to the bbox center.
    """
    if isinstance(pivot, (list, tuple)) and len(pivot) == 2 and coords is not None:
        return float(coords.parse_x(pivot[0])), float(coords.parse_y(pivot[1]))

    key = pivot if isinstance(pivot, str) else "mm"
    return _anchor_point(key, bbox)


def _anchor_point(key: str, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Resolve a two-letter anchor keyword to a bbox point.

    Uses the same Pillow anchor format as text: the first character selects the
    horizontal edge (``l``/``m``/``r``) and the second the vertical edge
    (``t``/``m``/``b``). Missing or unrecognized characters default to the
    middle, so ``"mm"`` (or ``""``) is the center.
    """
    left, top, right, bottom = bbox
    x_by = {"l": left, "m": (left + right) / 2, "r": right}
    y_by = {"t": top, "m": (top + bottom) / 2, "b": bottom}

    lowered = key.strip().lower()
    h = lowered[0] if len(lowered) > 0 else "m"
    v = lowered[1] if len(lowered) > 1 else "m"
    return float(x_by.get(h, (left + right) / 2)), float(y_by.get(v, (top + bottom) / 2))
