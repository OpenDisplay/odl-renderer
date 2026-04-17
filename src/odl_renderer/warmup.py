"""Pre-warm blocking I/O so it doesn't hit the event loop on first render."""

from __future__ import annotations

_DEFAULT_SIZES = (16, 24, 32, 48, 64, 96)


def warmup(sizes: tuple[int, ...] = _DEFAULT_SIZES) -> None:
    """Pre-load MDI fonts for common sizes into the module-level cache.

    Run this in a thread executor (e.g. ``loop.run_in_executor``) during
    application startup so the first render is not delayed by font I/O.
    Importing this module already triggers the MDI index load, so only the
    per-size font files need explicit pre-warming here.
    """
    from .elements.icons import _get_mdi_font

    for size in sizes:
        try:
            _get_mdi_font(size)
        except Exception:  # noqa: BLE001
            break
