"""Unit tests for warmup (finding A12: one bad size must not skip the rest)."""

from odl_renderer.warmup import warmup


def test_warmup_continues_past_a_failing_size(monkeypatch):
    attempted = []

    def fake_get_mdi_font(size):
        attempted.append(size)
        if size == 24:
            raise OSError("simulated font load failure")
        return object()

    # warmup imports _get_mdi_font from elements.icons inside the function.
    monkeypatch.setattr("odl_renderer.elements.icons._get_mdi_font", fake_get_mdi_font)

    warmup(sizes=(16, 24, 32))

    # All sizes attempted despite 24 failing — no early break.
    assert attempted == [16, 24, 32]
