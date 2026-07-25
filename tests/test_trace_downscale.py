"""Large images are downscaled before tracing.

Peak memory scales with pixel count — the extractor's LAB conversion alone
costs 24 bytes/pixel — so a 12 MP phone photo peaked near 1 GB and would
OOM-kill a small instance. Tracing normalises coordinates to 0..1 and
simplifies in that space, so the extra resolution buys nothing.
"""
from __future__ import annotations

import tempfile

import pytest
from PIL import Image

from cutter_pipeline import trace_outline
from cutter_pipeline.trace_outline import fit_for_tracing, trace_png_to_polygon


def _blob_image(size: tuple[int, int]) -> Image.Image:
    """A dark rounded-ish blob on white — traceable at any resolution."""
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    img.paste(Image.new("RGB", (w // 2, h // 2), (10, 10, 10)), (w // 4, h // 4))
    return img


def test_small_images_are_untouched(monkeypatch):
    monkeypatch.setattr(trace_outline, "MAX_TRACE_PIXELS", 2_000_000)
    img = _blob_image((800, 600))
    out, scale = fit_for_tracing(img)
    assert scale == 1.0
    assert out.size == (800, 600)


def test_large_images_are_capped_and_keep_aspect_ratio(monkeypatch):
    monkeypatch.setattr(trace_outline, "MAX_TRACE_PIXELS", 1_000_000)
    img = _blob_image((4000, 3000))
    out, scale = fit_for_tracing(img)
    w, h = out.size
    assert w * h <= 1_000_000
    assert scale < 1.0
    # 4:3 preserved to within a pixel of rounding
    assert abs((w / h) - (4000 / 3000)) < 0.01


def test_cap_can_be_disabled(monkeypatch):
    monkeypatch.setattr(trace_outline, "MAX_TRACE_PIXELS", 0)
    img = _blob_image((3000, 3000))
    out, scale = fit_for_tracing(img)
    assert scale == 1.0
    assert out.size == (3000, 3000)


def test_downscaling_does_not_change_the_traced_shape(monkeypatch, tmp_path):
    """The whole premise of the cap: the polygon must come out the same."""
    big = _blob_image((3000, 3000))
    path = tmp_path / "big.png"
    big.save(path)

    monkeypatch.setattr(trace_outline, "MAX_TRACE_PIXELS", 0)
    full = trace_png_to_polygon(str(path), str(tmp_path / "full.svg")).polygon

    monkeypatch.setattr(trace_outline, "MAX_TRACE_PIXELS", 500_000)
    capped = trace_png_to_polygon(str(path), str(tmp_path / "capped.svg")).polygon

    assert full is not None and capped is not None
    assert capped.area == pytest.approx(full.area, rel=0.02)
    # Centroids should land on top of each other in normalised space.
    assert capped.centroid.distance(full.centroid) < 0.01


def test_trace_still_succeeds_on_an_oversized_image(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_outline, "MAX_TRACE_PIXELS", 250_000)
    path = tmp_path / "huge.png"
    _blob_image((2600, 2600)).save(path)  # 6.8 MP in, 0.25 MP traced
    result = trace_png_to_polygon(str(path), str(tmp_path / "out.svg"))
    assert result.polygon is not None
    assert result.polygon.area > 0
