"""Unit tests for cutter_pipeline.image_extractor."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

# rembg loads a 170 MB model at import time when enabled — disable it for tests.
os.environ.setdefault("REMBG_ENABLED", "false")

from cutter_pipeline import image_extractor as ie  # noqa: E402


def _binary_outline(size=(96, 96)) -> Image.Image:
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, size[0] - 20, size[1] - 20], outline=0, width=2)
    return img


def _solid_silhouette(size=(96, 96)) -> Image.Image:
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    d.rectangle([24, 24, size[0] - 24, size[1] - 24], fill=0)
    return img


def _dashed_outline(size=(128, 128)) -> Image.Image:
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    # Dashed circle: 16 short arcs around the ring.
    for k in range(16):
        start = k * (360 / 16)
        d.arc([16, 16, size[0] - 16, size[1] - 16], start=start, end=start + 12, fill=0, width=3)
    return img


def _uniform_bg(size=(120, 120)) -> Image.Image:
    # Background colour is uniform but foreground is mid-tone so the classifier
    # bypasses the "binary" branch and lands in "simple_bg" via the corner-std check.
    img = Image.new("RGB", size, (220, 230, 240))
    d = ImageDraw.Draw(img)
    d.ellipse([30, 30, size[0] - 30, size[1] - 30], fill=(140, 110, 90))
    return img


# ── Classification ────────────────────────────────────────────────────────────

def test_classify_binary_outline():
    assert ie.classify_image(_solid_silhouette()) == "binary"


def test_classify_dashed_outline():
    assert ie.classify_image(_dashed_outline()) == "dashed"


def test_classify_uniform_background():
    assert ie.classify_image(_uniform_bg()) == "simple_bg"


def test_classify_complex_when_noisy():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    assert ie.classify_image(Image.fromarray(arr, "RGB")) == "complex"


# ── Mask extractors ───────────────────────────────────────────────────────────

def test_extract_mask_binary_returns_foreground():
    gray = np.array(_solid_silhouette().convert("L"))
    mask = ie.extract_mask_binary(gray, threshold=200)
    assert mask.dtype == bool
    assert mask.any()
    # The foreground rectangle covers ~ (48*48)/(96*96) ≈ 0.25 of the image.
    frac = mask.mean()
    assert 0.18 < frac < 0.32


def test_extract_mask_simple_bg_returns_foreground():
    rgb = np.array(_uniform_bg().convert("RGB"))
    mask = ie.extract_mask_simple_bg(rgb)
    assert mask.any()
    assert 0.1 < mask.mean() < 0.7


def test_extract_mask_dashed_bridges_gaps():
    gray = np.array(_dashed_outline().convert("L"))
    mask = ie.extract_mask_dashed(gray)
    assert mask.any()
    # The reconstructed silhouette should be a connected region, not many specks.
    from skimage import measure as _m
    n = int(_m.label(mask).max())
    assert n <= 2


def test_extract_mask_dashed_returns_empty_on_uniform_input():
    # A pure-white frame has no dashes to bridge — extractor should return empty.
    gray = np.full((64, 64), 255, dtype=np.uint8)
    mask = ie.extract_mask_dashed(gray)
    assert not mask.any()


# ── Public entry-point ────────────────────────────────────────────────────────

def test_extract_foreground_mask_auto_picks_binary_for_silhouette():
    mask, mode, warn = ie.extract_foreground_mask(_solid_silhouette(), mode="auto")
    assert mode == "binary"
    assert warn == ""
    assert mask.sum() > 0


def test_extract_foreground_mask_complex_disabled_raises(monkeypatch):
    monkeypatch.setattr(ie, "REMBG_ENABLED", False)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    with pytest.raises(ValueError, match="Background Removal"):
        ie.extract_foreground_mask(img, mode="auto")


def test_extract_foreground_mask_forced_mode_overrides_classifier():
    # Force "simple_bg" on an image that classify would call "binary".
    mask, mode, _warn = ie.extract_foreground_mask(_solid_silhouette(), mode="simple_bg")
    assert mode == "simple_bg"


def test_extract_foreground_mask_dashed_raises_when_gaps_too_wide():
    # A near-binary image but with no enclosable region: just scattered dots.
    img = Image.new("L", (96, 96), 255)
    d = ImageDraw.Draw(img)
    for x in range(8, 88, 20):
        for y in range(8, 88, 20):
            d.point((x, y), fill=0)
    # Classifier may or may not call this dashed; force the mode to exercise the path.
    with pytest.raises(ValueError, match="(?i)dashed"):
        ie.extract_foreground_mask(img, mode="dashed")
