"""
image_extractor.py
~~~~~~~~~~~~~~~~~~
Classifies an input image and extracts a binary foreground mask using the
best available strategy — no cloud services required.

Four paths:
  "binary"     – image is already a line-art / threshold-able outline.
                 Use simple luminance threshold (existing behaviour).
  "dashed"     – outline drawn as dashed/dotted strokes. Bridge the gaps
                 with morphological dilation, fill the enclosed interior,
                 then erode back to recover the silhouette.
  "simple_bg"  – image has a roughly uniform background (e.g. product shot
                 on white, logo on solid colour). Use LAB colour-distance
                 from sampled corners.
  "complex"    – photographic / textured background.  Tries rembg (local
                 neural net, no API key) if installed, otherwise falls back
                 to Felzenszwalb graph-cut segmentation with a quality
                 warning.

All paths return a boolean NumPy array (True = foreground) of the same
shape as the input image, ready for skimage.measure.find_contours.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, gaussian_filter
from skimage import measure, morphology
from skimage.color import rgb2lab
from skimage.segmentation import felzenszwalb

logger = logging.getLogger(__name__)

_rembg_enabled_raw = os.environ.get("REMBG_ENABLED", "true").strip().lower()
REMBG_ENABLED: bool = _rembg_enabled_raw not in ("false", "0", "no")
if not REMBG_ENABLED:
    logger.info("rembg is disabled via REMBG_ENABLED environment variable.")

# Pre-load the U2Net session once at startup so the model is not reloaded from
# disk on every request.  rembg.remove() creates a fresh session (and reloads
# the ~170 MB ONNX model) on each call when no session is supplied — this is a
# documented bottleneck acknowledged in the rembg README.
_rembg_session = None
if REMBG_ENABLED:
    try:
        from rembg import new_session as _rembg_new_session
        _rembg_session = _rembg_new_session("u2net")
        logger.info("rembg U2Net session initialised and model loaded into memory.")
    except Exception as _e:
        logger.warning("rembg session initialisation failed: %s", _e)

ImageMode = Literal["binary", "dashed", "simple_bg", "complex"]

# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_image(img: Image.Image) -> ImageMode:
    """Analyse a PIL image and return the recommended extraction mode."""
    gray = np.array(img.convert("L"))
    rgb  = np.array(img.convert("RGB"))

    # 1. Bimodal / binary-outline check
    #    If most pixels are near 0 (dark) or 255 (light) the image is already
    #    a rendered outline — use the existing threshold path.
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total   = gray.size
    low_pct  = hist[:50].sum()  / total   # very dark
    high_pct = hist[210:].sum() / total   # very light
    mid_pct  = 1.0 - low_pct - high_pct   # mid-tones

    if mid_pct < 0.12:
        # A near-binary image *could* still be a dashed/dotted outline rather
        # than a solid silhouette or single closed line. Detect that here so
        # the extractor can bridge the gaps before tracing.
        if _looks_dashed(gray):
            logger.info(
                "Image classified as DASHED OUTLINE — %.1f%% mid-tone pixels "
                "and foreground breaks into many small components. "
                "Using dilate-fill-erode extraction.",
                mid_pct * 100,
            )
            return "dashed"
        logger.info(
            "Image classified as BINARY OUTLINE — %.1f%% mid-tone pixels "
            "(threshold: <12%%). Using luminance threshold extraction.",
            mid_pct * 100,
        )
        return "binary"

    # 2. Uniform-background check
    #    Sample the four corners (5 % of each side).  If their pixel values
    #    are consistent the background is uniform enough for colour-distance
    #    extraction.
    h, w    = gray.shape
    margin  = max(10, int(min(h, w) * 0.05))
    corners = np.vstack([
        gray[:margin,  :margin ].ravel(),
        gray[:margin,  -margin:].ravel(),
        gray[-margin:, :margin ].ravel(),
        gray[-margin:, -margin:].ravel(),
    ])
    corner_std = float(np.std(corners))

    if corner_std < 28:
        logger.info(
            "Image classified as UNIFORM BACKGROUND — corner pixel std=%.1f "
            "(threshold: <28). Using LAB colour-distance extraction.",
            corner_std,
        )
        return "simple_bg"

    logger.info(
        "Image classified as COMPLEX / PHOTOGRAPHIC — corner pixel std=%.1f, "
        "%.1f%% mid-tone pixels. Extraction method: %s.",
        corner_std,
        mid_pct * 100,
        "rembg (U2Net)" if REMBG_ENABLED else "graph-cut (rembg disabled)",
    )
    return "complex"


# ──────────────────────────────────────────────────────────────────────────────
# Mask extractors — each returns a bool array (True = foreground)
# ──────────────────────────────────────────────────────────────────────────────

def extract_mask_binary(gray: np.ndarray, threshold: int = 200) -> np.ndarray:
    """Existing behaviour: simple luminance threshold."""
    return gray < threshold


def _looks_dashed(gray: np.ndarray, threshold: int = 200) -> bool:
    """
    Decide whether a near-binary image is a dashed/dotted outline rather
    than a solid silhouette or single closed-line outline.

    Heuristic: threshold the image and look at connected components of the
    dark foreground. A solid silhouette or a single closed-line outline
    has 1-2 dominant components; a dashed line breaks into many small
    pieces with no single dominant blob.
    """
    fg = gray < threshold
    if not fg.any():
        return False
    labels = measure.label(fg)
    n = int(labels.max())
    if n < 8:
        return False
    region_areas = np.bincount(labels.ravel())[1:]  # drop background
    largest = int(region_areas.max())
    total = int(region_areas.sum())
    return largest / total < 0.5


def extract_mask_dashed(
    gray: np.ndarray,
    threshold: int = 200,
    bridge_radius: int | None = None,
) -> np.ndarray:
    """
    Foreground extraction for outlines drawn as dashed or dotted strokes.

    1.  Threshold to recover the dark dashes.
    2.  Dilate the dashes just enough to merge them into a closed loop.
    3.  Flood-fill the now-enclosed interior.
    4.  Erode by the same radius to recover the original silhouette.
    5.  Clean up small specks and tiny holes.

    When *bridge_radius* is None, the smallest radius that successfully
    closes the loop is used — this preserves fine detail (small bumps,
    petals, scallops) that a larger radius would smooth away.
    """
    h, w = gray.shape
    raw = gray < threshold

    if bridge_radius is not None:
        radii = [bridge_radius]
    else:
        max_r = max(12, int(round(min(h, w) * 0.05)))
        radii = list(range(2, max_r + 1))

    chosen_r: int | None = None
    closed: np.ndarray | None = None
    for r in radii:
        struct = morphology.disk(r)
        dilated = binary_dilation(raw, structure=struct)
        filled = binary_fill_holes(dilated)
        # gain > 1.5 means fill_holes enclosed an interior — the dashes
        # now form a closed loop after dilation.
        if filled is not None and filled.sum() > dilated.sum() * 1.5:
            chosen_r = r
            closed = filled
            break

    if closed is None or chosen_r is None:
        return np.zeros_like(raw, dtype=bool)

    logger.info("Dashed extraction bridged at radius=%d.", chosen_r)
    mask = binary_erosion(closed, structure=morphology.disk(chosen_r))

    if not mask.any():
        return mask

    min_area = max(200, (h * w) // 200)
    mask = morphology.remove_small_objects(mask, max_size=min_area)
    mask = morphology.remove_small_holes(mask, max_size=min_area * 4)
    return mask


def extract_mask_simple_bg(
    rgb: np.ndarray,
    delta_e_threshold: float = 28.0,
    close_radius: int = 5,
    open_radius: int  = 2,
    min_object_px: int = 300,
    fill_hole_px:  int = 2000,
) -> np.ndarray:
    """
    Foreground extraction for images with a roughly uniform background.

    1.  Sample corner pixels → estimate background colour in LAB space.
    2.  Compute per-pixel ΔE (Euclidean distance in LAB).
    3.  Threshold → raw mask.
    4.  Morphological cleanup.
    """
    h, w   = rgb.shape[:2]
    margin = max(10, int(min(h, w) * 0.05))

    corner_pixels = np.vstack([
        rgb[:margin,  :margin ].reshape(-1, 3),
        rgb[:margin,  -margin:].reshape(-1, 3),
        rgb[-margin:, :margin ].reshape(-1, 3),
        rgb[-margin:, -margin:].reshape(-1, 3),
    ])
    # Median is robust to foreground objects that touch the corner edge
    bg_rgb = np.median(corner_pixels, axis=0).reshape(1, 1, 3).astype(np.float32) / 255.0

    lab_img = rgb2lab(rgb.astype(np.float32) / 255.0)
    bg_lab  = rgb2lab(bg_rgb)[0, 0]

    delta_e = np.sqrt(np.sum((lab_img - bg_lab) ** 2, axis=2))
    mask    = delta_e > delta_e_threshold

    # Morphological cleanup
    if close_radius > 0:
        mask = morphology.closing(mask, morphology.disk(close_radius))
    if open_radius > 0:
        mask = morphology.opening(mask, morphology.disk(open_radius))
    mask = morphology.remove_small_objects(mask, max_size=min_object_px)
    mask = morphology.remove_small_holes(mask, max_size=fill_hole_px)

    return mask


def extract_mask_complex(
    rgb: np.ndarray,
    *,
    scale: float  = 100.0,
    sigma: float  = 0.8,
    min_size: int = 200,
) -> tuple[np.ndarray, str]:
    """
    Foreground extraction for complex / photographic images.

    Tries rembg (local U2Net — no cloud, no API key) first.
    Falls back to Felzenszwalb graph-cut segmentation if rembg is not
    installed, returning a quality warning alongside the mask.

    Returns (mask, warning_message).  warning_message is empty string on
    success.
    """
    # ── Attempt rembg ──────────────────────────────────────────────────────
    if not REMBG_ENABLED:
        logger.info(
            "rembg is disabled (REMBG_ENABLED=false) — skipping to graph-cut fallback."
        )
        warning = (
            "rembg background removal is disabled. "
            "Falling back to graph-cut segmentation which may be less accurate "
            "for complex or photographic backgrounds."
        )
    else:
        if _rembg_session is not None:
            try:
                from rembg import remove as _rembg_remove  # type: ignore
                from PIL import Image as _PILImage

                pil_in  = _PILImage.fromarray(rgb)
                pil_out = _rembg_remove(pil_in, session=_rembg_session)  # reuses in-memory model
                alpha   = np.array(pil_out.split()[-1])     # A channel
                mask    = alpha > 10                         # near-transparent = background
                mask    = morphology.remove_small_objects(mask, max_size=300)
                mask    = morphology.remove_small_holes(mask, max_size=2000)
                logger.info("Extracting foreground with REMBG (cached U2Net session).")
                return mask, ""
            except ImportError:
                pass  # rembg not installed — continue to fallback

        # ── Felzenszwalb fallback ──────────────────────────────────────────────
        logger.warning(
            "rembg is not installed — falling back to GRAPH-CUT (Felzenszwalb) "
            "segmentation. Quality may be reduced for complex backgrounds. "
            "Install rembg[cpu] for best results: pip install 'rembg[cpu]'"
        )
        warning = (
            "Complex background detected. For best results install the 'rembg' "
            "package (pip install rembg) — it runs locally with no API key. "
            "Falling back to graph-cut segmentation which may be less accurate."
        )

    segments = felzenszwalb(rgb, scale=scale, sigma=sigma, min_size=min_size)

    # The foreground is assumed to be centred: take the segment that covers
    # the most of the central 50 % of the image.
    cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
    ch, cw = rgb.shape[0] // 4, rgb.shape[1] // 4
    centre  = segments[cy - ch : cy + ch, cx - cw : cx + cw]
    labels, counts = np.unique(centre, return_counts=True)
    fg_label = labels[np.argmax(counts)]

    mask = segments == fg_label
    mask = morphology.closing(mask, morphology.disk(5))
    mask = morphology.remove_small_objects(mask, max_size=300)
    mask = morphology.remove_small_holes(mask, max_size=2000)

    return mask, warning


# ──────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────────

def extract_foreground_mask(
    img: Image.Image,
    mode: ImageMode | Literal["auto"] = "auto",
    threshold: int = 200,
    delta_e_threshold: float = 28.0,
) -> tuple[np.ndarray, ImageMode, str]:
    """
    Extract a boolean foreground mask from *img*.

    Parameters
    ----------
    img               PIL image (any mode).
    mode              "auto" (default) classifies the image and picks the
                      best strategy; or pass "binary" / "simple_bg" /
                      "complex" to force a specific path.
    threshold         Luminance cut-off used in "binary" mode.
    delta_e_threshold ΔE cut-off used in "simple_bg" mode.

    Returns
    -------
    mask              bool ndarray, True = foreground.
    detected_mode     The mode that was actually used.
    warning           Non-empty string if quality may be reduced.
    """
    requested = mode
    if mode == "auto":
        mode = classify_image(img)
    else:
        logger.info("Extraction mode forced to '%s' (not auto-detected).", mode)

    gray = np.array(img.convert("L"))
    rgb  = np.array(img.convert("RGB"))
    warning = ""

    if mode == "binary":
        logger.info("Running extraction: BINARY threshold (luminance < %d).", threshold)
        mask = extract_mask_binary(gray, threshold=threshold)
    elif mode == "dashed":
        logger.info("Running extraction: DASHED outline (dilate-fill-erode, threshold=%d).", threshold)
        mask = extract_mask_dashed(gray, threshold=threshold)
        if not mask.any():
            raise ValueError(
                "Dashed outline detected but the gaps between dashes are too "
                "wide to bridge automatically. Try a higher-resolution scan, "
                "or fill the dashes in with a solid line."
            )
        # Soften the pixel-scale jaggies left by binary erosion so the
        # downstream find_contours produces a smooth curve. The 0.5
        # iso-level intersects the blurred boundary halfway, so the
        # silhouette size is preserved.
        return gaussian_filter(mask.astype(float), sigma=1.5), mode, warning
    elif mode == "simple_bg":
        logger.info("Running extraction: UNIFORM BACKGROUND colour-distance (ΔE threshold=%.1f).", delta_e_threshold)
        mask = extract_mask_simple_bg(rgb, delta_e_threshold=delta_e_threshold)
    else:  # complex
        if not REMBG_ENABLED:
            raise ValueError(
                "This image has a complex or photographic background that requires "
                "Background Removal, which is currently disabled. "
                "Try an image with a plain or uniform background instead."
            )
        logger.info("Running extraction: COMPLEX — attempting rembg, with graph-cut fallback.")
        mask, warning = extract_mask_complex(rgb)

    return mask.astype(float), mode, warning
