from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter
from shapely.geometry import LineString, Polygon
from skimage import measure

from cutter_pipeline.image_extractor import (
    ImageMode,
    extract_foreground_mask,
)


@dataclass
class TraceResult:
    polygon: Polygon
    svg_path: str
    svg_file: str
    extraction_mode: ImageMode = "binary"
    extraction_warning: str = ""


def _svg_from_coords(coords: list[tuple[float, float]]) -> str:
    d = f"M {coords[0][0]:.6f},{coords[0][1]:.6f} "
    for x, y in coords[1:]:
        d += f"L {x:.6f},{y:.6f} "
    d += "Z"
    return d


def trace_png_to_polygon(
    png_path: str,
    svg_out_path: str,
    threshold: int = 200,
    simplify_epsilon: float = 0.002,
    smooth_radius: float = 0.0,
    extraction_mode: Literal["auto", "binary", "dashed", "simple_bg", "complex"] = "auto",
    delta_e_threshold: float = 28.0,
) -> TraceResult:
    """
    Trace a PNG/JPG image to a Shapely polygon suitable for STL generation.

    Parameters
    ----------
    png_path          Path to the input image.
    svg_out_path      Where to write the preview SVG.
    threshold         Luminance cut-off for "binary" mode (0-255).
    simplify_epsilon  Douglas-Peucker tolerance for polygon simplification.
    smooth_radius     Gaussian blur radius applied before tracing (pixels).
    extraction_mode   "auto" (default) — automatically picks the best
                      strategy based on the image content:
                        "binary"    – pre-drawn outline / coloring-book art
                        "simple_bg" – subject on a uniform background
                        "complex"   – photographic / textured background
                      Pass an explicit value to override auto-detection.
    delta_e_threshold ΔE colour-distance threshold used in "simple_bg" mode.
    """
    img = Image.open(png_path)

    # Optional smoothing before extraction (helps soften aliased edges)
    if smooth_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=smooth_radius))

    binary, detected_mode, warning = extract_foreground_mask(
        img,
        mode=extraction_mode,
        threshold=threshold,
        delta_e_threshold=delta_e_threshold,
    )

    contours = measure.find_contours(binary, 0.5)
    if not contours:
        raise ValueError("No contours found. Try adjusting threshold or extraction mode.")

    contour = max(contours, key=lambda c: c.shape[0])
    arr = np.array(img.convert("L"))
    h, w = arr.shape

    pts = np.array(contour)
    # Flip Y to convert image (Y-down) to cartesian (Y-up), and flip X so the
    # printed cutter's cutting face matches the orientation of the source
    # image — without this mirror, STLs come out left/right reversed in slicers.
    # Normalise both axes by the SAME factor so non-square images keep their
    # aspect ratio (otherwise a wide image would be squashed horizontally).
    y = h - pts[:, 0]
    x = w - pts[:, 1]
    norm = float(max(w, h))
    pts_xy = np.column_stack([x / norm, y / norm])

    line = LineString(pts_xy)
    simple = line.simplify(simplify_epsilon, preserve_topology=True)

    coords = list(simple.coords)
    poly = Polygon(coords).buffer(0)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("Tracing produced invalid polygon. Try different simplify_epsilon or extraction mode.")

    svg_path_d = _svg_from_coords(coords)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">
  <g transform="translate(0,1) scale(1,-1)">
    <path d="{svg_path_d}" fill="black"/>
  </g>
</svg>
'''
    out = Path(svg_out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")

    return TraceResult(
        polygon=poly,
        svg_path=svg_path_d,
        svg_file=str(out),
        extraction_mode=detected_mode,
        extraction_warning=warning,
    )
