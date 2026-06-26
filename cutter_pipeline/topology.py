"""Classify outline topology for single-shape vs connected lattice grids."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage
from skimage import measure

from cutter_pipeline.lattice_extractor import LatticeDetectionError, extract_lattice_from_mask

Topology = Literal["single", "lattice"]
TopologyMode = Literal["auto", "single", "lattice"]


def _contour_bbox_area(contour: np.ndarray) -> float:
    rows, cols = contour[:, 0], contour[:, 1]
    return float((rows.max() - rows.min()) * (cols.max() - cols.min()))


def lattice_signals(binary: np.ndarray) -> dict:
    """Return debug stats used for lattice auto-detection."""
    labeled, n_components = ndimage.label(binary)
    contours = measure.find_contours(binary.astype(float), 0.5)
    bbox_areas = [_contour_bbox_area(c) for c in contours]

    similar_loops = False
    area_cv = None
    if len(bbox_areas) >= 4:
        mean_area = float(np.mean(bbox_areas))
        if mean_area > 0:
            area_cv = float(np.std(bbox_areas) / mean_area)
            similar_loops = area_cv < 0.3

    lattice_ok = False
    lattice_cols = None
    lattice_rows = None
    if n_components == 1 and similar_loops:
        try:
            lattice = extract_lattice_from_mask(binary)
            lattice_ok = lattice.cols >= 2 and lattice.rows >= 2
            lattice_cols = lattice.cols
            lattice_rows = lattice.rows
        except LatticeDetectionError:
            lattice_ok = False

    return {
        "n_components": int(n_components),
        "contour_count": len(contours),
        "area_cv": area_cv,
        "similar_loops": similar_loops,
        "lattice_ok": lattice_ok,
        "lattice_cols": lattice_cols,
        "lattice_rows": lattice_rows,
    }


def classify_topology(binary: np.ndarray) -> Topology:
    return "lattice" if lattice_signals(binary)["lattice_ok"] else "single"


def resolve_topology(requested: TopologyMode, binary: np.ndarray) -> Topology:
    if requested == "single":
        return "single"
    if requested == "lattice":
        return "lattice"
    return classify_topology(binary)


def grid_hint(binary: np.ndarray, resolved: Topology) -> str | None:
    """Suggest lattice mode when auto picked single but signals are strong."""
    if resolved != "single":
        return None
    stats = lattice_signals(binary)
    # Suggest grid mode if: single connected component AND multiple contours
    # Even if lattice extraction failed, the pattern suggests a grid
    if stats["n_components"] == 1 and stats["contour_count"] >= 4:
        cols = stats["lattice_cols"]
        rows = stats["lattice_rows"]
        if cols and rows:
            return f"Detected a {cols}×{rows} cell grid — try Grid / lattice mode."
        if stats["similar_loops"]:
            return "Detected similar cell loops — try Grid / lattice mode."
        return "Multiple contours detected — try Grid / lattice mode if this is a grid."
    return None
