"""Extract regular grid line geometry from connected lattice line-art."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LatticeGeometry:
    x_lines: list[float]
    y_lines: list[float]
    bounds: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y in pixels

    @property
    def cols(self) -> int:
        return max(0, len(self.x_lines) - 1)

    @property
    def rows(self) -> int:
        return max(0, len(self.y_lines) - 1)


class LatticeDetectionError(ValueError):
    """Raised when a regular grid cannot be extracted from the foreground mask."""


def _peak_positions(proj: np.ndarray, min_peaks: int, peak_frac: float = 0.3) -> list[float]:
    if proj.size == 0 or proj.max() <= 0:
        return []
    thresh = float(proj.max()) * peak_frac
    hits = np.where(proj >= thresh)[0]
    if len(hits) == 0:
        return []

    clusters: list[list[int]] = [[int(hits[0])]]
    for x in hits[1:]:
        if x - clusters[-1][-1] <= 3:
            clusters[-1].append(int(x))
        else:
            clusters.append([int(x)])

    centers = [float(np.mean(c)) for c in clusters]
    if len(centers) < min_peaks:
        return centers
    return centers


def _spacing_is_regular(positions: list[float], max_cv: float = 0.2) -> bool:
    if len(positions) < 2:
        return False
    gaps = np.diff(positions)
    if np.any(gaps <= 0):
        return False
    mean = float(gaps.mean())
    if mean <= 0:
        return False
    cv = float(gaps.std() / mean)
    return cv <= max_cv


def extract_lattice_from_mask(binary: np.ndarray) -> LatticeGeometry:
    """
    Detect vertical and horizontal grid line centers from a boolean foreground mask.

    Expects connected black grid line art (e.g. tic-tac-toe / brownie dividers).
    """
    if binary.ndim != 2:
        raise LatticeDetectionError("Expected a 2D foreground mask.")

    col_proj = binary.sum(axis=0).astype(float)
    row_proj = binary.sum(axis=1).astype(float)

    x_lines = _peak_positions(col_proj, min_peaks=2)
    y_lines = _peak_positions(row_proj, min_peaks=2)

    if len(x_lines) < 2 or len(y_lines) < 2:
        raise LatticeDetectionError(
            "Could not detect a regular grid. Try Single shape mode or thicken grid lines."
        )

    if not _spacing_is_regular(x_lines) or not _spacing_is_regular(y_lines):
        raise LatticeDetectionError(
            "Grid line spacing is irregular. Try Single shape mode or use an evenly spaced grid."
        )

    bounds = (x_lines[0], y_lines[0], x_lines[-1], y_lines[-1])
    return LatticeGeometry(x_lines=x_lines, y_lines=y_lines, bounds=bounds)
