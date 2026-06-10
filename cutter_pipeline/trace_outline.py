from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter
from shapely.geometry import LineString, Polygon
from skimage import measure

from cutter_pipeline.image_extractor import ImageMode, extract_foreground_mask
from cutter_pipeline.lattice_extractor import LatticeDetectionError, extract_lattice_from_mask
from cutter_pipeline.topology import Topology, TopologyMode, classify_topology, grid_hint, resolve_topology

TopologyParam = Literal["auto", "single", "lattice"]


@dataclass
class TraceResult:
    polygon: Polygon | None
    lattice: "LatticeGeometry | None"
    topology: Topology
    topology_requested: TopologyParam
    topology_detected: Topology | None
    contour_count: int
    cols: int | None
    rows: int | None
    grid_hint: str | None
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


def _contour_to_polygon(
    contour: np.ndarray,
    w: int,
    h: int,
    simplify_epsilon: float,
) -> Polygon:
    pts = np.array(contour)
    y = h - pts[:, 0]
    x = w - pts[:, 1]
    norm = float(max(w, h))
    pts_xy = np.column_stack([x / norm, y / norm])
    line = LineString(pts_xy)
    simple = line.simplify(simplify_epsilon, preserve_topology=True)
    coords = list(simple.coords)
    poly = Polygon(coords).buffer(0)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("Tracing produced invalid polygon.")
    return poly


def _write_single_svg(coords: list[tuple[float, float]], svg_out_path: Path) -> str:
    svg_path_d = _svg_from_coords(coords)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">
  <g transform="translate(0,1) scale(1,-1)">
    <path d="{svg_path_d}" fill="black"/>
  </g>
</svg>
'''
    svg_out_path.parent.mkdir(parents=True, exist_ok=True)
    svg_out_path.write_text(svg, encoding="utf-8")
    return svg_path_d


def _write_lattice_svg(lattice, w: int, h: int, svg_out_path: Path) -> str:
    norm = float(max(w, h))

    def norm_pt(x: float, y: float) -> tuple[float, float]:
        # Match single-shape mirror/orientation conventions.
        nx = (w - y) / norm
        ny = (h - x) / norm
        return nx, ny

    min_x, min_y, max_x, max_y = lattice.bounds
    x0, y0 = norm_pt(min_x, min_y)
    x1, y1 = norm_pt(max_x, min_y)
    x2, y2 = norm_pt(max_x, max_y)
    x3, y3 = norm_pt(min_x, max_y)

    lines: list[str] = []
    for x in lattice.x_lines:
        ax, ay = norm_pt(x, lattice.y_lines[0])
        bx, by = norm_pt(x, lattice.y_lines[-1])
        lines.append(
            f'<line x1="{ax:.6f}" y1="{ay:.6f}" x2="{bx:.6f}" y2="{by:.6f}" '
            f'stroke="#1f6feb" stroke-width="0.004"/>'
        )
    for y in lattice.y_lines:
        ax, ay = norm_pt(lattice.x_lines[0], y)
        bx, by = norm_pt(lattice.x_lines[-1], y)
        lines.append(
            f'<line x1="{ax:.6f}" y1="{ay:.6f}" x2="{bx:.6f}" y2="{by:.6f}" '
            f'stroke="#7ee787" stroke-width="0.004"/>'
        )

    frame = (
        f'<polygon points="{x0:.6f},{y0:.6f} {x1:.6f},{y1:.6f} {x2:.6f},{y2:.6f} {x3:.6f},{y3:.6f}" '
        f'fill="none" stroke="#e6edf3" stroke-width="0.005"/>'
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">
  <rect width="1" height="1" fill="white"/>
  <g transform="translate(0,1) scale(1,-1)">
    {frame}
    {chr(10).join(lines)}
  </g>
</svg>
'''
    svg_out_path.parent.mkdir(parents=True, exist_ok=True)
    svg_out_path.write_text(svg, encoding="utf-8")
    return f"lattice:{lattice.cols}x{lattice.rows}"


def _trace_single(
    binary: np.ndarray,
    w: int,
    h: int,
    simplify_epsilon: float,
    svg_out_path: Path,
) -> tuple[Polygon, str]:
    contours = measure.find_contours(binary.astype(float), 0.5)
    if not contours:
        raise ValueError("No contours found. Try adjusting threshold or extraction mode.")
    contour = max(contours, key=lambda c: c.shape[0])
    poly = _contour_to_polygon(contour, w, h, simplify_epsilon)
    coords = list(poly.exterior.coords)
    svg_path_d = _write_single_svg(coords, svg_out_path)
    return poly, svg_path_d


def trace_png(
    png_path: str,
    svg_out_path: str,
    threshold: int = 200,
    simplify_epsilon: float = 0.002,
    smooth_radius: float = 0.0,
    extraction_mode: Literal["auto", "binary", "dashed", "simple_bg", "complex"] = "auto",
    delta_e_threshold: float = 28.0,
    topology: TopologyParam = "auto",
) -> TraceResult:
    img = Image.open(png_path)
    if smooth_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=smooth_radius))

    binary, detected_mode, warning = extract_foreground_mask(
        img,
        mode=extraction_mode,
        threshold=threshold,
        delta_e_threshold=delta_e_threshold,
    )

    arr = np.array(img.convert("L"))
    h, w = arr.shape
    contours = measure.find_contours(binary.astype(float), 0.5)
    if not contours:
        raise ValueError("No contours found. Try adjusting threshold or extraction mode.")

    detected: Topology | None = classify_topology(binary) if topology == "auto" else None
    resolved = resolve_topology(topology, binary)
    hint = grid_hint(binary, resolved) if topology == "auto" else None
    svg_path = Path(svg_out_path)

    if resolved == "lattice":
        try:
            lattice = extract_lattice_from_mask(binary)
        except LatticeDetectionError as exc:
            if topology == "lattice":
                raise ValueError(str(exc)) from exc
            resolved = "single"
        else:
            svg_path_d = _write_lattice_svg(lattice, w, h, svg_path)
            return TraceResult(
                polygon=None,
                lattice=lattice,
                topology="lattice",
                topology_requested=topology,
                topology_detected=detected,
                contour_count=len(contours),
                cols=lattice.cols,
                rows=lattice.rows,
                grid_hint=hint,
                svg_path=svg_path_d,
                svg_file=str(svg_path),
                extraction_mode=detected_mode,
                extraction_warning=warning,
            )

    poly, svg_path_d = _trace_single(binary, w, h, simplify_epsilon, svg_path)
    return TraceResult(
        polygon=poly,
        lattice=None,
        topology="single",
        topology_requested=topology,
        topology_detected=detected,
        contour_count=len(contours),
        cols=None,
        rows=None,
        grid_hint=hint,
        svg_path=svg_path_d,
        svg_file=str(svg_path),
        extraction_mode=detected_mode,
        extraction_warning=warning,
    )


def trace_png_to_polygon(
    png_path: str,
    svg_out_path: str,
    threshold: int = 200,
    simplify_epsilon: float = 0.002,
    smooth_radius: float = 0.0,
    extraction_mode: Literal["auto", "binary", "dashed", "simple_bg", "complex"] = "auto",
    delta_e_threshold: float = 28.0,
    topology: TopologyParam = "auto",
) -> TraceResult:
    """Backward-compatible entry point; returns full TraceResult."""
    return trace_png(
        png_path,
        svg_out_path,
        threshold=threshold,
        simplify_epsilon=simplify_epsilon,
        smooth_radius=smooth_radius,
        extraction_mode=extraction_mode,
        delta_e_threshold=delta_e_threshold,
        topology=topology,
    )
