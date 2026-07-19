"""Build lattice traces directly from a user-specified grid — no image needed.

The Grid Builder lets a user say "4 columns × 3 rows, 30 mm cells" instead of
drawing/uploading a picture of a grid just to communicate its dimensions. Line
positions are generated in exact millimetres, so the resulting cutter matches
the requested cell sizes precisely (cells are measured between wall
centerlines; each opening is one wall thickness smaller).
"""

from __future__ import annotations

from pathlib import Path

from cutter_pipeline.lattice_extractor import LatticeGeometry
from cutter_pipeline.trace_outline import TraceResult, _write_lattice_svg

MAX_GRID_CELLS = 40
MIN_CELL_MM = 5.0
MAX_CELL_MM = 300.0
MAX_GRID_MM = 600.0


def lattice_from_spec(
    cols: int,
    rows: int,
    cell_w_mm: float,
    cell_h_mm: float | None = None,
    margin: float = 0.0,
) -> LatticeGeometry:
    """Build a regular LatticeGeometry whose line positions are millimetres.

    ``margin`` shifts all lines by a constant offset (used only to pad the SVG
    preview); it cancels out of the bounds and never affects the STL.
    """
    if cell_h_mm is None:
        cell_h_mm = cell_w_mm

    if not (1 <= cols <= MAX_GRID_CELLS) or not (1 <= rows <= MAX_GRID_CELLS):
        raise ValueError(
            f"Grid must be between 1×1 and {MAX_GRID_CELLS}×{MAX_GRID_CELLS} cells."
        )
    for label, value in (("width", cell_w_mm), ("height", cell_h_mm)):
        if not (MIN_CELL_MM <= value <= MAX_CELL_MM):
            raise ValueError(
                f"Cell {label} must be between {MIN_CELL_MM:g} and {MAX_CELL_MM:g} mm."
            )
    total_w = cols * cell_w_mm
    total_h = rows * cell_h_mm
    if total_w > MAX_GRID_MM or total_h > MAX_GRID_MM:
        raise ValueError(
            f"Grid footprint {total_w:g}×{total_h:g} mm exceeds the "
            f"{MAX_GRID_MM:g} mm maximum. Use fewer or smaller cells."
        )

    x_lines = [margin + i * cell_w_mm for i in range(cols + 1)]
    y_lines = [margin + j * cell_h_mm for j in range(rows + 1)]
    bounds = (x_lines[0], y_lines[0], x_lines[-1], y_lines[-1])
    return LatticeGeometry(x_lines=x_lines, y_lines=y_lines, bounds=bounds)


def build_grid_trace(
    cols: int,
    rows: int,
    cell_w_mm: float,
    cell_h_mm: float | None,
    svg_out_path: str,
) -> TraceResult:
    """Create a TraceResult for a manually specified grid, writing an SVG preview."""
    if cell_h_mm is None:
        cell_h_mm = cell_w_mm
    total_w = cols * cell_w_mm
    total_h = rows * cell_h_mm
    # Small margin so the preview frame doesn't touch the SVG viewBox edge.
    pad = 0.04 * max(total_w, total_h, 1.0)
    lattice = lattice_from_spec(cols, rows, cell_w_mm, cell_h_mm, margin=pad)
    svg_path = Path(svg_out_path)
    svg_path_d = _write_lattice_svg(lattice, total_w + 2 * pad, total_h + 2 * pad, svg_path)
    return TraceResult(
        polygon=None,
        lattice=lattice,
        topology="lattice",
        topology_requested="lattice",
        topology_detected="lattice",
        contour_count=0,
        cols=cols,
        rows=rows,
        grid_hint=None,
        svg_path=svg_path_d,
        svg_file=str(svg_path),
        extraction_mode="grid",
        extraction_warning="",
        grid_spec={
            "cols": cols,
            "rows": rows,
            "cell_w_mm": float(cell_w_mm),
            "cell_h_mm": float(cell_h_mm),
        },
    )


def grid_size_mm(spec: dict) -> tuple[float, float]:
    """Total grid footprint (wall centerline to centerline) for a grid spec."""
    return (
        spec["cols"] * spec["cell_w_mm"],
        spec["rows"] * spec["cell_h_mm"],
    )
