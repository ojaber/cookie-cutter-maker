from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from cutter_pipeline.image_extractor import extract_foreground_mask
from cutter_pipeline.lattice_cutter import lattice_height_mm, lattice_to_cookie_cutter_stl
from cutter_pipeline.lattice_extractor import extract_lattice_from_mask
from cutter_pipeline.stl_dispatch import generate_stl_from_trace
from cutter_pipeline.topology import classify_topology
from cutter_pipeline.trace_outline import trace_png_to_polygon


def _make_grid_mask(cols: int, rows: int, size: int = 300) -> np.ndarray:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    cell = size // max(cols, rows)
    for c in range(cols + 1):
        x = c * cell
        draw.line([(x, 0), (x, rows * cell)], fill=0, width=4)
    for r in range(rows + 1):
        y = r * cell
        draw.line([(0, y), (cols * cell, y)], fill=0, width=4)
    binary, _, _ = extract_foreground_mask(img, mode="binary", threshold=200)
    return binary


def test_extract_lattice_from_synthetic_grid() -> None:
    binary = _make_grid_mask(3, 4)
    lattice = extract_lattice_from_mask(binary)
    assert lattice.cols == 3
    assert lattice.rows == 4


def test_classify_topology_detects_lattice() -> None:
    binary = _make_grid_mask(2, 2)
    assert classify_topology(binary) == "lattice"


def test_trace_grid_asset_auto(tmp_path: Path) -> None:
    png = Path(__file__).parent / "assets" / "grid_3x4.png"
    if not png.exists():
        return
    svg = tmp_path / "grid.svg"
    traced = trace_png_to_polygon(str(png), str(svg), topology="auto", smooth_radius=0.0)
    assert traced.topology == "lattice"
    assert traced.cols == 3
    assert traced.rows == 4
    assert svg.exists()


def test_lattice_svg_has_correct_line_orientations(tmp_path: Path) -> None:
    """Vertical image grid lines must render as vertical SVG strokes (not swapped)."""
    png = Path(__file__).parent / "assets" / "grid_3x4.png"
    if not png.exists():
        return
    svg_path = tmp_path / "grid.svg"
    trace_png_to_polygon(str(png), str(svg_path), topology="lattice", smooth_radius=0.0)
    svg = svg_path.read_text(encoding="utf-8")

    import re

    lines = re.findall(
        r'<line x1="([0-9.+-]+)" y1="([0-9.+-]+)" x2="([0-9.+-]+)" y2="([0-9.+-]+)"',
        svg,
    )
    assert len(lines) == 9  # 4 vertical + 5 horizontal image lines

    vertical = [ln for ln in lines if abs(float(ln[0]) - float(ln[2])) < 1e-4]
    horizontal = [ln for ln in lines if abs(float(ln[1]) - float(ln[3])) < 1e-4]
    assert len(vertical) == 4
    assert len(horizontal) == 5

    for x1, y1, x2, y2 in lines:
        for v in (x1, y1, x2, y2):
            assert 0.0 <= float(v) <= 1.0, f"SVG coordinate out of viewBox: {v}"


def test_lattice_stl_from_grid_asset(tmp_path: Path) -> None:
    png = Path(__file__).parent / "assets" / "grid_3x4.png"
    if not png.exists():
        return
    traced = trace_png_to_polygon(str(png), str(tmp_path / "grid.svg"), topology="lattice", smooth_radius=0.0)
    stl_path = tmp_path / "grid.stl"
    meta = generate_stl_from_trace(traced, str(stl_path), target_width_mm=95.0)
    assert stl_path.exists()
    mesh = trimesh.load(stl_path, force="mesh")
    assert len(mesh.vertices) > 50
    assert meta["height_mm"] is not None
    assert meta["height_mm"] > 95.0


def test_lattice_height_scales_with_aspect_ratio() -> None:
    binary = _make_grid_mask(3, 4)
    lattice = extract_lattice_from_mask(binary)
    height = lattice_height_mm(lattice, 95.0)
    assert height > 95.0
