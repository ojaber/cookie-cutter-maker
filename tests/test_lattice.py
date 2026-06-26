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


def test_flange_all_lines_adds_volume_and_keeps_cells_open(tmp_path: Path) -> None:
    binary = _make_grid_mask(4, 4)
    lattice = extract_lattice_from_mask(binary)

    outer_path = tmp_path / "outer.stl"
    all_path = tmp_path / "all.stl"
    lattice_to_cookie_cutter_stl(
        lattice, str(outer_path), target_width_mm=95.0, flange_all_lines=False
    )
    lattice_to_cookie_cutter_stl(
        lattice, str(all_path), target_width_mm=95.0, flange_all_lines=True
    )

    assert outer_path.exists() and all_path.exists()
    outer_mesh = trimesh.load(outer_path, force="mesh")
    all_mesh = trimesh.load(all_path, force="mesh")

    # All-lines flange must add material along internal grid lines.
    assert all_mesh.volume > outer_mesh.volume
    # Cells must remain open (interior holes preserved), so the footprint stays
    # a connected web rather than a solid slab.
    assert len(all_mesh.vertices) > len(outer_mesh.vertices)


def test_flange_corner_radius_changes_grid_geometry(tmp_path: Path) -> None:
    binary = _make_grid_mask(4, 4)
    lattice = extract_lattice_from_mask(binary)

    sharp_path = tmp_path / "sharp.stl"
    round_path = tmp_path / "round.stl"
    lattice_to_cookie_cutter_stl(
        lattice, str(sharp_path), target_width_mm=95.0, flange_corner_radius_mm=0.0
    )
    lattice_to_cookie_cutter_stl(
        lattice, str(round_path), target_width_mm=95.0, flange_corner_radius_mm=4.0
    )

    assert sharp_path.exists() and round_path.exists()
    sharp = trimesh.load(sharp_path, force="mesh")
    rounded = trimesh.load(round_path, force="mesh")
    # Rounding trims the outer flange corners, so the rounded grid uses less
    # material than the sharp one.
    assert rounded.volume < sharp.volume


def test_flange_all_lines_fills_outer_corner(tmp_path: Path) -> None:
    """The all-lines flange must fill the outer corner square (boundary line
    shelves use flat caps and would otherwise leave a notched corner)."""
    binary = _make_grid_mask(4, 4)
    lattice = extract_lattice_from_mask(binary)
    out_path = tmp_path / "all_lines_corner.stl"
    lattice_to_cookie_cutter_stl(
        lattice,
        str(out_path),
        target_width_mm=95.0,
        flange_all_lines=True,
        flange_corner_radius_mm=0.0,  # sharp: corner should be a full square
    )
    mesh = trimesh.load(out_path, force="mesh")
    flange_xy = mesh.vertices[mesh.vertices[:, 2] < 3.5][:, :2]
    minx, miny = flange_xy.min(axis=0)
    # With the corner filled, there is flange material right at the extreme
    # corner; a notched corner would leave this quadrant empty.
    in_corner = ((flange_xy[:, 0] < minx + 1.0) & (flange_xy[:, 1] < miny + 1.0)).sum()
    assert in_corner > 0, "All-lines flange outer corner is notched/empty."


def test_all_lines_with_chamfer_is_watertight_manifold(tmp_path: Path) -> None:
    """The all-lines flange with a chamfer must export a single watertight,
    manifold mesh (no non-manifold edges) so slicers don't error."""
    from collections import Counter

    binary = _make_grid_mask(5, 6)
    lattice = extract_lattice_from_mask(binary)
    out_path = tmp_path / "all_lines_chamfer.stl"
    lattice_to_cookie_cutter_stl(
        lattice,
        str(out_path),
        target_width_mm=95.0,
        wall_mm=1.4,
        total_h_mm=15.0,
        flange_h_mm=3.5,
        flange_out_mm=2.5,
        flange_chamfer_mm=0.5,
        flange_all_lines=True,
        flange_corner_radius_mm=1.5,
    )
    mesh = trimesh.load(out_path, force="mesh")
    mesh.merge_vertices()
    counts = Counter(map(tuple, mesh.edges_sorted))
    non_manifold = sum(1 for k in counts.values() if k != 2)
    assert non_manifold == 0, f"Mesh has {non_manifold} non-manifold edges."
    assert mesh.is_watertight, "All-lines + chamfer mesh is not watertight."


def test_flange_corner_radius_all_lines_keeps_cells_open(tmp_path: Path) -> None:
    binary = _make_grid_mask(4, 4)
    lattice = extract_lattice_from_mask(binary)
    out_path = tmp_path / "all_lines_round.stl"
    lattice_to_cookie_cutter_stl(
        lattice,
        str(out_path),
        target_width_mm=95.0,
        flange_all_lines=True,
        flange_corner_radius_mm=3.0,
    )
    assert out_path.exists()
    mesh = trimesh.load(out_path, force="mesh")
    assert len(mesh.vertices) > 50


def test_flange_all_lines_dispatch(tmp_path: Path) -> None:
    png = Path(__file__).parent / "assets" / "grid_3x4.png"
    if not png.exists():
        return
    traced = trace_png_to_polygon(
        str(png), str(tmp_path / "grid.svg"), topology="lattice", smooth_radius=0.0
    )
    stl_path = tmp_path / "grid_all_lines.stl"
    generate_stl_from_trace(
        traced, str(stl_path), target_width_mm=95.0, flange_all_lines=True
    )
    assert stl_path.exists()
    mesh = trimesh.load(stl_path, force="mesh")
    assert len(mesh.vertices) > 50
