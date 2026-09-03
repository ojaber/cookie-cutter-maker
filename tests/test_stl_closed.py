"""The exported cutter must be a closed surface with no holes in it.

The body used to be built as a bare tube: both ends were left open, which meant
`_union_solids` rejected it ("Not all meshes are volumes"), silently fell back
to concatenating the wall, flange and chamfer as three overlapping shells, and
exported an STL you could see straight through. In the browser preview, which
draws front faces only, the gap at the cutting edge showed up as the page
background painted along the top of the walls.

None of the existing tests noticed, because every one of them measures
dimensions and the dimensions were right the whole time. These check topology
instead: no boundary edges anywhere, and an actual surface closing the cutting
edge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from cutter_pipeline.grid_spec import build_grid_trace
from cutter_pipeline.stl_dispatch import generate_stl_from_trace
from cutter_pipeline.trace_outline import trace_png_to_polygon

ASSETS = Path(__file__).parent / "assets"

# What the web UI posts with its default advanced settings.
UI_DEFAULTS = dict(
    target_width_mm=95.0,
    wall_mm=1.4,
    total_h_mm=15.0,
    flange_h_mm=3.5,
    flange_out_mm=2.5,
    flange_chamfer_mm=0.5,
    flange_corner_radius_mm=1.5,
    bottom_wall_mm=0.1,
    cutting_wall_h_mm=2.0,
    cleanup_mm=0.5,
    tip_smooth_mm=0.6,
)


def _open_edge_count(mesh: trimesh.Trimesh) -> int:
    """Edges with only one face on them, i.e. the rim of a hole."""
    edges = np.sort(mesh.edges_sorted, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _horizontal_faces_at_top(path: Path) -> int:
    # Loaded unprocessed so nothing is merged or repaired on the way in.
    raw = trimesh.load(str(path), process=False)
    top = float(raw.bounds[1][2])
    flat = np.abs(raw.face_normals[:, 2]) > 0.9
    return int((flat & (raw.triangles_center[:, 2] > top - 0.05)).sum())


def _cutter_from_png(png: Path, out: Path) -> Path:
    traced = trace_png_to_polygon(str(png), str(out.with_suffix(".svg")))
    generate_stl_from_trace(traced, str(out), **UI_DEFAULTS)
    return out


@pytest.mark.parametrize("shape", ["heart", "cactus", "doggy"])
def test_single_shape_cutter_has_no_holes(shape, tmp_path):
    stl = _cutter_from_png(ASSETS / f"{shape}.png", tmp_path / f"{shape}.stl")
    mesh = trimesh.load(str(stl))

    assert _open_edge_count(mesh) == 0, (
        f"{shape}: the exported mesh has boundary edges, so there is a hole in "
        "it. The body is probably reaching _union_solids open again, which "
        "makes the boolean fail and fall back to concatenating loose shells."
    )
    assert _horizontal_faces_at_top(stl) > 0, (
        f"{shape}: nothing closes the cutting edge at the top of the wall."
    )


def test_grid_cutter_has_no_holes(tmp_path):
    stl = tmp_path / "grid.stl"
    traced = build_grid_trace(
        cols=3, rows=3, cell_w_mm=30.0, cell_h_mm=30.0,
        svg_out_path=str(stl.with_suffix(".svg")),
    )
    generate_stl_from_trace(traced, str(stl), **{**UI_DEFAULTS, "target_width_mm": 90.0})

    assert _open_edge_count(trimesh.load(str(stl))) == 0
    assert _horizontal_faces_at_top(stl) > 0


def test_cutter_is_a_single_connected_body(tmp_path):
    """The wall, flange and chamfer have to end up merged rather than merely
    stacked on top of one another."""
    stl = _cutter_from_png(ASSETS / "heart.png", tmp_path / "heart.stl")
    assert trimesh.load(str(stl)).body_count == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},                                                  # tapered cutting edge
        {"bottom_wall_mm": 1.4},                              # edge as thick as the wall
        {"cutting_wall_h_mm": 0.0},                           # no taper at all
        {"flange_chamfer_mm": 0.0},                           # no chamfer brace
        {"flange_out_mm": 0.0, "flange_chamfer_mm": 0.0},     # no grip rim either
    ],
    ids=["tapered", "square-edge", "no-taper", "no-chamfer", "no-flange"],
)
def test_no_holes_across_edge_and_flange_settings(overrides, tmp_path):
    """The end caps are built on one code path and skipped on another, so walk
    both, with and without the parts that get unioned on afterwards."""
    out = tmp_path / "c.stl"
    traced = trace_png_to_polygon(str(ASSETS / "heart.png"), str(out.with_suffix(".svg")))
    generate_stl_from_trace(traced, str(out), **{**UI_DEFAULTS, **overrides})
    assert _open_edge_count(trimesh.load(str(out))) == 0
