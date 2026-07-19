"""Tests for the Grid Builder (manual grid spec → lattice → STL)."""
from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from cutter_pipeline.grid_spec import (
    MAX_GRID_CELLS,
    build_grid_trace,
    grid_size_mm,
    lattice_from_spec,
)
from cutter_pipeline.stl_dispatch import generate_stl_from_trace
from cutter_pipeline.trace_meta import trace_result_from_dict, trace_result_to_dict


def test_lattice_from_spec_geometry() -> None:
    lattice = lattice_from_spec(4, 3, 30.0, 25.0)
    assert lattice.cols == 4
    assert lattice.rows == 3
    assert lattice.x_lines == [0.0, 30.0, 60.0, 90.0, 120.0]
    assert lattice.y_lines == [0.0, 25.0, 50.0, 75.0]
    assert lattice.bounds == (0.0, 0.0, 120.0, 75.0)


def test_lattice_from_spec_square_default() -> None:
    lattice = lattice_from_spec(2, 2, 40.0)
    assert lattice.y_lines == [0.0, 40.0, 80.0]


def test_lattice_from_spec_margin_only_shifts() -> None:
    plain = lattice_from_spec(3, 2, 20.0)
    padded = lattice_from_spec(3, 2, 20.0, margin=5.0)
    for a, b in zip(plain.x_lines, padded.x_lines):
        assert b - a == pytest.approx(5.0)
    # Bounds width/height are unchanged, so the STL scale is unaffected.
    assert padded.bounds[2] - padded.bounds[0] == pytest.approx(60.0)


@pytest.mark.parametrize(
    "cols,rows,cw,ch",
    [
        (0, 3, 30.0, 30.0),
        (3, 0, 30.0, 30.0),
        (MAX_GRID_CELLS + 1, 3, 30.0, 30.0),
        (3, 3, 2.0, 30.0),   # cell too small
        (3, 3, 30.0, 500.0), # cell too large
        (30, 3, 30.0, 30.0), # footprint 900mm > 600mm cap
    ],
)
def test_lattice_from_spec_rejects_bad_input(cols, rows, cw, ch) -> None:
    with pytest.raises(ValueError):
        lattice_from_spec(cols, rows, cw, ch)


def test_build_grid_trace_writes_svg_and_spec(tmp_path: Path) -> None:
    svg = tmp_path / "grid.svg"
    traced = build_grid_trace(3, 4, 30.0, None, str(svg))
    assert svg.exists()
    assert traced.topology == "lattice"
    assert traced.cols == 3
    assert traced.rows == 4
    assert traced.extraction_mode == "grid"
    assert traced.grid_spec == {
        "cols": 3,
        "rows": 4,
        "cell_w_mm": 30.0,
        "cell_h_mm": 30.0,
    }
    assert grid_size_mm(traced.grid_spec) == (90.0, 120.0)
    # SVG contains the expected number of grid lines.
    text = svg.read_text(encoding="utf-8")
    assert text.count("<line") == 4 + 5


def test_grid_spec_round_trips_through_trace_meta(tmp_path: Path) -> None:
    traced = build_grid_trace(2, 3, 25.0, 20.0, str(tmp_path / "g.svg"))
    restored = trace_result_from_dict(trace_result_to_dict(traced))
    assert restored.grid_spec == traced.grid_spec
    assert restored.lattice is not None
    assert restored.lattice.x_lines == traced.lattice.x_lines


def test_grid_spec_stl_has_exact_footprint(tmp_path: Path) -> None:
    """Cell pitch is wall-centerline to centerline; the outer flange buffers
    the centerline rectangle by flange_out, so the printed footprint is
    grid + 2*flange_out (flange_out > wall/2 here)."""
    wall = 1.4
    flange_out = 2.5
    traced = build_grid_trace(4, 3, 30.0, 25.0, str(tmp_path / "g.svg"))
    stl_path = tmp_path / "g.stl"
    meta = generate_stl_from_trace(
        traced,
        str(stl_path),
        target_width_mm=grid_size_mm(traced.grid_spec)[0],
        wall_mm=wall,
        total_h_mm=15.0,
        flange_h_mm=3.5,
        flange_out_mm=flange_out,
        flange_corner_radius_mm=0.0,
    )
    assert meta["height_mm"] == pytest.approx(75.0)
    mesh = trimesh.load(stl_path, force="mesh")
    extent_x = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
    extent_y = mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()
    assert extent_x == pytest.approx(120.0 + 2 * flange_out, abs=0.2)
    assert extent_y == pytest.approx(75.0 + 2 * flange_out, abs=0.2)
    # Internal walls land exactly on the requested 30mm cell pitch.
    xs = mesh.vertices[:, 0]
    for line_x in (30.0, 60.0, 90.0):
        near = xs[abs(xs - line_x) < wall]
        assert len(near) > 0, f"No wall geometry near x={line_x}"
