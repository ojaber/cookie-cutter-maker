"""Tests for STL extraction functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from cutter_pipeline.stl_extractor import extract_outline_from_stl


def test_extract_outline_from_stl_simple():
    """Test STL extraction with a simple cube."""
    # Create a simple cube STL
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        cube.export(tmp.name)
        tmp_path = tmp.name
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as svg_tmp:
            svg_path = svg_tmp.name
        
        result = extract_outline_from_stl(tmp_path, svg_path, simplify_epsilon=0.002)
        
        # Verify TraceResult structure
        assert result.polygon is not None
        assert result.lattice is None
        assert result.topology == "single"
        assert result.extraction_mode == "stl"
        assert result.extraction_warning == ""
        
        # Verify polygon is valid
        assert isinstance(result.polygon, Polygon)
        assert result.polygon.is_valid
        assert not result.polygon.is_empty
        
        # Verify SVG was created
        assert Path(svg_path).exists()
        svg_content = Path(svg_path).read_text()
        assert "<svg" in svg_content
        assert "<path" in svg_content
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(svg_path).unlink(missing_ok=True)


def test_extract_outline_from_stl_lattice_grid():
    x_positions = [0.0, 1.0, 2.0]
    y_positions = [0.0, 1.0, 2.0, 3.0]
    bars = []
    for x in x_positions:
        bar = trimesh.creation.box(extents=[0.12, 3.0, 1.0])
        bar.apply_translation([x, 1.5, 0.5])
        bars.append(bar)
    for y in y_positions:
        bar = trimesh.creation.box(extents=[2.0, 0.12, 1.0])
        bar.apply_translation([1.0, y, 0.5])
        bars.append(bar)
    grid = trimesh.util.concatenate(bars)

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        grid.export(tmp.name)
        tmp_path = tmp.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as svg_tmp:
            svg_path = svg_tmp.name

        result = extract_outline_from_stl(tmp_path, svg_path, simplify_epsilon=0.002, topology="auto")

        assert result.topology == "lattice"
        assert result.lattice is not None
        assert result.cols == 2
        assert result.rows == 3
        assert result.polygon is None
        assert Path(svg_path).exists()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(svg_path).unlink(missing_ok=True)


def test_extract_outline_from_stl_multiple_components():
    """Test STL extraction with multiple disconnected components."""
    # Create two separate boxes
    box1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box1.apply_translation([1.5, 0, 0])
    box2 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box2.apply_translation([-1.5, 0, 0])
    
    # Combine them
    combined = trimesh.util.concatenate([box1, box2])
    
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        combined.export(tmp.name)
        tmp_path = tmp.name
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as svg_tmp:
            svg_path = svg_tmp.name
        
        result = extract_outline_from_stl(tmp_path, svg_path, simplify_epsilon=0.002)
        
        # Verify TraceResult structure
        assert result.polygon is not None
        assert result.topology == "single"
        
        # Verify polygon is valid (should be union of both components)
        assert isinstance(result.polygon, Polygon)
        assert result.polygon.is_valid
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(svg_path).unlink(missing_ok=True)


def test_extract_outline_from_stl_empty_mesh():
    """Test that an empty STL file raises an error."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        Path(tmp.name).write_bytes(b"")
        tmp_path = tmp.name
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as svg_tmp:
            svg_path = svg_tmp.name
        
        try:
            extract_outline_from_stl(tmp_path, svg_path)
            assert False, "Should have raised ValueError for empty STL file"
        except ValueError as e:
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(svg_path).unlink(missing_ok=True)


def test_extract_outline_from_stl_no_faces():
    """Test that malformed STL content raises an error."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        Path(tmp.name).write_text("solid bad\nendsolid bad\n", encoding="utf-8")
        tmp_path = tmp.name
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as svg_tmp:
            svg_path = svg_tmp.name
        
        try:
            extract_outline_from_stl(tmp_path, svg_path)
            assert False, "Should have raised ValueError for malformed STL"
        except ValueError as e:
            assert "empty" in str(e).lower() or "invalid" in str(e).lower() or "no faces" in str(e).lower()
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(svg_path).unlink(missing_ok=True)


# ── Models made of side walls only ────────────────────────────────────────────
# Many cookie-cutter STLs are an outline extruded as walls, with no top or
# bottom faces. Seen from above every face is edge-on, so a footprint built
# from face areas is empty; the outline has to come from the silhouette.

_L_SHAPE = Polygon([(0, 0), (40, 0), (40, 10), (10, 10), (10, 30), (0, 30)])


def _walls_only(poly: Polygon, height: float = 10.0) -> trimesh.Trimesh:
    """Extrude ``poly`` and keep only the side faces."""
    solid = trimesh.creation.extrude_polygon(poly, height)
    sides = np.abs(solid.face_normals[:, 2]) < 0.01
    return solid.submesh([sides], append=True)


def _trace(mesh: trimesh.Trimesh, tmp_path: Path):
    stl = tmp_path / "m.stl"
    mesh.export(stl)
    return extract_outline_from_stl(str(stl), str(tmp_path / "m.svg"))


def test_walls_only_model_traces_its_silhouette(tmp_path):
    walls = _walls_only(_L_SHAPE)
    assert not (np.abs(walls.face_normals[:, 2]) > 0.01).any()
    result = _trace(walls, tmp_path)
    capped = _trace(trimesh.creation.extrude_polygon(_L_SHAPE, 10.0), tmp_path)
    # Same outline, same place, same way round — not mirrored, not a sliver.
    overlap = result.polygon.intersection(capped.polygon).area / capped.polygon.union(result.polygon).area
    assert overlap > 0.97, overlap


def test_bundled_example_models_trace(tmp_path):
    examples = Path(__file__).resolve().parent.parent / "examples"
    for name in ("dino", "911", "racecar", "drill"):
        mesh = trimesh.load(examples / f"{name}.stl", force="mesh")
        result = extract_outline_from_stl(str(examples / f"{name}.stl"), str(tmp_path / f"{name}.svg"))
        minx, miny, maxx, maxy = result.polygon.bounds
        expected = mesh.extents[1] / mesh.extents[0]
        assert (maxy - miny) / (maxx - minx) == pytest.approx(expected, rel=0.03), name
        # A real silhouette, not the biggest of a few slivers.
        assert result.polygon.area > 0.2, name


def test_capped_model_with_a_big_hole_keeps_it(tmp_path):
    """A washer covers well under half of its filled outline. Its exact face
    footprint is still right and must win over the hole-filled silhouette."""
    washer = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)], [[(4, 4), (36, 4), (36, 36), (4, 36)]])
    assert washer.area < 0.5 * Polygon(washer.exterior).area
    result = _trace(trimesh.creation.extrude_polygon(washer, 10.0), tmp_path)
    assert len(result.polygon.interiors) == 1
    assert result.polygon.area == pytest.approx(washer.area / 40 ** 2, rel=0.05)
