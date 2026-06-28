"""Tests for STL extraction functionality."""

import tempfile
from pathlib import Path

import numpy as np
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
