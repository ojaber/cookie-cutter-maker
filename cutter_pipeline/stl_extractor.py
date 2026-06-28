"""Extract 2D outlines from STL files for cookie cutter generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage import measure

from cutter_pipeline.lattice_extractor import LatticeDetectionError, extract_lattice_from_mask
from cutter_pipeline.topology import grid_hint
from cutter_pipeline.trace_outline import TopologyParam, TraceResult, _trace_single, _write_lattice_svg


def _polygon_svg_path(poly: Polygon) -> str:
    parts: list[str] = []
    ext = list(poly.exterior.coords)
    if ext:
        parts.append(f"M {ext[0][0]:.6f},{ext[0][1]:.6f} ")
        for x, y in ext[1:]:
            parts.append(f"L {x:.6f},{y:.6f} ")
        parts.append("Z ")
    for interior in poly.interiors:
        coords = list(interior.coords)
        if not coords:
            continue
        parts.append(f"M {coords[0][0]:.6f},{coords[0][1]:.6f} ")
        for x, y in coords[1:]:
            parts.append(f"L {x:.6f},{y:.6f} ")
        parts.append("Z ")
    return "".join(parts).strip()


def _write_projected_svg(geom: Polygon | MultiPolygon, svg_out_path: Path) -> str:
    polygons = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    d = " ".join(_polygon_svg_path(poly) for poly in polygons if not poly.is_empty)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">
  <g transform="translate(0,1) scale(1,-1)">
    <path d="{d}" fill="black" fill-rule="evenodd"/>
  </g>
</svg>
'''
    svg_out_path.parent.mkdir(parents=True, exist_ok=True)
    svg_out_path.write_text(svg, encoding="utf-8")
    return d


def _project_mesh(mesh: trimesh.Trimesh, raster_size: int = 1024) -> tuple[Polygon | MultiPolygon, np.ndarray]:
    vertices_2d = np.asarray(mesh.vertices[:, :2], dtype=float)
    min_coords = vertices_2d.min(axis=0)
    max_coords = vertices_2d.max(axis=0)
    range_coords = max_coords - min_coords
    norm = float(max(range_coords[0], range_coords[1]))
    if norm <= 0:
        raise ValueError("STL has no width or height in XY plane.")

    normalized = (vertices_2d - min_coords) / norm
    raster_range = np.where(range_coords > 0, range_coords, 1.0)
    raster_xy = (vertices_2d - min_coords) / raster_range
    face_polys: list[Polygon] = []
    for face in np.asarray(mesh.faces):
        coords = normalized[face]
        tri = Polygon(coords).buffer(0)
        if tri.is_empty or tri.area <= 0:
            continue
        face_polys.append(tri)
    if not face_polys:
        raise ValueError("Could not extract a valid outline from the STL.")

    geom = unary_union(face_polys).buffer(0)
    if geom.is_empty:
        raise ValueError("Could not extract a valid outline from the STL.")

    scale = float(raster_size - 3)
    width = max(8, int(np.ceil(range_coords[0] / norm * scale)) + 3)
    height = max(8, int(np.ceil(range_coords[1] / norm * scale)) + 3)
    px = raster_xy[:, 0] * float(width - 3) + 1
    py = (1.0 - raster_xy[:, 1]) * float(height - 3) + 1
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for face in np.asarray(mesh.faces):
        coords = [(float(px[i]), float(py[i])) for i in face]
        draw.polygon(coords, fill=255)
    binary = np.array(image) > 0
    return geom, binary


def extract_outline_from_stl(
    stl_path: str,
    svg_out_path: str,
    simplify_epsilon: float = 0.002,
    topology: TopologyParam = "auto",
) -> TraceResult:
    mesh = trimesh.load(stl_path, force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("STL file is empty or invalid.")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError("STL file has no faces.")

    projected_geom, binary = _project_mesh(mesh)
    contours = measure.find_contours(binary.astype(float), 0.5)
    if not contours:
        raise ValueError("Could not extract a valid outline from the STL.")

    h, w = binary.shape
    lattice = None
    try:
        lattice = extract_lattice_from_mask(binary)
    except LatticeDetectionError:
        lattice = None

    detected = "lattice" if topology == "auto" and lattice is not None else ("single" if topology == "auto" else None)
    resolved = topology if topology in ("single", "lattice") else ("lattice" if lattice is not None else "single")
    hint = grid_hint(binary, resolved) if topology == "auto" else None
    svg_path = Path(svg_out_path)

    if resolved == "lattice":
        if lattice is None:
            try:
                lattice = extract_lattice_from_mask(binary)
            except LatticeDetectionError as exc:
                if topology == "lattice":
                    raise ValueError(str(exc)) from exc
                resolved = "single"
        if lattice is not None and resolved == "lattice":
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
                extraction_mode="stl",
                extraction_warning="",
            )

    if isinstance(projected_geom, MultiPolygon):
        if len(projected_geom.geoms) == 1:
            poly = projected_geom.geoms[0]
        else:
            poly = max(projected_geom.geoms, key=lambda g: g.area)
    else:
        poly = projected_geom
    poly = poly.simplify(simplify_epsilon, preserve_topology=True).buffer(0)
    if poly.is_empty or not poly.is_valid:
        poly, _ = _trace_single(binary, w, h, simplify_epsilon, svg_path)
        svg_path_d = _write_projected_svg(poly, svg_path)
    else:
        svg_path_d = _write_projected_svg(poly, svg_path)

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
        extraction_mode="stl",
        extraction_warning="",
    )
