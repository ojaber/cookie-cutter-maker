"""Extract 2D outlines from STL files for cookie cutter generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage import measure

from cutter_pipeline.lattice_extractor import LatticeDetectionError, extract_lattice_from_mask
from cutter_pipeline.topology import grid_hint
from cutter_pipeline.trace_outline import TopologyParam, TraceResult, _write_lattice_svg


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


class _Projection:
    """A mesh seen from above, two ways.

    ``geom`` is the union of the faces' footprints, exact but only as good as
    the faces: a model made of side walls alone (an extruded outline with no
    top or bottom, which is common for cookie-cutter models) has faces that
    are all edge-on from above and project to nothing. ``binary`` is the same
    view rasterised, where those walls still draw as lines; ``filled`` closes
    them into the solid silhouette. Raster pixels map back to the same
    normalised coordinates as ``geom`` via :meth:`to_norm`.
    """

    def __init__(self, geom, binary, filled, width, height, span_x, span_y):
        self.geom = geom
        self.binary = binary
        self.filled = filled
        self._w, self._h = width, height
        self._sx, self._sy = span_x, span_y

    def to_norm(self, contour: np.ndarray) -> np.ndarray:
        rows, cols = contour[:, 0], contour[:, 1]
        x = (cols - 1) / float(self._w - 3) * self._sx
        y = (1.0 - (rows - 1) / float(self._h - 3)) * self._sy
        return np.column_stack([x, y])


def _project_mesh(mesh: trimesh.Trimesh, raster_size: int = 1024) -> _Projection:
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
    geom = unary_union(face_polys).buffer(0) if face_polys else None
    if geom is not None and geom.is_empty:
        geom = None

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
        # A face seen edge-on has no area, and PIL may draw nothing for it.
        # Its edges still mark where the wall stands, so draw those too.
        draw.line(coords + [coords[0]], fill=255, width=1)
    binary = np.array(image) > 0
    # Thicken by a pixel first so a wall drawn as a one-pixel diagonal cannot
    # leak the fill, then thin back so the silhouette keeps its size.
    grown = ndimage.binary_dilation(binary, iterations=1)
    filled = ndimage.binary_erosion(ndimage.binary_fill_holes(grown), iterations=1) | binary
    return _Projection(
        geom, binary, filled, width, height,
        float(range_coords[0] / norm), float(range_coords[1] / norm),
    )


def _silhouette_polygon(proj: _Projection, simplify_epsilon: float) -> Polygon:
    """The outer outline of the filled raster silhouette, in ``geom`` units."""
    contours = measure.find_contours(proj.filled.astype(float), 0.5)
    if not contours:
        raise ValueError("Could not extract a valid outline from the STL.")
    contour = max(contours, key=lambda c: Polygon(c).area if len(c) >= 4 else 0.0)
    line = LineString(proj.to_norm(contour)).simplify(simplify_epsilon, preserve_topology=True)
    poly = Polygon(list(line.coords)).buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or poly.area <= 0:
        raise ValueError("Could not extract a valid outline from the STL.")
    return poly


def _lattice_is_drawn(lattice, binary: np.ndarray, min_coverage: float = 0.8) -> bool:
    """True when every detected grid line is actually present in ``binary``.

    The detector only sees row and column totals, so a round outline (whose
    extremes pile up into two peaks each way) reads as a 1x1 grid. Walking
    along each line and checking the mask is drawn there tells them apart.
    """
    mask = ndimage.binary_dilation(binary, iterations=2)
    h, w = mask.shape
    x0, y0, x1, y1 = lattice.bounds
    ys = np.clip(np.linspace(y0, y1, 200).round().astype(int), 0, h - 1)
    xs = np.clip(np.linspace(x0, x1, 200).round().astype(int), 0, w - 1)
    for x in lattice.x_lines:
        if mask[ys, min(max(int(round(x)), 0), w - 1)].mean() < min_coverage:
            return False
    for y in lattice.y_lines:
        if mask[min(max(int(round(y)), 0), h - 1), xs].mean() < min_coverage:
            return False
    return True


def extract_outline_from_stl(
    stl_path: str,
    svg_out_path: str,
    simplify_epsilon: float = 0.0008,
    topology: TopologyParam = "auto",
) -> TraceResult:
    mesh = trimesh.load(stl_path, force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("STL file is empty or invalid.")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError("STL file has no faces.")

    proj = _project_mesh(mesh)
    binary = proj.binary
    contours = measure.find_contours(binary.astype(float), 0.5)
    if not contours:
        raise ValueError("Could not extract a valid outline from the STL.")

    h, w = binary.shape
    lattice = None
    try:
        lattice = extract_lattice_from_mask(binary)
    except LatticeDetectionError:
        lattice = None
    if lattice is not None and topology == "auto" and not _lattice_is_drawn(lattice, binary):
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

    # Prefer the exact face footprint. Fall back to the raster silhouette when
    # the faces barely cover it: a model of side walls only projects to slivers
    # (or nothing), and keeping the biggest sliver builds a junk cutter.
    silhouette = _silhouette_polygon(proj, simplify_epsilon)
    poly = None
    if proj.geom is not None:
        geom = proj.geom
        poly = max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom
        poly = poly.simplify(simplify_epsilon, preserve_topology=True).buffer(0)
        # Compare outer outlines, not material: a washer or a letter O covers
        # far less than its filled silhouette but is still exactly right.
        if poly.is_empty or not poly.is_valid or Polygon(poly.exterior).area < 0.5 * silhouette.area:
            poly = None
    if poly is None:
        poly = silhouette
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
