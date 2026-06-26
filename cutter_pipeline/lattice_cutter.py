"""Generate STL cookie cutters from lattice / grid line geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from cutter_pipeline.lattice_extractor import LatticeGeometry

MIN_WALL_MM = 0.45


def _create_lattice_chamfer(
    inner_poly: Polygon,
    outer_poly: Polygon,
    base_z: float,
    chamfer_depth: float,
) -> trimesh.Trimesh:
    """Create a chamfered transition between inner and outer polygons for lattice flanges."""
    samples = 64  # Fewer samples needed for rectangular lattice flanges
    from shapely.geometry.polygon import orient

    inner_oriented = orient(inner_poly, sign=1.0)
    outer_oriented = orient(outer_poly, sign=1.0)

    inner_coords = list(inner_oriented.exterior.coords)
    outer_coords = list(outer_oriented.exterior.coords)

    # Sample rings
    if inner_coords[0] != inner_coords[-1]:
        inner_coords = inner_coords + [inner_coords[0]]
    if outer_coords[0] != outer_coords[-1]:
        outer_coords = outer_coords + [outer_coords[0]]

    inner_line = LineString(inner_coords)
    outer_line = LineString(outer_coords)

    inner_ring = np.array([inner_line.interpolate(inner_line.length * (i / samples)).coords[0] for i in range(samples)])
    outer_ring = np.array([outer_line.interpolate(outer_line.length * (i / samples)).coords[0] for i in range(samples)])

    # Align phases to avoid twisted geometry
    distances = np.linalg.norm(outer_ring - inner_ring[0], axis=1)
    shift = int(np.argmin(distances))
    outer_ring = np.roll(outer_ring, -shift, axis=0)

    # Create chamfer with multiple steps for smooth transition
    chamfer_steps = max(2, int(np.ceil(chamfer_depth / 0.25)))
    rings: list[tuple[np.ndarray, float]] = []

    for step in range(chamfer_steps + 1):
        t = step / chamfer_steps
        z = base_z - (chamfer_depth * t)
        # Interpolate between inner and outer rings
        interpolated = inner_ring * (1 - t) + outer_ring * t
        rings.append((interpolated, z))

    # Generate faces between rings
    vertices = []
    faces = []

    for i, (ring, z) in enumerate(rings):
        vertices.extend(np.column_stack([ring, np.full((samples, 1), z)]))

    for i in range(len(rings) - 1):
        for j in range(samples):
            v0 = i * samples + j
            v1 = i * samples + ((j + 1) % samples)
            v2 = (i + 1) * samples + j
            v3 = (i + 1) * samples + ((j + 1) % samples)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)


def lattice_to_cookie_cutter_stl(
    lattice: LatticeGeometry,
    out_path: str,
    target_width_mm: float = 95.0,
    wall_mm: float = 1.0,
    total_h_mm: float = 25.0,
    flange_h_mm: float = 7.226,
    flange_out_mm: float = 5.0,
    flange_chamfer_mm: float = 0.0,
    bottom_wall_mm: float = None,
    cutting_wall_h_mm: float = None,
) -> str:
    wall_mm = max(wall_mm, MIN_WALL_MM)

    min_x, min_y, max_x, max_y = lattice.bounds
    grid_w_px = max_x - min_x
    grid_h_px = max_y - min_y
    if grid_w_px <= 0 or grid_h_px <= 0:
        raise ValueError("Invalid lattice bounds.")

    scale = target_width_mm / grid_w_px

    def to_mm(x: float, y: float) -> tuple[float, float]:
        return ((x - min_x) * scale, (y - min_y) * scale)

    segments: list[LineString] = []
    for x in lattice.x_lines:
        segments.append(LineString([to_mm(x, lattice.y_lines[0]), to_mm(x, lattice.y_lines[-1])]))
    for y in lattice.y_lines:
        segments.append(LineString([to_mm(lattice.x_lines[0], y), to_mm(lattice.x_lines[-1], y)]))

    # Determine if bottom taper is active
    use_taper = (
        bottom_wall_mm is not None
        and cutting_wall_h_mm is not None
        and bottom_wall_mm < wall_mm
    )
    if use_taper:
        cutting_wall_h_mm = max(0.0, min(cutting_wall_h_mm, total_h_mm))
        bottom_wall_mm = max(bottom_wall_mm, MIN_WALL_MM)

    def _make_lattice_union(half_wall: float):
        polys = [
            seg.buffer(half_wall, cap_style=2, join_style=2)
            for seg in segments
        ]
        return unary_union(polys)

    if use_taper:
        # Full-thickness section from z=0 to z=(total_h_mm - cutting_wall_h_mm)
        body_meshes = []
        full_h = total_h_mm - cutting_wall_h_mm
        if full_h > 0:
            full_union = _make_lattice_union(wall_mm / 2)
            if not full_union.is_empty:
                full_parts = list(full_union.geoms) if full_union.geom_type == "MultiPolygon" else [full_union]
                for part in full_parts:
                    if part.is_empty:
                        continue
                    mesh = trimesh.creation.extrude_polygon(part, full_h, engine="earcut")
                    body_meshes.append(mesh)
        # Taper slices: full wall at z=full_h, thin at z=total_h_mm (cutting tip)
        taper_steps = max(2, int(np.ceil(cutting_wall_h_mm / 0.5)))
        for step in range(taper_steps):
            t0 = step / taper_steps
            t1 = (step + 1) / taper_steps
            z0 = full_h + cutting_wall_h_mm * t0
            z1 = full_h + cutting_wall_h_mm * t1
            t_mid = (t0 + t1) / 2
            cur_wall = wall_mm + (bottom_wall_mm - wall_mm) * t_mid
            layer_union = _make_lattice_union(cur_wall / 2)
            if layer_union.is_empty:
                continue
            layer_parts = list(layer_union.geoms) if layer_union.geom_type == "MultiPolygon" else [layer_union]
            for part in layer_parts:
                if part.is_empty:
                    continue
                mesh = trimesh.creation.extrude_polygon(part, z1 - z0, engine="earcut")
                mesh.apply_translation([0, 0, z0])
                body_meshes.append(mesh)
    else:
        wall_polys = [
            seg.buffer(wall_mm / 2, cap_style=2, join_style=2)
            for seg in segments
        ]
        lattice_union = unary_union(wall_polys)
        if lattice_union.is_empty:
            raise ValueError("Lattice wall geometry is empty.")
        parts = list(lattice_union.geoms) if lattice_union.geom_type == "MultiPolygon" else [lattice_union]
        body_meshes = [
            trimesh.creation.extrude_polygon(part, total_h_mm, engine="earcut")
            for part in parts
            if not part.is_empty
        ]

    if not body_meshes:
        raise ValueError("Failed to extrude lattice walls.")

    body = trimesh.util.concatenate(body_meshes)
    body.merge_vertices()

    outer = Polygon(
        [
            to_mm(lattice.x_lines[0], lattice.y_lines[0]),
            to_mm(lattice.x_lines[-1], lattice.y_lines[0]),
            to_mm(lattice.x_lines[-1], lattice.y_lines[-1]),
            to_mm(lattice.x_lines[0], lattice.y_lines[-1]),
        ]
    )
    flange_outer = outer.buffer(flange_out_mm, join_style=2)
    flange_ring = flange_outer.difference(outer.buffer(wall_mm / 2, join_style=2))
    if not flange_ring.is_empty:
        flange_parts = list(flange_ring.geoms) if flange_ring.geom_type == "MultiPolygon" else [flange_ring]
        flange_meshes = [
            trimesh.creation.extrude_polygon(p, flange_h_mm, engine="earcut")
            for p in flange_parts
            if not p.is_empty
        ]
        if flange_meshes:
            flange = trimesh.util.concatenate(flange_meshes)
            # Add chamfer between flange and body if requested
            if flange_chamfer_mm > 0:
                chamfer = _create_lattice_chamfer(
                    outer, flange_outer, total_h_mm, flange_chamfer_mm
                )
                body = trimesh.util.concatenate([body, chamfer, flange])
            else:
                # Flange sits at z=0 (build plate / base)
                body = trimesh.util.concatenate([body, flange])
            body.merge_vertices()

    if body.volume < 0:
        body.invert()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body.export(str(out))
    return str(out)


def lattice_height_mm(lattice: LatticeGeometry, target_width_mm: float) -> float:
    min_x, min_y, max_x, max_y = lattice.bounds
    grid_w_px = max_x - min_x
    grid_h_px = max_y - min_y
    if grid_w_px <= 0:
        return target_width_mm
    return target_width_mm * (grid_h_px / grid_w_px)
