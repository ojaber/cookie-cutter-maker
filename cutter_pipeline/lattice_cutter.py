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
    wall_face: Polygon,
    base_z: float,
    chamfer_h_mm: float,
    chamfer_out_mm: float,
) -> trimesh.Trimesh | None:
    """Build a SOLID triangular brace in the inner corner where the grid walls
    meet the top of the flange shelf.

    The brace sits on the flange top (z=``base_z``) where it is widest
    (extending ``chamfer_out_mm`` outward from the grid wall face) and shrinks
    to zero width at z=``base_z`` + ``chamfer_h_mm`` up the wall, forming a
    fillet that braces the wall-to-flange junction. Built as stacked extruded
    ring slices so the result is a watertight solid the slicer unions with the
    body and flange.
    """
    if chamfer_h_mm <= 0 or chamfer_out_mm <= 0:
        return None

    steps = max(2, int(np.ceil(chamfer_h_mm / 0.3)))
    meshes: list[trimesh.Trimesh] = []
    for i in range(steps):
        z0 = base_z + chamfer_h_mm * (i / steps)
        z1 = base_z + chamfer_h_mm * ((i + 1) / steps)
        t_mid = (i + 0.5) / steps
        width = chamfer_out_mm * (1.0 - t_mid)
        if width <= 1e-6:
            continue
        ring = wall_face.buffer(width, join_style=2).difference(wall_face)
        if ring.is_empty:
            continue
        ring_parts = list(ring.geoms) if ring.geom_type == "MultiPolygon" else [ring]
        for part in ring_parts:
            if part.is_empty:
                continue
            mesh = trimesh.creation.extrude_polygon(part, z1 - z0, engine="earcut")
            mesh.apply_translation([0, 0, z0])
            meshes.append(mesh)

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def lattice_to_cookie_cutter_stl(
    lattice: LatticeGeometry,
    out_path: str,
    target_width_mm: float = 95.0,
    wall_mm: float = 1.0,
    total_h_mm: float = 25.0,
    flange_h_mm: float = 7.226,
    flange_out_mm: float = 5.0,
    flange_chamfer_mm: float = 0.5,
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
            parts_to_join = [body, flange]
            # Add a solid chamfer brace around the grid's outer wall face so the
            # wall-to-flange junction isn't a sharp 90-degree stress point.
            if flange_chamfer_mm > 0:
                wall_face = outer.buffer(wall_mm / 2, join_style=2)
                chamfer_out = min(flange_chamfer_mm, flange_out_mm)
                # Brace sits on top of the flange shelf and rises up the wall,
                # without exceeding the wall's remaining height.
                chamfer_h = min(flange_chamfer_mm, max(0.0, total_h_mm - flange_h_mm))
                chamfer = _create_lattice_chamfer(
                    wall_face, flange_h_mm, chamfer_h, chamfer_out
                )
                if chamfer is not None:
                    parts_to_join.append(chamfer)
            # Flange sits at z=0 (build plate / base)
            body = trimesh.util.concatenate(parts_to_join)
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
