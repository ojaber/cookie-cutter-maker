"""Generate STL cookie cutters from lattice / grid line geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from cutter_pipeline.lattice_extractor import LatticeGeometry

MIN_WALL_MM = 0.45


def lattice_to_cookie_cutter_stl(
    lattice: LatticeGeometry,
    out_path: str,
    target_width_mm: float = 95.0,
    wall_mm: float = 1.0,
    total_h_mm: float = 25.0,
    flange_h_mm: float = 7.226,
    flange_out_mm: float = 5.0,
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
            flange.apply_translation([0, 0, total_h_mm - flange_h_mm])
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
