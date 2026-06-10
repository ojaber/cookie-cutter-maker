"""Dispatch STL generation based on traced topology."""

from __future__ import annotations

from cutter_pipeline.lattice_cutter import lattice_height_mm, lattice_to_cookie_cutter_stl
from cutter_pipeline.stl_cutter import polygon_to_cookie_cutter_stl
from cutter_pipeline.trace_outline import TraceResult


def generate_stl_from_trace(
    traced: TraceResult,
    stl_path: str,
    *,
    target_width_mm: float = 95.0,
    wall_mm: float = 1.0,
    total_h_mm: float = 25.0,
    flange_h_mm: float = 7.226,
    flange_out_mm: float = 5.0,
    bevel_h_mm: float = 2.0,
    bevel_top_wall_mm: float = 0.5,
    cleanup_mm: float = 0.5,
    tip_smooth_mm: float = 0.6,
    drop_holes: bool = True,
    min_component_area_mm2: float = 25.0,
) -> dict:
    if traced.topology == "lattice":
        if traced.lattice is None:
            raise ValueError("Lattice topology selected but no lattice geometry was traced.")
        lattice_to_cookie_cutter_stl(
            traced.lattice,
            stl_path,
            target_width_mm=target_width_mm,
            wall_mm=wall_mm,
            total_h_mm=total_h_mm,
            flange_h_mm=flange_h_mm,
            flange_out_mm=flange_out_mm,
        )
        return {
            "height_mm": lattice_height_mm(traced.lattice, target_width_mm),
            "cols": traced.cols,
            "rows": traced.rows,
        }

    if traced.polygon is None:
        raise ValueError("Single-shape topology selected but no polygon was traced.")

    polygon_to_cookie_cutter_stl(
        traced.polygon,
        stl_path,
        target_width_mm=target_width_mm,
        wall_mm=wall_mm,
        total_h_mm=total_h_mm,
        flange_h_mm=flange_h_mm,
        flange_out_mm=flange_out_mm,
        bevel_h_mm=bevel_h_mm,
        bevel_top_wall_mm=bevel_top_wall_mm,
        cleanup_mm=cleanup_mm,
        tip_smooth_mm=tip_smooth_mm,
        drop_holes=drop_holes,
        min_component_area_mm2=min_component_area_mm2,
    )
    return {"height_mm": None, "cols": None, "rows": None}
