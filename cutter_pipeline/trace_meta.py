"""Serialize and deserialize trace results for job persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shapely.geometry import mapping, shape

from cutter_pipeline.lattice_extractor import LatticeGeometry
from cutter_pipeline.trace_outline import TraceResult


def lattice_to_dict(lattice: LatticeGeometry) -> dict[str, Any]:
    return {
        "x_lines": lattice.x_lines,
        "y_lines": lattice.y_lines,
        "bounds": list(lattice.bounds),
    }


def lattice_from_dict(data: dict[str, Any]) -> LatticeGeometry:
    return LatticeGeometry(
        x_lines=[float(x) for x in data["x_lines"]],
        y_lines=[float(y) for y in data["y_lines"]],
        bounds=tuple(float(v) for v in data["bounds"]),
    )


def trace_result_to_dict(traced: TraceResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "topology": traced.topology,
        "topology_requested": traced.topology_requested,
        "topology_detected": traced.topology_detected,
        "contour_count": traced.contour_count,
        "cols": traced.cols,
        "rows": traced.rows,
        "grid_hint": traced.grid_hint,
        "extraction_mode": traced.extraction_mode,
        "extraction_warning": traced.extraction_warning,
    }
    if traced.polygon is not None:
        payload["polygon"] = mapping(traced.polygon)
    if traced.lattice is not None:
        payload["lattice"] = lattice_to_dict(traced.lattice)
    return payload


def trace_result_from_dict(data: dict[str, Any]) -> TraceResult:
    polygon = shape(data["polygon"]) if data.get("polygon") else None
    lattice = lattice_from_dict(data["lattice"]) if data.get("lattice") else None
    return TraceResult(
        polygon=polygon,
        lattice=lattice,
        topology=data["topology"],
        topology_requested=data.get("topology_requested", "auto"),
        topology_detected=data.get("topology_detected"),
        contour_count=int(data.get("contour_count", 0)),
        cols=data.get("cols"),
        rows=data.get("rows"),
        grid_hint=data.get("grid_hint"),
        svg_path=data.get("svg_path", ""),
        svg_file=data.get("svg_file", ""),
        extraction_mode=data.get("extraction_mode", "binary"),
        extraction_warning=data.get("extraction_warning", ""),
    )


def save_trace_result(job_dir: Path, traced: TraceResult) -> Path:
    meta_path = job_dir / "trace_meta.json"
    meta_path.write_text(json.dumps(trace_result_to_dict(traced)), encoding="utf-8")
    if traced.polygon is not None:
        poly_path = job_dir / "polygon.json"
        poly_path.write_text(json.dumps(mapping(traced.polygon)), encoding="utf-8")
    return meta_path


def load_trace_result(job_dir: Path) -> TraceResult:
    meta_path = job_dir / "trace_meta.json"
    if not meta_path.exists():
        poly_path = job_dir / "polygon.json"
        if not poly_path.exists():
            raise FileNotFoundError("trace_meta.json not found for this job.")
        polygon = shape(json.loads(poly_path.read_text(encoding="utf-8")))
        return TraceResult(
            polygon=polygon,
            lattice=None,
            topology="single",
            topology_requested="single",
            topology_detected="single",
            contour_count=1,
            cols=None,
            rows=None,
            grid_hint=None,
            svg_path="",
            svg_file="",
        )
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return trace_result_from_dict(data)
