"""Serialize trace results for the client, and validate the ones it posts back."""

from __future__ import annotations

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
    if traced.grid_spec is not None:
        payload["grid_spec"] = traced.grid_spec
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
        grid_spec=data.get("grid_spec"),
    )


# ── Traces posted back by a client ────────────────────────────────────────────
# The server keeps nothing between requests: a trace is sent to the browser and
# the browser posts it back to have a cutter built. That makes the payload
# untrusted input headed straight for the mesher, so everything in it is
# checked for shape, type, size and range before a TraceResult is built.

MAX_TRACE_VERTICES = 50_000
MAX_TRACE_RINGS = 500
MAX_LATTICE_LINES = 201
_MAX_COORD = 1e7
_MAX_TEXT = 500


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid trace: {what} must be a number.")
    v = float(value)
    if v != v or abs(v) > _MAX_COORD:  # NaN or absurd magnitude
        raise ValueError(f"Invalid trace: {what} is out of range.")
    return v


def _opt_int(value: Any, what: str, lo: int, hi: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"Invalid trace: {what} must be a whole number from {lo} to {hi}.")
    return value


def _opt_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Invalid trace: text fields must be strings.")
    return value[:_MAX_TEXT]


def _polygon_from_payload(geo: Any):
    from shapely.geometry import MultiPolygon, Polygon

    if not isinstance(geo, dict) or geo.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("Invalid trace: polygon must be a GeoJSON Polygon or MultiPolygon.")
    coords = geo.get("coordinates")
    if not isinstance(coords, list) or not coords:
        raise ValueError("Invalid trace: polygon has no coordinates.")
    polygons_raw = [coords] if geo["type"] == "Polygon" else coords

    budget = {"vertices": 0, "rings": 0}

    def ring(raw: Any) -> list[tuple[float, float]]:
        if not isinstance(raw, list) or len(raw) < 4:
            raise ValueError("Invalid trace: every ring needs at least 4 points.")
        budget["rings"] += 1
        budget["vertices"] += len(raw)
        if budget["rings"] > MAX_TRACE_RINGS or budget["vertices"] > MAX_TRACE_VERTICES:
            raise ValueError("Invalid trace: the outline has too many points.")
        pts = []
        for p in raw:
            if not isinstance(p, list) or len(p) not in (2, 3):
                raise ValueError("Invalid trace: points must be [x, y] pairs.")
            pts.append((_finite(p[0], "x"), _finite(p[1], "y")))
        return pts

    polys = []
    for rings in polygons_raw:
        if not isinstance(rings, list) or not rings:
            raise ValueError("Invalid trace: polygon has an empty ring list.")
        shell = ring(rings[0])
        holes = [ring(r) for r in rings[1:]]
        polys.append(Polygon(shell, holes))
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if geom.is_empty or not geom.is_valid or geom.area <= 0:
        raise ValueError("Invalid trace: the outline is not a valid shape.")
    return geom


def _lattice_from_payload(raw: Any) -> LatticeGeometry:
    if not isinstance(raw, dict):
        raise ValueError("Invalid trace: lattice is missing.")
    lines = {}
    for key in ("x_lines", "y_lines"):
        values = raw.get(key)
        if not isinstance(values, list) or not 2 <= len(values) <= MAX_LATTICE_LINES:
            raise ValueError(f"Invalid trace: lattice needs 2 to {MAX_LATTICE_LINES} {key}.")
        nums = [_finite(v, key) for v in values]
        if any(b <= a for a, b in zip(nums, nums[1:])):
            raise ValueError(f"Invalid trace: lattice {key} must be strictly increasing.")
        lines[key] = nums
    # The mesher scales by the bounds but draws walls from the lines, so posted
    # bounds that disagree with the lines would build a cutter at the wrong
    # size. Every trace the server produces has bounds spanning the first and
    # last lines, so derive them rather than trust the payload.
    x, y = lines["x_lines"], lines["y_lines"]
    return LatticeGeometry(x_lines=x, y_lines=y, bounds=(x[0], y[0], x[-1], y[-1]))


def trace_result_from_payload(data: Any) -> TraceResult:
    """Validate a client-supplied trace and rebuild the TraceResult from it.

    Raises ValueError (a 400 at the API) on anything malformed.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid trace: expected a JSON object.")
    topology = data.get("topology")
    if topology not in ("single", "lattice"):
        raise ValueError("Invalid trace: topology must be 'single' or 'lattice'.")

    polygon = None
    lattice = None
    grid_spec = data.get("grid_spec")
    if grid_spec is not None:
        # Grid Builder traces are exact millimetres described by four numbers.
        # Rebuild the lattice from those rather than trusting the posted line
        # positions, so the cutter always matches the spec it claims to be.
        from cutter_pipeline.grid_spec import lattice_from_spec

        if not isinstance(grid_spec, dict):
            raise ValueError("Invalid trace: grid_spec must be an object.")
        cols = _opt_int(grid_spec.get("cols"), "grid cols", 1, 1000)
        rows = _opt_int(grid_spec.get("rows"), "grid rows", 1, 1000)
        if cols is None or rows is None:
            raise ValueError("Invalid trace: grid_spec needs cols and rows.")
        cell_w = _finite(grid_spec.get("cell_w_mm"), "cell_w_mm")
        cell_h = _finite(grid_spec.get("cell_h_mm"), "cell_h_mm")
        lattice = lattice_from_spec(cols, rows, cell_w, cell_h)
        grid_spec = {"cols": cols, "rows": rows, "cell_w_mm": cell_w, "cell_h_mm": cell_h}
        topology = "lattice"
    elif topology == "lattice":
        lattice = _lattice_from_payload(data.get("lattice"))
    else:
        polygon = _polygon_from_payload(data.get("polygon"))

    requested = data.get("topology_requested", "auto")
    if requested not in ("auto", "single", "lattice"):
        requested = "auto"
    detected = data.get("topology_detected")
    if detected not in (None, "single", "lattice"):
        detected = None
    contour_count = _opt_int(data.get("contour_count", 0), "contour_count", 0, 1_000_000) or 0
    cols = lattice.cols if lattice is not None else None
    rows = lattice.rows if lattice is not None else None

    return TraceResult(
        polygon=polygon,
        lattice=lattice,
        topology=topology,
        topology_requested=requested,
        topology_detected=detected,
        contour_count=contour_count,
        cols=cols,
        rows=rows,
        grid_hint=_opt_text(data.get("grid_hint")),
        svg_path="",
        svg_file="",
        extraction_mode=_opt_text(data.get("extraction_mode")) or "binary",
        extraction_warning=_opt_text(data.get("extraction_warning")) or "",
        grid_spec=grid_spec,
    )
