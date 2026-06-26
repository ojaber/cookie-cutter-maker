from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import box

from cutter_pipeline.stl_cutter import polygon_to_cookie_cutter_stl


def _radial_band_stats(mesh: trimesh.Trimesh, z_min: float, z_max: float) -> tuple[float, float]:
    verts = mesh.vertices
    band = verts[(verts[:, 2] >= z_min) & (verts[:, 2] <= z_max)]
    if len(band) == 0:
        return 0.0, 0.0
    center = band[:, :2].mean(axis=0)
    d = np.linalg.norm(band[:, :2] - center, axis=1)
    outer_r = float(np.percentile(d, 95))
    inner_r = float(np.percentile(d, 5))
    return outer_r, inner_r


def _min_top_wall_mm(mesh: trimesh.Trimesh) -> float:
    zmax = mesh.vertices[:, 2].max()
    top_xy = mesh.vertices[np.isclose(mesh.vertices[:, 2], zmax)][:, :2]
    if len(top_xy) < 4:
        return 0.0
    center = top_xy.mean(axis=0)
    radii = np.linalg.norm(top_xy - center, axis=1)
    order = np.argsort(radii)
    split = len(top_xy) // 2
    inner = top_xy[order[:split]]
    outer = top_xy[order[split:]]
    if len(inner) == 0 or len(outer) == 0:
        return 0.0
    d2 = ((outer[:, None, :] - inner[None, :, :]) ** 2).sum(axis=2)
    nearest = np.sqrt(d2.min(axis=1))
    return float(nearest.min())


def test_polygon_to_cookie_cutter_stl(tmp_path: Path) -> None:
    square = box(0, 0, 1, 1)
    out_path = tmp_path / "cutter.stl"

    polygon_to_cookie_cutter_stl(
        square,
        str(out_path),
        target_width_mm=80.0,
        wall_mm=1.2,
        total_h_mm=20.0,
        flange_h_mm=6.0,
        flange_out_mm=5.0,
        cleanup_mm=0.4,
        tip_smooth_mm=0.5,
        drop_holes=True,
        min_component_area_mm2=10.0,
    )

    assert out_path.exists()
    mesh = trimesh.load(out_path, force="mesh")

    assert mesh.vertices.shape[0] > 0
    # X extent should at least match target width (flange grows it further).
    assert mesh.extents[0] >= 80.0


def test_taper_reduces_top_wall_thickness(tmp_path: Path) -> None:
    square = box(0, 0, 1, 1)
    out_path = tmp_path / "taper.stl"

    polygon_to_cookie_cutter_stl(
        square,
        str(out_path),
        target_width_mm=60.0,
        wall_mm=1.0,
        total_h_mm=20.0,
        flange_h_mm=6.0,
        flange_out_mm=5.0,
        cleanup_mm=0.0,
        tip_smooth_mm=0.0,
        drop_holes=True,
        min_component_area_mm2=1.0,
        bevel_h_mm=2.0,
        bevel_top_wall_mm=0.5,
    )

    mesh = trimesh.load(out_path, force="mesh")
    zmax = mesh.vertices[:, 2].max()
    bevel_h = 2.0
    bevel_start = zmax - bevel_h
    top_outer, top_inner = _radial_band_stats(mesh, zmax - 0.2, zmax + 1e-6)
    mid_outer, mid_inner = _radial_band_stats(mesh, zmax - bevel_h - 0.1, zmax - bevel_h + 0.1)

    assert top_outer < mid_outer - 0.3, f"Taper too small (top {top_outer:.3f}, mid {mid_outer:.3f})"
    assert abs(top_inner - mid_inner) < 0.2, (
        f"Inner wall drifted (top {top_inner:.3f}, mid {mid_inner:.3f})"
    )

    bevel_band = mesh.vertices[mesh.vertices[:, 2] >= bevel_start - 1e-6]
    unique_bevel_z = np.unique(np.round(bevel_band[:, 2], 3))
    assert len(unique_bevel_z) >= 6, f"Expected gradual bevel profile, found only {len(unique_bevel_z)} levels"


def test_taper_top_wall_has_minimum_guard(tmp_path: Path) -> None:
    square = box(0, 0, 1, 1)
    out_path = tmp_path / "taper_guard.stl"

    polygon_to_cookie_cutter_stl(
        square,
        str(out_path),
        target_width_mm=60.0,
        wall_mm=1.0,
        total_h_mm=20.0,
        flange_h_mm=6.0,
        flange_out_mm=5.0,
        cleanup_mm=0.0,
        tip_smooth_mm=0.0,
        drop_holes=True,
        min_component_area_mm2=1.0,
        bevel_h_mm=2.0,
        bevel_top_wall_mm=0.2,  # Should clamp up to 0.45mm minimum.
    )

    mesh = trimesh.load(out_path, force="mesh")
    min_top_wall = _min_top_wall_mm(mesh)
    assert min_top_wall >= 0.43, f"Top wall guard failed, got {min_top_wall:.3f}mm"


def test_flange_corner_radius_rounds_corners(tmp_path: Path) -> None:
    square = box(0, 0, 1, 1)
    sharp_path = tmp_path / "sharp.stl"
    round_path = tmp_path / "round.stl"

    polygon_to_cookie_cutter_stl(
        square,
        str(sharp_path),
        target_width_mm=80.0,
        wall_mm=1.2,
        total_h_mm=20.0,
        flange_h_mm=6.0,
        flange_out_mm=5.0,
        flange_corner_radius_mm=0.0,
        cleanup_mm=0.0,
        tip_smooth_mm=0.0,
        drop_holes=True,
        min_component_area_mm2=1.0,
    )
    polygon_to_cookie_cutter_stl(
        square,
        str(round_path),
        target_width_mm=80.0,
        wall_mm=1.2,
        total_h_mm=20.0,
        flange_h_mm=6.0,
        flange_out_mm=5.0,
        flange_corner_radius_mm=4.0,
        cleanup_mm=0.0,
        tip_smooth_mm=0.0,
        drop_holes=True,
        min_component_area_mm2=1.0,
    )

    assert sharp_path.exists() and round_path.exists()
    sharp = trimesh.load(sharp_path, force="mesh")
    rounded = trimesh.load(round_path, force="mesh")
    # Rounding the corners trims the outer flange footprint, so the bounding
    # box of the rounded version should be no larger than the sharp one and the
    # geometry should differ.
    assert rounded.extents[0] <= sharp.extents[0] + 1e-6
    assert not np.allclose(sorted(sharp.extents), sorted(rounded.extents)) or (
        len(rounded.vertices) != len(sharp.vertices)
    )


def test_wall_thickness_has_minimum_guard(tmp_path: Path) -> None:
    square = box(0, 0, 1, 1)
    out_path = tmp_path / "wall_guard.stl"

    polygon_to_cookie_cutter_stl(
        square,
        str(out_path),
        target_width_mm=60.0,
        wall_mm=0.3,  # Should clamp up to 0.45mm minimum.
        total_h_mm=20.0,
        flange_h_mm=6.0,
        flange_out_mm=5.0,
        cleanup_mm=0.0,
        tip_smooth_mm=0.0,
        drop_holes=True,
        min_component_area_mm2=1.0,
        bevel_h_mm=2.0,
        bevel_top_wall_mm=0.2,
    )

    mesh = trimesh.load(out_path, force="mesh")
    min_top_wall = _min_top_wall_mm(mesh)
    assert min_top_wall >= 0.43, f"Wall guard failed, got {min_top_wall:.3f}mm"
