from pathlib import Path

import numpy as np
import trimesh
from cutter_pipeline.trace_outline import trace_png_to_polygon
from cutter_pipeline.stl_cutter import polygon_to_cookie_cutter_stl

ASSETS = Path(__file__).parent / "assets"

CASES = [
    ("cactus.png", 95.0),
    ("dino.png", 95.0),
    ("heart.png", 95.0),
    ("doggy.png", 95.0),
]


def _min_top_gap(mesh: trimesh.Trimesh, band_mm: float = 0.1) -> float:
    verts = mesh.vertices
    zmax = verts[:, 2].max()
    band = verts[verts[:, 2] > zmax - band_mm]
    if len(band) == 0:
        return 0.0
    center = band[:, :2].mean(axis=0)
    d = np.linalg.norm(band[:, :2] - center, axis=1)
    # simple 2-means on radial distance to split inner/outer
    c1, c2 = d.min(), d.max()
    for _ in range(8):
        labels = np.where(np.abs(d - c1) < np.abs(d - c2), 0, 1)
        if labels.sum() == 0 or labels.sum() == len(labels):
            break
        c1 = d[labels == 0].mean()
        c2 = d[labels == 1].mean()
    inner = band[labels == (0 if c1 < c2 else 1), :2]
    outer = band[labels == (1 if c1 < c2 else 0), :2]
    if len(inner) == 0 or len(outer) == 0:
        return 0.0
    from scipy.spatial import cKDTree

    return cKDTree(inner).query(outer, k=1)[0].min()


def test_shapes_top_gap(tmp_path: Path):
    for name, width in CASES:
        png = ASSETS / name
        traced = trace_png_to_polygon(str(png), str(tmp_path / f"{name}.svg"), topology="auto")
        out = tmp_path / f"{name}.stl"
        from cutter_pipeline.stl_dispatch import generate_stl_from_trace

        generate_stl_from_trace(traced, str(out), target_width_mm=width)
        mesh = trimesh.load(out, force="mesh")
        gap = _min_top_gap(mesh)
        assert gap >= 0.3, f"{name}: top gap too small ({gap:.3f} mm)"
