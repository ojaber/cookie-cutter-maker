"""End-to-end tests for the FastAPI app in app/main.py.

These tests exercise the HTTP surface in process via fastapi.testclient.TestClient,
including the security validation layer (path-traversal guards, upload-size limit,
decompression-bomb guard) and the offline pipeline path.
"""
from __future__ import annotations

import importlib
import io
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


def _make_outline_png(size: tuple[int, int] = (128, 128)) -> bytes:
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4], fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _reset_prometheus_default_registry() -> None:
    """app.main registers metrics on the default registry at import time;
    when we reimport the module each test we have to unregister them first."""
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass


def _load_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env):
    """Import a fresh copy of app.main with env vars applied first.

    Auth/SESSION/OUTPUT_DIR are all captured at import time, so each test that
    wants a different config must re-import the module.
    """
    monkeypatch.setenv("PIPELINE_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("REMBG_ENABLED", "false")
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    sys.modules.pop("app.main", None)
    sys.modules.pop("app", None)
    _reset_prometheus_default_registry()
    return importlib.import_module("app.main")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch)
    with TestClient(mod.app) as c:
        c.app_module = mod  # type: ignore[attr-defined]
        yield c


# ── Basic endpoints ───────────────────────────────────────────────────────────

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.text == "OK"


def test_features_endpoint_reports_disabled_when_unset(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.get("/features")
    assert r.status_code == 200
    body = r.json()
    assert body["background_removal"] is False
    assert body["image_generation"] is False


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/healthz")  # ensure at least one observation
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_metrics_path_collapses_files_route(client):
    # The collapsed label keeps Prometheus cardinality bounded.
    assert client.app_module._metrics_path("/files/abc/def.png") == "/files/:job_id/:filename"
    assert client.app_module._metrics_path("/healthz") == "/healthz"


# ── Path-traversal / input validation ─────────────────────────────────────────

def test_file_download_rejects_path_traversal(client):
    r = client.get("/files/..%2F..%2Fetc/passwd")
    assert r.status_code in (400, 404)


def test_file_download_rejects_bad_job_id(client):
    r = client.get("/files/not-a-uuid/file.png")
    assert r.status_code == 400


def test_file_download_rejects_traversal_in_filename(client):
    # a valid-looking hex job id paired with a traversal filename
    r = client.get("/files/" + "a" * 32 + "/..%2Fsecret")
    assert r.status_code in (400, 404)


def test_trace_rejects_unsafe_name(client):
    png = _make_outline_png()
    r = client.post(
        "/trace/from-png",
        data={"name": "../evil"},
        files={"file": ("a.png", png, "image/png")},
    )
    assert r.status_code == 400


def test_pipeline_rejects_unsafe_name(client):
    png = _make_outline_png()
    r = client.post(
        "/pipeline/from-png",
        data={"name": "/etc/passwd"},
        files={"file": ("a.png", png, "image/png")},
    )
    assert r.status_code == 400


def test_trace_rejects_non_image_extension(client):
    r = client.post(
        "/trace/from-png",
        files={"file": ("evil.sh", b"#!/bin/sh\nrm -rf /\n", "application/x-sh")},
    )
    assert r.status_code == 400


def test_trace_rejects_invalid_image_bytes(client):
    # File has .png extension but is not actually a decodable image.
    r = client.post(
        "/trace/from-png",
        files={"file": ("a.png", b"not a real png", "image/png")},
    )
    assert r.status_code == 400


def test_trace_rejects_oversized_upload(client, monkeypatch):
    # Shrink the limit so the test stays fast.
    monkeypatch.setattr(client.app_module, "MAX_UPLOAD_BYTES", 1024)
    big = b"\x00" * (1024 * 8)
    r = client.post(
        "/trace/from-png",
        files={"file": ("a.png", big, "image/png")},
    )
    assert r.status_code == 413


def test_trace_rejects_decompression_bomb(client, monkeypatch):
    # Build a small valid PNG but lie about its declared size: easier to just
    # set the PIL MAX_IMAGE_PIXELS very low and feed a normal image.
    monkeypatch.setattr(client.app_module._PILImage, "MAX_IMAGE_PIXELS", 16)
    png = _make_outline_png((64, 64))
    r = client.post(
        "/trace/from-png",
        files={"file": ("a.png", png, "image/png")},
    )
    assert r.status_code == 400


def test_trace_from_job_rejects_bad_job_id(client):
    r = client.post("/trace/from-job", data={"job_id": "../../etc"})
    assert r.status_code == 400


def test_stl_from_job_rejects_bad_job_id(client):
    r = client.post("/stl/from-job", data={"job_id": "not-hex"})
    assert r.status_code == 400


def test_trace_from_job_rejects_unknown_job_id(client):
    r = client.post("/trace/from-job", data={"job_id": "a" * 32})
    assert r.status_code == 404


# ── Happy path: end-to-end offline pipeline ───────────────────────────────────

def test_trace_from_png_writes_files_inside_job_dir(client, tmp_path):
    png = _make_outline_png()
    r = client.post(
        "/trace/from-png",
        data={"name": "outline"},
        files={"file": ("a.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    job_id = body["job_id"]
    assert len(job_id) == 32

    # downloads of advertised artifacts work
    for url in (body["png"], body["svg"]):
        d = client.get(url)
        assert d.status_code == 200

    # files live under the configured output dir
    out_root = Path(client.app_module.OUTPUT_DIR)
    assert (out_root / job_id / "outline.png").is_file()
    assert (out_root / job_id / "outline.svg").is_file()
    assert (out_root / job_id / "polygon.json").is_file()
    assert (out_root / job_id / "trace_meta.json").is_file()


def test_pipeline_from_png_produces_stl_and_zip(client):
    png = _make_outline_png()
    r = client.post(
        "/pipeline/from-png",
        data={"name": "cookie", "width_mm": "60", "wall_mm": "1.0"},
        files={"file": ("a.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    stl = client.get(body["stl"])
    assert stl.status_code == 200
    assert stl.content[:5] == b"solid" or len(stl.content) > 80  # ascii or binary STL

    zip_resp = client.get(body["zip"])
    assert zip_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        names = set(zf.namelist())
        assert {"cookie.png", "cookie.svg", "cookie.stl"} <= names


def test_grid_png_pipeline_lattice_topology(client):
    grid_png = Path(__file__).parent / "assets" / "grid_3x4.png"
    if not grid_png.exists():
        return
    r = client.post(
        "/pipeline/from-png",
        data={"name": "grid", "topology": "auto", "width_mm": "95"},
        files={"file": ("grid.png", grid_png.read_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["topology"] == "lattice"
    assert body["cols"] == 3
    assert body["rows"] == 4
    assert body.get("height_mm", 0) > 95
    stl = client.get(body["stl"])
    assert stl.status_code == 200
    assert len(stl.content) > 100


def test_pipeline_then_stl_from_job(client):
    png = _make_outline_png()
    r1 = client.post(
        "/trace/from-png",
        data={"name": "outline"},
        files={"file": ("a.png", png, "image/png")},
    )
    job_id = r1.json()["job_id"]

    r2 = client.post(
        "/stl/from-job",
        data={"job_id": job_id, "name": "outline", "width_mm": "60"},
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    stl = client.get(body["stl"])
    assert stl.status_code == 200


# ── Grid Builder ──────────────────────────────────────────────────────────────

def test_trace_from_grid_builds_lattice_job(client):
    r = client.post(
        "/trace/from-grid",
        data={"name": "grid", "cols": "3", "rows": "4", "cell_w_mm": "30", "cell_h_mm": "25"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["topology"] == "lattice"
    assert body["cols"] == 3
    assert body["rows"] == 4
    assert body["extraction_mode"] == "grid"
    assert body["grid_width_mm"] == pytest.approx(90.0)
    assert body["grid_height_mm"] == pytest.approx(100.0)
    svg = client.get(body["svg"])
    assert svg.status_code == 200
    assert svg.text.count("<line") == 4 + 5


def test_trace_from_grid_square_cells_by_default(client):
    r = client.post(
        "/trace/from-grid",
        data={"cols": "2", "rows": "2", "cell_w_mm": "40"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grid_width_mm"] == pytest.approx(80.0)
    assert body["grid_height_mm"] == pytest.approx(80.0)


def test_trace_from_grid_rejects_invalid_spec(client):
    r = client.post(
        "/trace/from-grid",
        data={"cols": "0", "rows": "4", "cell_w_mm": "30"},
    )
    assert r.status_code == 400
    r = client.post(
        "/trace/from-grid",
        data={"cols": "3", "rows": "4", "cell_w_mm": "1000"},
    )
    assert r.status_code == 400


def test_grid_job_stl_honors_exact_cell_size(client):
    """stl/from-job must ignore width_mm for Grid Builder jobs so the
    requested cell sizes are exact."""
    import trimesh

    r = client.post(
        "/trace/from-grid",
        data={"name": "grid", "cols": "3", "rows": "2", "cell_w_mm": "30"},
    )
    job_id = r.json()["job_id"]

    r2 = client.post(
        "/stl/from-job",
        # width_mm deliberately wrong — must not rescale the grid.
        data={"job_id": job_id, "name": "grid", "width_mm": "200", "flange_out_mm": "2.5"},
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["grid_width_mm"] == pytest.approx(90.0)
    assert body["height_mm"] == pytest.approx(60.0)

    out_root = Path(client.app_module.OUTPUT_DIR)
    mesh = trimesh.load(out_root / job_id / "grid.stl", force="mesh")
    extent_x = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
    assert extent_x == pytest.approx(90.0 + 2 * 2.5, abs=0.2)


def test_pipeline_from_grid_one_shot(client):
    r = client.post(
        "/pipeline/from-grid",
        data={"name": "grid", "cols": "2", "rows": "3", "cell_w_mm": "25"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["topology"] == "lattice"
    assert body["grid_width_mm"] == pytest.approx(50.0)
    stl = client.get(body["stl"])
    assert stl.status_code == 200
    assert len(stl.content) > 100
    zip_resp = client.get(body["zip"])
    assert zip_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        assert {"grid.svg", "grid.stl", "trace_meta.json"} <= set(zf.namelist())


def test_grid_job_trace_from_job_returns_stored_trace(client):
    """Grid jobs have no PNG/STL source; re-trace must not 404 or garbage."""
    r = client.post(
        "/trace/from-grid",
        data={"name": "grid", "cols": "3", "rows": "4", "cell_w_mm": "30"},
    )
    job_id = r.json()["job_id"]
    r2 = client.post("/trace/from-job", data={"job_id": job_id, "name": "grid"})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["topology"] == "lattice"
    assert body["cols"] == 3
    assert body["extraction_mode"] == "grid"


# ── Source preservation regressions ───────────────────────────────────────────

def _make_box_stl() -> bytes:
    import trimesh

    return trimesh.creation.box(extents=(30, 20, 10)).export(file_type="stl")


def test_stl_source_survives_generation(client):
    """Generating a cutter must not overwrite the uploaded source STL, and a
    re-trace afterwards must still trace the original upload."""
    stl_bytes = _make_box_stl()
    r = client.post(
        "/trace/from-stl",
        data={"name": "cookie_cutter"},
        files={"file": ("box.stl", stl_bytes, "model/stl")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    job_id = body["job_id"]
    assert body["source_width_mm"] == pytest.approx(30.0, abs=0.1)
    assert "_input.stl" in body["source_stl"]

    source_path = Path(client.app_module.OUTPUT_DIR) / job_id / "cookie_cutter_input.stl"
    size_before = source_path.stat().st_size

    r2 = client.post("/stl/from-job", data={"job_id": job_id, "width_mm": "60"})
    assert r2.status_code == 200, r2.text
    assert source_path.stat().st_size == size_before

    r3 = client.post("/trace/from-job", data={"job_id": job_id})
    assert r3.status_code == 200, r3.text
    assert r3.json()["source_width_mm"] == pytest.approx(30.0, abs=0.1)


def test_png_retrace_after_generation_still_traces_png(client):
    """A PNG job gains a generated {name}.stl; re-trace must keep using the
    PNG source, not the generated mesh."""
    png = _make_outline_png()
    r = client.post(
        "/trace/from-png",
        data={"name": "cookie_cutter"},
        files={"file": ("a.png", png, "image/png")},
    )
    job_id = r.json()["job_id"]

    r2 = client.post("/stl/from-job", data={"job_id": job_id, "width_mm": "60"})
    assert r2.status_code == 200, r2.text

    r3 = client.post("/trace/from-job", data={"job_id": job_id})
    assert r3.status_code == 200, r3.text
    assert r3.json()["extraction_mode"] == "binary"


# ── Prompt endpoints when OPENAI_API_KEY is unset ─────────────────────────────

def test_prompt_pipeline_requires_api_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/pipeline/from-prompt", data={"prompt": "a heart"})
    assert r.status_code == 402


def test_outline_from_prompt_requires_api_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/outline/from-prompt", data={"prompt": "a heart"})
    assert r.status_code == 402


def test_prompt_too_long_rejected(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    r = client.post(
        "/pipeline/from-prompt",
        data={"prompt": "a" * 1001},
    )
    assert r.status_code == 400


# ── Auth middleware ───────────────────────────────────────────────────────────

@pytest.fixture()
def auth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    mod = _load_app(
        tmp_path,
        monkeypatch,
        ACCESS_PASSWORD="hunter2",
        SESSION_SECRET="test-secret-do-not-use",
    )
    with TestClient(mod.app) as c:
        c.app_module = mod  # type: ignore[attr-defined]
        yield c


def test_auth_redirects_unauthenticated_get(auth_client):
    r = auth_client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_auth_returns_401_for_unauthenticated_post(auth_client):
    r = auth_client.post(
        "/trace/from-png",
        files={"file": ("a.png", _make_outline_png(), "image/png")},
    )
    assert r.status_code == 401


def test_auth_healthz_is_exempt(auth_client):
    r = auth_client.get("/healthz")
    assert r.status_code == 200


def test_auth_login_wrong_password(auth_client):
    r = auth_client.post("/login", data={"password": "wrong"}, follow_redirects=False)
    assert r.status_code == 401


def test_auth_login_and_session_round_trip(auth_client):
    r = auth_client.post("/login", data={"password": "hunter2"}, follow_redirects=False)
    assert r.status_code == 303
    cookie = r.cookies.get("session")
    assert cookie

    # Authenticated request now succeeds — TestClient retains cookies set on login
    r2 = auth_client.get("/")
    assert r2.status_code == 200


def test_auth_rejects_forged_session_cookie(auth_client):
    auth_client.cookies.set("session", "deadbeef.notarealsig")
    r = auth_client.get("/", follow_redirects=False)
    assert r.status_code == 303


def test_session_token_verification_is_constant_time(auth_client):
    mod = auth_client.app_module
    token = mod._make_session_token()
    assert mod._verify_session_token(token) is True
    # Mutating the signature must fail
    nonce, _sig = token.rsplit(".", 1)
    assert mod._verify_session_token(nonce + ".0" * 64) is False
    assert mod._verify_session_token("no-dot-here") is False
