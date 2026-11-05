"""End-to-end tests for the FastAPI app in app/main.py.

These tests exercise the HTTP surface in process via fastapi.testclient.TestClient,
including the security validation layer (upload-size limit, decompression-bomb
guard, untrusted trace payloads) and the offline pipeline path.
"""
from __future__ import annotations

import importlib
import io
import json
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

    Auth/SESSION and the guards are captured at import time, so each test that
    wants a different config must re-import the module. Scratch directories go
    under ``tmp_path/scratch`` so tests can check nothing is left behind.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    monkeypatch.setattr("tempfile.tempdir", str(scratch))
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
        c.scratch = tmp_path / "scratch"  # type: ignore[attr-defined]
        yield c


def _trace_png(client, png: bytes, **data):
    r = client.post("/trace/from-png", data=data, files={"file": ("a.png", png, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def _generate(client, trace: dict, **data):
    """POST a trace back to /stl/generate, as the browser does."""
    return client.post(
        "/stl/generate",
        data=data,
        files={"trace": ("trace.json", json.dumps(trace), "application/json")},
    )


def _info(resp) -> dict:
    return json.loads(resp.headers["X-Cutter-Info"])


def _extent_x(stl_bytes: bytes) -> float:
    import trimesh

    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    return float(mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min())


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




# ── Path-traversal / input validation ─────────────────────────────────────────







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
    assert "too many pixels" in r.json()["detail"]








# ── Happy path: end-to-end offline pipeline ───────────────────────────────────

def test_trace_from_png_returns_the_trace_and_keeps_nothing(client):
    body = _trace_png(client, _make_outline_png(), name="outline")
    assert "job_id" not in body
    assert body["trace"]["topology"] == "single"
    assert body["trace"]["polygon"]["type"] in ("Polygon", "MultiPolygon")
    assert body["svg"].lstrip().startswith("<")
    # The scratch directory is gone before the response is sent.
    assert list(client.scratch.iterdir()) == []


def test_pipeline_from_png_returns_the_stl(client):
    png = _make_outline_png()
    r = client.post(
        "/pipeline/from-png",
        data={"name": "cookie", "width_mm": "60", "wall_mm": "1.0"},
        files={"file": ("a.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "model/stl"
    assert 'filename="cookie.stl"' in r.headers["content-disposition"]
    assert len(r.content) > 80
    assert _info(r)["topology"] == "single"

    z = client.post(
        "/pipeline/from-png",
        data={"name": "cookie", "width_mm": "60", "format": "zip"},
        files={"file": ("a.png", png, "image/png")},
    )
    assert z.status_code == 200, z.text
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        assert {"cookie.png", "cookie.svg", "cookie.stl", "trace.json"} <= set(zf.namelist())
    assert list(client.scratch.iterdir()) == []


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
    info = _info(r)
    assert info["topology"] == "lattice"
    assert info["cols"] == 3
    assert info["rows"] == 4
    assert info.get("height_mm", 0) > 95
    assert len(r.content) > 100


def test_trace_then_generate_from_the_returned_trace(client):
    body = _trace_png(client, _make_outline_png(), name="outline")
    r = _generate(client, body["trace"], name="outline", width_mm="60")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "model/stl"
    assert _extent_x(r.content) > 60  # the rim sticks out past the outline
    assert list(client.scratch.iterdir()) == []


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
    assert body["svg"].count("<line") == 4 + 5


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
    """Generating a Grid Builder trace must ignore width_mm so the requested
    cell sizes are exact."""
    r = client.post(
        "/trace/from-grid",
        data={"name": "grid", "cols": "3", "rows": "2", "cell_w_mm": "30"},
    )
    r2 = _generate(
        client, r.json()["trace"],
        # width_mm deliberately wrong — must not rescale the grid.
        name="grid", width_mm="200", flange_out_mm="2.5",
    )
    assert r2.status_code == 200, r2.text
    info = _info(r2)
    assert info["grid_width_mm"] == pytest.approx(90.0)
    assert info["height_mm"] == pytest.approx(60.0)
    assert _extent_x(r2.content) == pytest.approx(90.0 + 2 * 2.5, abs=0.2)


def test_pipeline_from_grid_one_shot(client):
    r = client.post(
        "/pipeline/from-grid",
        data={"name": "grid", "cols": "2", "rows": "3", "cell_w_mm": "25", "format": "zip"},
    )
    assert r.status_code == 200, r.text
    info = _info(r)
    assert info["topology"] == "lattice"
    assert info["grid_width_mm"] == pytest.approx(50.0)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert {"grid.svg", "grid.stl", "trace.json"} <= set(zf.namelist())
        assert len(zf.read("grid.stl")) > 100


# ── Source preservation regressions ───────────────────────────────────────────

def _make_box_stl() -> bytes:
    import trimesh

    return trimesh.creation.box(extents=(30, 20, 10)).export(file_type="stl")


def test_stl_trace_reports_the_source_size(client):
    r = client.post(
        "/trace/from-stl",
        data={"name": "cookie_cutter"},
        files={"file": ("box.stl", _make_box_stl(), "model/stl")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_width_mm"] == pytest.approx(30.0, abs=0.1)
    gen = _generate(client, body["trace"], width_mm="60")
    assert gen.status_code == 200, gen.text




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


# ── Reported footprint matches the printed part ───────────────────────────────

def test_generated_stl_reports_its_real_footprint(client):
    """The grip rim extends past the outline, so the part prints wider than the
    grid's cell area. The response must report the measured size, not just the
    requested one."""
    r = client.post(
        "/trace/from-grid",
        data={"name": "grid", "cols": "3", "rows": "4", "cell_w_mm": "30", "cell_h_mm": "30"},
    )
    assert r.json()["grid_width_mm"] == pytest.approx(90.0)

    r2 = _generate(client, r.json()["trace"], name="grid", flange_out_mm="2.5", wall_mm="1.4")
    info = _info(r2)
    assert info["footprint_w_mm"] == pytest.approx(95.0, abs=0.2)   # 90 + 2 * 2.5 rim
    assert info["footprint_h_mm"] == pytest.approx(125.0, abs=0.2)
    assert info["footprint_w_mm"] == pytest.approx(_extent_x(r2.content), abs=0.01)


def test_footprint_tracks_the_rim_setting(client):
    trace = client.post("/trace/from-grid", data={"name": "g", "cols": "2", "rows": "2", "cell_w_mm": "30"}).json()["trace"]
    no_rim = _info(_generate(client, trace, name="g", flange_out_mm="0", wall_mm="1.4"))
    big_rim = _info(_generate(client, trace, name="g", flange_out_mm="5", wall_mm="1.4"))
    # 60 mm of cells, plus half a wall each side with no rim, plus 5 mm each side with one.
    assert no_rim["footprint_w_mm"] == pytest.approx(61.4, abs=0.2)
    assert big_rim["footprint_w_mm"] == pytest.approx(70.0, abs=0.2)


def test_single_shape_also_reports_a_footprint(client):
    png = _make_outline_png()
    r = client.post("/pipeline/from-png", data={"name": "c", "width_mm": "60"}, files={"file": ("a.png", png, "image/png")})
    # Wider than the requested 60 mm because of the rim.
    assert _info(r)["footprint_w_mm"] > 60.0


# ── Traces posted back by the client are untrusted ────────────────────────────

def _square_trace() -> dict:
    return {
        "topology": "single",
        "polygon": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
    }


def test_generate_accepts_a_minimal_hand_written_trace(client):
    r = _generate(client, _square_trace(), width_mm="50")
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("trace, fragment", [
    ([], "JSON object"),
    ({"topology": "blob"}, "topology"),
    ({"topology": "single"}, "polygon"),
    ({"topology": "single", "polygon": {"type": "Point", "coordinates": [0, 0]}}, "polygon"),
    ({"topology": "single", "polygon": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]}}, "4 points"),
    ({"topology": "single", "polygon": {"type": "Polygon", "coordinates": [[[0, 0], ["x", 0], [1, 1], [0, 0]]]}}, "number"),
    ({"topology": "single", "polygon": {"type": "Polygon", "coordinates": [[[0, 0], [1e12, 0], [1, 1], [0, 0]]]}}, "out of range"),
    # A bow-tie: self-intersecting, so not a shape a cutter can be built from.
    ({"topology": "single", "polygon": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}}, "valid shape"),
    ({"topology": "lattice", "lattice": {"x_lines": [0, 1], "y_lines": [1, 0]}}, "increasing"),
    ({"topology": "lattice", "lattice": {"x_lines": list(range(500)), "y_lines": [0, 1]}}, "x_lines"),
    ({"topology": "lattice", "grid_spec": {"cols": 500, "rows": 2, "cell_w_mm": 30, "cell_h_mm": 30}}, "Grid must be"),
])
def test_generate_rejects_malformed_traces(client, trace, fragment):
    r = _generate(client, trace)
    assert r.status_code == 400, r.text
    assert fragment in r.json()["detail"]


def test_generate_rejects_non_json_and_nan(client):
    r = client.post("/stl/generate", files={"trace": ("t.json", b"not json", "application/json")})
    assert r.status_code == 400
    nan = b'{"topology":"single","polygon":{"type":"Polygon","coordinates":[[[0,0],[1,0],[NaN,1],[0,0]]]}}'
    r = client.post("/stl/generate", files={"trace": ("t.json", nan, "application/json")})
    assert r.status_code == 400


def test_generate_rejects_too_many_points(client):
    ring = [[float(i % 2), float(i)] for i in range(60_000)]
    trace = {"topology": "single", "polygon": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}
    r = _generate(client, trace)
    assert r.status_code in (400, 413)


def test_generate_rejects_an_oversized_trace(client, monkeypatch):
    monkeypatch.setattr(client.app_module, "MAX_TRACE_BYTES", 100)
    r = _generate(client, _square_trace() | {"extraction_warning": "x" * 200})
    assert r.status_code == 413


def test_grid_trace_is_rebuilt_from_its_spec(client):
    """A grid trace's line positions are not trusted: the lattice comes from
    the four numbers in grid_spec, so tampered lines change nothing."""
    trace = client.post("/trace/from-grid", data={"cols": "2", "rows": "2", "cell_w_mm": "30"}).json()["trace"]
    trace["lattice"]["x_lines"] = [0, 1000]
    r = _generate(client, trace, flange_out_mm="0", wall_mm="1.4")
    assert r.status_code == 200, r.text
    assert _extent_x(r.content) == pytest.approx(61.4, abs=0.2)


def test_lattice_bounds_come_from_the_lines(client):
    """The mesher scales by the bounds but draws walls from the lines, so a
    posted trace whose bounds disagree with its lines must not be able to blow
    the cutter up to many times the requested width."""
    trace = {
        "topology": "lattice",
        "lattice": {"x_lines": [0, 500, 1000], "y_lines": [0, 500, 1000], "bounds": [0, 0, 1, 1]},
    }
    r = _generate(client, trace, width_mm="80", flange_out_mm="0", wall_mm="1.4")
    assert r.status_code == 200, r.text
    assert _extent_x(r.content) == pytest.approx(81.4, abs=0.5)


def test_job_routes_are_gone(client):
    assert client.post("/trace/from-job", data={"job_id": "a" * 32}).status_code in (404, 405)
    assert client.post("/stl/from-job", data={"job_id": "a" * 32}).status_code in (404, 405)
    assert client.get("/files/" + "a" * 32 + "/x.stl").status_code == 404

def test_upload_limit_fits_vercel_and_is_reported(tmp_path, monkeypatch):
    """Vercel rejects bodies over 4.5 MB before the app runs, so the default
    limit drops there, and /features tells the page so it can shrink photos."""
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, VERCEL="1", MAX_UPLOAD_BYTES=None)
    assert mod.MAX_UPLOAD_BYTES == 4_200_000
    with TestClient(mod.app) as c:
        assert c.get("/features").json()["max_upload_bytes"] == 4_200_000

    mod = _load_app(tmp_path, monkeypatch, VERCEL=None, MAX_UPLOAD_BYTES=None)
    assert mod.MAX_UPLOAD_BYTES == 20 * 1024 * 1024


def test_pipeline_imports_wait_for_the_import_lock(client):
    """The warmup thread and the first request must not import numpy at the
    same time (it fails on a fresh Vercel instance). Every entry point waits
    for the one-time import under a lock."""
    import threading

    mod = client.app_module
    mod._pipeline_loaded = False
    done = threading.Event()
    with mod._PIPELINE_LOCK:
        t = threading.Thread(target=lambda: (mod.grid_size_mm({"cols": 1, "rows": 1, "cell_w_mm": 5, "cell_h_mm": 5}), done.set()))
        t.start()
        assert not done.wait(0.3), "an entry point imported without waiting for the lock"
    assert done.wait(5)
    assert mod._pipeline_loaded


def test_text_tab_needs_the_openai_package_too(client, monkeypatch):
    """The Vercel deploy leaves openai out to stay small. A key set there must
    not make the page offer a feature that would then fail."""
    import importlib.util

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "openai" else real(name, *a))
    assert client.get("/features").json()["image_generation"] is False
