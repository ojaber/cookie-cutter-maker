"""Tests for the public-instance guards: rate limiting, heavy-endpoint
concurrency cap, security headers, and robots.txt."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.test_api import _load_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch)
    with TestClient(mod.app) as c:
        c.app_module = mod  # type: ignore[attr-defined]
        yield c


# ── Security headers ──────────────────────────────────────────────────────────

def test_security_headers_on_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "Referrer-Policy" in r.headers
    assert "Content-Security-Policy" in r.headers
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]


def test_docs_exempt_from_csp_but_not_other_headers(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "Content-Security-Policy" not in r.headers
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_csp_allows_viewer_cdn_fallback(client):
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "https://unpkg.com" in csp
    assert "img-src 'self' data: blob:" in csp


def test_csp_lets_the_viewer_read_held_files(client):
    # The 3D viewer fetches the STL the browser holds through a blob: URL.
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "connect-src 'self' blob:" in csp


def test_default_frame_policy_is_self_only(client):
    r = client.get("/")
    assert r.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in r.headers["Content-Security-Policy"]


def test_frame_ancestors_env_override(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, FRAME_ANCESTORS="'self' https://example.com")
    with TestClient(mod.app) as c:
        r = c.get("/")
        assert "frame-ancestors 'self' https://example.com" in r.headers["Content-Security-Policy"]
        assert "X-Frame-Options" not in r.headers


# ── robots.txt ────────────────────────────────────────────────────────────────

def test_robots_txt_disallows_the_api(client):
    r = client.get("/robots.txt")
    assert r.status_code == 200
    assert "Disallow: /stl/" in r.text


def test_robots_txt_exempt_from_auth(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(
        tmp_path,
        monkeypatch,
        ACCESS_PASSWORD="hunter2",
        SESSION_SECRET="test-secret-do-not-use",
    )
    with TestClient(mod.app) as c:
        r = c.get("/robots.txt", follow_redirects=False)
        assert r.status_code == 200


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_rate_limit_kicks_in_on_heavy_posts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, RATE_LIMIT_PER_MINUTE="2")
    with TestClient(mod.app) as c:
        data = {"name": "grid", "cols": "2", "rows": "2", "cell_w_mm": "30"}
        assert c.post("/trace/from-grid", data=data).status_code == 200
        assert c.post("/trace/from-grid", data=data).status_code == 200
        r = c.post("/trace/from-grid", data=data)
        assert r.status_code == 429
        assert r.headers["Retry-After"] == "60"


def test_rate_limit_does_not_affect_light_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, RATE_LIMIT_PER_MINUTE="1")
    with TestClient(mod.app) as c:
        for _ in range(5):
            assert c.get("/features").status_code == 200
        assert c.get("/healthz").status_code == 200


def test_rate_limit_disabled_when_zero(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, RATE_LIMIT_PER_MINUTE="0")
    with TestClient(mod.app) as c:
        data = {"name": "grid", "cols": "2", "rows": "2", "cell_w_mm": "30"}
        for _ in range(4):
            assert c.post("/trace/from-grid", data=data).status_code == 200


def test_rate_limit_tracks_forwarded_client_ip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, RATE_LIMIT_PER_MINUTE="1")
    with TestClient(mod.app) as c:
        data = {"name": "grid", "cols": "2", "rows": "2", "cell_w_mm": "30"}
        assert c.post("/trace/from-grid", data=data, headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 200
        assert c.post("/trace/from-grid", data=data, headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 429
        # A different client is not affected by the first one's bucket.
        assert c.post("/trace/from-grid", data=data, headers={"X-Forwarded-For": "5.6.7.8"}).status_code == 200


# ── Concurrency cap ───────────────────────────────────────────────────────────

def test_busy_server_returns_503(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, HEAVY_QUEUE_TIMEOUT_SECONDS="0.1")
    # All slots taken: a fresh zero-permit semaphore never becomes available.
    mod._heavy_semaphore = asyncio.Semaphore(0)
    with TestClient(mod.app) as c:
        r = c.post(
            "/trace/from-grid",
            data={"name": "grid", "cols": "2", "rows": "2", "cell_w_mm": "30"},
        )
        assert r.status_code == 503
        assert "busy" in r.json()["detail"].lower()
        assert r.headers["Retry-After"] == "15"


def test_concurrency_guard_disabled_when_zero(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_app(tmp_path, monkeypatch, HEAVY_CONCURRENCY="0", HEAVY_QUEUE_TIMEOUT_SECONDS="0.1")
    mod._heavy_semaphore = asyncio.Semaphore(0)  # would 503 if the guard ran
    with TestClient(mod.app) as c:
        r = c.post(
            "/trace/from-grid",
            data={"name": "grid", "cols": "2", "rows": "2", "cell_w_mm": "30"},
        )
        assert r.status_code == 200


# ── Re-tracing ────────────────────────────────────────────────────────────────

def test_retrace_returns_the_new_outline(client):
    """Re-tracing is the same upload with new settings: a different simplify
    tolerance must reach the SVG that comes back."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("L", (256, 256), color=255)
    ImageDraw.Draw(img).ellipse([40, 40, 216, 216], fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    def trace(simplify):
        r = client.post(
            "/trace/from-png",
            data={"name": "c", "simplify": simplify},
            files={"file": ("a.png", buf.getvalue(), "image/png")},
        )
        assert r.status_code == 200, r.text
        return r.json()["svg"]

    coarse, fine = trace("0.02"), trace("0.0002")
    # A circle traced finely has many more segments than one traced coarsely.
    assert fine.count(" L ") > coarse.count(" L ") * 2, (coarse.count(" L "), fine.count(" L "))


def test_generated_cutter_is_not_cached(client):
    import json

    trace = client.post("/trace/from-grid", data={"name": "g", "cols": "2", "rows": "2", "cell_w_mm": "30"}).json()["trace"]
    r = client.post("/stl/generate", files={"trace": ("t.json", json.dumps(trace), "application/json")})
    assert r.status_code == 200
    assert "no-store" in r.headers.get("Cache-Control", "").lower()
