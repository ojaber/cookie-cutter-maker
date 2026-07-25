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

def test_robots_txt_disallows_job_files(client):
    r = client.get("/robots.txt")
    assert r.status_code == 200
    assert "Disallow: /files/" in r.text


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


# ── Job artifacts must not be cached ──────────────────────────────────────────

def test_job_files_are_revalidated(client):
    """{name}.svg and {name}.stl are rewritten in place by a re-trace or a
    regenerate. Without a no-cache header the browser keeps showing the
    previous result, which looks exactly like the settings having no effect."""
    r = client.post("/trace/from-grid", data={"name": "g", "cols": "2", "rows": "2", "cell_w_mm": "30"})
    svg_url = r.json()["svg"]
    resp = client.get(svg_url)
    assert resp.status_code == 200
    assert "no-cache" in resp.headers.get("Cache-Control", "").lower()


def test_retrace_serves_the_new_content(client):
    """End to end: a different simplify tolerance must reach the served SVG."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("L", (256, 256), color=255)
    ImageDraw.Draw(img).ellipse([40, 40, 216, 216], fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    r = client.post(
        "/trace/from-png",
        data={"name": "c", "simplify": "0.02"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    job_id = r.json()["job_id"]
    coarse = client.get(r.json()["svg"]).text

    r2 = client.post("/trace/from-job", data={"job_id": job_id, "name": "c", "simplify": "0.0002"})
    fine = client.get(r2.json()["svg"]).text

    # A circle traced finely has many more segments than one traced coarsely.
    assert fine.count(" L ") > coarse.count(" L ") * 2, (coarse.count(" L "), fine.count(" L "))
