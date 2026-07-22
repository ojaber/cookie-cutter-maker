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
