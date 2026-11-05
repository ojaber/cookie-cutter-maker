import asyncio
import base64
import contextlib
import functools
import hashlib
import hmac
import os
import re
import io
import secrets
import sys
import json
import tempfile
import threading
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import anyio

from PIL import Image as _PILImage
# Cap pixel count to defuse decompression bombs before PIL allocates a giant array.
# Default PIL limit is ~89 MP — tighten to ~25 MP (≈5000×5000), which is more than
# enough for a cookie-cutter outline.
_PILImage.MAX_IMAGE_PIXELS = 25_000_000
import zipfile

_log = logging.getLogger(__name__)

# ── Lazy pipeline imports ─────────────────────────────────────────────────────
# numpy/scipy/scikit-image/trimesh take several seconds to import from cold
# storage. Importing them at module load blocks uvicorn from binding the port,
# so a host that sleeps when idle (e.g. Render's free tier) serves nothing at
# all until they finish. These wrappers defer each import to first call —
# python caches it in sys.modules, so the cost is paid once — and
# _warm_imports() preloads them in the background once the server is listening.
# Call sites stay unchanged.
#
# Every entry point goes through _load_pipeline(), which imports the stack once
# under a lock. Without it the warmup thread and the first request import
# numpy at the same moment on a fresh instance, and numpy fails when two
# threads initialise it together ("partially initialized module"). Render's
# long-lived server rarely hit that; Vercel starts a new instance per burst and
# hit it on the first request.

_PIPELINE_LOCK = threading.Lock()
_pipeline_loaded = False


def _load_pipeline() -> None:
    global _pipeline_loaded
    if _pipeline_loaded:
        return
    with _PIPELINE_LOCK:
        if _pipeline_loaded:
            return
        import cutter_pipeline.trace_outline  # noqa: F401
        import cutter_pipeline.stl_dispatch  # noqa: F401
        import cutter_pipeline.stl_extractor  # noqa: F401
        import cutter_pipeline.grid_spec  # noqa: F401
        import cutter_pipeline.trace_meta  # noqa: F401
        import cutter_pipeline.image_extractor  # noqa: F401
        import trimesh  # noqa: F401
        _pipeline_loaded = True


def trace_png_to_polygon(*args, **kwargs):
    _load_pipeline()
    from cutter_pipeline.trace_outline import trace_png_to_polygon as _f
    return _f(*args, **kwargs)


def build_grid_trace(*args, **kwargs):
    _load_pipeline()
    from cutter_pipeline.grid_spec import build_grid_trace as _f
    return _f(*args, **kwargs)


def grid_size_mm(*args, **kwargs):
    _load_pipeline()
    from cutter_pipeline.grid_spec import grid_size_mm as _f
    return _f(*args, **kwargs)


def generate_stl_from_trace(*args, **kwargs):
    _load_pipeline()
    from cutter_pipeline.stl_dispatch import generate_stl_from_trace as _f
    return _f(*args, **kwargs)


def trace_result_to_dict(*args, **kwargs):
    _load_pipeline()
    from cutter_pipeline.trace_meta import trace_result_to_dict as _f
    return _f(*args, **kwargs)


def trace_result_from_payload(*args, **kwargs):
    _load_pipeline()
    from cutter_pipeline.trace_meta import trace_result_from_payload as _f
    return _f(*args, **kwargs)


def extract_outline_from_stl(*args, **kwargs):
    _load_pipeline()
    from cutter_pipeline.stl_extractor import extract_outline_from_stl as _f
    return _f(*args, **kwargs)


def _openai_error_cls() -> type[BaseException]:
    """The OpenAIError class, or a sentinel that never matches when the
    package is unavailable, so `except _openai_error_cls()` stays valid."""
    try:
        from openai import OpenAIError
        return OpenAIError
    except Exception:
        class _NeverRaised(Exception):
            pass
        return _NeverRaised


def _generate_outline_png(prompt: str, out_path: str) -> None:
    """Prompt -> outline PNG. Requires the optional OpenAI dependency."""
    try:
        from cutter_pipeline.outline_openai import generate_outline_png as _f
    except Exception:
        raise HTTPException(status_code=500, detail="OpenAI image step unavailable.")
    _f(prompt, out_path)


def _warm_imports() -> None:
    """Preload the heavy geometry stack so the first real request doesn't wait
    on it. Runs in a worker thread after the server is already accepting
    connections, so it never delays startup."""
    started = time.perf_counter()
    try:
        _load_pipeline()
    except Exception:
        # Not fatal: whichever import failed will be retried on first use and
        # surface a real error to that request.
        _log.exception("Background warmup failed; imports will happen on first use.")
        return
    _log.info("Pipeline warmup finished in %.2fs.", time.perf_counter() - started)


def _env_number(name: str, default: float, cast=int):
    """Parse a numeric env var, tolerating unset/empty/garbage values."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return cast(default)
    try:
        return cast(raw)
    except ValueError:
        _log.warning("Ignoring invalid %s=%r — using default %s.", name, raw, default)
        return cast(default)

# ── Access control ─────────────────────────────────────────────────────────────
# Auth is enabled only when ACCESS_PASSWORD is set. If not set, the app is open.

ACCESS_PASSWORD: str = os.environ.get("ACCESS_PASSWORD", "").strip()

# Session signing secret — independent of the password.
# Prefer SESSION_SECRET env var (required for stable sessions across replicas/restarts).
# Falls back to a random value: sessions will be invalidated on every restart.
_SESSION_SECRET: str = ""
if ACCESS_PASSWORD:
    _SESSION_SECRET = os.environ.get("SESSION_SECRET", "").strip()
    if not _SESSION_SECRET:
        _SESSION_SECRET = secrets.token_hex(32)
        _log.warning(
            "SESSION_SECRET env var not set — using a randomly generated secret. "
            "Sessions will be invalidated on restart and will not work across "
            "multiple replicas. Set SESSION_SECRET for stable sessions."
        )

_AUTH_EXEMPT = {"/login", "/logout", "/healthz", "/favicon.ico", "/robots.txt"}

def _make_session_token() -> str:
    nonce = secrets.token_hex(16)
    sig = hmac.new(_SESSION_SECRET.encode(), nonce.encode(), hashlib.sha256).hexdigest()
    return f"{nonce}.{sig}"

def _verify_session_token(token: str) -> bool:
    try:
        nonce, sig = token.rsplit(".", 1)
        expected = hmac.new(_SESSION_SECRET.encode(), nonce.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

def _login_page(error: str = "") -> str:
    error_html = f'<p class="error">{error}</p>' if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Cookie Cutter Maker — Login</title>
  <meta name="color-scheme" content="light dark"/>
  <link rel="icon" type="image/png" href="/favicon.ico"/>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    :root{{--bg:#f7f4ee;--surface:#fff;--border:#e6e0d2;--text:#241f17;--muted:#5d5647;--accent:#b45309;--accent-hover:#94430a;--err:#b03030}}
    @media (prefers-color-scheme: dark){{
      :root{{--bg:#17130f;--surface:#211c16;--border:#373025;--text:#f1eadd;--muted:#b6ac9a;--accent:#e79552;--accent-hover:#f0a765;--err:#f28b82}}
    }}
    body{{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px}}
    .card{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:2rem;width:100%;max-width:380px;box-shadow:0 10px 30px rgba(46,35,15,.08)}}
    h1{{font-size:1.25rem;margin-bottom:.25rem}}
    .sub{{font-size:.85rem;color:var(--muted);margin-bottom:1.5rem}}
    label{{display:block;font-size:.85rem;font-weight:600;color:var(--muted);margin-bottom:.4rem}}
    input[type=password]{{width:100%;padding:.65rem .75rem;background:var(--bg);border:1px solid var(--border);border-radius:10px;color:var(--text);font-size:1rem;margin-bottom:1rem}}
    input[type=password]:focus{{outline:none;border-color:var(--accent)}}
    button{{width:100%;padding:.7rem;background:var(--accent);color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer}}
    @media (prefers-color-scheme: dark){{button{{color:#271303}}}}
    button:hover{{background:var(--accent-hover)}}
    .error{{color:var(--err);font-size:.85rem;margin-bottom:1rem}}
  </style>
</head>
<body>
  <div class="card">
    <h1>🍪 Cookie Cutter Maker</h1>
    <p class="sub">Enter the passphrase to continue.</p>
    {error_html}
    <form method="POST" action="/login">
      <label for="pw">Passphrase</label>
      <input type="password" id="pw" name="password" autofocus autocomplete="current-password"/>
      <button type="submit">Sign in</button>
    </form>
  </div>
</body>
</html>"""

_log.info(
    "Starting Cookie Cutter Maker — REMBG_ENABLED=%s OPENAI=%s AUTH=%s",
    os.environ.get("REMBG_ENABLED", "unset (default true)"),
    "yes" if os.environ.get("OPENAI_API_KEY") else "no",
    "enabled" if ACCESS_PASSWORD else "disabled (ACCESS_PASSWORD not set)",
)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fire-and-forget: the port is already bound by the time this runs.
    warmup = asyncio.create_task(anyio.to_thread.run_sync(_warm_imports))
    yield
    warmup.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await warmup


app = FastAPI(title="Cookie Cutter Maker", version="0.2.0", lifespan=_lifespan)


async def _run(func, *args, **kwargs):
    """Run blocking pipeline work on a worker thread so the event loop stays
    responsive (health checks, other requests) during heavy tracing/meshing."""
    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))

# Reject uploads larger than this to avoid disk/memory exhaustion.
# Vercel Functions reject request bodies over 4.5 MB before the app sees them,
# so default lower there (Vercel sets VERCEL=1) and let the page know: it
# shrinks big photos to fit before uploading. Elsewhere allow 20 MB.
_ON_VERCEL = bool(os.environ.get("VERCEL"))
MAX_UPLOAD_BYTES = _env_number(
    "MAX_UPLOAD_BYTES", 4_200_000 if _ON_VERCEL else 20 * 1024 * 1024
)

# ── Public-instance guards ─────────────────────────────────────────────────────
# Tracing and meshing are CPU/memory heavy; on a small public instance a burst
# of requests (or a bot) can OOM the box. Two in-memory, per-process guards:
#   * HEAVY_CONCURRENCY   — max pipeline jobs running at once (0 disables).
#   * RATE_LIMIT_PER_MINUTE — max heavy POSTs per client IP per minute
#     (0 disables).
# Suitable for the default single-instance deployment; front with a real rate
# limiter if you ever scale out.
HEAVY_CONCURRENCY = _env_number("HEAVY_CONCURRENCY", 2)
HEAVY_QUEUE_TIMEOUT_SECONDS = _env_number("HEAVY_QUEUE_TIMEOUT_SECONDS", 30, cast=float)
RATE_LIMIT_PER_MINUTE = _env_number("RATE_LIMIT_PER_MINUTE", 20)

_HEAVY_PATHS = {
    "/trace/from-png",
    "/trace/from-stl",
    "/trace/from-grid",
    "/stl/generate",
    "/pipeline/from-png",
    "/pipeline/from-stl",
    "/pipeline/from-grid",
    "/pipeline/from-prompt",
    "/outline/from-prompt",
}

_heavy_semaphore = asyncio.Semaphore(max(HEAVY_CONCURRENCY, 1))
_RATE_WINDOW_SECONDS = 60.0
_rate_buckets: dict[str, deque] = {}


def _client_ip(request: Request) -> str:
    # App Platform / most reverse proxies put the real client first in
    # X-Forwarded-For. Fall back to the socket peer for direct connections.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_RATE_BUCKET_LIMIT = 5000


def _rate_limited(ip: str) -> bool:
    """Sliding-window rate check. Returns True when the request should be rejected."""
    now = time.monotonic()
    bucket = _rate_buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > _RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        return True
    bucket.append(now)
    if len(_rate_buckets) > _RATE_BUCKET_LIMIT:
        # Drop buckets whose newest hit has aged out of the window. Pruning on
        # emptiness alone would leak: a bucket is only emptied when that same
        # IP comes back, so one-off callers would linger forever.
        stale = [
            key for key, b in _rate_buckets.items()
            if not b or now - b[-1] > _RATE_WINDOW_SECONDS
        ]
        for key in stale:
            del _rate_buckets[key]
    return False

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
def _safe_name(name: str) -> str:
    """Validate a user-supplied base filename used to build output paths."""
    name = (name or "").strip()
    if not _SAFE_NAME_RE.match(name) or ".." in name:
        raise HTTPException(
            status_code=400,
            detail="Invalid name: use letters, digits, '.', '_' or '-' only (max 64 chars).",
        )
    return name


async def _read_upload(file: UploadFile) -> bytes:
    """Read an UploadFile into memory while enforcing MAX_UPLOAD_BYTES."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _verify_image_bytes(content: bytes) -> None:
    """Confirm the upload is actually a decodable image before we run the pipeline on it."""
    from io import BytesIO
    try:
        with _PILImage.open(BytesIO(content)) as img:
            img.verify()
    except _PILImage.DecompressionBombError:
        limit_mp = _PILImage.MAX_IMAGE_PIXELS * 2 // 1_000_000
        raise HTTPException(
            status_code=400,
            detail=f"That image has too many pixels (over {limit_mp} megapixels). "
            "Resize it to a few thousand pixels across and try again.",
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


def _verify_stl_bytes(content: bytes) -> None:
    """Confirm the upload is a valid STL file before we run the pipeline on it."""
    _load_pipeline()
    import trimesh
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            mesh = trimesh.load(tmp.name, force="mesh")
            if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                raise HTTPException(status_code=400, detail="STL file is empty or invalid.")
            if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
                raise HTTPException(status_code=400, detail="STL file has no faces.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid STL.")


def _measure_stl_xy_mm(stl_path: Path) -> tuple[float, float]:
    _load_pipeline()
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("STL file is empty or invalid.")
    vertices_2d = mesh.vertices[:, :2]
    mins = vertices_2d.min(axis=0)
    maxs = vertices_2d.max(axis=0)
    return float(maxs[0] - mins[0]), float(maxs[1] - mins[1])


async def _add_stl_size_fields(result: dict[str, Any], stl_path: Path) -> None:
    try:
        source_width_mm, source_height_mm = await _run(_measure_stl_xy_mm, stl_path)
    except Exception:
        return
    result["source_width_mm"] = source_width_mm
    result["source_height_mm"] = source_height_mm

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)
OPENAI_INFLIGHT = Gauge(
    "openai_generate_inflight",
    "Number of in-flight OpenAI image generation calls",
)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    # Skip metrics endpoint itself to avoid recursion.
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        method = request.method
        REQUEST_COUNT.labels(method=method, path=path, status=status_code).inc()
        REQUEST_LATENCY.labels(method=method, path=path, status=status_code).observe(elapsed)


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def _public_guard_middleware(request: Request, call_next):
    """Rate-limit and concurrency-cap the heavy pipeline endpoints."""
    if request.method != "POST" or request.url.path not in _HEAVY_PATHS:
        return await call_next(request)
    if RATE_LIMIT_PER_MINUTE > 0 and _rate_limited(_client_ip(request)):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests — please wait a minute and try again."},
            headers={"Retry-After": "60"},
        )
    if HEAVY_CONCURRENCY <= 0:
        return await call_next(request)
    try:
        await asyncio.wait_for(_heavy_semaphore.acquire(), timeout=HEAVY_QUEUE_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError):
        return JSONResponse(
            status_code=503,
            content={"detail": "The server is busy right now — please try again in a moment."},
            headers={"Retry-After": "15"},
        )
    try:
        return await call_next(request)
    finally:
        _heavy_semaphore.release()


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if not ACCESS_PASSWORD:
        return await call_next(request)
    if request.url.path in _AUTH_EXEMPT:
        return await call_next(request)
    token = request.cookies.get("session")
    if not token or not _verify_session_token(token):
        if request.method == "GET":
            return RedirectResponse(url="/login", status_code=303)
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


# Who may embed the app in an iframe (CSP frame-ancestors). Locked to the same
# origin by default; set FRAME_ANCESTORS to allow another site, for example
# "'self' https://example.com".
FRAME_ANCESTORS = os.environ.get("FRAME_ANCESTORS", "").strip() or "'self'"

# The UI is a single inline-scripted page, so 'unsafe-inline' is required;
# unpkg.com is the CDN fallback for the three.js viewer. Swagger UI (/docs,
# /redoc) loads its bundle from jsdelivr, so those paths skip the CSP only.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://unpkg.com; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    # blob: lets the 3D viewer read the STL the browser is holding.
    "connect-src 'self' blob:; "
    "font-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    f"frame-ancestors {FRAME_ANCESTORS}"
)
_CSP_EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def _security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    if FRAME_ANCESTORS == "'self'":
        # X-Frame-Options cannot express an allow-list; when embedding is
        # opened up, the CSP frame-ancestors directive alone governs it.
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.url.path not in _CSP_EXEMPT_PATHS:
        response.headers.setdefault("Content-Security-Policy", _CSP)
    return response


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    """Return the real error message to the client while logging the stack."""
    logging.exception("Unhandled error during %s %s", request.method, request.url, exc_info=exc)
    detail = str(exc).strip() or exc.__class__.__name__
    return JSONResponse(status_code=500, content={"detail": detail})

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png")

@app.get("/login", include_in_schema=False)
def login_page():
    return HTMLResponse(_login_page())

@app.post("/login", include_in_schema=False)
async def login_submit(request: Request, password: str = Form(default="")):
    if password.strip() and hmac.compare_digest(password.strip(), ACCESS_PASSWORD):
        token = _make_session_token()
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie("session", token, httponly=True, samesite="lax", secure=True, max_age=60 * 60 * 24 * 7)
        return response
    await asyncio.sleep(1)  # slow down brute-force attempts
    error = "Incorrect passphrase. Please try again."
    return HTMLResponse(_login_page(error=error), status_code=401)

@app.get("/logout", include_in_schema=False)
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session")
    return response

@app.get("/healthz", include_in_schema=False)
def healthz():
    return Response(content="OK", media_type="text/plain")

@app.get("/robots.txt", include_in_schema=False)
def robots():
    # Nothing here is worth indexing beyond the page itself, and the POST
    # endpoints would only be noise.
    return Response(
        content="User-agent: *\nDisallow: /trace/\nDisallow: /stl/\nDisallow: /pipeline/\n",
        media_type="text/plain",
    )

@app.get("/features", include_in_schema=False)
def features():
    _load_pipeline()
    from cutter_pipeline.image_extractor import REMBG_ENABLED
    return {
        "background_removal": REMBG_ENABLED,
        "image_generation": bool(os.environ.get("OPENAI_API_KEY")),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }

# ── Stateless request handling ────────────────────────────────────────────────
# Nothing is kept between requests. Each one gets a scratch directory that is
# deleted before the response goes out; whatever the next step needs (the
# traced outline, the preview SVG, the generated STL) travels in the response
# and the browser posts it back when it wants more work done. The pipeline
# functions take file paths, which is the only reason a directory exists.

@contextlib.contextmanager
def _scratch_dir():
    with tempfile.TemporaryDirectory(prefix="ccm-") as tmp:
        yield Path(tmp)


# A trace is ~3 KB for a typical outline; this leaves room for very detailed
# ones while keeping a hostile client from posting something enormous.
MAX_TRACE_BYTES = _env_number("MAX_TRACE_BYTES", 4 * 1024 * 1024)


def _topology_fields(traced) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "topology": traced.topology,
        "topology_requested": traced.topology_requested,
        "topology_detected": traced.topology_detected,
        "contour_count": traced.contour_count,
        "cols": traced.cols,
        "rows": traced.rows,
    }
    if traced.grid_hint:
        fields["grid_hint"] = traced.grid_hint
    if traced.grid_spec:
        grid_w, grid_h = grid_size_mm(traced.grid_spec)
        fields["grid_spec"] = traced.grid_spec
        fields["grid_width_mm"] = grid_w
        fields["grid_height_mm"] = grid_h
    return fields


def _trace_response(traced, svg_path: Path) -> dict[str, Any]:
    """What a trace endpoint returns: the outline itself (posted back to
    /stl/generate), its SVG preview inline, and the topology summary."""
    result: dict[str, Any] = {
        "trace": trace_result_to_dict(traced),
        "svg": svg_path.read_text(encoding="utf-8") if svg_path.is_file() else "",
        "extraction_mode": traced.extraction_mode,
        **_topology_fields(traced),
    }
    if traced.extraction_warning:
        result["warning"] = traced.extraction_warning
    return result


def _log_image_upload(filename: str, content: bytes) -> None:
    from io import BytesIO
    try:
        with _PILImage.open(BytesIO(content)) as img:
            w, h = img.size
        _log.info("Image upload — file=%r size=%.1fKB dimensions=%dx%d", filename, len(content) / 1024, w, h)
    except Exception:
        _log.info("Image upload — file=%r size=%.1fKB", filename, len(content) / 1024)


def _openai_detail(exc: BaseException) -> str:
    """Prefer the nested OpenAI error message if available."""
    # Newer openai client exposes .body with {'error': {'message': ...}}
    body: Any = getattr(exc, "body", None)
    if isinstance(body, dict):
        msg = body.get("error", {}).get("message") or body.get("message")
        if msg:
            return str(msg)
    msg = getattr(exc, "message", None)
    if msg:
        return str(msg)
    return str(exc).strip() or exc.__class__.__name__


async def _prompt_to_png(prompt: str, png_path: Path) -> None:
    if len(prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt must be 1000 characters or fewer.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=402, detail="OPENAI_API_KEY not set. Upload an image instead.")
    _log.info("Prompt — %r", prompt.strip()[:500])
    OPENAI_INFLIGHT.inc()
    try:
        await anyio.to_thread.run_sync(_generate_outline_png, prompt, str(png_path))
    except _openai_error_cls() as e:
        status = getattr(e, "status_code", 500) or 500
        detail = _openai_detail(e)
        logging.warning(
            "OpenAI image generation failed (status=%s, prompt=%s): %s",
            status, prompt.strip()[:200], detail,
        )
        raise HTTPException(status_code=status, detail=detail)
    finally:
        OPENAI_INFLIGHT.dec()


# ── Shared form parameters ────────────────────────────────────────────────────

class TraceSettings:
    """Outline settings shared by every image-tracing endpoint."""

    def __init__(
        self,
        threshold: int = Form(200),
        simplify: float = Form(0.0008),
        smooth_radius: float = Form(1.0),
        extraction_mode: str = Form("auto"),
        delta_e_threshold: float = Form(28.0),
        topology: str = Form("auto"),
    ):
        self.threshold = threshold
        self.simplify = simplify
        self.smooth_radius = smooth_radius
        self.extraction_mode = extraction_mode
        self.delta_e_threshold = delta_e_threshold
        self.topology = topology

    def png_kwargs(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "simplify_epsilon": self.simplify,
            "smooth_radius": self.smooth_radius,
            "extraction_mode": self.extraction_mode,
            "delta_e_threshold": self.delta_e_threshold,
            "topology": self.topology,
        }

    def stl_kwargs(self) -> dict[str, Any]:
        return {"simplify_epsilon": self.simplify, "topology": self.topology}


class CutterSettings:
    """Physical cutter settings shared by every endpoint that builds an STL."""

    def __init__(
        self,
        width_mm: float = Form(95.0),
        wall_mm: float = Form(1.4),
        total_h_mm: float = Form(15.0),
        flange_h_mm: float = Form(3.5),
        flange_out_mm: float = Form(2.5),
        flange_chamfer_mm: float = Form(0.5),
        flange_all_lines: bool = Form(False),
        flange_corner_radius_mm: float = Form(1.5),
        bottom_wall_mm: float = Form(0.1),
        cutting_wall_h_mm: float = Form(2.0),
        cleanup_mm: float = Form(0.5),
        tip_smooth_mm: float = Form(0.6),
        keep_holes: bool = Form(False),
        min_component_area_mm2: float = Form(25.0),
    ):
        self.width_mm = width_mm
        self.kwargs = {
            "wall_mm": wall_mm,
            "total_h_mm": total_h_mm,
            "flange_h_mm": flange_h_mm,
            "flange_out_mm": flange_out_mm,
            "flange_chamfer_mm": flange_chamfer_mm,
            "flange_all_lines": flange_all_lines,
            "flange_corner_radius_mm": flange_corner_radius_mm,
            "bottom_wall_mm": bottom_wall_mm,
            "cutting_wall_h_mm": cutting_wall_h_mm,
            "cleanup_mm": cleanup_mm,
            "tip_smooth_mm": tip_smooth_mm,
            "drop_holes": not keep_holes,
            "min_component_area_mm2": min_component_area_mm2,
        }

    def target_width_mm(self, traced) -> float:
        # Grid Builder traces carry exact millimetre geometry — ignore the
        # generic width so the requested cell sizes are honoured precisely.
        if traced.grid_spec:
            return grid_size_mm(traced.grid_spec)[0]
        return self.width_mm


# ── Building and returning a cutter ───────────────────────────────────────────

async def _build_stl(traced, settings: CutterSettings, work: Path, name: str) -> tuple[Path, dict]:
    stl_path = work / f"{name}.stl"
    meta = await _run(
        generate_stl_from_trace,
        traced,
        str(stl_path),
        target_width_mm=settings.target_width_mm(traced),
        **settings.kwargs,
    )
    return stl_path, meta


def _cutter_info(traced, meta: dict, extra: dict | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {"extraction_mode": traced.extraction_mode, **_topology_fields(traced)}
    for key in ("height_mm", "footprint_w_mm", "footprint_h_mm"):
        if meta.get(key) is not None:
            info[key] = meta[key]
    if traced.extraction_warning:
        info["warning"] = traced.extraction_warning
    if extra:
        info.update(extra)
    return info


def _zip_bytes(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, data in files:
            zf.writestr(arcname, data)
    return buf.getvalue()


def _cutter_response(
    name: str,
    stl_path: Path,
    info: dict[str, Any],
    fmt: str = "stl",
    extras: list[Path] | None = None,
    trace: dict | None = None,
) -> Response:
    """Return the cutter itself as the response body.

    ``fmt="stl"`` sends the bare STL; ``fmt="zip"`` bundles it with whatever
    else the request produced. The size and topology summary rides along in
    the ``X-Cutter-Info`` header as JSON, so the body stays the file.
    """
    headers = {"X-Cutter-Info": json.dumps(info), "Cache-Control": "no-store"}
    if fmt == "zip":
        files = [(stl_path.name, stl_path.read_bytes())]
        files += [(p.name, p.read_bytes()) for p in (extras or []) if p.is_file()]
        if trace is not None:
            files.append(("trace.json", json.dumps(trace).encode("utf-8")))
        headers["Content-Disposition"] = f'attachment; filename="{name}.zip"'
        return Response(content=_zip_bytes(files), media_type="application/zip", headers=headers)
    headers["Content-Disposition"] = f'attachment; filename="{name}.stl"'
    return Response(content=stl_path.read_bytes(), media_type="model/stl", headers=headers)


def _safe_format(fmt: str) -> str:
    fmt = (fmt or "stl").strip().lower()
    if fmt not in ("stl", "zip"):
        raise HTTPException(status_code=400, detail="format must be 'stl' or 'zip'.")
    return fmt


# ── Step 1/2: trace a shape ───────────────────────────────────────────────────
# Re-tracing with new settings is the same call again: the browser still holds
# the source it uploaded, so it simply sends it back.

async def _accept_png(file: UploadFile) -> bytes:
    if not (file.filename or "").lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Upload a PNG/JPG outline image")
    content = await _read_upload(file)
    _verify_image_bytes(content)
    _log_image_upload(file.filename, content)
    return content


async def _accept_stl(file: UploadFile) -> bytes:
    if not (file.filename or "").lower().endswith(".stl"):
        raise HTTPException(status_code=400, detail="Upload an STL file")
    content = await _read_upload(file)
    await _run(_verify_stl_bytes, content)
    _log.info("STL upload — file=%r size=%.1fKB", file.filename, len(content) / 1024)
    return content


@app.post("/trace/from-png")
async def trace_from_png(
    file: UploadFile = File(...),
    name: str = Form("outline"),
    trace: TraceSettings = Depends(),
):
    name = _safe_name(name)
    content = await _accept_png(file)
    with _scratch_dir() as work:
        png_path = work / f"{name}.png"
        svg_path = work / f"{name}.svg"
        png_path.write_bytes(content)
        traced = await _run(trace_png_to_polygon, str(png_path), str(svg_path), **trace.png_kwargs())
        return _trace_response(traced, svg_path)


@app.post("/trace/from-stl")
async def trace_from_stl(
    file: UploadFile = File(...),
    name: str = Form("outline"),
    simplify: float = Form(0.0008),
    topology: str = Form("auto"),
):
    name = _safe_name(name)
    content = await _accept_stl(file)
    with _scratch_dir() as work:
        stl_path = work / f"{name}_input.stl"
        svg_path = work / f"{name}.svg"
        stl_path.write_bytes(content)
        traced = await _run(
            extract_outline_from_stl, str(stl_path), str(svg_path),
            simplify_epsilon=simplify, topology=topology,
        )
        result = _trace_response(traced, svg_path)
        await _add_stl_size_fields(result, stl_path)
        return result


@app.post("/trace/from-grid")
async def trace_from_grid(
    name: str = Form("cookie_cutter"),
    cols: int = Form(...),
    rows: int = Form(...),
    cell_w_mm: float = Form(...),
    cell_h_mm: float = Form(0.0),
):
    """Grid Builder: create a lattice trace from explicit grid dimensions —
    no image or STL upload required. Cell sizes are exact millimetres,
    measured between wall centerlines. Pass cell_h_mm=0 for square cells."""
    name = _safe_name(name)
    _log.info("Grid Builder — %dx%d cells of %.1fx%.1f mm", cols, rows, cell_w_mm, cell_h_mm or cell_w_mm)
    with _scratch_dir() as work:
        svg_path = work / f"{name}.svg"
        traced = await _run(
            build_grid_trace, cols, rows, cell_w_mm,
            cell_h_mm if cell_h_mm > 0 else None, str(svg_path),
        )
        return _trace_response(traced, svg_path)


@app.post("/outline/from-prompt")
async def outline_from_prompt(
    prompt: str = Form(...),
    name: str = Form("cookie_cutter"),
    smooth_radius: float = Form(1.0),
):
    """Draw an outline from a text prompt and trace it. The drawing comes back
    as ``png_base64`` so the browser can re-trace it later like any upload."""
    name = _safe_name(name)
    with _scratch_dir() as work:
        png_path = work / f"{name}.png"
        svg_path = work / f"{name}.svg"
        await _prompt_to_png(prompt, png_path)
        traced = await _run(
            trace_png_to_polygon, str(png_path), str(svg_path),
            smooth_radius=smooth_radius, topology="single",
        )
        result = _trace_response(traced, svg_path)
        result["png_base64"] = base64.b64encode(png_path.read_bytes()).decode("ascii")
        return result


# ── Step 3: build the cutter from a trace ─────────────────────────────────────

@app.post("/stl/generate")
async def stl_generate(
    trace: UploadFile = File(..., description="The `trace` object a /trace/* response returned, as JSON."),
    name: str = Form("cookie_cutter"),
    settings: CutterSettings = Depends(),
):
    """Build a cutter from a trace the client already holds. Returns the STL
    as the response body, with the size summary in ``X-Cutter-Info``."""
    name = _safe_name(name)
    raw = await trace.read(MAX_TRACE_BYTES + 1)
    if len(raw) > MAX_TRACE_BYTES:
        raise HTTPException(status_code=413, detail="Trace is too large.")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Invalid trace: not JSON.")
    traced = await _run(trace_result_from_payload, payload)
    with _scratch_dir() as work:
        stl_path, meta = await _build_stl(traced, settings, work, name)
        return _cutter_response(name, stl_path, _cutter_info(traced, meta))


# ── One-shot endpoints for scripts and the API ────────────────────────────────
# Source in, cutter out, in one request. ``format=zip`` adds the source, the
# outline SVG and the trace to the download.

@app.post("/pipeline/from-png")
async def pipeline_from_png(
    file: UploadFile = File(...),
    name: str = Form("cookie_cutter"),
    format: str = Form("stl"),
    trace: TraceSettings = Depends(),
    settings: CutterSettings = Depends(),
):
    name = _safe_name(name)
    fmt = _safe_format(format)
    content = await _accept_png(file)
    with _scratch_dir() as work:
        png_path = work / f"{name}.png"
        svg_path = work / f"{name}.svg"
        png_path.write_bytes(content)
        traced = await _run(trace_png_to_polygon, str(png_path), str(svg_path), **trace.png_kwargs())
        stl_path, meta = await _build_stl(traced, settings, work, name)
        return _cutter_response(
            name, stl_path, _cutter_info(traced, meta), fmt,
            extras=[png_path, svg_path], trace=trace_result_to_dict(traced),
        )


@app.post("/pipeline/from-stl")
async def pipeline_from_stl(
    file: UploadFile = File(...),
    name: str = Form("cookie_cutter"),
    format: str = Form("stl"),
    simplify: float = Form(0.0008),
    topology: str = Form("auto"),
    settings: CutterSettings = Depends(),
):
    name = _safe_name(name)
    fmt = _safe_format(format)
    content = await _accept_stl(file)
    with _scratch_dir() as work:
        input_path = work / f"{name}_input.stl"
        svg_path = work / f"{name}.svg"
        input_path.write_bytes(content)
        traced = await _run(
            extract_outline_from_stl, str(input_path), str(svg_path),
            simplify_epsilon=simplify, topology=topology,
        )
        stl_path, meta = await _build_stl(traced, settings, work, name)
        size: dict[str, Any] = {}
        await _add_stl_size_fields(size, input_path)
        return _cutter_response(
            name, stl_path, _cutter_info(traced, meta, size), fmt,
            extras=[input_path, svg_path], trace=trace_result_to_dict(traced),
        )


@app.post("/pipeline/from-prompt")
async def pipeline_from_prompt(
    prompt: str = Form(...),
    name: str = Form("cookie_cutter"),
    format: str = Form("stl"),
    smooth_radius: float = Form(1.0),
    settings: CutterSettings = Depends(),
):
    name = _safe_name(name)
    fmt = _safe_format(format)
    with _scratch_dir() as work:
        png_path = work / f"{name}.png"
        svg_path = work / f"{name}.svg"
        await _prompt_to_png(prompt, png_path)
        traced = await _run(
            trace_png_to_polygon, str(png_path), str(svg_path),
            smooth_radius=smooth_radius, topology="single",
        )
        prompt_path = work / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        stl_path, meta = await _build_stl(traced, settings, work, name)
        return _cutter_response(
            name, stl_path, _cutter_info(traced, meta), fmt,
            extras=[png_path, svg_path, prompt_path], trace=trace_result_to_dict(traced),
        )


@app.post("/pipeline/from-grid")
async def pipeline_from_grid(
    name: str = Form("cookie_cutter"),
    format: str = Form("stl"),
    cols: int = Form(...),
    rows: int = Form(...),
    cell_w_mm: float = Form(...),
    cell_h_mm: float = Form(0.0),
    settings: CutterSettings = Depends(),
):
    """One-shot Grid Builder: grid spec straight to STL."""
    name = _safe_name(name)
    fmt = _safe_format(format)
    _log.info("Grid Builder (pipeline) — %dx%d cells of %.1fx%.1f mm", cols, rows, cell_w_mm, cell_h_mm or cell_w_mm)
    with _scratch_dir() as work:
        svg_path = work / f"{name}.svg"
        traced = await _run(
            build_grid_trace, cols, rows, cell_w_mm,
            cell_h_mm if cell_h_mm > 0 else None, str(svg_path),
        )
        stl_path, meta = await _build_stl(traced, settings, work, name)
        return _cutter_response(
            name, stl_path, _cutter_info(traced, meta), fmt,
            extras=[svg_path], trace=trace_result_to_dict(traced),
        )
