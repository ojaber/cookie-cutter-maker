import asyncio
import hashlib
import hmac
import os
import re
import secrets
import sys
import uuid
import json
import logging
import time
from typing import Any
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import anyio

from PIL import Image as _PILImage
# Cap pixel count to defuse decompression bombs before PIL allocates a giant array.
# Default PIL limit is ~89 MP — tighten to ~25 MP (≈5000×5000), which is more than
# enough for a cookie-cutter outline.
_PILImage.MAX_IMAGE_PIXELS = 25_000_000
from cutter_pipeline.trace_outline import trace_png_to_polygon
from cutter_pipeline.stl_dispatch import generate_stl_from_trace
from cutter_pipeline.trace_meta import load_trace_result, save_trace_result
from shapely.geometry import shape, mapping
import trimesh
import zipfile
from openai import OpenAIError

# Optional: prompt->png requires OPENAI_API_KEY
try:
    from cutter_pipeline.outline_openai import generate_outline_png
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

_log = logging.getLogger(__name__)

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

_AUTH_EXEMPT = {"/login", "/logout", "/healthz", "/favicon.ico"}

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
  <link rel="icon" type="image/png" href="/favicon.ico"/>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center}}
    .card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:2rem;width:100%;max-width:360px}}
    h1{{font-size:1.25rem;margin-bottom:.25rem}}
    .sub{{font-size:.85rem;color:#94a3b8;margin-bottom:1.5rem}}
    label{{display:block;font-size:.85rem;color:#94a3b8;margin-bottom:.4rem}}
    input[type=password]{{width:100%;padding:.6rem .75rem;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:1rem;margin-bottom:1rem}}
    input[type=password]:focus{{outline:none;border-color:#6366f1}}
    button{{width:100%;padding:.65rem;background:#6366f1;color:#fff;border:none;border-radius:6px;font-size:1rem;cursor:pointer}}
    button:hover{{background:#4f46e5}}
    .error{{color:#fca5a5;font-size:.85rem;margin-bottom:1rem}}
  </style>
</head>
<body>
  <div class="card">
    <h1>Cookie Cutter Maker</h1>
    <p class="sub">Enter your passphrase to continue.</p>
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

app = FastAPI(title="Cookie Cutter Maker", version="0.2.0")

OUTPUT_DIR = Path(os.environ.get("PIPELINE_OUTPUT_DIR", "output")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reject uploads larger than this to avoid disk/memory exhaustion.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_name(name: str) -> str:
    """Validate a user-supplied base filename used to build output paths."""
    name = (name or "").strip()
    if not _SAFE_NAME_RE.match(name) or ".." in name:
        raise HTTPException(
            status_code=400,
            detail="Invalid name: use letters, digits, '.', '_' or '-' only (max 64 chars).",
        )
    return name


def _safe_job_id(job_id: str) -> str:
    """Validate that a job_id matches the 32-char hex format produced by _new_job_dir."""
    job_id = (job_id or "").strip().lower()
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id.")
    return job_id


def _safe_filename(filename: str) -> str:
    filename = (filename or "").strip()
    if not _SAFE_FILENAME_RE.match(filename) or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return filename


def _confined_path(job_id: str, *parts: str) -> Path:
    """Build a path inside OUTPUT_DIR/<job_id> and verify it cannot escape."""
    job_id = _safe_job_id(job_id)
    candidate = (OUTPUT_DIR / job_id).resolve()
    try:
        candidate.relative_to(OUTPUT_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id.")
    for part in parts:
        candidate = candidate / part
    resolved = candidate.resolve()
    try:
        resolved.relative_to(OUTPUT_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path.")
    return resolved


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
        raise HTTPException(status_code=400, detail="Image too large (decompression bomb guard).")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

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
        path = _metrics_path(request.url.path)
        method = request.method
        REQUEST_COUNT.labels(method=method, path=path, status=status_code).inc()
        REQUEST_LATENCY.labels(method=method, path=path, status=status_code).observe(elapsed)


def _metrics_path(path: str) -> str:
    """Collapse paths with per-job IDs so Prometheus label cardinality stays bounded."""
    if path.startswith("/files/"):
        return "/files/:job_id/:filename"
    return path


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

@app.get("/features", include_in_schema=False)
def features():
    from cutter_pipeline.image_extractor import REMBG_ENABLED
    return {
        "background_removal": REMBG_ENABLED,
        "image_generation": bool(os.environ.get("OPENAI_API_KEY")),
    }

def _new_job_dir() -> Path:
    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _write_zip(job_dir: Path, files: list[Path], base_name: str = "all") -> Path:
    zip_path = job_dir / f"{base_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists():
                zf.write(f, arcname=f.name)
    return zip_path


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
    return fields


def _persist_trace(job_dir: Path, traced) -> None:
    save_trace_result(job_dir, traced)


def _find_png(job_dir: Path, name: str) -> Path:
    candidate = job_dir / f"{name}.png"
    if candidate.exists():
        return candidate
    matches = list(job_dir.glob("*.png"))
    if matches:
        return matches[0]
    raise HTTPException(status_code=404, detail="PNG not found for this job. Upload or generate first.")


def _log_image_upload(filename: str, content: bytes, path: Path) -> None:
    try:
        with _PILImage.open(path) as img:
            w, h = img.size
        _log.info("Image upload — file=%r size=%.1fKB dimensions=%dx%d", filename, len(content) / 1024, w, h)
    except Exception:
        _log.info("Image upload — file=%r size=%.1fKB", filename, len(content) / 1024)

def _openai_detail(exc: OpenAIError) -> str:
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

@app.post("/trace/from-png")
async def trace_from_png(
    file: UploadFile = File(...),
    name: str = Form("outline"),
    threshold: int = Form(200),
    simplify: float = Form(0.002),
    smooth_radius: float = Form(1.0),
    extraction_mode: str = Form("auto"),
    delta_e_threshold: float = Form(28.0),
    topology: str = Form("auto"),
):
    name = _safe_name(name)
    if not (file.filename or "").lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Upload a PNG/JPG outline image")

    content = await _read_upload(file)
    _verify_image_bytes(content)

    job_dir = _new_job_dir()
    png_path = job_dir / f"{name}.png"
    svg_path = job_dir / f"{name}.svg"

    png_path.write_bytes(content)
    _log_image_upload(file.filename, content, png_path)
    traced = trace_png_to_polygon(
        str(png_path),
        str(svg_path),
        threshold=threshold,
        simplify_epsilon=simplify,
        smooth_radius=smooth_radius,
        extraction_mode=extraction_mode,
        delta_e_threshold=delta_e_threshold,
        topology=topology,
    )
    _persist_trace(job_dir, traced)

    result = {
        "job_id": job_dir.name,
        "svg": f"/files/{job_dir.name}/{name}.svg",
        "png": f"/files/{job_dir.name}/{name}.png",
        "extraction_mode": traced.extraction_mode,
        **_topology_fields(traced),
    }
    if traced.extraction_warning:
        result["warning"] = traced.extraction_warning
    return result

@app.post("/pipeline/from-png")
async def pipeline_from_png(
    file: UploadFile = File(...),
    name: str = Form("cookie_cutter"),
    width_mm: float = Form(95.0),
    wall_mm: float = Form(1.2),
    total_h_mm: float = Form(15.0),
    flange_h_mm: float = Form(7.0),
    flange_out_mm: float = Form(4.0),
    bottom_wall_mm: float = Form(0.4),
    cutting_wall_h_mm: float = Form(3.0),
    cleanup_mm: float = Form(0.5),
    tip_smooth_mm: float = Form(0.6),
    keep_holes: bool = Form(False),
    min_component_area_mm2: float = Form(25.0),
    threshold: int = Form(200),
    simplify: float = Form(0.002),
    smooth_radius: float = Form(1.0),
    extraction_mode: str = Form("auto"),
    delta_e_threshold: float = Form(28.0),
    topology: str = Form("auto"),
):
    name = _safe_name(name)
    if not (file.filename or "").lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Upload a PNG/JPG outline image")

    content = await _read_upload(file)
    _verify_image_bytes(content)

    job_dir = _new_job_dir()
    png_path = job_dir / f"{name}.png"
    svg_path = job_dir / f"{name}.svg"
    stl_path = job_dir / f"{name}.stl"

    png_path.write_bytes(content)
    _log_image_upload(file.filename, content, png_path)
    traced = trace_png_to_polygon(
        str(png_path),
        str(svg_path),
        threshold=threshold,
        simplify_epsilon=simplify,
        smooth_radius=smooth_radius,
        extraction_mode=extraction_mode,
        delta_e_threshold=delta_e_threshold,
        topology=topology,
    )
    _persist_trace(job_dir, traced)

    stl_meta = generate_stl_from_trace(
        traced,
        str(stl_path),
        target_width_mm=width_mm,
        wall_mm=wall_mm,
        total_h_mm=total_h_mm,
        flange_h_mm=flange_h_mm,
        flange_out_mm=flange_out_mm,
        bottom_wall_mm=bottom_wall_mm,
        cutting_wall_h_mm=cutting_wall_h_mm,
        cleanup_mm=cleanup_mm,
        tip_smooth_mm=tip_smooth_mm,
        drop_holes=not keep_holes,
        min_component_area_mm2=min_component_area_mm2,
    )

    zip_path = _write_zip(job_dir, [png_path, svg_path, stl_path, job_dir / "trace_meta.json"], base_name=name)

    result = {
        "job_id": job_dir.name,
        "png": f"/files/{job_dir.name}/{name}.png",
        "svg": f"/files/{job_dir.name}/{name}.svg",
        "stl": f"/files/{job_dir.name}/{name}.stl",
        "zip": f"/files/{job_dir.name}/{zip_path.name}",
        "extraction_mode": traced.extraction_mode,
        **_topology_fields(traced),
    }
    if stl_meta.get("height_mm") is not None:
        result["height_mm"] = stl_meta["height_mm"]
    if traced.extraction_warning:
        result["warning"] = traced.extraction_warning
    return result

@app.post("/pipeline/from-prompt")
async def pipeline_from_prompt(
    prompt: str = Form(...),
    name: str = Form("cookie_cutter"),
    width_mm: float = Form(95.0),
    wall_mm: float = Form(1.2),
    total_h_mm: float = Form(15.0),
    flange_h_mm: float = Form(7.0),
    flange_out_mm: float = Form(4.0),
    bottom_wall_mm: float = Form(0.4),
    cutting_wall_h_mm: float = Form(3.0),
    cleanup_mm: float = Form(0.5),
    tip_smooth_mm: float = Form(0.6),
    keep_holes: bool = Form(False),
    min_component_area_mm2: float = Form(25.0),
    smooth_radius: float = Form(1.0),
):
    name = _safe_name(name)
    if len(prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt must be 1000 characters or fewer.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=402, detail="OPENAI_API_KEY not set. Use /pipeline/from-png for offline mode.")
    if not HAS_OPENAI:
        raise HTTPException(status_code=500, detail="OpenAI image step unavailable.")

    _log.info("Prompt (pipeline) — %r", prompt.strip()[:500])

    job_dir = _new_job_dir()
    png_path = job_dir / f"{name}.png"
    svg_path = job_dir / f"{name}.svg"
    stl_path = job_dir / f"{name}.stl"

    OPENAI_INFLIGHT.inc()
    try:
        await anyio.to_thread.run_sync(generate_outline_png, prompt, str(png_path))
    except OpenAIError as e:
        status = getattr(e, "status_code", 500) or 500
        detail = _openai_detail(e)
        logging.warning(
            "OpenAI image generation failed (status=%s, prompt=%s): %s",
            status,
            prompt.strip()[:200],
            detail,
        )
        raise HTTPException(status_code=status, detail=detail)
    finally:
        OPENAI_INFLIGHT.dec()

    traced = trace_png_to_polygon(
        str(png_path),
        str(svg_path),
        smooth_radius=smooth_radius,
        topology="single",
    )
    # Save prompt for reference
    (job_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _persist_trace(job_dir, traced)

    generate_stl_from_trace(
        traced,
        str(stl_path),
        target_width_mm=width_mm,
        wall_mm=wall_mm,
        total_h_mm=total_h_mm,
        flange_h_mm=flange_h_mm,
        flange_out_mm=flange_out_mm,
        bottom_wall_mm=bottom_wall_mm,
        cutting_wall_h_mm=cutting_wall_h_mm,
        cleanup_mm=cleanup_mm,
        tip_smooth_mm=tip_smooth_mm,
        drop_holes=not keep_holes,
        min_component_area_mm2=min_component_area_mm2,
    )

    zip_path = _write_zip(
        job_dir,
        [png_path, svg_path, stl_path, job_dir / "prompt.txt", job_dir / "trace_meta.json"],
        base_name=name,
    )

    return {
        "job_id": job_dir.name,
        "png": f"/files/{job_dir.name}/{name}.png",
        "svg": f"/files/{job_dir.name}/{name}.svg",
        "stl": f"/files/{job_dir.name}/{name}.stl",
        "zip": f"/files/{job_dir.name}/{zip_path.name}",
        **_topology_fields(traced),
    }

@app.post("/outline/from-prompt")
async def outline_from_prompt(
    prompt: str = Form(...),
    name: str = Form("cookie_cutter"),
    smooth_radius: float = Form(1.0),
):
    name = _safe_name(name)
    if len(prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt must be 1000 characters or fewer.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=402, detail="OPENAI_API_KEY not set. Cannot generate prompt image.")
    if not HAS_OPENAI:
        raise HTTPException(status_code=500, detail="OpenAI image step unavailable.")

    _log.info("Prompt (outline) — %r", prompt.strip()[:500])

    job_dir = _new_job_dir()
    png_path = job_dir / f"{name}.png"
    svg_path = job_dir / f"{name}.svg"

    OPENAI_INFLIGHT.inc()
    try:
        await anyio.to_thread.run_sync(generate_outline_png, prompt, str(png_path))
    except OpenAIError as e:
        status = getattr(e, "status_code", 500) or 500
        detail = _openai_detail(e)
        logging.warning(
            "OpenAI outline failed (status=%s, prompt=%s): %s",
            status,
            prompt.strip()[:200],
            detail,
        )
        raise HTTPException(status_code=status, detail=detail)
    finally:
        OPENAI_INFLIGHT.dec()

    traced = trace_png_to_polygon(
        str(png_path),
        str(svg_path),
        smooth_radius=smooth_radius,
        topology="single",
    )
    (job_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _persist_trace(job_dir, traced)

    return {
        "job_id": job_dir.name,
        "png": f"/files/{job_dir.name}/{name}.png",
        "svg": f"/files/{job_dir.name}/{name}.svg",
        **_topology_fields(traced),
    }

@app.post("/trace/from-job")
async def trace_from_job(
    job_id: str = Form(...),
    name: str = Form("cookie_cutter"),
    threshold: int = Form(200),
    simplify: float = Form(0.002),
    smooth_radius: float = Form(1.0),
    extraction_mode: str = Form("auto"),
    delta_e_threshold: float = Form(28.0),
    topology: str = Form("auto"),
):
    name = _safe_name(name)
    job_dir = _confined_path(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job_id not found")

    png_path = _find_png(job_dir, name)
    svg_path = job_dir / f"{Path(png_path).stem}.svg"

    traced = trace_png_to_polygon(
        str(png_path),
        str(svg_path),
        threshold=threshold,
        simplify_epsilon=simplify,
        smooth_radius=smooth_radius,
        extraction_mode=extraction_mode,
        delta_e_threshold=delta_e_threshold,
        topology=topology,
    )
    _persist_trace(job_dir, traced)

    result = {
        "job_id": job_dir.name,
        "png": f"/files/{job_dir.name}/{png_path.name}",
        "svg": f"/files/{job_dir.name}/{svg_path.name}",
        "extraction_mode": traced.extraction_mode,
        **_topology_fields(traced),
    }
    if traced.extraction_warning:
        result["warning"] = traced.extraction_warning
    return result

@app.post("/stl/from-job")
async def stl_from_job(
    job_id: str = Form(...),
    name: str = Form("cookie_cutter"),
    width_mm: float = Form(95.0),
    wall_mm: float = Form(1.2),
    total_h_mm: float = Form(15.0),
    flange_h_mm: float = Form(7.0),
    flange_out_mm: float = Form(4.0),
    bottom_wall_mm: float = Form(0.4),
    cutting_wall_h_mm: float = Form(3.0),
    cleanup_mm: float = Form(0.5),
    tip_smooth_mm: float = Form(0.6),
    keep_holes: bool = Form(False),
    min_component_area_mm2: float = Form(25.0),
):
    name = _safe_name(name)
    job_dir = _confined_path(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job_id not found")

    try:
        traced = load_trace_result(job_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Trace data for this job not found. Trace or upload again.")
    stl_path = job_dir / f"{name}.stl"

    stl_meta = generate_stl_from_trace(
        traced,
        str(stl_path),
        target_width_mm=width_mm,
        wall_mm=wall_mm,
        total_h_mm=total_h_mm,
        flange_h_mm=flange_h_mm,
        flange_out_mm=flange_out_mm,
        bottom_wall_mm=bottom_wall_mm,
        cutting_wall_h_mm=cutting_wall_h_mm,
        cleanup_mm=cleanup_mm,
        tip_smooth_mm=tip_smooth_mm,
        drop_holes=not keep_holes,
        min_component_area_mm2=min_component_area_mm2,
    )

    files = [
        stl_path,
        job_dir / f"{name}.png",
        job_dir / f"{name}.svg",
        job_dir / "prompt.txt",
        job_dir / "polygon.json",
        job_dir / "trace_meta.json",
    ]
    zip_path = _write_zip(job_dir, files, base_name=name)

    result = {
        "job_id": job_id,
        "png": f"/files/{job_id}/{name}.png" if (job_dir / f"{name}.png").exists() else None,
        "svg": f"/files/{job_id}/{name}.svg" if (job_dir / f"{name}.svg").exists() else None,
        "stl": f"/files/{job_id}/{name}.stl",
        **_topology_fields(traced),
    }
    if stl_meta.get("height_mm") is not None:
        result["height_mm"] = stl_meta["height_mm"]

    return {
        **result,
        "zip": f"/files/{job_id}/{zip_path.name}",
    }

@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str):
    filename = _safe_filename(filename)
    path = _confined_path(job_id, filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
