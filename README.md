# Cookie Cutter Maker (PNG/SVG/STL -> STL) + Web UI (Docker)

Turn any picture into a 3D-printable cookie cutter. Inputs:
- **an image** (PNG/JPG — outline, logo, or photo; offline/local, no OpenAI cost),
- **an STL file** (3D model projected to 2D outline),
- **a grid spec** (Grid Builder — pick columns, rows and exact cell size; no image needed), or
- **a text prompt** (outline PNG via OpenAI Images API if you set `OPENAI_API_KEY`).

It includes:
- a Python pipeline (trace + STL generation),
- a **FastAPI** service wrapping the pipeline,
- a **guided web UI** — drag & drop (or paste) an image, check the traced
  outline, tweak a few sliders, preview the cutter in 3D and download the STL.
  Three steps, plain-language labels with tooltips, advanced settings tucked
  behind a disclosure, light/dark theme, works on phones,
- public-instance guards (per-IP rate limiting + a concurrency cap on the
  heavy pipeline endpoints), security headers and `robots.txt`,
- a free, auto-deploying setup for [Render](https://render.com) (native
  Python, no Docker needed) plus a `Dockerfile` for any container host.

## Quick start (Docker)

```bash
make docker-up        # or: mkdir -p output && docker compose up --build
```

(The `mkdir` matters on Linux: the container runs as a non-root user and
needs the bind-mounted `output/` dir to be writable by you, not root.)

Open:
- UI: http://localhost:8000
- API docs: http://localhost:8000/docs

Generated files land in `./output/<job_id>/`.

## Run tests

```bash
pip install -r requirements.txt
pytest
```

## License

MIT — original project © seaburr, modifications © ojaber. See `LICENSE`.

## Offline flow (recommended)

### From PNG
1. Create or download a **simple black shape on white background** PNG outline.
2. Upload it in the UI.
3. Adjust sliders (wall, flange size, height, smoothing).
4. Download STL.

### From STL
1. Upload an **STL file** (3D model) in the UI using the "STL Upload" tab.
2. The STL is projected along the Z-axis (top-down view) to extract a 2D outline.
3. Adjust sliders (wall, flange size, height, smoothing).
4. Download STL.

No OpenAI calls.

### Grid / lattice outlines

Connected grid line art (e.g. tic-tac-toe or brownie dividers) is supported via **Shape mode**:

- **Auto** — detects a regular connected grid and builds divider walls with one outer grip flange.
- **Grid / lattice** — force lattice mode for evenly spaced cell lines.

Single silhouettes (heart, star, etc.) continue to use the classic ring cutter path.

### Grid Builder (no image needed)

If you just want a divider grid, you don't need an image at all. The **Grid
Builder** tab lets you specify the grid directly:

- **Columns / Rows** — number of cells (1–40 each).
- **Cell width / height (mm)** — exact spacing between wall centerlines
  (5–300 mm; each opening is one wall thickness smaller than the cell).

Cell sizes are honored exactly — the width slider is fixed for Grid Builder
jobs so nothing gets rescaled. The equivalent API endpoints are
`POST /trace/from-grid` (two-step, then `POST /stl/from-job`) and
`POST /pipeline/from-grid` (one-shot STL + zip):

```bash
curl -X POST http://localhost:8000/pipeline/from-grid \
  -F cols=4 -F rows=3 -F cell_w_mm=30 -F cell_h_mm=25 -F name=brownie_grid
```

## Prompt flow (optional)

If you want prompt -> outline generation:
1. Set `OPENAI_API_KEY` in your environment (or docker-compose.yml)
2. Use the Prompt tab in the UI or `POST /pipeline/from-prompt`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(unset)_ | Enables prompt-to-outline generation via the OpenAI Images API. If unset, the "From text" tab is shown as unavailable and the endpoint returns HTTP 402. |
| `REMBG_ENABLED` | `true` | Set to `false` to disable rembg background removal for complex/photographic images. When disabled the pipeline falls back to graph-cut (Felzenszwalb) segmentation, which is faster but less accurate. Disable this if you are running on a memory-constrained instance — rembg loads a ~170 MB U2Net model into memory at startup. |
| `PIPELINE_OUTPUT_DIR` | `output` | Directory where generated job files (PNG, SVG, STL, ZIP) are written. |
| `JOB_TTL_HOURS` | `24` | Job directories older than this are deleted by a background sweeper so the disk doesn't fill up. Set to `0` to keep jobs forever. |
| `MAX_UPLOAD_BYTES` | `20971520` | Maximum accepted upload size (20 MB). |
| `ACCESS_PASSWORD` | _(unset)_ | If set, the whole app sits behind a passphrase login. Leave unset for an open/public site. |
| `SESSION_SECRET` | _(unset)_ | Signs login session cookies when `ACCESS_PASSWORD` is set. Set it to any long random string so sessions survive restarts/replicas. |
| `RATE_LIMIT_PER_MINUTE` | `20` | Max heavy pipeline POSTs per client IP per minute (HTTP 429 beyond that). `0` disables. |
| `HEAVY_CONCURRENCY` | `2` | Max trace/mesh jobs running at once; extra requests queue briefly and then get HTTP 503. `0` disables. |
| `HEAVY_QUEUE_TIMEOUT_SECONDS` | `30` | How long a request waits for a free job slot before returning 503. |
| `FRAME_ANCESTORS` | `'self'` | CSP `frame-ancestors` value — who may embed the app in an iframe. Set e.g. `'self' https://example.com` to allow embedding on another site. |

## Sharing it publicly

The defaults are chosen so you can put an instance on the open internet:

- **Abuse guards on by default** — per-IP rate limiting (`RATE_LIMIT_PER_MINUTE`)
  and a cap on concurrent pipeline jobs (`HEAVY_CONCURRENCY`) protect small
  instances from bursts and bots. Both are in-memory and per-process, which is
  right for the default single-instance deploy (run one uvicorn worker, or put
  a real rate limiter in front, if you scale out).
- **Uploads are bounded** (`MAX_UPLOAD_BYTES`, decompression-bomb guard) and
  **jobs are ephemeral** — artifacts are deleted after `JOB_TTL_HOURS` (24 h).
- **Security headers** (CSP, `X-Frame-Options`, `nosniff`, referrer policy) are
  set on every response, and `robots.txt` keeps crawlers out of `/files/`.
- Want it semi-private instead? Set `ACCESS_PASSWORD` (plus `SESSION_SECRET`)
  and the app shows a login page first.
- Keep `REMBG_ENABLED=false` on instances with less than ~2 GB of RAM.

## Deploy for free (Render)

The app runs on [Render](https://render.com)'s free tier as a native Python
web service — no Docker image, no credit card. Render connects to your GitHub
repo and **auto-deploys on every push** to the chosen branch. The repo ships a
`render.yaml` blueprint (free plan, `requirements-render.txt`, photo AI off) so
setup is just a few clicks.

**One-time setup (about 3 minutes):**

1. Create a free account at [render.com](https://render.com) (sign in with
   GitHub).
2. **New → Blueprint**, pick this repo, and Render reads `render.yaml`.
3. Click **Apply**. Render installs the dependencies and starts the app.

Your app goes live at `https://<name>.onrender.com` and re-deploys on every
push. Set optional env vars (`ACCESS_PASSWORD`, `OPENAI_API_KEY`, …) under the
service's **Environment** tab.

Free-tier trade-offs:
- **512 MB RAM**, so photo background-removal (the U2Net model) stays off —
  the blueprint sets `REMBG_ENABLED=false`. Simple outlines, logos, and
  photos on a plain background all still work.
- The service **sleeps after ~15 min of inactivity** and takes ~1 minute to
  wake on the next visit (fine for a hobby/share link).
- The disk is **ephemeral**: restarting or waking wipes every job directory,
  so a `job_id` from before a restart no longer exists. The browser keeps your
  source image/grid, so the UI rebuilds the job automatically and retries —
  you should never see a "job not found" dead end. Download links from an
  earlier session will stop working, though.

Prefer more headroom (photo AI on, no sleep)? The app is a standard FastAPI
service on port 8000, so any container or Python host works too — Google Cloud
Run, Fly.io, Railway, a small VPS, etc. Use `requirements.txt` (full) and the
`Dockerfile` there.

## CLI

PNG input:

```bash
python -m cutter_pipeline.cli --png examples/pajama_outline.png --outdir output --name pajama
python -m cutter_pipeline.cli --png ~/Downloads/3-4.png --topology auto --outdir output --name grid
```

STL input:

```bash
python -m cutter_pipeline.cli --stl examples/dino.stl --outdir output --name dino_cutter
```

Grid Builder (no input file):

```bash
python -m cutter_pipeline.cli --grid 4x3 --cell-mm 30 --cell-h-mm 25 --outdir output --name brownie_grid
```

Prompt input:

```bash
python -m cutter_pipeline.cli --prompt "a heart shape silhouette" --outdir output --name heart
```

## Test / smoke test

```bash
python -m cutter_pipeline.cli --png examples/pajama_outline.png --outdir output --name smoke_test
test -f output/smoke_test.stl
```

## Notes

- Many slicers show a closed-solid-with-void as "solid" unless you use section/cut view.
- The STL topology matches your “circle reference” style: constant ID, OD larger only in flange, slicer-friendly.
