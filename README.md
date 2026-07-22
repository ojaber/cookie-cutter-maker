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
- Docker build/run and Terraform for DigitalOcean App Platform.

## Quick start (Docker)

```bash
docker compose up --build
```

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

MIT License © seaburr

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

## Infrastructure (Terraform / DigitalOcean App Platform)

The `terraform/` directory contains configuration to deploy the app to [DigitalOcean App Platform](https://www.digitalocean.com/products/app-platform) at `cookies.seaburr.io`.

**First-time setup:**

```bash
cd terraform
terraform init
terraform apply \
  -var="do_token=<your-do-token>"
```

**With optional variables:**

```bash
terraform apply \
  -var="do_token=<your-do-token>" \
  -var="openai_api_key=<your-openai-key>" \
  -var="rembg_enabled=true" \
  -var="instance_size_slug=apps-s-1vcpu-1gb-fixed"
```

**Update existing infrastructure:**

```bash
cd terraform
terraform apply -var="do_token=<your-do-token>"
```

Terraform will show a plan of changes before applying. Key variables:

| Variable | Default | Description |
|---|---|---|
| `do_token` | _(required)_ | DigitalOcean personal access token. |
| `image_tag` | `latest` | Docker image tag to deploy from GHCR. |
| `region` | `atl` | App Platform region (`atl`, `nyc`, `ams`, `sfo`, `fra`, `lon`, `sgp`, `syd`, `tor`). |
| `instance_size_slug` | `apps-s-1vcpu-1gb-fixed` | App Platform instance size. |
| `instance_count` | `1` | Number of instances. |
| `rembg_enabled` | `false` | Enable rembg background removal (see above). |
| `openai_api_key` | _(unset)_ | Optional — enables prompt-to-outline generation. |

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
