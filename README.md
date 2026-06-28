# Cookie Cutter Maker (PNG/SVG/STL -> STL) + Local UI (Docker)

This repo generates cookie cutter STL files from:
- **an outline PNG** (offline/local; no OpenAI cost),
- **an STL file** (3D model projected to 2D outline), or
- **a text prompt** (outline PNG via OpenAI Images API if you set `OPENAI_API_KEY`).

It includes:
- a Python pipeline (trace + STL generation),
- a **FastAPI** service wrapping the pipeline,
- a simple **local web UI** for end-to-end use,
- Docker build/run.

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

## Prompt flow (optional)

If you want prompt -> outline generation:
1. Set `OPENAI_API_KEY` in your environment (or docker-compose.yml)
2. Use the Prompt tab in the UI or `POST /pipeline/from-prompt`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(unset)_ | Enables prompt-to-outline generation via the OpenAI Images API. If unset, the Prompt tab returns HTTP 402. |
| `REMBG_ENABLED` | `true` | Set to `false` to disable rembg background removal for complex/photographic images. When disabled the pipeline falls back to graph-cut (Felzenszwalb) segmentation, which is faster but less accurate. Disable this if you are running on a memory-constrained instance — rembg loads a ~170 MB U2Net model into memory at startup. |
| `PIPELINE_OUTPUT_DIR` | `output` | Directory where generated job files (PNG, SVG, STL, ZIP) are written. |

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
