# Plan: make the pipeline stateless

## Why

A single logical operation — pick a shape, check the outline, generate the
cutter — is spread across four or more HTTP requests that communicate through
a `job_id` naming a directory on local disk:

```
POST /trace/from-png   -> writes output/<job_id>/{name.png, name.svg, polygon.json, trace_meta.json}
POST /trace/from-job   -> reads  output/<job_id>/          (re-trace)
POST /stl/from-job     -> reads  output/<job_id>/trace_meta.json, writes name.stl + name.zip
GET  /files/<job_id>/… -> reads  output/<job_id>/          (previews, downloads)
```

That design assumes a durable filesystem shared by every request. On a free
host it isn't: the disk is ephemeral and the instance sleeps when idle, so any
restart deletes every job mid-session. The symptom was "Not Found" when
changing an outline setting.

Two mitigations already shipped, and they are worth keeping:

- the client rebuilds a lost job from the source it still holds and retries
  (`withJobRecovery`), and
- generated artifacts are pulled into memory as `blob:` URLs so downloads keep
  working after the server forgets them (`holdArtifacts`).

Both compensate for the assumption rather than removing it. This plan removes
it: **the server stops keeping anything between requests.**

## What actually has to survive between requests

Measured, not guessed:

| Item | Size | Who needs it |
|---|---|---|
| `trace_meta.json` (the traced outline) | **~3 KB** | step 3, to build the mesh |
| source image | as uploaded (≤20 MB cap) | re-tracing with new settings |
| generated STL | 0.09 MB (3×4 grid) – 2.15 MB (20×20 grid); 1.36 MB for the dino | the download |

The outline is the only thing that must round-trip, and it is tiny. The
browser already holds the source (a `File`) and, since option 2, the generated
artifacts. So the state is *already* client-side — the server copy is
redundant.

## Answering the obvious question: where do the files live?

**In the browser, and only for as long as the tab is open.**

- **Source image** — already there as a `File` from the picker/drop/paste. It
  never needs to be stored server-side; it is re-uploaded when a re-trace is
  requested.
- **Traced outline** — returned in the JSON response (~3 KB) and kept in a JS
  variable. Posted back when generating.
- **Generated STL/SVG/PNG/ZIP** — returned as the response body, wrapped in a
  `Blob`, exposed via `URL.createObjectURL`. This is exactly what option 2
  already does; the difference is the bytes come straight from the generate
  response instead of a follow-up `GET /files/…`.
- **On the server** — a temp directory that exists only for the duration of one
  request, deleted before it returns. Nothing persists; `/files/` and the TTL
  sweeper disappear.

Memory cost is a few MB per session, dominated by the STL (2.15 MB worst case
measured above) — negligible for a browser tab.

Trade-off to accept deliberately: **shareable links go away.** A URL like
`/files/<job_id>/cookie_cutter.stl` can currently be sent to someone else.
Stateless means the file exists only in the generating browser. On the current
free host those links already die at the next restart, so little is really
lost — but if durable sharing is ever wanted, that needs real object storage
(S3/R2), not local disk.

## Changes

### Server

1. **`POST /trace/from-png|stl|grid`** — unchanged inputs; response gains a
   `trace` field (the ~3 KB outline, the JSON already written to
   `trace_meta.json`) and an inline `svg` string instead of a `/files/` URL.
   Stop writing a job directory.
2. **`POST /stl/generate`** (replaces `/stl/from-job`) — accepts the `trace`
   payload plus the existing size parameters; returns the STL bytes directly
   (`application/sla`), or a ZIP when the extras are wanted. Work happens in a
   `TemporaryDirectory` that is removed on the way out.
3. **Re-tracing** needs the source, so `/trace/from-job` is dropped: the client
   re-posts the image to `/trace/from-png` with the new settings. It already
   does exactly this in the recovery path.
4. **Delete** `/files/{job_id}/{filename}`, `_new_job_dir`, `_confined_path`,
   the TTL sweeper, and `JOB_TTL_HOURS`. The path-traversal guards go with
   them — there are no user-named paths left to attack.
5. **Keep** `/pipeline/from-*` one-shot endpoints for CLI/API users, but have
   them stream the STL back rather than persist it.

Validation to preserve: a `trace` posted by a client is untrusted input. It
must be schema-checked (contour counts, coordinate ranges, array sizes) before
it reaches the mesher, with a size cap on the request body. This is the main
new security surface and the reason step 2 below exists.

### Client

6. Hold `traceState` (the outline JSON) alongside the existing `File`.
7. "Update outline" re-posts the source; "Generate" posts `traceState`.
8. Read the STL from the generate response into a `Blob`; feed the same blob to
   the three.js viewer and the download links. `holdArtifacts` collapses into
   this — no second fetch.
9. `withJobRecovery` and `JobGoneError` are deleted: with no job to lose, there
   is nothing to recover. Transient-error retries (502/503 while the instance
   wakes) stay — those are a property of the host, not the architecture.

### Tests

10. Rewrite the job-flow tests (`test_stl_source_survives_generation`,
    `test_grid_job_trace_from_job_returns_stored_trace`, …) around
    trace-in/STL-out. Drop the path-traversal suite with the routes it covers.
11. Add: malformed/oversized `trace` payloads are rejected; a generate request
    leaves nothing behind on disk (assert the temp dir is gone).

## Sequencing

Each step ships green, so it can be paused at any point:

1. Add `trace` + inline `svg` to the trace responses (additive, nothing breaks).
2. Add `POST /stl/generate` with strict payload validation, alongside the
   existing endpoint.
3. Switch the UI to the new endpoints; keep the old ones serving until it lands.
4. Delete the job directory, `/files/`, the sweeper, and their tests.
5. Simplify `render.yaml` — no `JOB_TTL_HOURS`, no disk assumptions.

## What this buys

- The whole class of "server forgot my job" failures disappears rather than
  being recovered from.
- No user data is stored server-side at all, so the "anyone with the URL can
  fetch it" caveat goes away.
- The app stops needing a writable disk, which makes it portable to hosts that
  never had one — including serverless, where the disk requirement (not the
  compute) was the blocker.
- Less code: two endpoints, the sweeper, the path guards, and the recovery
  logic all go.

## Cost

Roughly a day of focused work, and it is a breaking API change for anyone
scripting against `/trace/from-job`, `/stl/from-job`, or `/files/`. Worth
doing before the API has real users; the CLI is unaffected.
