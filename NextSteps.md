# Next steps (recommended)

## 1) Improve outline validation
- lattice mode handles connected grid line art (auto-detected or forced)
- single-shape mode still expects one primary silhouette
- add min/max area checks for single-shape uploads

## 2) Better tracing controls
Expose in UI/API:
- threshold
- simplify epsilon
- optional smoothing

## 3) STL profiles
Add selectable profiles:
- current: circle-reference topology
- sharpened cutting lip
- rounded/chamfered press edge
- different flange shapes

## 4) Caching
Hash the uploaded PNG + params -> reuse existing output.

## 5) Auth & rate limits (if you ever host)
- ~~passphrase auth~~ — done (`ACCESS_PASSWORD` + `SESSION_SECRET`)
- ~~per-IP rate limiting + concurrency cap~~ — done (`RATE_LIMIT_PER_MINUTE`, `HEAVY_CONCURRENCY`)
- per-user accounts / quotas (only if it ever needs multi-tenancy)
- object storage for artifacts
