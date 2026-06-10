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
- user auth
- quota controls
- object storage for artifacts
