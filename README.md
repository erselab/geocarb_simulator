# geosat_geometry — Long-Slit Geostationary Satellite Geometry Simulator

Efficient Python module for simulating the observation geometry of a long-slit
geostationary (GEO) instrument (e.g. **GeoCarb**).  Computes pixel centres,
footprint polygons, viewing and solar angles, satellite look vectors, and full
3-D ray paths through the atmosphere — everything needed to drive a radiative
transfer or atmospheric inversion model.

---

## Features

| Capability | Details |
|---|---|
| **Slit geometry** | All 500 pixels along a 3000 km slit in < 0.5 ms |
| **ScanBlock** | Pre-computed 2-D (N-S × E-W) geometry arrays ready for model ingestion |
| **Per-column solar angles** | Sun position advanced by integration time at each E-W step |
| **Day scheduling** | Chain multiple scan targets into a full-day observing schedule |
| **Ray tracing** | Satellite→ground ray intersections with spherical atmospheric shells |
| **Land/water fraction** | Compute per-pixel land fraction using Natural Earth coastlines |
| **Coarsening** | Bin fine (6 km) pixels onto any regular lat/lon grid (e.g. 0.5°×0.5°) |
| **Plotting** | Built-in map visualization with optional Cartopy basemap |
| **Save / Load** | Round-trip persistence via compressed `.npz` |

---

## Requirements

```
numpy >= 1.21
matplotlib >= 3.5
```

Optional (for map projections in `plot_scan_blocks`):
```
cartopy >= 0.21
```

Install with:
```bash
pip install numpy matplotlib
pip install cartopy          # optional
```

---

## Quick Start

```python
from geosat_geometry import LongSlitGeoSatellite, plot_scan_blocks, coarsen_scan_block
from datetime import datetime
import numpy as np

# 1 — Create a GeoCarb-like satellite
sat = LongSlitGeoSatellite(
    sat_lon_deg      = -95.0,   # sub-satellite longitude
    slit_length_km   = 3000.0,  # N-S slit length at nadir
    pixel_size_ew_km =    6.0,  # E-W pixel size at nadir
    pixel_size_ns_km =    6.0,  # N-S pixel size at nadir
    integration_time_s = 10.0,  # integration time per scan step
)

# 2 — Build a ScanBlock for a region of interest
block = sat.build_scan_block(
    slit_center_lat = 30.0,
    scan_start_lon  = -110.0,
    scan_end_lon    =  -85.0,
    t0_utc = datetime(2020, 7, 1, 18, 0, 0),
)
print(block)
# ScanBlock(500×401 px | lat=30.0° | lon=[-110.0°,-85.0°] | dur=66.7 min ...)

# 3 — Access 2-D geometry arrays  (axis 0 = N-S, axis 1 = E-W)
block['vzas']           # (500, 401)  viewing zenith angle [deg]
block['szas']           # (500, 401)  solar zenith angle   [deg]
block['sat_look_vecs']  # (500, 401, 3)  unit vector sat→gnd (ECEF)
block['corner_lats']    # (500, 401, 4)  footprint polygon corners

# 4 — Coarsen to 0.5° × 0.5° for a CTM
coarse = block.coarsen(dlat_deg=0.5, dlon_deg=0.5)

# 5 — Build a full-day observing schedule
from geosat_geometry import ScanTarget, build_day_schedule
targets = [
    ScanTarget(30.0, -120.0, -95.0, label='West'),
    ScanTarget(30.0, -95.0, -75.0, label='Central'),
    ScanTarget(30.0, -75.0, -60.0, label='East'),
]
day_blocks = build_day_schedule(sat, datetime(2020, 7, 1, 12, 0), targets)

# 6 — Add land/water fraction to a ScanBlock
from geosat_geometry import add_land_fraction
block_with_land = add_land_fraction(block)

# 7 — Plot
from geosat_geometry import plot_scan_blocks
fig, ax = plot_scan_blocks(block, field='vzas')

# 8 — Ray paths through atmosphere
alts = np.arange(0, 80, 1.0)   # 0–79 km shells
rays = sat.compute_ray_paths_vectorized(
    block['lats'].ravel(), block['lons'].ravel(), alts)
# rays['intercept_pts']  shape (N_pixels, 80, 3)  ECEF [km]
# rays['slant_lengths']  shape (N_pixels, 79)      km per layer

# 9 — Save / load
block.save('west_block.npz')
block2 = ScanBlock.load('west_block.npz')
```

---

## Module Contents

### Classes

#### `LongSlitGeoSatellite`

```python
LongSlitGeoSatellite(
    sat_lon_deg,
    slit_length_km     = 3000.0,
    pixel_size_ew_km   = 6.0,
    pixel_size_ns_km   = 6.0,
    integration_time_s = 10.0,
    sat_alt_km         = 35786.0,  # nominal GEO altitude
    scan_rate_kms      = None,     # defaults to pixel_size_ew / integration_time
)
```

#### Dataclasses

| Name | Description |
|---|---|
| `ObservationGeometry` | Per-pixel geometry for a single integration (lat/lon, angles, vectors) |
| `RayPath` | Atmospheric ray intercept points for a single ground pixel |
| `ScanTarget` | Scan region definition for `build_day_schedule()` |

Key attributes:

| Attribute | Description |
|---|---|
| `sat_ecef` | Satellite ECEF position [km], shape (3,) |
| `n_pixels` | Number of detector pixels along slit |
| `ifov_ew_rad` / `ifov_ns_rad` | Instantaneous FOV [radians] |

Key methods:

| Method | Returns | Notes |
|---|---|---|
| `build_scan_block(lat, lon0, lon1, t0_utc)` | `ScanBlock` | Primary interface |
| `slit_geometry_vectorized(lat, lon, dt_utc)` | `dict` | Single slit position |
| `compute_ray_paths_vectorized(lats, lons, alts)` | `dict` | Batch ray tracing |
| `ray_path(lat, lon, alts)` | `RayPath` | Single pixel ray path |
| `simulate_scan(lat, lon0, lon1, t0_utc)` | `list[ObservationGeometry]` | Legacy per-pixel list |
| `build_day_schedule(sat, t_start, targets)` | `list[ScanBlock]` | Sequence of blocks for a full day |
| `add_land_fraction(block)` | `ScanBlock` | Adds per-pixel land fraction field |
| `geocarb_demo()` | `dict` | Demo creating blocks, coarsening, and plotting |

---

#### `ScanBlock`

Pre-computed 2-D geometry block.  Dict-like: `block['vzas']`, `block.keys()`,
`'szas' in block`.  All arrays shaped **(n_rows, n_cols)** or **(n_rows, n_cols, K)**.

**2-D scalar fields**

| Field | Units | Description |
|---|---|---|
| `lats` | deg | Geodetic latitude of pixel centre |
| `lons` | deg | Longitude of pixel centre |
| `vzas` | deg | Viewing zenith angle |
| `vaas` | deg | Viewing azimuth angle (clockwise from North) |
| `szas` | deg | Solar zenith angle (NaN if no `t0_utc`) |
| `saas` | deg | Solar azimuth angle (NaN if no `t0_utc`) |
| `airmass_geometric` | — | sec(VZA) + sec(SZA) |
| `relative_azimuth` | deg | \|VAA − SAA\| ∈ [0°, 180°] |
| `pixel_counts`* | int | Fine pixels per coarse cell (*coarsened blocks only*) |

**3-D vector / corner fields**

| Field | Shape suffix | Description |
|---|---|---|
| `sat_look_vecs` | (3,) | Unit vector satellite → ground (ECEF) |
| `gnd_ecef` | (3,) | Ground point ECEF position [km] |
| `corner_lats` | (4,) | Footprint corners NW→NE→SE→SW [deg] |
| `corner_lons` | (4,) | Footprint corners NW→NE→SE→SW [deg] |

**Metadata attributes**

`sat_lon_deg`, `sat_ecef`, `slit_center_lat`, `scan_start_lon`, `scan_end_lon`,
`scan_lons`, `integration_time_s`, `scan_duration_s`, `t_start_utc`,
`t_end_utc`, `col_times`, `n_rows`, `n_cols`.

**Methods**

```python
block.coarsen(dlat_deg, dlon_deg, min_count=1)  # → ScanBlock
block.save('path/to/file')                       # writes .npz
ScanBlock.load('path/to/file.npz')              # → ScanBlock
block.shape   # (n_rows, n_cols)
```

---

### Functions

#### `coarsen_scan_block(block, dlat_deg, dlon_deg, min_count=1)`

Bin a fine-resolution ScanBlock onto a regular lat/lon grid.

```python
coarse = coarsen_scan_block(block, dlat_deg=0.5, dlon_deg=0.5)
coarse['pixel_counts']  # how many 6 km pixels fell in each 0.5° cell
```

Aggregation rules:

| Field type | Method |
|---|---|
| Scalar angles (VZA, SZA, airmass) | Arithmetic mean |
| Azimuth angles (VAA, SAA) | Circular mean — handles 0°/360° wrap |
| `sat_look_vecs` | Vector sum → re-normalised (mean direction) |
| `gnd_ecef` | Arithmetic mean of ECEF positions |
| `corner_lats/lons` | Derived directly from coarse grid edges |

Grid edges are aligned to global multiples of `dlat/dlon` so adjacent
blocks at the same resolution share a consistent grid.

---

#### `build_day_schedule(satellite, t_start, targets, t_end=None, repeat=True)`

Build a chronological sequence of `ScanBlock` objects for a full day of scans.
Each scan block advances the UTC time by `integration_time_s` per column and
can be labelled via `ScanTarget.label`.

#### `add_land_fraction(block, mask_resolution_deg=0.1, natural_earth_scale='110m')`

Compute per-pixel land fraction using Natural Earth coastline polygons. Adds a
`land_fraction` field to the returned `ScanBlock` (0.0 = water, 1.0 = land).

---

#### `plot_scan_blocks(blocks, field, mode, ...)`

```python
fig, ax = plot_scan_blocks(
    blocks,                    # ScanBlock or list[ScanBlock]
    field = 'vzas',            # any 2-D ScanBlock field
    mode  = 'footprints',      # 'footprints' | 'pixels' | 'outline'
    cmap  = 'viridis',
    vmin  = None, vmax = None, # defaults: 2nd/98th percentile
    alpha = 0.85,
    show_colorbar     = True,
    show_sat_subpoint = True,
    coastlines        = True,  # uses Cartopy if installed
    block_labels      = ['West', 'Central', 'East'],
    title             = None,
    figsize           = (13, 7),
)
```

---

#### `solar_position_ecef(dt_utc)` → `np.ndarray` shape (3,)

Low-precision (~1°) Sun unit vector in ECEF.  Sufficient for computing
solar geometry in observation simulations.

---

#### `geodetic_to_ecef(lat, lon, alt_km=0)` / `ecef_to_geodetic(xyz)`

WGS-84 coordinate transforms.  Both accept scalar or array inputs.

---

## Coordinate Systems

| System | Definition |
|---|---|
| **ECEF** | Earth-Centered Earth-Fixed, km. +X → (0°N, 0°E), +Z → North Pole |
| **ENU** | Local East-North-Up at a ground point |
| **Geodetic** | WGS-84 latitude, longitude [degrees], altitude [km] |
| **Angles** | Degrees throughout; azimuths clockwise from North |

---

## 3-D Ray Tracing

```python
alts = np.arange(0, 80, 1.0)          # shell altitudes [km]
rays = sat.compute_ray_paths_vectorized(
    block['lats'].ravel(),
    block['lons'].ravel(),
    alts,
)
```

Returns:

| Key | Shape | Description |
|---|---|---|
| `intercept_pts` | (N, M, 3) | ECEF [km] of shell entry point |
| `slant_lengths` | (N, M-1) | Path length in each layer [km] |
| `t_params` | (N, M) | Parametric t (0 = satellite, 1 = ground) |
| `lat_intercepts` | (N, M) | Geodetic latitude at intercept [deg] |
| `lon_intercepts` | (N, M) | Longitude at intercept [deg] |
| `alt_intercepts` | (N, M) | Altitude at intercept [km] (≈ `alts`) |

Shells are spherical centred on Earth's centre (radius = RE + altitude).
The satellite is always above all shells; the ray enters each shell from above
at `t1` (the smaller quadratic root).

---

## Scan Geometry Notes

- The slit is oriented **N–S**; detector rows run along the slit.
- The instrument **steps E–W** one pixel width per integration period.
- `build_scan_block` advances the UTC time by `integration_time_s` per column,
  so solar angles vary correctly across an E–W scan.
- For GeoCarb at −95° longitude, a full CONUS scan (~60°) takes ≈ 100 min.
- The satellite position is fixed (GEO); only solar geometry changes with time.

---

## Performance (Apple M-series, NumPy 2.x)

| Operation | Size | Time |
|---|---|---|
| `build_scan_block` | 500 px × 400 cols | ~30 ms |
| `coarsen_scan_block` | 200 k pixels → 0.5° grid | ~25 ms |
| `compute_ray_paths_vectorized` | 200 k pixels × 80 shells | ~800 ms |
| `plot_scan_blocks` (footprints) | 3 blocks | ~1 s |
