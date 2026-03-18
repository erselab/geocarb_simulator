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
| **Model sampling** | Bilinear interpolation of 3-D/4-D model fields along ray paths (`model_sampler.py`) |
| **Curvilinear grids** | WRF Lambert / polar-stereographic 2-D lat/lon grids via KDTree + inverse bilinear |
| **Limited-area domains** | Out-of-domain detection for regional models; `outside_domain` mask returned per intercept |
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

Optional (for map projections in `plot_scan_blocks` and land fraction):
```
cartopy >= 0.21
```

Optional (for curvilinear / 2-D grid support in `model_sampler`):
```
scipy >= 1.7
```

To run the demo notebook:
```
jupyterlab
```

Install with:
```bash
pip install numpy matplotlib
pip install scipy            # optional — curvilinear grids
pip install cartopy          # optional — map projections / land fraction
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

#### `geocarb_demo(verbose=True)`

End-to-end demonstration: builds GeoCarb-like scan blocks, exercises
`coarsen_scan_block`, save/load, ray tracing, and plotting.  Returns a dict
with keys `blocks`, `coarse_blocks`, and timing results.  Useful as a quick
integration test or for getting familiar with the API.

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

---

## `model_sampler.py` — 3-D Model Field Sampler

Bridges ScanBlock geometry with atmospheric model output.  Bilinearly
interpolates any 3-D (or 4-D) field along the ray paths defined by a block,
returning sampled values and slant-path lengths for column integration.

### Requirements

```
numpy >= 1.21
```

Optional (required for **curvilinear grids** only):
```
scipy >= 1.7    # scipy.spatial.KDTree
```

### Quick Start

```python
import numpy as np
from geosat_geometry import LongSlitGeoSatellite
from model_sampler import (
    sample_field_along_rays,
    integrate_along_rays,
    interp_model_alts_to_block,
    pressure_to_alt_std_atm,
)

sat   = LongSlitGeoSatellite(sat_lon_deg=-95.0)
block = sat.build_scan_block(30.0, -120.0, -95.0)

# Standard pressure levels → km using US Std Atm 1976
pres_hpa   = np.array([1000, 850, 700, 500, 300, 200, 100, 50, 10])
level_alts = pressure_to_alt_std_atm(pres_hpa)  # (nlev,)

# Load model field: (nlev, nlat, nlon) from NetCDF / numpy
model_lats = np.linspace(-90, 90, 181)
model_lons = np.linspace(-180, 180, 361)
co2_field  = ...   # (nlev, nlat, nlon)

# Sample + integrate
result = sample_field_along_rays(block, sat, co2_field,
                                 model_lats, model_lons, level_alts)
column = integrate_along_rays(result['sampled'], result['slant_lengths'])
# column: (n_rows, n_cols)  in units of [field_units × km]
```

### Array Conventions

| Array | Shape | Notes |
|---|---|---|
| `field` | `(nlev, nlat, nlon)` **or** `(ntime, nlev, nlat, nlon)` | Level is always leading (or second) axis |
| `model_lats` / `model_lons` | `(nlat,)` / `(nlon,)` **or** `(nlat, nlon)` / `(nlat, nlon)` | 1-D for regular grids; 2-D for curvilinear |
| `level_alts_km` | `(nlev,)` **or** `(nlev, n_rows, n_cols)` | 1-D shared; 3-D per-pixel (terrain-following) |
| `sampled` | `(nlev, n_rows, n_cols)` | Field at ray intercepts |
| `slant_lengths` | `(nlev-1, n_rows, n_cols)` | Layer path lengths [km] |
| `outside_domain` | `(nlev, n_rows, n_cols)` bool | True where intercept is outside model grid |

### Functions

#### `sample_field_along_rays(block, sat, field_3d, model_lats, model_lons, level_alts_km, *, fill_value=nan, valid_mask=None, time_idx=0)`

Sample a model field along all ray paths in a ScanBlock.

| Parameter | Description |
|---|---|
| `block` | `ScanBlock` from `build_scan_block` |
| `sat` | `LongSlitGeoSatellite` used to build the block |
| `field_3d` | `(nlev, nlat, nlon)` or `(ntime, nlev, nlat, nlon)` model field |
| `model_lats` | `(nlat,)` monotone **or** `(nlat, nlon)` curvilinear |
| `model_lons` | `(nlon,)` monotone **or** `(nlat, nlon)` curvilinear |
| `level_alts_km` | `(nlev,)` standard altitudes **or** `(nlev, n_rows, n_cols)` per-pixel |
| `time_idx` | Time slice to extract when field is 4-D (default 0) |

Returns a dict with keys: `sampled`, `slant_lengths`, `lat_intercepts`,
`lon_intercepts`, `outside_domain`.

---

#### `integrate_along_rays(sampled, slant_lengths, layer_weights=None, normalize=False)`

Trapezoidal slant-column integral:

```
column[i,j] = Σ_k  0.5*(f[k,i,j] + f[k+1,i,j]) * slant[k,i,j]
```

Optional `layer_weights (nlev-1,)` or `(nlev-1, n_rows, n_cols)` for
pressure-weighted or number-density-weighted integrals.

When `normalize=True`, returns the path-length-weighted column-average mole
fraction (analogous to XCO₂) instead of the raw integral:

```
X[i,j] = Σ_k f_avg[k] * slant[k] * w[k]  /  Σ_k slant[k] * w[k]
```

---

#### `interp_model_alts_to_block(alt_3d, model_lats, model_lons, block, *, valid_mask=None, time_idx=0)`

Bilinearly interpolate a 3-D (or 4-D) geopotential height field
`(nlev, nlat, nlon)` to every pixel centre, returning
`(nlev, n_rows, n_cols)` per-pixel altitudes suitable for passing as
`level_alts_km` to `sample_field_along_rays`.

Accepts both 1-D regular and 2-D curvilinear `model_lats`/`model_lons`.
Pixels outside a limited-area domain are filled with NaN.

---

#### `pressure_to_alt_std_atm(pressure_hpa)` / `alt_to_pressure_std_atm(alt_km)`

US Standard Atmosphere 1976, accurate to < 0.1 km from surface to 86 km.
Fully vectorised; accepts scalars or arrays.

---

### Grid Types

#### Regular (1-D) grids

Pass `model_lats` as a 1-D `(nlat,)` array (ascending or descending) and
`model_lons` as `(nlon,)`.  Both 0–360 and −180–180 longitude conventions
are handled automatically.  Uses NumPy advanced indexing (no `scipy`).

#### Curvilinear (2-D) grids

Pass both `model_lats` and `model_lons` as `(nlat, nlon)` 2-D arrays (e.g.
from a WRF Lambert conformal output).  Internally:

1. Builds a `scipy.spatial.KDTree` on 3-D Cartesian cell centres to find
   the nearest grid cell — handles the −180/180 longitude seam.
2. Solves the inverse bilinear map analytically (quadratic in `s`, then
   linear in `t`) to get fractional cell coordinates.
3. Marks cells where `|s|` or `|t|` exceeds the `_OOD_TOL = 0.1` threshold
   as `outside_domain = True`.

---

### Terrain-Following / Hybrid-Sigma Coordinates

```python
# 1. Load the model geopotential height field: gph_km (nlev, nlat, nlon)
level_alts_px = interp_model_alts_to_block(
    gph_km, model_lats, model_lons, block
)  # (nlev, n_rows, n_cols)

# 2. Sample with per-pixel altitudes
result = sample_field_along_rays(
    block, sat, co2_field, model_lats, model_lons,
    level_alts_km=level_alts_px,
)
```

When `level_alts_km` is 3-D, `sample_field_along_rays` uses a fully
vectorised per-pixel quadratic ray solver where the sphere radius
`r[i,j] = RE + level_alts_km[k,i,j]` varies per pixel per level — no
Python loops, ~20% overhead vs. fixed levels.

---

### Limited-Area Domains

For regional models (WRF, CMAQ, …) whose grid does not cover the full
ScanBlock:

```python
result = sample_field_along_rays(block, sat, field, lat_2d, lon_2d, alts)

# (nlev, n_rows, n_cols) bool — True where intercept is outside the grid
ood = result['outside_domain']

# Pixel-level summary — True if ALL levels are outside
ood_pixel = ood.all(axis=0)   # (n_rows, n_cols)

# Restrict column integral to pixels inside the domain
valid_inside = (block['pixel_counts'] > 0) & ~ood_pixel
column_inside = integrate_along_rays(
    result['sampled'], result['slant_lengths']
)
```

`sampled` is filled with `fill_value` (default NaN) at outside-domain
intercepts, so `integrate_along_rays` uses `np.nansum` and naturally
ignores them.

---

### Performance (`model_sampler`)

| Operation | Size | Time |
|---|---|---|
| `sample_field_along_rays` (regular grid) | 4 k coarse pixels × 10 levels | ~5 ms |
| `sample_field_along_rays` (curvilinear, KDTree build) | 4 k pixels, 60×80 WRF grid | ~50 ms |
| `interp_model_alts_to_block` | 4 k pixels × 10 levels | ~2 ms |
| Bilinear hot path | 200 k pixels × 50 levels | ~150 ms |

---

## `scan_sampler.py` — NetCDF Model-to-Output Pipeline

High-level wrapper that reads an atmospheric model NetCDF file, samples
requested tracer fields along scan-block ray paths, and writes
pressure-weighted column-average mole fractions together with full
observation geometry and land/water fraction to a CF-compliant NetCDF4
output file.

### Requirements

```
netCDF4 >= 1.6
numpy >= 1.21
```

Optional (land fraction):
```
cartopy >= 0.21
```

Optional (curvilinear model grids):
```
scipy >= 1.7
```

### Quick Start

```python
from datetime import datetime, timezone
from geosat_geometry import LongSlitGeoSatellite
from scan_sampler import NCConfig, run_scan_sampler

sat   = LongSlitGeoSatellite(sat_lon_deg=-95.0)
block = sat.build_scan_block(30.0, -120.0, -95.0,
                             t0_utc=datetime(2020, 7, 1, 18, tzinfo=timezone.utc))

# Default NCConfig works for files with variables named lat/lon/lev/time (hPa)
run_scan_sampler(
    nc_path   = 'model_output.nc',
    block     = block,
    sat       = sat,
    tracers   = ['CO2', 'CH4'],
    out_path  = 'xgas_west_scan.nc',
    t_utc     = datetime(2020, 7, 1, 18, tzinfo=timezone.utc),
    dlat_deg  = 0.5,   # coarsen to 0.5° before sampling
    dlon_deg  = 0.5,
)
```

### `NCConfig`

Dataclass that maps variable roles to names in the model NetCDF file.

| Field | Default | Description |
|---|---|---|
| `lat_var` | `"lat"` | Latitude coordinate — 1-D `(nlat,)` or 2-D `(nlat, nlon)` |
| `lon_var` | `"lon"` | Longitude coordinate |
| `lev_var` | `"lev"` | Pressure level coordinate [hPa] — 1-D `(nlev,)` |
| `time_var` | `"time"` | CF-convention time coordinate; `None` = no time dimension |
| `lev_in_pa` | `False` | Set `True` if `lev_var` is in Pa (values ÷ 100) |
| `gph_var` | `None` | Optional geopotential height variable for terrain-following coordinates |
| `gph_in_m` | `True` | Set `True` if `gph_var` is in metres (÷ 1000 → km) |

Common model conventions:

```python
# ERA5
cfg = NCConfig(lat_var='latitude', lon_var='longitude', lev_var='level')

# WRF (Pa pressure, 2-D lat/lon, geopotential height available)
cfg = NCConfig(lat_var='XLAT', lon_var='XLONG',
               lev_var='P_HYD', lev_in_pa=True,
               gph_var='Z', gph_in_m=True)
```

### `run_scan_sampler(...)`

```python
run_scan_sampler(
    nc_path,                      # str | Path — model NetCDF file
    block,                        # ScanBlock
    sat,                          # LongSlitGeoSatellite
    tracers,                      # list[str] — tracer variable names
    out_path,                     # str | Path — output NetCDF4 file
    t_utc        = None,          # datetime — selects nearest model time step
    dlat_deg     = None,          # float — coarsen block to this lat resolution [°]
    dlon_deg     = None,          # float — coarsen block to this lon resolution [°]
    nc_config    = NCConfig(),    # variable name mapping
    save_profiles = False,        # also write per-level sampled profiles
    land_mask_resolution_deg = 0.1,   # land mask raster resolution [°]
    natural_earth_scale = '110m',     # Natural Earth coastline scale
)
```

Returns the absolute path to the written file as a string.

### Output file variables

All arrays have shape `(row, col)` unless noted.

**Geometry**

| Variable | Units | Description |
|---|---|---|
| `lat`, `lon` | ° | Pixel-centre coordinates |
| `vza`, `vaa` | ° | Viewing zenith / azimuth angle |
| `sza`, `saa` | ° | Solar zenith / azimuth angle |
| `airmass_geometric` | — | sec(VZA) + sec(SZA) |
| `relative_azimuth` | ° | \|VAA − SAA\| |
| `land_fraction` | — | 0 = water, 1 = land (requires cartopy) |
| `valid` | — | 1 = valid pixel, 0 = empty / masked |
| `pixel_counts` | — | Fine pixels per cell (coarsened blocks only) |
| `outside_domain` | — | 1 where all ray intercepts are outside model grid |

**Pressure levels and tracers**

| Variable | Shape | Description |
|---|---|---|
| `pressure_levels` | `(nlev,)` | Model pressure levels [hPa], surface first |
| `x_{tracer}` | `(row, col)` | Pressure-weighted column-average mole fraction |
| `profile_{tracer}` | `(nlev, row, col)` | Sampled values at each level (`save_profiles=True` only) |

### Column-average formula

The pressure-weighted column average is:

```
X[i,j] = Σ_k  q̄[k,i,j] · Δp[k]
          ─────────────────────────────
          Σ_k  Δp[k]  (finite layers)
```

where `q̄[k] = ½(q[k] + q[k+1])` is the layer-mean mole fraction and
`Δp[k] = p[k] − p[k+1]` is the pressure thickness.  Out-of-domain layers
(NaN) are excluded from both numerator and denominator, so partially sampled
columns (e.g. at the edge of a limited-area model domain) still yield a
valid estimate.
