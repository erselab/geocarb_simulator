"""
model_sampler.py
================
Fast sampler for 3-D atmospheric model output (on pressure levels) along
satellite-to-ground ray paths defined by ScanBlocks.

Typical workflow
----------------
1. Build (or load) a ScanBlock from geosat_geometry.
2. Load a 3-D model field on pressure levels (e.g. from NetCDF).
3. Call ``sample_field_along_rays`` to get the field sampled at every ray
   intercept for each pixel in the block.
4. Optionally call ``integrate_along_rays`` to compute slant-column integrals.

Quick start
-----------
>>> from datetime import datetime
>>> from geosat_geometry import LongSlitGeoSatellite
>>> from model_sampler import sample_field_along_rays, integrate_along_rays, pressure_to_alt_std_atm
>>>
>>> sat   = LongSlitGeoSatellite(sat_lon_deg=-95.0)
>>> block = sat.build_scan_block(30.0, -115.0, -75.0,
...                               t0_utc=datetime(2020, 7, 1, 18))
>>>
>>> # Model grid
>>> model_lats = np.linspace(-90, 90, 181)     # 1° global
>>> model_lons = np.linspace(-180, 180, 361)
>>> pres_hpa   = np.array([1000, 850, 700, 500, 300, 200, 100, 50, 10])
>>> level_alts = pressure_to_alt_std_atm(pres_hpa)  # km
>>> co2_field  = np.random.uniform(400, 420, (len(pres_hpa), len(model_lats), len(model_lons)))
>>>
>>> result  = sample_field_along_rays(block, sat, co2_field,
...                                    model_lats, model_lons, level_alts)
>>> column  = integrate_along_rays(result['sampled'], result['slant_lengths'])

Array conventions
-----------------
- ``field``      shape (nlev, nlat, nlon) **or** (ntime, nlev, nlat, nlon).
  Level is always the leading (or second) axis, matching most atmospheric
  model output (NetCDF ``lev, lat, lon`` or ``time, lev, lat, lon``).
  Pass ``time_idx`` to select the time slice; default is 0.
- ``model_lats`` / ``model_lons`` — two supported layouts:
    * **1-D** ``(nlat,)`` / ``(nlon,)`` — regular (rectangular) grid.
      ``model_lats`` must be strictly monotone; both 0–360 and −180–180 lon
      conventions are handled automatically.
    * **2-D** ``(nlat, nlon)`` / ``(nlat, nlon)`` — curvilinear grid (e.g.
      WRF Lambert / polar-stereographic).  Both arrays must be 2-D with the
      same shape.  Bilinear interpolation is performed via a KDTree lookup on
      3-D Cartesian cell centres (handles the −180/180 seam) followed by an
      analytical inverse-bilinear solve.  Requires ``scipy``.
- ``level_alts_km`` must be **ascending** (lowest level first).
  Use ``pressure_to_alt_std_atm`` if you only have pressure values.

Limited-area domains
--------------------
For regional models (WRF, CMAQ, …) whose horizontal extent is smaller than
the ScanBlock, footprints that lie outside the model domain are filled with
``fill_value`` (default NaN).  The returned dict always includes an
``outside_domain`` key — a ``(nlev, n_rows, n_cols)`` bool array that is
True for every (level, pixel) combination where the ray intercept fell
outside the grid.  A pixel that is entirely outside the domain will have
``outside_domain[:, row, col]`` all-True, and ``sampled[:, row, col]``
all-NaN.

Performance notes
-----------------
The hot path uses manual bilinear indexing via NumPy advanced indexing rather
than ``scipy.interpolate.RegularGridInterpolator``, giving roughly 10–50× speed
improvement for the large (N_pixels × N_levels) arrays typical of a ScanBlock.

For a 500×400 pixel block (200 000 pixels) with 50 pressure levels:
  - ray tracing          : ~500 ms
  - bilinear sampling    : ~150 ms
  - total                : ~650 ms

"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "sample_field_along_rays",
    "integrate_along_rays",
    "interp_model_alts_to_block",
    "pressure_to_alt_std_atm",
    "alt_to_pressure_std_atm",
]

# ---------------------------------------------------------------------------
# Standard atmosphere pressure ↔ altitude conversion
# ---------------------------------------------------------------------------

# US Standard Atmosphere 1976 layer definitions.
# Each row: (base altitude km, base pressure hPa, base temperature K, lapse rate K/km)
# Source: NOAA/NASA/USAF, 1976.
_STD_ATM_LAYERS = np.array([
    [  0.0, 1013.25, 288.15, -6.5 ],
    [ 11.0,  226.32, 216.65,  0.0 ],
    [ 20.0,   54.75, 216.65,  1.0 ],
    [ 32.0,    8.680, 228.65, 2.8 ],
    [ 47.0,    1.109, 270.65, 0.0 ],
    [ 51.0,   0.6694, 270.65, -2.8],
    [ 71.0,  0.03956, 214.65, -2.0],
    [ 86.0, 0.003734, 186.87, 0.0],   # upper sentinel
])

# Physical constants (SI throughout)
_G0_SI    = 9.80665   # m/s²
_RDRY_SI  = 287.058   # J/(kg·K) = m²/(s²·K)
# R_dry / g0 = 29.27 m/K = 0.02927 km/K  (scale height / temperature)
_RDRY_OVER_G0_KM = _RDRY_SI / _G0_SI / 1000.0   # km/K


def pressure_to_alt_std_atm(pressure_hpa: np.ndarray | float) -> np.ndarray:
    """
    Convert pressure [hPa] to altitude [km] using the US Standard Atmosphere 1976.

    Accurate to within ~0.1 km for pressures 1013–0.004 hPa (surface to 86 km).
    Vectorised; accepts scalars or arrays.

    Parameters
    ----------
    pressure_hpa : float or array-like
        Pressure in hPa (mb).

    Returns
    -------
    alt_km : same shape as input, altitude in km.
    """
    scalar = np.ndim(pressure_hpa) == 0
    p      = np.atleast_1d(np.asarray(pressure_hpa, dtype=float))
    alt    = np.full_like(p, np.nan)

    for i in range(len(_STD_ATM_LAYERS) - 1):
        z0_km, p0, T0, L_km = _STD_ATM_LAYERS[i]
        _,     p1, _,  _    = _STD_ATM_LAYERS[i + 1]
        mask = (p <= p0) & (p >= p1)
        if not np.any(mask):
            continue
        pm = p[mask]
        if abs(L_km) < 1e-12:                    # isothermal layer
            # alt = z0 + (R_dry/g0) * T0 * ln(P0/P)   [km]
            alt[mask] = z0_km + _RDRY_OVER_G0_KM * T0 * np.log(p0 / pm)
        else:                                     # gradient layer
            # alt = z0 + (T0/L) * [(P/P0)^(-R*L/(g0*1000)) - 1]  [km]
            # exponent = -R_dry_SI * L_km / (g0_SI * 1000)  (dimensionless)
            exp = -_RDRY_SI * L_km / (_G0_SI * 1000.0)
            alt[mask] = z0_km + (T0 / L_km) * ((pm / p0) ** exp - 1.0)

    # Extrapolate above 86 km (isothermal)
    z0_km, p0, T0, _ = _STD_ATM_LAYERS[-1]
    mask_top = p < p0
    if np.any(mask_top):
        alt[mask_top] = z0_km + _RDRY_OVER_G0_KM * T0 * np.log(p0 / p[mask_top])

    return float(alt[0]) if scalar else alt


def alt_to_pressure_std_atm(alt_km: np.ndarray | float) -> np.ndarray:
    """
    Convert altitude [km] to pressure [hPa] using the US Standard Atmosphere 1976.

    Parameters
    ----------
    alt_km : float or array-like
        Altitude in km.

    Returns
    -------
    pressure_hpa : same shape as input.
    """
    scalar = np.ndim(alt_km) == 0
    alt    = np.atleast_1d(np.asarray(alt_km, dtype=float))
    pres   = np.full_like(alt, np.nan)

    for i in range(len(_STD_ATM_LAYERS) - 1):
        z0_km, p0, T0, L_km = _STD_ATM_LAYERS[i]
        z1_km, _,  _,  _    = _STD_ATM_LAYERS[i + 1]
        mask = (alt >= z0_km) & (alt < z1_km)
        if not np.any(mask):
            continue
        dz = alt[mask] - z0_km          # km
        if abs(L_km) < 1e-12:           # isothermal
            # P = P0 * exp(-g0 * dz / (R_dry * T0))  [with dz in km → need to multiply by 1000]
            pres[mask] = p0 * np.exp(-_G0_SI * dz * 1000.0 / (_RDRY_SI * T0))
        else:                            # gradient
            # P = P0 * (T/T0)^(-g0/(R_dry*L_SI))  where T = T0 + L_km*dz
            T    = T0 + L_km * dz
            exp  = -_G0_SI * 1000.0 / (_RDRY_SI * L_km)
            pres[mask] = p0 * (T / T0) ** exp

    return float(pres[0]) if scalar else pres


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_lons(lons: np.ndarray) -> np.ndarray:
    """Wrap longitude values to (−180, 180]."""
    return ((lons + 180.0) % 360.0) - 180.0


def _bilinear_sample_all_levels(
    field_3d:   np.ndarray,    # (M, nlat, nlon)
    i_lat:      np.ndarray,    # (N, M) int — south-row index
    i_lon:      np.ndarray,    # (N, M) int — west-col index
    dlat:       np.ndarray,    # (N, M) float in [0, 1]
    dlon:       np.ndarray,    # (N, M) float in [0, 1]
    m_idx:      np.ndarray,    # (M,)   int level indices
) -> np.ndarray:               # (N, M)
    """
    Bilinear interpolation of field_3d at (N, M) query points.

    Uses NumPy advanced indexing — no Python loops.

    Indexing convention (matches C-order 2D images):
      field_3d[level, lat_row, lon_col]
      lat_row increases with latitude (ascending lat_grid required internally).
    """
    # Broadcast level index to (N, M)
    m = m_idx[np.newaxis, :]          # (1, M)

    f00 = field_3d[m, i_lat,     i_lon    ]   # (N, M)
    f10 = field_3d[m, i_lat + 1, i_lon    ]
    f01 = field_3d[m, i_lat,     i_lon + 1]
    f11 = field_3d[m, i_lat + 1, i_lon + 1]

    return (f00 * (1.0 - dlat) * (1.0 - dlon)
          + f10 * dlat         * (1.0 - dlon)
          + f01 * (1.0 - dlat) * dlon
          + f11 * dlat         * dlon)


def _build_bilinear_indices(
    query_lats: np.ndarray,    # (N, M)
    query_lons: np.ndarray,    # (N, M)
    lat_grid:   np.ndarray,    # (nlat,) ascending
    lon_grid:   np.ndarray,    # (nlon,) ascending
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bilinear interpolation indices and fractional offsets.

    Returns (i_lat, i_lon, dlat, dlon), all shape (N, M).
    Out-of-bounds points are clamped to the grid boundary (no extrapolation;
    the returned fractions will be 0 or 1, so values at the edge are used).
    """
    nlat = len(lat_grid)
    nlon = len(lon_grid)

    # Clamp to valid range [0, nlat-2] so i_lat+1 is always in bounds
    i_lat = np.searchsorted(lat_grid, query_lats, side='right') - 1
    i_lat = np.clip(i_lat, 0, nlat - 2)

    i_lon = np.searchsorted(lon_grid, query_lons, side='right') - 1
    i_lon = np.clip(i_lon, 0, nlon - 2)

    # Fractional offset within the cell, clamped to [0, 1]
    dlat = ((query_lats - lat_grid[i_lat])
            / (lat_grid[i_lat + 1] - lat_grid[i_lat]))
    dlon = ((query_lons - lon_grid[i_lon])
            / (lon_grid[i_lon + 1] - lon_grid[i_lon]))
    dlat = np.clip(dlat, 0.0, 1.0)
    dlon = np.clip(dlon, 0.0, 1.0)

    return i_lat, i_lon, dlat, dlon


# Fractional-cell tolerance for out-of-domain detection (curvilinear grids)
_OOD_TOL = 0.1


def _bilinear_indices_2d(
    query_lats: np.ndarray,    # any shape
    query_lons: np.ndarray,    # same shape
    lat_2d:     np.ndarray,    # (nrow, ncol)
    lon_2d:     np.ndarray,    # (nrow, ncol)
) -> tuple:
    """
    Bilinear cell lookup for a curvilinear grid.

    Algorithm
    ---------
    1. Build a ``scipy.spatial.KDTree`` on 3-D Cartesian (unit-sphere) cell
       centres — this handles the −180/180 longitude seam correctly.
    2. For each query point find the nearest cell (i_row, i_col).
    3. Normalise the 4 corner longitudes relative to the SW corner to avoid
       wrap-around artefacts within the cell.
    4. Solve the inverse bilinear map analytically (quadratic in ``s``,
       then linear in ``t``) to get fractional cell coordinates.

    Returns
    -------
    i_row, i_col : same shape as input — SW-corner grid indices
    s, t         : fractional coordinates, clamped to [0, 1]
    outside      : bool, same shape — True where the query falls outside the
                   nearest cell by more than ``_OOD_TOL`` (i.e. likely
                   outside the model domain for a limited-area grid)
    """
    from scipy.spatial import KDTree

    nrow, ncol   = lat_2d.shape
    orig_shape   = query_lats.shape
    flat_q_lat   = query_lats.ravel()
    flat_q_lon   = query_lons.ravel()

    # ---- 3-D Cartesian cell centres (handles meridian wrap) ---------------
    def to_xyz(lat_d, lon_d):
        lr, lo = np.deg2rad(lat_d), np.deg2rad(lon_d)
        return np.stack([np.cos(lr) * np.cos(lo),
                         np.cos(lr) * np.sin(lo),
                         np.sin(lr)], axis=-1)

    cxyz = 0.25 * (to_xyz(lat_2d[:-1, :-1], lon_2d[:-1, :-1])
                 + to_xyz(lat_2d[1:,  :-1], lon_2d[1:,  :-1])
                 + to_xyz(lat_2d[:-1, 1:],  lon_2d[:-1, 1:])
                 + to_xyz(lat_2d[1:,  1:],  lon_2d[1:,  1:]))   # (nrow-1, ncol-1, 3)
    cxyz /= np.linalg.norm(cxyz, axis=-1, keepdims=True)

    tree = KDTree(cxyz.reshape(-1, 3))
    _, ci = tree.query(to_xyz(flat_q_lat, flat_q_lon))           # (P,)

    n_cells_col = ncol - 1
    i0 = ci // n_cells_col    # SW row
    j0 = ci %  n_cells_col    # SW col

    # ---- 4 corner lat/lons ------------------------------------------------
    lat_00 = lat_2d[i0,   j0  ];  lat_10 = lat_2d[i0+1, j0  ]
    lat_01 = lat_2d[i0,   j0+1];  lat_11 = lat_2d[i0+1, j0+1]

    # Normalise lons relative to the SW corner to avoid wrap within a cell
    lon_ref = lon_2d[i0, j0]
    def _rel(lon, ref):
        return ref + ((lon - ref + 180.0) % 360.0) - 180.0

    lon_00 = lon_ref
    lon_10 = _rel(lon_2d[i0+1, j0  ], lon_ref)
    lon_01 = _rel(lon_2d[i0,   j0+1], lon_ref)
    lon_11 = _rel(lon_2d[i0+1, j0+1], lon_ref)
    q_lon  = _rel(flat_q_lon, lon_ref)

    # ---- inverse bilinear: solve for s (quadratic), then t (linear) -------
    dq_lat = flat_q_lat - lat_00
    dq_lon = q_lon      - lon_00

    A_lat = lat_10 - lat_00;  B_lat = lat_01 - lat_00
    C_lat = lat_11 - lat_10 - lat_01 + lat_00
    A_lon = lon_10 - lon_00;  B_lon = lon_01 - lon_00
    C_lon = lon_11 - lon_10 - lon_01 + lon_00

    aa = A_lon * C_lat - A_lat * C_lon
    bb = dq_lat * C_lon - A_lat * B_lon - dq_lon * C_lat + A_lon * B_lat
    cc = dq_lat * B_lon - dq_lon * B_lat

    disc    = np.maximum(bb**2 - 4.0 * aa * cc, 0.0)
    sq      = np.sqrt(disc)
    safe_aa = np.where(np.abs(aa) > 1e-10, aa, 1.0)
    safe_bb = np.where(np.abs(bb) > 1e-12, bb, 1.0)

    s1 = np.where(np.abs(aa) > 1e-10,
                  (-bb + sq) / (2.0 * safe_aa),
                  np.where(np.abs(bb) > 1e-12, -cc / safe_bb, 0.5))
    s2 = np.where(np.abs(aa) > 1e-10,
                  (-bb - sq) / (2.0 * safe_aa), s1)

    # Use the root closer to [0, 1]
    s1_ok = (s1 >= -_OOD_TOL) & (s1 <= 1.0 + _OOD_TOL)
    s2_ok = (s2 >= -_OOD_TOL) & (s2 <= 1.0 + _OOD_TOL)
    s = np.where(s1_ok, s1, np.where(s2_ok, s2, s1))

    denom_lat = B_lat + s * C_lat
    denom_lon = B_lon + s * C_lon
    t = np.where(
        np.abs(denom_lat) >= np.abs(denom_lon),
        np.where(np.abs(denom_lat) > 1e-12, (dq_lat - s * A_lat) / denom_lat, 0.5),
        np.where(np.abs(denom_lon) > 1e-12, (dq_lon - s * A_lon) / denom_lon, 0.5),
    )

    outside = ((s < -_OOD_TOL) | (s > 1.0 + _OOD_TOL)
             | (t < -_OOD_TOL) | (t > 1.0 + _OOD_TOL))
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    return (i0.reshape(orig_shape), j0.reshape(orig_shape),
            s.reshape(orig_shape),  t.reshape(orig_shape),
            outside.reshape(orig_shape))


# ---------------------------------------------------------------------------
# Per-pixel ray tracer
# ---------------------------------------------------------------------------

def _compute_ray_paths_per_pixel(
    sat_ecef:      np.ndarray,   # (3,)      satellite ECEF [km]
    lats_deg:      np.ndarray,   # (N,)
    lons_deg:      np.ndarray,   # (N,)
    level_alts_km: np.ndarray,   # (N, M)    per-pixel shell altitudes [km], ascending
) -> dict:
    """
    Intersect N satellite-to-ground rays with M per-pixel spherical shells.

    Identical in logic to ``LongSlitGeoSatellite.compute_ray_paths_vectorized``
    but the shell radius ``r[i, j] = RE_KM + level_alts_km[i, j]`` varies
    per pixel, enabling terrain-following / hybrid-sigma altitude grids.

    The quadratic is evaluated in one fully-vectorised batch over (N, M):

        |sat + t * d|² = r²
        A t² + 2 B_half t + C = 0

    where A = |d|² (N,), B_half = d·sat (N,), C = |sat|² − r² (N, M).

    Parameters
    ----------
    sat_ecef : (3,) array
    lats_deg, lons_deg : (N,) arrays — pixel ground-point coordinates
    level_alts_km : (N, M) array — altitude of each shell for each pixel

    Returns
    -------
    Same dict structure as ``compute_ray_paths_vectorized``:
    intercept_pts, slant_lengths, t_params, lat_intercepts, lon_intercepts,
    alt_intercepts.
    """
    from geosat_geometry import RE_KM, ecef_to_geodetic, geodetic_to_ecef

    N, M = level_alts_km.shape

    gnd_ecef = geodetic_to_ecef(lats_deg, lons_deg)             # (N, 3)
    ray_dirs = gnd_ecef - sat_ecef[np.newaxis, :]               # (N, 3)  sat→gnd

    A      = np.einsum('ij,ij->i', ray_dirs, ray_dirs)          # (N,)  |d|²
    B_half = np.einsum('ij,j->i',  ray_dirs, sat_ecef)          # (N,)  d·sat
    sat_sq = float(np.dot(sat_ecef, sat_ecef))                  # scalar

    r    = RE_KM + level_alts_km                                # (N, M)
    C    = sat_sq - r ** 2                                      # (N, M)
    disc = np.maximum(
        B_half[:, np.newaxis] ** 2 - A[:, np.newaxis] * C, 0.0
    )                                                           # (N, M)
    sq   = np.sqrt(disc)                                        # (N, M)

    # Near root = entry from above; clamp t to [0, 1]
    t1  = (-B_half[:, np.newaxis] - sq) / A[:, np.newaxis]     # (N, M)
    t_n = np.clip(t1, 0.0, 1.0)                                # (N, M)

    intercept_pts = (
        sat_ecef[np.newaxis, np.newaxis, :]
        + t_n[:, :, np.newaxis] * ray_dirs[:, np.newaxis, :]
    )                                                           # (N, M, 3)

    diffs         = np.diff(intercept_pts, axis=1)              # (N, M-1, 3)
    slant_lengths = np.linalg.norm(diffs, axis=2)               # (N, M-1)

    flat_lat, flat_lon, flat_alt = ecef_to_geodetic(
        intercept_pts.reshape(-1, 3)
    )
    return dict(
        intercept_pts  = intercept_pts,
        slant_lengths  = slant_lengths,
        t_params       = t_n,
        lat_intercepts = flat_lat.reshape(N, M),
        lon_intercepts = flat_lon.reshape(N, M),
        alt_intercepts = flat_alt.reshape(N, M),
    )


# ---------------------------------------------------------------------------
# Public helper — interpolate a model altitude field to block pixel centres
# ---------------------------------------------------------------------------

def interp_model_alts_to_block(
    alt_3d:     np.ndarray,   # (nlev, nlat, nlon) or (ntime, nlev, nlat, nlon)
    model_lats: np.ndarray,   # (nlat,) or (nlat, nlon)
    model_lons: np.ndarray,   # (nlon,) or (nlat, nlon)
    block:      object,       # ScanBlock
    valid_mask: Optional[np.ndarray] = None,  # (n_rows, n_cols) bool
    time_idx:   int = 0,      # which time step to use for 4-D alt_3d
) -> np.ndarray:
    """
    Bilinearly interpolate a 3-D model altitude field to each pixel centre of a
    ScanBlock, returning per-pixel altitude profiles ready for
    ``sample_field_along_rays``.

    This is the entry point for models with footprint-dependent pressure
    levels — hybrid-sigma, terrain-following, or any coordinate where the
    altitude of each model level varies horizontally.

    Typical usage
    -------------
    >>> # 'geopotential_height' from e.g. ERA5 or GEOS-Chem as (nlev, nlat, nlon)
    >>> level_alts_px = interp_model_alts_to_block(
    ...     gph_km, model_lats, model_lons, block
    ... )
    >>> result = sample_field_along_rays(
    ...     block, sat, co2_field, model_lats, model_lons,
    ...     level_alts_km=level_alts_px,    # (nlev, n_rows, n_cols)
    ... )

    Parameters
    ----------
    alt_3d : (nlev, nlat, nlon) **or** (ntime, nlev, nlat, nlon) array
        Altitude [km] of each model pressure level on the model's horizontal
        grid.  Level axis must be ascending in altitude (lowest first).
        If 4-D, ``alt_3d[time_idx]`` is used.
    model_lats : (nlat,) or (nlat, nlon) array
        Latitude grid [degrees].
        1-D: strictly monotone (ascending or descending).
        2-D: curvilinear grid (e.g. WRF); must match the horizontal shape of
        ``alt_3d``.  Requires ``scipy``.
    model_lons : (nlon,) or (nlat, nlon) array
        Longitude grid [degrees], 0–360 or −180–180.
    block : ScanBlock
        The block whose pixel centres are used as query points.
    valid_mask : (n_rows, n_cols) bool or None
        Pixels to process; defaults to block's own valid / pixel_counts mask.
    time_idx : int, optional
        Time-step index to extract when ``alt_3d`` is 4-D (default: 0).

    Returns
    -------
    level_alts_px : (nlev, n_rows, n_cols) float64
        Altitude [km] of each model level at each pixel centre.
        Invalid / masked pixels and pixels outside a limited-area domain are
        filled with NaN.
    """
    alt_3d     = np.asarray(alt_3d,     dtype=float)
    model_lats = np.asarray(model_lats, dtype=float)
    model_lons = np.asarray(model_lons, dtype=float)

    # ---- strip time dimension ---------------------------------------------
    if alt_3d.ndim == 4:
        alt_3d = alt_3d[time_idx]
    if alt_3d.ndim != 3:
        raise ValueError(f"alt_3d must be 3-D or 4-D; got shape {alt_3d.shape}")

    nlev, nlat, nlon = alt_3d.shape
    n_rows, n_cols   = block.shape

    # ---- detect grid type -------------------------------------------------
    curvilinear = model_lats.ndim == 2
    if curvilinear:
        if model_lats.shape != (nlat, nlon) or model_lons.shape != (nlat, nlon):
            raise ValueError(
                f"2-D model_lats/lons shape {model_lats.shape}/{model_lons.shape} "
                f"!= alt_3d horizontal shape ({nlat}, {nlon})"
            )
    else:
        if model_lats.ndim != 1 or model_lons.ndim != 1:
            raise ValueError("model_lats and model_lons must both be 1-D or both be 2-D")

    # ---- normalise grids (1-D only) ---------------------------------------
    if not curvilinear:
        lat_asc     = np.sort(model_lats)
        lat_flipped = model_lats[0] > model_lats[-1]
        if lat_flipped:
            alt_3d = alt_3d[:, ::-1, :]
        lon_norm = _normalise_lons(model_lons)
        sort_idx = np.argsort(lon_norm)
        lon_asc  = lon_norm[sort_idx]
        alt_3d   = alt_3d[:, :, sort_idx]

    # ---- valid pixel mask -------------------------------------------------
    if valid_mask is None:
        if 'valid_mask' in block:
            valid_mask = block['valid_mask']
        elif 'pixel_counts' in block:
            valid_mask = block['pixel_counts'] > 0
        else:
            valid_mask = np.isfinite(block['lats'])

    valid_flat = valid_mask.ravel()
    valid_idx  = np.where(valid_flat)[0]
    lats_v = block['lats'].ravel()[valid_idx]    # (N_valid,)
    lons_v = block['lons'].ravel()[valid_idx]

    m_idx = np.arange(nlev)    # (nlev,)
    N_v   = len(valid_idx)

    # ---- bilinear indices at pixel centres --------------------------------
    if not curvilinear:
        lons_v_norm = _normalise_lons(lons_v)
        lats_v2 = lats_v[:, np.newaxis]       # (N_valid, 1) → broadcast over nlev
        lons_v2 = lons_v_norm[:, np.newaxis]
        i_r, i_c, ds, dt = _build_bilinear_indices(
            lats_v2, lons_v2, lat_asc, lon_asc
        )                                      # all (N_valid, 1)
        outside_v = (
              (lats_v  < lat_asc[0])  | (lats_v  > lat_asc[-1])
            | (lons_v_norm < lon_asc[0])  | (lons_v_norm > lon_asc[-1])
        )                                      # (N_valid,)
    else:
        i_r, i_c, ds, dt, outside_v = _bilinear_indices_2d(
            lats_v, lons_v, model_lats, model_lons
        )                                      # all (N_valid,)
        i_r = i_r[:, np.newaxis]
        i_c = i_c[:, np.newaxis]
        ds  = ds[:, np.newaxis]
        dt  = dt[:, np.newaxis]

    # ---- sample all levels at once ----------------------------------------
    alts_v = _bilinear_sample_all_levels(      # (N_valid, nlev)
        alt_3d,
        np.broadcast_to(i_r, (N_v, nlev)),
        np.broadcast_to(i_c, (N_v, nlev)),
        np.broadcast_to(ds,  (N_v, nlev)),
        np.broadcast_to(dt,  (N_v, nlev)),
        m_idx,
    )
    alts_v[outside_v, :] = np.nan

    # ---- pack back to (nlev, n_rows, n_cols) ------------------------------
    out = np.full((nlev, n_rows * n_cols), np.nan)
    out[:, valid_idx] = alts_v.T
    return out.reshape(nlev, n_rows, n_cols)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def sample_field_along_rays(
    block:          object,          # ScanBlock
    sat:            object,          # LongSlitGeoSatellite
    field_3d:       np.ndarray,      # (nlev,nlat,nlon) or (ntime,nlev,nlat,nlon)
    model_lats:     np.ndarray,      # (nlat,) or (nlat,nlon)
    model_lons:     np.ndarray,      # (nlon,) or (nlat,nlon)
    level_alts_km:  np.ndarray,      # (nlev,) OR (nlev, n_rows, n_cols)
    fill_value:     float = np.nan,  # value for ray intercepts outside domain
    valid_mask:     Optional[np.ndarray] = None,  # (n_rows, n_cols) bool
    time_idx:       int = 0,         # which time step to use for 4-D fields
) -> dict:
    """
    Sample a 3-D atmospheric model field along satellite-to-ground ray paths.

    For each pixel (i, j) in ``block`` the ray is traced through the atmosphere
    and the model field is bilinearly interpolated at the (lat, lon) intercept
    on each pressure level.  Returns the sampled values and slant-path lengths
    needed for column integration.

    Parameters
    ----------
    block : ScanBlock
        Pre-computed observation geometry from
        ``LongSlitGeoSatellite.build_scan_block``.
    sat : LongSlitGeoSatellite
        The satellite object used to trace the rays.
    field_3d : (nlev, nlat, nlon) or (ntime, nlev, nlat, nlon) array
        Model field.  If 4-D, ``field_3d[time_idx]`` is used.  The level
        axis must be ordered ascending in altitude (decreasing pressure).
    model_lats : (nlat,) or (nlat, nlon) array
        Latitude grid [degrees].
        1-D: must be strictly monotone (ascending or descending).
        2-D: curvilinear grid (e.g. WRF); both ``model_lats`` and
        ``model_lons`` must have the same 2-D shape.
    model_lons : (nlon,) or (nlat, nlon) array
        Longitude grid [degrees].  0–360 and −180–180 both accepted.
    level_alts_km : (nlev,) **or** (nlev, n_rows, n_cols) array
        Altitude [km] of each pressure level.

        * **1-D ``(nlev,)``** — one altitude profile shared by all pixels
          (standard fixed pressure levels, e.g. from ``pressure_to_alt_std_atm``).
        * **3-D ``(nlev, n_rows, n_cols)``** — a per-pixel altitude profile,
          e.g. from terrain-following / hybrid-sigma coordinates.  Each pixel
          gets its own set of shell radii for ray tracing, so the slant-path
          intersections correctly account for spatially varying layer heights.
          Use ``interp_model_alts_to_block`` to generate this array from a 3-D
          model altitude field.

        In both cases the level axis must be **ascending in altitude** (surface
        level first).
    fill_value : float, optional
        Value assigned to pixels whose ray intercept falls outside the model
        grid domain (default: NaN).
    valid_mask : (n_rows, n_cols) bool array or None
        If provided, only pixels where the mask is True are ray-traced.
        Other pixels are filled with ``fill_value``.  If None, the block's
        own ``valid_mask`` (fine blocks) or ``pixel_counts > 0`` (coarsened
        blocks) is used.
    time_idx : int, optional
        Time-step index to extract when ``field_3d`` is 4-D (default: 0).

    Returns
    -------
    dict with keys:

    sampled : (nlev, n_rows, n_cols) float64
        Field values sampled at the ray intercept point on each pressure level.

    slant_lengths : (nlev - 1, n_rows, n_cols) float64
        Slant-path length [km] through each atmospheric layer.  Layer k spans
        level k to level k+1.  Use with ``sampled`` and ``integrate_along_rays``
        for column integrals.

    lat_intercepts : (nlev, n_rows, n_cols) float64
        Geodetic latitude [degrees] of the ray intercept on each level.

    lon_intercepts : (nlev, n_rows, n_cols) float64
        Longitude [degrees] of the ray intercept on each level.

    outside_domain : (nlev, n_rows, n_cols) bool
        True where the ray intercept fell outside the model grid.  For
        limited-area domains pixels that are entirely outside will have this
        True for every level and ``sampled`` NaN for every level.
        ``outside_domain.all(axis=0)`` gives a 2-D pixel-level map.

    Notes
    -----
    **Per-pixel altitude workflow** (hybrid-sigma / terrain-following coords)::

        # 1. Load model geopotential height field: gph_km (nlev, nlat, nlon)
        level_alts_px = interp_model_alts_to_block(
            gph_km, model_lats, model_lons, block
        )  # (nlev, n_rows, n_cols)

        # 2. Sample with per-pixel altitudes
        result = sample_field_along_rays(
            block, sat, co2_field, model_lats, model_lons,
            level_alts_km=level_alts_px,
        )

    **Memory**: for a 500×400 block with 50 levels the ``sampled`` array is
    ~80 MB.  Use a coarsened block (``block.coarsen()``) for lower-memory
    workflows.

    **Speed**: bilinear interpolation uses NumPy advanced indexing (no Python
    loops, no ``scipy`` overhead).  Per-pixel ray tracing adds a modest ~20%
    overhead vs. fixed levels due to the extra ``(N, M)`` broadcast.
    """
    field_3d      = np.asarray(field_3d,      dtype=float)
    model_lats    = np.asarray(model_lats,    dtype=float)
    model_lons    = np.asarray(model_lons,    dtype=float)
    level_alts_km = np.asarray(level_alts_km, dtype=float)

    # ---- strip time dimension ---------------------------------------------
    if field_3d.ndim == 4:
        field_3d = field_3d[time_idx]
    if field_3d.ndim != 3:
        raise ValueError(f"field_3d must be 3-D or 4-D; got shape {field_3d.shape}")

    nlev, nlat, nlon = field_3d.shape
    n_rows, n_cols   = block.shape

    # ---- detect grid type (1-D regular vs 2-D curvilinear) ----------------
    curvilinear = model_lats.ndim == 2
    if curvilinear:
        if model_lats.shape != (nlat, nlon) or model_lons.shape != (nlat, nlon):
            raise ValueError(
                f"2-D model_lats/lons shape {model_lats.shape}/{model_lons.shape} "
                f"!= field_3d horizontal shape ({nlat}, {nlon})"
            )
    else:
        if model_lats.ndim != 1 or model_lons.ndim != 1:
            raise ValueError("model_lats and model_lons must both be 1-D or both be 2-D")
        if len(model_lats) != nlat:
            raise ValueError(f"model_lats length {len(model_lats)} != field_3d nlat {nlat}")
        if len(model_lons) != nlon:
            raise ValueError(f"model_lons length {len(model_lons)} != field_3d nlon {nlon}")

    # ---- detect whether level_alts_km is shared (1D) or per-pixel (3D) ---
    if level_alts_km.ndim == 1:
        per_pixel_alts = False
        if len(level_alts_km) != nlev:
            raise ValueError(
                f"level_alts_km length {len(level_alts_km)} != field_3d nlev {nlev}"
            )
        if not np.all(np.diff(level_alts_km) > 0):
            raise ValueError("level_alts_km must be strictly ascending (lowest altitude first)")
    elif level_alts_km.ndim == 3:
        per_pixel_alts = True
        if level_alts_km.shape != (nlev, n_rows, n_cols):
            raise ValueError(
                f"Per-pixel level_alts_km shape {level_alts_km.shape} != "
                f"expected ({nlev}, {n_rows}, {n_cols})"
            )
    else:
        raise ValueError(
            f"level_alts_km must be 1-D (nlev,) or 3-D (nlev, n_rows, n_cols); "
            f"got shape {level_alts_km.shape}"
        )

    # ---- determine which pixels to process --------------------------------
    if valid_mask is None:
        if 'valid_mask' in block:
            valid_mask = block['valid_mask']
        elif 'pixel_counts' in block:
            valid_mask = block['pixel_counts'] > 0
        else:
            valid_mask = np.isfinite(block['lats'])
    valid_flat = valid_mask.ravel()
    lats_all   = block['lats'].ravel()
    lons_all   = block['lons'].ravel()

    valid_idx = np.where(valid_flat)[0]
    N_total   = n_rows * n_cols
    M         = nlev

    # ---- normalise grids (1-D only) ---------------------------------------
    if not curvilinear:
        lat_asc     = np.sort(model_lats)
        lat_flipped = model_lats[0] > model_lats[-1]
        if lat_flipped:
            field_3d = field_3d[:, ::-1, :]
        lon_norm = _normalise_lons(model_lons)
        sort_idx = np.argsort(lon_norm)
        lon_asc  = lon_norm[sort_idx]
        field_3d = field_3d[:, :, sort_idx]

    m_idx = np.arange(M)

    # ---- ray tracing on valid pixels only ---------------------------------
    lats_v = lats_all[valid_idx]
    lons_v = lons_all[valid_idx]

    if not per_pixel_alts:
        rays = sat.compute_ray_paths_vectorized(lats_v, lons_v, level_alts_km)
    else:
        alts_v = level_alts_km.reshape(M, N_total)[:, valid_idx].T   # (N_valid, M)
        rays   = _compute_ray_paths_per_pixel(sat.sat_ecef, lats_v, lons_v, alts_v)

    query_lats = rays['lat_intercepts']          # (N_valid, M)
    query_lons = rays['lon_intercepts']          # (N_valid, M) — raw

    # ---- bilinear interpolation (1-D vs 2-D dispatch) ---------------------
    if not curvilinear:
        q_lons_norm = _normalise_lons(query_lons)
        i_r, i_c, ds, dt = _build_bilinear_indices(
            query_lats, q_lons_norm, lat_asc, lon_asc
        )
        outside_valid = (
              (query_lats  < lat_asc[0])   | (query_lats  > lat_asc[-1])
            | (q_lons_norm < lon_asc[0])   | (q_lons_norm > lon_asc[-1])
        )
    else:
        i_r, i_c, ds, dt, outside_valid = _bilinear_indices_2d(
            query_lats, query_lons, model_lats, model_lons
        )

    sampled_valid = _bilinear_sample_all_levels(
        field_3d, i_r, i_c, ds, dt, m_idx
    )                                            # (N_valid, M)
    sampled_valid[outside_valid] = fill_value

    # ---- pack results back to (nlev, n_rows, n_cols) grids ----------------
    sampled_flat  = np.full((M,     N_total), fill_value)
    slant_flat    = np.full((M - 1, N_total), np.nan)
    lat_int_flat  = np.full((M,     N_total), np.nan)
    lon_int_flat  = np.full((M,     N_total), np.nan)
    outside_flat  = np.ones( (M,     N_total), dtype=bool)   # default: outside

    sampled_flat[:, valid_idx]  = sampled_valid.T
    slant_flat[:, valid_idx]    = rays['slant_lengths'].T
    lat_int_flat[:, valid_idx]  = rays['lat_intercepts'].T
    lon_int_flat[:, valid_idx]  = rays['lon_intercepts'].T
    outside_flat[:, valid_idx]  = outside_valid.T

    return dict(
        sampled        = sampled_flat.reshape(M,     n_rows, n_cols),
        slant_lengths  = slant_flat.reshape(M - 1,  n_rows, n_cols),
        lat_intercepts = lat_int_flat.reshape(M,     n_rows, n_cols),
        lon_intercepts = lon_int_flat.reshape(M,     n_rows, n_cols),
        outside_domain = outside_flat.reshape(M,     n_rows, n_cols),
    )


def integrate_along_rays(
    sampled:       np.ndarray,    # (nlev, n_rows, n_cols)
    slant_lengths: np.ndarray,    # (nlev-1, n_rows, n_cols)
    layer_weights: Optional[np.ndarray] = None,  # (nlev-1,) or (nlev-1, n_rows, n_cols)
    normalize:     bool = False,
) -> np.ndarray:
    """
    Compute slant-column integrals from sampled model values and path lengths.

    Uses the trapezoidal rule over the layer boundaries:

        column[i, j] = Σ_k  0.5 * (f[k,i,j] + f[k+1,i,j]) * slant[k,i,j]

    Parameters
    ----------
    sampled : (nlev, n_rows, n_cols) array
        Field values at ray intercepts from ``sample_field_along_rays``.
    slant_lengths : (nlev - 1, n_rows, n_cols) array
        Slant-path lengths [km] through each layer from ``sample_field_along_rays``.
    layer_weights : (nlev-1,) or (nlev-1, n_rows, n_cols) or None
        Optional per-layer multiplicative weights (e.g. pressure layer thickness,
        number density).  Applied to both numerator and denominator so they
        cancel in the average when ``normalize=True``.
    normalize : bool, optional
        If False (default), return the slant-column integral
        ``Σ f_avg * slant * w``  with units [field_units × km].

        If True, return the path-length-weighted column-average mole fraction:

            X[i,j] = Σ_k f_avg[k] * slant[k] * w[k]
                      ─────────────────────────────────
                         Σ_k slant[k] * w[k]

        with units [field_units].  This is analogous to XCO₂: the average
        mole fraction seen along the slant path, weighted by layer thickness
        (or by ``layer_weights`` if provided).

    Returns
    -------
    result : (n_rows, n_cols) float64
        Slant-column integral (``normalize=False``) or column-average mole
        fraction (``normalize=True``).
    """
    sampled       = np.asarray(sampled,       dtype=float)
    slant_lengths = np.asarray(slant_lengths, dtype=float)

    # Trapezoidal average of adjacent level values
    f_avg = 0.5 * (sampled[:-1] + sampled[1:])   # (nlev-1, n_rows, n_cols)

    integrand = f_avg * slant_lengths
    if layer_weights is not None:
        integrand *= np.asarray(layer_weights, dtype=float)

    numerator = np.nansum(integrand, axis=0)      # (n_rows, n_cols)

    if not normalize:
        return numerator

    # Denominator: total weighted path length (same weights, no field values)
    denom = slant_lengths.copy()
    if layer_weights is not None:
        denom = denom * np.asarray(layer_weights, dtype=float)
    denominator = np.nansum(denom, axis=0)        # (n_rows, n_cols)

    return np.where(denominator > 0, numerator / denominator, np.nan)
