"""
scan_sampler.py
===============
High-level wrapper: reads atmospheric model output from a NetCDF file,
samples tracer fields along GeoCarb scan-block ray paths, and writes
pressure-weighted column-average mole fractions together with full
observation geometry and land/water fraction to a CF-compliant NetCDF4
output file.

Typical workflow
----------------
1. Build (or load) a ScanBlock from geosat_geometry.
2. Call ``run_scan_sampler``, pointing at a model NetCDF file.
3. Read or plot the output NetCDF.

Quick start
-----------
>>> from datetime import datetime, timezone
>>> from geosat_geometry import LongSlitGeoSatellite
>>> from scan_sampler import NCConfig, run_scan_sampler
>>>
>>> sat   = LongSlitGeoSatellite(sat_lon_deg=-95.0)
>>> block = sat.build_scan_block(30.0, -120.0, -95.0,
...                              t0_utc=datetime(2020, 7, 1, 18, tzinfo=timezone.utc))
>>>
>>> # Defaults work for GEOS-Chem / ERA5 with variables named lat/lon/lev/time
>>> run_scan_sampler(
...     nc_path   = 'model_output.nc',
...     block     = block,
...     sat       = sat,
...     tracers   = ['CO2', 'CH4'],
...     out_path  = 'xgas_west_scan.nc',
...     t_utc     = datetime(2020, 7, 1, 18, tzinfo=timezone.utc),
...     dlat_deg  = 0.5,   # coarsen to 0.5° before sampling
...     dlon_deg  = 0.5,
... )

NCConfig — adapting to your model's variable names
---------------------------------------------------
>>> # WRF output (Pa pressure, 2-D lat/lon, geopotential height available)
>>> cfg = NCConfig(lat_var='XLAT', lon_var='XLONG',
...                lev_var='P_HYD', lev_in_pa=True,
...                gph_var='Z', gph_in_m=True)
>>>
>>> # ERA5 (hPa levels, latitude/longitude variable names)
>>> cfg = NCConfig(lat_var='latitude', lon_var='longitude', lev_var='level')

Output file variables
---------------------
All geometry and column-average arrays have shape ``(row, col)``.

Geometry:
  lat, lon, vza, vaa, sza, saa, airmass_geometric, relative_azimuth,
  land_fraction, valid, pixel_counts (coarsened blocks), outside_domain

Sampling:
  pressure_levels  — (nlev,) model pressure levels [hPa]
  x_{tracer}       — pressure-weighted column-average mole fraction [same units as model]
  profile_{tracer} — (nlev, row, col) sampled values at each level [save_profiles=True only]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from geosat_geometry import LongSlitGeoSatellite, ScanBlock, add_land_fraction
from model_sampler import (
    interp_model_alts_to_block,
    pressure_to_alt_std_atm,
    sample_field_along_rays,
)

__all__ = ["NCConfig", "run_scan_sampler"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NCConfig:
    """
    Maps variable roles to names in the model NetCDF file.

    Defaults follow common conventions used by GEOS-Chem, ERA5, and many
    CTM outputs.  Override any field to match your model's naming scheme.

    Parameters
    ----------
    lat_var : str
        Latitude coordinate variable name.  May be 1-D ``(nlat,)`` for
        regular grids or 2-D ``(nlat, nlon)`` for curvilinear grids (e.g. WRF).
    lon_var : str
        Longitude coordinate variable name, same shape rules as ``lat_var``.
    lev_var : str
        Pressure-level coordinate variable name.  Must be 1-D ``(nlev,)``.
        Values are interpreted as hPa unless ``lev_in_pa=True``.
    time_var : str or None
        Time coordinate variable name (CF-convention, decoded via
        ``netCDF4.num2date``).  Set to ``None`` if the file has no time
        dimension; the first (only) index is used.
    lev_in_pa : bool
        Set ``True`` if the pressure coordinate is in Pa instead of hPa
        (values will be divided by 100).
    gph_var : str or None
        Optional geopotential height variable name, shape
        ``(nlev, nlat, nlon)`` or ``(ntime, nlev, nlat, nlon)``.  When
        provided, per-pixel terrain-following altitudes are used for ray
        tracing instead of the standard-atmosphere conversion.  Requires
        that the level axis matches ``lev_var``.
    gph_in_m : bool
        Set ``True`` if ``gph_var`` is in metres (divided by 1000 → km).
        Set ``False`` if already in km.
    """
    lat_var:  str            = "lat"
    lon_var:  str            = "lon"
    lev_var:  str            = "lev"
    time_var: Optional[str]  = "time"
    lev_in_pa: bool          = False
    gph_var:  Optional[str]  = None
    gph_in_m: bool           = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_time_idx(ds, t_utc: Optional[datetime], time_var: Optional[str]) -> int:
    """Return the time-dimension index nearest to *t_utc*, or 0 if unavailable."""
    import netCDF4

    if time_var is None or time_var not in ds.variables:
        return 0
    if t_utc is None:
        return 0

    tvar = ds.variables[time_var]
    units    = getattr(tvar, "units",    None)
    calendar = getattr(tvar, "calendar", "standard")
    if units is None:
        return 0

    try:
        cf_times = netCDF4.num2date(tvar[:], units=units, calendar=calendar)
    except Exception:
        return 0

    # Normalise t_utc to UTC-aware
    if t_utc.tzinfo is None:
        t_utc = t_utc.replace(tzinfo=timezone.utc)

    def _to_seconds(ct) -> float:
        try:
            dt = datetime(ct.year, ct.month, ct.day,
                          ct.hour, ct.minute, ct.second,
                          tzinfo=timezone.utc)
            return abs((dt - t_utc).total_seconds())
        except Exception:
            return float("inf")

    diffs = np.array([_to_seconds(ct) for ct in cf_times])
    idx   = int(np.argmin(diffs))

    gap_h = diffs[idx] / 3600.0
    if gap_h > 6.0:
        warnings.warn(
            f"Nearest model time is {gap_h:.1f} h from requested t_utc={t_utc}. "
            "Using it anyway.",
            stacklevel=4,
        )
    return idx


def _read_nc(
    nc_path: Union[str, Path],
    tracers: Sequence[str],
    t_utc:   Optional[datetime],
    cfg:     NCConfig,
) -> dict:
    """
    Open a model NetCDF file and return grids, pressure levels, and tracer fields.

    Returns a dict with keys:
      lat_grid      — (nlat,) or (nlat, nlon)
      lon_grid      — (nlon,) or (nlat, nlon)
      pres_hpa      — (nlev,) descending (surface = index 0, TOA = index -1)
      level_alts_km — (nlev,) standard-atmosphere altitudes for pres_hpa
      gph_km        — (nlev, nlat, nlon) or None
      fields        — {tracer: (nlev, nlat, nlon) float64}
      time_idx      — int
    """
    import netCDF4

    with netCDF4.Dataset(str(nc_path), "r") as ds:
        avail = list(ds.variables.keys())

        # ---- time index -------------------------------------------------------
        time_idx = _find_time_idx(ds, t_utc, cfg.time_var)

        # ---- lat / lon --------------------------------------------------------
        for vname, role in [(cfg.lat_var, "latitude"), (cfg.lon_var, "longitude")]:
            if vname not in ds.variables:
                raise KeyError(
                    f"{role} variable '{vname}' not found in {Path(nc_path).name}. "
                    f"Available variables: {avail}"
                )
        lat_grid = np.asarray(ds.variables[cfg.lat_var][:], dtype=float)
        lon_grid = np.asarray(ds.variables[cfg.lon_var][:], dtype=float)

        # ---- pressure levels --------------------------------------------------
        if cfg.lev_var not in ds.variables:
            raise KeyError(
                f"Pressure variable '{cfg.lev_var}' not found in {Path(nc_path).name}."
            )
        pres = np.asarray(ds.variables[cfg.lev_var][:], dtype=float).ravel()
        if cfg.lev_in_pa:
            pres = pres / 100.0   # Pa → hPa

        # Ensure descending pressure (index 0 = surface, highest pressure)
        flip_lev = pres[0] < pres[-1]
        if flip_lev:
            pres = pres[::-1]

        # ---- geopotential height (optional) -----------------------------------
        gph_km = None
        if cfg.gph_var is not None:
            if cfg.gph_var not in ds.variables:
                warnings.warn(
                    f"gph_var='{cfg.gph_var}' not found in {Path(nc_path).name}; "
                    "falling back to standard-atmosphere altitude conversion.",
                    stacklevel=4,
                )
            else:
                gph_raw = np.asarray(ds.variables[cfg.gph_var][:], dtype=float)
                if gph_raw.ndim == 4:
                    gph_raw = gph_raw[time_idx]
                if gph_raw.ndim != 3:
                    warnings.warn(
                        f"gph_var '{cfg.gph_var}' has unexpected shape {gph_raw.shape}; "
                        "ignoring and using standard-atmosphere altitudes.",
                        stacklevel=4,
                    )
                else:
                    if cfg.gph_in_m:
                        gph_raw = gph_raw / 1000.0
                    if flip_lev:
                        gph_raw = gph_raw[::-1, ...]
                    gph_km = gph_raw

        level_alts_km = pressure_to_alt_std_atm(pres)

        # ---- tracer fields ----------------------------------------------------
        fields: Dict[str, np.ndarray] = {}
        for tracer in tracers:
            if tracer not in ds.variables:
                raise KeyError(
                    f"Tracer variable '{tracer}' not found in {Path(nc_path).name}. "
                    f"Available variables: {avail}"
                )
            arr = np.asarray(ds.variables[tracer][:], dtype=float)
            if arr.ndim == 4:
                arr = arr[time_idx]
            if arr.ndim != 3:
                raise ValueError(
                    f"Tracer '{tracer}' has unexpected shape {arr.shape} after "
                    "time-index selection; expected (nlev, nlat, nlon)."
                )
            if flip_lev:
                arr = arr[::-1, ...]
            fields[tracer] = arr

        return dict(
            lat_grid      = lat_grid,
            lon_grid      = lon_grid,
            pres_hpa      = pres,
            level_alts_km = level_alts_km,
            gph_km        = gph_km,
            fields        = fields,
            time_idx      = time_idx,
        )


def _pressure_weighted_col_avg(
    sampled:  np.ndarray,   # (nlev, n_rows, n_cols)
    pres_hpa: np.ndarray,   # (nlev,) descending (surface first)
) -> np.ndarray:            # (n_rows, n_cols)
    """
    Pressure-weighted column-average mole fraction:

        X[i,j] = Σ_k  q_avg[k,i,j] * Δp[k]
                 ─────────────────────────────────
                 Σ_k  Δp[k]  [finite layers only]

    where q_avg[k] = 0.5*(q[k] + q[k+1]) and Δp[k] = p[k] - p[k+1] > 0.
    NaN layers (out-of-domain intercepts) are excluded from both numerator
    and denominator, so partially sampled columns still get a valid estimate.
    """
    f_avg = 0.5 * (sampled[:-1] + sampled[1:])        # (nlev-1, n_rows, n_cols)
    dp    = pres_hpa[:-1] - pres_hpa[1:]               # (nlev-1,) all positive
    dp_3d = dp[:, np.newaxis, np.newaxis]

    numerator   = np.nansum(f_avg * dp_3d, axis=0)
    denominator = np.nansum(
        np.where(np.isfinite(f_avg), dp_3d, 0.0), axis=0
    )
    return np.where(denominator > 0, numerator / denominator, np.nan)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_scan_sampler(
    nc_path:       Union[str, Path],
    block:         ScanBlock,
    sat:           LongSlitGeoSatellite,
    tracers:       Sequence[str],
    out_path:      Union[str, Path],
    t_utc:         Optional[datetime] = None,
    dlat_deg:      Optional[float]    = None,
    dlon_deg:      Optional[float]    = None,
    nc_config:     Optional[NCConfig] = None,
    save_profiles: bool               = False,
    land_mask_resolution_deg: float   = 0.1,
    natural_earth_scale:      str     = "110m",
) -> str:
    """
    Sample atmospheric model tracers along scan-block ray paths and write
    pressure-weighted column-average mole fractions to a NetCDF4 file.

    Parameters
    ----------
    nc_path : str or Path
        Atmospheric model NetCDF file.  Must contain a pressure-level
        coordinate and 3-D (or 4-D with time) tracer fields.
    block : ScanBlock
        Pre-computed scan geometry from ``LongSlitGeoSatellite.build_scan_block``.
    sat : LongSlitGeoSatellite
        Satellite object used to trace rays.
    tracers : sequence of str
        Names of tracer variables to process (e.g. ``['CO2', 'CH4', 'CO']``).
        Must match variable names in the netCDF file (or as remapped by
        ``nc_config``).
    out_path : str or Path
        Output NetCDF4 file path (created or overwritten).
    t_utc : datetime, optional
        UTC time used to select the nearest model time step.  Timezone-naive
        datetimes are assumed UTC.  If ``None`` or the file has no time
        dimension, the first time step is used.
    dlat_deg : float, optional
        Coarsen the block to this latitude resolution [degrees] before
        sampling.  Must be supplied together with ``dlon_deg``.  Recommended
        when the fine (6 km) block is much finer than the model grid.
    dlon_deg : float, optional
        Coarsen the block to this longitude resolution [degrees].
    nc_config : NCConfig, optional
        Maps variable roles to names in the model NetCDF.  Defaults to
        ``NCConfig()`` (common: lat/lon/lev/time, levels in hPa).
    save_profiles : bool, optional
        If True, also write per-level sampled values as ``profile_{tracer}``
        in the output file with shape ``(nlev, row, col)``.  Increases
        output file size by ``nlev ×`` for each tracer.  Default False.
    land_mask_resolution_deg : float, optional
        Resolution of the rasterised land/water mask [°].  Default 0.1°
        (~11 km), appropriate for 6 km GeoCarb pixels.  Passed to
        ``add_land_fraction``.
    natural_earth_scale : str, optional
        Natural Earth coastline dataset scale: ``'110m'`` (default, fast),
        ``'50m'``, or ``'10m'``.  Higher resolution improves coastline
        accuracy at the cost of a longer first-call build.

    Returns
    -------
    str
        Absolute path to the written output file.

    Output variables (shape ``(row, col)`` unless noted)
    -----------------------------------------------------
    Geometry:
        lat, lon                — pixel-centre coordinates [°]
        vza, vaa                — viewing zenith / azimuth angle [°]
        sza, saa                — solar zenith / azimuth angle [°]
        airmass_geometric       — sec(VZA) + sec(SZA)
        relative_azimuth        — |VAA − SAA| [°]
        land_fraction           — 0.0 (water) → 1.0 (land)
        valid                   — 1 where pixel is valid, else 0
        pixel_counts            — fine pixels per cell (coarsened blocks only)
        outside_domain          — 1 where all ray intercepts are outside model grid

    Sampling:
        pressure_levels         — ``(nlev,)`` model pressure levels [hPa]
        x_{tracer}              — pressure-weighted column-average mole fraction
        profile_{tracer}        — ``(nlev, row, col)`` sampled values per level
                                  (only written when ``save_profiles=True``)

    Notes
    -----
    The column average is defined as:

        X[i,j] = Σ_k  q̄[k,i,j] · Δp[k]  /  Σ_k  Δp[k]

    where q̄[k] = ½(q[k] + q[k+1]) is the layer-mean mole fraction and
    Δp[k] = p[k] − p[k+1] > 0 is the pressure thickness.  This is the
    standard approximation for column-average dry-air mole fractions
    (analogous to XCO₂) used in most CTM forward simulations.

    If ``gph_var`` is set in ``nc_config``, per-pixel terrain-following
    altitudes are interpolated to the block's pixel centres via
    ``interp_model_alts_to_block`` before ray tracing, providing correct
    layer boundaries for models with hybrid-sigma vertical coordinates.
    """
    import netCDF4

    if nc_config is None:
        nc_config = NCConfig()

    if (dlat_deg is None) != (dlon_deg is None):
        raise ValueError("dlat_deg and dlon_deg must both be provided or both omitted.")

    # ---- 1. Coarsen if requested -------------------------------------------
    if dlat_deg is not None:
        work_block = block.coarsen(dlat_deg, dlon_deg)
    else:
        work_block = block

    # ---- 2. Land / water fraction ------------------------------------------
    # add_land_fraction requires cartopy; skip gracefully if unavailable.
    try:
        work_block = add_land_fraction(
            work_block,
            mask_resolution_deg = land_mask_resolution_deg,
            natural_earth_scale = natural_earth_scale,
        )
    except ImportError:
        warnings.warn(
            "cartopy is not installed — land_fraction will not be written to output. "
            "Install with: conda install -c conda-forge cartopy",
            stacklevel=2,
        )

    # ---- 3. Valid-pixel mask ------------------------------------------------
    if "valid_mask" in work_block:
        valid = work_block["valid_mask"]
    elif "pixel_counts" in work_block:
        valid = work_block["pixel_counts"] > 0
    else:
        valid = np.isfinite(work_block["lats"])

    n_rows, n_cols = work_block.shape

    # ---- 4. Read model data ------------------------------------------------
    nc_data       = _read_nc(nc_path, tracers, t_utc, nc_config)
    lat_grid      = nc_data["lat_grid"]
    lon_grid      = nc_data["lon_grid"]
    pres_hpa      = nc_data["pres_hpa"]
    level_alts_km = nc_data["level_alts_km"]
    gph_km        = nc_data["gph_km"]
    fields        = nc_data["fields"]
    nlev          = len(pres_hpa)

    # ---- 5. Per-pixel altitudes from geopotential height (optional) --------
    if gph_km is not None:
        level_alts_arg = interp_model_alts_to_block(
            gph_km, lat_grid, lon_grid, work_block, valid_mask=valid
        )   # (nlev, n_rows, n_cols)
    else:
        level_alts_arg = level_alts_km   # (nlev,)

    # ---- 6. Sample each tracer along ray paths -----------------------------
    sampled_all: Dict[str, np.ndarray] = {}
    slant_lengths  = None
    outside_domain = None

    for tracer in tracers:
        result = sample_field_along_rays(
            work_block, sat,
            fields[tracer],
            lat_grid, lon_grid,
            level_alts_km = level_alts_arg,
            fill_value    = np.nan,
            valid_mask    = valid,
        )
        sampled_all[tracer] = result["sampled"]   # (nlev, n_rows, n_cols)

        # Ray geometry is the same for every tracer — cache on first pass
        if slant_lengths is None:
            slant_lengths  = result["slant_lengths"]    # (nlev-1, n_rows, n_cols)
            outside_domain = result["outside_domain"]   # (nlev,   n_rows, n_cols)

    # ---- 7. Pressure-weighted column averages ------------------------------
    col_avgs: Dict[str, np.ndarray] = {}
    for tracer in tracers:
        xgas = _pressure_weighted_col_avg(sampled_all[tracer], pres_hpa)
        col_avgs[tracer] = np.where(valid, xgas, np.nan)

    # ---- 8. Write output NetCDF4 ------------------------------------------
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(str(out_path), "w", format="NETCDF4") as ds:

        # Global attributes
        ds.title             = "GeoCarb scan sampler output"
        ds.Conventions       = "CF-1.8"
        ds.satellite_lon_deg = float(work_block._meta.get("sat_lon_deg", np.nan))
        ds.source_model_file = str(Path(nc_path).name)
        ds.tracers           = ", ".join(tracers)
        ds.weighting_method  = "pressure-weighted column average (Δp weights)"
        ds.t_scan_start_utc  = (str(work_block.t_start_utc)
                                if work_block.t_start_utc else "unknown")
        ds.t_scan_end_utc    = (str(work_block.t_end_utc)
                                if work_block.t_end_utc   else "unknown")
        if t_utc is not None:
            ds.t_model_utc   = str(t_utc)
        if dlat_deg is not None:
            ds.coarsen_dlat_deg = float(dlat_deg)
            ds.coarsen_dlon_deg = float(dlon_deg)

        # Dimensions
        ds.createDimension("row",  n_rows)
        ds.createDimension("col",  n_cols)
        ds.createDimension("nlev", nlev)

        # Helper: create a compressed float variable
        def _fvar(name, dims, units=None, long_name=None):
            v = ds.createVariable(name, "f4", dims,
                                  zlib=True, complevel=4,
                                  fill_value=np.float32(9.96921e+36))
            if units:     v.units     = units
            if long_name: v.long_name = long_name
            return v

        def _ivar(name, dims, long_name=None):
            v = ds.createVariable(name, "i4", dims,
                                  zlib=True, complevel=4,
                                  fill_value=np.int32(-9999))
            if long_name: v.long_name = long_name
            return v

        def _bvar(name, dims, long_name=None):
            v = ds.createVariable(name, "i1", dims, zlib=True, complevel=4)
            if long_name: v.long_name = long_name
            return v

        def _write_f(v, arr):
            """Write float array, masking NaN/inf as fill_value."""
            v[:] = np.ma.masked_invalid(np.asarray(arr, dtype=np.float32))

        # ---- geometry --------------------------------------------------
        _write_f(_fvar("lat", ("row", "col"),
                        units="degrees_north",
                        long_name="pixel-centre geodetic latitude"),
                 work_block["lats"])

        _write_f(_fvar("lon", ("row", "col"),
                        units="degrees_east",
                        long_name="pixel-centre longitude"),
                 work_block["lons"])

        _write_f(_fvar("vza", ("row", "col"),
                        units="degrees",
                        long_name="viewing zenith angle"),
                 work_block["vzas"])

        _write_f(_fvar("vaa", ("row", "col"),
                        units="degrees",
                        long_name="viewing azimuth angle (clockwise from North)"),
                 work_block["vaas"])

        _write_f(_fvar("sza", ("row", "col"),
                        units="degrees",
                        long_name="solar zenith angle"),
                 work_block["szas"])

        _write_f(_fvar("saa", ("row", "col"),
                        units="degrees",
                        long_name="solar azimuth angle (clockwise from North)"),
                 work_block["saas"])

        _write_f(_fvar("airmass_geometric", ("row", "col"),
                        units="1",
                        long_name="geometric airmass sec(VZA) + sec(SZA)"),
                 work_block["airmass_geometric"])

        _write_f(_fvar("relative_azimuth", ("row", "col"),
                        units="degrees",
                        long_name="relative azimuth angle |VAA - SAA|"),
                 work_block["relative_azimuth"])

        # ---- land / water fraction -------------------------------------
        if "land_fraction" in work_block:
            _write_f(_fvar("land_fraction", ("row", "col"),
                            units="1",
                            long_name="land fraction (0=water, 1=land)"),
                     work_block["land_fraction"])

        # ---- valid mask and pixel counts --------------------------------
        vv = _bvar("valid", ("row", "col"),
                   long_name="valid pixel flag (1=valid, 0=invalid or masked)")
        vv[:] = valid.astype(np.int8)

        if "pixel_counts" in work_block:
            cv = _ivar("pixel_counts", ("row", "col"),
                       long_name="number of fine-resolution pixels in this cell")
            cv[:] = work_block["pixel_counts"].astype(np.int32)

        # ---- outside-domain mask ----------------------------------------
        if outside_domain is not None:
            ood_pixel = outside_domain.all(axis=0)   # True if every level outside
            ov = _bvar("outside_domain", ("row", "col"),
                       long_name="1 where all ray intercepts are outside model grid")
            ov[:] = np.where(valid, ood_pixel.astype(np.int8), np.int8(0))

        # ---- pressure levels -------------------------------------------
        pv = _fvar("pressure_levels", ("nlev",),
                   units="hPa",
                   long_name="model pressure levels (index 0 = surface, index -1 = TOA)")
        pv[:] = pres_hpa.astype(np.float32)

        # ---- column averages -------------------------------------------
        for tracer in tracers:
            _write_f(
                _fvar(f"x_{tracer}", ("row", "col"),
                      long_name=f"pressure-weighted column-average {tracer} mole fraction"),
                col_avgs[tracer],
            )

        # ---- optional per-level profiles --------------------------------
        if save_profiles:
            for tracer in tracers:
                _write_f(
                    _fvar(f"profile_{tracer}", ("nlev", "row", "col"),
                          long_name=f"sampled {tracer} at ray intercepts per pressure level"),
                    sampled_all[tracer],
                )

    return str(out_path)
