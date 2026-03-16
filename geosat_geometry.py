"""
geosat_geometry.py
==================
Observation geometry simulator for a long-slit geostationary satellite.

Computes per-pixel:
  - Geodetic center (lat, lon)
  - Footprint polygon corners (4 corners, CCW order)
  - Viewing geometry (VZA, VAA)
  - Solar geometry (SZA, SAA)
  - Satellite-to-ground look vector in ECEF
  - 3D ray-shell intercept points for atmospheric ray tracing
  - Land/water fraction (via add_land_fraction)

Designed for instruments like GeoCarb:
  - GEO orbit, fixed longitude
  - Long slit (~3000 km N-S)
  - Pixel size ~6 km × 6 km at nadir
  - E-W scanning via mirror stepping (one pixel width per integration step)
  - 10-second integration time

Coordinate systems
------------------
ECEF  : Earth-Centered Earth-Fixed, km, WGS-84 ellipsoid.
        +X toward (0°N, 0°E), +Z toward North Pole.
ENU   : East-North-Up local frame at ground point.
Angles: degrees throughout; azimuths measured clockwise from North.

ScanBlock
---------
build_scan_block() returns a ScanBlock — a dict-like object of 2D arrays:
  axis 0 = N-S pixels (slit rows, 0 = southernmost)
  axis 1 = E-W steps  (scan columns, 0 = westernmost)

Contains lats, lons, vzas, vaas, szas, saas, sat_look_vecs, gnd_ecef,
corner_lats/lons, airmass_geometric, relative_azimuth.
Includes timing metadata and can be saved/loaded with .save()/.load().

Land/water fraction
-------------------
add_land_fraction(block) returns a new ScanBlock with a ``land_fraction``
field (0.0 = all water, 1.0 = all land) added.  Requires cartopy.
The land mask is rasterised from Natural Earth polygons and cached so that
repeated calls over the same region are fast.

Day scheduling
--------------
build_day_schedule(satellite, t_start, targets) chains ScanBlocks in time
for a full day of observations.  Define scan regions with ScanTarget, pass
an ordered list, and the utility cycles through them (repeat=True by default)
from t_start to t_end (default: +24 h), advancing the clock after each scan.

Plotting
--------
plot_scan_blocks() supports a ``projection`` keyword:
  'platecarree'   — standard lat/lon (default)
  'geostationary' — satellite-view disk centred on the sub-satellite point
                    (requires cartopy; satellite longitude taken from block
                    metadata automatically)

For 3D ray tracing
------------------
Each observation provides:
  sat_ecef      : (3,) satellite position [km]
  gnd_ecef      : (3,) ground intercept [km]
  sat_look_vecs : (3,) unit vector satellite → ground

compute_ray_paths_vectorized() returns ECEF intercept points on spherical
shells at requested altitudes — feed directly into a 3D atmospheric model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# WGS-84 constants
# ---------------------------------------------------------------------------
RE_KM      = 6378.137           # equatorial radius [km]
RP_KM      = 6356.752           # polar radius [km]
F_WGS84    = 1.0 / 298.257223563
E2_WGS84   = 2 * F_WGS84 - F_WGS84**2   # first eccentricity squared
SAT_ALT_KM = 35786.0            # nominal GEO altitude above equator [km]

__all__ = [
    "LongSlitGeoSatellite",
    "ObservationGeometry",
    "RayPath",
    "ScanBlock",
    "ScanTarget",
    "plot_scan_blocks",
    "coarsen_scan_block",
    "add_land_fraction",
    "build_day_schedule",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "solar_position_ecef",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObservationGeometry:
    """Full geometry for one pixel during one integration."""
    # Center
    lat_deg:      float        # geodetic latitude  [deg]
    lon_deg:      float        # longitude          [deg]
    # Footprint corners — NW, NE, SE, SW order (CCW from above)
    corner_lats:  np.ndarray   # shape (4,)  [deg]
    corner_lons:  np.ndarray   # shape (4,)  [deg]
    # Viewing angles at ground point
    vza_deg:      float        # viewing zenith angle  [deg]
    vaa_deg:      float        # viewing azimuth angle [deg, N=0 E=90]
    # Solar angles at ground point
    sza_deg:      float        # solar zenith angle  [deg]
    saa_deg:      float        # solar azimuth angle [deg]
    # 3D positions (ECEF, km)
    sat_ecef:     np.ndarray   # shape (3,) satellite position
    gnd_ecef:     np.ndarray   # shape (3,) ground point
    look_vec:     np.ndarray   # shape (3,) unit vector sat→gnd
    # Scan indices
    scan_col:     int          # E-W scan step index
    slit_row:     int          # N-S pixel index (0 = southernmost)
    # Optional timestamp
    utc_time:     Optional[datetime] = None


@dataclass
class RayPath:
    """
    Satellite-to-ground ray intersections with atmospheric shells.

    altitudes_km : (M,)   shell altitudes [km], ascending
    intercept_pts: (M, 3) ECEF coordinates [km] of entry into each shell
    slant_lengths: (M-1,) path length within each layer [km]
    t_params     : (M,)   parametric t along ray (0=satellite, 1=ground)
    lat_pts      : (M,)   geodetic latitude at each intercept [deg]
    lon_pts      : (M,)   longitude at each intercept [deg]
    """
    altitudes_km:  np.ndarray
    intercept_pts: np.ndarray
    slant_lengths: np.ndarray
    t_params:      np.ndarray
    lat_pts:       np.ndarray
    lon_pts:       np.ndarray


@dataclass
class ScanTarget:
    """
    Definition of one scan region, for use with build_day_schedule().

    Parameters
    ----------
    slit_center_lat : float
        Nominal latitude of the slit centre [°].
    scan_start_lon : float
        West edge of the scan [° E].
    scan_end_lon : float
        East edge of the scan [° E].
    label : str or None
        Optional name stored in the resulting ScanBlock's metadata under
        the key ``'label'``.
    n_cols : int or None
        Number of E-W scan steps.  If None (default) the count is derived
        from the ground distance divided by pixel_size_ew_km, matching the
        default behaviour of build_scan_block().
    """
    slit_center_lat: float
    scan_start_lon:  float
    scan_end_lon:    float
    label:           Optional[str] = None
    n_cols:          Optional[int] = None


# ---------------------------------------------------------------------------
# ScanBlock — 2D pre-computed geometry for a rectangular scan region
# ---------------------------------------------------------------------------

class ScanBlock:
    """
    Pre-computed observation geometry for a rectangular scan region.

    Behaves like a read-only dict for array data; metadata is available as
    attributes.  Build with LongSlitGeoSatellite.build_scan_block().

    Array layout
    ------------
    All 2D arrays have shape (n_rows, n_cols):
      axis 0 — N-S pixels along the slit (row 0 = southernmost)
      axis 1 — E-W scan steps (col 0 = westernmost / scan_start_lon)

    3D arrays have shape (n_rows, n_cols, K):
      sat_look_vecs : K=3  unit vector satellite→ground (ECEF)
      gnd_ecef      : K=3  ground point ECEF [km]
      corner_lats   : K=4  footprint corners NW,NE,SE,SW [deg]
      corner_lons   : K=4  footprint corners NW,NE,SE,SW [deg]

    Metadata attributes
    -------------------
    sat_lon_deg, sat_ecef, slit_center_lat, scan_start_lon, scan_end_lon,
    scan_lons (n_cols,), integration_time_s, scan_duration_s,
    t_start_utc, t_end_utc, col_times (list or None), n_rows, n_cols.

    Dict-like access
    ----------------
    block['vzas']          → (n_rows, n_cols) array
    block.keys()           → all array field names
    'szas' in block        → True
    block.vzas             → same as block['vzas']

    Persistence
    -----------
    block.save('path/to/file')       # writes .npz (adds extension if needed)
    block = ScanBlock.load('path')   # round-trips all arrays + metadata
    """

    # 2D scalar fields
    _FIELDS_2D = (
        'lats', 'lons', 'vzas', 'vaas', 'szas', 'saas',
        'airmass_geometric', 'relative_azimuth',
    )
    # 3D vector/corner fields
    _FIELDS_3D = ('sat_look_vecs', 'gnd_ecef', 'corner_lats', 'corner_lons')

    def __init__(self, data: dict, meta: dict) -> None:
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_meta', meta)

    # ---- dict-like interface ------------------------------------------------

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    # ---- attribute access (arrays + metadata) --------------------------------

    def __getattr__(self, name: str):
        if name in ('_data', '_meta'):        # guard against infinite recursion
            raise AttributeError(name)
        if name in self._data:
            return self._data[name]
        if name in self._meta:
            return self._meta[name]
        raise AttributeError(f"ScanBlock has no attribute '{name}'")

    # ---- shape ---------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_rows, n_cols) — (N-S pixels, E-W scan steps)."""
        return (self._meta['n_rows'], self._meta['n_cols'])

    @property
    def n_rows(self) -> int:
        return self._meta['n_rows']

    @property
    def n_cols(self) -> int:
        return self._meta['n_cols']

    # ---- save / load ---------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save to a compressed .npz file.

        Datetimes are stored as ISO-8601 strings; lists of datetimes as
        1-D string arrays.  Adds '.npz' extension if not present.
        """
        if not str(path).endswith('.npz'):
            path = str(path) + '.npz'
        arrays = dict(self._data)
        for k, v in self._meta.items():
            tag = f'_meta_{k}'
            if v is None:
                arrays[tag] = np.array('__None__')
            elif isinstance(v, datetime):
                arrays[tag] = np.array(v.isoformat())
            elif isinstance(v, list):
                if v and isinstance(v[0], datetime):
                    arrays[tag] = np.array([t.isoformat() for t in v])
                else:
                    arrays[tag] = np.array(v)
            elif isinstance(v, np.ndarray):
                arrays[tag] = v
            else:
                arrays[tag] = np.array(v)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> 'ScanBlock':
        """Load a ScanBlock previously saved with .save()."""
        npz = np.load(path, allow_pickle=False)
        data, raw_meta = {}, {}
        for key in npz.files:
            (raw_meta if key.startswith('_meta_') else data)[key] = npz[key]

        meta = {}
        for k, arr in raw_meta.items():
            name = k[6:]         # strip '_meta_' prefix
            arr  = np.asarray(arr)
            if arr.ndim == 0:
                val = arr.item()
                if isinstance(val, str):
                    if val == '__None__':
                        meta[name] = None
                    else:
                        try:
                            meta[name] = datetime.fromisoformat(val)
                        except ValueError:
                            meta[name] = val
                elif isinstance(val, (int, np.integer)):
                    meta[name] = int(val)
                else:
                    meta[name] = val
            elif arr.ndim == 1 and arr.dtype.kind in ('U', 'S', 'O'):
                meta[name] = [datetime.fromisoformat(str(s))
                               for s in arr if str(s) != '__None__']
            else:
                meta[name] = arr
        return cls(data, meta)

    # ---- coarsen -------------------------------------------------------------

    def coarsen(self,
                dlat_deg:  float,
                dlon_deg:  float,
                min_count: int = 1) -> 'ScanBlock':
        """
        Coarsen to a regular lat/lon grid.  Convenience wrapper around
        coarsen_scan_block(); see that function for full documentation.

        Parameters
        ----------
        dlat_deg, dlon_deg : float
            Target grid resolution [degrees].
        min_count : int
            Minimum fine pixels required for a valid coarse cell.

        Returns
        -------
        ScanBlock on a regular (dlat_deg × dlon_deg) grid.
        """
        return coarsen_scan_block(self, dlat_deg, dlon_deg, min_count)

    # ---- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        m    = self._meta
        t    = m.get('t_start_utc')
        tstr = f", t={t.strftime('%Y-%m-%dT%H:%MZ')}" if isinstance(t, datetime) else ""
        cstr = (f" [coarsened {m['dlat_deg']}°×{m['dlon_deg']}°]"
                if 'dlat_deg' in m else "")
        return (
            f"ScanBlock({m['n_rows']}×{m['n_cols']} px | "
            f"lat={m['slit_center_lat']:.1f}° | "
            f"lon=[{m['scan_start_lon']:.1f}°,{m['scan_end_lon']:.1f}°] | "
            f"dur={m['scan_duration_s']/60:.1f} min{tstr}{cstr})"
        )


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg: np.ndarray,
                     lon_deg: np.ndarray,
                     alt_km:  float | np.ndarray = 0.0) -> np.ndarray:
    """
    Geodetic (lat, lon, alt) → ECEF [km].

    Supports scalar or array inputs; broadcasts alt_km.
    Returns shape (..., 3).
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N   = RE_KM / np.sqrt(1.0 - E2_WGS84 * np.sin(lat)**2)
    x = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - E2_WGS84) + alt_km) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ECEF [km] → geodetic (lat_deg, lon_deg, alt_km).

    Iterative Bowring method; xyz shape (..., 3).
    Returns lat_deg, lon_deg, alt_km each shape (...).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    p   = np.sqrt(x**2 + y**2)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - E2_WGS84))
    for _ in range(6):
        N       = RE_KM / np.sqrt(1.0 - E2_WGS84 * np.sin(lat)**2)
        lat_new = np.arctan2(z + E2_WGS84 * N * np.sin(lat), p)
        if np.max(np.abs(lat_new - lat)) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    N   = RE_KM / np.sqrt(1.0 - E2_WGS84 * np.sin(lat)**2)
    alt = np.where(np.abs(np.cos(lat)) > 1e-10,
                   p / np.cos(lat) - N,
                   np.abs(z) / np.sin(np.abs(lat)) - N * (1.0 - E2_WGS84))
    return np.rad2deg(lat), np.rad2deg(lon), alt


def _local_enu(lat_deg: float, lon_deg: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East, North, Up unit vectors in ECEF at a geodetic point."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    east  = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon),
                      -np.sin(lat) * np.sin(lon),
                       np.cos(lat)])
    up    = np.array([ np.cos(lat) * np.cos(lon),
                       np.cos(lat) * np.sin(lon),
                       np.sin(lat)])
    return east, north, up


# ---------------------------------------------------------------------------
# Low-precision solar position (accuracy ~1°)
# ---------------------------------------------------------------------------

def solar_position_ecef(dt_utc: datetime) -> np.ndarray:
    """
    Return Sun direction unit vector in ECEF.

    Simplified from Reda & Andreas (2004).  Accuracy ~1°, adequate for
    computing solar angles in an observation geometry context.
    """
    jd = (dt_utc - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0
    T  = jd / 36525.0
    L0 = np.radians((280.46646 + 36000.76983 * T) % 360.0)
    M  = np.radians((357.52911 + 35999.05029 * T) % 360.0)
    C  = np.radians((1.914602 - 0.004817 * T) * np.sin(M)
                    + 0.019993 * np.sin(2 * M))
    sun_lon = L0 + C
    eps  = np.radians(23.439291 - 0.013004 * T)
    sx   = np.cos(sun_lon)
    sy   = np.cos(eps) * np.sin(sun_lon)
    sz   = np.sin(eps) * np.sin(sun_lon)
    gmst = np.radians((280.46061837 + 360.98564736629 * jd) % 360.0)
    x    = sx * np.cos(gmst) + sy * np.sin(gmst)
    y    = -sx * np.sin(gmst) + sy * np.cos(gmst)
    return np.array([x, y, sz])


# ---------------------------------------------------------------------------
# Small geodetic offset utility
# ---------------------------------------------------------------------------

def _offset_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray,
                   dn_km: float, de_km: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offset (lat, lon) by dn_km northward and de_km eastward.
    Uses meridional/normal radii of curvature on the WGS-84 ellipsoid.
    Handles scalar or array lat_deg/lon_deg.
    """
    lat    = np.deg2rad(lat_deg)
    N_r    = RE_KM / np.sqrt(1.0 - E2_WGS84 * np.sin(lat)**2)
    M_r    = RE_KM * (1.0 - E2_WGS84) / (1.0 - E2_WGS84 * np.sin(lat)**2)**1.5
    new_lat = lat_deg + np.degrees(dn_km / M_r)
    new_lon = lon_deg + np.degrees(de_km / (N_r * np.cos(lat)))
    # Ensure both outputs share the same shape (handles scalar + array)
    shape  = np.broadcast_shapes(np.shape(new_lat), np.shape(new_lon))
    return (np.broadcast_to(new_lat, shape).copy(),
            np.broadcast_to(new_lon, shape).copy())


# ---------------------------------------------------------------------------
# Ray – WGS-84 ellipsoid intersection
# ---------------------------------------------------------------------------

def _ray_ellipsoid_intersect(sat_ecef: np.ndarray,
                              look_dirs: np.ndarray) -> np.ndarray:
    """
    Intersect rays originating at the satellite with the WGS-84 ellipsoid.

    Parameters
    ----------
    sat_ecef  : (3,) satellite ECEF position [km]
    look_dirs : (..., 3) unit direction vectors from satellite toward Earth

    Returns
    -------
    (..., 3) ECEF intersection points [km]
    """
    a2     = RE_KM ** 2
    b2     = RE_KM ** 2 * (1.0 - E2_WGS84)
    inv_sq = np.array([1.0 / a2, 1.0 / a2, 1.0 / b2])
    S    = sat_ecef                                               # (3,)
    D    = look_dirs                                              # (..., 3)
    A    = np.sum(D ** 2       * inv_sq, axis=-1)                # (...)
    B    = 2.0 * np.sum(S * D * inv_sq, axis=-1)                 # (...)
    C    = float(np.sum(S ** 2 * inv_sq)) - 1.0                  # scalar
    disc = B ** 2 - 4.0 * A * C
    hit  = disc >= 0.0                                            # ray hits Earth
    # Avoid sqrt of negative; use 0 as a safe placeholder where disc < 0,
    # then overwrite those entries with NaN so callers can detect misses.
    t    = np.where(hit,
                    (-B - np.sqrt(np.where(hit, disc, 0.0))) / (2.0 * A),
                    np.nan)                                       # (...)
    return S + t[..., np.newaxis] * D                            # (..., 3)  NaN for misses


# ---------------------------------------------------------------------------
# Internal vectorised geometry kernel (shared by slit_geometry_vectorized
# and build_scan_block)
# ---------------------------------------------------------------------------

def _viewing_geometry_vectorized(
        lats_f: np.ndarray,
        lons_f: np.ndarray,
        sat_ecef: np.ndarray,
        sun_per_pixel: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute full per-pixel viewing/solar geometry for flat arrays of ground
    points.

    Parameters
    ----------
    lats_f, lons_f : (N,)
    sat_ecef       : (3,)
    sun_per_pixel  : (N, 3) or None — if None, solar angles are NaN.

    Returns
    -------
    dict with keys: vzas, vaas, szas, saas, look_to_sat, sat_look_vecs,
                    gnd_ecef, up_vecs, north_vecs, east_vecs,
                    airmass_geometric, relative_azimuth
    All shape (N,) or (N, 3).
    """
    gnd_ecef    = geodetic_to_ecef(lats_f, lons_f)                  # (N, 3)
    vec_to_sat  = sat_ecef[np.newaxis, :] - gnd_ecef                # (N, 3)
    norms       = np.linalg.norm(vec_to_sat, axis=1, keepdims=True)
    look_to_sat = vec_to_sat / norms                                 # ground→sat

    lat_r = np.deg2rad(lats_f)
    lon_r = np.deg2rad(lons_f)
    clat, slat = np.cos(lat_r), np.sin(lat_r)
    clon, slon = np.cos(lon_r), np.sin(lon_r)

    up_vecs    = np.stack([ clat * clon,  clat * slon,  slat                ], axis=1)
    north_vecs = np.stack([-slat * clon, -slat * slon,  clat                ], axis=1)
    east_vecs  = np.stack([-slon,         clon,          np.zeros_like(slon)], axis=1)

    cos_vza = np.einsum('ij,ij->i', look_to_sat, up_vecs)
    vzas    = np.degrees(np.arccos(np.clip(cos_vza, -1.0, 1.0)))

    horiz = look_to_sat - cos_vza[:, np.newaxis] * up_vecs
    vaas  = (np.degrees(np.arctan2(
                 np.einsum('ij,ij->i', horiz, east_vecs),
                 np.einsum('ij,ij->i', horiz, north_vecs))) % 360.0)

    N = len(lats_f)
    if sun_per_pixel is not None:
        cos_sza = np.einsum('ij,ij->i', up_vecs, sun_per_pixel)
        szas    = np.degrees(np.arccos(np.clip(cos_sza, -1.0, 1.0)))
        sun_h   = sun_per_pixel - cos_sza[:, np.newaxis] * up_vecs
        saas    = (np.degrees(np.arctan2(
                      np.einsum('ij,ij->i', sun_h, east_vecs),
                      np.einsum('ij,ij->i', sun_h, north_vecs))) % 360.0)
    else:
        szas = np.full(N, np.nan)
        saas = np.full(N, np.nan)
        cos_sza = None

    with np.errstate(divide='ignore', invalid='ignore'):
        am = np.where(vzas < 89.9, 1.0 / np.cos(np.deg2rad(vzas)), np.inf)
        if cos_sza is not None:
            am += np.where(szas < 89.9, 1.0 / np.cos(np.deg2rad(szas)), np.inf)

    rel_az = np.abs(vaas - saas)
    rel_az = np.where(rel_az > 180.0, 360.0 - rel_az, rel_az)

    return dict(
        vzas=vzas, vaas=vaas, szas=szas, saas=saas,
        look_to_sat=look_to_sat,
        sat_look_vecs=-look_to_sat,
        gnd_ecef=gnd_ecef,
        up_vecs=up_vecs, north_vecs=north_vecs, east_vecs=east_vecs,
        airmass_geometric=am, relative_azimuth=rel_az,
    )


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class LongSlitGeoSatellite:
    """
    Observation geometry simulator for a long-slit geostationary satellite.

    Parameters
    ----------
    sat_lon_deg : float
        Sub-satellite point longitude [degrees East].
    slit_length_km : float
        Physical slit length projected on ground at nadir [km], N-S direction.
    pixel_size_ew_km : float
        Pixel size in the E-W (along-scan) direction at nadir [km].
    pixel_size_ns_km : float
        Pixel size in the N-S (along-slit) direction at nadir [km].
    integration_time_s : float
        Integration time per scan step [seconds].
    sat_alt_km : float
        Satellite altitude above the equator [km].  Default: 35786 km.
    scan_rate_kms : float or None
        E-W scan rate [km/s at sub-satellite point].
        If None (default), set to pixel_size_ew_km / integration_time_s.

    Key derived attributes
    ----------------------
    n_pixels   : int    — number of detector pixels along the slit
    sat_ecef   : (3,)   — satellite ECEF position [km]
    ifov_ew_rad: float  — instantaneous FOV in E-W direction [rad]
    ifov_ns_rad: float  — instantaneous FOV in N-S direction [rad]

    Example (GeoCarb-like)
    ----------------------
    >>> sat = LongSlitGeoSatellite(sat_lon_deg=-95.0, slit_length_km=3000.0,
    ...                             pixel_size_ew_km=6.0, pixel_size_ns_km=6.0,
    ...                             integration_time_s=10.0)
    >>> block = sat.build_scan_block(30.0, -115.0, -75.0,
    ...                               t0_utc=datetime(2020, 7, 1, 18))
    >>> fig, ax = plot_scan_blocks(block, field='vzas')
    """

    def __init__(self,
                 sat_lon_deg:        float,
                 slit_length_km:     float = 3000.0,
                 pixel_size_ew_km:   float = 6.0,
                 pixel_size_ns_km:   float = 6.0,
                 integration_time_s: float = 10.0,
                 sat_alt_km:         float = SAT_ALT_KM,
                 scan_rate_kms:      Optional[float] = None):

        self.sat_lon_deg        = float(sat_lon_deg)
        self.slit_length_km     = float(slit_length_km)
        self.pixel_size_ew_km   = float(pixel_size_ew_km)
        self.pixel_size_ns_km   = float(pixel_size_ns_km)
        self.integration_time_s = float(integration_time_s)
        self.sat_alt_km         = float(sat_alt_km)

        self.sat_ecef   = geodetic_to_ecef(0.0, self.sat_lon_deg,
                                           alt_km=self.sat_alt_km)
        self.n_pixels   = int(round(slit_length_km / pixel_size_ns_km))
        self.scan_rate_kms = (float(scan_rate_kms) if scan_rate_kms is not None
                               else self.pixel_size_ew_km / self.integration_time_s)
        self.ifov_ew_rad = np.arctan(pixel_size_ew_km / sat_alt_km)
        self.ifov_ns_rad = np.arctan(pixel_size_ns_km / sat_alt_km)

    # ------------------------------------------------------------------
    # Slit pixel centre positions
    # ------------------------------------------------------------------

    def slit_centers(self,
                     center_lat_deg: float = 0.0,
                     center_lon_deg: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Geodetic centres of all pixels along the slit.

        Returns lats_deg, lons_deg arrays of shape (n_pixels,).
        Pixel 0 is southernmost.

        Each pixel is found by rotating the satellite-to-centre look vector
        by the appropriate N-S angular step in the focal plane (constant
        azimuth, varying elevation) and intersecting with the WGS-84
        ellipsoid.  This correctly reproduces the slight longitude curvature
        of an off-nadir slit in geographic coordinates.
        """
        if center_lon_deg is None:
            center_lon_deg = self.sat_lon_deg

        # Look vector from satellite to slit centre
        gnd_center = geodetic_to_ecef(float(center_lat_deg),
                                      float(center_lon_deg))     # (3,)
        L0 = gnd_center - self.sat_ecef
        L0 = L0 / np.linalg.norm(L0)

        # N-S slit direction in the focal plane: component of Earth's North
        # pole perpendicular to the current look vector
        north    = np.array([0.0, 0.0, 1.0])
        slit_dir = north - np.dot(north, L0) * L0
        slit_dir = slit_dir / np.linalg.norm(slit_dir)

        # Angular offset for each pixel (negative = south, positive = north)
        alphas = ((np.arange(self.n_pixels) - (self.n_pixels - 1) / 2.0)
                  * self.ifov_ns_rad)                             # (n_pixels,)

        # Rotated look directions then ellipsoid intersection
        look_dirs = (np.cos(alphas)[:, np.newaxis] * L0
                     + np.sin(alphas)[:, np.newaxis] * slit_dir) # (n_pixels, 3)
        gnd_pts = _ray_ellipsoid_intersect(self.sat_ecef, look_dirs)
        lats, lons, _ = ecef_to_geodetic(gnd_pts)
        return lats, lons

    # ------------------------------------------------------------------
    # Footprint polygons
    # ------------------------------------------------------------------

    def footprint_polygons(self,
                           lats_deg: np.ndarray,
                           lons_deg: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        4-corner footprint polygons for an array of pixel centres.

        Corners ordered NW → NE → SE → SW (CCW when viewed from above).
        Returns corner_lats, corner_lons each shape (N, 4) [degrees].
        """
        lats  = np.asarray(lats_deg)
        lons  = np.asarray(lons_deg)
        lat_r = np.deg2rad(lats)
        N_r   = RE_KM / np.sqrt(1.0 - E2_WGS84 * np.sin(lat_r)**2)
        M_r   = RE_KM * (1.0 - E2_WGS84) / (1.0 - E2_WGS84 * np.sin(lat_r)**2)**1.5
        dN    =  self.pixel_size_ns_km / 2.0
        dE    =  self.pixel_size_ew_km / 2.0
        dlat  = np.degrees(dN / M_r)
        dlon  = np.degrees(dE / (N_r * np.cos(lat_r)))
        corner_lats = np.stack([lats + dlat, lats + dlat,
                                 lats - dlat, lats - dlat], axis=1)
        corner_lons = np.stack([lons - dlon, lons + dlon,
                                 lons + dlon, lons - dlon], axis=1)
        return corner_lats, corner_lons

    # ------------------------------------------------------------------
    # Vectorised slit geometry — single slit position
    # ------------------------------------------------------------------

    def slit_geometry_vectorized(self,
                                  center_lat_deg: float,
                                  center_lon_deg: float,
                                  dt_utc: Optional[datetime] = None
                                  ) -> dict:
        """
        Compute full observation geometry for every pixel in one slit position.

        All operations are vectorised over the slit (no Python loops).

        Returns
        -------
        dict with keys (all shape (n_pixels,) unless noted):
          lats, lons, vzas, vaas, szas, saas,
          look_vecs (ground→sat), sat_look_vecs (sat→gnd),
          corner_lats, corner_lons (n_pixels, 4),
          gnd_ecef, sat_ecef, airmass_geometric, relative_azimuth.
        """
        lats, lons = self.slit_centers(center_lat_deg, center_lon_deg)
        sun = solar_position_ecef(dt_utc) if dt_utc is not None else None
        sun_pp = (np.broadcast_to(sun[np.newaxis, :],
                                   (self.n_pixels, 3)).copy()
                  if sun is not None else None)

        g = _viewing_geometry_vectorized(lats, lons, self.sat_ecef, sun_pp)
        corner_lats, corner_lons = self.footprint_polygons(lats, lons)

        return dict(
            lats=lats, lons=lons,
            vzas=g['vzas'], vaas=g['vaas'],
            szas=g['szas'], saas=g['saas'],
            look_vecs=g['look_to_sat'],
            sat_look_vecs=g['sat_look_vecs'],
            corner_lats=corner_lats,
            corner_lons=corner_lons,
            gnd_ecef=g['gnd_ecef'],
            sat_ecef=self.sat_ecef.copy(),
            airmass_geometric=g['airmass_geometric'],
            relative_azimuth=g['relative_azimuth'],
            valid_mask=np.isfinite(lats),   # False where ray misses the Earth
        )

    # ------------------------------------------------------------------
    # Build ScanBlock — full 2D geometry for a rectangular scan region
    # ------------------------------------------------------------------

    def build_scan_block(self,
                         slit_center_lat: float,
                         scan_start_lon:  float,
                         scan_end_lon:    float,
                         t0_utc:          Optional[datetime] = None,
                         n_cols:          Optional[int] = None,
                         ) -> ScanBlock:
        """
        Build a pre-computed ScanBlock for a rectangular scan region.

        The slit is held at slit_center_lat (N-S) while stepping E-W from
        scan_start_lon to scan_end_lon.  Solar angles are computed per-column
        using the correct UTC time for each integration step.

        Parameters
        ----------
        slit_center_lat : float
            Nominal latitude of the slit centre [degrees].
        scan_start_lon, scan_end_lon : float
            Longitude extent [degrees East].  Start is the west edge.
        t0_utc : datetime or None
            UTC time at the first scan column.  Each subsequent column
            advances by integration_time_s.  If None, solar angles are NaN.
        n_cols : int or None
            Number of E-W scan steps.  If None, derived from the ground
            distance at slit_center_lat divided by pixel_size_ew_km.

        Returns
        -------
        ScanBlock
            All 2D arrays have shape (n_rows, n_cols) with:
              axis 0 = N-S slit pixels (row 0 = southernmost)
              axis 1 = E-W scan steps  (col 0 = scan_start_lon)
        """
        # ---- column longitudes ----
        if n_cols is None:
            lat_r    = np.deg2rad(slit_center_lat)
            km_range = (abs(scan_end_lon - scan_start_lon)
                        * np.pi / 180.0 * RE_KM * np.cos(lat_r))
            n_cols   = max(1, int(round(km_range / self.pixel_size_ew_km)))
        scan_lons = np.linspace(scan_start_lon, scan_end_lon, n_cols)
        n_rows    = self.n_pixels

        # ---- pixel centres (n_rows, n_cols) ----
        # For each scan column, rotate the look vector to the slit centre in
        # the N-S focal-plane direction and ray-cast to the ellipsoid.  This
        # correctly models the slight longitude curvature of an off-nadir slit.

        # Column slit-centre ground positions and look vectors — (n_cols, 3)
        gnd_cols  = geodetic_to_ecef(np.full(n_cols, slit_center_lat),
                                     scan_lons)                          # (n_cols, 3)
        L0_cols   = gnd_cols - self.sat_ecef[np.newaxis, :]             # (n_cols, 3)
        L0_cols   = L0_cols / np.linalg.norm(L0_cols, axis=1,
                                              keepdims=True)

        # N-S slit direction per column — (n_cols, 3)
        north     = np.array([0.0, 0.0, 1.0])
        slit_dirs = north - (L0_cols @ north)[:, np.newaxis] * L0_cols  # (n_cols, 3)
        slit_dirs = slit_dirs / np.linalg.norm(slit_dirs, axis=1,
                                                keepdims=True)

        # Angular offsets for each row pixel — (n_rows,)
        alphas = ((np.arange(n_rows) - (n_rows - 1) / 2.0)
                  * self.ifov_ns_rad)

        # Look directions for every (row, col) pair — (n_rows, n_cols, 3)
        look_2d = (np.cos(alphas)[:, np.newaxis, np.newaxis]
                   * L0_cols[np.newaxis, :, :]
                   + np.sin(alphas)[:, np.newaxis, np.newaxis]
                   * slit_dirs[np.newaxis, :, :])

        # Ray-ellipsoid intersection → geodetic lat/lon
        gnd_2d              = _ray_ellipsoid_intersect(self.sat_ecef, look_2d)
        lats_2d, lons_2d, _ = ecef_to_geodetic(gnd_2d)                  # (n_rows, n_cols)

        # ---- flatten (C/row-major order) ----
        # flat index k  →  row = k // n_cols,  col = k % n_cols
        lats_f = lats_2d.ravel()   # (n_rows*n_cols,)
        lons_f = lons_2d.ravel()

        # ---- solar angles: per-column time, broadcast across rows ----
        sun_per_pixel = None
        col_times     = None
        if t0_utc is not None:
            col_times  = [t0_utc + timedelta(seconds=j * self.integration_time_s)
                          for j in range(n_cols)]
            sun_vecs   = np.array([solar_position_ecef(dt)
                                   for dt in col_times])                # (n_cols, 3)
            # For C-order ravel of (n_rows, n_cols): col of pixel k = k % n_cols
            col_idx    = np.tile(np.arange(n_cols), n_rows)             # (n_rows*n_cols,)
            sun_per_pixel = sun_vecs[col_idx]                           # (N, 3)

        # ---- all geometry in one vectorised pass ----
        g = _viewing_geometry_vectorized(lats_f, lons_f,
                                          self.sat_ecef, sun_per_pixel)

        # ---- footprint corners (flat, then reshape) ----
        lat_r_f = np.deg2rad(lats_f)
        N_r_f   = RE_KM / np.sqrt(1.0 - E2_WGS84 * np.sin(lat_r_f)**2)
        M_r_f   = (RE_KM * (1.0 - E2_WGS84)
                   / (1.0 - E2_WGS84 * np.sin(lat_r_f)**2)**1.5)
        dN  = self.pixel_size_ns_km / 2.0
        dE  = self.pixel_size_ew_km / 2.0
        dlat = np.degrees(dN / M_r_f)
        dlon = np.degrees(dE / (N_r_f * np.cos(lat_r_f)))
        corner_lats_f = np.stack([lats_f + dlat, lats_f + dlat,
                                   lats_f - dlat, lats_f - dlat], axis=1)
        corner_lons_f = np.stack([lons_f - dlon, lons_f + dlon,
                                   lons_f + dlon, lons_f - dlon], axis=1)

        # ---- reshape helpers ----
        def r2(a):    return a.reshape(n_rows, n_cols)
        def r3(a, k): return a.reshape(n_rows, n_cols, k)

        scan_duration_s = (n_cols - 1) * self.integration_time_s
        t_end_utc       = col_times[-1] if col_times else None

        data = {
            'lats':              r2(lats_f),
            'lons':              r2(lons_f),
            'valid_mask':        r2(np.isfinite(lats_f)),   # False where ray misses Earth
            'vzas':              r2(g['vzas']),
            'vaas':              r2(g['vaas']),
            'szas':              r2(g['szas']),
            'saas':              r2(g['saas']),
            'airmass_geometric': r2(g['airmass_geometric']),
            'relative_azimuth':  r2(g['relative_azimuth']),
            'sat_look_vecs':     r3(g['sat_look_vecs'],   3),
            'gnd_ecef':          r3(g['gnd_ecef'],        3),
            'corner_lats':       r3(corner_lats_f,        4),
            'corner_lons':       r3(corner_lons_f,        4),
        }
        meta = {
            'sat_lon_deg':        self.sat_lon_deg,
            'sat_ecef':           self.sat_ecef.copy(),
            'slit_center_lat':    float(slit_center_lat),
            'scan_start_lon':     float(scan_start_lon),
            'scan_end_lon':       float(scan_end_lon),
            'scan_lons':          scan_lons,
            'integration_time_s': self.integration_time_s,
            'scan_duration_s':    float(scan_duration_s),
            't_start_utc':        t0_utc,
            't_end_utc':          t_end_utc,
            'col_times':          col_times,
            'n_rows':             n_rows,
            'n_cols':             n_cols,
        }
        return ScanBlock(data, meta)

    # ------------------------------------------------------------------
    # Ray-path intercept points (vectorised over pixels and shells)
    # ------------------------------------------------------------------

    def compute_ray_paths_vectorized(self,
                                      lats_deg:     np.ndarray,
                                      lons_deg:     np.ndarray,
                                      altitudes_km: np.ndarray
                                      ) -> dict:
        """
        Intersect satellite-to-ground rays with spherical atmospheric shells.

        Ray: P(t) = sat_ecef + t*(gnd_ecef - sat_ecef), t=0 at satellite,
        t=1 at ground.  Returns the entry intercept from above.

        Parameters
        ----------
        lats_deg, lons_deg : (N,) ground point coordinates [degrees].
        altitudes_km       : (M,) shell altitudes [km], sorted ascending.

        Returns
        -------
        dict:
          intercept_pts  : (N, M, 3) ECEF [km]
          slant_lengths  : (N, M-1)  path length in each layer [km]
          t_params       : (N, M)    parametric t ∈ [0, 1]
          lat_intercepts : (N, M)    geodetic latitude [deg]
          lon_intercepts : (N, M)    longitude [deg]
          alt_intercepts : (N, M)    altitude [km] (≈ altitudes_km)
        """
        altitudes_km = np.asarray(altitudes_km, dtype=float)
        lats_deg     = np.asarray(lats_deg,     dtype=float)
        lons_deg     = np.asarray(lons_deg,     dtype=float)
        N = len(lats_deg)
        M = len(altitudes_km)

        gnd_ecef = geodetic_to_ecef(lats_deg, lons_deg)
        ray_dirs = gnd_ecef - self.sat_ecef[np.newaxis, :]   # sat→gnd
        sat      = self.sat_ecef

        # |sat + t*d|^2 = r^2  →  A t^2 + 2 B_half t + C = 0
        A      = np.einsum('ij,ij->i', ray_dirs, ray_dirs)   # |d|^2
        B_half = np.einsum('ij,j->i',  ray_dirs, sat)        # d·sat
        sat_sq = float(np.dot(sat, sat))

        intercept_pts = np.zeros((N, M, 3))
        t_params      = np.zeros((N, M))

        for j, alt in enumerate(altitudes_km):
            r    = RE_KM + alt
            C    = sat_sq - r**2
            disc = np.maximum(B_half**2 - A * C, 0.0)
            sq   = np.sqrt(disc)
            # t1 (near root) = entry from above; already in [0,1] space
            t1   = (-B_half - sq) / A
            t_n  = np.clip(t1, 0.0, 1.0)
            intercept_pts[:, j, :] = sat[np.newaxis, :] + t_n[:, np.newaxis] * ray_dirs
            t_params[:, j]         = t_n

        diffs         = np.diff(intercept_pts, axis=1)
        slant_lengths = np.linalg.norm(diffs, axis=2)

        flat_lat, flat_lon, flat_alt = ecef_to_geodetic(
            intercept_pts.reshape(-1, 3))
        return dict(
            intercept_pts=intercept_pts,
            slant_lengths=slant_lengths,
            t_params=t_params,
            lat_intercepts=flat_lat.reshape(N, M),
            lon_intercepts=flat_lon.reshape(N, M),
            alt_intercepts=flat_alt.reshape(N, M),
        )

    # ------------------------------------------------------------------
    # Single-observation ray path (convenience wrapper)
    # ------------------------------------------------------------------

    def ray_path(self, lat_deg: float, lon_deg: float,
                 altitudes_km: np.ndarray) -> RayPath:
        """Ray path for a single ground point.  For batch use prefer
        compute_ray_paths_vectorized()."""
        r = self.compute_ray_paths_vectorized(
            np.array([lat_deg]), np.array([lon_deg]),
            np.asarray(altitudes_km))
        return RayPath(
            altitudes_km=np.asarray(altitudes_km),
            intercept_pts=r['intercept_pts'][0],
            slant_lengths=r['slant_lengths'][0],
            t_params=r['t_params'][0],
            lat_pts=r['lat_intercepts'][0],
            lon_pts=r['lon_intercepts'][0],
        )

    # ------------------------------------------------------------------
    # Full scan simulation (list of ObservationGeometry objects)
    # ------------------------------------------------------------------

    def simulate_scan(self,
                      scan_center_lat: float,
                      scan_start_lon:  float,
                      scan_end_lon:    float,
                      t0_utc:          Optional[datetime] = None,
                      ) -> List[ObservationGeometry]:
        """
        Simulate a full E-W scan; returns one ObservationGeometry per pixel
        per column step.  Ordered by scan column (outer) then slit row (inner).
        """
        lat_r    = np.deg2rad(scan_center_lat)
        km_range = abs(scan_end_lon - scan_start_lon) * np.pi / 180.0 * RE_KM
        n_cols   = max(1, int(round(km_range / self.pixel_size_ew_km)))
        scan_lons = np.linspace(scan_start_lon, scan_end_lon, n_cols)

        observations = []
        for col_idx, clon in enumerate(scan_lons):
            dt = (t0_utc + timedelta(seconds=col_idx * self.integration_time_s)
                  if t0_utc is not None else None)
            g  = self.slit_geometry_vectorized(scan_center_lat, clon, dt)
            for row_idx in range(self.n_pixels):
                observations.append(ObservationGeometry(
                    lat_deg=float(g['lats'][row_idx]),
                    lon_deg=float(g['lons'][row_idx]),
                    corner_lats=g['corner_lats'][row_idx],
                    corner_lons=g['corner_lons'][row_idx],
                    vza_deg=float(g['vzas'][row_idx]),
                    vaa_deg=float(g['vaas'][row_idx]),
                    sza_deg=float(g['szas'][row_idx]),
                    saa_deg=float(g['saas'][row_idx]),
                    sat_ecef=self.sat_ecef.copy(),
                    gnd_ecef=g['gnd_ecef'][row_idx],
                    look_vec=g['sat_look_vecs'][row_idx],
                    scan_col=col_idx,
                    slit_row=row_idx,
                    utc_time=dt,
                ))
        return observations

    def __repr__(self) -> str:
        return (f"LongSlitGeoSatellite(lon={self.sat_lon_deg}°, "
                f"slit={self.slit_length_km:.0f} km × {self.n_pixels} px, "
                f"pixel={self.pixel_size_ns_km}×{self.pixel_size_ew_km} km, "
                f"τ={self.integration_time_s}s)")


# ---------------------------------------------------------------------------
# Plot utility
# ---------------------------------------------------------------------------

#: Human-readable labels for ScanBlock fields used in plot colorbars.
_FIELD_LABELS = {
    'vzas':              'Viewing Zenith Angle [°]',
    'vaas':              'Viewing Azimuth Angle [°]',
    'szas':              'Solar Zenith Angle [°]',
    'saas':              'Solar Azimuth Angle [°]',
    'airmass_geometric': 'Geometric Airmass',
    'relative_azimuth':  'Relative Azimuth [°]',
    'lats':              'Latitude [°]',
    'lons':              'Longitude [°]',
    'pixel_counts':      'Fine Pixels per Coarse Cell',
    'land_fraction':     'Land Fraction',
}


def plot_scan_blocks(
        blocks,
        field:              str            = 'vzas',
        mode:               str            = 'footprints',
        ax                                 = None,
        cmap:               str            = 'viridis',
        vmin:               Optional[float] = None,
        vmax:               Optional[float] = None,
        alpha:              float          = 0.85,
        show_colorbar:      bool           = True,
        colorbar_label:     Optional[str]  = None,
        show_sat_subpoint:  bool           = True,
        coastlines:         bool           = True,
        title:              Optional[str]  = None,
        block_labels:       Optional[List[str]] = None,
        outline_color:      str            = 'k',
        outline_lw:         float          = 0.3,
        figsize:            Tuple[int,int] = (13, 7),
        projection:         str            = 'platecarree',
) -> Tuple:
    """
    Plot a collection of ScanBlocks on a map.

    Parameters
    ----------
    blocks : ScanBlock or list[ScanBlock]
        One or more scan blocks to display.
    field : str
        2D array field to colour by.  Any key from ScanBlock works, e.g.
        'vzas', 'szas', 'airmass_geometric', 'relative_azimuth', 'lats',
        'land_fraction'.
    mode : str
        'footprints' — fill every pixel polygon (accurate but slow for
                       very large scans; uses PolyCollection).
        'pixels'     — scatter plot at pixel centres (fast, approximate).
        'outline'    — draw only the block boundary filled with median colour.
    ax : matplotlib Axes or None
        Existing axes to draw into.  If None, a new figure is created.
        Pass a cartopy GeoAxes to get automatic coastlines and projections.
    cmap, vmin, vmax : colormap and range.  vmin/vmax default to the 2nd/98th
        percentile of all finite values across all blocks.
    alpha : float
        Fill transparency.
    show_colorbar : bool
    colorbar_label : str or None  — defaults to a human-readable field label.
    show_sat_subpoint : bool  — mark the satellite sub-point (★, lat=0).
    coastlines : bool  — add coastlines/land/ocean if cartopy is available;
                         silently skipped if cartopy is not installed.
    title : str or None
    block_labels : list[str] or None  — optional text label per block.
    outline_color, outline_lw : edge colour and width for polygon outlines.
    figsize : (w, h) inches.
    projection : str
        Map projection to use when cartopy is available.

        ``'platecarree'`` (default)
            Standard rectilinear lat/lon axes.  A padded extent is set
            automatically from the block data.
        ``'geostationary'``
            Full-disk satellite view centred on the sub-satellite longitude
            (read from block metadata).  Requires cartopy.  Grid-line labels
            are suppressed because cartopy cannot place them on the disk edge.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.

    Notes
    -----
    Cartopy is an optional dependency.  If not installed the function falls
    back to a plain latitude/longitude Axes with no basemap.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if isinstance(blocks, ScanBlock):
        blocks = [blocks]

    # ---- try cartopy ----
    _cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _cartopy = True
        print("Cartopy found: using map projections and basemap features.")
    except ImportError:
        pass

    # ---- validate projection arg ----
    _VALID_PROJ = {'platecarree', 'geostationary'}
    if projection not in _VALID_PROJ:
        raise ValueError(
            f"projection must be one of {_VALID_PROJ}; got '{projection}'")
    _use_geo = (projection == 'geostationary') and _cartopy

    # ---- create figure / axes ----
    if ax is None:
        if _cartopy:
            import cartopy.crs as ccrs
            if _use_geo and blocks:
                sat_lon = float(blocks[0]._meta['sat_lon_deg'])
                _proj = ccrs.Geostationary(
                    central_longitude=sat_lon,
                    satellite_height=SAT_ALT_KM * 1_000.0)
            else:
                _proj = ccrs.PlateCarree()
            fig, ax = plt.subplots(figsize=figsize,
                                   subplot_kw={'projection': _proj})
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Only attach a geographic transform when ax is actually a GeoAxes.
    # Use an explicit isinstance check rather than hasattr — cartopy's GeoAxes
    # uses __getattr__ delegation internally which can make hasattr unreliable.
    # If the caller supplied a plain matplotlib Axes (e.g. ax=plt.subplots()[1])
    # we fall back to plain lat/lon plotting throughout.
    if _cartopy:
        from cartopy.mpl.geoaxes import GeoAxes as _GeoAxesClass
        _is_geoaxes = isinstance(ax, _GeoAxesClass)
    else:
        _is_geoaxes = False
    transform_kw = {}
    if _is_geoaxes:
        import cartopy.crs as ccrs
        transform_kw = {'transform': ccrs.PlateCarree()}

    # Re-derive _use_geo from the actual axes projection so that callers who
    # pass in a pre-existing Geostationary GeoAxes are handled correctly.
    if _is_geoaxes and _cartopy:
        import cartopy.crs as ccrs
        _use_geo = isinstance(ax.projection, ccrs.Geostationary)

    # ---- basemap ----
    if coastlines and _is_geoaxes:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND,      facecolor='#f0efe8', zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='#555555')
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, color='#888888',
                       linestyle=':')

    # ---- global colour range ----
    all_vals = []
    for blk in blocks:
        if field in blk:
            v = blk[field].ravel()
            all_vals.append(v[np.isfinite(v)])
    if all_vals:
        flat = np.concatenate(all_vals)
        _vmin = vmin if vmin is not None else float(np.nanpercentile(flat, 2))
        _vmax = vmax if vmax is not None else float(np.nanpercentile(flat, 98))
    else:
        _vmin, _vmax = 0.0, 1.0

    norm     = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
    cmap_obj = plt.get_cmap(cmap)
    sm       = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])

    all_lons_list, all_lats_list = [], []

    # ---- draw each block ----
    for b_idx, blk in enumerate(blocks):
        n_rows, n_cols = blk.shape
        vals = blk[field] if field in blk else np.zeros((n_rows, n_cols))

        all_lons_list.append(blk['lons'].ravel())
        all_lats_list.append(blk['lats'].ravel())

        if mode == 'footprints':
            # corner_lats/lons: (n_rows, n_cols, 4) — NW NE SE SW
            clat = blk['corner_lats'].reshape(-1, 4)   # (N, 4)
            clon = blk['corner_lons'].reshape(-1, 4)
            face_colors = cmap_obj(norm(vals.ravel()))
            if _use_geo and _is_geoaxes:
                # PolyCollection doesn't go through cartopy's reprojection
                # pipeline, so passing transform=PlateCarree() on a
                # Geostationary axes places polygons in wrong locations.
                # Pre-project lon/lat corners to native (x, y) metres instead.
                import cartopy.crs as ccrs
                N = clat.shape[0]
                xy = ax.projection.transform_points(
                    ccrs.PlateCarree(),
                    clon.ravel(), clat.ravel())        # (N*4, 3)
                verts    = xy[:, :2].reshape(N, 4, 2)  # drop z, keep x/y
                poly_kw  = {}                          # already in native coords
            else:
                # PolyCollection expects (N, vertices, 2) = (N, 4, (x, y))
                verts   = np.stack([clon, clat], axis=-1)   # (N, 4, 2)
                poly_kw = transform_kw
            pc = PolyCollection(verts,
                                facecolors=face_colors,
                                edgecolors=outline_color,
                                linewidths=outline_lw,
                                alpha=alpha,
                                **poly_kw)
            ax.add_collection(pc)

        elif mode == 'pixels':
            lats_f = blk['lats'].ravel()
            lons_f = blk['lons'].ravel()
            ax.scatter(lons_f, lats_f,
                       c=vals.ravel(), cmap=cmap_obj, norm=norm,
                       s=1.5, alpha=alpha, linewidths=0,
                       **transform_kw)

        elif mode == 'outline':
            lats = blk['lats']
            lons = blk['lons']
            # Trace the block boundary: top row → right col → bottom row
            # reversed → left col reversed
            blon = np.concatenate([lons[ 0, :], lons[:, -1],
                                    lons[-1, ::-1], lons[::-1,  0]])
            blat = np.concatenate([lats[ 0, :], lats[:, -1],
                                    lats[-1, ::-1], lats[::-1,  0]])
            med_color = cmap_obj(norm(float(np.nanmedian(vals))))
            ax.fill(blon, blat,
                    color=med_color, alpha=alpha * 0.45,
                    **transform_kw)
            ax.plot(np.append(blon, blon[0]),
                    np.append(blat, blat[0]),
                    color=outline_color, lw=1.2,
                    **transform_kw)
        else:
            raise ValueError(f"mode must be 'footprints', 'pixels', or "
                             f"'outline'; got '{mode}'")

        # optional per-block label at centroid
        if block_labels and b_idx < len(block_labels):
            cx = float(np.nanmean(blk['lons']))
            cy = float(np.nanmean(blk['lats']))
            ax.text(cx, cy, block_labels[b_idx],
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc='k', alpha=0.4),
                    zorder=20, **transform_kw)

    # ---- satellite sub-point ----
    if show_sat_subpoint and blocks:
        sat_lon = float(blocks[0]._meta['sat_lon_deg'])
        ax.plot(sat_lon, 0.0, marker='*', ms=14, color='crimson',
                zorder=15, label=f'Satellite ({sat_lon:.1f}°E)',
                **transform_kw)

    # ---- map extent ----
    all_lons = np.concatenate(all_lons_list)
    all_lats = np.concatenate(all_lats_list)
    pad_lon, pad_lat = 4.0, 3.0
    x0 = all_lons.min() - pad_lon
    x1 = all_lons.max() + pad_lon
    y0 = all_lats.min() - pad_lat
    y1 = all_lats.max() + pad_lat

    if _is_geoaxes:
        # GeoAxes (cartopy) — use geographic extent / gridlines
        import cartopy.crs as ccrs
        if _use_geo:
            # Show the full Earth disk as seen by the satellite.
            # ax.gridlines() raises errors on Geostationary axes in most
            # cartopy versions because the projection does not support
            # gridline computation — skip it for this projection.
            ax.set_global()
        else:
            ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                              color='gray', alpha=0.6, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
    else:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel('Longitude [°E]')
        ax.set_ylabel('Latitude [°]')
        ax.grid(True, lw=0.3, alpha=0.5)

    # ---- colorbar ----
    if show_colorbar:
        label = colorbar_label or _FIELD_LABELS.get(field, field)
        plt.colorbar(sm, ax=ax, label=label, shrink=0.75, pad=0.04)

    # ---- title ----
    if title:
        ax.set_title(title, fontsize=11)
    else:
        t = blocks[0]._meta.get('t_start_utc')
        tstr = (f'  —  {t.strftime("%Y-%m-%d %H:%M UTC")}'
                if isinstance(t, datetime) else '')
        ax.set_title(f'Scan blocks  |  {_FIELD_LABELS.get(field, field)}{tstr}',
                     fontsize=11)

    if show_sat_subpoint:
        ax.legend(loc='lower right', fontsize=8)

    return fig, ax


# ---------------------------------------------------------------------------
# Coarsening utility
# ---------------------------------------------------------------------------

def coarsen_scan_block(block:     ScanBlock,
                        dlat_deg:  float,
                        dlon_deg:  float,
                        min_count: int = 1) -> ScanBlock:
    """
    Coarsen a ScanBlock onto a regular lat/lon grid.

    Each coarse cell aggregates all fine pixels whose centres fall within
    it, producing a lower-resolution ScanBlock suitable for driving
    coarser atmospheric models (e.g. 0.5° × 0.5° CTMs).

    Parameters
    ----------
    block : ScanBlock
        Input fine-resolution scan block.
    dlat_deg : float
        Target grid spacing in latitude [degrees].
    dlon_deg : float
        Target grid spacing in longitude [degrees].
    min_count : int
        Minimum number of fine pixels required for a valid coarse cell.
        Cells with fewer contributing pixels are filled with NaN.

    Returns
    -------
    ScanBlock
        Coarsened block on a regular (dlat_deg × dlon_deg) grid with shape
        (n_rows_c, n_cols_c).

        All standard ScanBlock fields are present.  Two extras are added:

        Data arrays
          ``pixel_counts``  (n_rows_c, n_cols_c) int32 — number of fine
          pixels that contributed to each coarse cell.

        Metadata
          ``dlat_deg``, ``dlon_deg``            — coarse resolution [°]
          ``coarsened_from_shape`` (2,) ndarray — original fine block shape

    Notes
    -----
    Grid alignment
        Edges are snapped to global multiples of dlat/dlon (e.g. 0°, 0.5°,
        1.0°, … for dlat=0.5), so adjacent blocks at the same resolution
        share a consistent grid.

    Angle aggregation
        Scalar angles (vzas, szas) use arithmetic mean.
        Azimuth angles (vaas, saas, relative_azimuth) use **circular mean**
        via atan2(Σ sin θ, Σ cos θ) to handle the 0°/360° wrap correctly.

    Unit vectors
        ``sat_look_vecs`` are averaged then re-normalised (mean direction).

    Position vectors
        ``gnd_ecef`` is averaged without normalisation (representative
        position of the cell).

    Footprint corners
        ``corner_lats/lons`` are derived directly from the coarse grid
        edges, not from averaging fine pixel corners.
    """
    lats_f = block['lats'].ravel()
    lons_f = block['lons'].ravel()

    # ---- aligned regular grid -----------------------------------------------
    lat_lo = np.floor(np.nanmin(lats_f) / dlat_deg) * dlat_deg
    lat_hi = np.ceil( np.nanmax(lats_f) / dlat_deg) * dlat_deg
    lon_lo = np.floor(np.nanmin(lons_f) / dlon_deg) * dlon_deg
    lon_hi = np.ceil( np.nanmax(lons_f) / dlon_deg) * dlon_deg

    lat_edges = np.arange(lat_lo, lat_hi + dlat_deg * 0.5, dlat_deg)
    lon_edges = np.arange(lon_lo, lon_hi + dlon_deg * 0.5, dlon_deg)
    n_rows_c  = len(lat_edges) - 1
    n_cols_c  = len(lon_edges) - 1
    n_cells   = n_rows_c * n_cols_c

    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # ---- assign each fine pixel to a coarse cell ----------------------------
    # np.digitize: returns i s.t. lat_edges[i-1] <= lat < lat_edges[i]
    lat_bin = np.digitize(lats_f, lat_edges) - 1   # 0-based row index
    lon_bin = np.digitize(lons_f, lon_edges) - 1   # 0-based col index
    in_grid = ((lat_bin >= 0) & (lat_bin < n_rows_c) &
               (lon_bin >= 0) & (lon_bin < n_cols_c))
    cell_idx = lat_bin * n_cols_c + lon_bin         # linear coarse cell index

    # ---- aggregation helpers -------------------------------------------------

    def _bin_scalar(arr_f: np.ndarray, circular: bool = False) -> np.ndarray:
        """Mean of a flat (N,) field into (n_rows_c, n_cols_c)."""
        ok   = in_grid & np.isfinite(arr_f)
        cidx = cell_idx[ok]
        cnts = np.bincount(cidx, minlength=n_cells)
        if circular:
            rad   = np.deg2rad(arr_f[ok])
            sin_s = np.bincount(cidx, weights=np.sin(rad), minlength=n_cells)
            cos_s = np.bincount(cidx, weights=np.cos(rad), minlength=n_cells)
            result = np.where(cnts >= min_count,
                              np.degrees(np.arctan2(sin_s, cos_s)) % 360.0,
                              np.nan)
        else:
            wsum   = np.bincount(cidx, weights=arr_f[ok], minlength=n_cells)
            result = np.where(cnts >= min_count,
                              wsum / np.maximum(cnts, 1), np.nan)
        return result.reshape(n_rows_c, n_cols_c)

    def _bin_vector(arr_f3: np.ndarray, normalize: bool) -> np.ndarray:
        """Mean of a flat (N, 3) field into (n_rows_c, n_cols_c, 3).

        normalize=True  → sum then re-normalise to unit length (use for
                           unit-vector fields such as sat_look_vecs).
        normalize=False → arithmetic mean (sum / count), used for position
                           vectors such as gnd_ecef.
        """
        ok   = in_grid & np.all(np.isfinite(arr_f3), axis=1)
        cidx = cell_idx[ok]
        cnts = np.bincount(cidx, minlength=n_cells)          # (n_cells,)
        out  = np.zeros((n_cells, 3))
        for k in range(3):
            out[:, k] = np.bincount(cidx, weights=arr_f3[ok, k],
                                    minlength=n_cells)
        if normalize:
            # Mean direction: normalise the vector sum
            nrm = np.linalg.norm(out, axis=1, keepdims=True)
            out /= np.where(nrm > 1e-10, nrm, 1.0)
        else:
            # Arithmetic mean: divide sum by count (guard against zero)
            cnt_safe = np.where(cnts > 0, cnts, 1)[:, np.newaxis]
            out /= cnt_safe
        out[cnts < min_count] = np.nan
        return out.reshape(n_rows_c, n_cols_c, 3)

    # ---- pixel counts -------------------------------------------------------
    counts_2d = np.bincount(cell_idx[in_grid],
                             minlength=n_cells).reshape(n_rows_c, n_cols_c
                             ).astype(np.int32)
    mask = counts_2d < min_count

    # ---- bin all fields -----------------------------------------------------
    vzas_c = _bin_scalar(block['vzas'].ravel())
    vaas_c = _bin_scalar(block['vaas'].ravel(),            circular=True)
    szas_c = _bin_scalar(block['szas'].ravel())
    saas_c = _bin_scalar(block['saas'].ravel(),            circular=True)
    am_c   = _bin_scalar(block['airmass_geometric'].ravel())
    # relative_azimuth lives in [0°, 180°] — arithmetic mean is correct here
    ra_c   = _bin_scalar(block['relative_azimuth'].ravel(), circular=False)

    slv_c  = _bin_vector(block['sat_look_vecs'].reshape(-1, 3), normalize=True)
    gnd_c  = _bin_vector(block['gnd_ecef'].reshape(-1, 3),      normalize=False)

    # ---- coarse footprint corners from grid edges ---------------------------
    # NW, NE, SE, SW  (axis 2 = corner index)
    lat_n = lat_edges[1:]    # north edge of each row: (n_rows_c,)
    lat_s = lat_edges[:-1]   # south edge
    lon_w = lon_edges[:-1]   # west  edge of each col: (n_cols_c,)
    lon_e = lon_edges[1:]    # east  edge

    corner_lats_c = np.stack([
        np.broadcast_to(lat_n[:, np.newaxis], (n_rows_c, n_cols_c)).copy(),
        np.broadcast_to(lat_n[:, np.newaxis], (n_rows_c, n_cols_c)).copy(),
        np.broadcast_to(lat_s[:, np.newaxis], (n_rows_c, n_cols_c)).copy(),
        np.broadcast_to(lat_s[:, np.newaxis], (n_rows_c, n_cols_c)).copy(),
    ], axis=2)
    corner_lons_c = np.stack([
        np.broadcast_to(lon_w[np.newaxis, :], (n_rows_c, n_cols_c)).copy(),
        np.broadcast_to(lon_e[np.newaxis, :], (n_rows_c, n_cols_c)).copy(),
        np.broadcast_to(lon_e[np.newaxis, :], (n_rows_c, n_cols_c)).copy(),
        np.broadcast_to(lon_w[np.newaxis, :], (n_rows_c, n_cols_c)).copy(),
    ], axis=2)
    corner_lats_c[mask] = np.nan
    corner_lons_c[mask] = np.nan

    # lats/lons — regular grid centers; NaN for empty cells
    lats_2d = np.where(mask, np.nan,
                        np.broadcast_to(lat_centers[:, np.newaxis],
                                        (n_rows_c, n_cols_c)).copy())
    lons_2d = np.where(mask, np.nan,
                        np.broadcast_to(lon_centers[np.newaxis, :],
                                        (n_rows_c, n_cols_c)).copy())

    # ---- assemble output ----------------------------------------------------
    data = {
        'lats':              lats_2d,
        'lons':              lons_2d,
        'vzas':              vzas_c,
        'vaas':              vaas_c,
        'szas':              szas_c,
        'saas':              saas_c,
        'airmass_geometric': am_c,
        'relative_azimuth':  ra_c,
        'sat_look_vecs':     slv_c,
        'gnd_ecef':          gnd_c,
        'corner_lats':       corner_lats_c,
        'corner_lons':       corner_lons_c,
        'pixel_counts':      counts_2d,
    }
    meta = {
        'sat_lon_deg':           block._meta['sat_lon_deg'],
        'sat_ecef':              block._meta['sat_ecef'],
        'slit_center_lat':       float(lat_centers[n_rows_c // 2]),
        'scan_start_lon':        float(lon_edges[0]),
        'scan_end_lon':          float(lon_edges[-1]),
        'scan_lons':             lon_centers,
        'integration_time_s':    block._meta['integration_time_s'],
        'scan_duration_s':       block._meta['scan_duration_s'],
        't_start_utc':           block._meta['t_start_utc'],
        't_end_utc':             block._meta['t_end_utc'],
        'col_times':             None,
        'n_rows':                n_rows_c,
        'n_cols':                n_cols_c,
        'dlat_deg':              float(dlat_deg),
        'dlon_deg':              float(dlon_deg),
        'coarsened_from_shape':  np.array(block.shape),
    }
    return ScanBlock(data, meta)


# ---------------------------------------------------------------------------
# Day scheduling utility
# ---------------------------------------------------------------------------

def build_day_schedule(
        satellite: 'LongSlitGeoSatellite',
        t_start:   datetime,
        targets:   list,
        t_end:     Optional[datetime] = None,
        repeat:    bool               = True,
) -> List['ScanBlock']:
    """
    Build a chronological sequence of ScanBlocks for a full day of scanning.

    Scan blocks are chained in time: the next block starts one integration
    step after the last column of the previous block (i.e. the total clock
    advance per block is ``n_cols × integration_time_s``).

    Parameters
    ----------
    satellite : LongSlitGeoSatellite
    t_start : datetime
        UTC time of the first scan column of the first block.
    targets : list of ScanTarget, tuple, or dict
        Ordered list of scan regions.  Each entry may be:

        * a ``ScanTarget`` instance,
        * a tuple ``(slit_center_lat, scan_start_lon, scan_end_lon)``
          with optional extra fields matching ScanTarget,
        * a dict whose keys match ScanTarget field names.
    t_end : datetime or None
        Stop producing blocks once the *next* scan would start at or after
        this time.  Defaults to ``t_start + 24 hours``.
    repeat : bool
        If True (default) cycle through *targets* repeatedly until *t_end*.
        If False execute each target once then stop.

    Returns
    -------
    list of ScanBlock
        Chronologically ordered blocks.  Each block's ``t_start_utc`` and
        ``t_end_utc`` metadata are correct.  If a ``ScanTarget`` carries a
        ``label``, it is stored in the block's ``_meta`` dict under 'label'.

    Example
    -------
    >>> sat = LongSlitGeoSatellite(sat_lon_deg=-95.0, slit_length_km=3000.0,
    ...                             pixel_size_ew_km=6.0, pixel_size_ns_km=6.0,
    ...                             integration_time_s=10.0)
    >>> targets = [
    ...     ScanTarget(30.0, -120.0, -95.0, label='West'),
    ...     ScanTarget(30.0,  -95.0, -75.0, label='Central'),
    ...     ScanTarget(30.0,  -75.0, -60.0, label='East'),
    ... ]
    >>> day = build_day_schedule(sat, datetime(2020, 7, 1, 12, 0), targets)
    >>> print(f"{len(day)} blocks, "
    ...       f"ending {day[-1].t_end_utc.strftime('%H:%M UTC')}")
    """
    if t_end is None:
        t_end = t_start + timedelta(hours=24)

    # Normalise targets to ScanTarget instances
    norm: List[ScanTarget] = []
    for t in targets:
        if isinstance(t, ScanTarget):
            norm.append(t)
        elif isinstance(t, (tuple, list)):
            norm.append(ScanTarget(*t))
        elif isinstance(t, dict):
            norm.append(ScanTarget(**t))
        else:
            raise TypeError(
                f"targets must be ScanTarget, tuple, list, or dict; got {type(t)}")

    if not norm:
        return []

    blocks: List[ScanBlock] = []
    t_current = t_start

    while True:
        made_progress = False
        for tgt in norm:
            if t_current >= t_end:
                return blocks
            blk = satellite.build_scan_block(
                tgt.slit_center_lat,
                tgt.scan_start_lon,
                tgt.scan_end_lon,
                t0_utc=t_current,
                n_cols=tgt.n_cols,
            )
            if tgt.label is not None:
                blk._meta['label'] = tgt.label
            blocks.append(blk)
            # Advance clock: total scan takes n_cols integrations
            t_current += timedelta(
                seconds=blk.scan_duration_s + satellite.integration_time_s)
            made_progress = True

        if not repeat or not made_progress:
            break

    return blocks


# ---------------------------------------------------------------------------
# Land / water fraction
# ---------------------------------------------------------------------------

#: Module-level cache: cache_key -> (integral_image, lat_edges, lon_edges)
_LAND_MASK_CACHE: dict = {}


def _build_land_mask_for_extent(
        lat0: float, lat1: float, lon0: float, lon1: float,
        resolution_deg: float, scale: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterise Natural Earth land polygons onto a regular grid.

    The grid covers [lat0-pad .. lat1+pad] × [lon0-pad .. lon1+pad] where
    pad = resolution_deg, to ensure pixel-corner queries don't fall outside.

    Returns
    -------
    mask      : (n_lat, n_lon) float32  — 1.0 land, 0.0 water
    lat_edges : (n_lat+1,) bin edges [°]
    lon_edges : (n_lon+1,) bin edges [°]
    """
    try:
        import cartopy.io.shapereader as shpreader
        import matplotlib.path as mpath
    except ImportError as exc:
        raise ImportError(
            "add_land_fraction requires cartopy and matplotlib. "
            "Install cartopy with:  conda install -c conda-forge cartopy"
        ) from exc

    pad = resolution_deg
    lat_edges = np.arange(lat0 - pad, lat1 + pad + resolution_deg * 0.5,
                           resolution_deg)
    lon_edges = np.arange(lon0 - pad, lon1 + pad + resolution_deg * 0.5,
                           resolution_deg)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    n_lat, n_lon = len(lat_c), len(lon_c)

    if n_lat == 0 or n_lon == 0:
        return (np.zeros((n_lat, n_lon), dtype=np.float32),
                lat_edges, lon_edges)

    LON, LAT = np.meshgrid(lon_c, lat_c)
    pts     = np.column_stack([LON.ravel(), LAT.ravel()])
    in_land = np.zeros(len(pts), dtype=bool)

    reader = shpreader.natural_earth(resolution=scale,
                                      category='physical', name='land')
    for geom in reader.geometries():
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            bx0, by0, bx1, by1 = poly.bounds
            # Quick AABB reject: skip polygons that don't overlap the extent
            if bx1 < lon0 - pad or bx0 > lon1 + pad:
                continue
            if by1 < lat0 - pad or by0 > lat1 + pad:
                continue
            coords = np.array(poly.exterior.coords)
            path   = mpath.Path(coords)
            in_land |= path.contains_points(pts)

    mask = in_land.reshape(n_lat, n_lon).astype(np.float32)
    return mask, lat_edges, lon_edges


def add_land_fraction(
        block:               'ScanBlock',
        mask_resolution_deg: float = 0.1,
        natural_earth_scale: str   = '110m',
) -> 'ScanBlock':
    """
    Add a per-pixel ``land_fraction`` field to a ScanBlock.

    Each pixel's ``land_fraction`` is the fraction of land-mask cells within
    its lat/lon bounding box (derived from ``corner_lats`` / ``corner_lons``).
    Values range from 0.0 (entirely water) to 1.0 (entirely land); pixels
    that straddle a coastline get intermediate values.

    The land mask is rasterised from Natural Earth polygons on the first call
    for a given region and resolution, then cached so that subsequent calls
    over the same extent are instantaneous (2D summed-area-table lookup).

    Parameters
    ----------
    block : ScanBlock
        Input scan block.
    mask_resolution_deg : float
        Resolution of the rasterised land mask [°].  Default 0.1°, which
        gives ~11 km cells — appropriate for ~6 km GeoCarb pixels.
        Use 0.05° for finer sub-pixel fractions (slower first build).
    natural_earth_scale : str
        Natural Earth dataset scale: ``'110m'`` (default, fast),
        ``'50m'``, or ``'10m'``.  Higher resolution improves coastline
        accuracy at the cost of a longer first-call build.

    Returns
    -------
    ScanBlock
        A new ScanBlock with ``'land_fraction'`` (n_rows, n_cols) float32
        added.  The original block is not modified.

    Requires
    --------
    cartopy (≥ 0.18) and matplotlib.

    Notes
    -----
    The cache key includes the block extent and resolution, so blocks from
    different scan regions each get their own mask slice.  Clear the cache
    between sessions with ``geosat_geometry._LAND_MASK_CACHE.clear()``.
    """
    clat = block['corner_lats'].reshape(-1, 4)   # (N, 4) NW NE SE SW
    clon = block['corner_lons'].reshape(-1, 4)

    lat_lo = clat.min(axis=1)
    lat_hi = clat.max(axis=1)
    lon_lo = clon.min(axis=1)
    lon_hi = clon.max(axis=1)

    extent_lat0 = float(np.nanmin(lat_lo))
    extent_lat1 = float(np.nanmax(lat_hi))
    extent_lon0 = float(np.nanmin(lon_lo))
    extent_lon1 = float(np.nanmax(lon_hi))

    cache_key = (round(extent_lat0, 4), round(extent_lat1, 4),
                 round(extent_lon0, 4), round(extent_lon1, 4),
                 mask_resolution_deg, natural_earth_scale)

    if cache_key not in _LAND_MASK_CACHE:
        mask, lat_edges, lon_edges = _build_land_mask_for_extent(
            extent_lat0, extent_lat1, extent_lon0, extent_lon1,
            mask_resolution_deg, natural_earth_scale)
        # Build 2-D summed area table (integral image) for O(1) box queries.
        # Pad with a zero row/column so that index 0 means "zero area counted".
        integral = np.cumsum(np.cumsum(mask.astype(np.float64), axis=0), axis=1)
        ii = np.pad(integral, ((1, 0), (1, 0)), mode='constant')
        _LAND_MASK_CACHE[cache_key] = (ii, lat_edges, lon_edges)

    ii, lat_edges, lon_edges = _LAND_MASK_CACHE[cache_key]

    # For each pixel query the summed area table over [lat_lo..lat_hi] x [lon_lo..lon_hi].
    # searchsorted returns indices into lat_edges / lon_edges (bin-edge arrays).
    # The padded integral image means row/col 0 contributes nothing.
    r0 = np.searchsorted(lat_edges, lat_lo, side='left')
    r1 = np.searchsorted(lat_edges, lat_hi, side='right')
    c0 = np.searchsorted(lon_edges, lon_lo, side='left')
    c1 = np.searchsorted(lon_edges, lon_hi, side='right')

    max_r = ii.shape[0] - 1
    max_c = ii.shape[1] - 1
    r0 = np.clip(r0, 0, max_r)
    r1 = np.clip(r1, 0, max_r)
    c0 = np.clip(c0, 0, max_c)
    c1 = np.clip(c1, 0, max_c)

    sums   = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
    counts = (r1 - r0) * (c1 - c0)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)

    new_data = dict(block._data)
    new_data['land_fraction'] = frac.reshape(block.shape)
    return ScanBlock(new_data, dict(block._meta))


# ---------------------------------------------------------------------------
# Demo / quick-start
# ---------------------------------------------------------------------------

def geocarb_demo(verbose: bool = True) -> dict:
    """
    GeoCarb-like demo: build scan blocks, demonstrate save/load, coarsening,
    and plotting.

    Returns dict with 'sat', 'blocks', 'coarse_blocks', 'rays'.
    """
    import time

    import matplotlib.pyplot as plt

    sat = LongSlitGeoSatellite(
        sat_lon_deg      = -95.0,
        slit_length_km   = 3000.0,
        pixel_size_ew_km =    6.0,
        pixel_size_ns_km =    6.0,
        integration_time_s = 10.0,
    )
    if verbose:
        print(sat)

    dt0 = datetime(2020, 7, 1, 17, 30, 0)   # start time

    # ---- build three adjacent scan blocks covering the continental US ----
    block_defs = [
        ('West',   30.0, -120.0,  -95.0),
        ('Central',30.0,  -95.0,  -75.0),
        ('East',   30.0,  -75.0,  -60.0),
    ]
    blocks = []
    t_offset = 0.0
    for name, lat, lon0, lon1 in block_defs:
        t_start = dt0 + timedelta(seconds=t_offset)
        t0 = time.perf_counter()
        blk = sat.build_scan_block(lat, lon0, lon1, t0_utc=t_start)
        t1 = time.perf_counter()
        blocks.append(blk)
        if verbose:
            print(f"\n{name}: {blk}")
            print(f"  build time : {(t1-t0)*1e3:.0f} ms")
            print(f"  VZA  range : {blk['vzas'].min():.1f}° – {blk['vzas'].max():.1f}°")
            print(f"  SZA  range : {blk['szas'].min():.1f}° – {blk['szas'].max():.1f}°")
            print(f"  Airmass    : {blk['airmass_geometric'].min():.2f} – "
                  f"{blk['airmass_geometric'].max():.2f}")
        # Advance time offset so next block starts after this one finishes
        t_offset += blk.scan_duration_s

    # ---- save / load round-trip ----
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmp_path = f.name
    blocks[0].save(tmp_path)
    reloaded = ScanBlock.load(tmp_path)
    os.unlink(tmp_path)
    if verbose:
        assert np.allclose(reloaded['vzas'], blocks[0]['vzas'])
        print(f"\nSave/load round-trip ✓  (vzas max diff = "
              f"{np.abs(reloaded['vzas'] - blocks[0]['vzas']).max():.2e})")

    # ---- ray paths through the atmosphere ----
    alts  = np.arange(0.0, 80.0, 1.0)
    blk0  = blocks[0]
    lats_f = blk0['lats'].ravel()
    lons_f = blk0['lons'].ravel()
    t0 = time.perf_counter()
    rays = sat.compute_ray_paths_vectorized(lats_f, lons_f, alts)
    t1 = time.perf_counter()
    if verbose:
        N, M = len(lats_f), len(alts)
        print(f"\nRay paths: {N:,} pixels × {M} shells "
              f"in {(t1-t0)*1e3:.0f} ms")

    # ---- coarsen all three blocks to 0.5° × 0.5° ----
    t0 = time.perf_counter()
    coarse_blocks = [blk.coarsen(0.5, 0.5) for blk in blocks]
    t1 = time.perf_counter()
    if verbose:
        print(f"\nCoarsened to 0.5° × 0.5°  ({(t1-t0)*1e3:.0f} ms total):")
        for name, cblk in zip(['West', 'Central', 'East'], coarse_blocks):
            total_fine = cblk['pixel_counts'].sum()
            nonempty   = int((cblk['pixel_counts'] > 0).sum())
            print(f"  {name}: {cblk}  |  "
                  f"{nonempty} filled cells, "
                  f"{total_fine:,} fine pixels binned  |  "
                  f"VZA {cblk['vzas'][np.isfinite(cblk['vzas'])].min():.1f}°"
                  f"–{cblk['vzas'][np.isfinite(cblk['vzas'])].max():.1f}°")

        # Validate: coarse VZA mean should be close to fine VZA mean
        for name, blk, cblk in zip(['West', 'Central', 'East'],
                                    blocks, coarse_blocks):
            fine_mean   = float(np.nanmean(blk['vzas']))
            coarse_mean = float(np.nanmean(cblk['vzas']))
            print(f"  {name} VZA mean  fine={fine_mean:.3f}°  "
                  f"coarse={coarse_mean:.3f}°  "
                  f"Δ={abs(fine_mean - coarse_mean):.4f}°")

        # Validate: sat_look_vecs should remain unit vectors after averaging
        slv = coarse_blocks[0]['sat_look_vecs']
        finite_mask = np.all(np.isfinite(slv), axis=-1)
        norms = np.linalg.norm(slv[finite_mask], axis=-1)
        print(f"\n  sat_look_vecs norm range (coarsened): "
              f"{norms.min():.6f} – {norms.max():.6f}  (should be ≈1.0)")

        # Validate: gnd_ecef altitudes should be near 0 km
        ge = coarse_blocks[0]['gnd_ecef']
        finite_mask_g = np.all(np.isfinite(ge), axis=-1)
        flat_lat, flat_lon, flat_alt = ecef_to_geodetic(ge[finite_mask_g])
        print(f"  gnd_ecef altitude range after coarsening: "
              f"{flat_alt.min():.3f} – {flat_alt.max():.3f} km  (should be ≈0)")

    # ---- plot: fine footprints (left) vs coarsened outline (right) ----
    if verbose:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plot_scan_blocks(blocks, field='vzas', mode='footprints',
                         ax=axes[0], coastlines=False,
                         block_labels=['West', 'Central', 'East'],
                         title='Fine resolution — VZA')
        plot_scan_blocks(coarse_blocks, field='vzas', mode='footprints',
                         ax=axes[1], coastlines=False,
                         block_labels=['West', 'Central', 'East'],
                         title='Coarsened 0.5°×0.5° — VZA')
        plt.tight_layout()
        plt.savefig('scan_blocks_demo.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved → scan_blocks_demo.png")
        plt.show()

    return dict(sat=sat, blocks=blocks, coarse_blocks=coarse_blocks,
                rays=rays, alts=alts)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = geocarb_demo(verbose=True)
