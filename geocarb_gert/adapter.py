"""``ScanBlock`` → GERT: per-pixel geometry and scenes.

`geosat_geometry.ScanBlock` already carries everything GERT needs per pixel:
solar/viewing zenith angles and the relative azimuth.  This module is the thin
translation layer; it performs no radiative transfer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from gert.geometry import Geometry
from gert.osse import Scene, SurfaceSpec


def gsd_km(sat) -> float:
    """Nominal ground sample distance [km] of a `LongSlitGeoSatellite`.

    The mean of the E-W and N-S pixel sizes; they are equal in the default
    GeoCarb configuration (6 km).
    """
    return 0.5 * (float(sat.pixel_size_ew_km) + float(sat.pixel_size_ns_km))


def pixel_geometry(block, row: int, col: int) -> Geometry:
    """`gert.Geometry` for one ScanBlock pixel.

    ``relative_azimuth`` from the ScanBlock is used directly as GERT's ``raa``.
    """
    return Geometry(sza=float(block["szas"][row, col]),
                    vza=float(block["vzas"][row, col]),
                    raa=float(block["relative_azimuth"][row, col]))


def sample_geometries(block, n: int = 10, seed: int = 0,
                      max_sza: float = 70.0) -> list:
    """Sample ``n`` valid, well-illuminated pixels → ``[(row, col, Geometry)]``.

    Pixels failing ``valid_mask`` or exceeding ``max_sza`` are excluded: at very
    high solar zenith the plane-parallel assumption and the photon budget both
    degrade, and such pixels are not representative design points.
    """
    valid = np.asarray(block["valid_mask"], bool) & (np.asarray(block["szas"]) <= max_sza)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        raise ValueError("no valid pixels with sza <= %.1f deg" % max_sza)
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows.size, size=min(n, rows.size), replace=False)
    return [(int(rows[k]), int(cols[k]), pixel_geometry(block, rows[k], cols[k]))
            for k in pick]


def scene_from_profile(atm, albedo, kind: str = "lambertian",
                       layers: Optional[list] = None) -> Scene:
    """Wrap an `AtmosphericProfile` + surface albedo into a `gert.osse.Scene`.

    ``albedo`` is per-band (length = number of instrument windows).  Model-driven
    scenes come from ``model_sampler.sample_field_along_rays`` — build the
    `AtmosphericProfile` from those samples and pass it here.
    """
    return Scene(atmosphere=atm,
                 surface=SurfaceSpec(kind=kind, albedo=list(np.atleast_1d(albedo))),
                 layers=list(layers or []))