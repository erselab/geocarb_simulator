"""geocarb_gert — adapter wiring the GeoCarb scan simulator to the GERT library.

This package is deliberately thin.  It contains **no radiative transfer and no
retrieval code**: `gert` supplies those.  What lives here is everything that is
specific to *how GeoCarb flies and what it is*:

* :mod:`geocarb_gert.instrument` — the four GeoCarb spectral bands.
* :mod:`geocarb_gert.radiometry` — the mission-side GSD/dwell → noise mapping.
  GeoCarb **stares** from GEO, so integration time is a free parameter,
  decoupled from GSD (unlike a LEO pushbroom where ``t_int = GSD/v_ground``).
* :mod:`geocarb_gert.adapter`   — ``ScanBlock`` → per-pixel `gert.Geometry`
  and `gert.osse.Scene`.

See ``docs/STATUS_AND_ROADMAP.md`` §3.7 in the gert repo for the layering rule.
"""
__version__ = "0.0.1"

from .instrument import GEOCARB_BANDS, build_geocarb_instrument
from .radiometry import (GEOCARB_REF, base_noise_model, model_for,
                         staring_sweep_models, etendue_factor_for_gsd)
from .adapter import pixel_geometry, sample_geometries, gsd_km, scene_from_profile
from .scene import reference_atmosphere, albedo_for, SCENE_TYPES

__all__ = [
    "GEOCARB_BANDS", "build_geocarb_instrument",
    "GEOCARB_REF", "base_noise_model", "model_for", "staring_sweep_models",
    "etendue_factor_for_gsd",
    "pixel_geometry", "sample_geometries", "gsd_km", "scene_from_profile",
    "reference_atmosphere", "albedo_for", "SCENE_TYPES",
]
