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
from .paths import gert_root, describe_gert_root  # noqa: F401
from .radiometry import (GEOCARB_REF, base_noise_model, model_for,
                         staring_sweep_models, etendue_factor_for_gsd,
                         RADIOMETRIC_SPEC_BY_FPA, linear_shot_noise_params,
                         geocarb_noise_model, geocarb_noise_model_multi)
from .adapter import pixel_geometry, sample_geometries, gsd_km, scene_from_profile
from .scene import reference_atmosphere, albedo_for, SCENE_TYPES, hires_spectra_for
from .focalplane import (FocalPlaneModel, uniform_scene, edge_scene,
                         random_scene, barcode_scene, nearest_bin_scene)
from .gd_polynomials import (xy_to_wavelength_slit, wavelength_slit_to_xy, rows_crossed,
                             DISPERSION_ASCENDING)
from . import gd_render
from . import along_slit_scene
from . import cross_band
from .robust_stats import median_mad, mad_outlier_mask, chi2_outlier_mask, robust_mean_std

__all__ = [
    "GEOCARB_BANDS", "build_geocarb_instrument",
    "GEOCARB_REF", "base_noise_model", "model_for", "staring_sweep_models",
    "etendue_factor_for_gsd",
    "RADIOMETRIC_SPEC_BY_FPA", "linear_shot_noise_params",
    "geocarb_noise_model", "geocarb_noise_model_multi",
    "pixel_geometry", "sample_geometries", "gsd_km", "scene_from_profile",
    "reference_atmosphere", "albedo_for", "SCENE_TYPES", "hires_spectra_for",
    "FocalPlaneModel", "uniform_scene", "edge_scene", "random_scene", "barcode_scene",
    "gert_root",
    "describe_gert_root",
    "nearest_bin_scene",
    "xy_to_wavelength_slit", "wavelength_slit_to_xy", "rows_crossed", "gd_render",
    "along_slit_scene", "DISPERSION_ASCENDING",
    "median_mad", "mad_outlier_mask", "chi2_outlier_mask", "robust_mean_std",
]
