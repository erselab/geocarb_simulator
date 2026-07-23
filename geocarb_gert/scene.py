"""Scenes for GeoCarb simulations — the mission supplies its own atmosphere.

GERT is a library: it takes an `AtmosphericProfile` as an argument and never
owns one.  This module builds them, either from a standard atmosphere (for
design studies and smoke tests) or from model fields sampled along the ScanBlock
ray paths (for real OSSEs).

Deliberately does **not** import gert's ``settings.py`` — that is a script
convenience of the gert repo (it reads a cwd-relative ``site_settings.yml``),
not part of the public API.
"""
from __future__ import annotations

import numpy as np

from gert.atmosphere import AtmosphericProfile
from gert.levels import gert_levels

from model_sampler import pressure_to_alt_std_atm

_MW_RATIO = 0.018015 / 0.028964      # M_H2O / M_dry ≈ 0.622

# Well-mixed dry-air mole fractions.  ``o2`` is needed by the O2 A-band window,
# ``co``/``n2o`` by the 2.32 µm CH4/CO window.
_WELL_MIXED = {"co2": 415e-6, "ch4": 1.90e-6, "n2o": 330e-9,
               "co": 100e-9, "o2": 0.2095}

# ── Surface scenes ──────────────────────────────────────────────────────────
# Representative Lambertian-equivalent surface albedo per GeoCarb band, keyed by
# the ``SpectralWindow.label`` used in :mod:`geocarb_gert.instrument`.  The four
# bands sit at roughly [0.76, 1.61, 2.06, 2.32] µm.
#
# These drive the band-to-band *signal* differences that a scalar SNR cannot
# express: vegetation is bright in the NIR (just past the red edge) but darkens
# sharply through the SWIR as leaf water absorbs; water is near-black beyond
# 1 µm; desert is the bright, slowly-varying case.  Values are typical clear-sky
# nadir reflectances, not a specific measured spectrum.
_BAND_ALBEDO = {
    #  band          desert  forest  grass   water
    "O2_A":       {"desert": 0.30, "forest": 0.45, "grass": 0.42, "water": 0.04},
    "CO2_weak":   {"desert": 0.45, "forest": 0.25, "grass": 0.30, "water": 0.02},
    "CO2_strong": {"desert": 0.45, "forest": 0.12, "grass": 0.18, "water": 0.015},
    "CH4_CO":     {"desert": 0.40, "forest": 0.08, "grass": 0.14, "water": 0.015},
}

SCENE_TYPES = ("desert", "forest", "grass", "water")


def albedo_for(instrument, surface: str) -> np.ndarray:
    """Per-band Lambertian albedo vector for a surface type, in window order.

    Looks up each ``instrument.windows[i].label`` in :data:`_BAND_ALBEDO`, so it
    stays correct when the instrument is built from a subset of bands (e.g. a
    1.6 µm-only design variant).  Pass the result as ``albedo`` to
    :func:`geocarb_gert.adapter.scene_from_profile`.
    """
    surface = str(surface)
    if surface not in SCENE_TYPES:
        raise ValueError(f"unknown surface {surface!r}; choose from {SCENE_TYPES}")
    try:
        return np.array([_BAND_ALBEDO[w.label][surface] for w in instrument.windows],
                        dtype=float)
    except KeyError as e:
        raise KeyError(f"no albedo defined for band {e.args[0]!r}; "
                       f"add it to geocarb_gert.scene._BAND_ALBEDO") from None


def hires_spectra_for(fm, surface_types=SCENE_TYPES) -> dict:
    """Hi-res radiance spectra for each surface type, all bands at once.

    Runs ``fm`` once per surface type at that surface's :func:`albedo_for`,
    for use as the base spectra in :func:`geocarb_gert.focalplane.
    random_scene` (a patchwork of real surface types along the slit) — e.g.
    ``geocarb_gert.focalplane.random_scene([res[s].I_hires[b] for s in
    SCENE_TYPES], ...)`` for band ``b``.

    Parameters
    ----------
    fm : gert.ForwardModel
        Already constructed (atmosphere, ABSCO, instrument, geometry, solver).
    surface_types : sequence of str
        Defaults to all of :data:`SCENE_TYPES`.

    Returns
    -------
    dict[str, gert.ForwardResult]
        Keyed by surface type. Each result's ``.wn_band_hires[b]`` /
        ``.I_hires[b]`` give band ``b``'s hi-res wavenumber grid and
        radiance for that surface (same ``wn_band_hires[b]`` across surface
        types, since it depends only on the instrument/geometry).
    """
    nb = len(fm.instrument.windows)
    return {s: fm.run(albedo=list(albedo_for(fm.instrument, s)), albedo_slope=[0.0] * nb)
           for s in surface_types}


def _std_temperature(z_km: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere 1976 temperature [K] (troposphere → mesosphere)."""
    z = np.asarray(z_km, dtype=float)
    T = np.empty_like(z)
    T = np.where(z < 11.0, 288.15 - 6.5 * z, T)
    T = np.where((z >= 11.0) & (z < 20.0), 216.65, T)
    T = np.where((z >= 20.0) & (z < 32.0), 216.65 + 1.0 * (z - 20.0), T)
    T = np.where((z >= 32.0) & (z < 47.0), 228.65 + 2.8 * (z - 32.0), T)
    T = np.where(z >= 47.0, 270.65, T)
    return T


def reference_atmosphere(p_surface_pa: float = 101325.0,
                         h2o_surface_vmr: float = 1.0e-2,
                         h2o_scale_height_km: float = 2.0,
                         xco2_ppm: float | None = None) -> AtmosphericProfile:
    """A US-Standard-Atmosphere `AtmosphericProfile` on the GERT level grid.

    Water vapour decays exponentially with altitude; other gases are well mixed.
    ``xco2_ppm`` overrides the default CO₂ mole fraction (useful for OSSE truth
    vs prior).
    """
    p = np.asarray(gert_levels(float(p_surface_pa)), dtype=float)   # TOA → surface
    z_km = np.asarray(pressure_to_alt_std_atm(p / 100.0), dtype=float)
    T = _std_temperature(z_km)

    h2o = h2o_surface_vmr * np.exp(-z_km / float(h2o_scale_height_km))
    gases = {g: np.full_like(p, v) for g, v in _WELL_MIXED.items()}
    if xco2_ppm is not None:
        gases["co2"] = np.full_like(p, float(xco2_ppm) * 1e-6)
    gases["h2o"] = h2o

    w = _MW_RATIO * h2o                # mass mixing ratio [kg/kg dry]
    q = w / (1.0 + w)                  # specific humidity

    return AtmosphericProfile(p_levels=p, T_levels=T, q_levels=q, gases=gases)