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