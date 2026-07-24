"""Along-slit-varying truth atmosphere for the composition/pressure stress test.

Every scene built elsewhere in this package (``focalplane.py``, ``scene.py``)
holds the *atmosphere* fixed along the slit and only varies surface albedo.
This module does the opposite: it builds a truth atmosphere whose gas
columns (CO2, CH4, CO, H2O) and surface pressure vary continuously along
the ~2800 km slit, for testing whether keystone/smile row-crossing confuses
a retrieval when the along-slit variability is geophysical, not just
radiometric.

Design (see ``scripts/gd_along_slit_atm_profiles.py`` for the plots this was
approved from): baselines from :func:`geocarb_gert.scene.reference_atmosphere`
/ ``_WELL_MIXED`` (CO2 415 ppm, CH4 1900 ppb, CO 100 ppb, H2O surface 1% VMR,
p_surface 1013.25 hPa). Each quantity gets a broad smooth background
gradient; CO2 and CO share a broad co-located plume (a wildfire-in-mountains
scenario, combustion co-emission) plus a topographic pressure depression
nearby; CH4 gets its own broad plume at a different location (an
independent wetland/oil-and-gas-type source); H2O gets a smooth
arid-to-humid climatological gradient across the whole transect, no plume.
Small (~10 km wide, a few detector rows) "hot spot" point sources are
layered on top -- CO2 and CO hot spots are coincident (combustion
co-emission); CH4's are independent.

Coordinate convention: physical along-slit distance is linear in eta
(eta=-1..+1 -> -1400..+1400 km, ~2800 km total, matching this repo's
LongSlitGeoSatellite's slit_length_km=3000 default closely enough for a
stress-test truth scene).
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import numpy as np

from gert.atmosphere import AtmosphericProfile
from gert.levels import gert_levels

from model_sampler import pressure_to_alt_std_atm

from .scene import _MW_RATIO, _std_temperature, _WELL_MIXED
from .gd_render import available_cpus

SLIT_HALF_KM = 1400.0        # -> 2800 km total slit length

# -- baselines (geocarb_gert.scene.reference_atmosphere / _WELL_MIXED) --
XCO2_BG_PPM = 415.0
XCH4_BG_PPB = 1900.0
XCO_BG_PPB = 100.0
H2O_BG_VMR = 1.0e-2      # surface VMR, fraction
P_BG_HPA = 1013.25

# Small, localized "hot spots" -- point-source-scale features, deliberately
# narrower than the along-slit ground sample distance (~2.7 km/row for the
# ~2800 km slit over 1024 rows) times a handful of rows, i.e. only a few
# rows wide -- to stress-test whether keystone/smile row-crossing preserves
# or smears a genuinely localized source, as opposed to the broad regional
# plumes/gradients below. Placed away from the broad features so each shows
# up in isolation.
HOTSPOTS_CO2 = [(-1100.0, 10.0, 5.0), (1050.0, 9.0, 4.0)]     # (x0_km, width_km, amp_ppm)
HOTSPOTS_CH4 = [(-50.0, 12.0, 35.0), (1050.0, 9.0, 25.0)]     # (x0_km, width_km, amp_ppb) -- second co-located with the CO2 one at +1050 km (a small combined facility)
HOTSPOTS_CO = [(-1100.0, 10.0, 35.0), (1050.0, 9.0, 28.0)]    # (x0_km, width_km, amp_ppb) -- coincident with both CO2 hot spots (combustion co-emission)


def _gauss(x, x0, width, amp):
    return amp * np.exp(-0.5 * ((x - x0) / width) ** 2)


def xco2_ppm(x_km):
    background = XCO2_BG_PPM + 2.0 * np.sin(2 * np.pi * (x_km + 1400) / 3200)
    plume = _gauss(x_km, x0=-500.0, width=60.0, amp=6.0)
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CO2)
    return background + plume + hotspots


def xco_ppb(x_km):
    background = XCO_BG_PPB + 10.0 * np.sin(2 * np.pi * (x_km + 1400) / 2600 + 1.0)
    plume = _gauss(x_km, x0=-500.0, width=55.0, amp=160.0)   # co-located with the CO2 plume
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CO)
    return background + plume + hotspots


def xch4_ppb(x_km):
    background = XCH4_BG_PPB + 15.0 * np.sin(2 * np.pi * (x_km + 1400) / 2200 + 2.5)
    plume = _gauss(x_km, x0=650.0, width=100.0, amp=45.0)   # separate location from CO2/CO
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CH4)
    return background + plume + hotspots


def h2o_surface_vmr(x_km):
    # smooth arid -> humid climatological gradient across the whole transect
    t = 0.5 * (1.0 + np.tanh(x_km / 500.0))
    return 0.005 + t * (0.018 - 0.005)


def p_surface_hpa(x_km):
    # `pressure_to_alt_std_atm` (model_sampler.py) only extrapolates for
    # p <= standard sea level (1013.25 hPa) -- it returns NaN above that
    # (found 2026-07-24: a symmetric +/-3 hPa sinusoid here briefly exceeded
    # 1013.25 and silently corrupted T_levels/h2o downstream). Kept
    # asymmetric (oscillates *below* P_BG_HPA only) so the background term
    # can never push p_surface above standard sea level, regardless of x_km.
    background = (P_BG_HPA - 0.1) - 3.0 * (1.0 + np.sin(2 * np.pi * (x_km + 1400) / 2800 + 0.5))
    mountain = _gauss(x_km, x0=-250.0, width=140.0, amp=-250.0)   # topographic depression
    return background + mountain


def atmosphere_at(x_km: float, h2o_scale_height_km: float = 2.0) -> AtmosphericProfile:
    """The true ``AtmosphericProfile`` at one along-slit position [km].

    Generalizes ``geocarb_gert.scene.reference_atmosphere`` to also vary
    CH4, CO, and surface pressure (that function only overrides CO2/H2O).
    """
    p_surface_pa = float(p_surface_hpa(x_km)) * 100.0
    p = np.asarray(gert_levels(p_surface_pa), dtype=float)      # TOA -> surface
    z_km = np.asarray(pressure_to_alt_std_atm(p / 100.0), dtype=float)
    T = _std_temperature(z_km)

    h2o = float(h2o_surface_vmr(x_km)) * np.exp(-z_km / float(h2o_scale_height_km))
    gases = {g: np.full_like(p, v) for g, v in _WELL_MIXED.items()}
    gases["co2"] = np.full_like(p, float(xco2_ppm(x_km)) * 1e-6)
    gases["ch4"] = np.full_like(p, float(xch4_ppb(x_km)) * 1e-9)
    gases["co"] = np.full_like(p, float(xco_ppb(x_km)) * 1e-9)
    gases["h2o"] = h2o

    w = _MW_RATIO * h2o
    q = w / (1.0 + w)
    return AtmosphericProfile(p_levels=p, T_levels=T, q_levels=q, gases=gases)


# -- globals populated in build_lookup_radiance() before the Pool is forked --
_G_LOOKUP = {}


def _lookup_sample(i: int):
    g = _G_LOOKUP
    x_km = g["x_samples_km"][i]
    atm = atmosphere_at(x_km, g["h2o_scale_height_km"])
    fm = g["fm_cls"](atm, g["absco"], g["inst"], g["geo"],
                     solver=g["solver_cls"](), solar_spectrum=g["solar"])
    res = fm.run(albedo=g["albedo"], albedo_slope=[0.0])
    return i, res.I_hires[0]


def build_lookup_radiance(
    absco, inst, geo, solar, albedo,
    n_samples: int = 400,
    h2o_scale_height_km: float = 2.0,
    n_workers: int | None = None,
) -> tuple[np.ndarray, Callable]:
    """Precompute hi-res spectra at ``n_samples`` along-slit positions and
    return ``(wn_hires, radiance)`` where ``radiance(eta) -> spectrum`` does
    linear interpolation between the two nearest precomputed samples.

    The along-slit hot spots are ~10 km wide; with the default
    ``n_samples=400`` over the 2800 km slit, sample spacing is 7 km --
    roughly 3 samples across each hot spot's FWHM, resolving it without
    excessive forward-model cost. Increase ``n_samples`` for finer features.

    Each sample needs its own ``ForwardModel`` (a fresh one per sample
    atmosphere) -- embarrassingly parallel like ``gd_render.image()``, same
    fork + copy-on-write pattern so ``absco``/``solar`` aren't duplicated
    per worker.
    """
    from gert.forward_model import ForwardModel
    from gert.rt_solver import SingleScatterSolver

    x_samples_km = np.linspace(-SLIT_HALF_KM, SLIT_HALF_KM, n_samples)

    if n_workers is None:
        n_workers = available_cpus()
    if mp.current_process().daemon:
        n_workers = 1

    _G_LOOKUP.update(dict(x_samples_km=x_samples_km, absco=absco, inst=inst, geo=geo,
                          solar=solar, albedo=list(albedo), h2o_scale_height_km=h2o_scale_height_km,
                          fm_cls=ForwardModel, solver_cls=SingleScatterSolver))

    # one throwaway call to get wn_hires (identical for every sample -- a
    # property of the Instrument/SpectralWindow, not the atmosphere)
    _, spec0 = _lookup_sample(0)
    n_hires = len(spec0)
    spectra = np.empty((n_samples, n_hires), dtype=float)
    spectra[0] = spec0

    remaining = range(1, n_samples)
    if n_workers <= 1:
        for i in remaining:
            _, spec = _lookup_sample(i)
            spectra[i] = spec
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, spec in pool.imap_unordered(_lookup_sample, remaining, chunksize=4):
                spectra[i] = spec

    wn_hires = inst.windows[0].wn_hires   # atmosphere-independent; no extra forward call needed

    def radiance(eta):
        eta = np.atleast_1d(np.asarray(eta, dtype=float))
        x_km = eta * SLIT_HALF_KM
        idx_hi = np.clip(np.searchsorted(x_samples_km, x_km), 1, n_samples - 1)
        idx_lo = idx_hi - 1
        x_lo, x_hi = x_samples_km[idx_lo], x_samples_km[idx_hi]
        w_hi = np.clip((x_km - x_lo) / (x_hi - x_lo), 0.0, 1.0)
        w_lo = 1.0 - w_hi
        return w_lo[:, None] * spectra[idx_lo] + w_hi[:, None] * spectra[idx_hi]

    return wn_hires, radiance
