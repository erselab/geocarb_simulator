"""Mission-side radiometry: GeoCarb GSD / dwell → a `gert.RadiometricNoise`.

**This is the mission half of the seam.**  `gert.radiometry` models what an
instrument *is* (photon budget, detector noise, saturation) given étendue,
per-channel bandwidth and ``t_int``.  How ``t_int`` relates to GSD is a property
of how the mission *flies*, and therefore lives here:

* **GeoCarb stares from GEO.**  Integration time is set by the dwell/scan
  schedule and is **independent of GSD**.  Signal ``N_e ∝ GSD²·t_int``, so
  shot-limited ``SNR ∝ GSD·√t_int`` and read-limited ``SNR ∝ GSD²·t_int``.
* A LEO pushbroom instead has ``t_int = GSD / v_ground`` (``N_e ∝ GSD³``).
  That coupling is *not* used here — see `gert.pushbroom_integration_time`.

Because a single quoted SNR cannot separate étendue, optical throughput and QE,
we calibrate a lumped throughput from a spec point via
`gert.RadiometricNoise.from_snr_spec` and then rescale it: for a staring
instrument the étendue scales as ``(GSD/GSD_ref)²`` at fixed optics.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from gert.instrument import LinearShotNoise
from gert.radiometry import DetectorSpec, RadiometricNoise

from .mission_config import GeoCarbInstrumentConfig as _GeoCarbInstrumentConfig

_cfg = _GeoCarbInstrumentConfig.from_yaml()

# GeoCarb reference design point. Sourced from input/geocarb_instrument.yml's
# geometry: block at import time (Phase B of the config-consolidation plan)
# -- was a typed-inline dict that had to be kept in sync with
# geosat_geometry.LongSlitGeoSatellite's own constructor defaults "by
# convention" (a prior code comment's own words); now the same YAML backs
# both, via GeoCarbInstrumentConfig.satellite(), so they agree by
# construction instead.
GEOCARB_REF = {
    "gsd_km": _cfg.geometry.pixel_size_ns_km,
    "t_int_s": _cfg.geometry.integration_time_s,
    "sat_alt_km": _cfg.geometry.sat_alt_km,
}

# Real per-band radiometric calibration from instrument testing (all
# radiances in W/m^2/sr/um), keyed by FPA index (0=O2/SIF, 1=WCO2, 2=SCO2,
# 3=CH4/CO -- same order as geocarb_gert.instrument.GEOCARB_BANDS):
#   I_ref, SNR_ref -- SNR measured at a reference radiance.
#   I_min          -- the "minimum measurable signal": the radiance at
#                     which SNR = 1.
#   I_max          -- the "maximum measurable signal": the saturation cap.
# I_ref/SNR_ref and I_min together pin down LinearShotNoise's two free
# parameters exactly (sigma(I)^2 = N0^2 + N1*I): sigma(I_ref) = I_ref/SNR_ref
# and sigma(I_min) = I_min (SNR=1 by definition of "minimum measurable
# signal") is a 2x2 linear system in (N0^2, N1) -- see
# :func:`linear_shot_noise_params`.
# Sourced from input/geocarb_instrument.yml's noise.by_fpa: block at import
# time (Phase B of the config-consolidation plan) -- was a typed-inline
# dict.
RADIOMETRIC_SPEC_BY_FPA = _cfg.noise_by_fpa


def linear_shot_noise_params(spec: dict) -> tuple:
    """Solve `LinearShotNoise`'s ``(N0, N1)`` from a two-point radiometric
    calibration: ``sigma(I_ref) = I_ref/SNR_ref`` and ``sigma(I_min) =
    I_min`` (SNR=1 at the minimum measurable signal, by definition).

    Parameters
    ----------
    spec : dict
        One entry of :data:`RADIOMETRIC_SPEC_BY_FPA` -- needs ``I_ref``,
        ``SNR_ref``, ``I_min``.

    Returns
    -------
    (N0, N1) : tuple of float
    """
    I_ref, SNR_ref, I_min = spec["I_ref"], spec["SNR_ref"], spec["I_min"]
    sigma_ref = I_ref / SNR_ref
    N1 = (sigma_ref ** 2 - I_min ** 2) / (I_ref - I_min)
    N0 = float(np.sqrt(max(I_min ** 2 - N1 * I_min, 0.0)))
    return N0, N1


def geocarb_noise_model(fpa: int) -> LinearShotNoise:
    """`LinearShotNoise` for one GeoCarb band, calibrated from real
    instrument-test data in :data:`RADIOMETRIC_SPEC_BY_FPA`.

    Includes the saturation cap (``I_max``), so
    ``geocarb_noise_model(fpa).saturation_mask(...)`` is meaningful directly.
    """
    spec = RADIOMETRIC_SPEC_BY_FPA[fpa]
    N0, N1 = linear_shot_noise_params(spec)
    return LinearShotNoise(N0=N0, N1=N1, I_max=spec["I_max"])


def geocarb_noise_model_multi(fpas) -> LinearShotNoise:
    """`LinearShotNoise` for several GeoCarb bands at once (e.g. a joint
    retrieval) -- per-band ``N0``/``N1``/``I_max`` arrays, in ``fpas`` order.
    """
    models = [geocarb_noise_model(fpa) for fpa in fpas]
    return LinearShotNoise(N0=[m.N0 for m in models], N1=[m.N1 for m in models],
                           I_max=[m.I_max for m in models])


def etendue_factor_for_gsd(gsd_km: float, gsd_ref_km: float = GEOCARB_REF["gsd_km"]) -> float:
    """Étendue scaling for a staring instrument: ``A·Ω ∝ GSD²`` at fixed optics."""
    return (float(gsd_km) / float(gsd_ref_km)) ** 2


def base_noise_model(snr_ref: float, radiance_ref: float, wl_ref_um: float,
                     dlambda_ref_um: float,
                     t_int_s: float = GEOCARB_REF["t_int_s"],
                     detector: Optional[DetectorSpec] = None) -> RadiometricNoise:
    """Calibrate the reference-design noise model from a quoted SNR spec.

    The returned model corresponds to the *reference* GSD (``GEOCARB_REF``).
    Use :func:`model_for` to move to another (GSD, dwell).
    """
    return RadiometricNoise.from_snr_spec(
        snr_ref=snr_ref, radiance_ref=radiance_ref, wl_ref_um=wl_ref_um,
        dlambda_ref_um=dlambda_ref_um, t_int_s=t_int_s, detector=detector)


def model_for(base: RadiometricNoise, gsd_km: float,
              t_int_s: float = GEOCARB_REF["t_int_s"],
              gsd_ref_km: float = GEOCARB_REF["gsd_km"]) -> RadiometricNoise:
    """A noise model at a different (GSD, dwell) — staring coupling.

    ``t_int`` is passed through independently of GSD.  That independence *is*
    the GEO advantage: a smaller footprint can be paid for with a longer stare
    rather than a bigger aperture.
    """
    return base.scaled(etendue_factor=etendue_factor_for_gsd(gsd_km, gsd_ref_km),
                       t_int_s=float(t_int_s))


def staring_sweep_models(base: RadiometricNoise, gsds_km, t_ints_s,
                         gsd_ref_km: float = GEOCARB_REF["gsd_km"]) -> dict:
    """Build a labelled grid of noise models over (GSD × dwell).

    Suitable for `gert.osse.run_design_sweep`, which holds ``K`` fixed while
    only ``Sy`` varies — so the whole grid costs one forward model.
    """
    models = {}
    for g in gsds_km:
        for t in t_ints_s:
            models[f"{g:g} km / {t:g} s"] = model_for(base, g, t, gsd_ref_km)
    return models