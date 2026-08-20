"""GeoCarb spectral bands as a `gert.Instrument`.

The four GeoCarb bands, with the interfering species restricted to those the
ABSCO table actually covers densely.  Verified against ``absco.h5``: all four
bands have dense 0.01 cm⁻¹ coverage for every molecule listed here.

"""
from __future__ import annotations

from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument

from .mission_config import GeoCarbInstrumentConfig as _GeoCarbInstrumentConfig

# (label, wn_min, wn_max, molecules, resolving_power)
# GeoCarb resolving power λ/Δλ ≈ 20 000 (O2 A) … ≈ 12 000 (2.3 µm).
# Sourced from input/geocarb_instrument.yml's bands: block at import time
# (Phase B of the config-consolidation plan) -- was a typed-inline literal
# list. Kept as a module constant since it's imported bare in several
# places (e.g. scripts/gd_test.py's GEOCARB_BANDS[fpa]); a non-default
# instrument config must go through GeoCarbInstrumentConfig.bands directly.
GEOCARB_BANDS = _GeoCarbInstrumentConfig.from_yaml().bands


def build_geocarb_instrument(bands=GEOCARB_BANDS, channels_per_fwhm: int = 3,
                             hires_spacing: float = 0.01,
                             snr: float = 300.0,
                             noise_model=None) -> Instrument:
    """Build the GeoCarb `gert.Instrument`.

    Parameters
    ----------
    bands : sequence of (label, wn_min, wn_max, molecules, resolving_power)
    channels_per_fwhm : int
        Spectral sampling.  Note this sets the per-channel *bandwidth* used by
        the radiometric noise model, which is distinct from the ILS FWHM.
    noise_model : gert NoiseModel, optional
        e.g. a `gert.RadiometricNoise`.  When given it overrides ``snr``.
    """
    windows = []
    for label, wn_min, wn_max, mols, R in bands:
        wn_c = 0.5 * (wn_min + wn_max)
        fwhm_cm = wn_c / float(R)              # Δν = ν / R
        windows.append(SpectralWindow(
            wn_min=wn_min, wn_max=wn_max,
            ils=ILS(type="gaussian", fwhm=fwhm_cm),
            molecules=list(mols), label=label,
            hires_spacing=hires_spacing,
            channels_per_fwhm=channels_per_fwhm,
        ))
    return Instrument(windows=windows, snr=snr, noise_model=noise_model)