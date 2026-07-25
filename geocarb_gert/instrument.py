"""GeoCarb spectral bands as a `gert.Instrument`.

The four GeoCarb bands, with the interfering species restricted to those the
ABSCO table actually covers densely.  Verified against ``absco.h5``: all four
bands have dense 0.01 cm⁻¹ coverage for every molecule listed here.

Note the O2 band uses the ``o2`` table (the 760 nm A-band) — *not* ``o2_1p27``,
which is the 1.27 µm band used by EM27/SUN.
"""
from __future__ import annotations

from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument

# (label, wn_min, wn_max, molecules, resolving_power)
# GeoCarb resolving power λ/Δλ ≈ 20 000 (O2 A) … ≈ 12 000 (2.3 µm).
GEOCARB_BANDS = [
    ("O2_A",     12950.0, 13190.0, ["o2", "h2o"],                 19000.0),
    ("CO2_weak",  6166.0,  6286.0, ["co2", "h2o"],                20000.0),
    ("CO2_strong", 4810.0, 4897.0, ["co2", "h2o"],                20000.0),
    ("CH4_CO",    4258.60, 4350.69, ["ch4", "co", "h2o", "n2o"],  12000.0),
    # CH4_CO's wn range was [4208.0, 4318.0] until 2026-07-24 -- centered at
    # 4263.0, ~41 cm-1 below the real per-pixel band's centre (~4304.6,
    # confirmed independently against ground-test wavelength data of
    # [2.2989, 2.3461] um = [4262.4, 4349.9] cm-1). That's not the usual
    # "nominal snugly narrower than real" pattern the other three bands show
    # (e.g. FPA0: nominal [12950,13190] vs real [12957,13233], ~40 cm-1
    # short on a 240 cm-1 band) -- it was a ~41 cm-1 shift on a 110 cm-1
    # band, so rectified/undistorted pipelines were silently missing about
    # a third of every row's real spectral content at the high-wavenumber
    # end. Corrected to real_wavenumber_range(3, margin_cm1=0.0) exactly
    # (geocarb_gert/gd_polynomials.py's own per-pixel dispersion-polynomial
    # envelope), so the nominal grid now fully contains the real range with
    # zero margin -- matches ground-test data. See KEYSTONE_SMILE_BIAS_PLAN.md.
]


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