"""Real GeoCarb geometric-distortion (GD) polynomials, from ground-test data.

Loads the EM27/SUN-refined 4th-degree 2D polynomial coefficients
(``gcmap_em27.csv``, at the repo root) that describe the actual measured
mapping between detector pixels and (wavelength, slit position) for each of
GeoCarb's four FPAs. Source: ground-test geometric-distortion characterization
(``keystone_report.pdf`` Sec. 4.8.1 "Initial Fitting of Polynomial Models from
Test Data" and Sec. 4.8.6 "Refinement Using EM27/SUN Spectra"); see
``KEYSTONE_SMILE_BIAS_PLAN.md`` Sec. 9 for how the coefficient table and its
coordinate conventions were validated in this repo.

Coordinate conventions (validated against the report's own figures):
    x, y        FPA pixel coordinates, 0..1023
    wavelength  microns
    s           slit angle, degrees, roughly in [-2, 2]

The CSV holds all 8 transformations from the report's Table 5 (32 coefficient
sets = 8 transformations x 4 FPAs), but only the two requested directions are
wrapped here:

    rectification  (x, y)                    -> (wavelength, s)   [A, B]
    projection     (wavelength, s)            -> (x, y)           [C, D]

FPA index order matches ``geocarb_gert.instrument.GEOCARB_BANDS``: 0=O2_A,
1=CO2_weak, 2=CO2_strong, 3=CH4_CO.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

_CSV_PATH = Path(__file__).resolve().parent.parent / "gcmap_em27.csv"

_TERMS = ["const", "x", "y", "xx", "xy", "yy", "xxx", "xxy", "xyy", "yyy",
          "xxxx", "xxxy", "xxyy", "xyyy", "yyyy"]

N_FPA = 4
N_PX = 1024   # real detector rows/columns

# Whether wavenumber increases with raw column index, at each FPA's own
# column-to-wavelength mapping (xy_to_wavelength_slit at a representative
# row). Alternates FPA to FPA -- confirmed by the user 2026-07-24: internal
# reflections/beam splitters in the real optical path flip the dispersion
# direction for alternating bands, not a calibration artifact.
#
#   FPA0 (O2_A)       False -- descending wavenumber with column
#   FPA1 (CO2_weak)   True  -- ascending wavenumber with column
#   FPA2 (CO2_strong) False -- descending wavenumber with column
#   FPA3 (CH4_CO)     True  -- ascending wavenumber with column (expected;
#                               not yet verified end-to-end pending the
#                               ABSCO ch4/h2o/co coverage extension, see
#                               KEYSTONE_SMILE_BIAS_PLAN.md)
#
# This matters because gert.ForwardModel always returns y/y_ret in
# ascending-*wavelength* order (= descending wavenumber) regardless of
# obs_grid's input order (gert/instrument.py's SpectralWindow.wn_instrument
# sorts ascending, then wl_instrument reverses it). A native-grid
# retrieval's y_dist -- built by indexing the raw rendered row in column
# order -- only lines up with that convention for FPA0/FPA2; FPA1/FPA3 need
# an explicit reversal. Every earlier native-grid script in this study
# happened to only ever use FPA2, so this asymmetry went unnoticed until
# scripts/gd_band_stress_test.py exercised FPA1 and every retrieval failed
# catastrophically, even at the slit centre where truth == prior -- see
# KEYSTONE_SMILE_BIAS_PLAN.md Sec. 9o. Don't assume either direction:
# check ``nu_row[0] < nu_row[-1]`` (or use DISPERSION_ASCENDING) and
# reverse if needed, the way gd_band_stress_test.py's _worker() does.
DISPERSION_ASCENDING = {0: False, 1: True, 2: False, 3: True}


@lru_cache(maxsize=1)
def _coeffs(csv_path: str = str(_CSV_PATH)) -> dict:
    """Load and cache the GD coefficient table.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{column: {term: coefficient}}``, e.g. ``coeffs["A0"]["xy"]``.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    columns = header[1:]
    out = {col: {} for col in columns}
    for row in body:
        term = row[0]
        for col, val in zip(columns, row[1:]):
            out[col][term] = float(val)
    for col, term_coeffs in out.items():
        missing = set(_TERMS) - term_coeffs.keys()
        if missing:
            raise ValueError(f"{csv_path}: column {col!r} missing terms {missing}")
    return out


def _poly2d(term_coeffs: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Evaluate a 4th-degree bivariate polynomial sum_k c_k * term_k(a, b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    values = {
        "const": np.ones_like(a), "x": a, "y": b,
        "xx": a**2, "xy": a * b, "yy": b**2,
        "xxx": a**3, "xxy": a**2 * b, "xyy": a * b**2, "yyy": b**3,
        "xxxx": a**4, "xxxy": a**3 * b, "xxyy": a**2 * b**2,
        "xyyy": a * b**3, "yyyy": b**4,
    }
    return sum(term_coeffs[t] * values[t] for t in _TERMS)


def _check_fpa(fpa: int) -> None:
    if not 0 <= fpa < N_FPA:
        raise ValueError(f"fpa must be 0-{N_FPA - 1}, got {fpa}")


def xy_to_wavelength_slit(fpa: int, x, y):
    """Rectify detector pixel coordinates to (wavelength, slit position).

    Parameters
    ----------
    fpa : int
        FPA index, 0-3 (see module docstring for band order).
    x, y : array-like
        FPA pixel coordinates.

    Returns
    -------
    wavelength : ndarray, microns
    s : ndarray, slit angle in degrees
    """
    _check_fpa(fpa)
    coeffs = _coeffs()
    wavelength = _poly2d(coeffs[f"A{fpa}"], x, y)
    s = _poly2d(coeffs[f"B{fpa}"], x, y)
    return wavelength, s


@lru_cache(maxsize=N_FPA)
def real_wavenumber_range(fpa: int, margin_cm1: float = 10.0):
    """This FPA's real per-pixel channel-center wavenumber range, + margin.

    The nominal ``GEOCARB_BANDS`` (wn_min, wn_max) is generally *narrower*
    than the real per-pixel range (e.g. FPA0: nominal [12950, 13190], real
    [12957, 13233] -- up to ~40 cm-1 short). Any hi-res grid or retrieval
    window used with the real GD curves (:mod:`geocarb_gert.gd_render`) must
    cover this wider range, or ILS convolution silently zero-weights pixels
    outside it. ``margin_cm1`` adds a safety buffer beyond the observed
    real min/max (for the ILS's own truncation half-width).

    Returns
    -------
    (wn_min, wn_max) : tuple of float
    """
    _check_fpa(fpa)
    cols = np.arange(N_PX, dtype=float)
    nu_min, nu_max = np.inf, -np.inf
    for i in range(0, N_PX, 8):          # sparse row sampling is enough for an envelope
        lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(N_PX, float(i)))
        nu_row = 1.0e4 / lam_row
        nu_min = min(nu_min, float(nu_row.min()))
        nu_max = max(nu_max, float(nu_row.max()))
    # Snap outward to a 0.01 cm-1 grid -- ABSCOTable.wn_index requires an exact
    # match to its own (0.01-spaced) grid, not just any float in range.
    lo = np.floor((nu_min - margin_cm1) * 100.0) / 100.0
    hi = np.ceil((nu_max + margin_cm1) * 100.0) / 100.0
    return lo, hi


def rows_crossed(fpa: int, row) -> np.ndarray:
    """Keystone row-crossing at detector row(s) ``row`` -- how many physical
    rows this row's own dispersion trace spans across its full column range.

    Same quantity as ``keystone_report.pdf`` Fig. 38 (there evaluated per
    slit position via the E(x,s) polynomial); here evaluated directly per
    detector row via B(x, row), which is what matters for picking rows to
    test (KEYSTONE_SMILE_BIAS_PLAN.md Sec. 9h): 0 at FPA2's row 25 (its real
    keystone-null row, Sec. 9c), growing to ~10 at the slit ends, matching
    the ground-test report.

    Parameters
    ----------
    fpa : int
    row : array-like

    Returns
    -------
    ndarray -- rows crossed (same shape as ``row``)
    """
    _check_fpa(fpa)
    row = np.asarray(row, dtype=float)
    _, s_lo = xy_to_wavelength_slit(fpa, np.zeros_like(row) + 4.0, row)
    _, s_hi = xy_to_wavelength_slit(fpa, np.zeros_like(row) + (N_PX - 5.0), row)
    _, s_ref0 = xy_to_wavelength_slit(fpa, np.array([4.0]), np.array([0.0]))
    _, s_ref1 = xy_to_wavelength_slit(fpa, np.array([4.0]), np.array([float(N_PX - 1)]))
    deg_per_row = float(np.abs(s_ref1 - s_ref0)[0]) / (N_PX - 1)
    return np.abs(s_hi - s_lo) / deg_per_row


@lru_cache(maxsize=N_FPA)
def slit_eta_reference(fpa: int) -> tuple:
    """``(s_center, s_half)`` [deg]: the slit-image centre and half-length that
    define this FPA's ``eta`` (2026-09-20).

    ``eta = (s - s_center) / s_half`` so that ``eta = -1 / +1`` are the two
    ends of the slit image and ``eta = 0`` its centre -- for EVERY FPA, so equal
    ``eta`` means the same fractional position along the (shared) physical slit
    in every band. Replaces the earlier ``eta = s / s_max(fpa)``, which used
    each FPA's own larger edge ``|s|`` and so put the two slit ends at different
    ``eta`` on every FPA (only the wider end reached +/-1) and did not centre on
    the slit image.

    Assumptions (user, 2026-09-20, for these experiments): the slit image fills
    the FPA rows 0..1023 at the LONG-WAVELENGTH end of the band (so the
    reference column is whichever of column 0 / 1023 has the longer wavelength
    -- 1023 for FPA0 and FPA2, 0 for FPA1 and FPA3), and the keystone mapping
    ``s(row, col)`` itself is taken as perfect. The absolute ``s`` calibration
    is NOT assumed comparable across FPAs (differences are attributed to
    misalignment/defocus), which is why each band is centred and scaled by its
    own slit image rather than by a shared ``s``.
    """
    lam, _ = xy_to_wavelength_slit(fpa, np.array([0.0, N_PX - 1.0]), np.full(2, N_PX / 2 - 0.5))
    col_long = 0.0 if lam[0] > lam[1] else N_PX - 1.0
    _, s_ends = xy_to_wavelength_slit(fpa, np.full(2, col_long), np.array([0.0, N_PX - 1.0]))
    return float(np.mean(s_ends)), float(abs(s_ends[1] - s_ends[0]) / 2.0)


def eta_of_s(fpa: int, s):
    """Real slit angle ``s`` [deg] -> ``eta`` (see :func:`slit_eta_reference`)."""
    c, h = slit_eta_reference(fpa)
    return (np.asarray(s, dtype=float) - c) / h


def s_of_eta(fpa: int, eta):
    """Inverse of :func:`eta_of_s`."""
    c, h = slit_eta_reference(fpa)
    return c + h * np.asarray(eta, dtype=float)


def wavelength_slit_to_xy(fpa: int, wavelength, s):
    """Project (wavelength, slit position) to detector pixel coordinates.

    Parameters
    ----------
    fpa : int
        FPA index, 0-3 (see module docstring for band order).
    wavelength : array-like, microns
    s : array-like
        Slit angle in degrees.

    Returns
    -------
    x, y : ndarray, FPA pixel coordinates
    """
    _check_fpa(fpa)
    coeffs = _coeffs()
    x = _poly2d(coeffs[f"C{fpa}"], wavelength, s)
    y = _poly2d(coeffs[f"D{fpa}"], wavelength, s)
    return x, y


# ---------------------------------------------------------------------------
# Assumed (mismatched) calibration -- for the imperfect-knowledge experiment,
# see KEYSTONE_SMILE_BIAS_PLAN.md Sec. 12. `xy_to_wavelength_slit` above is
# always the REAL calibration, used to render truth (gd_render.image()/
# rectify() must stay on it). The functions below build a separate ASSUMED
# calibration -- what a retrieval pipeline believes the mapping is, when the
# ground-test polynomial fit has residual noise and/or the on-orbit
# instrument has thermally drifted from that fit -- for use only in
# retrieval-side position bookkeeping (gd_band_stress_test.py's _worker()).
# ---------------------------------------------------------------------------

_LOW_ORDER_TERMS = _TERMS[:6]   # const, x, y, xx, xy, yy -- smooth, not per-pixel


def _rescaled_random_field(seed, target_rms: float, x: np.ndarray, y: np.ndarray) -> dict:
    """A random low-order 2D polynomial (see _LOW_ORDER_TERMS), rescaled so
    its RMS over (x, y) equals target_rms. {} (no perturbation) if seed is
    None or target_rms <= 0 -- a "noise" term that's off unless both a
    magnitude and a seed are given."""
    if seed is None or target_rms <= 0:
        return {}
    rng = np.random.default_rng(seed)
    raw = {t: (rng.normal() if t in _LOW_ORDER_TERMS else 0.0) for t in _TERMS}
    field = _poly2d(raw, x, y)
    achieved = float(np.sqrt(np.mean(field ** 2)))
    if achieved < 1e-12:
        return {}
    scale = target_rms / achieved
    return {t: c * scale for t, c in raw.items()}


def perturbed_coeffs(fpa: int, wn_bias_cm1: float = 0.0, wn_noise_rms_cm1: float = 0.0,
                     slit_bias_km: float = 0.0, slit_noise_rms_km: float = 0.0,
                     seed: int | None = None, slit_half_km: float = 1400.0) -> dict:
    """Build an ASSUMED coefficient set for one FPA: the real A{fpa}/B{fpa}
    term dicts, each with a low-order perturbation added -- a deterministic
    bias (thermal-drift stand-in) plus a random smooth field at a target RMS
    (ground-test fit-noise stand-in), specified in physical units (cm-1 for
    wavelength, km for slit position) and converted to the polynomial's own
    units (microns, degrees) via a band-centre scale factor.

    See KEYSTONE_SMILE_BIAS_PLAN.md Sec. 12 for the experiment this
    supports and why the perturbation is smooth/low-order rather than
    per-pixel: a finite polynomial fit's own residual error is spatially
    smooth, not iid noise, and a real thermal drift shifts the whole
    dispersion/keystone relation, not individual pixels.

    Parameters
    ----------
    fpa : int
    wn_bias_cm1 : float
        Deterministic wavenumber bias [cm-1] (thermal-drift stand-in).
        Positive = assumed wavenumber too high (assumed wavelength too low).
    wn_noise_rms_cm1 : float
        RMS of a random smooth wavenumber perturbation [cm-1] (ground-test
        fit-noise stand-in). Only applied if `seed` is not None.
    slit_bias_km : float
        Deterministic along-slit position bias [km].
    slit_noise_rms_km : float
        RMS of a random smooth along-slit position perturbation [km]. Only
        applied if `seed` is not None.
    seed : int, optional
        Seed for the noise draws (the wavelength and slit noise fields use
        `seed` and `seed + 1` respectively, so they're independent draws
        from one experiment seed). None disables both noise terms
        regardless of the *_rms args -- bias-only perturbation.
    slit_half_km : float
        Real half-slit-length [km] used for the deg<->km conversion --
        pass `geocarb_gert.along_slit_scene.SLIT_HALF_KM` (1400.0) to match
        the rest of this codebase's convention; not imported directly here
        to avoid a circular import.

    Returns
    -------
    dict
        {"A{fpa}": {...term dict...}, "B{fpa}": {...term dict...}} -- ready
        to substitute into a `_poly2d` call in place of `_coeffs()`'s own
        entries (see `xy_to_wavelength_slit_assumed`).
    """
    _check_fpa(fpa)
    coeffs = _coeffs()
    cols = np.arange(N_PX, dtype=float)
    rows = np.arange(N_PX, dtype=float)
    gx, gy = np.meshgrid(cols, rows)   # full pixel grid, for RMS calibration

    # Band-centre scale factors converting the physical-unit bias/RMS
    # (cm-1, km) into the polynomial's own units (microns, degrees). Uses
    # the REAL mapping (xy_to_wavelength_slit, not this function) since the
    # conversion factor itself should reflect the true instrument, not an
    # already-perturbed one.
    lam_c, _ = xy_to_wavelength_slit(fpa, np.array([N_PX / 2]), np.array([N_PX / 2]))
    lam_c = float(lam_c[0])
    um_per_cm1 = lam_c ** 2 / 1.0e4             # |dlambda/dnu| at band centre
    _, sm = slit_eta_reference(fpa)             # slit-image half-length [deg] <-> slit_half_km
    deg_per_km = sm / slit_half_km

    wn_bias_um = -wn_bias_cm1 * um_per_cm1      # wavenumber up -> wavelength down
    wn_noise_um = wn_noise_rms_cm1 * um_per_cm1
    slit_bias_deg = slit_bias_km * deg_per_km
    slit_noise_deg = slit_noise_rms_km * deg_per_km

    A = dict(coeffs[f"A{fpa}"])
    B = dict(coeffs[f"B{fpa}"])
    A["const"] = A["const"] + wn_bias_um
    B["const"] = B["const"] + slit_bias_deg

    rand_A = _rescaled_random_field(seed, wn_noise_um, gx, gy)
    rand_B = _rescaled_random_field(None if seed is None else seed + 1, slit_noise_deg, gx, gy)
    for t, c in rand_A.items():
        A[t] = A.get(t, 0.0) + c
    for t, c in rand_B.items():
        B[t] = B.get(t, 0.0) + c

    return {f"A{fpa}": A, f"B{fpa}": B}


def xy_to_wavelength_slit_assumed(fpa: int, x, y, mismatch: dict):
    """Same as `xy_to_wavelength_slit`, but evaluated against an ASSUMED
    (perturbed) coefficient set from `perturbed_coeffs()` instead of the
    real one -- the retrieval-side "what the pipeline believes the mapping
    is" counterpart. Never use this to render truth: gd_render.image()/
    rectify() must stay on the real `xy_to_wavelength_slit`, or there is no
    actual mismatch to measure."""
    _check_fpa(fpa)
    wavelength = _poly2d(mismatch[f"A{fpa}"], x, y)
    s = _poly2d(mismatch[f"B{fpa}"], x, y)
    return wavelength, s
