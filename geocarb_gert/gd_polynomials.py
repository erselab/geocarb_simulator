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
