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
