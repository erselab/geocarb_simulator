"""Focal-plane image renderer driven by the real ground-test GD polynomials.

Renders at the real detector's native 1024x1024 resolution, using the actual
measured ``xy_to_wavelength_slit`` curves from :mod:`geocarb_gert.
gd_polynomials` (see ``KEYSTONE_SMILE_BIAS_PLAN.md`` Sec. 9b/9c/9f for how
those curves were validated) -- not the analytic keystone/smile/clocking
formulas :class:`geocarb_gert.focalplane.FocalPlaneModel` uses.

**Non-separable by construction.** Each pixel ``(i, j)`` is evaluated at its
own true (wavelength, slit-position) pair, ``A(j, i)``/``B(j, i)`` -- there is
no per-row "pick one representative slit position" step, and no resampling
across rows. An earlier version of this module *did* factor rendering into a
per-row spectral step followed by a cross-row spatial resample (mirroring
``FocalPlaneModel``'s architecture) -- that reintroduces a real error even on
a spatially uniform scene: because smile varies (slowly) with row, borrowing
a neighbouring row's already-dispersion-shifted content mixes two different
wavelength calibrations together. Evaluating every pixel at its own row's
correct dispersion curve, with no cross-row borrowing, removes that. Along-
slit mixing between true neighbouring positions comes only from the real N/S
PSF blur (the one physical step for which cross-row mixing is *correct*, not
an artifact) -- so detector row ``i``'s rendered spectrum, ``image(...)[i,
:]``, is the genuine single-physical-row truth, never averaged.

A "scene" here is the same convention used throughout ``geocarb_gert.
focalplane``: a shared hi-res wavenumber grid ``wn_hires`` plus a callable
``radiance(eta) -> S_hires`` giving the hi-res spectrum at object slit
position ``eta`` (scalar *or* array -- required here, since a full row's
worth of per-pixel true slit positions is evaluated in one vectorized call),
with ``eta`` on the standard ``[-1, 1]`` convention (see :func:`uniform_scene`,
:func:`edge_scene`, :func:`random_scene`, :func:`geocarb_gert.focalplane.
barcode_scene`). Real slit angle ``s`` [deg] is mapped to this convention via
``eta = s / s_max(fpa)``, where ``s_max(fpa)`` is that FPA's own real ``|s|``
at the detector edge (row 0 or 1023) -- so ``eta = -1`` and ``eta = +1``
always correspond to the true top and bottom of that FPA's slit.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np

from gert.instrument import ILS

from .focalplane import gaussian_blur_rows
from .gd_polynomials import N_FPA, N_PX, wavelength_slit_to_xy, xy_to_wavelength_slit


@lru_cache(maxsize=N_FPA)
def s_max(fpa: int) -> float:
    """This FPA's real |slit angle| [deg] at the detector edge (row 0 or 1023).

    Used to map real slit angle to the standard ``eta in [-1, 1]`` scene
    convention: ``eta = s / s_max(fpa)``.
    """
    _, s = xy_to_wavelength_slit(fpa, np.full(2, N_PX / 2 - 0.5), np.array([0.0, N_PX - 1.0]))
    return float(np.abs(s).max())


def real_row_eta(fpa: int) -> np.ndarray:
    """Each detector row's real slit position, in the ``eta in [-1, 1]`` convention.

    Evaluated at the row's own centre column. A diagnostic/plotting
    convenience -- not used internally by :func:`image`, which evaluates
    every pixel's own true slit position rather than one per-row value.
    """
    y = np.arange(N_PX, dtype=float)
    _, s = xy_to_wavelength_slit(fpa, np.full(N_PX, N_PX / 2 - 0.5), y)
    return s / s_max(fpa)


def _diagonal_ils_convolve(wn_hires: np.ndarray, S_row: np.ndarray,
                           nu_row: np.ndarray, ils: ILS) -> np.ndarray:
    """Convolve a *different* hi-res spectrum against each output centre.

    Unlike ``ILS.convolve`` (one shared spectrum, many centres), each column
    ``j`` here has its own ``S_row[j]`` -- needed because every pixel in a row
    may sample a different true slit position (keystone), hence a different
    scene spectrum, not just a different wavelength. Same truncated-Gaussian
    kernel and normalisation as ``ILS.convolve(..., exact_center=True)``.

    Windowed: ``wn_hires`` is assumed uniformly spaced (true for every hi-res
    grid used in this codebase -- ``SpectralWindow.wn_hires`` and every
    ``np.arange``/``np.linspace`` grid built for these renders), so each
    centre's truncated-Gaussian support window is sliced by direct index
    arithmetic instead of computing ``wn_hires - wn_c`` and masking over the
    *full* array on every one of the ~1e6 (row, col) calls this makes for a
    full 1024x1024 render. Full-array masking was the dominant cost (the
    window is typically ~1-2% of ``n_hires``).

    Parameters
    ----------
    wn_hires : ndarray, shape (n_hires,)
        Uniformly spaced, ascending.
    S_row : ndarray, shape (n_cols, n_hires)
    nu_row : ndarray, shape (n_cols,)
    ils : gert.instrument.ILS

    Returns
    -------
    ndarray, shape (n_cols,)
    """
    sig = ils.sigma
    half = ils.ils_half_width * sig
    spacing = wn_hires[1] - wn_hires[0]
    half_idx = max(1, int(np.ceil(half / spacing)))
    n_hires = len(wn_hires)
    wn0 = wn_hires[0]

    out = np.empty(len(nu_row))
    for j, wn_c in enumerate(nu_row):
        idx_c = int(round((wn_c - wn0) / spacing))
        lo = max(0, idx_c - half_idx)
        hi = min(n_hires, idx_c + half_idx + 1)
        delta = wn_hires[lo:hi] - wn_c
        G = np.exp(-0.5 * (delta / sig) ** 2)
        Gsum = G.sum()
        out[j] = (G * S_row[j, lo:hi]).sum() / Gsum if Gsum > 0 else 0.0
    return out


def image(
    fpa: int,
    wn_hires: np.ndarray,
    radiance: Callable[[float], np.ndarray],
    ils: ILS,
    spatial_psf_fwhm_px: float = 1.5,
) -> np.ndarray:
    """Render the real-detector focal-plane image for one FPA.

    Parameters
    ----------
    fpa : int
        FPA index, 0-3 (see ``geocarb_gert.gd_polynomials`` for band order).
    wn_hires : ndarray, shape (n_hires,)
        Hi-res wavenumber grid [cm-1], monotonically increasing, covering the
        band with ILS wings.
    radiance : callable
        ``radiance(eta) -> S_hires`` -- the hi-res spectrum at object slit
        position ``eta`` (must accept an array ``eta`` and return shape
        ``eta.shape + (n_hires,)``; see module docstring).
    ils : gert.instrument.ILS
        Spectral response used for the per-pixel convolution.
    spatial_psf_fwhm_px : float
        Along-slit (N/S) PSF FWHM [detector pixels] -- a *different* blur
        from ``ils`` (which is spectral, in cm-1). GeoCarb measured ~1.5 px
        (KEYSTONE_SMILE_BIAS_PLAN.md Sec. 1/9). Set 0 to disable.

    Returns
    -------
    ndarray, shape (1024, 1024)
        ``A[i, j]`` -- detector row ``i``, column ``j``, in radiance units.
        Row ``i`` is the single-physical-row truth: never averaged with
        neighbouring rows (the only cross-row mixing is the final PSF blur).
    """
    cols = np.arange(N_PX, dtype=float)
    sm = s_max(fpa)

    # Every pixel at its own true (wavelength, slit-position) -- no per-row
    # representative eta, no cross-row borrowing.
    A = np.empty((N_PX, N_PX), dtype=float)
    for i in range(N_PX):
        lam_row, s_row = xy_to_wavelength_slit(fpa, cols, np.full(N_PX, float(i)))
        nu_row = 1.0e4 / lam_row                      # microns -> cm-1
        eta_row_true = s_row / sm                     # true slit position per column
        S_row = np.asarray(radiance(eta_row_true), dtype=float)   # (N_PX, n_hires)
        A[i] = _diagonal_ils_convolve(wn_hires, S_row, nu_row, ils)

    # Along-slit (N/S) PSF blur -- the one genuine cross-row mixing step.
    return gaussian_blur_rows(A, spatial_psf_fwhm_px)


def rectify(fpa: int, A: np.ndarray, s_grid: np.ndarray, wn_grid: np.ndarray,
           order: int = 1) -> np.ndarray:
    """Rectify a raw detector image onto a regular (slit, wavenumber) grid.

    This is what a real L1B pipeline does to turn a raw detector image into
    "one spectrum per slit position": for each point on the *output* regular
    grid, use the inverse polynomial mapping (``wavelength_slit_to_xy``, the
    C/D pair) to find where it came from in the *raw* image, then
    interpolate there -- standard inverse-mapping image resampling, and the
    simulated counterpart to the real GD-polynomial rectification described
    in KEYSTONE_SMILE_BIAS_PLAN.md Sec. 9 (``keystone_report.pdf`` Sec.
    4.8.1: "the polynomials encode the mapping of any location in the image
    (x,y) to a slit position and wavelength (lambda,s)"; rectification
    applies that mapping to resample onto a regular grid).

    Not exact even net of interpolation error: A/B and C/D are independently
    fit polynomials, not exact inverses of each other (§9h's round-trip
    check found sub-pixel but nonzero discrepancy) -- exactly as real,
    independently-fit rectification/projection polynomials are not exact
    inverses either (``keystone_report.pdf`` §4.8.1: "the resulting
    polynomials are not exactly invertible among each other").

    Parameters
    ----------
    fpa : int
    A : ndarray, shape (1024, 1024)
        Raw detector image, as returned by :func:`image`.
    s_grid : ndarray, shape (n_s,)
        Target slit-angle grid [deg], ascending.
    wn_grid : ndarray, shape (n_wn,)
        Target wavenumber grid [cm-1], ascending.
    order : int
        Interpolation order passed to ``scipy.ndimage.map_coordinates``
        (1 = bilinear, matching the real pipeline's linear regridding).

    Returns
    -------
    ndarray, shape (n_s, n_wn)
        Rectified spectrogram ``R[k, l]`` at ``(s_grid[k], wn_grid[l])``.
        ``nan`` where the source pixel falls outside the raw image (off the
        detector).
    """
    from scipy.ndimage import map_coordinates
    lam_grid = 1.0e4 / wn_grid                       # cm-1 -> microns
    SS, LL = np.meshgrid(s_grid, lam_grid, indexing="ij")   # (n_s, n_wn)
    x, y = wavelength_slit_to_xy(fpa, LL, SS)
    return map_coordinates(A, [y, x], order=order, mode="constant", cval=np.nan)
