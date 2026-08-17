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

import multiprocessing as mp
import os
from functools import lru_cache
from typing import Callable

import numpy as np

from gert.instrument import ILS

from .focalplane import gaussian_blur_rows
from .gd_polynomials import N_FPA, N_PX, wavelength_slit_to_xy, xy_to_wavelength_slit


def available_cpus() -> int:
    """CPUs actually usable by this job (respects the SLURM cgroup
    allocation, unlike ``os.cpu_count()`` which reports the whole node's
    physical core count regardless of what's actually allocated)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


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
    # Matches gert.instrument.ILS.convolve's own _window() exactly: one extra
    # hi-res point of margin (+1) plus a final exact `|delta| <= half` cutoff
    # below -- confirmed by scripts/gd_diagonal_ils_convolve_test.py to have
    # been a real, if tiny (~3e-7 relative in line depth), discrepancy from
    # the reference convolution before this fix.
    half_idx = max(1, int(np.ceil(half / spacing)) + 1)
    n_hires = len(wn_hires)
    wn0 = wn_hires[0]

    out = np.empty(len(nu_row))
    for j, wn_c in enumerate(nu_row):
        idx_c = int(round((wn_c - wn0) / spacing))
        lo = max(0, idx_c - half_idx)
        hi = min(n_hires, idx_c + half_idx + 1)
        delta = wn_hires[lo:hi] - wn_c
        mask = np.abs(delta) <= half
        G = np.exp(-0.5 * (delta[mask] / sig) ** 2)
        Gsum = G.sum()
        out[j] = (G * S_row[j, lo:hi][mask]).sum() / Gsum if Gsum > 0 else 0.0
    return out


# -- globals populated in image() before the Pool is forked, so every worker
# -- inherits them (including the arbitrary `radiance` closure, which isn't
# -- generally picklable) via copy-on-write instead of needing IPC transfer --
_G_RENDER = {}


def _render_row(i: int):
    g = _G_RENDER
    lam_row, s_row = xy_to_wavelength_slit(g["fpa"], g["cols"], np.full(N_PX, float(i)))
    nu_row = 1.0e4 / lam_row                      # microns -> cm-1
    eta_row_true = s_row / g["sm"]                # true slit position per column
    S_row = np.asarray(g["radiance"](eta_row_true), dtype=float)   # (N_PX, n_hires)
    return i, _diagonal_ils_convolve(g["wn_hires"], S_row, nu_row, g["ils"])


def image(
    fpa: int,
    wn_hires: np.ndarray,
    radiance: Callable[[float], np.ndarray],
    ils: ILS,
    spatial_psf_fwhm_px: float = 1.5,
    n_workers: int | None = None,
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
    n_workers : int, optional
        Rows are independent (only the final PSF blur mixes them), so
        rendering is embarrassingly parallel. Defaults to
        :func:`available_cpus`. Set 1 to force the plain serial loop (e.g.
        for debugging). Automatically falls back to serial if called from
        inside an already-parallel worker process (a nested
        ``multiprocessing.Pool`` would otherwise raise "daemonic processes
        are not allowed to have children").

    Returns
    -------
    ndarray, shape (1024, 1024)
        ``A[i, j]`` -- detector row ``i``, column ``j``, in radiance units.
        Row ``i`` is the single-physical-row truth: never averaged with
        neighbouring rows (the only cross-row mixing is the final PSF blur).
    """
    cols = np.arange(N_PX, dtype=float)
    sm = s_max(fpa)

    if n_workers is None:
        n_workers = available_cpus()
    if mp.current_process().daemon:
        n_workers = 1   # already inside a worker process -- can't nest Pools

    A = np.empty((N_PX, N_PX), dtype=float)
    if n_workers <= 1:
        # Every pixel at its own true (wavelength, slit-position) -- no
        # per-row representative eta, no cross-row borrowing.
        for i in range(N_PX):
            lam_row, s_row = xy_to_wavelength_slit(fpa, cols, np.full(N_PX, float(i)))
            nu_row = 1.0e4 / lam_row                      # microns -> cm-1
            eta_row_true = s_row / sm                     # true slit position per column
            S_row = np.asarray(radiance(eta_row_true), dtype=float)   # (N_PX, n_hires)
            A[i] = _diagonal_ils_convolve(wn_hires, S_row, nu_row, ils)
    else:
        _G_RENDER.update(dict(fpa=fpa, cols=cols, sm=sm, wn_hires=wn_hires,
                              radiance=radiance, ils=ils))
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, row in pool.imap_unordered(_render_row, range(N_PX), chunksize=8):
                A[i] = row

    # Along-slit (N/S) PSF blur -- the one genuine cross-row mixing step.
    return gaussian_blur_rows(A, spatial_psf_fwhm_px)


def predict_neighborhood(
    fpa: int,
    rows: np.ndarray,
    wn_hires: np.ndarray,
    radiance: Callable[[float], np.ndarray],
    ils: ILS,
    spatial_psf_fwhm_px: float = 1.5,
    pad: int = 4,
) -> np.ndarray:
    """:func:`image`'s own per-row loop, restricted to a small row window --
    the forward operator for the joint multi-atmosphere block
    (`JOINT_ROW_INVERSION_PLAN.md` §4, Phase 1's ``predict_neighborhood``).

    Renders a padded window around ``rows`` (needed because
    ``gaussian_blur_rows`` mixes across rows, so pixels just outside the
    requested window still influence it) and returns only the requested,
    unpadded rows -- otherwise identical to :func:`image`: every pixel is
    still evaluated at its own true ``(eta, nu)``, no cross-row borrowing
    before the final PSF blur.

    Parameters
    ----------
    rows : ndarray of int
        Detector rows to return, any subset/order (need not be contiguous
        or sorted, though a contiguous block is the intended use).
    pad : int
        Extra rows rendered on each side of ``[rows.min(), rows.max()]``
        solely for correct PSF-blur edge handling, then discarded. Should
        be a few multiples of ``spatial_psf_fwhm_px``; the default (4)
        matches ``gaussian_blur_rows``'s own edge-extension half-width for
        FWHM~1.5px.

    Returns
    -------
    ndarray, shape (len(rows), 1024)
    """
    rows = np.asarray(rows, dtype=int)
    row_lo, row_hi = int(rows.min()) - pad, int(rows.max()) + pad
    rows_padded = np.arange(max(0, row_lo), min(N_PX, row_hi + 1))
    cols = np.arange(N_PX, dtype=float)
    sm = s_max(fpa)

    A_pad = np.empty((len(rows_padded), N_PX), dtype=float)
    for k, i in enumerate(rows_padded):
        lam_row, s_row = xy_to_wavelength_slit(fpa, cols, np.full(N_PX, float(i)))
        nu_row = 1.0e4 / lam_row
        eta_row_true = s_row / sm
        S_row = np.asarray(radiance(eta_row_true), dtype=float)
        A_pad[k] = _diagonal_ils_convolve(wn_hires, S_row, nu_row, ils)

    A_pad = gaussian_blur_rows(A_pad, spatial_psf_fwhm_px)
    idx = np.searchsorted(rows_padded, rows)
    return A_pad[idx]


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
