"""Robust (outlier-resistant) summary statistics for retrieval diagnostics.

A converged, non-diverged row can still be a catastrophic outlier -- most
visibly in barcode scenes, where a handful of bar-edge rows can carry a
bias or chi2 orders of magnitude above the rest of the slit and dominate a
plain mean/std even though the fit formally "converged". This module
centralizes the median-absolute-deviation (MAD) based outlier handling
that :mod:`scripts.gd_plot`/:mod:`scripts.gd_plot_grid` already applied to
chi2 (each previously carried its own private copy of the same function),
and generalizes it to any quantity -- gas/pressure bias included, which is
where a large reported std most often traces back to a small number of
rows rather than genuine broad scatter.

All three functions are pure numpy, no gert/geocarb_gert-specific
assumptions, so they're equally usable on chi2 (right-skewed, always
positive -- use ``log=True``) or on a bias series (signed, roughly
symmetric around zero -- use ``log=False``, the default).
"""
from __future__ import annotations

import numpy as np


def median_mad(values) -> tuple:
    """Median and the MAD-based robust sigma estimate.

    ``1.4826 * median(|x - median(x)|)`` is the scale factor that makes
    the MAD a consistent estimator of the standard deviation for normally
    distributed data, while remaining insensitive to extreme outliers
    (unlike ``np.std``, whose breakdown point is 0 -- a single huge value
    can inflate it arbitrarily).

    Parameters
    ----------
    values : array-like

    Returns
    -------
    (median, robust_sigma, n) : tuple of float, float, int
        ``robust_sigma`` is 0.0 if ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), 0.0, 0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, 1.4826 * mad, arr.size


def mad_outlier_mask(values, n_mad: float = 8.0, log: bool = False) -> np.ndarray:
    """Boolean mask, True where ``values`` is a MAD outlier.

    Same method and default threshold (8 robust-sigma) used throughout
    this project's own chi2-outlier filtering (``gd_plot.py``/
    ``gd_plot_grid.py``'s prior private ``_chi2_outlier_mask``). Chi2-like
    quantities (right-skewed, strictly positive) should pass ``log=True``
    so the MAD is computed in log space, where the distribution is closer
    to symmetric; bias-like quantities (signed, roughly symmetric around
    zero already) should use the default ``log=False``.

    Parameters
    ----------
    values : array-like
    n_mad : float
        Outlier threshold in robust-sigma units.
    log : bool
        Compute the MAD in log10 space (for right-skewed, positive-only
        quantities like chi2). Values <= 0 are floored to 1e-300 first so
        ``log10`` never sees a non-positive input.

    Returns
    -------
    ndarray of bool, same shape as ``values`` -- True = outlier.
    NaN/non-finite entries are always marked as outliers (they carry no
    usable information and would otherwise silently propagate into any
    downstream mean/std).
    """
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.ones(arr.shape, dtype=bool)
    work = np.log10(np.maximum(arr, 1e-300)) if log else arr
    med, sigma, n = median_mad(work[finite])
    out = np.ones(arr.shape, dtype=bool)
    if sigma < 1e-12:
        # degenerate (near-constant) distribution -- MAD collapses to ~0,
        # so fall back to a large absolute threshold rather than flagging
        # every point that differs from the median at all.
        out[finite] = np.abs(work[finite] - med) > 1e6
        return out
    out[finite] = np.abs(work[finite] - med) > n_mad * sigma
    return out


def robust_mean_std(values, n_mad: float = 8.0) -> dict:
    """Plain and MAD-outlier-trimmed mean/std of ``values``, side by side.

    Removes points more than ``n_mad`` robust-sigma from the median
    (computed directly on ``values``, not on some proxy like chi2) before
    computing the trimmed mean/std -- so a handful of catastrophic rows no
    longer dominate the reported scatter for the quantity itself, not just
    for whatever separately-computed diagnostic (e.g. chi2) happened to
    flag them.

    Returns
    -------
    dict with keys:
        n              -- number of finite input values
        mean, std      -- plain statistics over all finite values
        median, mad_sigma -- robust center/scale (see `median_mad`)
        n_outliers     -- how many finite values were flagged
        trimmed_mean, trimmed_std, n_trimmed -- statistics after dropping
            outliers (equal to `mean`/`std`/`n` when nothing was flagged)
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    median=float("nan"), mad_sigma=0.0, n_outliers=0,
                    trimmed_mean=float("nan"), trimmed_std=float("nan"), n_trimmed=0)
    med, sigma, n = median_mad(arr)
    outlier = mad_outlier_mask(arr, n_mad=n_mad, log=False)
    kept = arr[~outlier]
    if kept.size == 0:
        kept = arr  # every point flagged (degenerate case) -- fall back to all of them
    return dict(
        n=int(arr.size), mean=float(np.mean(arr)), std=float(np.std(arr)),
        median=med, mad_sigma=sigma, n_outliers=int(outlier.sum()),
        trimmed_mean=float(np.mean(kept)), trimmed_std=float(np.std(kept)),
        n_trimmed=int(kept.size),
    )
