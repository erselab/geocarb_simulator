"""Assemble the real, information-bearing along-slit state from a set of
independently-solved windows -- no interpolation, no manufactured points.

Replaces the row-interpolating approach this module used to take (stitch
onto a fixed 1024-row grid, keep only the diagonal variance at each row).
That representation silently implied up to 1024 independent measurements
when the actual retrieved degrees of freedom are the sum of each window's
own G bins -- usually far fewer, and never independent between adjacent
rows inside one window (they are all linear combinations of the same G
bin values). Concretely, for a `co2_ppm`-only sweep at G~10-49 per window
over 58 windows, the previous 1024-point array over-reported the DOF by
roughly 5-20x.

`stack_windows_along_slit` builds the vector that actually has that many
degrees of freedom: every window's own retrieved bins, concatenated, with
their REAL joint covariance -- which is exactly block-diagonal, since
each window was solved fully independently (no shared data, no shared
prior term crosses a window boundary). No cross-window covariance is
fabricated; where two windows' rows genuinely interact (an overlap band,
`build_window_tiles(..., overlap=N)`) that interaction is handled by
projecting this state onto a query grid (`along_slit_query.query_state`),
never by writing something into `cov` here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StackedState:
    """The concatenation of one named row (e.g. `co2_ppm`) across every
    window that retrieved it, with the real block-diagonal joint
    covariance -- the information-bearing state, at its own native
    resolution. Length `M` is the sum of each contributing window's own
    `n` for this row (its G for a free row, however many positions a
    frozen one was given), NOT tied to any detector-row count.
    """

    eta: np.ndarray          # (M,) each entry's own bin-centre eta
    values: np.ndarray       # (M,) physical-unit value at that bin
    cov: np.ndarray          # (M, M) block-diagonal joint covariance, physical units
    row_lo: np.ndarray       # (M,) int -- owning window's row_lo (its solved row range)
    row_hi: np.ndarray       # (M,) int -- owning window's row_hi
    window_id: np.ndarray    # (M,) int -- index into the `windows` list this entry came from


def _cov_block(window: dict, solve: str, name: str) -> np.ndarray:
    """One window's own ``(n, n)`` physical-units posterior covariance for
    row `name`, read directly out of its SAVED ``cov``/``slices`` rather
    than reconstructed via a throwaway single-row `StateSpec` -- a
    reconstructed spec's own `slices()` always starts at 0, which silently
    disagrees with the real packed-vector offset whenever the original
    solve had more than one free row before this one (e.g. `co2_ppm` +
    `p_surface_hpa` in the `co2p` experiments). A frozen row has no
    posterior uncertainty by construction -- zero matrix, not a raise.
    """
    rec = window[solve]["params"][name]
    n = len(np.atleast_1d(rec["positions"]))
    if not rec["free"]:
        return np.zeros((n, n))
    start, stop = window[solve]["slices"][name]
    cov_scale = np.asarray(window[solve]["cov"], dtype=float)[start:stop, start:stop]
    if rec["kind"] == "scale":
        prior = np.asarray(rec["prior"], dtype=float)
        return cov_scale * np.outer(prior, prior)
    return cov_scale


def stack_windows_along_slit(windows, name: str, solve: str = "hires") -> StackedState:
    """Concatenate row `name` across every window that has it, block-
    diagonal covariance and all. Windows lacking `name` entirely (e.g. a
    solve that only freed `co2_ppm`, queried for `p_surface_hpa`) are
    skipped; a window where `name` was frozen still contributes its prior
    value with a zero-variance block, matching how a frozen row already
    behaves everywhere else in this codebase (present, just certain).
    """
    eta_parts, val_parts, cov_blocks = [], [], []
    lo_parts, hi_parts, wid_parts = [], [], []

    for wi, w in enumerate(windows):
        if solve not in w or name not in w[solve].get("params", {}):
            continue
        rec = w[solve]["params"][name]
        positions = np.atleast_1d(np.asarray(rec["positions"], dtype=float))
        values = np.atleast_1d(np.asarray(rec["values"], dtype=float))
        n = positions.size
        eta_parts.append(positions)
        val_parts.append(values)
        cov_blocks.append(_cov_block(w, solve, name))
        lo_parts.append(np.full(n, int(w["row_lo"])))
        hi_parts.append(np.full(n, int(w["row_hi"])))
        wid_parts.append(np.full(n, wi))

    if not eta_parts:
        return StackedState(eta=np.zeros(0), values=np.zeros(0), cov=np.zeros((0, 0)),
                            row_lo=np.zeros(0, dtype=int), row_hi=np.zeros(0, dtype=int),
                            window_id=np.zeros(0, dtype=int))

    eta = np.concatenate(eta_parts)
    values = np.concatenate(val_parts)
    M = eta.size
    cov = np.zeros((M, M))
    off = 0
    for block in cov_blocks:
        k = block.shape[0]
        cov[off:off + k, off:off + k] = block
        off += k

    return StackedState(eta=eta, values=values, cov=cov,
                        row_lo=np.concatenate(lo_parts), row_hi=np.concatenate(hi_parts),
                        window_id=np.concatenate(wid_parts))
