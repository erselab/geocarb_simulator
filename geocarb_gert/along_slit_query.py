"""Project a `along_slit_state.StackedState` onto a caller-supplied grid,
returning an estimate AND its full covariance -- the postprocessing step,
kept separate from `along_slit_state` so the information-bearing state
itself never has to know what any particular caller wants to plot.

The state's own covariance (`StackedState.cov`) is exactly block-diagonal
-- real, not approximate, since each window was solved independently. This
module never adds a fabricated cross-window term to it; every number in
the returned `cov_q` comes from one linear operator `W` applied to that
real covariance (`cov_q = W @ stacked.cov @ W.T`), so within-window
smearing (adjacent query points sharing the same bins) and cross-window
correlation (an overlap-band query point whose weight row spans two
windows) both fall out of the SAME propagation, with no separate
diagonal-only special case.

The one deliberately non-propagated addition is `disagreement_inflation`:
where two windows both cover a query point, their own independent
estimates there are expected to differ by more than their stated
uncertainties alone predict, because of the documented window-boundary
overshoot/undershoot bias (`along_slit_merge`'s own former docstring, now
here: the `exponential` prior's stationary form has no edge-weakening, so
a window's trailing edge systematically over-extrapolates and the next
window's leading edge under-commits, with opposite sign). That is a bias
effect, not sampling noise `W @ cov @ W.T` can represent, so it is folded
in as an explicit diagonal addition at exactly the query points where
more than one window contributes -- clearly separable from the propagated
part by construction (see `n_covering` in the return value).
"""
from __future__ import annotations

import numpy as np

from .along_slit_state import StackedState
from .joint_state import ParamSpec, StateSpec


def _window_weight_row(stacked: StackedState, window_id: int, eta_q: float,
                       state_interp: str) -> tuple[np.ndarray, np.ndarray]:
    """This window's own interpolation weights (over its own bins only) at
    one query eta -- `(idx, weights)` where `idx` indexes into the full
    stacked vector. Reuses `StateSpec.interp_weights`, which only needs
    `positions` (not packing order), so a throwaway single-row `StateSpec`
    is safe here -- unlike the real covariance slice, which is not (see
    `along_slit_state._cov_block`).
    """
    mask = stacked.window_id == window_id
    idx = np.where(mask)[0]
    positions = stacked.eta[mask]
    p = ParamSpec(name="x", positions=positions, prior=positions, sigma=1.0,
                 corr_length=1.0, free=True, kind="scale")
    w = StateSpec([p]).interp_weights(np.array([eta_q]), "x", state_interp)[0]
    return idx, w


def query_state(stacked: StackedState, row_query, eta_query, state_interp: str = "linear",
                disagreement_inflation: bool = True):
    """Estimate + full covariance at an arbitrary grid.

    `row_query` and `eta_query` are parallel arrays (same length as each
    other, any length): `row_query[g]` decides which window(s) cover query
    point `g` (a window covers it iff `row_lo <= row_query[g] <= row_hi`),
    `eta_query[g]` is the actual interpolation abscissa. Kept as two
    separate arrays rather than one, so this module stays free of any
    FPA/geometry dependency -- the caller (who already has both, e.g. the
    standard 1024-row grid's own `eta_rows`) supplies them; nothing here
    computes eta from row or vice versa.

    Returns `(values_q, cov_q, n_covering)`:
    - `values_q`: `(Q,)`, nan where no window covers that point.
    - `cov_q`: `(Q, Q)`, the FULL covariance -- `.diagonal()` for just the
      per-point variance, or use the whole matrix for a further linear
      combination (e.g. a segment average) without re-deriving it.
    - `n_covering`: `(Q,)` int, how many windows contributed to each
      point -- 0 (gap, nan), 1 (ordinary), or >1 (overlap-band blend).
    """
    row_query = np.atleast_1d(np.asarray(row_query, dtype=float))
    eta_query = np.atleast_1d(np.asarray(eta_query, dtype=float))
    if row_query.shape != eta_query.shape:
        raise ValueError(f"row_query {row_query.shape} and eta_query {eta_query.shape} "
                         f"must have the same shape")
    Q = eta_query.size
    M = stacked.eta.size
    window_ids = np.unique(stacked.window_id) if M else np.zeros(0, dtype=int)

    W = np.zeros((Q, M))
    excess = np.zeros(Q)
    n_covering = np.zeros(Q, dtype=int)

    lo_by_window = {wi: int(stacked.row_lo[stacked.window_id == wi][0]) for wi in window_ids}
    hi_by_window = {wi: int(stacked.row_hi[stacked.window_id == wi][0]) for wi in window_ids}

    for g in range(Q):
        covering = [wi for wi in window_ids
                   if lo_by_window[wi] <= row_query[g] <= hi_by_window[wi]]
        n_covering[g] = len(covering)
        if not covering:
            continue

        candidates = []  # (idx, weights, var)
        for wi in covering:
            idx, w = _window_weight_row(stacked, wi, eta_query[g], state_interp)
            cov_block = stacked.cov[np.ix_(idx, idx)]
            var = float(w @ cov_block @ w)
            candidates.append((idx, w, max(var, 0.0)))

        if len(candidates) == 1:
            idx, w, _ = candidates[0]
            W[g, idx] = w
            continue

        inv = np.array([1.0 / max(v, 1e-30) for _, _, v in candidates])
        alphas = inv / inv.sum()
        row = np.zeros(M)
        for a, (idx, w, _) in zip(alphas, candidates):
            row[idx] += a * w
        W[g, :] = row

        if disagreement_inflation:
            cand_vals = np.array([w @ stacked.values[idx] for idx, w, _ in candidates])
            vmerged = float(alphas @ cand_vals)
            excess[g] = float(np.sum(alphas * (cand_vals - vmerged) ** 2))

    values_q = np.where(n_covering > 0, W @ stacked.values, np.nan)
    cov_q = W @ stacked.cov @ W.T
    if disagreement_inflation and excess.any():
        cov_q = cov_q + np.diag(excess)
    gap = n_covering == 0
    cov_q[gap, :] = np.nan
    cov_q[:, gap] = np.nan

    return values_q, cov_q, n_covering
