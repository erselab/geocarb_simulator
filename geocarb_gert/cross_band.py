"""Cross-band ground co-registration: pairing rows of two GeoCarb bands (FPAs)
that see the same true along-slit position, for a multi-band joint retrieval.

See KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11 for the full plan and rationale --
in particular Sec. 11b (why real slit angle `s` [deg] is the correct shared
coordinate, confirmed by the ground-test calibration methodology: co-aligned
lasers hitting one shared point on the scanning mirror, so `s` means the
same physical direction in every FPA's own fitted polynomial) and Sec. 11c
(why nearest-real-row pairing is preferred as a first cut over resampling
either band onto a shared grid -- gd_render.rectify()'s interpolation is the
dominant bias source found everywhere else in this study, Sec. 9l/9m).

Deliberately keyed on real slit angle `s`, never on either band's own
normalized `eta = s / s_max(fpa)` -- `s_max` differs slightly per FPA
(2.2899/2.2820/2.2970/2.2858 deg for FPA0-3), so equal `eta` in two bands is
*not* the same true angle: the resulting mismatch is real and systematic,
growing toward the slit edges (confirmed 2026-07-27 -- up to ~1.6 rows at
the edge for the FPA0+FPA2 pair Sec. 11 proposes, ~3.4 rows for the worst
pair FPA1+FPA2), landing exactly where keystone row-crossing (Sec. 9w) is
already largest. Using real `s` throughout avoids that entirely.
"""
from __future__ import annotations

import numpy as np

from .gd_polynomials import N_PX, xy_to_wavelength_slit
from .gd_render import s_max as _s_max  # noqa: F401  (re-exported for convenience)


def real_s_of_row(fpa: int, rows=None, col: float | None = None) -> np.ndarray:
    """Real slit angle [deg] at each of `fpa`'s rows, evaluated at a fixed
    representative column -- the same convention already used elsewhere in
    this codebase for `x_km_of_row` (e.g. `gd_band_stress_test.py`).

    Parameters
    ----------
    fpa : int
    rows : array-like, optional
        Row indices to evaluate. Default: every row, 0..N_PX-1.
    col : float, optional
        Column to evaluate at. Default: detector centre (N_PX/2 - 0.5).

    Returns
    -------
    ndarray, real slit angle [deg], same shape as `rows`.
    """
    if rows is None:
        rows = np.arange(N_PX, dtype=float)
    rows = np.asarray(rows, dtype=float)
    if col is None:
        col = N_PX / 2 - 0.5
    _, s = xy_to_wavelength_slit(fpa, np.full_like(rows, col), rows)
    return s


def nearest_row_pairing(fpa_a: int, fpa_b: int, rows_a=None) -> dict:
    """For each row of `fpa_a`, find the real row of `fpa_b` closest in true
    slit angle `s` -- the co-registration mechanism from
    KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11c/11d item 3 (nearest-native-row, the
    simpler first cut ahead of PSF-area-weighted combination).

    No interpolation of either band's spectrum -- this only decides *which*
    already-real row of `fpa_b` to pair with each already-real row of
    `fpa_a`. Rows of `fpa_a` whose real angle falls outside `fpa_b`'s own
    covered range are dropped, not extrapolated.

    Parameters
    ----------
    fpa_a, fpa_b : int
    rows_a : array-like, optional
        Rows of `fpa_a` to pair. Default: every row, 0..N_PX-1.

    Returns
    -------
    dict with:
        rows_a : ndarray[int]          -- input rows of fpa_a (valid only)
        rows_b : ndarray[int]          -- matched rows of fpa_b
        s_a, s_b : ndarray[float]      -- each row's own real slit angle [deg]
        mismatch_deg : ndarray[float]  -- s_b - s_a, real angle residual
        mismatch_rows : ndarray[float] -- mismatch_deg / fpa_b's mean row spacing
        n_dropped : int                -- rows_a outside fpa_b's covered range
    """
    if rows_a is None:
        rows_a = np.arange(N_PX, dtype=float)
    rows_a = np.asarray(rows_a, dtype=float)

    s_a = real_s_of_row(fpa_a, rows_a)
    rows_b_all = np.arange(N_PX, dtype=float)
    s_b_all = real_s_of_row(fpa_b, rows_b_all)

    lo, hi = min(s_b_all.min(), s_b_all.max()), max(s_b_all.min(), s_b_all.max())
    in_range = (s_a >= lo) & (s_a <= hi)
    n_dropped = int((~in_range).sum())

    # Brute-force nearest-neighbour: |difference| matrix, at most N_PX x N_PX
    # (~1e6 entries, trivial) -- doesn't assume s_b_all is monotonic in row
    # (it should be globally, but this doesn't rely on it).
    diff = np.abs(s_a[in_range, None] - s_b_all[None, :])
    best = np.argmin(diff, axis=1)

    rows_a_valid = rows_a[in_range].astype(int)
    rows_b_matched = rows_b_all[best].astype(int)
    s_a_valid = s_a[in_range]
    s_b_matched = s_b_all[best]
    mismatch_deg = s_b_matched - s_a_valid

    row_spacing_b = float(np.abs(np.diff(s_b_all)).mean())
    mismatch_rows = mismatch_deg / row_spacing_b

    return dict(rows_a=rows_a_valid, rows_b=rows_b_matched,
               s_a=s_a_valid, s_b=s_b_matched,
               mismatch_deg=mismatch_deg, mismatch_rows=mismatch_rows,
               n_dropped=n_dropped)


def fpas_tag(fpas) -> str:
    """Canonical filename tag for an ordered list of >=2 FPA indices, e.g.
    [0, 2] -> "fpa0_fpa2", [0, 1, 2, 3] -> "fpa0_fpa1_fpa2_fpa3" -- the
    naming convention shared by gd_joint_band_test.py and its two plotting
    scripts, so a run's output filenames and a plot's expected input
    filenames can never drift apart independently."""
    return "_".join(f"fpa{fpa}" for fpa in fpas)


def nearest_row_pairing_multi(fpas, rows_ref=None) -> dict:
    """Generalizes nearest_row_pairing to >=2 bands for a multi-band joint
    retrieval: pairs every band in `fpas[1:]` to `fpas[0]` (the along-slit
    position reference) independently via nearest_row_pairing, then
    restricts to the `fpas[0]` rows that fall within *every* other band's
    covered range (the N-way intersection, not just one pairwise range).

    Each individual pairing is independently bounded (<=0.5-row mismatch,
    confirmed 2026-07-27 for the FPA0+FPA2 pair -- see
    KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11f), so error does not compound as
    more bands are added -- only the valid along-slit coverage can shrink,
    since it's an intersection across all bands' ranges.

    Parameters
    ----------
    fpas : sequence of int, len >= 2
        Ordered FPA indices; fpas[0] is the along-slit position reference.
    rows_ref : array-like, optional
        Rows of fpas[0] to pair. Default: every row, 0..N_PX-1.

    Returns
    -------
    dict with:
        fpas : list[int]                   -- as given
        rows : dict[int, ndarray[int]]     -- per-FPA matched row, same
                                               length/order for every FPA
        s : dict[int, ndarray[float]]      -- per-FPA real slit angle [deg]
        mismatch_deg : dict[int, ndarray]  -- vs. reference, 0 for fpas[0]
        mismatch_rows : dict[int, ndarray] -- vs. reference, 0 for fpas[0]
        n_dropped : int                    -- reference rows dropped by the
                                               intersection
    """
    fpas = list(fpas)
    ref = fpas[0]
    if rows_ref is None:
        rows_ref = np.arange(N_PX, dtype=float)
    rows_ref = np.asarray(rows_ref, dtype=float)

    pairings = {fpa: nearest_row_pairing(ref, fpa, rows_ref) for fpa in fpas[1:]}

    valid_int = set(rows_ref.astype(int).tolist())
    for p in pairings.values():
        valid_int &= set(p["rows_a"].tolist())
    valid_rows_ref = np.array(sorted(valid_int), dtype=int)
    n_dropped = int(len(rows_ref) - len(valid_rows_ref))

    rows = {ref: valid_rows_ref}
    s = {ref: real_s_of_row(ref, valid_rows_ref.astype(float))}
    mismatch_deg = {ref: np.zeros(len(valid_rows_ref))}
    mismatch_rows = {ref: np.zeros(len(valid_rows_ref))}

    for fpa, p in pairings.items():
        idx_of = {int(ra): i for i, ra in enumerate(p["rows_a"])}
        sel = np.array([idx_of[r] for r in valid_rows_ref], dtype=int)
        rows[fpa] = p["rows_b"][sel]
        s[fpa] = p["s_b"][sel]
        mismatch_deg[fpa] = p["mismatch_deg"][sel]
        mismatch_rows[fpa] = p["mismatch_rows"][sel]

    return dict(fpas=fpas, rows=rows, s=s, mismatch_deg=mismatch_deg,
               mismatch_rows=mismatch_rows, n_dropped=n_dropped)
