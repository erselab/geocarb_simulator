"""Shared-eta window tiling for a multi-band joint block (docs/MULTIBAND_PLAN.md
Sec.4.3, decided 2026-09-20: the tiling must respect BOTH bands' keystone maxima).

Single-band tiling (``gd_joint_block_retrieve.build_window_tiles``) picks each
window's half-width in ROWS as ``window_scale * 2.2 * rows_crossed`` evaluated at
the window's CENTRE row. With two bands that is not enough: keystone differs by
band and by position (0 at FPA2's null row, ~10 rows at the slit ends) and the
maximum inside a window is not at its centre. Here the tiling lives on the shared
slit-image ``eta`` axis (:func:`geocarb_gert.gd_polynomials.eta_of_s`, Sec.33), and
a window's half-width is ``window_scale * 2.2 * k`` where ``k`` is the LARGEST
keystone crossing -- expressed in eta -- of ANY band over the rows inside the
candidate window (``crossing="max"``, the multi-band default; ``"center"``
reproduces the single-band rule for regression tests).

Each band's rows for a window are the rows whose nominal eta (centre column,
matching ``_eta_of(fpa, 512, rows)`` used for anchors) lies in the window's eta
interval, so every row belongs to exactly one window (before ``overlap``). The
window's STATE and ANCHORS must additionally cover the eta extent of every pixel
of those rows across ALL columns -- :func:`window_eta_extent` /
:func:`joint_anchor_eta_range` give that, unioned over bands.

The real GD mapping polynomials are used throughout (keystone always on; never
zero -- user, 2026-09-20).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .gd_polynomials import N_PX, eta_of_s, rows_crossed, xy_to_wavelength_slit

_KEY_COL_LO, _KEY_COL_HI = 4.0, N_PX - 5.0     # same columns `rows_crossed` uses
_CENTER_COL = 512.0                            # nominal-eta column (matches the driver's anchors)


def _eta(fpa: int, cols, rows) -> np.ndarray:
    _, s = xy_to_wavelength_slit(fpa, np.asarray(cols, dtype=float), np.asarray(rows, dtype=float))
    return eta_of_s(fpa, s)


@lru_cache(maxsize=8)
def band_tables(fpa: int) -> dict:
    """Per-row geometry of one band: nominal eta (centre column), the keystone
    crossing in eta (|eta(col 1019) - eta(col 4)|, the ``rows_crossed`` columns),
    and the eta extent of the whole row over every column."""
    rows = np.arange(N_PX, dtype=float)
    eta_c = _eta(fpa, np.full(N_PX, _CENTER_COL), rows)
    if not np.all(np.diff(eta_c) > 0):
        raise ValueError(f"FPA{fpa}: nominal row eta is not strictly increasing with row")
    k = np.abs(_eta(fpa, np.full(N_PX, _KEY_COL_HI), rows) - _eta(fpa, np.full(N_PX, _KEY_COL_LO), rows))
    cols = np.arange(N_PX, dtype=float)
    lo = np.empty(N_PX)
    hi = np.empty(N_PX)
    for i in range(N_PX):
        e = _eta(fpa, cols, np.full(N_PX, float(i)))
        lo[i], hi[i] = e.min(), e.max()
    return dict(eta_c=eta_c, k_eta=k, eta_lo=lo, eta_hi=hi,
                spacing=float(np.mean(np.diff(eta_c))))


@dataclass
class MBWindow:
    """One multi-band window: an eta interval and each band's own row range."""
    eta_lo: float
    eta_hi: float
    rows: dict = field(default_factory=dict)       # {fpa: (row_lo, row_hi)}  (after `overlap`)
    kmax_eta: dict = field(default_factory=dict)   # {fpa: largest keystone crossing in this window, eta}

    @property
    def width_eta(self) -> float:
        return self.eta_hi - self.eta_lo


def build_window_tiles_multiband(fpas, min_window: int = 4, window_scale: float = 1.0,
                                 overlap: int = 0, crossing: str = "max") -> list:
    """Tile the shared eta axis for ``fpas`` -- see the module docstring.

    Tiles are laid out in ROWS of the reference band ``fpas[0]`` (so that ONE band with
    ``crossing="center"`` reproduces ``gd_joint_block_retrieve.build_window_tiles``
    exactly); the other bands' rows for each window are then the rows whose nominal
    eta falls in the reference window's eta interval. The half-width rule
    (``r = round(window_scale * 2.2 * k)``, in reference rows, at least ``min_window``)
    uses ``k`` = the largest keystone crossing of ANY band (``rows_crossed``, converted
    from that band's rows to reference rows through the two bands' row spacing in eta).

    Parameters
    ----------
    min_window : int
        Minimum half-width in reference-band rows (as in the single-band tiling).
    window_scale : float
        Multiplier on the ``2.2 * crossing`` half-width rule.
    overlap : int
        Rows of overlap added to each INTERIOR boundary, per band (as single-band).
    crossing : "max" | "center"
        "max": largest crossing of any band over the rows inside the candidate window
        (multi-band default). "center": each band's crossing at the candidate window's
        centre only, then the max over bands (the single-band rule for one band).
    """
    if crossing not in ("max", "center"):
        raise ValueError(f"crossing must be 'max' or 'center', got {crossing!r}")
    fpas = tuple(fpas)
    ref = fpas[0]
    tabs = {f: band_tables(f) for f in fpas}
    tr = tabs[ref]
    lo = max(t["eta_c"][0] for t in tabs.values())      # intersection of the bands' coverage
    hi = min(t["eta_c"][-1] for t in tabs.values())
    row_min = int(np.searchsorted(tr["eta_c"], lo, side="left"))
    row_max = int(np.searchsorted(tr["eta_c"], hi, side="right") - 1)
    k_rows = {f: rows_crossed(f, np.arange(N_PX, dtype=float)) for f in fpas}   # in band f's rows

    def k_ref_rows(a_eta, b_eta):
        """Largest crossing of any band, in reference-band rows, over eta in [a_eta, b_eta]."""
        best = 0.0
        for f, t in tabs.items():
            m = (t["eta_c"] >= a_eta) & (t["eta_c"] <= b_eta)
            if crossing == "max" and m.any():
                kf = float(k_rows[f][m].max())
            else:
                kf = float(k_rows[f][int(np.argmin(np.abs(t["eta_c"] - 0.5 * (a_eta + b_eta))))])
            best = max(best, kf * t["spacing"] / tr["spacing"])
        return best

    ref_tiles = []
    row_start = row_min
    while row_start <= row_max:
        r = min_window
        for _ in range(6):
            end_c = min(row_start + 2 * r, row_max)
            k = k_ref_rows(tr["eta_c"][row_start], tr["eta_c"][end_c])
            r_new = max(min_window, int(round(window_scale * 2.2 * k)))
            if r_new == r:
                break
            r = r_new
        row_end = min(row_start + 2 * r, row_max)
        ref_tiles.append((row_start, row_end))
        row_start = row_end + 1

    tiles = []
    for j, (r0, r1) in enumerate(ref_tiles):
        last = j == len(ref_tiles) - 1
        a = float(tr["eta_c"][r0])
        b = float(tr["eta_c"][r1 + 1]) if not last else float(tr["eta_c"][r1])
        w = MBWindow(eta_lo=a, eta_hi=b)
        for f, t in tabs.items():
            if f == ref:
                lo_r, hi_r = r0, r1
            else:
                m = (t["eta_c"] >= a) & ((t["eta_c"] <= b) if last else (t["eta_c"] < b))
                idx = np.where(m)[0]
                if idx.size == 0:
                    raise ValueError(f"window {j} [{a:.4f},{b:.4f}] has no FPA{f} rows")
                if not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
                    raise ValueError(f"FPA{f} rows in window {j} are not contiguous")
                lo_r, hi_r = int(idx[0]), int(idx[-1])
            if overlap > 0:
                if j > 0:
                    lo_r = max(0, lo_r - overlap)
                if not last:
                    hi_r = min(N_PX - 1, hi_r + overlap)
            w.rows[f] = (lo_r, hi_r)
            m = (t["eta_c"] >= a) & (t["eta_c"] <= b)
            w.kmax_eta[f] = float(k_rows[f][m].max() * t["spacing"]) if m.any() else 0.0
        tiles.append(w)
    return tiles


def window_eta_extent(fpa: int, row_lo: int, row_hi: int, pad: int = 0) -> tuple:
    """(min, max) eta over EVERY column of rows ``[row_lo - pad, row_hi + pad]``
    (clipped to the detector) -- the eta range a window's state and anchors must
    cover for that band, keystone included."""
    t = band_tables(fpa)
    a, b = max(0, row_lo - pad), min(N_PX - 1, row_hi + pad)
    return float(t["eta_lo"][a:b + 1].min()), float(t["eta_hi"][a:b + 1].max())


def eta_to_row(fpa: int, eta) -> np.ndarray:
    """Nearest detector row (centre column) for `eta` (scalar or array) -- the
    inverse of `band_tables(fpa)["eta_c"]`, itself strictly increasing with row
    (checked in `band_tables`). 2026-09-23 (user: "the ability to specify values
    explicitly instead of just pointing to a reference pkl file" -- a geometry
    config's tile eta_lo/eta_hi, hand-typed or exported, can stand in for
    `rows_by_fpa` directly via this, so a config never STRICTLY needs a prior
    run's own row ranges). A nearest-row lookup for window PLACEMENT, same
    spirit as `gd_joint_block_retrieve.py`'s own local `eta_to_row` (x-axis
    placement only) -- never used to reconstruct a physical quantity.
    """
    eta_c = band_tables(fpa)["eta_c"]
    idx = np.searchsorted(eta_c, np.atleast_1d(eta))
    idx = np.clip(idx, 0, len(eta_c) - 1)
    rows = idx if np.ndim(eta) else int(idx[0])
    return rows


def joint_anchor_eta_range(window: MBWindow, pad: int) -> tuple:
    """Union over bands of :func:`window_eta_extent` -- the eta range the SHARED
    anchor grid must span so no band's pixel falls outside it."""
    ext = [window_eta_extent(f, r0, r1, pad) for f, (r0, r1) in window.rows.items()]
    return min(e[0] for e in ext), max(e[1] for e in ext)
