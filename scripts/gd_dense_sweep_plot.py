#!/usr/bin/env python3
"""Diagnostics for the dense along-slit sweep (gd_dense_sweep.py):
look for systematic patterns in the rectification-interpolation residuals
that might explain the -50 to -65 ppm bias found in gd_rectify_retrieve.py
(KEYSTONE_SMILE_BIAS_PLAN.md Sec. 9l).

Produces plots/gd_dense_sweep_fpa2.png, 2x3 panels:
1. CO2 bias vs row, dense, native vs rectified, order=0 (no dispersion
   correction).
2. CO2 bias vs row, dense, native vs rectified, order=2.
3. Where each pipeline diverges / goes off-detector, by row.
4. Rectified-pipeline post-fit spectral residual, row x wavenumber heatmap,
   order=0 -- same shared wn_grid as order=2 (no interpolation needed for
   this comparison, unlike a native-vs-rectified one).
5. Rectified-pipeline post-fit spectral residual, row x wavenumber heatmap,
   order=2 -- the main "systematic pattern" diagnostic.
6. Mean |residual| per row vs rows_crossed(row), order=2 -- does
   interpolation error scale with keystone severity, or is it row-independent?

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_dense_sweep_plot.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geocarb_gert.gd_polynomials import rows_crossed

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_good(b) -> bool:
    """True for a genuinely converged retrieval.

    Before 2026-07-23's gert fix, GERTRetrieval's dx_norm convergence
    criterion could report converged=True with a non-finite chi2 (a state
    that had already blown up); this needed a manual chi2-sanity filter here
    to compensate. gert now guarantees converged=True implies a finite chi2
    (see gert/retrieval.py's divergence check), so `_conv` and `_diverged`
    alone are sufficient -- a converged-but-high-chi2 row (common for
    order=0, which has no dispersion to absorb geometric distortion) is a
    real, if poor, result now, not a symptom to filter out.
    """
    return (b is not None and not b.get("_diverged") and b.get("_conv")
            and np.isfinite(b.get("co2", np.nan)))


def _plot_bias_vs_row(ax, out, rows, order):
    for pipeline, color in (("native", "tab:blue"), ("rectified", "tab:orange")):
        ks, biases = [], []
        for k in rows:
            v = out[(pipeline, order, int(k))]
            b = v["bias"]
            if _is_good(b):
                ks.append(k); biases.append(b["co2"])
        ax.plot(ks, biases, ".", ms=2, label=f"{pipeline} (order={order})", color=color, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    disp_note = "no dispersion correction" if order == 0 else "dispersion order 2"
    ax.set_title(f"CO2 bias vs row ({disp_note})")
    ax.set_xlabel("row"); ax.set_ylabel("CO2 bias [ppm]")
    ax.legend(fontsize=8)


def _plot_residual_heatmap(ax, fig, out, rows, wn_grid, order):
    n_wn = len(wn_grid)
    heat = np.full((len(rows), n_wn), np.nan)
    for i, k in enumerate(rows):
        v = out[("rectified", order, int(k))]
        resid, nu = v["residual"], v["nu"]
        if resid is None or not _is_good(v["bias"]):
            continue
        # residual/nu are in ascending-wavelength (descending wn) order --
        # reverse back to ascending-wn to match wn_grid's own ordering.
        resid_wn_order = resid[::-1]
        nu_wn_order = nu[::-1]
        idx = np.searchsorted(wn_grid, nu_wn_order)
        heat[i, idx] = resid_wn_order
    vmax = np.nanpercentile(np.abs(heat), 98)
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[wn_grid[0], wn_grid[-1], rows[-1], rows[0]])
    disp_note = "no dispersion correction" if order == 0 else "dispersion order 2"
    ax.set_title(f"Rectified-pipeline post-fit residual ({disp_note})\n"
                f"row x wavenumber (color range: +/-{vmax:.3g})")
    ax.set_xlabel("wavenumber [cm$^{-1}$]"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, shrink=0.8, label="y_meas - y_ret")


def main() -> int:
    with open(REPO_ROOT / "results" / "gd_dense_sweep.pkl", "rb") as f:
        d = pickle.load(f)
    out, rows, wn_grid, FPA = d["out"], d["rows"], d["wn_grid"], d["FPA"]
    rows = np.array(rows)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    _plot_bias_vs_row(axes[0, 0], out, rows, order=0)
    _plot_bias_vs_row(axes[0, 1], out, rows, order=2)

    # -- Panel: divergence / off-detector / stalled-at-max-iter map --
    # (the pre-2026-07-23-fix "falsely converged" category -- conv=True with
    # a non-finite chi2 -- is gone: gert now guarantees that can't happen.)
    ax = axes[0, 2]
    for order, color, yoff in ((0, "tab:red", 0.3), (2, "tab:purple", 0.15)):
        div_rows = [k for k in rows if (b := out[("rectified", order, int(k))]["bias"]) is not None and b.get("_diverged")]
        ax.plot(div_rows, [yoff] * len(div_rows), "x", color=color, ms=4,
               label=f"rectified order={order} diverged (n={len(div_rows)})")
    for order, color, yoff in ((0, "tab:brown", -0.15), (2, "tab:pink", -0.3)):
        stalled = [k for k in rows if (b := out[("rectified", order, int(k))]["bias"]) is not None
                  and not b.get("_diverged") and not b.get("_conv")]
        ax.plot(stalled, [yoff] * len(stalled), "+", color=color, ms=5,
               label=f"rectified order={order} hit max_iter, not converged (n={len(stalled)})")
    offdet_rows = [k for k in rows if out[("rectified", 2, int(k))]["bias"] is None]
    ax.plot(offdet_rows, [0.0] * len(offdet_rows), "o", color="gray", ms=3,
           label=f"rectified off-detector (n={len(offdet_rows)})")
    ax.set_title("Where the rectified pipeline fails\n(native pipeline: converges cleanly at every row, either order)")
    ax.set_xlabel("row"); ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper center")

    _plot_residual_heatmap(axes[1, 0], fig, out, rows, wn_grid, order=0)
    _plot_residual_heatmap(axes[1, 1], fig, out, rows, wn_grid, order=2)

    # -- Panel: mean |residual| per row vs rows_crossed (order=2) --
    ax = axes[1, 2]
    rc = rows_crossed(FPA, rows)
    for pipeline, color in (("native", "tab:blue"), ("rectified", "tab:orange")):
        mean_abs = []
        for k in rows:
            v = out[(pipeline, 2, int(k))]
            resid = v["residual"]
            good = resid is not None and _is_good(v["bias"])
            mean_abs.append(np.abs(resid).mean() if good else np.nan)
        ax.plot(rc, mean_abs, ".", ms=2, label=pipeline, color=color, alpha=0.6)
    ax.set_title("Mean |post-fit residual| vs local keystone (rows_crossed)\n(order=2)")
    ax.set_xlabel("rows_crossed(row)"); ax.set_ylabel("mean |residual|")
    ax.legend(fontsize=8)

    fig.suptitle(f"Dense along-slit sweep -- FPA{FPA}, uniform desert scene, {len(rows)} rows", fontsize=13)
    fig.tight_layout()
    out_path = REPO_ROOT / "plots" / "gd_dense_sweep_fpa2.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
