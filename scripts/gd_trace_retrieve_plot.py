#!/usr/bin/env python3
"""Compare the trace-and-select sweep (`gd_trace_retrieve_sweep.py`,
`JOINT_ROW_INVERSION_PLAN.md` §0) against native/undistorted across the
whole FPA2 realistic-scene slit, not just the two §11p hot-spot rows the
original go/no-go test checked.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_trace_retrieve_plot.py \\
        --tolerance-km 0.5
Output: plots/gd_trace_vs_native_fpa2.png
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geocarb_gert import chi2_outlier_mask, robust_mean_std

REPO_ROOT = Path(__file__).resolve().parent.parent


def _native_undistorted_series(pipeline: str, quantity: str = "co2"):
    with open(REPO_ROOT / "results" / "gd_joint_fpa2.pkl", "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    rows_, vals_, chi2_ = [], [], []
    for (pl, order, rt), v in out.items():
        if pl != pipeline or v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        if quantity not in v:
            continue
        rows_.append(rt[0]); vals_.append(v[quantity]); chi2_.append(v["_chi2"])
    rows_ = np.array(rows_); vals_ = np.array(vals_); chi2_ = np.array(chi2_)
    outlier = chi2_outlier_mask(chi2_, n_mad=8.0)
    order = np.argsort(rows_[~outlier])
    return rows_[~outlier][order], vals_[~outlier][order]


def _trace_series(tolerance_km: float, quantity: str = "co2"):
    path = REPO_ROOT / "results" / f"gd_trace_fpa2_tol{tolerance_km:g}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    rows_, vals_, chi2_, npix_ = [], [], [], []
    for row, entry in out.items():
        nl = entry["nl"]
        if not nl.get("_conv") or nl.get("_diverged") or quantity not in nl:
            continue
        rows_.append(row); vals_.append(nl[quantity]); chi2_.append(nl["_chi2"])
        npix_.append(entry["n_pixels"])
    rows_ = np.array(rows_); vals_ = np.array(vals_); chi2_ = np.array(chi2_); npix_ = np.array(npix_)
    outlier = chi2_outlier_mask(chi2_, n_mad=8.0)
    order = np.argsort(rows_[~outlier])
    return rows_[~outlier][order], vals_[~outlier][order], npix_[~outlier][order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance-km", type=float, default=0.5)
    args = ap.parse_args()

    r_n, v_n = _native_undistorted_series("native")
    r_u, v_u = _native_undistorted_series("undistorted")
    r_t, v_t, n_t = _trace_series(args.tolerance_km)

    stats_n = robust_mean_std(v_n, n_mad=8.0)
    stats_u = robust_mean_std(v_u, n_mad=8.0)
    stats_t = robust_mean_std(v_t, n_mad=8.0)

    print(f"{'pipeline':14s} {'n':>6s} {'mean':>10s} {'std':>10s} {'median':>10s} {'MAD-sig':>10s}")
    for label, s in [("native", stats_n), ("undistorted", stats_u),
                     (f"trace(tol={args.tolerance_km:g}km)", stats_t)]:
        print(f"{label:14s} {s['n']:6d} {s['mean']:+10.4g} {s['std']:10.4g} "
             f"{s['median']:+10.4g} {s['mad_sigma']:10.4g}")

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.3]})

    ax = axes[0]
    ax.plot(r_n, v_n, ".", ms=2, color="tab:red", alpha=0.5, label=f"native (mean={stats_n['mean']:+.2f})")
    ax.plot(r_u, v_u, ".", ms=2, color="tab:green", alpha=0.5, label=f"undistorted (mean={stats_u['mean']:+.2f})")
    ax.plot(r_t, v_t, ".", ms=3, color="tab:blue", alpha=0.6,
           label=f"trace-and-select, tol={args.tolerance_km:g}km (mean={stats_t['mean']:+.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_title("FPA2 realistic scene, no noise: CO2 bias vs row, all three pipelines")
    ax.legend(fontsize=8, markerscale=4, loc="upper right")

    ax = axes[1]
    ax.plot(r_n, v_n, ".", ms=3, color="tab:red", alpha=0.6, label="native")
    ax.plot(r_t, v_t, ".", ms=4, color="tab:blue", alpha=0.7, label=f"trace-and-select (tol={args.tolerance_km:g}km)")
    ax.axvspan(895, 925, color="gold", alpha=0.15, label="§11p hot spot (row~910)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_title("native vs trace-and-select only (undistorted omitted for clarity)")
    ax.legend(fontsize=8, markerscale=3, loc="upper right")

    ax = axes[2]
    ax.plot(r_t, n_t, "-", color="tab:blue", lw=1.0)
    ax.set_ylabel("pixels\nincluded")
    ax.set_xlabel("detector row")
    ax.set_title("trace-and-select sample count per row", fontsize=9)

    fig.tight_layout()
    plots_dir = REPO_ROOT / "plots" / "trace_and_select"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_trace_vs_native_fpa2_tol{args.tolerance_km:g}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
