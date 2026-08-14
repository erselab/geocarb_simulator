#!/usr/bin/env python3
"""Full-experiment comparison: trace-and-select vs. all three original
pipelines (native, rectified, undistorted), FPA2 realistic scene, no
noise -- the "vs. the other methods" figure requested after the whole-slit
sweep in `gd_trace_retrieve_sweep.py` (`JOINT_ROW_INVERSION_PLAN.md` §0b).

Rectified's bias here is ~-62 ppm (§9l/§9m's known interpolation
artifact), roughly an order of magnitude past the other three -- plotting
it on the same linear axis as native/undistorted/trace-and-select would
flatten the interesting comparison into a flat line, so it gets its own
panel, same convention as this project's own summary tables always
separating it out as "the dominant, already-understood bias source" (§11l).

Note: rectified's x-axis is `_shared_s_grid`'s own grid index (§11's
co-registration mechanism), not a native detector row number -- for a
single FPA the two are close enough for a qualitative row-axis comparison
here, but not an exact identity.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_trace_all_pipelines_plot.py \\
        --tolerance-km 0.5
Output: plots/gd_trace_all_pipelines_fpa2.png
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


def _joint_series(pipeline: str, quantity: str = "co2"):
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
    rows_, vals_, chi2_ = [], [], []
    for row, entry in out.items():
        nl = entry["nl"]
        if not nl.get("_conv") or nl.get("_diverged") or quantity not in nl:
            continue
        rows_.append(row); vals_.append(nl[quantity]); chi2_.append(nl["_chi2"])
    rows_ = np.array(rows_); vals_ = np.array(vals_); chi2_ = np.array(chi2_)
    outlier = chi2_outlier_mask(chi2_, n_mad=8.0)
    order = np.argsort(rows_[~outlier])
    return rows_[~outlier][order], vals_[~outlier][order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance-km", type=float, default=0.5)
    args = ap.parse_args()

    r_n, v_n = _joint_series("native")
    r_r, v_r = _joint_series("rectified")
    r_u, v_u = _joint_series("undistorted")
    r_t, v_t = _trace_series(args.tolerance_km)

    series = {
        "native": (r_n, v_n, "tab:red"),
        "rectified": (r_r, v_r, "tab:purple"),
        "undistorted": (r_u, v_u, "tab:green"),
        f"trace-and-select (tol={args.tolerance_km:g}km)": (r_t, v_t, "tab:blue"),
    }
    stats = {name: robust_mean_std(v, n_mad=8.0) for name, (r, v, c) in series.items()}

    print(f"{'pipeline':32s} {'n':>6s} {'mean':>10s} {'std':>10s} {'median':>10s} {'MAD-sig':>10s}")
    for name, s in stats.items():
        print(f"{name:32s} {s['n']:6d} {s['mean']:+10.4g} {s['std']:10.4g} "
             f"{s['median']:+10.4g} {s['mad_sigma']:10.4g}")

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.6], hspace=0.35)

    # panel A: the three fine-scale pipelines
    ax = fig.add_subplot(gs[0])
    for name in ["native", "undistorted", f"trace-and-select (tol={args.tolerance_km:g}km)"]:
        r, v, c = series[name]
        ax.plot(r, v, ".", ms=2.5, color=c, alpha=0.6,
               label=f"{name} (mean={stats[name]['mean']:+.2f}, std={stats[name]['std']:.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_title("FPA2 realistic scene, no noise: native / undistorted / trace-and-select", fontsize=11)
    ax.legend(fontsize=8, markerscale=4, loc="upper right")

    # panel B: rectified alone, own scale -- ~10x the others (§9l/§9m/§11l)
    ax = fig.add_subplot(gs[1])
    r, v, c = series["rectified"]
    ax.plot(r, v, ".", ms=2.5, color=c, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_xlabel("detector row (rectified: shared-grid index, ≈ row for a single FPA)")
    ax.set_title(f"rectified, own scale -- mean={stats['rectified']['mean']:+.1f} ppm, "
                f"~{abs(stats['rectified']['mean']/stats['native']['mean']):.0f}x native's "
                f"magnitude (§9l/§9m interpolation bias, already known)", fontsize=10)

    # panel C: robust-stats table, same convention as gd_plot.py's own summary pages
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    header = f"{'pipeline':32s} {'n':>6s} {'mean':>10s} {'std':>10s} {'median':>10s} {'MAD-sig':>10s}"
    lines = [header, "-" * len(header)]
    for name, s in stats.items():
        lines.append(f"{name:32s} {s['n']:6d} {s['mean']:+10.4g} {s['std']:10.4g} "
                     f"{s['median']:+10.4g} {s['mad_sigma']:10.4g}")
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, family="monospace",
           fontsize=9.5, va="top", ha="left")

    fig.suptitle("Trace-and-select vs. all three original pipelines (CO2 bias)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plots_dir = REPO_ROOT / "plots" / "trace_and_select"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_trace_all_pipelines_fpa2_tol{args.tolerance_km:g}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
