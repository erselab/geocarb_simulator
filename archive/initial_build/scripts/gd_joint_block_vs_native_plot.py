#!/usr/bin/env python3
"""Whole-slit head-to-head: joint block (coarse and hi-res, from the
adaptive window sweep, scripts/gd_joint_block_whole_slit_sweep.py)
against native, rectified, and undistorted (results/gd_joint_fpa2.pkl,
already computed for all 1024 rows) -- the "does this actually help,
across the real complexity of the whole slit" comparison, not just the
single hot spot.

Undistorted is not the right ceiling for the joint block the way it is
for native (undistorted removes the real spatial PSF blur entirely, which
the joint block's own forward model always keeps -- see the conversation
this script follows from), so it's shown here for continuity/reference
only, not as a target the joint block should be expected to approach.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_vs_native_plot.py
Output: plots/joint_block/gd_joint_block_vs_native_fpa2.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_retrieve import FPA, _eta_of  # noqa: E402

from geocarb_gert import along_slit_scene as als, chi2_outlier_mask, robust_mean_std  # noqa: E402


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


def _reconstruct_joint_block_bias():
    """Same stitching logic as gd_joint_block_whole_slit_plot.py: rebuild
    the true/coarse/hires CO2 profile at native row resolution from the
    sweep's own saved state vectors, then convert to bias (posterior -
    true), matching native's own retrieved-minus-true convention exactly."""
    with open(REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}.pkl", "rb") as f:
        d = pickle.load(f)
    windows = sorted(d["results"].values(), key=lambda r: r["row_lo"])

    rows_all, bias_coarse, bias_hires = [], [], []
    for w in windows:
        row_lo, row_hi = w["row_lo"], w["row_hi"]
        rows_win = np.arange(row_lo, row_hi + 1)
        eta_win = _eta_of(FPA, np.full(len(rows_win), 512.0), rows_win.astype(float))
        true_win = als.xco2_ppm(eta_win * als.SLIT_HALF_KM)

        bin_centers = w["bin_centers"]
        retrieved_ppm_coarse = w["prior_co2_ppm_bins"] * w["x_coarse"]
        retrieved_ppm_hires = w["prior_co2_ppm_bins"] * w["x_hires"]

        if len(bin_centers) > 1:
            edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
            idx = np.searchsorted(edges, eta_win)
            coarse_win = retrieved_ppm_coarse[idx]
            hires_win = np.interp(eta_win, bin_centers, retrieved_ppm_hires)
        else:
            coarse_win = np.full(len(rows_win), retrieved_ppm_coarse[0])
            hires_win = np.full(len(rows_win), retrieved_ppm_hires[0])

        rows_all.append(rows_win)
        bias_coarse.append(coarse_win - true_win)
        bias_hires.append(hires_win - true_win)

    return np.concatenate(rows_all), np.concatenate(bias_coarse), np.concatenate(bias_hires)


def main() -> int:
    r_n, v_n = _joint_series("native")
    r_r, v_r = _joint_series("rectified")
    r_u, v_u = _joint_series("undistorted")
    r_jb, v_coarse, v_hires = _reconstruct_joint_block_bias()

    series = {
        "native": (r_n, v_n, "tab:red"),
        "undistorted": (r_u, v_u, "tab:green"),
        "joint block, coarse": (r_jb, v_coarse, "tab:orange"),
        "joint block, hi-res": (r_jb, v_hires, "tab:blue"),
    }
    stats = {name: robust_mean_std(v, n_mad=8.0) for name, (r, v, c) in series.items()}
    stats["rectified"] = robust_mean_std(v_r, n_mad=8.0)

    print(f"{'pipeline':24s} {'n':>6s} {'mean':>10s} {'std':>10s} {'median':>10s} {'MAD-sig':>10s}")
    for name in list(series.keys()) + ["rectified"]:
        s = stats[name]
        print(f"{name:24s} {s['n']:6d} {s['mean']:+10.4g} {s['std']:10.4g} "
             f"{s['median']:+10.4g} {s['mad_sigma']:10.4g}")

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.7], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    for name, (r, v, c) in series.items():
        ax.plot(r, v, ".", ms=2.2, color=c, alpha=0.55,
               label=f"{name} (mean={stats[name]['mean']:+.2f}, std={stats[name]['std']:.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_title("FPA2 realistic scene, no noise: whole-slit CO2 bias, native vs. joint block", fontsize=11)
    ax.legend(fontsize=8, markerscale=4, loc="upper right")

    ax = fig.add_subplot(gs[1])
    ax.plot(r_r, v_r, ".", ms=2.2, color="tab:purple", alpha=0.55)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_xlabel("detector row")
    ax.set_title(f"rectified, own scale -- mean={stats['rectified']['mean']:+.1f} ppm "
                f"(§9l/§9m interpolation bias, already known, not the point of this comparison)",
                fontsize=10)

    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    header = f"{'pipeline':24s} {'n':>6s} {'mean':>10s} {'std':>10s} {'median':>10s} {'MAD-sig':>10s}"
    lines = [header, "-" * len(header)]
    for name in list(series.keys()) + ["rectified"]:
        s = stats[name]
        lines.append(f"{name:24s} {s['n']:6d} {s['mean']:+10.4g} {s['std']:10.4g} "
                     f"{s['median']:+10.4g} {s['mad_sigma']:10.4g}")
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, family="monospace",
           fontsize=9.5, va="top", ha="left")

    fig.suptitle("Joint block (whole-slit adaptive sweep) vs. native/rectified/undistorted", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_joint_block_vs_native_fpa{FPA}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
