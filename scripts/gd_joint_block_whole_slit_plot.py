#!/usr/bin/env python3
"""Stitch the 58 independent windows of the whole-slit joint-block sweep
(scripts/gd_joint_block_whole_slit_sweep.py) into slit-wide comparisons:
true/coarse/hires CO2 profile, retrieval bias, per-window fit quality, and
the adaptive window/G sizing itself. Every quantity here is reconstructed
from the sweep's own saved state vectors (x_coarse, x_hires per window) --
no re-solving.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_whole_slit_plot.py \\
        [results/gd_joint_block_whole_slit_fpa2_gratio1.pkl]
Output: plots/joint_block/<input stem>.png
"""
from __future__ import annotations

import argparse
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

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=str, nargs="?",
                    default=str(REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}.pkl"),
                    help="path to a gd_joint_block_whole_slit_sweep.py (or _merge.py) output pickle")
    args = ap.parse_args()
    in_path = Path(args.input)

    with open(in_path, "rb") as f:
        d = pickle.load(f)
    results = d["results"]
    windows = sorted(results.values(), key=lambda r: r["row_lo"])

    rows_all, true_all, coarse_all, hires_all = [], [], [], []
    width_all, G_all, resid_c_all, resid_h_all, row_mid_all = [], [], [], [], []

    for w in windows:
        row_lo, row_hi = w["row_lo"], w["row_hi"]
        rows_win = np.arange(row_lo, row_hi + 1)
        eta_win = _eta_of(FPA, np.full(len(rows_win), 512.0), rows_win.astype(float))
        true_win = als.xco2_ppm(eta_win * als.SLIT_HALF_KM)

        bin_centers = w["bin_centers"]
        coarse_edges = 0.5 * (bin_centers[:-1] + bin_centers[1:]) if len(bin_centers) > 1 else np.array([])
        retrieved_ppm_coarse = w["prior_co2_ppm_bins"] * w["x_coarse"]
        retrieved_ppm_hires = w["prior_co2_ppm_bins"] * w["x_hires"]

        if len(bin_centers) > 1:
            idx = np.searchsorted(coarse_edges, eta_win)
            coarse_win = retrieved_ppm_coarse[idx]
            hires_win = np.interp(eta_win, bin_centers, retrieved_ppm_hires)
        else:
            coarse_win = np.full(len(rows_win), retrieved_ppm_coarse[0])
            hires_win = np.full(len(rows_win), retrieved_ppm_hires[0])

        rows_all.append(rows_win)
        true_all.append(true_win)
        coarse_all.append(coarse_win)
        hires_all.append(hires_win)
        width_all.append(w["width"])
        G_all.append(w["G"])
        resid_c_all.append(w["resid_coarse_rms"])
        resid_h_all.append(w["resid_hires_rms"])
        row_mid_all.append(0.5 * (row_lo + row_hi))

    rows_all = np.concatenate(rows_all)
    true_all = np.concatenate(true_all)
    coarse_all = np.concatenate(coarse_all)
    hires_all = np.concatenate(hires_all)
    bias_coarse = coarse_all - true_all
    bias_hires = hires_all - true_all

    print(f"{len(windows)} windows, {len(rows_all)} rows total")
    print(f"coarse: mean={bias_coarse.mean():+.4f} rms={np.sqrt(np.mean(bias_coarse**2)):.4f} "
         f"max|bias|={np.max(np.abs(bias_coarse)):.4f} ppm")
    print(f"hires:  mean={bias_hires.mean():+.4f} rms={np.sqrt(np.mean(bias_hires**2)):.4f} "
         f"max|bias|={np.max(np.abs(bias_hires)):.4f} ppm")
    worst_row_c = rows_all[np.argmax(np.abs(bias_coarse))]
    worst_row_h = rows_all[np.argmax(np.abs(bias_hires))]
    print(f"worst coarse bias at row {worst_row_c}, worst hires bias at row {worst_row_h}")

    rc_fine = rows_crossed(FPA, rows_all.astype(float))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.6, 1.0, 1.0]})

    ax = axes[0]
    ax.plot(rows_all, true_all, color="black", lw=1.1, label="true", zorder=5)
    ax.plot(rows_all, coarse_all, color="tab:orange", lw=0.9, alpha=0.85, label="coarse posterior")
    ax.plot(rows_all, hires_all, color="tab:blue", lw=0.9, alpha=0.85, label="hi-res posterior")
    ax.set_ylabel("CO2 [ppm]")
    ax.set_title(f"FPA{FPA} whole-slit joint block sweep: true vs. retrieved CO2, {len(windows)} independent windows", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper right", markerscale=2)

    ax = axes[1]
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(rows_all, bias_coarse, color="tab:orange", lw=0.8,
           label=f"coarse (rms={np.sqrt(np.mean(bias_coarse**2)):.3f}, max={np.max(np.abs(bias_coarse)):.3f} ppm)")
    ax.plot(rows_all, bias_hires, color="tab:blue", lw=0.8,
           label=f"hi-res (rms={np.sqrt(np.mean(bias_hires**2)):.3f}, max={np.max(np.abs(bias_hires)):.3f} ppm)")
    ax.set_ylabel("retrieval bias\n(posterior - true) [ppm]")
    ax.set_title("retrieval bias across the whole slit", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[2]
    ax.step(row_mid_all, resid_c_all, where="mid", color="tab:orange", lw=1.2, label="coarse")
    ax.step(row_mid_all, resid_h_all, where="mid", color="tab:blue", lw=1.2, label="hi-res")
    ax.set_yscale("log")
    ax.set_ylabel("per-window\nresid RMS")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("per-window residual RMS (fit quality)", fontsize=10.5)

    ax = axes[3]
    ax2 = ax.twinx()
    ax.step(row_mid_all, width_all, where="mid", color="0.4", lw=1.2, label="window width [rows]")
    ax2.step(row_mid_all, G_all, where="mid", color="#A0631B", lw=1.2, label="G [bins]")
    ax.plot(rows_all, rc_fine * 4, color="0.75", lw=0.8, ls="--", label="rows_crossed x4 (scale ref.)")
    ax.set_ylabel("window width [rows]")
    ax2.set_ylabel("G [bins]", color="#A0631B")
    ax.set_xlabel("detector row")
    ax.set_title("adaptive window size and bin count (tied to local keystone)", fontsize=10.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{in_path.stem}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
