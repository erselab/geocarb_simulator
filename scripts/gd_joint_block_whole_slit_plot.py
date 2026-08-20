#!/usr/bin/env python3
"""Stitch the 58 independent windows of the whole-slit joint-block sweep
(scripts/gd_joint_block_whole_slit_sweep.py) into slit-wide comparisons:
true/coarse/hires CO2 profile, retrieval bias, per-window fit quality, and
the adaptive window/G sizing itself. Every quantity here is reconstructed
from the sweep's own saved StateSpec snapshots (`w["coarse"]["params"]
["co2_ppm"]`/`w["hires"]["params"]["co2_ppm"]`, already-scaled physical ppm
values per window) -- no re-solving.

Reads `params["co2_ppm"]["values"]` directly rather than reconstructing
via `w["prior_co2_ppm_bins"] * w["x_hires"]` (fixed 2026-08-20): that
legacy top-level `prior_co2_ppm_bins` field is ALWAYS exact truth,
kept only for old back-compat plotters -- it does not reflect
`--prior-fields`, so the old reconstruction silently gave the wrong
posterior for any non-"exact" prior run (right by coincidence only when
prior=truth, since `prior_co2_ppm_bins` happens to equal the real prior
in that one case). `gd_joint_block_matrix.py`'s own `stitch()` was
never affected -- it always read `params[...]["values"]` this same way.

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

    # --hires-only sweeps (--anchor-density / --state-interp variants) carry no
    # x_coarse/resid_coarse_rms at all, since the coarse solve is unaffected by
    # those flags and re-running it would just reproduce the baseline. Detect
    # that once here and drop the coarse curves rather than KeyError.
    has_coarse = all("x_coarse" in w for w in windows)
    if not has_coarse:
        print("no x_coarse in this pickle (hi-res-only sweep) -- plotting hi-res only")

    rows_all, true_all, coarse_all, hires_all, prior_all = [], [], [], [], []
    width_all, G_all, resid_c_all, resid_h_all, row_mid_all = [], [], [], [], []

    for w in windows:
        row_lo, row_hi = w["row_lo"], w["row_hi"]
        rows_win = np.arange(row_lo, row_hi + 1)
        eta_win = _eta_of(FPA, np.full(len(rows_win), 512.0), rows_win.astype(float))
        true_win = als.xco2_ppm(eta_win * als.SLIT_HALF_KM)

        bin_centers = w["bin_centers"]
        coarse_edges = 0.5 * (bin_centers[:-1] + bin_centers[1:]) if len(bin_centers) > 1 else np.array([])
        # Already-scaled physical ppm values, correct under any --prior-fields
        # (see module docstring) -- positions are guaranteed == bin_centers by
        # construction (state_spec_from_scene stores exactly the positions it
        # was called with), so bin_centers is still the right x-axis to
        # interpolate/bin against below.
        retrieved_ppm_coarse = (np.asarray(w["coarse"]["params"]["co2_ppm"]["values"])
                                if has_coarse else None)
        retrieved_ppm_hires = np.asarray(w["hires"]["params"]["co2_ppm"]["values"])
        # Same shape/positions as retrieved_ppm_hires (state_spec_from_scene
        # stores prior at exactly its own row's positions) -- under
        # --prior-fields exact this equals true_win exactly (prior=truth),
        # so it's a real regression test as well as a plotting feature:
        # plotting it costs nothing extra under the common case, and shows
        # the prior's own error under any imperfect --prior-fields.
        prior_ppm_hires = np.asarray(w["hires"]["params"]["co2_ppm"]["prior"])

        if len(bin_centers) > 1:
            idx = np.searchsorted(coarse_edges, eta_win)
            coarse_win = retrieved_ppm_coarse[idx] if has_coarse else None
            hires_win = np.interp(eta_win, bin_centers, retrieved_ppm_hires)
            prior_win = np.interp(eta_win, bin_centers, prior_ppm_hires)
        else:
            coarse_win = (np.full(len(rows_win), retrieved_ppm_coarse[0])
                          if has_coarse else None)
            hires_win = np.full(len(rows_win), retrieved_ppm_hires[0])
            prior_win = np.full(len(rows_win), prior_ppm_hires[0])

        rows_all.append(rows_win)
        true_all.append(true_win)
        if has_coarse:
            coarse_all.append(coarse_win)
        hires_all.append(hires_win)
        prior_all.append(prior_win)
        width_all.append(w["width"])
        G_all.append(w["G"])
        if has_coarse:
            resid_c_all.append(w["resid_coarse_rms"])
        resid_h_all.append(w["resid_hires_rms"])
        row_mid_all.append(0.5 * (row_lo + row_hi))

    rows_all = np.concatenate(rows_all)
    true_all = np.concatenate(true_all)
    hires_all = np.concatenate(hires_all)
    prior_all = np.concatenate(prior_all)
    bias_hires = hires_all - true_all
    bias_prior = prior_all - true_all
    if has_coarse:
        coarse_all = np.concatenate(coarse_all)
        bias_coarse = coarse_all - true_all

    print(f"{len(windows)} windows, {len(rows_all)} rows total")
    print(f"prior:  mean={bias_prior.mean():+.4f} rms={np.sqrt(np.mean(bias_prior**2)):.4f} "
         f"max|bias|={np.max(np.abs(bias_prior)):.4f} ppm")
    if has_coarse:
        print(f"coarse: mean={bias_coarse.mean():+.4f} rms={np.sqrt(np.mean(bias_coarse**2)):.4f} "
             f"max|bias|={np.max(np.abs(bias_coarse)):.4f} ppm")
    print(f"hires:  mean={bias_hires.mean():+.4f} rms={np.sqrt(np.mean(bias_hires**2)):.4f} "
         f"max|bias|={np.max(np.abs(bias_hires)):.4f} ppm")
    worst_row_h = rows_all[np.argmax(np.abs(bias_hires))]
    if has_coarse:
        print(f"worst coarse bias at row {rows_all[np.argmax(np.abs(bias_coarse))]}, "
             f"worst hires bias at row {worst_row_h}")
    else:
        print(f"worst hires bias at row {worst_row_h}")

    rc_fine = rows_crossed(FPA, rows_all.astype(float))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.6, 1.0, 1.0]})

    # Prior only gets its own trace/legend entry under an imperfect prior
    # (--prior-fields != "exact"). NOT a np.allclose(prior_all, true_all)
    # check: even under "exact" the prior trace is a PIECEWISE-LINEAR
    # interpolation of the state's own bin-center values (same as
    # hires_win), which differs from the continuous true_win by the
    # already-documented interpolation-representation-error floor (real,
    # but a separate effect from "prior is wrong" -- would almost always
    # read as "differs" and defeat the point of this check).
    prior_differs = d.get("prior_fields", "exact") != "exact"

    ax = axes[0]
    ax.plot(rows_all, true_all, color="black", lw=1.1, label="true", zorder=5)
    if prior_differs:
        ax.plot(rows_all, prior_all, color="0.5", lw=0.9, ls="--", label="prior", zorder=4)
    if has_coarse:
        ax.plot(rows_all, coarse_all, color="tab:orange", lw=0.9, alpha=0.85, label="coarse posterior")
    ax.plot(rows_all, hires_all, color="tab:blue", lw=0.9, alpha=0.85, label="hi-res posterior")
    ax.set_ylabel("CO2 [ppm]")
    ax.set_title(f"FPA{FPA} whole-slit joint block sweep: true vs. retrieved CO2, {len(windows)} independent windows", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper right", markerscale=2)

    ax = axes[1]
    ax.axhline(0, color="black", lw=0.6)
    if prior_differs:
        ax.plot(rows_all, bias_prior, color="0.5", lw=0.8, ls="--",
               label=f"prior (rms={np.sqrt(np.mean(bias_prior**2)):.3f}, max={np.max(np.abs(bias_prior)):.3f} ppm)")
    if has_coarse:
        ax.plot(rows_all, bias_coarse, color="tab:orange", lw=0.8,
               label=f"coarse (rms={np.sqrt(np.mean(bias_coarse**2)):.3f}, max={np.max(np.abs(bias_coarse)):.3f} ppm)")
    ax.plot(rows_all, bias_hires, color="tab:blue", lw=0.8,
           label=f"hi-res (rms={np.sqrt(np.mean(bias_hires**2)):.3f}, max={np.max(np.abs(bias_hires)):.3f} ppm)")
    ax.set_ylabel("bias\n(- true) [ppm]" if prior_differs else "retrieval bias\n(posterior - true) [ppm]")
    ax.set_title("prior and retrieval bias across the whole slit" if prior_differs
                else "retrieval bias across the whole slit", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[2]
    if has_coarse:
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
