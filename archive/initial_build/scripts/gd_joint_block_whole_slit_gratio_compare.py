#!/usr/bin/env python3
"""Direct old (G=width/3) vs. new (G=width, ground-footprint-tied)
comparison on the real whole-slit production run
(docs/JOINT_BLOCK_MIGRATION_PLAN.md Sec.10), in the same style as the
profile and residual-RMS plots used throughout this investigation.
Nothing here re-solves anything -- both pickles already have per-window
x_coarse/x_hires/bin_centers/resid_*_rms saved, this only re-plots them.

Panel 1 (per window, three familiar windows from Sec.9): true vs. coarse
vs. hires profile, old default overlaid with the new one.
Panel 2: per-window residual RMS (spectral fit quality) across the WHOLE
slit, old vs. new, coarse and hires -- the only "residual" quantity
actually saved (see chat: full per-pixel residual arrays were never
written to the pickle, only this scalar RMS per window).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_whole_slit_gratio_compare.py
Output: plots/joint_block/gd_joint_block_whole_slit_gratio_compare_fpa2.png
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
from geocarb_gert import along_slit_scene as als  # noqa: E402

OLD_PATH = REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}.pkl"
NEW_PATH = REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}_gratio1.pkl"

# familiar windows from Sec.9 (row_lo, row_hi as tiled -- looked up by
# whichever tile CONTAINS these rows, since the two runs' tilings are
# identical -- same build_window_tiles(FPA), unaffected by --g-ratio)
LOOKUP_ROWS = [338, 904, 112]
LABELS = {338: "broad plume peak (331-345)", 904: "east hot spot (884-924)", 112: "west hot spot (108-116)"}


def find_window(results, row):
    for (lo, hi), w in results.items():
        if lo <= row <= hi:
            return w
    raise KeyError(row)


def profile(w):
    rows_win = np.arange(w["row_lo"], w["row_hi"] + 1)
    width = w["width"]
    eta_win = _eta_of(FPA, np.full(width, 512.0), rows_win.astype(float))
    true_win = als.xco2_ppm(eta_win * als.SLIT_HALF_KM)
    bin_centers = w["bin_centers"]
    retrieved_hires = w["prior_co2_ppm_bins"] * w["x_hires"]
    hires_win = np.interp(eta_win, bin_centers, retrieved_hires) if w["G"] > 1 else np.full(width, retrieved_hires[0])
    return rows_win, true_win, hires_win


def main() -> int:
    with open(OLD_PATH, "rb") as f:
        d_old = pickle.load(f)
    with open(NEW_PATH, "rb") as f:
        d_new = pickle.load(f)
    print(f"old: g_ratio={d_old['g_ratio']:g}, new: g_ratio={d_new['g_ratio']:g}")

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig = plt.figure(figsize=(13, 3.4 * len(LOOKUP_ROWS) + 4.5))
    gs = fig.add_gridspec(len(LOOKUP_ROWS) + 1, 1,
                          height_ratios=[1] * len(LOOKUP_ROWS) + [1.4], hspace=0.45)

    for k, row in enumerate(LOOKUP_ROWS):
        w_old = find_window(d_old["results"], row)
        w_new = find_window(d_new["results"], row)
        rows_old, true_old, hires_old = profile(w_old)
        rows_new, true_new, hires_new = profile(w_new)

        ax = fig.add_subplot(gs[k])
        ax.plot(rows_new, true_new, color="black", lw=1.5, label="true", zorder=5)
        ax.plot(rows_old, hires_old, color="tab:blue", lw=1.1,
               label=f"hi-res, G=width/3 (old default, G={w_old['G']}, resid_rms={w_old['resid_hires_rms']:.2e})")
        ax.plot(rows_new, hires_new, color="tab:red", lw=1.1, ls="--",
               label=f"hi-res, G=width (new, G={w_new['G']}, resid_rms={w_new['resid_hires_rms']:.2e})")
        ax.set_title(f"rows {w_new['row_lo']}-{w_new['row_hi']} ({LABELS[row]})", fontsize=10.5)
        ax.set_xlabel("detector row"); ax.set_ylabel("CO2 [ppm]")
        ax.legend(fontsize=7.5, loc="best")

    # ---- residual RMS comparison across the whole slit ----
    ax2 = fig.add_subplot(gs[len(LOOKUP_ROWS)])
    for label, d, color_h, color_c in [("old (G=width/3)", d_old, "tab:blue", "#8fb8e0"),
                                       ("new (G=width)", d_new, "tab:red", "#e8a598")]:
        windows = sorted(d["results"].values(), key=lambda r: r["row_lo"])
        row_mid = [0.5 * (w["row_lo"] + w["row_hi"]) for w in windows]
        resid_h = [w["resid_hires_rms"] for w in windows]
        resid_c = [w["resid_coarse_rms"] for w in windows]
        ax2.step(row_mid, resid_h, where="mid", color=color_h, lw=1.3, label=f"hi-res, {label}")
        ax2.step(row_mid, resid_c, where="mid", color=color_c, lw=1.0, ls=":", label=f"coarse, {label}")
    ax2.set_yscale("log")
    ax2.set_xlabel("detector row")
    ax2.set_ylabel("per-window\nresid RMS")
    ax2.set_title("per-window residual RMS across the whole slit -- the only saved 'residual' "
                 "(scalar per window; full per-pixel residual arrays were not written to the pickle)",
                 fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper right", ncol=2)

    fig.suptitle(f"FPA{FPA}: whole-slit production run, G=width/3 (old) vs. G=width (new)", fontsize=13)
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_joint_block_whole_slit_gratio_compare_fpa{FPA}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
