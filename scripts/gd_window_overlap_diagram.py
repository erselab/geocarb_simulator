#!/usr/bin/env python3
"""How `build_window_tiles(..., overlap=N)` actually changes two adjacent
windows' own bin placement -- the concrete follow-up to the conversation
about window-overlap merging (docs/PROJECT_STATUS.md Sec.6,
`geocarb_gert/along_slit_state.py`, `geocarb_gert/along_slit_query.py`).

Same convention as `gd_plume_bins_vs_anchors_diagram.py`: real per-pixel
eta scatter as the background, state bins marked as horizontal lines on
top. Two adjacent windows (window i, window i+1), each with its own
independently-placed G bins (`pixel_density_bin_centers`, sized off ITS
OWN width via `G = round(width/g_ratio)`) -- shown once under `overlap=0`
(today's exact tiling: windows meet with no shared rows, no shared
eta-range) and once under `overlap=N>0` (both windows widened at their
shared boundary, so their row ranges -- and hence their own bin-eta-ranges
-- genuinely overlap).

The point this makes visually: overlap does NOT create shared bins. Each
window's G bins are independently re-placed across its own (now wider)
eta range; window i's bins and window i+1's bins in the overlap band are
two completely different, independently-optimized sets of eta positions
that never coincide. The two windows only ever meet when `along_slit_query.
query_state` is asked for a value at a specific query point: it separately
interpolates each covering window's own bins onto that point and blends
the two resulting estimates (and their real covariance) -- see that
module's own docstring for why (a live example of that blend is out of
scope for a bin-placement diagram; this one is purely about where the
bins themselves end up).

Run:  PYTHONPATH=. python3 scripts/gd_window_overlap_diagram.py \\
        [--window-index 1] [--overlap 3] [--g-ratio 1]
Output: plots/joint_block/gd_window_overlap_r<lo>-<hi>_ovlp<N>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_retrieve import FPA, _eta_of  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
from gd_joint_block_whole_slit_sweep import build_window_tiles  # noqa: E402

COLOR_I = "#3D5A80"      # window i
COLOR_I1 = "#E07A5F"     # window i+1
COLOR_OVERLAP = "#F2CC8F"


def _window_bins(fpa, lo, hi, g_ratio):
    rows_win = np.arange(lo, hi + 1)
    width = len(rows_win)
    G = max(2, round(width / g_ratio))
    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(fpa, cols, np.full(1024, float(r))) for r in rows_win])
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    return G, bin_centers, eta_all


def _panel(ax, fpa, lo_i, hi_i, lo_i1, hi_i1, g_ratio, title):
    lo_plot, hi_plot = min(lo_i, lo_i1) - 1, max(hi_i, hi_i1) + 1
    rows_plot = np.arange(lo_plot, hi_plot + 1)
    cols = np.arange(1024.0)
    col_sub = np.arange(0, 1024, 6)

    # background: real per-pixel eta, all rows shown -- x=row (constant per
    # scatter call), y=eta (varies across the column subsample at that row),
    # same convention as gd_plume_bins_vs_anchors_diagram.py's own panel 1.
    eta_pad = np.stack([_eta_of(fpa, cols, np.full(1024, float(r))) for r in rows_plot])
    for i, r in enumerate(rows_plot):
        ax.scatter(np.full(len(col_sub), r), eta_pad[i, col_sub], s=3, color="0.6", alpha=0.4, zorder=1)

    # shade each window's own row range, and the overlap (intersection)
    # distinctly -- axVspan, not axhspan: row is the X axis here, not Y.
    ax.axvspan(lo_i - 0.5, hi_i + 0.5, color=COLOR_I, alpha=0.10, zorder=0)
    ax.axvspan(lo_i1 - 0.5, hi_i1 + 0.5, color=COLOR_I1, alpha=0.10, zorder=0)
    ov_lo, ov_hi = max(lo_i, lo_i1), min(hi_i, hi_i1)
    if ov_lo <= ov_hi:
        ax.axvspan(ov_lo - 0.5, ov_hi + 0.5, color=COLOR_OVERLAP, alpha=0.55, zorder=0,
                  label=f"shared rows ({ov_hi - ov_lo + 1})")

    G_i, bc_i, _ = _window_bins(fpa, lo_i, hi_i, g_ratio)
    G_i1, bc_i1, _ = _window_bins(fpa, lo_i1, hi_i1, g_ratio)
    for bc in bc_i:
        ax.axhline(bc, color=COLOR_I, lw=1.1, alpha=0.85, zorder=2)
    for bc in bc_i1:
        ax.axhline(bc, color=COLOR_I1, lw=1.1, alpha=0.85, ls="--", zorder=2)
    ax.plot([], [], color=COLOR_I, lw=1.5, label=f"window i bins (rows {lo_i}-{hi_i}, G={G_i})")
    ax.plot([], [], color=COLOR_I1, lw=1.5, ls="--", label=f"window i+1 bins (rows {lo_i1}-{hi_i1}, G={G_i1})")

    ax.set_xlim(lo_plot - 0.5, hi_plot + 0.5)
    ax.set_xlabel("detector row")
    ax.set_ylabel(r"pixel $\eta$")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-index", type=int, default=1,
                    help="index i of the (window i, window i+1) pair to show, into the "
                         "overlap=0 tiling's own list (default 1: the same pair inspected "
                         "in conversation, rows 9-17 / 18-26 at min_window=4)")
    ap.add_argument("--overlap", type=int, default=3)
    ap.add_argument("--g-ratio", type=float, default=1.0)
    ap.add_argument("--min-window", type=int, default=4)
    args = ap.parse_args()

    tiles0 = build_window_tiles(FPA, min_window=args.min_window, window_scale=1.0, overlap=0)
    tilesN = build_window_tiles(FPA, min_window=args.min_window, window_scale=1.0, overlap=args.overlap)
    i = args.window_index
    lo_i0, hi_i0 = tiles0[i]
    lo_i10, hi_i10 = tiles0[i + 1]
    lo_iN, hi_iN = tilesN[i]
    lo_i1N, hi_i1N = tilesN[i + 1]

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    _panel(axes[0], FPA, lo_i0, hi_i0, lo_i10, hi_i10, args.g_ratio,
          f"overlap=0 (today's default): windows meet, no shared rows,\n"
          f"bins independently placed on disjoint eta ranges")
    _panel(axes[1], FPA, lo_iN, hi_iN, lo_i1N, hi_i1N, args.g_ratio,
          f"overlap={args.overlap}: both windows widened at the shared boundary --\n"
          f"row ranges overlap, but the bins are STILL two independent sets")

    fig.suptitle(f"FPA{FPA}, g_ratio={args.g_ratio:g}: window overlap changes each window's own "
                f"bin placement, never creates a shared bin", fontsize=13)
    fig.tight_layout()
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_window_overlap_r{lo_i0}-{hi_i10}_ovlp{args.overlap}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
