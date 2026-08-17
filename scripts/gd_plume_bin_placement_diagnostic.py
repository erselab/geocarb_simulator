#!/usr/bin/env python3
"""Direct look at bin PLACEMENT and pixel-per-bin counts at the finest G
tested in gd_plume_bin_density_sweep.py, prompted by the natural question
that sweep raises: with G approaching (or exceeding) the window's own row
count, are bins actually resolving new information, or mostly just
re-hitting the same pixels?

No radiative transfer involved -- this is pure geometry (bin_centers via
gd_joint_block_diagnostics.pixel_density_bin_centers, nearest-bin pixel
counts via its own bin_assign), so it runs in seconds, unlike the sweep
scripts.

For each of the same three windows, at both the production G and the
finest G tested there:
  - bin_centers (quantiles of the REAL per-pixel eta distribution --
    width x 1024 points, not one per row, since keystone/smile means
    different columns of the same row sit at different eta -- see
    gd_plume_bin_density_sweep.py's own module docstring and the
    conversation this follows up on).
  - nearest-bin pixel counts (same assignment gd_joint_block_diagnostics
    uses for its own per-bin chi2 diagnostic), reported per bin and
    summarized (min/max/empty-bin count).
  - for each bin, which detector rows its assigned pixels actually come
    from (row_lo..row_hi span) -- directly answers "is this bin mostly
    redrawing pixels already covered by its neighbor," since two bins
    with heavily overlapping row spans are working with much the same
    physical rows, distinguished only by which columns (keystone-shifted
    eta) fell on which side of the quantile edge between them.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_bin_placement_diagnostic.py
Output: plots/joint_block/gd_plume_bin_placement_fpa2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_retrieve import FPA, _eta_of  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers, bin_assign  # noqa: E402
from gd_joint_block_whole_slit_sweep import G_RATIO  # noqa: E402
from gd_plume_bin_density_sweep import WINDOWS  # noqa: E402


def analyze(row_lo: int, row_hi: int, G: int):
    rows_win = np.arange(row_lo, row_hi + 1)
    cols = np.arange(1024.0)
    eta_by_row = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])  # (width, 1024)
    eta_flat = eta_by_row.ravel()
    row_flat = np.repeat(rows_win, 1024)

    bin_centers = pixel_density_bin_centers(eta_flat, G)
    idx = bin_assign(bin_centers, eta_flat)
    counts = np.bincount(idx, minlength=G)

    row_span = np.full((G, 2), -1.0)
    for g in range(G):
        rows_g = row_flat[idx == g]
        if len(rows_g) > 0:
            row_span[g] = [rows_g.min(), rows_g.max()]

    return dict(row_lo=row_lo, row_hi=row_hi, G=G, rows_win=rows_win, cols=cols,
               eta_by_row=eta_by_row, bin_centers=bin_centers, counts=counts, row_span=row_span)


def main() -> int:
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(len(WINDOWS), 3, figsize=(16, 3.8 * len(WINDOWS)))

    for k, (row_lo, row_hi, wlabel, g_values) in enumerate(WINDOWS):
        prod_G = max(2, int(round((row_hi - row_lo + 1) / G_RATIO)))
        fine_G = max(g_values)
        base = analyze(row_lo, row_hi, prod_G)
        fine = analyze(row_lo, row_hi, fine_G)

        # --- panel 1: eta vs row for every pixel (subsampled columns for
        # readability), with bin-center lines for both G levels ---
        ax = axes[k, 0]
        col_sub = np.arange(0, 1024, 8)
        rows_win = base["rows_win"]
        for i, r in enumerate(rows_win):
            ax.scatter(np.full(len(col_sub), r), base["eta_by_row"][i, col_sub],
                      s=2, color="0.55", alpha=0.5, zorder=1)
        for bc in base["bin_centers"]:
            ax.axhline(bc, color="tab:orange", lw=0.9, alpha=0.8, zorder=2)
        for bc in fine["bin_centers"]:
            ax.axhline(bc, color="tab:blue", lw=0.5, alpha=0.5, zorder=3)
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel}): pixel eta + bin centers\n"
                    f"orange=production G={prod_G}, blue=finest G={fine_G}", fontsize=9.5)
        ax.set_xlabel("detector row"); ax.set_ylabel(r"pixel $\eta$")

        # --- panel 2: pixel count per bin, finest G ---
        ax = axes[k, 1]
        ax.bar(np.arange(fine_G), fine["counts"], color="tab:blue", width=0.85)
        ax.axhline(fine["counts"].mean(), color="0.3", lw=0.8, ls="--",
                  label=f"mean={fine['counts'].mean():.0f}")
        n_empty = int(np.sum(fine["counts"] == 0))
        ax.set_title(f"pixels per bin, finest G={fine_G} "
                    f"(min={fine['counts'].min()}, max={fine['counts'].max()}, empty={n_empty})",
                    fontsize=9.5)
        ax.set_xlabel("bin index (eta order)"); ax.set_ylabel("pixel count")
        ax.legend(fontsize=7.5)

        # --- panel 3: each bin's own row span (min..max row of its
        # assigned pixels) at finest G -- overlapping spans between
        # neighboring bins directly show shared-row reuse ---
        ax = axes[k, 2]
        for g in range(fine_G):
            lo, hi = fine["row_span"][g]
            if lo < 0:
                continue
            ax.plot([lo, hi], [g, g], color="tab:blue", lw=2.5, solid_capstyle="butt")
        ax.set_title(f"row span of each bin's assigned pixels, finest G={fine_G}", fontsize=9.5)
        ax.set_xlabel("detector row"); ax.set_ylabel("bin index (eta order)")

    fig.suptitle(f"FPA{FPA}: bin placement and pixel counts at production vs. finest tested G", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_plume_bin_placement_fpa{FPA}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")

    print("\nsummary (finest G per window):")
    for row_lo, row_hi, wlabel, g_values in WINDOWS:
        fine_G = max(g_values)
        fine = analyze(row_lo, row_hi, fine_G)
        width = row_hi - row_lo + 1
        rows_per_bin_span = [hi - lo + 1 for lo, hi in fine["row_span"] if lo >= 0]
        print(f"  rows {row_lo}-{row_hi} ({wlabel}), width={width}, G={fine_G}: "
             f"pixel count/bin min={fine['counts'].min()} max={fine['counts'].max()} "
             f"mean={fine['counts'].mean():.0f} empty_bins={int(np.sum(fine['counts']==0))}; "
             f"row-span/bin min={min(rows_per_bin_span)} max={max(rows_per_bin_span)} "
             f"mean={np.mean(rows_per_bin_span):.2f} (1 row-only bins: "
             f"{sum(1 for s in rows_per_bin_span if s==1)}/{fine_G})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
