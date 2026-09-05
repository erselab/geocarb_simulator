#!/usr/bin/env python3
"""Column-level follow-up to gd_plume_bin_placement_diagnostic.py, answering
directly: in the 108-116 west-hot-spot window at its finest tested G=18,
the 6 bins confined to a single INTERIOR row (bins 3, 6, 8, 10, 12, 14, on
rows 109, 111-115 -- the two edge rows 108/116 each split into two confined
bins instead and are shown for context but not what this was asked about)
each sit in the MIDDLE of their own row's 1024 columns. What flanks them on
either side is not empty space -- it is columns belonging to a NEIGHBORING
bin that ALSO reaches into the adjacent row, i.e. shared with that row too.
This makes the shared/not-shared split explicit per row: which columns are
this row's alone (owned only by its own confined bin, no other row touches
them) versus which columns are shared (owned by a bin that also appears one
row up or down).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_single_row_bin_columns.py \\
        [--row-lo 108] [--row-hi 116] [--G 18]
Output: plots/joint_block/gd_plume_single_row_bin_columns_fpa2_r108-116_G18.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_retrieve import FPA, _eta_of  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers, bin_assign  # noqa: E402

NOT_SHARED_COLOR = "#3E8E5B"      # one color for every "not shared" (confined) bin
SHARED_COLORS = ["#E07A5F", "#3D5A80", "#F2CC8F", "#9B5DE5",
                 "#00A6A6", "#EE6C4D", "#7A9E9F", "#C1666B"]  # one per shared bin, colors repeat across rows it spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-lo", type=int, default=108)
    ap.add_argument("--row-hi", type=int, default=116)
    ap.add_argument("--G", type=int, default=18)
    args = ap.parse_args()
    row_lo, row_hi, G = args.row_lo, args.row_hi, args.G

    rows_win = np.arange(row_lo, row_hi + 1)
    width = len(rows_win)
    cols = np.arange(1024.0)
    eta_by_row = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    eta_flat = eta_by_row.ravel()

    bin_centers = pixel_density_bin_centers(eta_flat, G)
    idx = bin_assign(bin_centers, eta_flat).reshape(width, 1024)

    row_span = {}       # bin -> (row_lo, row_hi) it appears in
    col_range = {}       # (bin, row) -> (col_lo, col_hi), contiguous within that row
    for g in range(G):
        rows_g, cols_g = np.where(idx == g)
        if len(rows_g) == 0:
            continue
        row_span[g] = (int(rows_g.min()) + row_lo, int(rows_g.max()) + row_lo)
        for r in np.unique(rows_g):
            c = cols_g[rows_g == r]
            col_range[(g, int(r) + row_lo)] = (int(c.min()), int(c.max()))
    not_shared_bins = sorted(g for g, (lo, hi) in row_span.items() if lo == hi)
    shared_bins = sorted(g for g, (lo, hi) in row_span.items() if lo != hi)
    # the 6 confined-to-an-interior-row bins specifically asked about (excludes
    # the edge rows row_lo/row_hi, which split into two confined bins each
    # instead of one -- a different, edge-boundary effect, not this question)
    interior_not_shared = [g for g in not_shared_bins if row_span[g][0] not in (row_lo, row_hi)]

    shared_color_of = {g: SHARED_COLORS[i % len(SHARED_COLORS)] for i, g in enumerate(shared_bins)}

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, ax = plt.subplots(figsize=(13, 0.62 * width + 2.5))

    # paint each row's contiguous bin segments directly (not an imshow of bin
    # index -- shared bins need a per-bin color that stays visually linked
    # across the two rows they touch, not a generic categorical colormap)
    for i, r in enumerate(rows_win):
        row_bins = np.unique(idx[i])
        for g in row_bins:
            c_lo, c_hi = col_range[(int(g), int(r))]
            color = NOT_SHARED_COLOR if g in not_shared_bins else shared_color_of[g]
            ax.axhspan(r - 0.48, r + 0.48, xmin=c_lo / 1024, xmax=(c_hi + 1) / 1024,
                      facecolor=color, edgecolor="white", linewidth=0.8, zorder=2)
            label = f"bin {g}" if (c_hi - c_lo) > 60 else ""
            is_focus = g in interior_not_shared
            ax.text((c_lo + c_hi) / 2, r, label, ha="center", va="center",
                   fontsize=8.5 if is_focus else 7, fontweight="bold" if is_focus else "normal",
                   color="white", zorder=3)

    ax.set_xlim(0, 1024)
    ax.set_ylim(row_hi + 0.55, row_lo - 0.55)
    ax.set_yticks(rows_win)
    ax.set_xlabel("detector column")
    ax.set_ylabel("detector row")
    ax.set_title(f"FPA{FPA}: rows {row_lo}-{row_hi}, G={G} -- shared vs. not-shared columns per row\n"
                f"green = NOT shared (bin confined to this row only) -- the {len(interior_not_shared)} "
                f"interior bins {interior_not_shared} are the ones asked about; "
                f"other colors = SHARED (bin also reaches into the row above/below, same color both places)",
                fontsize=10.5)

    fig.tight_layout()
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_plume_single_row_bin_columns_fpa{FPA}_r{row_lo}-{row_hi}_G{G}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    print(f"\nthe {len(interior_not_shared)} interior not-shared bins {interior_not_shared} "
         f"(rows {[row_span[g][0] for g in interior_not_shared]}):")
    for g in interior_not_shared:
        r = row_span[g][0]
        c_lo, c_hi = col_range[(g, r)]
        n_not_shared = c_hi - c_lo + 1
        neighbors = [g2 for g2 in shared_bins if r in range(row_span[g2][0], row_span[g2][1] + 1)]
        n_shared = 1024 - n_not_shared
        print(f"  row {r}: bin {g} owns cols [{c_lo},{c_hi}] ({n_not_shared} px, "
             f"{n_not_shared/1024*100:.0f}% of the row) NOT shared with any other row; "
             f"remaining {n_shared} px ({n_shared/1024*100:.0f}%) belong to shared bins "
             f"{neighbors} (each also reaches into the row above and/or below)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
