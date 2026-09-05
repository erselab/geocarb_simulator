#!/usr/bin/env python3
"""One picture tying together everything discussed about how bins, anchors,
rows, columns, and eta relate for the 108-116 west-hot-spot window (G=18
finest tested) -- the concrete follow-up to the correction in docs/JOINT_
BLOCK_MIGRATION_PLAN.md Sec.9 ("the column-sharing analysis described the
COARSE bin partition, not the hi-res forward model's real resolution").

Three things live in the same (row, column) space but partition it
completely differently, and this script puts all three in one figure:

  - G=18 STATE BINS (`pixel_density_bin_centers`): quantiles of the real
    per-pixel eta distribution over the core window's own rows. These are
    the retrieval's free parameters (`x`, what Gauss-Newton actually
    solves for) and also directly determine the COARSE posterior via
    `bin_assign` (nearest-bin, hard partition).
  - G_eff=17 ANCHORS (`build_forward_hires`'s own `anchor_etas`): one per
    detector row in the PADDED window (width + 2*PAD), each a single eta
    value at column 512 only. This is where RT is ACTUALLY run for the
    hi-res model -- one independent `spectrum_for(...)` call per anchor,
    fed by the G-dim state INTERPOLATED onto the anchor's own eta (state-
    space interpolation, `np.interp(anchor_etas, bin_centers, x)`), never
    the other way around.
  - Real PIXELS (row, col): each one gets nearest-neighbor assigned to
    exactly one bin (coarse scheme) and, independently, to exactly one
    anchor (hi-res scheme) -- two different hard partitions of the same
    pixel grid, per `docs/JOINT_BLOCK_MIGRATION_PLAN.md` Sec.9's own
    "no pixel assigned twice, but the two schemes disagree" discussion.

Panel 1: eta vs. row for the full PADDED range (104-120, since anchors
extend past the 9 core output rows), with G=18 bin-center lines and
G_eff=17 anchor points both marked directly on the real per-pixel eta
scatter -- literally where the spectra are computed (anchors) vs. how
the coarse state is binned (bin centers).
Panel 2: which STATE BIN each core-window pixel is nearest to (the coarse
partition).
Panel 3: which ANCHOR each core-window pixel is nearest to (the real
hi-res partition) -- directly comparable to panel 2, same rows/columns,
different partition.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_bins_vs_anchors_diagram.py \\
        [--row-lo 108] [--row-hi 116] [--G 18]
Output: plots/joint_block/gd_plume_bins_vs_anchors_fpa2_r108-116_G18.png
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
from gd_joint_block_diagnostics import pixel_density_bin_centers, bin_assign  # noqa: E402
from gd_joint_block_whole_slit_sweep import PAD, ROW_MAX_IDX  # noqa: E402

SEGMENT_COLORS = ["#E07A5F", "#3D5A80", "#F2CC8F", "#9B5DE5", "#00A6A6", "#EE6C4D",
                 "#7A9E9F", "#C1666B", "#81B29A", "#F4845F", "#5B5F97", "#B5838D",
                 "#6D9DC5", "#F18F01", "#48A9A6", "#D6A2E8", "#4C956C", "#E4572E"]


def owner_map(idx_row_by_col: np.ndarray, n_owners: int):
    """idx_row_by_col: (n_rows, 1024) owner index per pixel -> per-row
    contiguous (owner, col_lo, col_hi) segments."""
    segments = []
    for i in range(idx_row_by_col.shape[0]):
        row_idx = idx_row_by_col[i]
        change = np.where(np.diff(row_idx) != 0)[0]
        starts = np.concatenate(([0], change + 1))
        ends = np.concatenate((change, [len(row_idx) - 1]))
        segments.append(list(zip(row_idx[starts], starts, ends)))
    return segments


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

    # core-window per-pixel eta (what bin_centers is quantiled over, and
    # what both partitions get applied to)
    eta_core = np.stack([_eta_of(FPA, cols, np.full(1024, float(r))) for r in rows_win])
    eta_core_flat = eta_core.ravel()

    # G=18 state bins (coarse partition)
    bin_centers = pixel_density_bin_centers(eta_core_flat, G)
    bin_idx = bin_assign(bin_centers, eta_core_flat).reshape(width, 1024)

    # G_eff anchors (hi-res partition) -- padded row range, one eta per row at col=512
    anchor_rows = np.arange(max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD) + 1)
    anchor_etas = _eta_of(FPA, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float))
    order = np.argsort(anchor_etas)
    anchor_rows_sorted = anchor_rows[order]
    anchor_etas_sorted = anchor_etas[order]
    G_eff = len(anchor_rows)
    anchor_edges = 0.5 * (anchor_etas_sorted[:-1] + anchor_etas_sorted[1:])
    anchor_idx_core = np.searchsorted(anchor_edges, eta_core_flat).reshape(width, 1024)
    anchor_row_owner = anchor_rows_sorted[anchor_idx_core]  # actual row number, not sort-index

    # per-pixel eta over the FULL padded range, for panel 1's background scatter
    eta_pad = np.stack([_eta_of(FPA, cols, np.full(1024, float(r))) for r in anchor_rows])

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.6, 1.6], hspace=0.32)

    # ---- panel 1: eta vs row, full padded range, bins + anchors marked ----
    ax = fig.add_subplot(gs[0])
    ax.axvspan(row_lo - 0.5, row_hi + 0.5, color="0.93", zorder=0, label="core output rows")
    col_sub = np.arange(0, 1024, 6)
    for i, r in enumerate(anchor_rows):
        ax.scatter(np.full(len(col_sub), r), eta_pad[i, col_sub], s=3, color="0.55", alpha=0.45, zorder=1)
    for bc in bin_centers:
        ax.axhline(bc, color="tab:orange", lw=0.8, alpha=0.75, zorder=2)
    ax.plot([], [], color="tab:orange", lw=1.5, label=f"G={G} state bin centers (coarse partition)")
    ax.scatter(anchor_rows_sorted, anchor_etas_sorted, marker="*", s=180, color="black",
              edgecolor="white", linewidth=0.6, zorder=5,
              label=f"G_eff={G_eff} anchors (RT actually computed here, col=512 only)")
    for r, e in zip(anchor_rows_sorted, anchor_etas_sorted):
        ax.axvline(r, color="0.85", lw=0.4, zorder=0)
    ax.set_xlabel("detector row")
    ax.set_ylabel(r"pixel $\eta$")
    ax.set_title(f"FPA{FPA} rows {row_lo}-{row_hi} (core, shaded) + PAD={PAD} padding: "
                f"per-pixel $\\eta$ (gray), state bins (orange lines), anchors (black stars)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    # ---- panel 2: coarse (state-bin) ownership map, core rows only ----
    ax2 = fig.add_subplot(gs[1])
    bin_segments = owner_map(bin_idx, G)
    for i, r in enumerate(rows_win):
        for owner, c_lo, c_hi in bin_segments[i]:
            color = SEGMENT_COLORS[int(owner) % len(SEGMENT_COLORS)]
            ax2.axhspan(r - 0.48, r + 0.48, xmin=c_lo / 1024, xmax=(c_hi + 1) / 1024,
                       facecolor=color, edgecolor="white", linewidth=0.7, zorder=2)
            if c_hi - c_lo > 55:
                ax2.text((c_lo + c_hi) / 2, r, f"bin {int(owner)}", ha="center", va="center",
                        fontsize=7.5, color="white", zorder=3)
    ax2.set_xlim(0, 1024); ax2.set_ylim(row_hi + 0.55, row_lo - 0.55)
    ax2.set_yticks(rows_win)
    ax2.set_xlabel("detector column"); ax2.set_ylabel("detector row")
    ax2.set_title(f"COARSE partition: which of the G={G} state bins each pixel is nearest to "
                f"(what build_forward / the coarse posterior actually uses)", fontsize=10.5)

    # ---- panel 3: hi-res (anchor) ownership map, core rows only ----
    ax3 = fig.add_subplot(gs[2])
    # relabel owners as sort-rank so color/segment logic (contiguous-run based on
    # value change) works the same way as panel 2
    rank_of_row = {int(r): i for i, r in enumerate(anchor_rows_sorted)}
    anchor_rank_core = np.vectorize(rank_of_row.get)(anchor_row_owner)
    anchor_segments = owner_map(anchor_rank_core, G_eff)
    for i, r in enumerate(rows_win):
        for owner_rank, c_lo, c_hi in anchor_segments[i]:
            owner_row = int(anchor_rows_sorted[int(owner_rank)])
            color = SEGMENT_COLORS[owner_row % len(SEGMENT_COLORS)]
            ax3.axhspan(r - 0.48, r + 0.48, xmin=c_lo / 1024, xmax=(c_hi + 1) / 1024,
                       facecolor=color, edgecolor="white", linewidth=0.7, zorder=2)
            if c_hi - c_lo > 55:
                ax3.text((c_lo + c_hi) / 2, r, f"row {owner_row}'s\nanchor", ha="center", va="center",
                        fontsize=7, color="white", zorder=3)
    ax3.set_xlim(0, 1024); ax3.set_ylim(row_hi + 0.55, row_lo - 0.55)
    ax3.set_yticks(rows_win)
    ax3.set_xlabel("detector column"); ax3.set_ylabel("detector row")
    ax3.set_title(f"HI-RES partition: which of the G_eff={G_eff} row anchors each pixel is nearest to "
                f"(what build_forward_hires actually uses -- nearly every pixel maps to ITS OWN row's anchor here, "
                f"since rows_crossed is small at this window)", fontsize=10.5)

    fig.suptitle(f"FPA{FPA} rows {row_lo}-{row_hi}: bins vs. anchors vs. real pixels", fontsize=13.5, y=0.995)
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_plume_bins_vs_anchors_fpa{FPA}_r{row_lo}-{row_hi}_G{G}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
