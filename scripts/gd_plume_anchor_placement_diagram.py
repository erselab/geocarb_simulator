#!/usr/bin/env python3
"""Direct visual comparison of the two anchor placement schemes tested in
gd_plume_anchor_density_sweep_v2.py, for all three windows: OLD (row-
uniform -- one anchor per detector row, eta taken at column 512 only) vs.
NEW (pixel-density -- G_eff anchors placed at quantiles of the real
per-pixel eta distribution over the full padded window). Both computed
at the SAME G_eff = width + 2*PAD used throughout that sweep, so this is
literally what each scheme handed the retrieval at matched anchor count.

OLD anchors have a natural row position (one per row, by construction),
so they're plotted as points at their own (row, eta) location -- these
trace the smooth row-vs-eta curve at even ROW spacing. NEW anchors have
no such row association (they're quantiles of the full 2D pixel
population, not tied to one row each), so they're plotted as horizontal
lines spanning the padded row range, at their own eta value -- their
VERTICAL spacing pattern is the thing to look at, and how it compares to
OLD's even spacing directly shows what pixel-density placement changed.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_anchor_placement_diagram.py
Output: plots/joint_block/gd_plume_anchor_placement_fpa2.png
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
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
from gd_joint_block_whole_slit_sweep import PAD, ROW_MAX_IDX  # noqa: E402

WINDOWS = [
    (331, 345, "broad plume peak"),
    (884, 924, "east hot spot (ATBD 890-935 ref)"),
    (108, 116, "west hot spot (worst-case)"),
]


def main() -> int:
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(len(WINDOWS), 2, figsize=(15, 4.2 * len(WINDOWS)),
                             gridspec_kw={"width_ratios": [3.2, 1]})

    for k, (row_lo, row_hi, wlabel) in enumerate(WINDOWS):
        width = row_hi - row_lo + 1
        pad_rows = np.arange(max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD) + 1)
        G_eff = len(pad_rows)
        cols = np.arange(1024.0)

        eta_pad = np.stack([_eta_of(FPA, cols, np.full(1024, float(r))) for r in pad_rows])
        eta_pad_flat = eta_pad.ravel()

        anchor_etas_old = _eta_of(FPA, np.full(G_eff, 512.0), pad_rows.astype(float))
        anchor_etas_new = np.sort(pixel_density_bin_centers(eta_pad_flat, G_eff))

        # ---- left panel: row vs eta, both anchor schemes overlaid on real pixels ----
        ax = axes[k, 0]
        ax.axvspan(row_lo - 0.5, row_hi + 0.5, color="0.93", zorder=0, label="core output rows")
        col_sub = np.arange(0, 1024, 6)
        for i, r in enumerate(pad_rows):
            ax.scatter(np.full(len(col_sub), r), eta_pad[i, col_sub], s=3, color="0.6", alpha=0.4, zorder=1)
        for e in anchor_etas_new:
            ax.axhline(e, color="tab:red", lw=0.8, alpha=0.75, zorder=2)
        ax.plot([], [], color="tab:red", lw=1.5, label=f"NEW: pixel-density anchors (G_eff={G_eff})")
        ax.plot(pad_rows, anchor_etas_old, "o-", color="tab:blue", ms=5, lw=1.2, zorder=4,
               label=f"OLD: row-uniform anchors (G_eff={G_eff})")
        ax.set_xlabel("detector row")
        ax.set_ylabel(r"pixel / anchor $\eta$")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel}): where each scheme's anchors land", fontsize=10.5)
        if k == 0:
            ax.legend(fontsize=8, loc="upper left")

        # ---- right panel: anchor spacing (consecutive eta differences), both schemes ----
        ax2 = axes[k, 1]
        spacing_old = np.diff(np.sort(anchor_etas_old))
        spacing_new = np.diff(anchor_etas_new)
        idx_old = np.arange(len(spacing_old))
        idx_new = np.arange(len(spacing_new))
        ax2.plot(idx_old, spacing_old, "o-", color="tab:blue", ms=4, label="OLD spacing")
        ax2.plot(idx_new, spacing_new, "o-", color="tab:red", ms=4, label="NEW spacing")
        ax2.set_xlabel("anchor index (eta order)")
        ax2.set_ylabel(r"consecutive anchor $\Delta\eta$")
        ax2.set_title("anchor-to-anchor spacing", fontsize=10.5)
        if k == 0:
            ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"FPA{FPA}: anchor placement, OLD (row-uniform) vs. NEW (pixel-density), matched G_eff",
                fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_plume_anchor_placement_fpa{FPA}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
