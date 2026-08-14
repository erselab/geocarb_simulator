#!/usr/bin/env python3
"""Visualize the trace-and-select pixel scheme from
`JOINT_ROW_INVERSION_PLAN.md` §0: for a target along-slit position `eta0`
(defined at a chosen center row's own column=512 convention, matching
`x_km_of_row` throughout this repo), evaluate **every** (row, column)
pixel's true position against `eta0` and mark exactly which ones are
within the area-overlap tolerance -- not just one per row. For several
target rows spanning very different keystone amplitudes (`rows_crossed`),
so the qualitative difference between a near-null-keystone target and a
high-keystone target is visible directly.

A common misreading (corrected here, 2026-08-12): it's tempting to assume
one column per row is the natural unit -- pick whichever column comes
closest to `eta0` and check if *that one* is within tolerance. That's
what an earlier version of this script did, and it hides something
important: near a keystone-null row, essentially the *entire* row (all
1024 columns) sits within tolerance, because keystone means the row's own
`eta` barely changes with column at all there -- there's no reason to
throw away 1023 perfectly good columns just because a script only checked
the single nearest one. This version checks all of them.

Each panel shows, for one target row `i_c`:
  - a background field of `eta(row, col) - eta0` over the row window x all
    1024 columns (the real, known geometric mapping, via
    `xy_to_wavelength_slit` -- exactly what `gd_render.image()` itself
    evaluates per pixel), diverging colormap centered at 0 in physical km
  - a contour at exactly 0 -- the smooth iso-eta projection of that ground
    location across the FPA
  - a **solid green overlay** on *every* (row, column) pixel within
    `--tolerance-km` of eta0 -- exactly the pixels the trace-and-select
    scheme would include in the retrieval. This can be a wide swath (most
    or all of a row) near keystone-null rows, and a narrow, tilted band
    that cuts across many rows near high-keystone rows.
  - a **red x** at the single closest-available column on any row that has
    *zero* included pixels -- that row contributes nothing for this eta0
  - a status strip just right of each panel, one cell per row, shaded by
    how many columns that row contributes (darker green = more), red = none

`--tolerance-km` is the area-overlap criterion, exposed as a CLI
parameter so it can be swept independently. Per the user's own
90%-area-overlap criterion (2026-08-12): a pixel-center-to-eta-curve
distance under ~0.2 km is the relevant threshold (default here).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_trace_pixels.py
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_trace_pixels.py --tolerance-km 2.0
Output: plots/gd_toy_trace_pixels_fpa2.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from geocarb_gert import along_slit_scene as als
from geocarb_gert.gd_polynomials import rows_crossed, xy_to_wavelength_slit
from geocarb_gert.gd_render import s_max

REPO_ROOT = Path(__file__).resolve().parent.parent
FPA = 2
N_PX = 1024
CENTER_ROWS = [25, 100, 300, 512, 700, 950]   # spans null-keystone -> near-max, exact rows_crossed already known from §11p/§11o
MIN_WINDOW = 4   # rows on each side, floor for near-null-keystone panels so they still show context
STRIP_GAP_PX = 25    # columns of blank space between the detector panel and the status strip
STRIP_WIDTH_PX = 15  # columns of margin kept to the right of the status strip


def _eta_of(fpa: int, cols, rows) -> np.ndarray:
    _, s = xy_to_wavelength_slit(fpa, cols, rows)
    return s / s_max(fpa)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance-km", type=float, default=0.2,
                    help="max |eta-eta0| (converted to km, i.e. pixel-center-to-eta-curve "
                         "distance) for a (row, column) pixel to count as included in the "
                         "retrieval (default 0.2 km, the user's own ~90%%-area-overlap "
                         "criterion; try 2.0 for a looser ~one-native-pixel criterion)")
    args = ap.parse_args()
    tol = args.tolerance_km

    km_per_eta = als.SLIT_HALF_KM   # eta in [-1,1] spans the full slit; x_km = eta * SLIT_HALF_KM (see _band_setup)
    cols_full = np.arange(N_PX, dtype=float)
    green_cmap = ListedColormap(["#00cc00"])

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    for ax, i_c in zip(axes.flat, CENTER_ROWS):
        k_c = rows_crossed(FPA, i_c)
        # window sized to this row's own keystone reach (roughly 2x the
        # expected kept-run half-width), not a fixed size -- otherwise a
        # short, contiguous, line-tracking run of kept rows gets lost in a
        # sea of empty/gray space at low keystone.
        window = max(MIN_WINDOW, int(round(2.2 * k_c)))
        rows_win = np.arange(max(0, i_c - window), min(N_PX, i_c + window + 1))
        eta0 = _eta_of(FPA, 512.0, float(i_c))

        # eta(row, col) - eta0, in km, over the full column range -- every
        # (row, column) pixel individually, not just one per row
        col_mesh, row_mesh = np.meshgrid(cols_full, rows_win)
        eta_mesh = _eta_of(FPA, col_mesh, row_mesh)
        d_km_mesh = (eta_mesh - eta0) * km_per_eta
        kept_mask = np.abs(d_km_mesh) <= tol

        vmax = np.abs(d_km_mesh).max()
        im = ax.pcolormesh(col_mesh, row_mesh, d_km_mesh, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.contour(col_mesh, row_mesh, d_km_mesh, levels=[0.0], colors="k", linewidths=1.5)

        # every included pixel, not just one per row -- solid green overlay,
        # transparent (NaN) everywhere else so the background still shows through
        overlay = np.where(kept_mask, 1.0, np.nan)
        ax.pcolormesh(col_mesh, row_mesh, overlay, cmap=green_cmap, shading="auto",
                      alpha=0.85, zorder=4)

        kept_counts = kept_mask.sum(axis=1)     # per-row: how many columns included
        n_pixels = int(kept_counts.sum())
        rows_with_any = int((kept_counts > 0).sum())

        # rows with ZERO included pixels: mark the single closest-available
        # column for reference, so it's clear how close that row could get
        for k, i in enumerate(rows_win):
            if kept_counts[k] == 0:
                j = int(np.argmin(np.abs(d_km_mesh[k])))
                ax.scatter([j], [i], s=50, marker="x", color="red", linewidths=1.6, zorder=6)

        # per-row status strip: shaded by how many columns that row
        # contributes (0 = red, more columns = darker green)
        strip_col = N_PX + STRIP_GAP_PX
        max_count = max(kept_counts.max(), 1)
        strip_colors = []
        for c in kept_counts:
            if c == 0:
                strip_colors.append("red")
            else:
                g = 0.35 + 0.55 * (1 - c / max_count)   # more columns -> darker green
                strip_colors.append((g * 0.2, 0.55 + 0.45 * (c / max_count), g * 0.2))
        ax.scatter([strip_col] * len(rows_win), rows_win, s=60, marker="s",
                  color=strip_colors, edgecolors="0.3", linewidths=0.3, zorder=6, clip_on=False)

        ax.set_title(f"row {i_c}  (rows_crossed={k_c:.2f})\n"
                    f"{n_pixels} pixels included, across {rows_with_any}/{len(rows_win)} rows "
                    f"(tol={tol:g} km)", fontsize=10)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_xlim(0, strip_col + STRIP_WIDTH_PX)
        if i_c == CENTER_ROWS[0]:
            handles = [Patch(facecolor="#00cc00", alpha=0.85, label="included pixel(s)"),
                      plt.Line2D([], [], color="red", marker="x", linestyle="none",
                                 label="row has zero included pixels\n(closest available shown)")]
            ax.legend(handles=handles, fontsize=7, loc="upper left")
        fig.colorbar(im, ax=ax, label="eta-eta0 [km]", shrink=0.85, pad=0.12)

    fig.suptitle(f"FPA{FPA}: which (row, column) pixels does trace-and-select actually keep?\n"
                f"tolerance = {tol:g} km pixel-center-to-eta-curve distance, evaluated per-pixel "
                f"(not one column per row) -- right-hand strip shaded by included-column count",
                fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    plots_dir = REPO_ROOT / "plots" / "toy_diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_toy_trace_pixels_fpa{FPA}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
