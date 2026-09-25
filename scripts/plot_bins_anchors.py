"""How the state-vector bins and the RT anchors are laid out (2026-09-25), for the summary document.

Panel A: one tile (tile 12 of FPA0+FPA2) in detail, along-slit position [km]: every detector row of both bands
(bar = the eta extent of that row over ALL 1024 columns, i.e. including keystone; tick = its nominal centre-column
position), the padded anchor grid (one RT run each; also the albedo grid), the bin centres (where the gas /
pressure / humidity / temperature states live), and the tile's core interval.
Panel B: bin spacing along the whole slit for the three band pairings (from the exported geometry configs). Bins sit at
evenly spaced quantiles of the real pixel-eta distribution, so interior gaps are ~1 detector row (2.85 km) while each
tile's first and last gap stretch (3-15 km) because few pixels fall in the extreme tails.
    PYTHONPATH=.:<gert> python scripts/plot_bins_anchors.py
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.multiband_geometry import band_tables  # noqa: E402

H = als.SLIT_HALF_KM
ink, sec, grid = "#0b0b0b", "#52514e", "#e6e5e0"
C = {0: "#2a78d6", 2: "#eb6834", "bin": "#8e44ad", "anchor": "#1baf7a", "core": "#c3c2b7"}

geom = json.load(open(REPO / "geometry_config_noaero_realistic.json"))["tiles"]
t = geom[12]
r0, r2 = t["rows_by_fpa"]["0"], t["rows_by_fpa"]["2"]
d = pickle.load(open(REPO / f"results/realistic_prior/multiband/mb_fpa0-2_r0-{r0[0]}-{r0[1]}_r2-{r2[0]}-{r2[1]}_free-co2-p-h2o-t-albedo_cover_g1.0_etaslit_prior-realistic.pkl", "rb"))
anch = np.sort(np.asarray(d["anchor_etas"])) * H
bins = np.asarray(d["bin_centers"]) * H

fig = plt.figure(figsize=(12, 10.5), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0])
ax = fig.add_subplot(gs[0])
ymap = {}
y = 0.0
for f in (0, 2):
    lo, hi = t["rows_by_fpa"][str(f)]
    bt = band_tables(f)
    for r in range(lo, hi + 1):
        ax.plot([bt["eta_lo"][r] * H, bt["eta_hi"][r] * H], [y, y], color=C[f], lw=3, alpha=0.55, solid_capstyle="butt")
        ax.plot(bt["eta_c"][r] * H, y, "|", color=C[f], ms=7, mew=1.4)
        y += 1.0
    ax.text(anch.min() - 1, y - 0.5 * (hi - lo + 1), f"FPA{f} rows {lo}-{hi}", color=C[f], ha="right", va="center", fontsize=9)
    y += 1.5
y_anchor, y_bin = y + 1.0, y + 5.0
ax.plot(anch, np.full(anch.size, y_anchor), "|", color=C["anchor"], ms=9, mew=0.9)
ax.text(anch.min() - 1, y_anchor, f"RT anchors\n(n={anch.size}, {np.diff(anch).mean():.2f} km apart;\nalso the albedo grid)", ha="right", va="center", fontsize=9, color=C["anchor"])
ax.plot(bins, np.full(bins.size, y_bin), "v", color=C["bin"], ms=7)
ax.text(anch.min() - 1, y_bin, f"bin centres\n(G={bins.size}; gas, p, H$_2$O, T)", ha="right", va="center", fontsize=9, color=C["bin"])
ax.axvspan(t["eta_lo"] * H, t["eta_hi"] * H, color=C["core"], alpha=0.35, lw=0)
ax.text(0.5 * (t["eta_lo"] + t["eta_hi"]) * H, y_bin + 1.8, "tile core (non-overlapping) interval", ha="center", fontsize=9, color=sec)
ax.set_xlim(anch.min() - 34, anch.max() + 3)
ax.set_ylim(-1, y_bin + 3.2)
ax.set_yticks([])
ax.set_xlabel("along-slit position [km]", color=ink)
ax.set_title("A. One tile in detail (tile 12, FPA0+FPA2): every row's footprint, the RT anchors, and the state bins", loc="left", fontsize=11, color=ink)
ax.text(anch.max(), 0.5, "bar = a row's extent over all 1024 columns (keystone); tick = its centre column", ha="right", va="bottom", fontsize=8, color=sec)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)

ax2 = fig.add_subplot(gs[1])
cfgs = ((2, "geometry_config_noaero_realistic.json", "FPA0+FPA2 (33 tiles)", "#eb6834"),
        (1, "geometry_config_fpa0-1_realistic.json", "FPA0+FPA1 (60 tiles)", "#1baf7a"),
        (3, "geometry_config_fpa0-3_realistic.json", "FPA0+FPA3 (47 tiles)", "#8e44ad"))
for pair, fn, lab, col in cfgs:
    xi, si, xe, se = [], [], [], []
    for tt in json.load(open(REPO / fn))["tiles"]:
        b = np.asarray(tt["bin_centers"]) * H
        g, m = np.diff(b), 0.5 * (b[1:] + b[:-1])
        if g.size >= 3:
            xi += list(m[1:-1]); si += list(g[1:-1])
            xe += [m[0], m[-1]]; se += [g[0], g[-1]]
    ax2.plot(xi, si, ".", ms=3, color=col, alpha=0.55, label=f"{lab}: gaps between interior bins")
    ax2.plot(xe, se, "D", ms=4.5, mfc="none", mec=col, mew=1.2, alpha=0.9, label=f"{lab}: first and last gap of each tile")
ax2.axhline(2.76, color=sec, ls=":", lw=1)
ax2.text(-1395, 2.9, "one detector row ≈ 2.76 km", fontsize=8, color=sec, va="bottom")
ax2.set_ylabel("distance between neighbouring bins [km]", color=ink)
ax2.set_xlabel("along-slit position [km]", color=ink)
ax2.set_title("B. Bin spacing along the slit: about one detector row apart inside a tile, stretching at each tile's two outer gaps",
              loc="left", fontsize=11, color=ink)
ax2.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2, markerscale=1.6)
ax2.grid(True, color=grid, lw=0.8)
for s_ in ("top", "right"):
    ax2.spines[s_].set_visible(False)
out = REPO / "plots/bins_and_anchors.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out.name)
