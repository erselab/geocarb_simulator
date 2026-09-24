"""Pairwise correlation grid of retrieval ERRORS (retrieved - truth) between state rows, within
one multiband sweep -- seaborn.pairplot, scatter off-diagonal + histograms on the diagonal
(2026-09-23, user: "create grids of scatter plots that show the correlation between different
state variable errors within the runs... seaborn has a function that can do this, with
histograms of the error on the main diagonal"). Useful for spotting real degeneracies directly
(e.g. p_surface error correlating with amplitude_aerosol error) rather than inferring them from
posterior covariance alone.

co2_ppm/p_surface_hpa/h2o_surface_vmr/t_offset_k/amplitude_aerosol/height_aerosol are all FREE
on the SAME shared bin_centers grid within a tile (checked directly: identical `positions`
arrays). albedo_O2_A/albedo_CO2_strong sit on their own, much finer grid (224 vs ~41 points/tile
-- also checked directly: the two albedo rows share ONE grid with each other). The master grid
for this plot is albedo's own, finer one -- interpolating the coarse fields (np.interp, already
smooth by construction of being on a coarse retrieval grid) UP onto it, rather than downsampling
albedo's real per-bin retrieval onto the coarser grid, keeps every row's own actual retrieved
albedo value intact instead of smoothing away real fine-scale structure to match the sparser
fields.

Each row's own boundary bins are dropped first (2026-09-22, same DROP_BOUNDARY convention as
summarize_multiband_sweep.py -- least redundantly covered by nearby anchors).

    PYTHONPATH=.:<gert> python scripts/plot_multiband_error_pairplot.py [--prior realistic] [--aerosol]
"""
import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from geocarb_gert import along_slit_scene as als  # noqa: E402

PRIOR = sys.argv[sys.argv.index("--prior") + 1] if "--prior" in sys.argv else "structural"
AEROSOL = "--aerosol" in sys.argv
PSUF = "" if PRIOR == "structural" else f"_prior-{PRIOR}"
ASUF = "_aero" if AEROSOL else ""
SINGLE = "--single-band" in sys.argv     # FPA2-only merged sweep (no aerosol); same tiles/bins as the multi-band run
FREE_TAG = "co2-p-h2o-t-albedo-amplitude-height" if AEROSOL else "co2-p-h2o-t-albedo"

SHARED_ROWS = ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"]
if AEROSOL:
    SHARED_ROWS += ["amplitude_aerosol", "height_aerosol"]
ALBEDO_ROWS = [("albedo_O2_A", "O2_A"), ("albedo_CO2_strong", "CO2_strong")]
LABELS = {"co2_ppm": "CO2 [ppm]", "p_surface_hpa": "p [hPa]", "h2o_surface_vmr": "h2o (vmr)",
         "t_offset_k": "T [K]", "amplitude_aerosol": "amp_aer", "height_aerosol": "height_aer [Pa]",
         "albedo_O2_A": "albedo O2_A", "albedo_CO2_strong": "albedo CO2_str"}


def truth_of(name, x, label=None):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x, label))
    if name in als.SURFACE_FIELDS:
        return np.asarray(als.SURFACE_FIELDS[name](x))
    return np.asarray(als.STATE_FIELDS[name](x))


SB_DIR = "results/realistic_prior/gd_joint_block_whole_slit_fpa2_gratio1_adens4_free-co2-p-h2o-t-albedo_nwin33_analytic_prior-realistic_valb_spos-anchor_fapos-anchor_etaslit"
if SINGLE:
    assert not AEROSOL and PRIOR == "realistic", "--single-band data exists only for the realistic-prior no-aerosol arm"
    ALBEDO_ROWS = [("albedo", "CO2_strong")]
    LABELS["albedo"] = "albedo CO2_str"
    _sb = pickle.load(open(glob.glob(str(REPO / SB_DIR / "*.pkl"))[0], "rb"))["results"]
    files = [_sb[k]["hires"] for k in sorted(_sb) if "hires" in _sb[k]]
else:
  files = sorted(glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-{FREE_TAG}_cover_g1.0_etaslit{PSUF}{ASUF}.pkl")))
rows_out = []
for f in files:
    params = f["params"] if SINGLE else pickle.load(open(f, "rb"))["joint"]["params"]
    x_coarse = np.asarray(params[SHARED_ROWS[0]]["positions"]) * als.SLIT_HALF_KM
    x_fine = np.asarray(params[ALBEDO_ROWS[0][0]]["positions"]) * als.SLIT_HALF_KM   # the master grid
    if x_coarse.size <= 2 or x_fine.size <= 2:
        continue
    keep_fine = slice(1, -1)                    # drop this tile's own boundary bins (fine grid)
    x_fine_k = x_fine[keep_fine]
    tile_err = {}
    for name, label in ALBEDO_ROWS:
        p = params[name]
        tile_err[name] = np.asarray(p["values"])[keep_fine] - truth_of(name, x_fine_k, label)
    for name in SHARED_ROWS:
        p = params[name]
        e_coarse = np.asarray(p["values"])[1:-1] - truth_of(name, x_coarse)[1:-1]   # drop ITS OWN boundary too
        tile_err[name] = np.interp(x_fine_k, x_coarse[1:-1], e_coarse)
    tile_err["along-slit [km]"] = x_fine_k              # NOT a plotted variable -- scatter color only
    rows_out.append(pd.DataFrame(tile_err))
df = pd.concat(rows_out, ignore_index=True)
df = df.rename(columns=LABELS)
plot_vars = [c for c in df.columns if c != "along-slit [km]"]
print(f"{len(files)} tiles, {len(df)} bins pooled")
print("\ncorrelation matrix:")
print(df[plot_vars].corr().round(3).to_string())

def annotate_corr(x, y, **kws):
    """Upper triangle (2026-09-23, user: 'labels that have the names of the pairwise variables
    and the correlation between them'): each pair's own column names + Pearson r, text size/color
    scaled by |r| so strong pairs are visually obvious without cross-referencing the printed
    matrix above."""
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.set_facecolor(plt.cm.RdBu_r((r + 1) / 2, alpha=0.25))
    ax.annotate(f"{x.name}\nvs\n{y.name}\nr = {r:+.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
               ha="center", va="center", fontsize=8 + 6 * abs(r), fontweight="bold" if abs(r) > 0.5 else "normal")


ETA_NORM = plt.Normalize(vmin=-1400, vmax=1400)


def scatter_by_eta(x, y, **kws):
    """Lower triangle (2026-09-23, user: 'color the scatterplot points by eta value') -- each
    point's along-slit km position, not just which tile it came from, so a correlation that's
    actually localized to one part of the slit (e.g. the low-albedo region flagged separately)
    is visible directly rather than looking like a uniform cloud."""
    c = df.loc[x.index, "along-slit [km]"]
    plt.scatter(x, y, c=c, cmap="coolwarm", norm=ETA_NORM, s=8, alpha=0.45, edgecolor="none")


g = sns.PairGrid(df, vars=plot_vars)
g.map_diag(plt.hist, bins=40)
g.map_lower(scatter_by_eta)
g.map_upper(annotate_corr)
g.figure.suptitle(f"{'Single-band (FPA2)' if SINGLE else 'Multiband'} error correlations ({PRIOR} prior{', aerosol' if AEROSOL else ', no aerosol'})",
                  y=1.01)
g.figure.colorbar(plt.cm.ScalarMappable(norm=ETA_NORM, cmap="coolwarm"), ax=g.axes, shrink=0.5,
                  label="along-slit position [km]", location="right", pad=0.02)
out = REPO / f"plots/{"singleband" if SINGLE else "multiband"}_error_pairplot{PSUF}{ASUF}.png"
g.figure.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
