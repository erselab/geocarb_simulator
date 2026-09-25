"""Error-correlation pairplots for every run set (2026-09-25): FPA1/2/3 pairings x {single-band,
multi-band} x {noiseless, noisy seeds 1-3 pooled} -> 12 figures. Same design as
plot_multiband_error_pairplot.py: histograms on the diagonal, lower triangle = scatter colored by
along-slit position, upper triangle = names + Pearson r. The finest grid (albedo) is the master;
coarser state rows are np.interp'd onto it; each tile's boundary bins are dropped first.
Gases the pairing's bands cannot constrain (CH4/CO in FPA1, CO2 in FPA3: retrieved == prior exactly,
zero Jacobian) are left out -- their 'error' is just the prior error.

    PYTHONPATH=.:<gert> python scripts/plot_error_pairplots.py
Outputs: plots/error_pairplot_fpa<n>_{singleband|multiband}_{noiseless|noise-seeds123}_prior-realistic.png
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
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402

LABELS = {1: "CO2_weak", 2: "CO2_strong", 3: "CH4_CO"}
COL = {"co2_ppm": "CO2 [ppm]", "ch4_ppb": "CH4 [ppb]", "co_ppb": "CO [ppb]", "p_surface_hpa": "p [hPa]",
       "h2o_surface_vmr": "h2o (vmr)", "t_offset_k": "T [K]", "albedo_O2_A": "albedo O2_A"}
SHARED = {1: ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"],
          2: ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"],
          3: ["ch4_ppb", "co_ppb", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"]}
ETA_NORM = plt.Normalize(vmin=-1400, vmax=1400)
SB_DRIVER = "--sb-driver" in sys.argv      # default: the single band is run through the multi-band code (same anchor grid)


def config(pair):
    if pair == 2:
        return "co2-p-h2o-t-albedo", 33, REPO / "geometry_config_noaero_realistic.json"
    return "co2-ch4-co-p-h2o-t-albedo", {1: 60, 3: 47}[pair], REPO / f"geometry_config_fpa0-{pair}_realistic.json"


def truth(name, x, label=None):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x, label))
    return np.asarray(als.STATE_FIELDS[name](x))


def tile_frame(params, pair, arm):
    """One tile's error DataFrame on the albedo (finest) grid."""
    lab = LABELS[pair]
    alb = ({f"albedo {lab}": (f"albedo_{lab}", lab)} if (arm == "mb" or not SB_DRIVER) else {f"albedo {lab}": ("albedo", lab)})
    if arm == "mb":
        alb["albedo O2_A"] = ("albedo_O2_A", "O2_A")
    first = next(iter(alb.values()))[0]
    xf = np.asarray(params[first]["positions"]) * als.SLIT_HALF_KM
    xk = xf[1:-1]
    d = {}
    for colname, (pname, band) in alb.items():
        d[colname] = (np.asarray(params[pname]["values"])[1:-1] - truth(None, xk, band))
    xc = np.asarray(params[SHARED[pair][0]]["positions"]) * als.SLIT_HALF_KM
    for n in SHARED[pair]:
        e = (np.asarray(params[n]["values"]) - truth(n, xc))[1:-1]
        d[COL[n]] = np.interp(xk, xc[1:-1], e)
    d["along-slit [km]"] = xk
    return pd.DataFrame(d)


def frames(pair, arm, seed):
    tag, nw, geom = config(pair)
    nsuf = f"_noise{seed}" if seed else ""
    if arm == "sb" and not SB_DRIVER:      # single band through the multi-band code (default since 2026-09-25)
        get = lambda key: pickle.load(open(REPO / f"results/realistic_prior/multiband/mb_fpa{pair}_r{pair}-{key[0]}-{key[1]}_free-{tag}_cover_g1.0_etaslit_prior-realistic_geomcfg{nsuf}.pkl", "rb"))["joint"]["params"]
    elif arm == "sb":
        sd = (f"results/realistic_prior/gd_joint_block_whole_slit_fpa{pair}_gratio1_adens4_free-{tag}_nwin{nw}"
              f"_analytic_prior-realistic_valb_spos-anchor_fapos-anchor_etaslit{nsuf}")
        res = pickle.load(open(glob.glob(str(REPO / sd / "*.pkl"))[0], "rb"))["results"]
        get = lambda key: res[key]["hires"]["params"]
    else:
        mb = {}
        for g in glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-{pair}_r0-*_r{pair}-*_free-{tag}_cover_g1.0_etaslit_prior-realistic{nsuf}.pkl")):
            d = pickle.load(open(g, "rb"))
            mb[tuple(d["rows_by_fpa"][pair])] = d["joint"]["params"]
        get = lambda key: mb[key]
    return [tile_frame(get(tuple(t["rows_by_fpa"][str(pair)])), pair, arm) for t in json.load(open(geom))["tiles"]]


def annotate_corr(x, y, **kws):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.set_facecolor(plt.cm.RdBu_r((r + 1) / 2, alpha=0.25))
    ax.annotate(f"{x.name}\nvs\n{y.name}\nr = {r:+.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes, ha="center",
               va="center", fontsize=7 + 6 * abs(r), fontweight="bold" if abs(r) > 0.5 else "normal")


for pair in (1, 2, 3):
    for arm, armname in (("sb", "singleband"), ("mb", "multiband")):
        for cond, seeds in (("noiseless", [None]), ("noise-seeds123", [1, 2, 3])):
            df = pd.concat([f for s in seeds for f in frames(pair, arm, s)], ignore_index=True)
            plot_vars = [c for c in df.columns if c != "along-slit [km]"]

            def scatter_by_eta(x, y, **kws):
                plt.scatter(x, y, c=df.loc[x.index, "along-slit [km]"], cmap="coolwarm", norm=ETA_NORM, s=6,
                            alpha=0.4, edgecolor="none")

            g = sns.PairGrid(df, vars=plot_vars)
            g.map_diag(plt.hist, bins=40)
            g.map_lower(scatter_by_eta)
            g.map_upper(annotate_corr)
            what = f"single-band FPA{pair}" if arm == "sb" else f"multi-band FPA0+FPA{pair}"
            ncond = "noiseless" if cond == "noiseless" else "noisy, seeds 1-3 pooled"
            g.figure.suptitle(f"Error correlations: {what}, {ncond} (realistic prior, no aerosol; {len(df)} bins)", y=1.01)
            g.figure.colorbar(plt.cm.ScalarMappable(norm=ETA_NORM, cmap="coolwarm"), ax=g.axes, shrink=0.5,
                              label="along-slit position [km]", location="right", pad=0.02)
            out = REPO / f"plots/error_pairplot_fpa{pair}_{armname}_{cond}_prior-realistic.png"
            g.figure.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(g.figure)
            print("saved", out.name, len(df), flush=True)
