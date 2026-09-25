"""Histograms of retrieval error (retrieved - truth) for every run set (2026-09-25).

For each band pairing (FPA1, FPA2, FPA3) and each condition -- noiseless, and noisy with seeds 1-3
pooled -- one figure with a panel per state variable overlaying the single-band (FPA<n>) and
multi-band (FPA0+FPA<n>) error distributions (identical tiles/bins; boundary bins dropped) plus the
prior's own error for scale. Albedo is on a different grid in each arm; its histogram pools each
arm's own bins. Pairs whose band has no lines for a gas (CH4/CO in FPA1, CO2 in FPA3) show the
prior error only (all three histograms coincide) and are still plotted for completeness.
Histogram x-range is set by the retrievals' 99.5th percentile; prior errors beyond it are cut off (counts kept in the density).

    PYTHONPATH=.:<gert> python scripts/plot_error_histograms.py
Outputs: plots/error_hist_fpa<n>_{noiseless|noise-seeds123}_prior-realistic.png
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

LABELS = {1: "CO2_weak", 2: "CO2_strong", 3: "CH4_CO"}
NAMES = {"co2_ppm": "CO$_2$ [ppm]", "ch4_ppb": "CH$_4$ [ppb]", "co_ppb": "CO [ppb]",
         "p_surface_hpa": "p$_{surf}$ [hPa]", "h2o_surface_vmr": "H$_2$O vmr", "t_offset_k": "T offset [K]",
         "albedo": "albedo"}
C = {"sb": "#8e44ad", "mb": "#1baf7a", "prior": "#eb6834"}


def config(pair):
    if pair == 2:
        return "co2-p-h2o-t-albedo", 33, REPO / "geometry_config_noaero_realistic.json", \
            ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k", "albedo"]
    return "co2-ch4-co-p-h2o-t-albedo", {1: 60, 3: 47}[pair], REPO / f"geometry_config_fpa0-{pair}_realistic.json", \
        ["co2_ppm", "ch4_ppb", "co_ppb", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k", "albedo"]


def truth(name, x, label):
    if name == "albedo":
        return np.asarray(als.SURFACE_FIELDS["albedo"](x, label))
    return np.asarray(als.STATE_FIELDS[name](x))


def load(pair, seed):
    tag, nw, geom, rows = config(pair)
    nsuf = f"_noise{seed}" if seed else ""
    sb_dir = (f"results/realistic_prior/gd_joint_block_whole_slit_fpa{pair}_gratio1_adens4_free-{tag}_nwin{nw}"
              f"_analytic_prior-realistic_valb_spos-anchor_fapos-anchor_etaslit{nsuf}")
    # single-band arm: by default the same band through the multi-band code (same anchor grid as the multi-band arm;
    # 2026-09-25, see scripts/compare_albedo_codepaths.py); --sb-driver = the original single-band driver's results
    SB_DRIVER = "--sb-driver" in sys.argv
    if SB_DRIVER:
        sb = pickle.load(open(glob.glob(str(REPO / sb_dir / "*.pkl"))[0], "rb"))["results"]
    mb = {}
    for g in glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-{pair}_r0-*_r{pair}-*_free-{tag}_cover_g1.0_etaslit_prior-realistic{nsuf}.pkl")):
        d = pickle.load(open(g, "rb"))
        mb[tuple(d["rows_by_fpa"][pair])] = d["joint"]["params"]
    out = {n: dict(sb=[], mb=[], prior=[]) for n in rows}
    for t in json.load(open(geom))["tiles"]:
        key = tuple(t["rows_by_fpa"][str(pair)])
        if SB_DRIVER:
            ps = sb[key]["hires"]["params"]
        else:
            ps = pickle.load(open(REPO / f"results/realistic_prior/multiband/mb_fpa{pair}_r{pair}-{key[0]}-{key[1]}_free-{tag}_cover_g1.0_etaslit_prior-realistic_geomcfg{nsuf}.pkl", "rb"))["joint"]["params"]
        pm = mb[key]
        for n in rows:
            mname = f"albedo_{LABELS[pair]}" if n == "albedo" else n
            sname = n if (SB_DRIVER or n != "albedo") else mname
            xs = np.asarray(ps[sname]["positions"]) * als.SLIT_HALF_KM
            xm = np.asarray(pm[mname]["positions"]) * als.SLIT_HALF_KM
            k = slice(1, -1)
            out[n]["sb"].append((np.asarray(ps[sname]["values"]) - truth(n, xs, LABELS[pair]))[k])
            out[n]["mb"].append((np.asarray(pm[mname]["values"]) - truth(n, xm, LABELS[pair]))[k])
            out[n]["prior"].append((np.asarray(ps[sname]["prior"]) - truth(n, xs, LABELS[pair]))[k])
    return {n: {q: np.concatenate(v) for q, v in d.items()} for n, d in out.items()}, rows


def pooled(pair, seeds):
    res = [load(pair, s) for s in seeds]
    rows = res[0][1]
    return {n: {q: np.concatenate([r[0][n][q] for r in res]) for q in ("sb", "mb", "prior")} for n in rows}, rows


for pair in (1, 2, 3):
    for cond, seeds in (("noiseless", [None]), ("noise-seeds123", [1, 2, 3])):
        data, rows = pooled(pair, seeds)
        nc = 2
        nr = (len(rows) + 1) // 2
        fig, axs = plt.subplots(nr, nc, figsize=(11, 3.1 * nr), squeeze=False, constrained_layout=True)
        for a, n in zip(axs.ravel(), rows):
            d = data[n]
            lim = np.percentile(np.abs(np.concatenate([d["sb"], d["mb"]])), 99.5) * 1.1
            lim = lim if lim > 0 else 1.0
            bins = np.linspace(-lim, lim, 61)
            for q, lab in (("prior", "prior"), ("sb", f"single-band FPA{pair}"), ("mb", f"multi-band FPA0+{pair}")):
                v = d[q]
                a.hist(v, bins=bins, histtype="step", lw=1.6, density=True, color=C[q],
                       label=f"{lab}: mean {v.mean():.3g}, sd {v.std():.3g}")
            a.set_title(NAMES[n] + " error", loc="left", fontsize=10)
            a.set_ylabel("density")
            a.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
            a.legend(frameon=False, fontsize=7, loc="best")
            a.grid(True, color="#e6e5e0", lw=0.8)
            for s_ in ("top", "right"):
                a.spines[s_].set_visible(False)
        for a in axs.ravel()[len(rows):]:
            a.set_visible(False)
        ncond = "noiseless" if cond == "noiseless" else "noisy, seeds 1-3 pooled"
        fig.suptitle(f"Error histograms (retrieved - truth): FPA{pair} single-band vs FPA0+FPA{pair}, {ncond}, realistic prior, no aerosol",
                     fontsize=10)
        out = REPO / f"plots/error_hist_fpa{pair}_{cond}_prior-realistic.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("saved", out.name, flush=True)
