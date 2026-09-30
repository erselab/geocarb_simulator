"""4-band per-arm state-error summaries, reusing this project's established conventions rather than a new
design (2026-09-30, user: "repeat the former plotting scripts that show all the state variable errors on a
single panel plot oriented vertically. Also show the errors as histograms. Please also do the pairplot for
each case."):

  * VERTICAL panel -- one row of thin panels sharing along-slit position [km] on the Y axis (the orientation
    `plot_state_with_fpa_images.py` uses), one panel per free state row, X = error (retrieved - truth).
  * HISTOGRAMS -- step histograms of pooled error per row, grid layout, prior error overlaid for scale (the
    convention `plot_error_histograms.py` uses).
  * PAIRPLOT -- seaborn pairwise error-correlation grid (the convention `plot_multiband_error_pairplot.py`
    uses): diagonal histograms, lower triangle scattered/colored by along-slit position, upper triangle
    annotated with Pearson r.

One set of all three per ARM (no-aerosol, no-aerosol+noise1, aerosol=smoke, aerosol=smoke+noise1 by default;
pass --sulfate to do the two sulfate arms instead once those sweeps finish).

    PYTHONPATH=.:<gert> python scripts/plot_mb4_state_summary.py [--sulfate]
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

DIR = REPO / "results/realistic_prior/multiband"
BASE = "mb_fpa0-1-2-3_r0-*_r1-*_r2-*_r3-*_free-co2-ch4-co-p-h2o-t-albedo"
SUF = "_cover_g1.0_etaslit_prior-realistic"
AERO_TYPE = "sulfate" if "--sulfate" in sys.argv else "smoke"

ARMS = {
    "no-aerosol":              f"{DIR}/{BASE}{SUF}.pkl",
    "no-aerosol+noise1":       f"{DIR}/{BASE}{SUF}_noise1.pkl",
    f"aerosol({AERO_TYPE})":         f"{DIR}/{BASE}-amplitude-height{SUF}_aero_{AERO_TYPE}_lmfast.pkl",
    f"aerosol({AERO_TYPE})+noise1":  f"{DIR}/{BASE}-amplitude-height{SUF}_aero_{AERO_TYPE}_noise1_lmfast.pkl",
}

SHARED_ROWS = [("co2_ppm", "CO$_2$ [ppm]"), ("ch4_ppb", "CH$_4$ [ppb]"), ("co_ppb", "CO [ppb]"),
               ("p_surface_hpa", "p [hPa]"), ("h2o_surface_vmr", "h2o (vmr)"), ("t_offset_k", "T [K]"),
               ("amplitude_aerosol", "amp_aer"), ("height_aerosol", "height_aer [Pa]")]
ALBEDO_ROWS = [("albedo_O2_A", "O2_A", "albedo O2_A"), ("albedo_CO2_weak", "CO2_weak", "albedo CO2_weak"),
               ("albedo_CO2_strong", "CO2_strong", "albedo CO2_strong"), ("albedo_CH4_CO", "CH4_CO", "albedo CH4/CO")]


def truth_of(name, x_km, label=None):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x_km, label))
    if name in als.SURFACE_FIELDS:
        return np.asarray(als.SURFACE_FIELDS[name](x_km))
    return np.asarray(als.STATE_FIELDS[name](x_km))


def load(pattern):
    return {tuple(pickle.load(open(f, "rb"))["rows_by_fpa"][0]): pickle.load(open(f, "rb"))
            for f in glob.glob(pattern)}


def build_frame(tiles):
    """One pooled DataFrame per arm: shared coarse rows interpolated onto albedo's own fine grid per tile
    (same convention as plot_multiband_error_pairplot.py), plus a prior-error twin set of columns."""
    rows_out, prior_out = [], []
    for d in tiles.values():
        params = d["joint"]["params"]
        if "albedo_O2_A" not in params:
            continue
        x_fine = np.asarray(params["albedo_O2_A"]["positions"], dtype=float)[1:-1] * als.SLIT_HALF_KM
        if x_fine.size <= 2:
            continue
        tile_err, tile_prior = {}, {}
        for name, label, _ in ALBEDO_ROWS:
            if name not in params:
                continue
            p = params[name]
            v = np.asarray(p["values"], dtype=float)[1:-1]
            pr = np.asarray(p["prior"], dtype=float)[1:-1]
            t = truth_of(name, x_fine, label)
            tile_err[name] = v - t
            tile_prior[name] = pr - t
        for name, _ in SHARED_ROWS:
            if name not in params:
                continue
            p = params[name]
            x_coarse = np.asarray(p["positions"], dtype=float) * als.SLIT_HALF_KM
            v = np.asarray(p["values"], dtype=float)
            pr = np.asarray(p["prior"], dtype=float)
            t = truth_of(name, x_coarse)
            e_coarse = (v - t)[1:-1]
            p_coarse = (pr - t)[1:-1]
            xc = x_coarse[1:-1]
            tile_err[name] = np.interp(x_fine, xc, e_coarse)
            tile_prior[name] = np.interp(x_fine, xc, p_coarse)
        tile_err["along-slit [km]"] = x_fine
        rows_out.append(pd.DataFrame(tile_err))
        prior_out.append(pd.DataFrame(tile_prior))
    return pd.concat(rows_out, ignore_index=True), pd.concat(prior_out, ignore_index=True)


LABELS = {n: lab for n, lab in SHARED_ROWS}
LABELS.update({n: lab for n, _, lab in ALBEDO_ROWS})

arm_frames = {}
for arm, pat in ARMS.items():
    tiles = load(pat)
    print(f"{arm}: {len(tiles)} tiles")
    if not tiles:
        continue
    err, prior = build_frame(tiles)
    arm_frames[arm] = (err, prior)

# ============================================================ 1. VERTICAL panel, one figure per arm =====
for arm, (err, prior) in arm_frames.items():
    rows = [n for n, _ in SHARED_ROWS if n in err.columns] + [n for n, _, _ in ALBEDO_ROWS if n in err.columns]
    fig, axs = plt.subplots(1, len(rows), figsize=(1.7 * len(rows), 9), sharey=True)
    for a, name in zip(np.atleast_1d(axs), rows):
        a.scatter(err[name], err["along-slit [km]"], s=2, alpha=0.25, color="#1baf7a", label="error")
        a.axvline(0.0, color="k", lw=0.8, alpha=0.5)
        a.set_title(LABELS[name], fontsize=9)
        a.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        a.tick_params(axis="x", labelsize=7, rotation=45)
    np.atleast_1d(axs)[0].set_ylabel("along-slit position [km]")
    fig.suptitle(f"State errors (retrieved - truth) vs. along-slit position -- {arm}", fontsize=11)
    fig.tight_layout()
    out = REPO / f"plots/mb4_vertical_panel_{arm.replace('(', '').replace(')', '').replace('+', '_')}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.name)

# ============================================================ 2. HISTOGRAMS, one figure per arm =========
for arm, (err, prior) in arm_frames.items():
    rows = [n for n, _ in SHARED_ROWS if n in err.columns] + [n for n, _, _ in ALBEDO_ROWS if n in err.columns]
    nc = 2
    nr = (len(rows) + 1) // 2
    fig, axs = plt.subplots(nr, nc, figsize=(11, 3.0 * nr), squeeze=False, constrained_layout=True)
    for a, name in zip(axs.ravel(), rows):
        v, p = err[name].to_numpy(), prior[name].to_numpy()
        lim = np.percentile(np.abs(v), 99.5) * 1.1
        lim = lim if lim > 0 else 1.0
        bins = np.linspace(-lim, lim, 61)
        a.hist(p, bins=bins, histtype="step", lw=1.4, density=True, color="#eb6834",
              label=f"prior: mean {p.mean():.3g}, sd {p.std():.3g}")
        a.hist(v, bins=bins, histtype="step", lw=1.6, density=True, color="#1baf7a",
              label=f"retrieved: mean {v.mean():.3g}, sd {v.std():.3g}")
        a.set_title(LABELS[name] + " error", loc="left", fontsize=10)
        a.set_ylabel("density")
        a.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        a.legend(frameon=False, fontsize=7, loc="best")
        a.grid(True, color="#e6e5e0", lw=0.8)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    for a in axs.ravel()[len(rows):]:
        a.set_visible(False)
    fig.suptitle(f"Error histograms (retrieved - truth) -- {arm}", fontsize=11)
    out = REPO / f"plots/mb4_error_hist_{arm.replace('(', '').replace(')', '').replace('+', '_')}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.name)

# ============================================================ 3. PAIRPLOT, one figure per arm ===========
ETA_NORM = plt.Normalize(vmin=-1400, vmax=1400)


def annotate_corr(x, y, **kws):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.set_facecolor(plt.cm.RdBu_r((r + 1) / 2, alpha=0.25))
    ax.annotate(f"{x.name}\nvs\n{y.name}\nr = {r:+.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
               ha="center", va="center", fontsize=7 + 6 * abs(r), fontweight="bold" if abs(r) > 0.5 else "normal")


def scatter_by_eta(x, y, **kws):
    c = df.loc[x.index, "along-slit [km]"]
    plt.scatter(x, y, c=c, cmap="coolwarm", norm=ETA_NORM, s=6, alpha=0.4, edgecolor="none")


for arm, (err, prior) in arm_frames.items():
    rows = [n for n, _ in SHARED_ROWS if n in err.columns] + [n for n, _, _ in ALBEDO_ROWS if n in err.columns]
    df = err.rename(columns=LABELS)
    plot_vars = [LABELS[n] for n in rows]
    print(f"\n{arm} correlation matrix:")
    print(df[plot_vars].corr().round(3).to_string())
    g = sns.PairGrid(df, vars=plot_vars)
    g.map_diag(plt.hist, bins=40)
    g.map_lower(scatter_by_eta)
    g.map_upper(annotate_corr)
    g.figure.suptitle(f"Error correlations -- {arm}", y=1.01)
    g.figure.colorbar(plt.cm.ScalarMappable(norm=ETA_NORM, cmap="coolwarm"), ax=g.axes, shrink=0.5,
                      label="along-slit position [km]", location="right", pad=0.02)
    out = REPO / f"plots/mb4_error_pairplot_{arm.replace('(', '').replace(')', '').replace('+', '_')}.png"
    g.figure.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    print("saved", out.name)
