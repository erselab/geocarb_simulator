"""Single-band (FPA2) vs multi-band (FPA0+FPA2) retrievals of the SAME free set (co2/p/h2o/T/albedo,
no aerosol, single_scatter, realistic prior), directly overlaid (2026-09-24). The single-band sweep
was run with --geometry-config exported from the multi-band run, so both share IDENTICAL tiles and
bin_centers (verified 33/33 tiles, bit-identical `positions` per row) -- co2/p/h2o/T are compared
bin-for-bin with no interpolation. Albedo sits on a different grid in each (single-band 'albedo',
~121 points/tile; multi-band 'albedo_CO2_strong', its own anchor grid) so it is overlaid only,
not bin-matched. Boundary bins dropped per DROP_BOUNDARY (2026-09-22).

Outputs (plots/): <row>_singleband_vs_multiband_prior-realistic.png per row (value + error panels),
singleband_vs_multiband_error_scatter_prior-realistic.png (per-bin single-band error vs multi-band
error, colored by along-slit position, 1:1 line).

    PYTHONPATH=.:<gert> python scripts/plot_singleband_vs_multiband.py
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

SB_DIR = "results/realistic_prior/gd_joint_block_whole_slit_fpa2_gratio1_adens4_free-co2-p-h2o-t-albedo_nwin33_analytic_prior-realistic_valb_spos-anchor_fapos-anchor_etaslit"
GEOM = REPO / "geometry_config_noaero_realistic.json"
ROWS = [("co2_ppm", "CO$_2$ [ppm]"), ("p_surface_hpa", "surface pressure [hPa]"),
        ("h2o_surface_vmr", "H$_2$O surface vmr"), ("t_offset_k", "T offset [K]")]

sb = pickle.load(open(glob.glob(str(REPO / SB_DIR / "*.pkl"))[0], "rb"))["results"]
geom = json.load(open(GEOM))["tiles"]
mb = {}
for g in glob.glob(str(REPO / "results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo_cover_g1.0_etaslit_prior-realistic.pkl")):
    d = pickle.load(open(g, "rb"))
    mb[tuple(d["rows_by_fpa"][2])] = d["joint"]["params"]
tile_params = []
for t in geom:
    rows = tuple(t["rows_by_fpa"]["2"])
    tile_params.append((sb[rows]["hires"]["params"], mb[rows]))
print(f"{len(tile_params)} tiles (single-band and multi-band, identical bins)")


def truth(name, x, label=None):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x, label))
    return np.asarray(als.STATE_FIELDS[name](x))


C = {"truth": "#2a78d6", "prior": "#eb6834", "sb": "#8e44ad", "mb": "#1baf7a"}
ink, sec, grid = "#0b0b0b", "#52514e", "#e6e5e0"
xx = np.linspace(-1400, 1400, 3000)


def style(axes):
    for a in axes:
        a.grid(True, color=grid, lw=0.8)
        for s_ in (a.spines["top"], a.spines["right"]):
            s_.set_visible(False)
        a.tick_params(colors=sec)


pooled = {}
for name, ylabel in ROWS:
    X, P, S, M = [], [], [], []
    for ps, pm in tile_params:
        assert np.array_equal(ps[name]["positions"], pm[name]["positions"]), name
        x = np.asarray(ps[name]["positions"]) * als.SLIT_HALF_KM
        k = slice(1, -1)
        X += list(x[k]); P += list(np.asarray(ps[name]["prior"])[k])
        S += list(np.asarray(ps[name]["values"])[k]); M += list(np.asarray(pm[name]["values"])[k])
    X, P, S, M = map(np.array, (X, P, S, M))
    T = truth(name, X)
    es, em, ep = S - T, M - T, P - T
    rs, rm, rp = (np.sqrt(np.mean(e ** 2)) for e in (es, em, ep))
    pooled[name] = (X, es, em)
    print(f"{ylabel:24s} single-band rms {rs:.4g}  multi-band rms {rm:.4g}  prior rms {rp:.4g}  "
          f"(multi-band better in {100 * np.mean(np.abs(em) < np.abs(es)):.0f}% of bins)")
    o = np.argsort(X)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw=dict(height_ratios=[1.2, 1]))
    style(ax)
    a = ax[0]
    a.plot(xx, truth(name, xx), color=C["truth"], lw=1.6, label="truth")
    a.plot(X[o], P[o], ".", ms=1.5, color=C["prior"], alpha=0.5, mew=0, label="prior (realistic)")
    a.plot(X[o], S[o], "o", ms=2.6, color=C["sb"], mew=0, label="posterior, single-band (FPA2)")
    a.plot(X[o], M[o], "o", ms=2.6, color=C["mb"], mew=0, alpha=0.7, label="posterior, multi-band (FPA0+FPA2)")
    a.set_ylabel(ylabel, color=ink)
    a.set_title(f"{ylabel}: single-band vs. multi-band, same bins (no aerosol, single_scatter, realistic prior)",
               loc="left", color=ink, fontsize=12)
    a.legend(frameon=False, loc="upper right", fontsize=8)
    a = ax[1]
    a.axhline(0, color="#c3c2b7", lw=1)
    a.plot(X[o], es[o], "o", ms=2.6, color=C["sb"], mew=0, label=f"single-band - truth (rms {rs:.3g})")
    a.plot(X[o], em[o], "o", ms=2.6, color=C["mb"], mew=0, alpha=0.7, label=f"multi-band - truth (rms {rm:.3g})")
    a.set_xlabel("along-slit position [km]", color=ink)
    a.set_ylabel("error", color=ink)
    a.legend(frameon=False, loc="lower right", fontsize=8)
    plt.tight_layout()
    out = REPO / f"plots/{name}_singleband_vs_multiband_prior-realistic.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)

# albedo: overlay only (different grids)
Xs, Es, Xm, Em = [], [], [], []
for ps, pm in tile_params:
    xs = np.asarray(ps["albedo"]["positions"]) * als.SLIT_HALF_KM
    xm = np.asarray(pm["albedo_CO2_strong"]["positions"]) * als.SLIT_HALF_KM
    Xs += list(xs[1:-1]); Es += list((np.asarray(ps["albedo"]["values"]) - truth("albedo", xs, "CO2_strong"))[1:-1])
    Xm += list(xm[1:-1]); Em += list((np.asarray(pm["albedo_CO2_strong"]["values"]) - truth("albedo", xm, "CO2_strong"))[1:-1])
Xs, Es, Xm, Em = map(np.array, (Xs, Es, Xm, Em))
rs, rm = np.sqrt(np.mean(Es ** 2)), np.sqrt(np.mean(Em ** 2))
print(f"{'albedo (CO2 strong)':24s} single-band rms {rs:.4g}  multi-band rms {rm:.4g}  (different grids -- overlay only)")
fig, ax = plt.subplots(figsize=(11, 4.5))
style([ax])
ax.axhline(0, color="#c3c2b7", lw=1)
ax.plot(Xs, Es, "o", ms=1.8, color=C["sb"], mew=0, alpha=0.6, label=f"single-band - truth (rms {rs:.3g}, {Xs.size} pts)")
ax.plot(Xm, Em, "o", ms=1.8, color=C["mb"], mew=0, alpha=0.6, label=f"multi-band - truth (rms {rm:.3g}, {Xm.size} pts)")
ax.set_xlabel("along-slit position [km]", color=ink)
ax.set_ylabel("albedo (CO$_2$ strong) error", color=ink)
ax.set_title("Albedo error: single-band vs. multi-band (each on its own grid -- not bin-matched)", loc="left", color=ink, fontsize=12)
ax.legend(frameon=False, loc="upper right", fontsize=8)
plt.tight_layout()
out = REPO / "plots/albedo_CO2_strong_singleband_vs_multiband_prior-realistic.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print("  saved", out)

# per-bin scatter grid
fig, axs = plt.subplots(2, 2, figsize=(11, 10.5), constrained_layout=True)
for a, (name, ylabel) in zip(axs.ravel(), ROWS):
    X, es, em = pooled[name]
    sc = a.scatter(em, es, c=X, cmap="coolwarm", vmin=-1400, vmax=1400, s=10, alpha=0.6, edgecolor="none")
    lim = max(np.abs(em).max(), np.abs(es).max()) * 1.05
    a.plot([-lim, lim], [-lim, lim], color="#52514e", lw=1, ls="--")
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel(f"multi-band error", color=ink); a.set_ylabel(f"single-band error", color=ink)
    r = np.corrcoef(em, es)[0, 1]
    a.set_title(f"{ylabel}  \n(r = {r:+.2f}; multi-band closer in {100 * np.mean(np.abs(em) < np.abs(es)):.0f}%)",
               loc="left", fontsize=9, color=ink, pad=8)
    style([a])
fig.colorbar(sc, ax=axs, shrink=0.6, label="along-slit position [km]")
fig.suptitle("Per-bin retrieval error: single-band vs. multi-band (identical bins; points above the dashed 1:1 line = single-band worse)",
             fontsize=10)
out = REPO / "plots/singleband_vs_multiband_error_scatter_prior-realistic.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
