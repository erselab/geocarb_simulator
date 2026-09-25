"""Single-band (FPA2; or FPA1/FPA3 with --pair) vs multi-band (FPA0+FPA<same>) retrievals of the SAME free set (co2/p/h2o/T/albedo,
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

# --pair 1|3 (2026-09-24): single-band FPA<pair> vs multi-band FPA0+FPA<pair>, free set
# co2,ch4,co,p,h2o,T,albedo (no aerosol). Default (no --pair) = the original FPA2 vs FPA0+FPA2 comparison.
PAIR = int(sys.argv[sys.argv.index("--pair") + 1]) if "--pair" in sys.argv else 2
NZ = int(sys.argv[sys.argv.index("--noise-seed") + 1]) if "--noise-seed" in sys.argv else None   # noisy-observation runs (2026-09-25)
NSUF = f"_noise{NZ}" if NZ is not None else ""
LABEL = {1: "CO2_weak", 2: "CO2_strong", 3: "CH4_CO"}[PAIR]
if PAIR == 2:
    FREE_TAG, NW, GEOM = "co2-p-h2o-t-albedo", 33, REPO / "geometry_config_noaero_realistic.json"
    OUT = "singleband_vs_multiband"
else:
    FREE_TAG, NW, GEOM = "co2-ch4-co-p-h2o-t-albedo", {1: 60, 3: 47}[PAIR], REPO / f"geometry_config_fpa0-{PAIR}_realistic.json"
    OUT = f"fpa{PAIR}_singleband_vs_multiband"
SB_DIR = f"results/realistic_prior/gd_joint_block_whole_slit_fpa{PAIR}_gratio1_adens4_free-{FREE_TAG}_nwin{NW}_analytic_prior-realistic_valb_spos-anchor_fapos-anchor_etaslit{NSUF}"
BAND = f"FPA{PAIR}"
MBTAG = f"FPA0+{BAND}"
ROWS = [("co2_ppm", "CO$_2$ [ppm]")]
if PAIR != 2:
    ROWS += [("ch4_ppb", "CH$_4$ [ppb]"), ("co_ppb", "CO [ppb]")]
ROWS += [("p_surface_hpa", "surface pressure [hPa]"),
         ("h2o_surface_vmr", "H$_2$O surface vmr"), ("t_offset_k", "T offset [K]")]

# Single-band arm (2026-09-25): by default the SAME single band run through the multi-band code (gd_multiband_window.py
# --fpas <n> --geometry-config ...), so both arms share the code path and "cover" anchor grid; --sb-driver uses the
# original single-band driver's results instead (its different anchor grid inflated the apparent single-band albedo
# advantage; see scripts/compare_albedo_codepaths.py).
SB_DRIVER = "--sb-driver" in sys.argv
if SB_DRIVER:
    sb = pickle.load(open(glob.glob(str(REPO / SB_DIR / "*.pkl"))[0], "rb"))["results"]
ALB_SB = "albedo" if SB_DRIVER else f"albedo_{LABEL}"


def sb_params(rows):
    if SB_DRIVER:
        return sb[rows]["hires"]["params"]
    f = REPO / f"results/realistic_prior/multiband/mb_fpa{PAIR}_r{PAIR}-{rows[0]}-{rows[1]}_free-{FREE_TAG}_cover_g1.0_etaslit_prior-realistic_geomcfg{NSUF}.pkl"
    return pickle.load(open(f, "rb"))["joint"]["params"]
geom = json.load(open(GEOM))["tiles"]
mb = {}
for g in glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-{PAIR}_r0-*_r{PAIR}-*_free-{FREE_TAG}_cover_g1.0_etaslit_prior-realistic{NSUF}.pkl")):
    d = pickle.load(open(g, "rb"))
    mb[tuple(d["rows_by_fpa"][PAIR])] = d["joint"]["params"]
tile_params = []
for t in geom:
    rows = tuple(t["rows_by_fpa"][str(PAIR)])
    tile_params.append((sb_params(rows), mb[rows]))
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
    a.plot(X[o], S[o], "o", ms=2.6, color=C["sb"], mew=0, label=f"posterior, single-band ({BAND})")
    a.plot(X[o], M[o], "o", ms=2.6, color=C["mb"], mew=0, alpha=0.7, label=f"posterior, multi-band ({MBTAG})")
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
    out = REPO / f"plots/{name}_{OUT}_prior-realistic{NSUF}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)

# albedo: overlay only (different grids)
Xs, Es, Xm, Em = [], [], [], []
for ps, pm in tile_params:
    xs = np.asarray(ps[ALB_SB]["positions"]) * als.SLIT_HALF_KM
    xm = np.asarray(pm[f"albedo_{LABEL}"]["positions"]) * als.SLIT_HALF_KM
    Xs += list(xs[1:-1]); Es += list((np.asarray(ps[ALB_SB]["values"]) - truth("albedo", xs, LABEL))[1:-1])
    Xm += list(xm[1:-1]); Em += list((np.asarray(pm[f"albedo_{LABEL}"]["values"]) - truth("albedo", xm, LABEL))[1:-1])
Xs, Es, Xm, Em = map(np.array, (Xs, Es, Xm, Em))
rs, rm = np.sqrt(np.mean(Es ** 2)), np.sqrt(np.mean(Em ** 2))
print(f"{f'albedo ({LABEL})':24s} single-band rms {rs:.4g}  multi-band rms {rm:.4g}  (different grids -- overlay only)")
fig, ax = plt.subplots(figsize=(11, 4.5))
style([ax])
ax.axhline(0, color="#c3c2b7", lw=1)
ax.plot(Xs, Es, "o", ms=1.8, color=C["sb"], mew=0, alpha=0.6, label=f"single-band - truth (rms {rs:.3g}, {Xs.size} pts)")
ax.plot(Xm, Em, "o", ms=1.8, color=C["mb"], mew=0, alpha=0.6, label=f"multi-band - truth (rms {rm:.3g}, {Xm.size} pts)")
ax.set_xlabel("along-slit position [km]", color=ink)
ax.set_ylabel("albedo ({LABEL}) error", color=ink)
ax.set_title("Albedo error: single-band vs. multi-band (each on its own grid -- not bin-matched)", loc="left", color=ink, fontsize=12)
ax.legend(frameon=False, loc="upper right", fontsize=8)
plt.tight_layout()
out = REPO / f"plots/albedo_{LABEL}_{OUT}_prior-realistic{NSUF}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print("  saved", out)

# per-bin scatter grid
nr = (len(ROWS) + 1) // 2
fig, axs = plt.subplots(nr, 2, figsize=(11, 5.25 * nr), constrained_layout=True, squeeze=False)
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
for a in axs.ravel()[len(ROWS):]:
    a.set_visible(False)
fig.colorbar(sc, ax=axs, shrink=0.6, label="along-slit position [km]")
fig.suptitle("Per-bin retrieval error: single-band vs. multi-band (identical bins; points above the dashed 1:1 line = single-band worse)",
             fontsize=10)
out = REPO / f"plots/{OUT}_error_scatter_prior-realistic{NSUF}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
