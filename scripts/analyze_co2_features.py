"""CO2 prior / truth / posterior along the slit for the two-band sweep, near vs away from the CO2 features
(the -500 km plume, 6 ppm, 60 km wide; hot spots at -1100 km, 10 km, 5 ppm and +1050 km, 9 km, 4 ppm), plus where the
background error concentrates. Also makes plots/co2_multiband_prior_truth_posterior[_<prior>].png.

    PYTHONPATH=.:<gert> python scripts/analyze_co2_features.py [--prior realistic]
"""
import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.multiband_geometry import build_window_tiles_multiband  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402

PRIOR = sys.argv[sys.argv.index("--prior") + 1] if "--prior" in sys.argv else "structural"
PSUF = "" if PRIOR == "structural" else f"_prior-{PRIOR}"
feats = [(-500.0, 60.0, 6.0)] + list(als.HOTSPOTS_CO2)
tiles = build_window_tiles_multiband((0, 2), gjr.MIN_WINDOW, 1.0, 2)
by = {tuple(t.rows[0]): i for i, t in enumerate(tiles)}
X, P, V, TILE = [], [], [], []
for f in sorted(glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo_cover_g1.0_etaslit{PSUF}.pkl"))):
    d = pickle.load(open(f, "rb"))
    k = by.get(tuple(d["rows_by_fpa"][0]))
    if k is None:
        continue
    p = d["joint"]["params"]["co2_ppm"]
    x = np.asarray(p["positions"]) * als.SLIT_HALF_KM
    X += list(x); P += list(p["prior"]); V += list(p["values"]); TILE += [k] * len(x)
X, P, V, TILE = map(np.array, (X, P, V, TILE))
T = als.xco2_ppm(X)
ep, ev = P - T, V - T
z = np.min([np.abs(X - x0) / w for x0, w, a in feats], axis=0)
alb = np.asarray(als.SURFACE_FIELDS["albedo"](X, "CO2_strong"))
print(f"prior = {PRIOR}: {len(X)} bins from {len(set(TILE.tolist()))} tiles")
print(f"{'region':30s} {'n':>5s} {'prior rms':>10s} {'prior bias':>11s} {'post rms':>9s} {'post bias':>10s}")
for name, m in (("near  (< 1 feature width)", z < 1), ("edge  (1-3 widths)", (z >= 1) & (z < 3)), ("away  (>= 3 widths)", z >= 3)):
    print(f"{name:30s} {m.sum():5d} {np.sqrt(np.mean(ep[m] ** 2)):10.3f} {ep[m].mean():+11.3f} {np.sqrt(np.mean(ev[m] ** 2)):9.3f} {ev[m].mean():+10.3f}")
print("\nfraction of each feature's enhancement OVER THE PRIOR recovered (bins within 1 width):")
for x0, w, a in feats:
    m = np.abs(X - x0) < w
    if m.sum():
        enh, rec = T[m] - P[m], V[m] - P[m]
        print(f"  x0={x0:7.0f} km w={w:4.0f} amp={a:4.1f}: n={m.sum():3d}, truth-prior {enh.mean():+.2f}, posterior-prior {rec.mean():+.2f} -> {100 * rec.sum() / enh.sum():5.1f}%, posterior rms err {np.sqrt(np.mean(ev[m] ** 2)):.2f}")
away = z >= 3
print(f"\naway from features: posterior better than the prior in {100 * np.mean(np.abs(ev[away]) < np.abs(ep[away])):.0f}% of bins")
for lo, hi in ((0, 0.08), (0.08, 0.2), (0.2, 0.3), (0.3, 1.0)):
    m = away & (alb >= lo) & (alb < hi)
    if m.sum():
        print(f"  albedo(CO2_strong) {lo:.2f}-{hi:.2f}: n={m.sum():4d} posterior rms {np.sqrt(np.mean(ev[m] ** 2)):.3f}  prior rms {np.sqrt(np.mean(ep[m] ** 2)):.3f}")
xs = np.arange(-1400, 1401, 200)
for lo, hi in zip(xs[:-1], xs[1:]):
    m = away & (X >= lo) & (X < hi)
    if m.sum():
        print(f"  x {lo:6d}..{hi:5d} km: away bins {m.sum():4d}  posterior rms {np.sqrt(np.mean(ev[m] ** 2)):.3f}  prior rms {np.sqrt(np.mean(ep[m] ** 2)):.3f}")

C = {"truth": "#2a78d6", "prior": "#eb6834", "post": "#1baf7a"}
ink, sec, grid = "#0b0b0b", "#52514e", "#e6e5e0"
o = np.argsort(X)
xx = np.linspace(-1400, 1400, 3000)
prior_fn = als.PRIOR_FIELD_SETS[PRIOR]["co2_ppm"]
fig, ax = plt.subplots(3, 1, figsize=(11, 10.5), gridspec_kw=dict(height_ratios=[1.15, 1, 1]))
for a in ax:
    a.grid(True, color=grid, lw=0.8)
    for s_ in (a.spines["top"], a.spines["right"]):
        s_.set_visible(False)
    a.tick_params(colors=sec)
a = ax[0]
a.plot(xx, als.xco2_ppm(xx), color=C["truth"], lw=1.6, label="truth")
a.plot(xx, prior_fn(xx), color=C["prior"], lw=1.6, label=f"prior ({PRIOR})")
a.plot(X[o], V[o], "o", ms=2.6, color=C["post"], mew=0, label="posterior (retrieved bins)")
a.set_ylabel("CO$_2$ [ppm]", color=ink)
a.set_title(f"CO$_2$ along the slit: truth, {PRIOR} prior and two-band posterior", loc="left", color=ink, fontsize=12)
a.legend(frameon=False, loc="upper right", fontsize=9)
a = ax[1]
a.axhline(0, color="#c3c2b7", lw=1)
a.plot(xx, prior_fn(xx) - als.xco2_ppm(xx), color=C["prior"], lw=1.6, label="prior - truth")
a.plot(X[o], (V - T)[o], "o", ms=2.6, color=C["post"], mew=0, label="posterior - truth")
a.set_ylabel("error [ppm]", color=ink)
a.legend(frameon=False, loc="lower right", fontsize=9)
a = ax[2]
m = (X > -700) & (X < -300)
xz = xx[(xx > -700) & (xx < -300)]
a.plot(xz, als.xco2_ppm(xz), color=C["truth"], lw=1.6, label="truth")
a.plot(xz, prior_fn(xz), color=C["prior"], lw=1.6, label="prior")
a.plot(X[m], V[m], "o", ms=4, color=C["post"], mew=0, label="posterior")
a.set_xlabel("along-slit position [km]", color=ink)
a.set_ylabel("CO$_2$ [ppm]", color=ink)
a.set_title("Zoom on the -500 km plume", loc="left", color=ink, fontsize=11)
plt.tight_layout()
out = REPO / f"plots/co2_multiband_prior_truth_posterior{PSUF}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
