"""Prior / truth / posterior along the slit for every OTHER free row of the two-band aerosol-arm
sweep (p_surface, h2o, T, both albedo rows, amplitude_aerosol, height_aerosol) -- analyze_co2_
features.py's own counterpart for CO2. One two-panel plot per row (value + error), each row's
tile boundary anchors dropped (2026-09-22, user: least redundantly covered, see summarize_
multiband_sweep.py's own DROP_BOUNDARY).

    PYTHONPATH=.:<gert> python scripts/plot_multiband_slit_state.py [--prior realistic] [--aerosol]
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
AEROSOL = "--aerosol" in sys.argv
PSUF = "" if PRIOR == "structural" else f"_prior-{PRIOR}"
ASUF = "_aero" if AEROSOL else ""
FREE_TAG = "co2-p-h2o-t-albedo-amplitude-height" if AEROSOL else "co2-p-h2o-t-albedo"

ROWS = [("p_surface_hpa", None, "surface pressure [hPa]"), ("h2o_surface_vmr", None, "H$_2$O surface vmr"),
        ("t_offset_k", None, "T offset [K]"), ("albedo_O2_A", "O2_A", "albedo (O$_2$-A)"),
        ("albedo_CO2_strong", "CO2_strong", "albedo (CO$_2$ strong)")]
if AEROSOL:
    ROWS += [("amplitude_aerosol", None, "aerosol amplitude"), ("height_aerosol", None, "aerosol height [Pa]")]

tiles = build_window_tiles_multiband((0, 2), gjr.MIN_WINDOW, 1.0, 2)
by = {tuple(t.rows[0]): i for i, t in enumerate(tiles)}
files = sorted(glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-{FREE_TAG}_cover_g1.0_etaslit{PSUF}{ASUF}.pkl")))
res = []
for f in files:
    d = pickle.load(open(f, "rb"))
    if tuple(d["rows_by_fpa"][0]) in by:
        res.append(d)
print(f"{len(res)}/{len(tiles)} tiles present")


def truth_fn(x, name, label):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x, label))
    if name in als.SURFACE_FIELDS:
        return np.asarray(als.SURFACE_FIELDS[name](x))
    return np.asarray(als.STATE_FIELDS[name](x))


def prior_fn(x, name, label):
    key = "albedo" if label else name
    fields = als.SURFACE_PRIOR_FIELD_SETS.get(PRIOR, als.SURFACE_FIELDS) if (label or name in als.SURFACE_FIELDS) \
        else als.PRIOR_FIELD_SETS.get(PRIOR, als.STATE_FIELDS)
    fn = fields.get(key, als.SURFACE_FIELDS.get(key) if label else als.STATE_FIELDS.get(key))
    return np.asarray(fn(x, label)) if label else np.asarray(fn(x))


C = {"truth": "#2a78d6", "prior": "#eb6834", "post": "#1baf7a"}
ink, sec, grid = "#0b0b0b", "#52514e", "#e6e5e0"
xx = np.linspace(-1400, 1400, 3000)

for name, label, ylabel in ROWS:
    X, P, V = [], [], []
    for d in res:
        p = d["joint"]["params"][name]
        x = np.asarray(p["positions"]) * als.SLIT_HALF_KM
        pr, vv = np.asarray(p["prior"]), np.asarray(p["values"])
        if x.size > 2:
            x, pr, vv = x[1:-1], pr[1:-1], vv[1:-1]
        X += list(x); P += list(pr); V += list(vv)
    X, P, V = map(np.array, (X, P, V))
    o = np.argsort(X)
    T = truth_fn(X, name, label)
    ep, ev = P - T, V - T
    ra, rp = np.sqrt(np.mean(ev ** 2)), np.sqrt(np.mean(ep ** 2))
    print(f"{ylabel:28s} n={X.size:5d}  retrieved rms {ra:.4g}  prior rms {rp:.4g}")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw=dict(height_ratios=[1.2, 1]))
    for a in ax:
        a.grid(True, color=grid, lw=0.8)
        for s_ in (a.spines["top"], a.spines["right"]):
            s_.set_visible(False)
        a.tick_params(colors=sec)
    a = ax[0]
    a.plot(xx, truth_fn(xx, name, label), color=C["truth"], lw=1.6, label="truth")
    a.plot(xx, prior_fn(xx, name, label), color=C["prior"], lw=1.6, label=f"prior ({PRIOR})")
    a.plot(X[o], V[o], "o", ms=2.6, color=C["post"], mew=0, label="posterior (retrieved bins)")
    a.set_ylabel(ylabel, color=ink)
    title_suffix = " (aerosol arm)" if AEROSOL else ""
    a.set_title(f"{ylabel} along the slit: truth, {PRIOR} prior and two-band posterior{title_suffix}",
               loc="left", color=ink, fontsize=12)
    a.legend(frameon=False, loc="upper right", fontsize=9)
    a = ax[1]
    a.axhline(0, color="#c3c2b7", lw=1)
    a.plot(xx, prior_fn(xx, name, label) - truth_fn(xx, name, label), color=C["prior"], lw=1.6,
          label=f"prior - truth (rms {rp:.3g})")
    a.plot(X[o], ev[o], "o", ms=2.6, color=C["post"], mew=0, label=f"posterior - truth (rms {ra:.3g})")
    a.set_xlabel("along-slit position [km]", color=ink)
    a.set_ylabel("error", color=ink)
    a.legend(frameon=False, loc="lower right", fontsize=9)
    plt.tight_layout()
    out = REPO / f"plots/{name}_multiband_prior_truth_posterior{PSUF}{ASUF}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
