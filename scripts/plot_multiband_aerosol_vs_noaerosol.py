"""No-aerosol vs. aerosol multiband arms, directly overlaid (2026-09-23, user: "the no-aerosol
vs. aerosol, multi-band results are key" comparison discussed). Both arms share IDENTICAL bin
positions per tile (checked directly -- same fpas/g_ratio/overlap/tiling, only --free/--aerosol/
--solver differ), so this overlays real bin-for-bin values with no interpolation, unlike the
single-vs-multiband comparison which needs a geometry config.

One two-panel plot per shared row (value + error), truth/prior common to both, two posterior
scatter series (no-aerosol single_scatter vs aerosol xrtm) so the aerosol confound is visible
directly. Boundary bins dropped per DROP_BOUNDARY convention.

    PYTHONPATH=.:<gert> python scripts/plot_multiband_aerosol_vs_noaerosol.py [--prior realistic]
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

PRIOR = sys.argv[sys.argv.index("--prior") + 1] if "--prior" in sys.argv else "structural"
PSUF = "" if PRIOR == "structural" else f"_prior-{PRIOR}"

ROWS = [("co2_ppm", None, "CO$_2$ [ppm]"), ("p_surface_hpa", None, "surface pressure [hPa]"),
        ("h2o_surface_vmr", None, "H$_2$O surface vmr"), ("t_offset_k", None, "T offset [K]"),
        ("albedo_O2_A", "O2_A", "albedo (O$_2$-A)"), ("albedo_CO2_strong", "CO2_strong", "albedo (CO$_2$ strong)")]


def truth_fn(x, name, label):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x, label))
    if name in als.SURFACE_FIELDS:
        return np.asarray(als.SURFACE_FIELDS[name](x))
    return np.asarray(als.STATE_FIELDS[name](x))


def load_arm(free_tag, aero_suf):
    files = sorted(glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-{free_tag}_cover_g1.0_etaslit{PSUF}{aero_suf}.pkl")))
    return [pickle.load(open(f, "rb")) for f in files]


noaero = load_arm("co2-p-h2o-t-albedo", "")
aero = load_arm("co2-p-h2o-t-albedo-amplitude-height", "_aero")
print(f"{len(noaero)} no-aerosol tiles, {len(aero)} aerosol tiles")
by_rows_aero = {tuple(d["rows_by_fpa"][0]): d for d in aero}

C = {"truth": "#2a78d6", "prior": "#eb6834", "noaero": "#1baf7a", "aero": "#c0392b"}
ink, sec, grid = "#0b0b0b", "#52514e", "#e6e5e0"
xx = np.linspace(-1400, 1400, 3000)

for name, label, ylabel in ROWS:
    Xn, Pn, Vn, Va = [], [], [], []
    for dn in noaero:
        da = by_rows_aero.get(tuple(dn["rows_by_fpa"][0]))
        if da is None:
            continue
        pn, pa = dn["joint"]["params"][name], da["joint"]["params"][name]
        x = np.asarray(pn["positions"]) * als.SLIT_HALF_KM
        if x.size <= 2:
            continue
        keep = slice(1, -1)
        Xn += list(x[keep]); Pn += list(np.asarray(pn["prior"])[keep])
        Vn += list(np.asarray(pn["values"])[keep]); Va += list(np.asarray(pa["values"])[keep])
    Xn, Pn, Vn, Va = map(np.array, (Xn, Pn, Vn, Va))
    o = np.argsort(Xn)
    T = truth_fn(Xn, name, label)
    en, ea, ep = Vn - T, Va - T, Pn - T
    rn, ra, rp = (np.sqrt(np.mean(e ** 2)) for e in (en, ea, ep))
    print(f"{ylabel:24s} no-aero rms {rn:.4g}  aero rms {ra:.4g}  prior rms {rp:.4g}")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw=dict(height_ratios=[1.2, 1]))
    for a_ in ax:
        a_.grid(True, color=grid, lw=0.8)
        for s_ in (a_.spines["top"], a_.spines["right"]):
            s_.set_visible(False)
        a_.tick_params(colors=sec)
    a_ = ax[0]
    a_.plot(xx, truth_fn(xx, name, label), color=C["truth"], lw=1.6, label="truth")
    a_.plot(xx, [np.nan] * len(xx), color=C["prior"], lw=1.6, label=f"prior ({PRIOR})")   # legend only (prior scatter below)
    a_.plot(Xn[o], Pn[o], ".", ms=1.5, color=C["prior"], alpha=0.5, mew=0)
    a_.plot(Xn[o], Vn[o], "o", ms=2.6, color=C["noaero"], mew=0, label="posterior, no aerosol (single_scatter)")
    a_.plot(Xn[o], Va[o], "o", ms=2.6, color=C["aero"], mew=0, alpha=0.7, label="posterior, aerosol (xrtm)")
    a_.set_ylabel(ylabel, color=ink)
    a_.set_title(f"{ylabel}: no-aerosol vs. aerosol multiband posterior ({PRIOR} prior)", loc="left", color=ink, fontsize=12)
    a_.legend(frameon=False, loc="upper right", fontsize=8)
    a_ = ax[1]
    a_.axhline(0, color="#c3c2b7", lw=1)
    a_.plot(Xn[o], en[o], "o", ms=2.6, color=C["noaero"], mew=0, label=f"no aerosol - truth (rms {rn:.3g})")
    a_.plot(Xn[o], ea[o], "o", ms=2.6, color=C["aero"], mew=0, alpha=0.7, label=f"aerosol - truth (rms {ra:.3g})")
    a_.set_xlabel("along-slit position [km]", color=ink)
    a_.set_ylabel("error", color=ink)
    a_.legend(frameon=False, loc="lower right", fontsize=8)
    plt.tight_layout()
    out = REPO / f"plots/{name}_multiband_aerosol_vs_noaerosol{PSUF}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
