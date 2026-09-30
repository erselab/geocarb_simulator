"""4-band: non-scattering (no aerosol) vs. scattering (aerosol=smoke) arms, each with and without
observation noise (seed 1) -- 4 series overlaid, one panel per free state row, error (retrieved - truth)
vs. along-slit position, plus a separate chi^2-along-the-slit plot (2026-09-30, user: "Please run the plots
that compare non-scattering with and without noise to aerosols with and without noise for all state variables
versus eta. Also please plot the chi^2 along the slit.").

All four arms share the SAME 33-tile FPA0+FPA1+FPA2+FPA3 tiling and the same free gas/surface rows
(co2_ppm, ch4_ppb, co_ppb, p_surface_hpa, h2o_surface_vmr, t_offset_k, and 4 per-band albedo rows); the
aerosol arms additionally free amplitude_aerosol/height_aerosol, plotted only for those two.

Per-tile chi^2 = sum(weighted resid^2) (dof-normalized, i.e. reduced chi^2 = that / joint["dof"]) -- resid
is the full per-datapoint weighted residual, not resolved by eta, so this is necessarily one point per TILE
(at its centre along-slit position), not a continuous curve like the state-row error panels.

    PYTHONPATH=.:<gert> python scripts/plot_multiband4_scattering_noise_comparison.py
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

DIR = REPO / "results/realistic_prior/multiband"
BASE = "mb_fpa0-1-2-3_r0-*_r1-*_r2-*_r3-*_free-co2-ch4-co-p-h2o-t-albedo"
SUF = "_cover_g1.0_etaslit_prior-realistic"

ARMS = {
    "no-aerosol":          (f"{DIR}/{BASE}{SUF}.pkl",                                  "tab:green"),
    "no-aerosol + noise":  (f"{DIR}/{BASE}{SUF}_noise1.pkl",                           "tab:olive"),
    "aerosol (smoke)":     (f"{DIR}/{BASE}-amplitude-height{SUF}_aero_smoke_lmfast.pkl", "tab:red"),
    "aerosol (smoke) + noise": (f"{DIR}/{BASE}-amplitude-height{SUF}_aero_smoke_noise1_lmfast.pkl", "tab:orange"),
}

ROWS = [("co2_ppm", None, "CO$_2$ [ppm]"), ("ch4_ppb", None, "CH$_4$ [ppb]"), ("co_ppb", None, "CO [ppb]"),
        ("p_surface_hpa", None, "surface pressure [hPa]"), ("h2o_surface_vmr", None, "H$_2$O surface vmr"),
        ("t_offset_k", None, "T offset [K]"), ("albedo_O2_A", "O2_A", "albedo (O$_2$-A)"),
        ("albedo_CO2_weak", "CO2_weak", "albedo (CO$_2$ weak)"), ("albedo_CO2_strong", "CO2_strong", "albedo (CO$_2$ strong)"),
        ("albedo_CH4_CO", "CH4_CO", "albedo (CH$_4$/CO)"),
        ("amplitude_aerosol", None, "aerosol amplitude"), ("height_aerosol", None, "aerosol height [Pa]")]


def truth_of(name, x_km, label=None):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x_km, label))
    if name in als.SURFACE_FIELDS:
        return np.asarray(als.SURFACE_FIELDS[name](x_km))
    return np.asarray(als.STATE_FIELDS[name](x_km))


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        d = pickle.load(open(f, "rb"))
        out[tuple(d["rows_by_fpa"][0])] = d
    return out


arm_data = {name: load(pat) for name, (pat, _) in ARMS.items()}
for name, d in arm_data.items():
    print(f"{name}: {len(d)} tiles")
common = sorted(set.intersection(*(set(d) for d in arm_data.values())))
print(f"{len(common)} tiles common to all 4 arms\n")

# ---- one figure per state row: error (retrieved - truth) vs along-slit km, 4 series overlaid ----------------
for name, label, title in ROWS:
    fig, ax = plt.subplots(figsize=(11, 4))
    any_data = False
    for arm, (pat, color) in ARMS.items():
        rows_out = []
        for k in common:
            d = arm_data[arm][k]
            params = d["joint"]["params"]
            if name not in params:
                continue
            p = params[name]
            x = np.asarray(p["positions"], dtype=float) * als.SLIT_HALF_KM
            v = np.asarray(p["values"], dtype=float)
            t = truth_of(name, x, label)
            keep = slice(1, -1)                      # drop this tile's own boundary bins
            rows_out.append((x[keep], v[keep] - t[keep]))
        if not rows_out:
            continue
        any_data = True
        xs = np.concatenate([r[0] for r in rows_out])
        es = np.concatenate([r[1] for r in rows_out])
        order = np.argsort(xs)
        ax.plot(xs[order], es[order], "-", color=color, lw=0.8, alpha=0.85, label=arm)
    if not any_data:
        plt.close(fig)
        continue
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("along-slit position [km]")
    ax.set_ylabel(f"retrieved - truth: {title}")
    ax.set_title(f"{title}: non-scattering vs. scattering, with/without noise")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    out = REPO / f"plots/mb4_scatter_noise_comparison_{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

# ---- chi^2 along the slit: one point per tile per arm ---------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for arm, (pat, color) in ARMS.items():
    xc, chi2, chi2_red = [], [], []
    for k in common:
        d = arm_data[arm][k]
        j = d["joint"]
        x = np.asarray(j["params"]["co2_ppm"]["positions"], dtype=float) * als.SLIT_HALF_KM
        xc.append(float(np.mean(x)))
        c2 = float(np.sum(np.asarray(j["resid"], dtype=float) ** 2))
        chi2.append(c2)
        chi2_red.append(c2 / float(j["dof"]))
    order = np.argsort(xc)
    xc = np.asarray(xc)[order]
    axes[0].plot(xc, np.asarray(chi2)[order], "o-", color=color, ms=4, lw=1, label=arm)
    axes[1].plot(xc, np.asarray(chi2_red)[order], "o-", color=color, ms=4, lw=1, label=arm)
axes[0].set_ylabel(r"$\chi^2$ (per tile)")
axes[0].set_yscale("log")
axes[1].axhline(1.0, color="k", lw=0.8, alpha=0.5, ls="--")
axes[1].set_ylabel(r"reduced $\chi^2$ ($\chi^2$/dof)")
axes[1].set_yscale("log")
axes[1].set_xlabel("along-slit position [km] (tile centre)")
axes[0].set_title(r"$\chi^2$ along the slit: non-scattering vs. scattering, with/without noise")
axes[0].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
fig.tight_layout()
out = REPO / "plots/mb4_chi2_along_slit.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
