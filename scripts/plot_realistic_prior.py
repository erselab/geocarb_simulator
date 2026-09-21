"""Plots of the "realistic" prior against the truth and the old "structural" prior (plots/realistic_prior_{fields,err}.png).

    PYTHONPATH=.:<gert> python scripts/plot_realistic_prior.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402

x = np.linspace(-1400, 1400, 5601)
C = {"truth": "#2a78d6", "real": "#eb6834", "struct": "#1baf7a"}
ink, sec, grid = "#0b0b0b", "#52514e", "#e6e5e0"
F = [("co2_ppm", "CO$_2$ [ppm]", "abs"), ("ch4_ppb", "CH$_4$ [ppb]", "abs"), ("co_ppb", "CO [ppb]", "abs"),
     ("t_offset_k", "T offset [K]", "abs"), ("p_surface_hpa", "surface pressure [hPa]", "abs"),
     ("h2o_surface_vmr", "H$_2$O surface vmr", "rel"), ("amplitude_aerosol", "aerosol amplitude", "rel"),
     ("height_aerosol", "aerosol height [Pa]", "abs"), ("thickness_aerosol", "aerosol thickness [Pa]", "rel"),
     ("albedo:O2_A", "albedo (O$_2$-A)", "rel"), ("albedo:CO2_strong", "albedo (CO$_2$ strong)", "rel")]


def get(name, key):
    base = name.split(":")[0]
    label = name.split(":")[1] if ":" in name else None
    if base in als.SURFACE_FIELDS:
        f = als.SURFACE_FIELDS[base] if key == "truth" else als.SURFACE_PRIOR_FIELD_SETS[key][base]
        return np.asarray(f(x, label))
    f = als.STATE_FIELDS[base] if key == "truth" else als.PRIOR_FIELD_SETS[key][base]
    return np.asarray(f(x))


def style(a):
    a.grid(True, color=grid, lw=0.7)
    for s in (a.spines["top"], a.spines["right"]):
        s.set_visible(False)
    a.tick_params(colors=sec, labelsize=8)
    for s in a.spines.values():
        s.set_color("#c3c2b7")


def panels(kind):
    fig, axs = plt.subplots(4, 3, figsize=(15, 13))
    axs = axs.ravel()
    for a, (n, lab, ek) in zip(axs, F):
        style(a)
        t, r, s = get(n, "truth"), get(n, "realistic"), get(n, "structural")
        if kind == "fields":
            a.plot(x, t, color=C["truth"], lw=1.4, label="truth")
            a.plot(x, s, color=C["struct"], lw=1.3, ls="--", label='old "structural" prior')
            a.plot(x, r, color=C["real"], lw=1.6, label="realistic prior")
            a.set_ylabel(lab, fontsize=9, color=ink)
            if n == "height_aerosol":                      # terrain-following: the layer sits above the surface pressure
                a.plot(x, als.p_surface_hpa(x) * 100.0, color="#8a8980", lw=1.0, ls=":", label="surface pressure (truth)")
        else:
            if ek == "abs":
                er, es = r - t, s - t
                unit = lab.split("[")[-1].rstrip("]") if "[" in lab else ""
                yl = f"prior - truth  [{unit}]" if unit else "prior - truth"
            else:
                er, es, yl = 100 * (r / t - 1), 100 * (s / t - 1), "prior / truth - 1  [%]"
            a.axhline(0, color="#8a8980", lw=1.0)
            a.plot(x, es, color=C["struct"], lw=1.3, ls="--", label='old "structural" prior')
            a.plot(x, er, color=C["real"], lw=1.6, label="realistic prior")
            a.set_ylabel(yl, fontsize=9, color=ink)
        a.set_title(lab, loc="left", fontsize=10, color=ink)
        a.set_xlabel("along-slit position [km]", fontsize=8, color=sec)
    ax = axs[-1]
    ax.axis("off")
    h, l = axs[0].get_legend_handles_labels()
    if kind == "fields":
        h2, l2 = axs[7].get_legend_handles_labels()
        h, l = h + [h2[-1]], l + [l2[-1]]
    ax.legend(h, l, loc="center", frameon=False, fontsize=11)
    ttl = {"fields": "Realistic prior vs truth vs old structural prior",
           "err": "Prior error (prior - truth): the realistic prior never crosses zero; the old prior sits on zero"}[kind]
    fig.suptitle(ttl, x=0.01, ha="left", fontsize=13, color=ink)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPO / f"plots/realistic_prior_{kind}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)




def aerosol_scene():
    """plots/aerosol_scene.png: the aerosol scene along the slit, stacked on one x axis with the mountain shaded and the
    CO2 features marked, so the AOD bumps can be read against the terrain (2026-09-21)."""
    fig, axs = plt.subplots(4, 1, figsize=(11.5, 12), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1, 1, 0.9]))
    ps = als.p_surface_hpa(x)
    mtn = ps < 900.0
    tau_t, tau_s = np.asarray(als.tau_aerosol(x)), np.asarray(als.tau_aerosol_prior(x))
    amp_r = np.asarray(als.amplitude_aerosol_prior_realistic(x))
    th_r = np.asarray(als.thickness_aerosol_prior_realistic(x))
    tau_r = amp_r * th_r * np.sqrt(2 * np.pi)                                  # tau = amplitude * sigma * sqrt(2 pi)
    series = [
        ("aerosol optical depth (O$_2$-A reference)", None, (tau_t, tau_r, tau_s)),
        ("aerosol amplitude [1/Pa]  (= AOD / (thickness $\\sqrt{2\\pi}$))", None,
         (np.asarray(als.amplitude_aerosol(x)), amp_r, np.asarray(als.amplitude_aerosol_prior(x)))),
        ("layer centre pressure [hPa]", "height",
         (np.asarray(als.height_aerosol(x)) / 100, np.asarray(als.height_aerosol_prior_realistic(x)) / 100,
          np.asarray(als.height_aerosol_prior(x)) / 100)),
        ("layer thickness (Gaussian sigma) [hPa]", None,
         (np.asarray(als.thickness_aerosol(x)) / 100, th_r / 100, np.asarray(als.thickness_aerosol_prior(x)) / 100)),
    ]
    for a, (lab, special, (t, r, s)) in zip(axs, series):
        style(a)
        a.axvspan(x[mtn].min(), x[mtn].max(), color="#efeee9", zorder=0)             # the mountain: surface pressure < 900 hPa
        for x0, w in ((-500.0, 60.0), (-1100.0, 10.0), (1050.0, 9.0)):               # the CO2 plume and hot spots
            a.axvline(x0, color="#b9b8ae", lw=0.9, ls=(0, (3, 3)), zorder=1)
        a.plot(x, s, color=C["struct"], lw=1.3, ls="--", label='old "structural" prior')
        a.plot(x, r, color=C["real"], lw=1.6, label="realistic prior")
        a.plot(x, t, color=C["truth"], lw=1.8, label="truth")
        if special == "height":
            a.plot(x, ps, color="#8a8980", lw=1.1, ls=":", label="surface pressure (truth)")
            a.invert_yaxis()
        a.set_ylabel(lab, fontsize=9, color=ink)
    axs[0].annotate("urban AOD bump\n(CO$_2$ plume, -500 km)", (-500, tau_t[np.argmin(np.abs(x + 500))]), xytext=(-980, 0.30),
                    fontsize=9, color=sec, arrowprops=dict(arrowstyle="-", color="#8a8980", lw=0.8))
    axs[0].annotate("haze (+200 km)", (200, tau_t.max()), xytext=(330, 0.335), fontsize=9, color=sec)
    axs[0].text(x[mtn].mean(), axs[0].get_ylim()[1] * 0.93, "mountain\n(p$_s$ < 900 hPa)", ha="center", va="top", fontsize=9, color=sec)
    for x0, lab in ((-1100, "CO$_2$ hot spot"), (1050, "CO$_2$ hot spot")):
        axs[0].text(x0, axs[0].get_ylim()[1] * 0.97, lab, rotation=90, ha="right", va="top", fontsize=8, color="#8a8980")
    axs[0].legend(frameon=False, loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 0.62))
    axs[2].legend(frameon=False, loc="lower right", fontsize=8, ncol=2)
    axs[-1].set_xlabel("along-slit position [km]", fontsize=9, color=ink)
    fig.suptitle("Aerosol scene along the slit: the AOD bump at -500 km is the urban signal, on the mountain's western flank; "
                 "the layer height follows the terrain", x=0.01, ha="left", fontsize=11, color=ink)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPO / "plots/aerosol_scene.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    panels("fields")
    panels("err")
    aerosol_scene()
