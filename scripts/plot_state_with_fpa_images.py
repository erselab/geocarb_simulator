"""Along-slit truth/prior state profiles drawn vertically beside the four FPA truth images (2026-09-25), one
row of panels sharing ONE vertical axis = along-slit position [km], so a feature in a state variable lines up
with the same slit position in every FPA image. Each FPA image is drawn with its own row->eta->km mapping
(band_tables eta_c), columns oriented with wavelength increasing to the right (FPA1/FPA3 flipped). Skinny state
panels: truth (solid) and realistic prior (dashed); albedo panel overlays each band's albedo. Needs the images
rendered by render_fpa_images.py (results/fpa_images/).
    PYTHONPATH=.:<gert> python scripts/plot_state_with_fpa_images.py
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
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.multiband_geometry import band_tables  # noqa: E402

H = als.SLIT_HALF_KM
x = np.linspace(-H, H, int(round(2 * H / 0.5)) + 1)          # the 0.5 km anchor grid (albedo has 0.5 km texture)
NAMES = {0: "FPA0 O2-A", 1: "FPA1 CO2 weak", 2: "FPA2 CO2 strong", 3: "FPA3 CH4/CO"}
LABELS = {0: "O2_A", 1: "CO2_weak", 2: "CO2_strong", 3: "CH4_CO"}
BAND_COLORS = {0: "#2a78d6", 1: "#1baf7a", 2: "#eb6834", 3: "#8e44ad"}
VARS = [("co2_ppm", "CO$_2$ [ppm]"), ("ch4_ppb", "CH$_4$ [ppb]"), ("co_ppb", "CO [ppb]"),
        ("p_surface_hpa", "p$_{surf}$ [hPa]"), ("h2o_surface_vmr", "H$_2$O [vmr]"), ("t_offset_k", "T offset [K]")]
prior_f = als.PRIOR_FIELD_SETS["realistic"]
prior_alb = als.SURFACE_PRIOR_FIELD_SETS["realistic"]["albedo"]
UNIT = "W m$^{-2}$ $\\mu$m$^{-1}$ sr$^{-1}$"

ratios = [1.0] * len(VARS) + [1.0, 3.6] * 4
fig, axs = plt.subplots(1, len(ratios), figsize=(sum(ratios) * 1.45, 11), sharey=True,
                        gridspec_kw=dict(width_ratios=ratios, wspace=0.08), constrained_layout=False)
ink = "#0b0b0b"
for a, (name, lab) in zip(axs, VARS):
    a.plot(np.asarray(als.STATE_FIELDS[name](x)), x, color="#111", lw=1.4, label="truth")
    a.plot(np.asarray(prior_f[name](x)), x, color="#eb6834", lw=1.2, ls="--", label="prior")
    a.set_title(lab, fontsize=9, color=ink)
    a.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    a.tick_params(axis="x", labelsize=7, rotation=45)
axs[0].legend(frameon=False, fontsize=7, loc="lower left")
axs[0].set_ylabel("along-slit position [km]", color=ink)

state_axes = list(axs[:len(VARS)])
for f in range(4):
    aa, a = axs[len(VARS) + 2 * f], axs[len(VARS) + 2 * f + 1]
    # this band's albedo (truth solid, prior dashed) right beside its own image
    aa.plot(np.asarray(prior_alb(x, LABELS[f])), x, color="#eb6834", lw=0.6, ls="-", alpha=0.8)
    aa.plot(np.asarray(als.SURFACE_FIELDS["albedo"](x, LABELS[f])), x, color="#111", lw=0.6)
    aa.set_title(f"albedo\n{LABELS[f]}", fontsize=8, color=ink)
    aa.tick_params(axis="x", labelsize=7, rotation=45)
    state_axes.append(aa)
    img = np.load(REPO / f"results/fpa_images/fpa{f}_truth.npy")
    lam, _ = xy_to_wavelength_slit(f, np.array([0.0, 1023.0]), np.array([512.0, 512.0]))
    if lam[1] < lam[0]:
        img = img[:, ::-1]
    eta_c = band_tables(f)["eta_c"]
    lo, hi = np.percentile(img, [1, 99])
    im = a.imshow(img, origin="lower", cmap="viridis", vmin=lo, vmax=hi, aspect="auto", interpolation="nearest",
                  extent=[0, 1023, eta_c[0] * H, eta_c[-1] * H])
    a.set_title(f"{NAMES[f]}\n{min(lam):.3f}-{max(lam):.3f} $\\mu$m", fontsize=9, color=ink)
    a.set_xlabel("column, $\\lambda$ increasing $\\rightarrow$" + ("" if lam[1] > lam[0] else " (flipped)"), fontsize=8)
    cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.03, pad=0.07)
    cb.set_label(f"truth radiance [{UNIT}]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
axs[len(VARS)].plot([], [], color="#111", lw=1, label="truth")
axs[len(VARS)].plot([], [], color="#eb6834", lw=1, label="prior")
axs[len(VARS)].legend(frameon=False, fontsize=6, loc="lower right")
for a in axs:
    a.set_ylim(-H, H)
    a.set_yticks(np.arange(-1400, 1401, 200))
    a.tick_params(axis="y", labelsize=8)
    if a in state_axes:
        a.grid(True, axis="y", color="#e6e5e0", lw=0.8)
    else:
        a.grid(True, axis="y", color="white", lw=0.5, ls=":", alpha=0.35)
fig.suptitle("Along-slit state (truth vs realistic prior) beside the FPA truth images: one shared vertical axis, so features line up", fontsize=12, y=0.99)
out = REPO / "plots/state_profiles_with_fpa_images.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print("saved", out.name)
