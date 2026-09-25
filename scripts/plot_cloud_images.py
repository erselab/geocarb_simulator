"""Radiance images of the tapering-edge cloud in all four bands (2026-09-25); reads results/cloud_images/ (render_cloud_images.py).

  plots/cloud_images_mid_moderate.png : for the mid-altitude, tau0 = 2 cloud, every band as [clear | cloudy | cloudy/clear]
  plots/cloud_images_ratio_fpa<n>.png : cloudy/clear ratio images for all 9 scenarios (rows: altitude, columns: tau0)
  plots/cloud_images_profiles.png     : row-summed radiance ratio along the slit, the four bands per scenario
Vertical axis = along-slit position [km] (each row's centre column); columns oriented with wavelength increasing to the
right (FPA1/FPA3 flipped). Radiance in W m^-2 um^-1 sr^-1.
    PYTHONPATH=.:<gert> python scripts/plot_cloud_images.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert.cloud_scene import ALTITUDES_HPA, OPTICAL_DEPTHS, scenarios  # noqa: E402

D = REPO / "results/cloud_images"
NAMES = {0: "FPA0 O2-A", 1: "FPA1 CO2 weak", 2: "FPA2 CO2 strong", 3: "FPA3 CH4/CO"}
FLIP = {0: False, 1: True, 2: False, 3: True}
UNIT = "W m$^{-2}$ $\\mu$m$^{-1}$ sr$^{-1}$"
SC = scenarios()


def load(name, f):
    p = D / f"{name}_fpa{f}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    A = z["A"][:, ::-1] if FLIP[f] else z["A"]
    return A, z["x_km"]


def show(ax, img, x_km, **kw):
    return ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest", extent=[0, 1023, x_km[0], x_km[-1]], **kw)


# ---- 1. mid_moderate, four bands, clear | cloudy | ratio ----
fig, axs = plt.subplots(4, 3, figsize=(13, 17), constrained_layout=True)
for f in range(4):
    c, k = load("clear", f), load("mid_moderate", f)
    if c is None or k is None:
        continue
    (Ac, x), (Ak, _) = c, k
    vmax = np.percentile(Ak, 99)
    im0 = show(axs[f, 0], Ac, x, cmap="viridis", vmin=0, vmax=vmax)
    show(axs[f, 1], Ak, x, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(im0, ax=axs[f, :2].tolist(), shrink=0.8, label=f"radiance [{UNIT}]")
    R = Ak / np.maximum(Ac, 1e-9 * vmax)
    im2 = show(axs[f, 2], R, x, cmap="magma", norm=LogNorm(vmin=1.0, vmax=max(2.0, np.percentile(R, 99))))
    fig.colorbar(im2, ax=axs[f, 2], shrink=0.8, label="cloudy / clear")
    for j, t in enumerate(("clear", "cloudy (mid, tau0 = 2)", "ratio")):
        axs[f, j].set_title(f"{NAMES[f]}: {t}", loc="left", fontsize=10)
        axs[f, j].set_xlabel("column, wavelength increasing" + (" (flipped)" if FLIP[f] else ""), fontsize=8)
    axs[f, 0].set_ylabel("along-slit position [km]")
fig.suptitle("A tapering-edge liquid cloud (600 hPa, optical depth 2 at the core, centre +700 km) in each band's radiance", fontsize=12)
fig.savefig(REPO / "plots/cloud_images_mid_moderate.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---- 2. ratio images, all scenarios, per band ----
for f in range(4):
    c = load("clear", f)
    if c is None:
        continue
    Ac, x = c
    fig, axs = plt.subplots(3, 3, figsize=(13, 12), sharex=True, sharey=True, constrained_layout=True)
    vmax = 2.0
    for i, (an, p) in enumerate(ALTITUDES_HPA.items()):
        for j, (tn, tau) in enumerate(OPTICAL_DEPTHS.items()):
            k = load(f"{an}_{tn}", f)
            a = axs[i, j]
            a.set_title(f"{an} ({p:g} hPa), tau0 = {tau:g}", loc="left", fontsize=9)
            if k is None:
                a.text(0.5, 0.5, "not rendered", transform=a.transAxes, ha="center")
                continue
            R = k[0] / np.maximum(Ac, 1e-9 * np.percentile(k[0], 99))
            vmax = max(vmax, np.percentile(R, 99.5))
            im = show(a, R, x, cmap="magma", norm=LogNorm(vmin=1.0, vmax=None))
            ims = im
    for a in axs[:, 0]:
        a.set_ylabel("along-slit position [km]")
    for a in axs[2, :]:
        a.set_xlabel("column, wavelength increasing" + (" (flipped)" if FLIP[f] else ""), fontsize=8)
    # one common log scale per figure
    for a in axs.ravel():
        for im in a.get_images():
            im.set_norm(LogNorm(vmin=1.0, vmax=vmax))
    fig.colorbar(ims, ax=axs.ravel().tolist(), shrink=0.7, label="cloudy / clear radiance")
    fig.suptitle(f"{NAMES[f]}: cloud signature (cloudy / clear) for 3 altitudes x 3 optical depths", fontsize=12)
    fig.savefig(REPO / f"plots/cloud_images_ratio_fpa{f}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

# ---- 3. along-slit profile of the row-summed ratio, four bands per scenario ----
fig, axs = plt.subplots(3, 3, figsize=(14, 9.5), sharex=True, constrained_layout=True)
COL = {0: "#2a78d6", 1: "#1baf7a", 2: "#eb6834", 3: "#8e44ad"}
for i, (an, p) in enumerate(ALTITUDES_HPA.items()):
    for j, (tn, tau) in enumerate(OPTICAL_DEPTHS.items()):
        a = axs[i, j]
        for f in range(4):
            c, k = load("clear", f), load(f"{an}_{tn}", f)
            if c is None or k is None:
                continue
            a.plot(c[1], k[0].sum(axis=1) / c[0].sum(axis=1), color=COL[f], lw=1.5, label=NAMES[f])
        a.axhline(1, color="#c3c2b7", lw=0.8)
        a.set_yscale("log")
        a.set_title(f"{an} ({p:g} hPa), tau0 = {tau:g}", loc="left", fontsize=10)
        if j == 0:
            a.set_ylabel("row-summed radiance, cloudy / clear")
        if i == 2:
            a.set_xlabel("along-slit position [km]")
axs[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("How much the cloud brightens each band along the slit (radiance summed over all columns of each row)", fontsize=12)
fig.savefig(REPO / "plots/cloud_images_profiles.png", dpi=120, bbox_inches="tight")
print("saved")
