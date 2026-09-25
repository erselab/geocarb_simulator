"""Full-detector (1024x1024) radiance images per FPA (2026-09-25): realistic-prior scene, truth,
truth + shot noise (band's geocarb_noise_model, seed 1) -- rendered by render_fpa_images.py into
results/fpa_images/. One figure per FPA (plots/fpa_images_fpa<n>.png), shared color scale set by the
truth image's 1st-99th percentile so the three panels are directly comparable; plus a
prior-minus-truth panel. Image index is [detector row, detector column]; columns are flipped for FPA1/FPA3 (reversed dispersion) so
wavelength increases left->right in every FPA.
Radiance units: W m^-2 um^-1 sr^-1 (gert convention, geocarb_gert/radiometry.py).
    PYTHONPATH=.:<gert> python scripts/plot_fpa_images.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402


def lam_range(f):
    lam, _ = xy_to_wavelength_slit(f, np.array([0.0, 1023.0]), np.array([512.0, 512.0]))
    return float(min(lam)), float(max(lam)), bool(lam[1] > lam[0])

NAMES = {0: "FPA0 O2-A", 1: "FPA1 CO2 weak", 2: "FPA2 CO2 strong", 3: "FPA3 CH4/CO"}
d = REPO / "results/fpa_images"
for f in range(4):
    try:
        img = {k: np.load(d / f"fpa{f}_{k}.npy") for k in ("prior", "truth", "noise")}
    except FileNotFoundError as e:
        print(f"FPA{f}: missing {e.filename} -- skipped")
        continue
    l0, l1, asc = lam_range(f)
    if not asc:      # FPA1/FPA3 dispersion runs the other way: flip columns so wavelength increases left->right in every FPA
        img = {k: v[:, ::-1] for k, v in img.items()}
    lo, hi = np.percentile(img["truth"], [1, 99])
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.8), constrained_layout=True)
    for a, (k, t) in zip(axs[:3], (("prior", "realistic-prior scene"), ("truth", "truth"), ("noise", "truth + noise"))):
        im = a.imshow(img[k], origin="lower", cmap="viridis", vmin=lo, vmax=hi, aspect="equal", interpolation="nearest")
        a.set_title(t, loc="left", fontsize=10)
    fig.colorbar(im, ax=list(axs[:3]), shrink=0.9, label="radiance [W m$^{-2}$ $\\mu$m$^{-1}$ sr$^{-1}$]")
    diff = img["prior"] - img["truth"]
    v = np.percentile(np.abs(diff), 99) or 1.0
    a = axs[3]
    im2 = a.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal", interpolation="nearest")
    a.set_title("prior scene - truth", loc="left", fontsize=10)
    fig.colorbar(im2, ax=a, shrink=0.9, label="radiance difference [W m$^{-2}$ $\\mu$m$^{-1}$ sr$^{-1}$]")
    for a in axs:
        a.set_xlabel("detector column, wavelength increasing ->" + ("" if asc else " (flipped)"))
        a.set_ylabel("detector row")
    fig.suptitle(f"{NAMES[f]} ({l0:.3f}-{l1:.3f} um): full-detector radiance, realistic prior vs truth vs truth + noise", fontsize=12)
    out = REPO / f"plots/fpa_images_fpa{f}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.name)
