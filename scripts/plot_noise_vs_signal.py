"""Noise vs signal for the 'truth + noise' FPA images (2026-09-25).

Row 1: realized noise (noisy - truth image, results/fpa_images/) vs the pixel's own radiance, per band, with the
analytic sigma(I) = sqrt(N0^2 + N1 |I|) (the band's LinearShotNoise) and +/-1 sigma curves.
Row 2: the same sigma (analytic, per pixel) vs the proxy signal = albedo(x) * cos(SZA), x = each pixel's slit
position -- to show how well that proxy predicts the noise. Because sigma is a deterministic function of the
pixel's radiance only, the radiance itself is the natural 'signal'; albedo*cos(SZA) omits the solar
irradiance (very different per band) and every absorption line, so pixels at one albedo*cos(SZA) span a wide
range of radiance/noise.
    PYTHONPATH=.:<gert> python scripts/plot_noise_vs_signal.py
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
from geocarb_gert.gd_polynomials import N_PX, eta_of_s, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402

NAMES = {0: "FPA0 O2-A", 1: "FPA1 CO2 weak", 2: "FPA2 CO2 strong", 3: "FPA3 CH4/CO"}
LABELS = {0: "O2_A", 1: "CO2_weak", 2: "CO2_strong", 3: "CH4_CO"}
COS_SZA = float(np.cos(np.radians(31.894852612221843)))      # the single scene geometry (sample_geometries seed 0)
UNIT = "W m$^{-2}$ $\\mu$m$^{-1}$ sr$^{-1}$"
d = REPO / "results/fpa_images"
rng = np.random.default_rng(0)
cols, rows = np.meshgrid(np.arange(N_PX, dtype=float), np.arange(N_PX, dtype=float))

fig, axs = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
for f in range(4):
    A = np.load(d / f"fpa{f}_truth.npy")
    noise = np.load(d / f"fpa{f}_noise.npy") - A
    nm = geocarb_noise_model(f)
    sig = np.sqrt(nm.N0 ** 2 + nm.N1 * np.abs(A))
    z = noise / sig
    print(f"FPA{f}: N0={nm.N0:.4g} N1={nm.N1:.4g} I_max={nm.I_max:.4g}; realized noise/sigma mean {z.mean():+.3f} sd {z.std():.3f}; "
          f"sigma {sig.min():.3g}..{sig.max():.3g}; SNR {np.percentile(A/sig,5):.0f}..{np.percentile(A/sig,95):.0f} (5-95%)", flush=True)
    idx = rng.choice(A.size, 60000, replace=False)
    Ai, ni = A.ravel()[idx], noise.ravel()[idx]
    ax = axs[0, f]
    ax.scatter(Ai, ni, s=1.5, alpha=0.25, color="#8e44ad", edgecolor="none", rasterized=True, label="realized noise (60k pixels)")
    grid = np.linspace(0, A.max() * 1.02, 300)
    sg = np.sqrt(nm.N0 ** 2 + nm.N1 * grid)
    ax.plot(grid, sg, color="#111", lw=1.4, label="$\\pm\\sigma(I)=\\sqrt{N_0^2+N_1 I}$")
    ax.plot(grid, -sg, color="#111", lw=1.4)
    ax.axvline(nm.I_max, color="#c2185b", ls="--", lw=1, label=f"I_max = {nm.I_max:.3g}")
    ax.set_title(f"{NAMES[f]}: noise vs radiance", loc="left", fontsize=10)
    ax.set_xlabel(f"pixel radiance [{UNIT}]")
    ax.set_ylabel(f"noise [{UNIT}]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(True, color="#e6e5e0", lw=0.8)
    # proxy signal albedo*cos(SZA), per pixel slit position
    _, s = xy_to_wavelength_slit(f, cols.ravel(), rows.ravel())
    x_km = eta_of_s(f, s) * als.SLIT_HALF_KM
    proxy = np.asarray(als.SURFACE_FIELDS["albedo"](x_km, LABELS[f])) * COS_SZA
    ax = axs[1, f]
    ax.scatter(proxy[idx], sig.ravel()[idx], s=1.5, alpha=0.25, color="#1baf7a", edgecolor="none", rasterized=True)
    ax.set_title(f"{NAMES[f]}: $\\sigma$ vs albedo$\\cdot\\cos$(SZA)", loc="left", fontsize=10)
    ax.set_xlabel("albedo(x) $\\cdot$ cos(SZA)  [-]")
    ax.set_ylabel(f"$\\sigma$ [{UNIT}]")
    ax.grid(True, color="#e6e5e0", lw=0.8)
    r_rad = np.corrcoef(A.ravel()[idx], sig.ravel()[idx])[0, 1]
    r_proxy = np.corrcoef(proxy[idx], sig.ravel()[idx])[0, 1]
    ax.text(0.03, 0.95, f"corr($\\sigma$, radiance) = {r_rad:.3f}\ncorr($\\sigma$, albedo$\\cdot$cosSZA) = {r_proxy:.3f}",
            transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))
    print(f"   corr(sigma, radiance) {r_rad:.3f}  corr(sigma, albedo*cosSZA) {r_proxy:.3f}", flush=True)
fig.suptitle("Noise vs signal for the truth + noise images (signal-dependent shot noise, seed 1)", fontsize=12)
out = REPO / "plots/noise_vs_signal_fpa0-3.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out.name)
