"""Is the amplitude-height trade-off a curved valley?  (2026-09-26)

The 1D scan (scan_aerosol_height.py) shows the radiance is only mildly nonlinear in height alone (linearization error 3-5% for a 30 hPa
step), yet the 4-band solve's Gauss-Newton step fails at half length. Amplitude and height trade off (more AOD lower down looks like less
AOD higher up), so this maps the noise-weighted misfit chi2(amplitude, height) at one anchor (x = -340 km, all four bands, smoke Mie,
XRTM) and draws it in three coordinate systems -- (amp, h), (ln amp, h), (ln amp, ln h) -- so the valley's curvature and
the best parameterization can be judged directly. Also prints the valley path (best h per amplitude) and how straight it is in each.
Saves plots/aerosol_amp_height_valley.png.
"""
import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ["GEOCARB_AEROSOL_TYPE"] = "smoke"
import gd_joint_block_retrieve as gjr  # noqa: E402
import gd_multiband_window as gmw  # noqa: E402
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402
from geocarb_gert.spectrum import simulate_spectrum  # noqa: E402

X = np.array([-340.0])
absco, solar, geo, atm_center = gmw.load_inputs()
atm = {n: float(als.STATE_FIELDS[n](X)[0]) for n in ("co2_ppm", "ch4_ppb", "co_ppb", "h2o_surface_vmr", "p_surface_hpa", "t_offset_k")}
amp_t = float(als.SURFACE_FIELDS["amplitude_aerosol"](X, "x")[0])
thick = float(als.SURFACE_FIELDS["thickness_aerosol"](X, "x")[0])
h_t = float(als.SURFACE_FIELDS["height_aerosol"](X, "x")[0])
AMP = amp_t * np.exp(np.linspace(np.log(0.4), np.log(3.0), 15))
HH = np.linspace(44000.0, 72000.0, 15)
BANDS = {}
for f in range(4):
    label = gjr.GEOCARB_BANDS[f][0]
    _, wide, _ = gjr.band_basics(f, atm_center, absco, geo, solar)
    BANDS[f] = (wide, {"albedo": float(als.SURFACE_FIELDS["albedo"](X, label)[0]), "thickness_aerosol": thick})


def I_of(job):
    f, amp, h = job
    wide, sfc = BANDS[f]
    return simulate_spectrum(atm, {**sfc, "amplitude_aerosol": amp, "height_aerosol": h}, absco, wide, geo, solar,
                             aerosol_type="smoke", solver="xrtm").I_hires


with mp.get_context("fork").Pool(min(22, os.cpu_count() or 8)) as pool:
    jobs = [(f, float(a), float(h)) for f in range(4) for a in AMP for h in HH]
    res = dict(zip(jobs, pool.map(I_of, jobs, chunksize=4)))
    ref = {f: I_of((f, amp_t, h_t)) for f in range(4)}
sig = {f: geocarb_noise_model(f).sigma([ref[f]], [None]).ravel() for f in range(4)}
chi2 = np.array([[sum(float(np.sum(((res[(f, float(a), float(h))] - ref[f]) / sig[f]) ** 2)) for f in range(4)) for h in HH] for a in AMP])
print(f"anchor x=-340: truth amp {amp_t:.2e}, height {h_t:.0f} Pa; chi2 grid {chi2.shape}, min {chi2.min():.3g}", flush=True)
# valley path: best h for each amplitude (parabolic refinement in h)
path = []
for i, a in enumerate(AMP):
    j = int(np.argmin(chi2[i]))
    if 0 < j < len(HH) - 1:
        c = np.polyfit(HH[j - 1:j + 2], chi2[i, j - 1:j + 2], 2)
        path.append(-c[1] / (2 * c[0]))
    else:
        path.append(HH[j])
path = np.array(path)
for name, xs, ys in (("h vs amp", AMP, path), ("h vs ln amp", np.log(AMP), path), ("ln h vs ln amp", np.log(AMP), np.log(path))):
    c = np.polyfit(xs, ys, 1)
    r = ys - np.polyval(c, xs)
    print(f"valley path {name:16s}: linear-fit rms residual {100 * np.sqrt(np.mean(r ** 2)) / np.ptp(ys):.1f}% of its range; slope {c[0]:.4g}", flush=True)
print("valley path (amp/amp_true -> best height [Pa]): " + "  ".join(f"{a / amp_t:.2f}:{p:.0f}" for a, p in zip(AMP[::2], path[::2])), flush=True)
fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
L = np.log10(np.maximum(chi2, chi2.min() + 1e-6))
for a, (xl, X_, Y_) in zip(axs, (("amplitude", AMP, HH), ("ln amplitude", np.log(AMP), HH), ("ln amplitude", np.log(AMP), np.log(HH)))):
    cs = a.contourf(Y_ if False else X_, Y_, L.T, levels=20, cmap="viridis")
    a.plot(X_, (path if a is not axs[2] else np.log(path)), "w-", lw=1.5, label="best height per amplitude")
    a.plot(np.log(amp_t) if a is not axs[0] else amp_t, h_t if a is not axs[2] else np.log(h_t), "r*", ms=13, label="truth")
    a.set_xlabel(xl)
axs[0].set_ylabel("height [Pa]"); axs[1].set_ylabel("height [Pa]"); axs[2].set_ylabel("ln height")
axs[0].legend(frameon=False, loc="upper right", fontsize=8)
fig.colorbar(cs, ax=axs, shrink=0.8, label="log10 chi2 (four bands, one anchor)")
fig.suptitle("Amplitude-height misfit valley in three parameterizations (x = -340 km; truth star)", fontsize=12)
fig.savefig(REPO / "plots/aerosol_amp_height_valley.png", dpi=120, bbox_inches="tight")
print("saved plots/aerosol_amp_height_valley.png")
