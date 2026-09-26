"""How nonlinear is the radiance in aerosol layer height, per band?  (2026-09-26)

One anchor of tile 12 (x = -340 km, surface ~798 hPa, scene-truth atmosphere/albedo/aerosol amplitude and thickness, smoke Mie): the
XRTM radiance I(h) for the layer-centre pressure h scanned over 40000..76000 Pa in every band, plus the analytic dI/dh at two
reference heights. Reports
  * the linearization error |I(h0+d) - I(h0) - K d| / |K d| against the height change d [Pa] -- the size of step over which the
    Gauss-Newton model is trustworthy, and how it compares with the layer thickness (sigma = 84 hPa) and the layer spacing;
  * the same in log-pressure (a step d/h0 in ln h is the same physical change here, so this only asks whether ln h linearizes better);
  * the noise-weighted misfit chi2(h) = sum ((I(h) - I(h_true))/sigma_I)^2 -- the shape of the objective along height at one anchor.
    python scripts/scan_aerosol_height.py
Saves plots/aerosol_height_scan.png.
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
from geocarb_gert.spectrum import simulate_spectrum, spectrum_and_jacobian  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402

X = np.array([-340.0])
absco, solar, geo, atm_center = gmw.load_inputs()
atm = {n: float(als.STATE_FIELDS[n](X)[0]) for n in ("co2_ppm", "ch4_ppb", "co_ppb", "h2o_surface_vmr", "p_surface_hpa", "t_offset_k")}
amp = float(als.SURFACE_FIELDS["amplitude_aerosol"](X, "x")[0])
thick = float(als.SURFACE_FIELDS["thickness_aerosol"](X, "x")[0])
h_true = float(als.SURFACE_FIELDS["height_aerosol"](X, "x")[0])
H = np.linspace(40000.0, 76000.0, 37)
BANDS = {}
for f in range(4):
    label = gjr.GEOCARB_BANDS[f][0]
    _, wide, _ = gjr.band_basics(f, atm_center, absco, geo, solar)
    BANDS[f] = (wide, {"albedo": float(als.SURFACE_FIELDS["albedo"](X, label)[0]), "amplitude_aerosol": amp, "thickness_aerosol": thick})
print(f"anchor x=-340: p_sfc {atm['p_surface_hpa']:.1f} hPa, amp {amp:.2e}, thickness {thick:.0f} Pa, true height {h_true:.0f} Pa", flush=True)


def I_of(job):
    f, h = job
    wide, sfc = BANDS[f]
    return simulate_spectrum(atm, {**sfc, "height_aerosol": h}, absco, wide, geo, solar, aerosol_type="smoke", solver="xrtm").I_hires


def K_of(job):
    f, h = job
    wide, sfc = BANDS[f]
    return spectrum_and_jacobian(atm, ["height_aerosol"], absco, wide, geo, solar, surface={**sfc, "height_aerosol": h},
                                 aerosol_type="smoke", solver="xrtm")[1]["height_aerosol"]


refs = [h_true, 61600.0]
with mp.get_context("fork").Pool(min(20, os.cpu_count() or 8)) as pool:
    jobs = [(f, float(h)) for f in range(4) for h in H]
    Is = dict(zip(jobs, pool.map(I_of, jobs)))
    kj = [(f, r) for f in range(4) for r in refs]
    Ks = dict(zip(kj, pool.map(K_of, kj)))
    It = {f: I_of((f, h_true)) for f in range(4)}

fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for f in range(4):
    sig = geocarb_noise_model(f).sigma([It[f]], [None]).ravel()
    chi2 = [float(np.sum(((Is[(f, float(h))] - It[f]) / sig) ** 2)) for h in H]
    axs[0, 0].plot(H / 100, np.array(chi2) / max(chi2[0], 1e-30) if False else np.log10(np.maximum(chi2, 1e-3)), label=f"FPA{f}")
    print(f"\nFPA{f}: chi2 at h_true {0:.0f}; along h: " + " ".join(f"{h / 100:.0f}:{c:.3g}" for h, c in zip(H[::4], chi2[::4])), flush=True)
    for ref in refs:
        K = Ks[(f, ref)]
        i0 = int(np.argmin(np.abs(H - ref)))
        h0 = H[i0]
        I0 = Is[(f, float(h0))]
        row = []
        for d in (-6000, -3000, -1500, 1500, 3000, 6000):
            j = int(np.argmin(np.abs(H - (h0 + d))))
            dd = H[j] - h0
            if dd == 0:
                continue
            lin = np.linalg.norm(Is[(f, float(H[j]))] - I0 - K * dd) / (np.linalg.norm(K * dd) + 1e-300)
            row.append(f"{dd:+.0f}Pa:{100 * lin:.1f}%")
            if ref == refs[0] and f == 0:
                pass
        print(f"   linearization error at h0={h0:.0f} Pa (analytic K): " + "  ".join(row), flush=True)
axs[0, 0].set_xlabel("layer centre pressure [hPa]"); axs[0, 0].set_ylabel("log10 chi2 vs the truth spectrum")
axs[0, 0].axvline(h_true / 100, color="k", ls=":"); axs[0, 0].legend(frameon=False)
axs[0, 0].set_title("noise-weighted misfit along aerosol height (one anchor)", loc="left")
for f in range(4):
    K = Ks[(f, refs[0])]
    i0 = int(np.argmin(np.abs(H - refs[0])))
    lin_err = [np.linalg.norm(Is[(f, float(h))] - Is[(f, float(H[i0]))] - K * (h - H[i0])) / (np.linalg.norm(Is[(f, float(H[i0]))]) + 1e-300) for h in H]
    axs[0, 1].plot(H / 100, 100 * np.array(lin_err), label=f"FPA{f}")
    axs[1, 0].plot(H / 100, (np.array([Is[(f, float(h))].mean() for h in H]) / Is[(f, float(H[i0]))].mean() - 1) * 100, label=f"FPA{f}")
axs[0, 1].set_title("linearization error about the true height (% of I)", loc="left"); axs[0, 1].set_xlabel("layer centre pressure [hPa]")
axs[1, 0].set_title("band-mean radiance change vs the true height [%]", loc="left"); axs[1, 0].set_xlabel("layer centre pressure [hPa]")
axs[1, 1].axis("off")
axs[1, 1].text(0.02, 0.9, f"anchor -340 km: p_sfc {atm['p_surface_hpa']:.0f} hPa\namp {amp:.2e}, thickness {thick / 100:.0f} hPa\ntrue height {h_true / 100:.0f} hPa\nsmoke Mie, XRTM two-stream", va="top")
fig.savefig(REPO / "plots/aerosol_height_scan.png", dpi=120, bbox_inches="tight")
print("saved plots/aerosol_height_scan.png")
