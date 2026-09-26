"""What a non-scattering O2-A-only retrieval sees of a tapering-edge cloud (2026-09-25).

Truth: XRTM with the cloud (geocarb_gert/cloud_scene.py); retrieval: FPA0 only, single_scatter, no cloud/aerosol, free
surface pressure + O2-A albedo, realistic prior, tiles 42-54 (476-943 km). Compared with the clear-sky control (same XRTM
truth, no cloud) so the multiple-scattering vs single_scatter offset drops out. Per scenario (3 altitudes x 3 optical depths):
  * retrieved surface pressure minus the prior ("delta Ps", what an A-band cloud flag looks at), scenario vs control;
  * retrieved minus control (the pure cloud effect);
  * the fit residual per tile relative to the control.
Prints the along-slit half-width over which |cloud effect| exceeds 1, 5, 10 hPa.
    PYTHONPATH=.:<gert> python scripts/plot_cloud_o2a.py
Outputs: plots/cloud_o2a_pressure.png, plots/cloud_o2a_residual.png
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
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.cloud_scene import ALTITUDES_HPA, OPTICAL_DEPTHS, Cloud  # noqa: E402

H = als.SLIT_HALF_KM
TILES = range(42, 55)


def load(tau, p):
    fs = sorted(glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0_r0-*_free-p-albedo_cover_g1.0_etaslit_prior-realistic_cloud-tau{tau:g}-p{p:g}.pkl")))
    x, pr, pe, res, xt, rr = [], [], [], [], [], []
    for f in fs:
        d = pickle.load(open(f, "rb"))
        q = d["joint"]["params"]["p_surface_hpa"]
        xs = np.asarray(q["positions"]) * H
        x += list(xs); pr += list(q["values"]); pe += list(q["prior"])
        xt.append(0.5 * (xs.min() + xs.max())); rr.append(d["resid_rms"])
    o = np.argsort(x)
    return np.array(x)[o], np.array(pr)[o], np.array(pe)[o], np.array(xt), np.array(rr), len(fs)


xc, pc, prior, xt0, rr0, n0 = load(0.0, 850.0)
print(f"control: {n0} tiles")
fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True, constrained_layout=True)
fig2, axs2 = plt.subplots(3, 3, figsize=(15, 9), sharex=True, constrained_layout=True)
print(f"{'scenario':16s} {'max |effect| [hPa]':>20s}  half-width where |effect| > 1 / 5 / 10 hPa [km from cloud centre]")
for i, (an, p) in enumerate(ALTITUDES_HPA.items()):
    for j, (tn, tau) in enumerate(OPTICAL_DEPTHS.items()):
        x, ps, _, xt, rr, n = load(tau, p)
        cloud = Cloud(tau0=tau, p_centre_hpa=p)
        a = axs[i, j]
        if n < len(TILES):
            a.text(0.5, 0.5, f"incomplete ({n}/{len(TILES)} tiles)", transform=a.transAxes, ha="center")
            continue
        eff = ps - np.interp(x, xc, pc)
        a.plot(x, ps - prior, color="#8e44ad", lw=1.4, label="scenario: retrieved - prior")
        a.plot(xc, pc - prior, color="#8a93a1", lw=1.0, ls="--", label="clear control")
        a.plot(x, eff, color="#eb6834", lw=1.4, label="cloud effect (scenario - control)")
        a.axhline(0, color="#c3c2b7", lw=0.8)
        b = a.twinx()
        b.fill_between(x, cloud.tau(x), color="#2a78d6", alpha=0.18, lw=0)
        b.set_ylim(0, max(OPTICAL_DEPTHS.values()) * 1.1 if False else tau * 1.6)
        b.set_yticks([])
        a.set_title(f"{an} ({p:g} hPa), tau0 = {tau:g}", loc="left", fontsize=10)
        if j == 0:
            a.set_ylabel("surface pressure [hPa]")
        if i == 2:
            a.set_xlabel("along-slit position [km]")
        if i == 0 and j == 0:
            a.legend(frameon=False, fontsize=7, loc="lower left")
        hw = []
        for thr in (1, 5, 10):
            m = np.abs(eff) > thr
            hw.append(f"{np.max(np.abs(x[m] - cloud.x0_km)):.0f}" if m.any() else "-")
        print(f"{an + '_' + tn:16s} {np.max(np.abs(eff)):20.2f}  " + " / ".join(hw))
        a2 = axs2[i, j]
        a2.plot(xt, rr / np.interp(xt, xt0, rr0), "o-", color="#1baf7a")
        a2.axhline(1, color="#c3c2b7", lw=0.8)
        b2 = a2.twinx()
        b2.fill_between(x, cloud.tau(x), color="#2a78d6", alpha=0.18, lw=0)
        b2.set_ylim(0, tau * 1.6); b2.set_yticks([])
        a2.set_title(f"{an} ({p:g} hPa), tau0 = {tau:g}", loc="left", fontsize=10)
        if j == 0:
            a2.set_ylabel("fit residual rms / clear control")
        if i == 2:
            a2.set_xlabel("along-slit position [km]")
fig.suptitle("O2-A-only, non-scattering retrieval of surface pressure through a tapering-edge cloud (blue shading = cloud optical depth)", fontsize=11)
fig2.suptitle("Fit residual (rms per tile) relative to the clear-sky control (blue shading = cloud optical depth)", fontsize=11)
fig.savefig(REPO / "plots/cloud_o2a_pressure.png", dpi=120, bbox_inches="tight")
fig2.savefig(REPO / "plots/cloud_o2a_residual.png", dpi=120, bbox_inches="tight")
print("saved plots/cloud_o2a_pressure.png, plots/cloud_o2a_residual.png")
