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
# --noise: the same experiment with shot noise (seeds 1-3; each seed's clear control has the identical noise draw, so the
# scenario-minus-control difference cancels most of the noise). Lines are per seed; the residual panel shows the seed mean
# with the min-max band.
NOISE = "--noise" in sys.argv
SEEDS = (1, 2, 3) if NOISE else (None,)
OUTSUF = "_noise" if NOISE else ""


def load(tau, p, seed=None):
    ns = f"_noise{seed}" if seed else ""
    fs = sorted(glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0_r0-*_free-p-albedo_cover_g1.0_etaslit_prior-realistic{ns}_cloud-tau{tau:g}-p{p:g}.pkl")))
    x, pr, pe, res, xt, rr = [], [], [], [], [], []
    for f in fs:
        d = pickle.load(open(f, "rb"))
        q = d["joint"]["params"]["p_surface_hpa"]
        xs = np.asarray(q["positions"]) * H
        x += list(xs); pr += list(q["values"]); pe += list(q["prior"])
        xt.append(0.5 * (xs.min() + xs.max())); rr.append(d["resid_rms"])
    o = np.argsort(x)
    return np.array(x)[o], np.array(pr)[o], np.array(pe)[o], np.array(xt), np.array(rr), len(fs)


CTRL = {sd: load(0.0, 850.0, sd) for sd in SEEDS}
print("control tiles:", {sd: v[5] for sd, v in CTRL.items()})
fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True, constrained_layout=True)
fig2, axs2 = plt.subplots(3, 3, figsize=(15, 9), sharex=True, constrained_layout=True)
SCOL = {None: "#8e44ad", 1: "#8e44ad", 2: "#1baf7a", 3: "#eb6834"}
print(f"{'scenario':16s} {'max |effect| [hPa], per seed':>32s}  tau where mean residual/control first exceeds 1.1 / 1.5 (outermost tile)")
for i, (an, p) in enumerate(ALTITUDES_HPA.items()):
    for j, (tn, tau) in enumerate(OPTICAL_DEPTHS.items()):
        cloud = Cloud(tau0=tau, p_centre_hpa=p)
        a, a2 = axs[i, j], axs2[i, j]
        runs = {sd: load(tau, p, sd) for sd in SEEDS}
        if any(r[5] < len(TILES) for r in runs.values()):
            for aa in (a, a2):
                aa.text(0.5, 0.5, "incomplete", transform=aa.transAxes, ha="center")
            continue
        maxeff, ratios = [], []
        for sd, (x, ps, _, xt, rr, n) in runs.items():
            xc, pc, prior, xt0, rr0, _ = CTRL[sd]
            eff = ps - np.interp(x, xc, pc)
            maxeff.append(np.max(np.abs(eff)))
            a.plot(x, eff, color=SCOL[sd], lw=1.0, alpha=0.85, label=("cloud effect" if sd in (None, 1) else None) if not NOISE else f"seed {sd}")
            ratios.append(rr / np.interp(xt, xt0, rr0))
        a.plot(xc, pc - prior, color="#8a93a1", lw=0.9, ls="--", label="clear control (retrieved - prior)")
        a.axhline(0, color="#c3c2b7", lw=0.8)
        b = a.twinx(); b.fill_between(x, cloud.tau(x), color="#2a78d6", alpha=0.18, lw=0); b.set_ylim(0, tau * 1.6); b.set_yticks([])
        a.set_title(f"{an} ({p:g} hPa), tau0 = {tau:g}", loc="left", fontsize=10)
        R = np.array(ratios)
        a2.fill_between(xt, R.min(0), R.max(0), color="#1baf7a", alpha=0.25, lw=0)
        a2.plot(xt, R.mean(0), "o-", color="#1baf7a")
        a2.axhline(1, color="#c3c2b7", lw=0.8)
        b2 = a2.twinx(); b2.fill_between(x, cloud.tau(x), color="#2a78d6", alpha=0.18, lw=0); b2.set_ylim(0, tau * 1.6); b2.set_yticks([])
        a2.set_title(f"{an} ({p:g} hPa), tau0 = {tau:g}", loc="left", fontsize=10)
        thr = []
        for t_ in (1.1, 1.5):
            idx = np.where(R.mean(0) > t_)[0]
            thr.append(f"{min(cloud.tau(xt[idx[0]]), cloud.tau(xt[idx[-1]])):.3f}" if idx.size else "-")
        print(f"{an + '_' + tn:16s} " + " ".join(f"{m:8.1f}" for m in maxeff) + f"   {thr[0]:>7s} / {thr[1]:>7s}")
        if j == 0:
            a.set_ylabel("surface pressure [hPa]"); a2.set_ylabel("fit residual rms / clear control")
        if i == 2:
            a.set_xlabel("along-slit position [km]"); a2.set_xlabel("along-slit position [km]")
        if i == 0 and j == 0:
            a.legend(frameon=False, fontsize=7, loc="lower left")
ttl = "with noise, 3 seeds" if NOISE else "no noise"
fig.suptitle(f"O2-A-only, non-scattering retrieval: cloud effect on retrieved surface pressure ({ttl}; blue = cloud optical depth)", fontsize=11)
fig2.suptitle(f"Fit residual per tile relative to the clear control ({ttl}; blue = cloud optical depth)", fontsize=11)
fig.savefig(REPO / f"plots/cloud_o2a_pressure{OUTSUF}.png", dpi=120, bbox_inches="tight")
fig2.savefig(REPO / f"plots/cloud_o2a_residual{OUTSUF}.png", dpi=120, bbox_inches="tight")
print(f"saved plots/cloud_o2a_pressure{OUTSUF}.png, plots/cloud_o2a_residual{OUTSUF}.png")
