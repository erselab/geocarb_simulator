"""Validate the analytic surface-pressure Jacobian under XRTM + aerosol against the RT finite difference (2026-09-26).

For anchors spanning the slit (mountain, urban bump, haze peak, background) and all four bands, with the smoke Mie properties:
  FD        : GEOCARB_PSURF_FD=1 (2 extra XRTM calls, the previous behaviour)
  analytic  : gert psurf_seed slot + aerosol layer-weight shift (the new default)
  partial   : the no-aerosol composition only, i.e. WITHOUT the two new terms (shows what they contribute)
Reports cosine, relative L2 error and worst pointwise error (over points with |FD| > 1e-3 of its max) vs FD.
    python scripts/check_psurf_jacobian.py [--aerosol-type smoke]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import gd_joint_block_retrieve as gjr  # noqa: E402
import gd_multiband_window as gmw  # noqa: E402
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert import jacobians as jac  # noqa: E402
from geocarb_gert.spectrum import simulate_spectrum, spectrum_and_jacobian  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--aerosol-type", default="smoke")
ap.add_argument("--xs", default="-250,-500,200,700")
a = ap.parse_args()
os.environ["GEOCARB_AEROSOL_TYPE"] = a.aerosol_type

absco, solar, geo, atm_center = gmw.load_inputs()
print(f"aerosol type {a.aerosol_type}; columns: cosine, rel L2 err, worst rel err (analytic vs FD | partial vs FD)", flush=True)
for f in range(4):
    label = gjr.GEOCARB_BANDS[f][0]
    _, wide_inst, _ = gjr.band_basics(f, atm_center, absco, geo, solar)
    for xk in [float(t) for t in a.xs.split(",")]:
        x = np.array([xk])
        atm = {n: float(als.STATE_FIELDS[n](x)[0]) for n in ("co2_ppm", "ch4_ppb", "co_ppb", "h2o_surface_vmr", "p_surface_hpa", "t_offset_k")}
        sfc = {"albedo": float(als.SURFACE_FIELDS["albedo"](x, label)[0])}
        for k in ("amplitude_aerosol", "height_aerosol", "thickness_aerosol"):
            sfc[k] = float(als.SURFACE_FIELDS[k](x, label)[0])
        os.environ.pop("GEOCARB_PSURF_FD", None)
        t0 = time.time()
        _, d_an = spectrum_and_jacobian(atm, ["p_surface_hpa"], absco, wide_inst, geo, solar, surface=sfc,
                                        aerosol_type=a.aerosol_type, solver="xrtm")
        t_an = time.time() - t0
        os.environ["GEOCARB_PSURF_FD"] = "1"
        t0 = time.time()
        _, d_fd = spectrum_and_jacobian(atm, ["p_surface_hpa"], absco, wide_inst, geo, solar, surface=sfc,
                                        aerosol_type=a.aerosol_type, solver="xrtm")
        t_fd = time.time() - t0
        os.environ.pop("GEOCARB_PSURF_FD", None)
        sr = simulate_spectrum(atm, sfc, absco, wide_inst, geo, solar, jacobians=True, aerosol_type=a.aerosol_type, solver="xrtm")
        base = jac._p_surface_dI_dparam_analytic(sr.result, atm)
        fd, an = d_fd["p_surface_hpa"], d_an["p_surface_hpa"]

        def m(y):
            cos = float(np.dot(y, fd) / (np.linalg.norm(y) * np.linalg.norm(fd)))
            rel = float(np.linalg.norm(y - fd) / np.linalg.norm(fd))
            mask = np.abs(fd) > 1e-3 * np.abs(fd).max()
            worst = float(np.max(np.abs(y[mask] - fd[mask]) / np.abs(fd[mask])))
            return cos, rel, worst
        ca, cb = m(an), m(base)
        print(f"FPA{f} x={xk:7.0f} km p={atm['p_surface_hpa']:6.1f} hPa tau_amp={sfc['amplitude_aerosol']:.2e} | "
              f"analytic cos {ca[0]:.6f} rel {ca[1]:.2e} worst {ca[2]:.2e} | partial cos {cb[0]:.6f} rel {cb[1]:.2e} worst {cb[2]:.2e} | "
              f"time analytic {t_an:.0f}s FD {t_fd:.0f}s", flush=True)
