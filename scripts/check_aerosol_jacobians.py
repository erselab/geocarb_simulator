"""Are the analytic aerosol-row Jacobians (height, amplitude) and p_surface consistent with the forward model? (2026-09-26)

The joint-state diagnostic (diagnose_aerosol_height.py) found the analytic height column only 0.88 cosine / 51% off, amplitude 13% and
p_surface 3.7% off the central finite difference of the joint forward model at the last iterate of the 4-band aerosol solve. This isolates
the per-anchor, per-band Jacobian: for each of a few anchor states (scene truth, and the retrieved-like state with the solve's aerosol
amplitude/height) it compares spectrum_and_jacobian's analytic dI/d(height, amplitude, p_surface) with a central FD of simulate_spectrum.
    python scripts/check_aerosol_jacobians.py [--aerosol-type smoke]
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import gd_joint_block_retrieve as gjr  # noqa: E402
import gd_multiband_window as gmw  # noqa: E402
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.spectrum import simulate_spectrum, spectrum_and_jacobian  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--aerosol-type", default="smoke")
ap.add_argument("--xs", default="-280,-250,-220")
a = ap.parse_args()
os.environ["GEOCARB_AEROSOL_TYPE"] = a.aerosol_type
absco, solar, geo, atm_center = gmw.load_inputs()
st = pickle.load(open(REPO / "results/realistic_prior/multiband/check_aero4_tile12_analytic_psurf.pkl", "rb"))["joint"]["params"]
ret_amp = float(np.mean(st["amplitude_aerosol"]["values"]))
ret_h = float(np.mean(st["height_aerosol"]["values"]))
print(f"retrieved tile-12 mean amplitude {ret_amp:.3e}, height {ret_h:.0f} Pa (truth at -250 km: amp {als.SURFACE_FIELDS['amplitude_aerosol'](np.array([-250.0]), 'x')[0]:.3e}, "
      f"height {als.SURFACE_FIELDS['height_aerosol'](np.array([-250.0]), 'x')[0]:.0f} Pa)", flush=True)
ROWS = ["height_aerosol", "amplitude_aerosol", "p_surface_hpa"]
for f in range(4):
    label = gjr.GEOCARB_BANDS[f][0]
    _, wide_inst, _ = gjr.band_basics(f, atm_center, absco, geo, solar)
    for xk in [float(t) for t in a.xs.split(",")]:
        x = np.array([xk])
        atm = {n: float(als.STATE_FIELDS[n](x)[0]) for n in ("co2_ppm", "ch4_ppb", "co_ppb", "h2o_surface_vmr", "p_surface_hpa", "t_offset_k")}
        base = {"albedo": float(als.SURFACE_FIELDS["albedo"](x, label)[0]),
                "thickness_aerosol": float(als.SURFACE_FIELDS["thickness_aerosol"](x, label)[0])}
        for tag, amp, h in (("truth", float(als.SURFACE_FIELDS["amplitude_aerosol"](x, label)[0]), float(als.SURFACE_FIELDS["height_aerosol"](x, label)[0])),
                            ("retrieved-like", ret_amp, ret_h)):
            sfc = {**base, "amplitude_aerosol": amp, "height_aerosol": h}
            _, d = spectrum_and_jacobian(atm, ROWS, absco, wide_inst, geo, solar, surface=sfc, aerosol_type=a.aerosol_type, solver="xrtm")
            out = []
            for row in ROWS:
                if row == "p_surface_hpa":
                    val, key, tgt = atm["p_surface_hpa"], "p_surface_hpa", "atm"
                else:
                    val, key, tgt = sfc[row], row, "sfc"
                hh = 1e-3 * val
                def I(v):
                    at, sf = dict(atm), dict(sfc)
                    (at if tgt == "atm" else sf)[key] = v
                    return simulate_spectrum(at, sf, absco, wide_inst, geo, solar, aerosol_type=a.aerosol_type, solver="xrtm").I_hires
                fd = (I(val + hh) - I(val - hh)) / (2 * hh)
                an = d[row]
                cos = float(np.dot(an, fd) / (np.linalg.norm(an) * np.linalg.norm(fd) + 1e-300))
                rel = float(np.linalg.norm(an - fd) / (np.linalg.norm(fd) + 1e-300))
                out.append(f"{row.split('_')[0]:9s} cos {cos:.5f} rel {rel:.2e}")
            print(f"FPA{f} x={xk:5.0f} {tag:14s} amp {amp:.2e} h {h:8.0f} | " + " | ".join(out), flush=True)
