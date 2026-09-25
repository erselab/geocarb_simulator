"""How long does one XRTM cloud anchor take in each band? (2026-09-25) Sizing check for the cloud experiments."""
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ["GEOCARB_AEROSOL_TYPE"] = "cloud_water"
import gd_joint_block_retrieve as gjr  # noqa: E402
import gd_multiband_window as gmw  # noqa: E402
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.cloud_scene import Cloud  # noqa: E402
from geocarb_gert.spectrum import simulate_spectrum  # noqa: E402

absco, solar, geo, atm_center = gmw.load_inputs()
cloud = Cloud(tau0=2.0, p_centre_hpa=600.0)
x = 700.0
atm = {n: float(als.STATE_FIELDS[n](np.array([x]))[0]) for n in ("co2_ppm", "ch4_ppb", "co_ppb", "h2o_surface_vmr", "p_surface_hpa", "t_offset_k")}
cf = cloud.surface_fields()
print("cloud tau at x=700:", float(cloud.tau(x)), flush=True)
for f in range(4):
    _, wide_inst, alb = gjr.band_basics(f, atm_center, absco, geo, solar)
    sfc_clear = {"albedo": float(als.SURFACE_FIELDS["albedo"](np.array([x]), gjr.GEOCARB_BANDS[f][0])[0])}
    sfc_cloud = {**sfc_clear, **{k: float(fn(np.array([x]))[0]) for k, fn in cf.items()}}
    out = []
    for lab, sfc, solver in (("single_scatter clear", sfc_clear, "single_scatter"), ("xrtm clear", sfc_clear, "xrtm"), ("xrtm cloud", sfc_cloud, "xrtm")):
        t0 = time.time()
        I = simulate_spectrum(atm, sfc, absco, wide_inst, geo, solar, solver=solver).I_hires
        out.append((lab, time.time() - t0, float(np.mean(I))))
    print(f"FPA{f}: " + " | ".join(f"{l} {t:.1f}s (mean I {m:.4g})" for l, t, m in out), flush=True)
