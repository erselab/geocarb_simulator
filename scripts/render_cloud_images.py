"""Render sub-slit detector radiance images of a scene with a tapering-edge cloud, all four bands (2026-09-25).

One task = one scenario (clear control, or an altitude x optical-depth cloud from cloud_scene.scenarios()). For each FPA it
renders the detector rows that see x in [X_LO, X_HI] km through geocarb_gert.joint_state.render_at_anchors (RT run fresh at
every anchor, real per-pixel footprints, keystone and PSF) with the XRTM two-stream solver, so the clear and cloudy images
use the SAME physics and their ratio isolates the cloud. Anchors are ANCHOR_KM apart.
    python scripts/render_cloud_images.py --scenario mid_moderate [--workers 16]
Output: results/cloud_images/<scenario>_fpa<n>.npz  (A = radiance [rows, 1024], rows, x_km of each row's centre column)
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
os.environ["GEOCARB_AEROSOL_TYPE"] = "cloud_water"
import gd_joint_block_retrieve as gjr  # noqa: E402
import gd_multiband_window as gmw  # noqa: E402
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.cloud_scene import scenarios  # noqa: E402
from geocarb_gert.joint_state import render_at_anchors  # noqa: E402
from geocarb_gert.multiband_geometry import band_tables  # noqa: E402
from geocarb_gert.spectrum import simulate_spectrum  # noqa: E402

X_LO, X_HI = 430.0, 970.0      # km along the slit: the cloud (700 +/- 180 km) plus clear margins
ANCHOR_KM = 1.0
PAD = 4
H = als.SLIT_HALF_KM

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True, help="clear | <altitude>_<thickness>, e.g. mid_moderate")
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--fpas", default="0,1,2,3")
a = ap.parse_args()
cloud = None if a.scenario == "clear" else scenarios()[a.scenario]
out = REPO / "results/cloud_images"
out.mkdir(parents=True, exist_ok=True)

absco, solar, geo, atm_center = gmw.load_inputs()
for f in [int(t) for t in a.fpas.split(",")]:
    t0 = time.time()
    label = gjr.GEOCARB_BANDS[f][0]
    _, wide_inst, albedo0 = gjr.band_basics(f, atm_center, absco, geo, solar)
    bt = band_tables(f)
    xc = bt["eta_c"] * H
    rows = np.where((xc >= X_LO) & (xc <= X_HI))[0]
    lo_eta = bt["eta_lo"][rows[0] - PAD:rows[-1] + PAD + 1].min()
    hi_eta = bt["eta_hi"][rows[0] - PAD:rows[-1] + PAD + 1].max()
    etas = np.arange(lo_eta, hi_eta + 1e-9, ANCHOR_KM / H)
    x = etas * H
    atm_params = {n: np.asarray(als.STATE_FIELDS[n](x), dtype=float) for n in als.STATE_FIELDS}
    surf = {"albedo": np.asarray(als.SURFACE_FIELDS["albedo"](x, label), dtype=float)}
    if cloud is not None:
        surf.update({k: np.asarray(fn(x), dtype=float) for k, fn in cloud.surface_fields().items()})

    def spectrum(atm_p, surf_p=None):
        return simulate_spectrum(atm_p, dict(surf_p or {}), absco, wide_inst, geo, solar, solver="xrtm",
                                 aerosol_type="cloud_water").I_hires

    print(f"FPA{f} {a.scenario}: {rows.size} rows, {etas.size} anchors ({x.min():.0f}..{x.max():.0f} km)", flush=True)
    A = render_at_anchors(f, np.arange(rows[0] - 0, rows[-1] + 1), etas, atm_params, surf, spectrum,
                          wide_inst.windows[0].wn_hires, wide_inst.windows[0].ils, pad=PAD, n_workers=a.workers)
    np.savez_compressed(out / f"{a.scenario}_fpa{f}.npz", A=np.asarray(A), rows=np.arange(rows[0], rows[-1] + 1), x_km=xc[rows[0]:rows[-1] + 1])
    print(f"  saved ({time.time() - t0:.0f}s)", flush=True)
