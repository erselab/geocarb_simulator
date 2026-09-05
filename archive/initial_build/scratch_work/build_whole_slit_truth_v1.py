#!/usr/bin/env python3
"""Build both whole-slit truth images (Mode 1 dense, Mode 2 representative)
at overlap=0, via the new shared no-spectral-interpolation forward model
(render_at_anchors/footprint_average_scene). 2026-09-02, user-directed
design.
"""
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics  # noqa: E402
from gd_joint_block_whole_slit_sweep import _make_state_spectrum  # noqa: E402
from gd_build_resolution_matched_truth import build_whole_slit_truth  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402

fpa = FPA
band_label = GEOCARB_BANDS[fpa][0]
print("Loading absco/solar...", flush=True)
block = gg.geocarb_demo(verbose=False)["blocks"][0]
_, _, geo = sample_geometries(block, n=1, seed=0)[0]
absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
atm_center = als.atmosphere_at(0.0)
snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)
wn_hires = wide_win.wn_hires
ils = wide_win.ils
print("done.\n", flush=True)

out = {}
for mode, kwargs in [
    ("representative", dict(g_ratio=1.0, anchor_density=16)),
    ("dense", dict(dx_km=0.5)),
]:
    print(f"=== building whole-slit truth, mode={mode} {kwargs} ===", flush=True)
    t0 = time.time()
    A = build_whole_slit_truth(fpa, mode, spectrum, wn_hires, ils, band_label,
                               n_workers=16, **kwargs)
    print(f"  done in {time.time()-t0:.0f}s, shape={A.shape}, "
         f"range=[{A.min():.4g},{A.max():.4g}]", flush=True)
    out[mode] = A

out_path = REPO_ROOT / "scratch_work" / "whole_slit_truth_v1.pkl"
with open(out_path, "wb") as f:
    pickle.dump(out, f)
print(f"\nsaved {out_path}")
