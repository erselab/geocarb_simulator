#!/usr/bin/env python3
"""Mode-2 representative truth at anchor_density=4 (2026-09-04, user: "we
can probably speed things up a bit by using anchor density=4" for these
imperfect-prior ablations). Same mechanism as build_whole_slit_truth_v1.py
(render_at_anchors/footprint_average_scene, no spectral interpolation),
just a separate anchor_density -- kept as its own file/output (not
overwriting whole_slit_truth_v1.pkl's ad16 entry) since representability
is defined relative to a SPECIFIC anchor_density: an ad16 truth is not
representable at ad4 bin/anchor spacing and vice versa, so any retrieval
run at ad4 must be scored against an ad4-built truth, not the existing
ad16 one.
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

print("=== building whole-slit truth, mode=representative, anchor_density=4 ===", flush=True)
t0 = time.time()
A = build_whole_slit_truth(fpa, "representative", spectrum, wn_hires, ils, band_label,
                           g_ratio=1.0, anchor_density=4, n_workers=16)
print(f"  done in {time.time()-t0:.0f}s, shape={A.shape}, "
     f"range=[{A.min():.4g},{A.max():.4g}]", flush=True)

out_path = REPO_ROOT / "scratch_work" / "whole_slit_truth_ad4.pkl"
with open(out_path, "wb") as f:
    pickle.dump({"representative": A}, f)
print(f"\nsaved {out_path}")
