#!/usr/bin/env python3
"""Experiment B (2026-09-04, user): co2_ppm + p_surface_hpa free only,
scored against Mode-1 DENSE truth (500m, NOT representable at this
retrieval's own ad4 bin/anchor grid -- a genuine representability gap,
unlike every ad4 experiment so far this session) -- with every FROZEN
row (ch4_ppb, co_ppb, h2o_surface_vmr, AND albedo) fixed at the exact
continuous truth (the same STATE_FIELDS/SURFACE_FIELDS the 500m dense
truth image was itself rendered from), not the structural prior. Free
rows (co2_ppm, p_surface_hpa) keep their usual structural (imperfect)
prior, consistent with every other run this session.

This isolates: does a genuine representability gap (dense, non-
representable truth) degrade the co2/p_surface retrieval differently
than Sec.14's frozen-row-contamination ablation found under a
REPRESENTABLE (Mode-2) truth -- with frozen-row error controlled for
(exact) either way, so any NEW degradation here is attributable to the
representability gap itself, not to frozen-row prior error.

Reuses scratch_work/whole_slit_truth_v1.pkl's "dense" entry directly --
Mode-1 dense truth is built at a fixed dx_km=0.5 (500m) independent of
the RETRIEVAL's own anchor_density (ad4 here vs. the ad16 config used
when that truth was originally built), so it needs no rebuilding.
"""
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import gd_test as gdt  # noqa: E402
import gd_joint_block_whole_slit_sweep as sweep  # noqa: E402
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from gd_joint_block_retrieve import FPA  # noqa: E402

with open(REPO_ROOT / "scratch_work" / "whole_slit_truth_v1.pkl", "rb") as f:
    truth = pickle.load(f)
A_TRUTH = truth["dense"]

FREE_ATM_ROWS = {"co2_ppm", "p_surface_hpa"}
_merged_fields = {
    name: (als.STATE_FIELDS_PRIOR[name] if name in FREE_ATM_ROWS else als.STATE_FIELDS[name])
    for name in als.STATE_FIELDS
}
als.PRIOR_FIELD_SETS["dense_frozen_exact"] = _merged_fields
# albedo is FROZEN here (not in --free) -- per the user's instruction,
# frozen variables use the exact 500m truth values, so this is
# SURFACE_FIELDS (exact), NOT SURFACE_FIELDS_PRIOR (structural) --
# different from Sec.14.1's own "frozen_exact" registration, where
# albedo was itself free and kept its structural prior.
als.SURFACE_PRIOR_FIELD_SETS["dense_frozen_exact"] = als.SURFACE_FIELDS

print("dense_frozen_exact atmosphere prior rows:", flush=True)
for name in als.STATE_FIELDS:
    src = "structural (free, unchanged)" if name in FREE_ATM_ROWS else "EXACT TRUTH (frozen, ablated)"
    print(f"  {name}: {src}", flush=True)
print("  albedo: EXACT TRUTH (frozen)", flush=True)


def _fake_band_setup_cached(fpa, atm_center, absco, geo, solar, snr, n_lookup_samples,
                            n_workers, uniform, barcode, barcode_bars, noise, noise_seed,
                            realistic_barcode=False, vary_albedo=False, use_cache=True,
                            fields=None, surface_fields=None, resolution_tag=None):
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[fpa]
    from gert.instrument import Instrument, SpectralWindow, ILS
    from gd_joint_block_retrieve import real_wavenumber_range
    wn_min, wn_max = real_wavenumber_range(fpa, margin_cm1=10.0)
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=list(mols), label=label, hires_spacing=0.01, channels_per_fwhm=3)
    print(f"  [FAKE band_setup_cached] injecting precomputed Mode-1 DENSE (500m) "
         f"truth array (shape={A_TRUTH.shape}) instead of rendering", flush=True)
    return dict(fpa=fpa, label=label, mols=mols, R=R, wn_min=wn_min, wn_max=wn_max,
               fwhm_cm=fwhm_cm, albedo=None, A=A_TRUTH, wn_hires=wide_win.wn_hires,
               radiance=None, ils=wide_win.ils, wn_grid=None, noise_n0=None, noise_n1=None,
               noise_i_max=None, sat_mask=None, noise_arr=None, x_km_of_row=None, xtrue_of_row=None)


gdt._band_setup_cached = _fake_band_setup_cached
sweep.gdt._band_setup_cached = _fake_band_setup_cached

RESDIR = REPO_ROOT / "results" / "realistic_prior" / "config_matrix" / "resolution_matched_v1" / "results"
RESDIR.mkdir(parents=True, exist_ok=True)

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
anchor_workers = int(os.environ.get("ANCHOR_WORKERS", "8"))

sys.argv = [
    "gd_joint_block_whole_slit_sweep.py",
    "--g-ratio", "1", "--free", "co2_ppm,p_surface_hpa", "--vary-albedo",
    "--prior-fields", "dense_frozen_exact", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", "1",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / "co2p-58-g1-ad4-pf-densefrozenexact-DENSETRUTH_analytic.pkl"),
]

raise SystemExit(sweep.main())
