#!/usr/bin/env python3
"""Experiment B, anchor-albedo variant (2026-09-05, user: "I'd like to
give the retrieval the true albedo at every anchor for this retrieval
version and keep it frozen"). Identical to Experiment B
(retrieval_co2p_dense_truth_frozenexact_ad4.py) -- co2_ppm+p_surface_hpa
free, Mode-1 DENSE (500m) truth, ch4/co/h2o/albedo frozen at exact truth
-- except the frozen ALBEDO row now lives on this window's own ANCHOR
grid (--surface-positions anchor, new flag) instead of sharing the
atmosphere rows' (much coarser) bin_centers grid.

Isolates: Sec.16.2 found co2/p_surface errors 100-250x worse than the
equivalent representable-truth case, with a working (not yet confirmed)
explanation that a frozen row's own piecewise-linear BIN-TO-BIN
reconstruction can't capture real structure below bin spacing -- most
likely albedo's own fine texture -- even when each bin's own node value
is exact. Giving albedo its true value at every ANCHOR (the same fine
grid the forward model already renders from, matching the retrieval's
own ad4 resolution) removes that gap for albedo specifically, while
ch4/co/h2o stay on the coarser bin_centers grid unchanged. If this alone
closes most of the blowup, it confirms albedo's sub-bin texture was the
driver; if the blowup persists, the explanation needs revisiting.
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

with open(REPO_ROOT / "archive" / "initial_build" / "scratch_work" / "whole_slit_truth_v1.pkl", "rb") as f:
    truth = pickle.load(f)
A_TRUTH = truth["dense"]

FREE_ATM_ROWS = {"co2_ppm", "p_surface_hpa"}
_merged_fields = {
    name: (als.STATE_FIELDS_PRIOR[name] if name in FREE_ATM_ROWS else als.STATE_FIELDS[name])
    for name in als.STATE_FIELDS
}
als.PRIOR_FIELD_SETS["dense_frozen_exact"] = _merged_fields
als.SURFACE_PRIOR_FIELD_SETS["dense_frozen_exact"] = als.SURFACE_FIELDS

print("dense_frozen_exact atmosphere prior rows (albedo on ANCHOR grid this time):", flush=True)
for name in als.STATE_FIELDS:
    src = "structural (free, unchanged)" if name in FREE_ATM_ROWS else "EXACT TRUTH (frozen, bin_centers grid)"
    print(f"  {name}: {src}", flush=True)
print("  albedo: EXACT TRUTH (frozen, ANCHOR grid -- --surface-positions anchor)", flush=True)


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
    "--surface-positions", "anchor",
    "--prior-fields", "dense_frozen_exact", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", "1",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / "co2p-58-g1-ad4-pf-densefrozenexact-anchoralbedo-DENSETRUTH_analytic.pkl"),
]

raise SystemExit(sweep.main())
