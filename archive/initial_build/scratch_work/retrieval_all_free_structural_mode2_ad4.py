#!/usr/bin/env python3
"""Experiment A (2026-09-04, user): imperfect (structural) prior, Mode-2
representative truth, FPA2, with ALL currently-modeled state elements
free: co2_ppm, p_surface_hpa, h2o_surface_vmr, albedo (temperature is not
yet a state parameter -- see docs/PROJECT_STATUS.md's plan for adding
T_offset_K; ch4_ppb/co_ppb stay frozen since FPA2 has zero sensitivity to
either). Goal: with everything free at once (a real "we won't know
anything" configuration, one step closer to a genuine retrieval), look
for cross-talk between the four free rows and how it interacts with
keystone -- building on Sec.14's single-free-set (co2/p_surface/albedo)
findings, now with h2o_surface_vmr added to the mix.

Same mechanism as the ad4 structural/frozen_exact array scripts: inject
the precomputed Mode-2 representative truth (representability=0 by
construction at this retrieval's own ad4 bin/anchor grid) via a fake
_band_setup_cached, so the run goes straight to solving.
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
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from gd_joint_block_retrieve import FPA  # noqa: E402

with open(REPO_ROOT / "scratch_work" / "whole_slit_truth_ad4.pkl", "rb") as f:
    truth = pickle.load(f)
A_TRUTH = truth["representative"]


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
    print(f"  [FAKE band_setup_cached] injecting precomputed Mode-2 representative "
         f"truth array (ad4, shape={A_TRUTH.shape}) instead of rendering", flush=True)
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
    "--g-ratio", "1",
    "--free", "co2_ppm,p_surface_hpa,h2o_surface_vmr,albedo", "--vary-albedo",
    "--prior-fields", "structural", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", "1",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / "co2ph2oalbedo-58-g1-ad4-pf-structural-MODE2TRUTH_analytic.pkl"),
]

raise SystemExit(sweep.main())
