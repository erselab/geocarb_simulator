#!/usr/bin/env python3
"""Experiment B, free-albedo variant (2026-09-05, user: "let's try
unfreezing albedo"). Same config as Experiment B (co2_ppm+p_surface_hpa
free, Mode-1 DENSE/500m truth, ch4/co/h2o frozen at exact truth on the
bin_centers grid) except albedo is now FREE (on the default shared
bin_centers grid, same as every other free-albedo experiment this
session -- e.g. Sec.14/16.1's own convention) instead of frozen, with
its own structural (imperfect) prior -- matching how every OTHER free
row in this experiment family is treated (start from an imperfect
prior, let GN move it), not an exact prior (which wouldn't test
anything, since a free row starting at truth has no reason to move).

Isolates: does letting GN actually FIT albedo (even at coarse bin-
center resolution) do better or worse than PINNING it exactly at bin
centers (Sec.16.2's original 100-250x blowup) or at every anchor
(this session's anchor-albedo variant, ~90% closed)? If free albedo at
bin-center resolution does comparably well to (or better than) the
anchor-frozen case, that would suggest the RETRIEVAL fitting albedo's
coarse-scale structure matters more than pinning its fine-scale texture
exactly -- a different mechanism than the working "sub-bin texture"
explanation.
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
# albedo is FREE here -- structural (imperfect) prior, matching every
# other free-albedo experiment this session (Sec.14/16.1's own
# convention: a free row starts from an imperfect prior, not truth).
als.SURFACE_PRIOR_FIELD_SETS["dense_frozen_exact"] = als.SURFACE_FIELDS_PRIOR

print("dense_frozen_exact atmosphere prior rows (albedo now FREE):", flush=True)
for name in als.STATE_FIELDS:
    src = "structural (free, unchanged)" if name in FREE_ATM_ROWS else "EXACT TRUTH (frozen, bin_centers grid)"
    print(f"  {name}: {src}", flush=True)
print("  albedo: structural (FREE, bin_centers grid -- unfrozen this run)", flush=True)


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
    "--g-ratio", "1", "--free", "co2_ppm,p_surface_hpa,albedo", "--vary-albedo",
    "--prior-fields", "dense_frozen_exact", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", "1",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / "co2palbedo-58-g1-ad4-pf-densefrozenexact-freealbedo-DENSETRUTH_analytic.pkl"),
]

raise SystemExit(sweep.main())
