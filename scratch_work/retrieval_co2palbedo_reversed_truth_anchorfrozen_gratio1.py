#!/usr/bin/env python3
"""Keystone-vs-truth-structure control experiment (2026-09-06, user:
"In order to isolate the effects of keystone versus the structure in
the truth, we can run parallel experiments with the truth profile
along the slit reversed"). Identical to Sec.6's g_ratio config
(co2_ppm/p_surface_hpa/albedo free, structural prior; ch4_ppb/co_ppb/
h2o_surface_vmr frozen exact on the ANCHOR grid) except EVERY truth
field is spatially mirrored (fn(x_km) -> fn(-x_km)) -- the instrument
geometry (row<->eta mapping, keystone curve) is completely untouched,
so rows_crossed at row 1013-1023 is identical to every other run this
session; only WHICH physical along-slit content sits there changes.

If rows 1013-1023 still shows comparably large error under this
REVERSED truth, that is decisive confirmation the effect is a geometry
(keystone) effect (Sec.5), not a coincidence of unlucky truth structure
at that specific location. Run at g_ratio=1 (not 0.5/0.25) to keep this
a clean, fast, single-variable control rather than compounding it with
the resolution question Sec.6 already answered separately.

Uses the REVERSED Mode-1 dense truth image
(scratch_work/whole_slit_truth_reversed.pkl,
scratch_work/build_whole_slit_truth_reversed.py) and applies the SAME
field-reversal monkeypatch here too, so the frozen rows' own "exact
truth on the anchor grid" values match what the truth image was
actually rendered from (not the original, un-reversed fields).
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


def _reversed1(fn):
    return lambda x_km, _fn=fn: _fn(-np.asarray(x_km, dtype=float))


def _reversed2(fn):
    return lambda x_km, label, _fn=fn: _fn(-np.asarray(x_km, dtype=float), label)


_ORIG_STATE_FIELDS = dict(als.STATE_FIELDS)
_ORIG_STATE_FIELDS_PRIOR = dict(als.STATE_FIELDS_PRIOR)
_ORIG_SURFACE_FIELDS = dict(als.SURFACE_FIELDS)
_ORIG_SURFACE_FIELDS_PRIOR = dict(als.SURFACE_FIELDS_PRIOR)

als.STATE_FIELDS = {name: _reversed1(fn) for name, fn in _ORIG_STATE_FIELDS.items()}
als.STATE_FIELDS_PRIOR = {name: _reversed1(fn) for name, fn in _ORIG_STATE_FIELDS_PRIOR.items()}
als.SURFACE_FIELDS = {name: _reversed2(fn) for name, fn in _ORIG_SURFACE_FIELDS.items()}
als.SURFACE_FIELDS_PRIOR = {name: _reversed2(fn) for name, fn in _ORIG_SURFACE_FIELDS_PRIOR.items()}
print("Reversed STATE_FIELDS/STATE_FIELDS_PRIOR/SURFACE_FIELDS/"
     "SURFACE_FIELDS_PRIOR installed (matching the reversed truth image).", flush=True)

with open(REPO_ROOT / "scratch_work" / "whole_slit_truth_reversed.pkl", "rb") as f:
    truth = pickle.load(f)
A_TRUTH = truth["dense"]

FREE_ATM_ROWS = {"co2_ppm", "p_surface_hpa"}
_merged_fields = {
    name: (als.STATE_FIELDS_PRIOR[name] if name in FREE_ATM_ROWS else als.STATE_FIELDS[name])
    for name in als.STATE_FIELDS
}
als.PRIOR_FIELD_SETS["dense_frozen_exact_reversed"] = _merged_fields
als.SURFACE_PRIOR_FIELD_SETS["dense_frozen_exact_reversed"] = als.SURFACE_FIELDS_PRIOR

print("dense_frozen_exact_reversed atmosphere prior rows:", flush=True)
for name in als.STATE_FIELDS:
    src = "structural (FREE, bin_centers grid)" if name in FREE_ATM_ROWS else "EXACT REVERSED TRUTH (frozen, ANCHOR grid)"
    print(f"  {name}: {src}", flush=True)
print("  albedo: structural (FREE, bin_centers grid)", flush=True)


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
    print(f"  [FAKE band_setup_cached] injecting precomputed REVERSED Mode-1 DENSE "
         f"(500m) truth array (shape={A_TRUTH.shape}) instead of rendering", flush=True)
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
    "--frozen-atmosphere-positions", "anchor",
    "--prior-fields", "dense_frozen_exact_reversed", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", "1",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / "co2palbedo-58-g1-ad4-pf-anchorfrozen-REVERSEDTRUTH_analytic.pkl"),
]

raise SystemExit(sweep.main())
