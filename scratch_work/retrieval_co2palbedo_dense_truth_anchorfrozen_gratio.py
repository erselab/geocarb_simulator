#!/usr/bin/env python3
"""Whole-slit sweep, finer g_ratio exploration (2026-09-05, user: "I'd
like to try a whole slit sweep with co2, pressure, and albedo unfrozen
and the others set at the anchor truth values. Use g_ratios of 0.5 and
0.25" -- explicitly superseding the earlier "skip g_ratio=0.5" guidance,
which the user says was based on the old, since-replaced truth-
rendering mechanism where finer g_ratio didn't matter much).

co2_ppm, p_surface_hpa, albedo all FREE (structural prior, shared
bin_centers grid -- albedo's own convention when free, matching every
other free-albedo experiment this session). ch4_ppb/co_ppb/
h2o_surface_vmr frozen at their EXACT value on the ANCHOR grid (this
session's --frozen-atmosphere-positions anchor mechanism, built
specifically for this kind of test). Mode-1 DENSE (500m) truth -- the
genuinely non-representable truth this whole investigation has used;
"anchor truth values" for the frozen rows only differs meaningfully
from plain bin-center-exact under genuine sub-bin structure, which only
Mode-1 dense truth has by construction.

G_RATIO is read from the GRATIO env var so the same driver serves both
0.5 and 0.25 without duplicating the file.
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
# albedo is FREE this run -- structural prior, matching every other
# free-albedo experiment this session.
als.SURFACE_PRIOR_FIELD_SETS["dense_frozen_exact"] = als.SURFACE_FIELDS_PRIOR

G_RATIO = os.environ.get("GRATIO", "0.5")
print(f"g_ratio={G_RATIO}", flush=True)
print("dense_frozen_exact atmosphere prior rows:", flush=True)
for name in als.STATE_FIELDS:
    src = "structural (FREE, bin_centers grid)" if name in FREE_ATM_ROWS else "EXACT TRUTH (frozen, ANCHOR grid)"
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
gratio_tag = G_RATIO.replace(".", "p")

sys.argv = [
    "gd_joint_block_whole_slit_sweep.py",
    "--g-ratio", G_RATIO, "--free", "co2_ppm,p_surface_hpa,albedo", "--vary-albedo",
    "--frozen-atmosphere-positions", "anchor",
    "--prior-fields", "dense_frozen_exact", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", G_RATIO,
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / f"co2palbedo-58-g{gratio_tag}-ad4-pf-anchorfrozen-DENSETRUTH_analytic.pkl"),
]

raise SystemExit(sweep.main())
