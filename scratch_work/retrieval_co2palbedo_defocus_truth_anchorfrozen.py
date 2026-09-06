#!/usr/bin/env python3
"""Defocus (wide-PSF) experiment (2026-09-06, user: "I'd like to plan
some experiments with wider PSFs to simulate a defocusing effect ...
It doesn't affect the spectral resolution, just the spatial blurring").

Same base config as Sec.6/7/8's anchor-frozen driver (co2_ppm/
p_surface_hpa/albedo free, structural prior; ch4_ppb/co_ppb/
h2o_surface_vmr frozen exact on the ANCHOR grid), `g_ratio=1` fixed (kept
a single-variable control, matching Sec.7's own choice), but against a
truth image rendered with a WIDER along-slit PSF
(`build_whole_slit_truth_defocused.py`) instead of the nominal 1.5px one.

Reads TWO independent env vars:
  TRUTH_PSF_FWHM_PX     -- which defocused truth image to load (selects
                           scratch_work/whole_slit_truth_defocus_fwhm{N}px.pkl)
  RETRIEVAL_PSF_FWHM_PX -- what PSF the RETRIEVAL's own forward model
                           assumes (--retrieval-psf-fwhm-px)
Neither has a fallback default -- both must be set explicitly at
submission time (this session's own GRATIO near-miss lesson: never rely
on a driver's silent default matching intent by coincidence).

TRUTH_PSF_FWHM_PX == RETRIEVAL_PSF_FWHM_PX simulates a KNOWN, modeled
defocus (the mechanism moved and the retrieval was told). TRUTH_PSF_
FWHM_PX > RETRIEVAL_PSF_FWHM_PX (typically RETRIEVAL_PSF_FWHM_PX=1.5,
nominal) simulates an UNCORRECTED/uncalibrated defocus event.
"""
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import gd_test as gdt  # noqa: E402
import gd_joint_block_whole_slit_sweep as sweep  # noqa: E402
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from gd_joint_block_retrieve import FPA  # noqa: E402

TRUTH_PSF_FWHM_PX = os.environ["TRUTH_PSF_FWHM_PX"]
RETRIEVAL_PSF_FWHM_PX = os.environ["RETRIEVAL_PSF_FWHM_PX"]
print(f"TRUTH_PSF_FWHM_PX={TRUTH_PSF_FWHM_PX}  RETRIEVAL_PSF_FWHM_PX={RETRIEVAL_PSF_FWHM_PX}", flush=True)

truth_fwhm_tag = str(float(TRUTH_PSF_FWHM_PX)).replace(".", "p")
truth_path = REPO_ROOT / "scratch_work" / f"whole_slit_truth_defocus_fwhm{truth_fwhm_tag}px.pkl"
with open(truth_path, "rb") as f:
    truth = pickle.load(f)
A_TRUTH = truth["dense"]
print(f"loaded {truth_path}, shape={A_TRUTH.shape}", flush=True)

FREE_ATM_ROWS = {"co2_ppm", "p_surface_hpa"}
_merged_fields = {
    name: (als.STATE_FIELDS_PRIOR[name] if name in FREE_ATM_ROWS else als.STATE_FIELDS[name])
    for name in als.STATE_FIELDS
}
als.PRIOR_FIELD_SETS["dense_frozen_exact_defocus"] = _merged_fields
als.SURFACE_PRIOR_FIELD_SETS["dense_frozen_exact_defocus"] = als.SURFACE_FIELDS_PRIOR

print("dense_frozen_exact_defocus atmosphere prior rows:", flush=True)
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
    print(f"  [FAKE band_setup_cached] injecting precomputed DEFOCUSED (PSF FWHM="
         f"{TRUTH_PSF_FWHM_PX}px) Mode-1 DENSE truth array (shape={A_TRUTH.shape}) "
         f"instead of rendering", flush=True)
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
retr_fwhm_tag = str(float(RETRIEVAL_PSF_FWHM_PX)).replace(".", "p")

sys.argv = [
    "gd_joint_block_whole_slit_sweep.py",
    "--g-ratio", "1", "--free", "co2_ppm,p_surface_hpa,albedo", "--vary-albedo",
    "--frozen-atmosphere-positions", "anchor",
    "--retrieval-psf-fwhm-px", RETRIEVAL_PSF_FWHM_PX,
    "--prior-fields", "dense_frozen_exact_defocus", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", "1",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / f"co2palbedo-58-g1-ad4-pf-anchorfrozen-DEFOCUS_truth{truth_fwhm_tag}_retr{retr_fwhm_tag}_analytic.pkl"),
]

raise SystemExit(sweep.main())
