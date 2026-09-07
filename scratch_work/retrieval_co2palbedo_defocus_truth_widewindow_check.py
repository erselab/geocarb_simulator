#!/usr/bin/env python3
"""One-off diagnostic (2026-09-06, user: "Yes, let's do that" -- rerun a
matched defocus config with a much wider min_window to check whether the
edge-clamp confound Sec.9's per-bin analysis found shrinks as expected).

Sec.9's "even MATCHED defocus gets worse with wider FWHM" finding turned
out to be confounded: `state_spec_from_scene`'s free parameters
(co2_ppm/p_surface_hpa/albedo) live only on a window's own local
`bin_centers`, but `predict_neighborhood`'s PSF blur needs a padded render
region beyond the window (`pad`, auto-scaled via `default_pad_for_psf` to
4/8/14/22 rows at FWHM 1.5/3/5/8px) -- `_row_interp1d`'s `fill_value=(lo,
hi)` CLAMPS the free state to the nearest window-edge value out there
rather than extrapolating any real gradient, so wider PSF -> wider pad ->
more of a window's edge-row signal drawn from a badly-represented region.

This script reruns ONE much-wider window (`--min-window 20` instead of the
standard `--n-windows 58` tiling -- rows 205-245, width 41, vs. the
original 11-row window at the same slit location) at both nominal
(1.5px) and defocused (FWHM=8px) matched PSF. If the edge-clamp
mechanism is the dominant driver, pad/width ratio drops from ~200%
(22/11) to ~54% (22/41) and the FWHM=8 vs nominal degradation should be
much smaller in relative terms than it was for the narrow window
(narrow: rms 4.92 at 8px matched vs. Sec.6's own nominal baseline for a
DIFFERENT, bin-frozen-representability-confounded config -- not directly
comparable; the real within-this-script comparison is nominal-wide vs.
defocused-wide, both run under the IDENTICAL wide tiling and free/frozen
config, so the only thing that differs is truth/retrieval PSF).

Reads TRUTH_PSF_FWHM_PX / RETRIEVAL_PSF_FWHM_PX (same convention as
retrieval_co2palbedo_defocus_truth_anchorfrozen.py) plus MIN_WINDOW
(the wider window radius) -- task_id/n_tasks come from SLURM_ARRAY_*
as usual, but now index into the min_window's OWN tiling, not the
standard 58-window one, so the caller must independently confirm which
task_id maps to the target row range under this min_window (done via
`build_window_tiles(fpa, min_window=<MW>, window_scale=1.0)` at
submission time, not derived automatically here).
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
MIN_WINDOW = os.environ["MIN_WINDOW"]
print(f"TRUTH_PSF_FWHM_PX={TRUTH_PSF_FWHM_PX}  RETRIEVAL_PSF_FWHM_PX={RETRIEVAL_PSF_FWHM_PX}  "
     f"MIN_WINDOW={MIN_WINDOW}", flush=True)

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
PRIOR_NAME = f"dense_frozen_exact_defocus_truth{truth_fwhm_tag}_widecheck"
als.PRIOR_FIELD_SETS[PRIOR_NAME] = _merged_fields
als.SURFACE_PRIOR_FIELD_SETS[PRIOR_NAME] = als.SURFACE_FIELDS_PRIOR

print(f"{PRIOR_NAME} atmosphere prior rows:", flush=True)
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
    "--prior-fields", PRIOR_NAME, "--min-window", MIN_WINDOW, "--anchor-density", "4",
    "--resolution-matched-g-ratio-bins", "1",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / f"co2palbedo-widecheck-mw{MIN_WINDOW}-g1-ad4-pf-anchorfrozen-DEFOCUS_truth{truth_fwhm_tag}_retr{retr_fwhm_tag}_analytic.pkl"),
]

raise SystemExit(sweep.main())
