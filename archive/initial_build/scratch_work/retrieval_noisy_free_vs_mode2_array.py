#!/usr/bin/env python3
"""Isolate whether the pressure/CO2/H2O/albedo degeneracy itself (not a
confound) drives the window-boundary oscillations (user, 2026-09-02):
FROZEN rows (ch4_ppb, co_ppb, h2o_surface_vmr) pinned to EXACT truth --
zero confound from a wrong background composition, unlike the earlier
"structural" run, which pinned them to the wrong (structural-prior)
values. FREE rows (co2_ppm, p_surface_hpa, albedo) start from truth PLUS
independent, reproducible per-bin Gaussian noise (not "structural"'s own
systematic large-scale bias shape) at each row's own existing prior
sigma (10%/2%/20% -- the same numbers the retrieval's own Sa already
assumes, so the injected uncertainty is self-consistent with what the
solver believes going in).

Truth = Mode 2 "representative" (zero representability gap, exactly
reproducible by the retrieval's own forward code) -- the clean baseline
with no model-mismatch confound either, so any remaining boundary
wobble is attributable to the free-row degeneracy alone.

Windows distributed via --task-id/--n-tasks (one per window, following
the earlier per-window-job finding), each with real anchor-level
parallelism (--anchor-workers).
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
from geocarb_gert import along_slit_scene as als  # noqa: E402

with open(REPO_ROOT / "scratch_work" / "whole_slit_truth_v1.pkl", "rb") as f:
    truth = pickle.load(f)
A_TRUTH = truth["representative"]

# -- build the mixed "frozen exact, free noisy" field set -------------
SIGMA_FRAC = {"co2_ppm": 0.10, "p_surface_hpa": 0.02, "albedo": 0.20}
SEED_SALT = {"co2_ppm": 101, "p_surface_hpa": 202, "albedo": 303}


def _make_noisy(true_fn, sigma_frac, seed_salt):
    def noisy(x_km):
        x_km = np.asarray(x_km, dtype=float)
        true_val = np.asarray(true_fn(x_km), dtype=float)
        seed = (seed_salt + (hash(x_km.tobytes()) % 1_000_000_007)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return true_val * (1.0 + sigma_frac * rng.standard_normal(x_km.size))
    return noisy


def _make_noisy_surface(true_fn, sigma_frac, seed_salt):
    def noisy(x_km, label):
        x_km = np.asarray(x_km, dtype=float)
        true_val = np.asarray(true_fn(x_km, label), dtype=float)
        seed = (seed_salt + (hash(x_km.tobytes()) % 1_000_000_007)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return true_val * (1.0 + sigma_frac * rng.standard_normal(x_km.size))
    return noisy


noisy_fields = dict(als.STATE_FIELDS)  # frozen rows: exact truth, unmodified
for _name in ["co2_ppm", "p_surface_hpa"]:
    noisy_fields[_name] = _make_noisy(als.STATE_FIELDS[_name], SIGMA_FRAC[_name], SEED_SALT[_name])
als.PRIOR_FIELD_SETS["noisy_free_exact_frozen"] = noisy_fields

noisy_surface_fields = {
    "albedo": _make_noisy_surface(als.SURFACE_FIELDS["albedo"], SIGMA_FRAC["albedo"], SEED_SALT["albedo"])
}
als.SURFACE_PRIOR_FIELD_SETS["noisy_free_exact_frozen"] = noisy_surface_fields


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
anchor_workers = int(os.environ.get("ANCHOR_WORKERS", "16"))

# Deliberately NO --resolution-matched-g-ratio-bins: prior_fields
# "noisy_free_exact_frozen" is looked up straight from
# als.PRIOR_FIELD_SETS (the non-resolution-matched code path), which is
# fine here -- resolution_matched_fields and raw STATE_FIELDS agree
# EXACTLY at each bin's own center (both equal the true value there;
# resolution_matched_fields is defined to interpolate exactly through
# its own anchor nodes), so evaluating the frozen rows' exact-truth
# functions at bin centers gives byte-identical numbers to what the
# Mode-2 truth render itself used at those same points.
sys.argv = [
    "gd_joint_block_whole_slit_sweep.py",
    "--g-ratio", "1", "--free", "co2_ppm,p_surface_hpa,albedo", "--vary-albedo",
    "--prior-fields", "noisy_free_exact_frozen", "--n-windows", "58", "--anchor-density", "16",
    "--hires-only", "--jacobian", "analytic", "--prior-form", "exponential",
    "--overlap", "0", "--n-workers", "1", "--anchor-workers", str(anchor_workers),
    "--task-id", str(task_id), "--n-tasks", str(n_tasks),
    "--out", str(RESDIR / "co2p_albedo-58-g1-ad16-pf-noisyfree-MODE2TRUTH_analytic.pkl"),
]

raise SystemExit(sweep.main())
