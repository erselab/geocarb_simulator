#!/usr/bin/env python3
"""Per-bin diagnostic for docs/PROJECT_STATUS.md Sec.10's modulation result.

Two competing explanations were on the table for why sub_bin_modulation
(g=truth) makes native-resolution albedo error WORSE, not better, even
with a structural (no-fine-texture) prior and bins ~11x wider than the
texture's own correlation length (500m):

  (A) Error-AMPLIFICATION: the modulated Jacobian column carries a factor
      g(eta)/mean_g_k that a plain linear column doesn't, so any residual
      retrieval error gets amplified by however much g(eta) deviates from
      its own bin's footprint mean.

  (B) A SCORING artifact: `_native_error_summary` (gd_information_density_
      bins_demo.py) scores the raw retrieved bin_value_k -- via `ParamSpec.
      apply()`, unaffected by sub_bin_modulation -- against truth sampled
      at a single POINT (the bin's own center), never against the actual
      modulated curve or a footprint-consistent target. Under modulation
      the self-consistent target for bin_value_k is `mean_g_k`, a
      FOOTPRINT AVERAGE of the true texture -- which can differ from one
      point sample inside that footprint even for a perfect fit, once the
      texture's correlation length (500m) is much shorter than the bin
      (~5.7km here).

This script solves the SAME window/settings as that result (structural
prior, ALBEDO_CORR_KM=0.5km, g_ratio=1, --window-index 38) once with no
modulation and once with sub_bin_modulation(g=truth), then compares, per
bin:
  - bin_value_k (both solves)
  - mean_g_k (footprint average of truth over the bin's own footprint --
    for g=truth this IS the footprint-averaged truth, no separate
    computation needed)
  - point_truth_k (truth at the bin's own center -- what the existing
    scoring actually compares against)

and reports both hypotheses' own diagnostic:
  (A) correlation between the modulation-INDUCED extra error
      (delta_mod_k - delta_nomod_k, both vs point_truth_k) and the
      amplification proxy |g_range_k| (local RMS deviation of g within
      the bin's own footprint, from _albedo_fine_field directly)
  (B) correlation between that same induced error and the point-vs-
      footprint-mean gap (mean_g_k - point_truth_k) -- AND whether
      bin_value_mod_k tracks mean_g_k more tightly than it tracks
      point_truth_k (the direct check that modulation is doing what the
      self-consistency identity says it should).

Run:  PYTHONPATH=. python3 scripts/gd_modulation_diagnostic.py
"""
from __future__ import annotations

import functools
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, _eta_of, band_basics  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
from gd_joint_block_whole_slit_sweep import build_window_tiles  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from geocarb_gert.joint_state import (combined_information_weighted_bin_centers,  # noqa: E402
                                      bin_footprint_means)
import gd_information_density_bins_demo as demo  # noqa: E402

WINDOW_INDEX = 38
G_RATIO = 1.0
ANCHOR_DENSITY = 16


def _oracle_g(eta, fpa):
    return als.albedo_for_label(eta * als.SLIT_HALF_KM, GEOCARB_BANDS[fpa][0], include_fine=True)


def main() -> int:
    fpa = FPA
    print("Building scene...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
    band = gdt._band_setup_cached(fpa, atm_center, absco, geo, solar, snr, 400, None,
                                  False, False, None, False, 0, False, vary_albedo=True)
    wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    tiles = build_window_tiles(fpa, min_window=4, window_scale=1.0)
    row_lo, row_hi = tiles[WINDOW_INDEX]
    print(f"window {WINDOW_INDEX}: rows {row_lo}-{row_hi}")

    cols = np.arange(1024.0)
    rows_win = np.arange(row_lo, row_hi + 1)
    eta_all = np.stack([_eta_of(fpa, cols, np.full(1024, float(r))) for r in rows_win])
    G = max(2, round(len(rows_win) / G_RATIO))
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    print(f"G={G} bins")

    shared_grid = combined_information_weighted_bin_centers(
        bin_centers.min(), bin_centers.max(), G,
        {"albedo": lambda eta: als.albedo_info_density(eta * als.SLIT_HALF_KM),
         "co2_ppm": lambda eta: np.full_like(eta, 0.1)})

    surface_fields = als.SURFACE_FIELDS_PRIOR  # structural: no fine texture in the prior

    demo._CTX.update(dict(fpa=fpa, row_lo=row_lo, row_hi=row_hi, band=band, absco=absco,
                          wide_inst=wide_inst, geo=geo, solar=solar, albedo=albedo,
                          anchor_density=ANCHOR_DENSITY, surface_fields=surface_fields,
                          return_avk=True))
    jobs = {
        "nomod": dict(bin_centers=shared_grid),
        "mod_truth": dict(bin_centers=shared_grid,
                          row_sub_bin_modulation={"albedo": {
                              "g_fn": functools.partial(_oracle_g, fpa=fpa)}}),
    }
    names = list(jobs.keys())
    print(f"solving {names} in parallel...", flush=True)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=2) as pool:
        outs = pool.map(demo._solve_worker, [jobs[n] for n in names])
    print(f"  done in {time.time()-t0:.0f}s")
    w = dict(zip(names, outs))

    # -- per-bin arrays --------------------------------------------------
    rec_nomod = w["nomod"]["hires"]["params"]["albedo"]
    rec_mod = w["mod_truth"]["hires"]["params"]["albedo"]
    positions = np.asarray(rec_nomod["positions"], dtype=float)   # shared_grid, eta units
    assert np.array_equal(positions, np.asarray(rec_mod["positions"], dtype=float))
    bin_value_nomod = np.asarray(rec_nomod["values"], dtype=float)
    bin_value_mod = np.asarray(rec_mod["values"], dtype=float)

    band_label = GEOCARB_BANDS[fpa][0]
    x_km = positions * als.SLIT_HALF_KM
    point_truth_k = als.albedo_for_label(x_km, band_label, include_fine=True)

    # mean_g_k for g=truth IS the footprint-averaged truth -- same helper
    # modulation itself uses for its own normalization, called here with
    # the identical g_fn so this is not a reimplementation.
    g_fn = functools.partial(_oracle_g, fpa=fpa)
    sbm = bin_footprint_means(positions, g_fn, state_interp="linear")
    mean_g_k = sbm.mean_g

    # amplification proxy: local RMS deviation of g within each bin's own
    # footprint (same fine grid bin_footprint_means used internally, but
    # we only need the fine albedo field's own local std here, not the
    # full interp_weights machinery) -- approximate via the true fine
    # field's std over each bin's [prev_center, next_center] span.
    xf, f = als._albedo_fine_field()
    edges = np.concatenate([[positions[0] - (positions[1] - positions[0])],
                            0.5 * (positions[:-1] + positions[1:]),
                            [positions[-1] + (positions[-1] - positions[-2])]])
    edges_km = edges * als.SLIT_HALF_KM
    g_local_std_k = np.zeros_like(positions)
    for k in range(len(positions)):
        mask = (xf >= edges_km[k]) & (xf < edges_km[k + 1])
        g_local_std_k[k] = f[mask].std() if mask.sum() > 1 else 0.0

    delta_nomod = bin_value_nomod - point_truth_k
    delta_mod = bin_value_mod - point_truth_k
    delta_induced = delta_mod - delta_nomod
    artifact_gap = mean_g_k - point_truth_k     # point-vs-footprint-mean gap

    def corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n=== per-bin summary ===")
    print(f"n_bins: {len(positions)}")
    print(f"mean |delta_nomod|: {np.mean(np.abs(delta_nomod)):.4f}")
    print(f"mean |delta_mod|:   {np.mean(np.abs(delta_mod)):.4f}")
    print(f"mean |delta_induced| (extra error from modulation): {np.mean(np.abs(delta_induced)):.4f}")
    print(f"mean |artifact_gap| (mean_g_k - point_truth_k):     {np.mean(np.abs(artifact_gap)):.4f}")
    print()
    print("--- Hypothesis (A): amplification ---")
    print(f"corr(|delta_induced|, g_local_std_k):        {corr(np.abs(delta_induced), g_local_std_k):.3f}")
    print()
    print("--- Hypothesis (B): scoring artifact ---")
    print(f"corr(delta_induced, artifact_gap):            {corr(delta_induced, artifact_gap):.3f}")
    print(f"corr(|delta_induced|, |artifact_gap|):         {corr(np.abs(delta_induced), np.abs(artifact_gap)):.3f}")
    print()
    print("--- does bin_value_mod track mean_g_k (self-consistency check)? ---")
    resid_vs_meang = bin_value_mod - mean_g_k
    print(f"mean |bin_value_mod - mean_g_k|:        {np.mean(np.abs(resid_vs_meang)):.4f}")
    print(f"mean |bin_value_mod - point_truth_k|:   {np.mean(np.abs(delta_mod)):.4f}")
    print(f"  (if modulation is working as designed, the first number should be")
    print(f"   noticeably smaller than the second -- bin_value_mod chasing the")
    print(f"   footprint mean, not the point value)")
    print()
    print("--- direct comparison: scoring against footprint mean instead of point ---")
    delta_mod_vs_footprint = bin_value_mod - mean_g_k
    delta_nomod_vs_footprint = bin_value_nomod - mean_g_k
    print(f"mean |nomod - point_truth|:      {np.mean(np.abs(delta_nomod)):.4f}")
    print(f"mean |nomod - footprint_mean|:   {np.mean(np.abs(delta_nomod_vs_footprint)):.4f}")
    print(f"mean |mod   - point_truth|:      {np.mean(np.abs(delta_mod)):.4f}")
    print(f"mean |mod   - footprint_mean|:   {np.mean(np.abs(delta_mod_vs_footprint)):.4f}")

    # -- averaging kernel: does modulation genuinely degrade conditioning? -
    print("\n=== averaging kernel (degrees-of-freedom-for-signal) ===")
    for label, key in [("nomod", "nomod"), ("mod_truth", "mod_truth")]:
        rec = w[key]["hires"]
        avk = np.asarray(rec["avk"], dtype=float)
        sl = rec["slices"]["albedo"]
        avk_albedo = avk[sl[0]:sl[1], sl[0]:sl[1]]
        dof_total = float(np.trace(avk))
        dof_albedo = float(np.trace(avk_albedo))
        diag_albedo = np.diag(avk_albedo)
        print(f"{label:12s} trace(AVK) whole solve: {dof_total:6.2f} / {avk.shape[0]} free  "
             f"| albedo-only: {dof_albedo:6.2f} / {avk_albedo.shape[0]} bins  "
             f"| mean diag(AVK)_albedo: {diag_albedo.mean():.4f}  "
             f"min: {diag_albedo.min():.4f}  max: {diag_albedo.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
