#!/usr/bin/env python3
"""Prior-information-driven albedo bin placement -- a synthetic demo.

The concrete follow-up to the conversation about a shared, information-
weighted state grid (docs/PROJECT_STATUS.md Sec.8): today `albedo`'s bins
are placed by `geocarb_gert.joint_state.albedo_positions_for`, a flat
`surface_density=3` uniform oversampling of the gas grid's own span --
three times as many bins as `co2_ppm`/`p_surface_hpa`, everywhere, whether
or not anything actually supports that much local resolution.

This demo swaps that flat rule for `information_weighted_bin_centers`
(joint_state.py), driven by `als.albedo_info_density` -- a SYNTHETIC stand-
in for "how much a real external product (MODIS albedo, ~500m) would tell
us here," peaked at each land-cover patch boundary, floor elsewhere. SAME
total bin budget as today's flat scheme, just redistributed toward where
the (synthetic) prior information actually is -- directly targeting the
12.8x/3.5x boundary-error concentration found in Sec.7.7-7.9.

Two parts:
  1. A bin-placement diagram (same convention as gd_window_overlap_diagram.py
     / gd_plume_bins_vs_anchors_diagram.py): today's flat placement vs. the
     new info-weighted one, for one window straddling a real patch boundary.
  2. A before/after accuracy comparison: solves that SAME window twice (flat
     vs. info-weighted bins, same total G, everything else identical),
     scores native-resolution error (along_slit_state.stack_windows_along_
     slit, no row-interpolation smearing -- Sec.7.4's own methodology) split
     near/far from the boundary, the same split Sec.7.7 used.

Deliberately self-contained rather than reusing `gd_joint_block_whole_slit_
sweep.py::_solve_window` -- that function has no extension point for a
custom surface-position rule and this demo isn't meant to add one to the
production CLI; it directly calls the same underlying primitives
(`state_spec_from_scene`, `build_forward_state`, `gauss_newton_state`,
`jac.make_spectrum_jac`) that function itself uses, at `--hires-only`
resolution only (matching the convention every other demo script in this
project uses).

RESULT (2026-08-25, --window-index 54, rows 884-924, G=40 albedo bins,
anchor_density=16): the redistribution mechanism itself works exactly as
intended -- the diagram shows real, dramatic clustering of bins right at
the boundary -- but the accuracy result is NOT the naive "concentrate
resolution where information exists -> improve accuracy there" outcome.
Native-resolution mean|error| near the boundary got WORSE (0.0122 ->
0.0185), while far-field improved (0.0011 -> 0.0003, fewer bins there now,
but each with more anchor support than before). Checked one obvious
confound: the tightly-packed near-boundary bins are NOT anchor-starved to
the same degree the ad1 pathology (Sec.7.6) was (min anchors-per-gap only
drops from 6 to 3, not to 0), so that alone doesn't explain it. Best
current hypothesis, not confirmed: packing bins closer together than
ALBEDO_CORR_KM (10km) makes neighbors strongly prior-correlated, which
isn't the same as the DATA independently resolving them, and may interact
badly with Gauss-Newton conditioning right where the extra resolution was
supposed to help -- the natural next diagnostic is comparing trace(AVK)
per bin between the two schemes (return_avk, Sec.7.10), not yet done. A
real, useful negative result: naively moving bins toward "where
information exists" without also confirming the DATA (anchor grid) can
support that redistribution can make things worse exactly where you
meant to help.

RESULT 2 (2026-08-25, --shared-grid, --window-index 38, rows 418-438, G=19,
anchor_density=16): extends the comparison with a THIRD scheme -- co2_ppm/
p_surface_hpa/albedo all sharing ONE combined_information_weighted_bin_
centers grid (driven by albedo's own boundary proxy, since no CO2 proxy
was built), with albedo alone set to state_interp="nearest" (co2/p_surface
stay "linear"). Same three-way pattern holds at this smaller window too --
flat: near 0.0031/far 0.0015; info-weighted (still separate grid, still
"linear"): near 0.0069/far 0.0013 (near got worse again, consistent with
RESULT above). Shared grid + nearest for albedo: near 0.0065/far 0.0064 --
overall WORSE than the flat baseline (0.0065 vs 0.0022), but the near/far
GAP essentially vanishes. That's a real, informative difference in
character, not just magnitude: "nearest" has no leakage between adjacent
bins at all (each anchor gets exactly its own bin's value, no blending
across the boundary), which removes the specific asymmetric leakage/blur
pattern Sec.7.9 documented for LINEAR interpolation -- but piecewise-
constant assignment is a cruder approximation everywhere else in the
window that isn't right at a hard edge, which is the likely source of the
uniformly-elevated (not just near-boundary) error here. Matches the
tradeoff named in conversation before building this: "nearest" trades
the linear leakage pattern for a different, more uniform but not smaller,
error source -- not a strict improvement, a different one.

Run:  PYTHONPATH=. python3 scripts/gd_information_density_bins_demo.py
        [--window-index 54] [--anchor-density 16] [--shared-grid]
Output: plots/joint_block/gd_information_density_bins_r<lo>-<hi>.png,
        plus a printed before/after accuracy table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, _eta_of, band_basics  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
from gd_joint_block_whole_slit_sweep import (build_window_tiles, _make_state_spectrum,  # noqa: E402
                                             PAD, ROW_MAX_IDX, _SIGMA_FLOOR)
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from geocarb_gert.joint_state import (build_forward_state, gauss_newton_state,  # noqa: E402
                                      state_spec_from_scene, albedo_positions_for,
                                      information_weighted_bin_centers,
                                      combined_information_weighted_bin_centers)
from geocarb_gert.along_slit_state import stack_windows_along_slit  # noqa: E402
from geocarb_gert import jacobians as jac  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402

COLOR_FLAT = "#3D5A80"
COLOR_INFO = "#E07A5F"


def _bin_placements(bin_centers, surface_density):
    """Both placement schemes, same total bin count, for one window."""
    flat = albedo_positions_for(bin_centers, surface_density)
    G = len(flat)
    info = information_weighted_bin_centers(bin_centers.min(), bin_centers.max(), G,
                                            lambda eta: als.albedo_info_density(eta * als.SLIT_HALF_KM))
    return flat, info


def _plot_bin_placement(fpa, row_lo, row_hi, flat, info, out_path):
    """Same visual convention as gd_window_overlap_diagram.py: real
    per-pixel eta scatter as background, bin centers as horizontal lines."""
    rows_win = np.arange(row_lo, row_hi + 1)
    cols = np.arange(1024.0)
    col_sub = np.arange(0, 1024, 6)
    eta_all = np.stack([_eta_of(fpa, cols, np.full(1024, float(r))) for r in rows_win])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    for ax, positions, color, title in [
        (axes[0], flat, COLOR_FLAT, f"today: flat surface_density (G={len(flat)})"),
        (axes[1], info, COLOR_INFO, f"info-weighted (same G={len(info)})"),
    ]:
        for i, r in enumerate(rows_win):
            ax.scatter(np.full(len(col_sub), r), eta_all[i, col_sub], s=3, color="0.6",
                      alpha=0.4, zorder=1)
        for p in positions:
            ax.axhline(p, color=color, lw=0.9, alpha=0.85, zorder=2)
        ax.set_xlabel("detector row")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel(r"pixel $\eta$")
    fig.suptitle(f"FPA{fpa}, rows {row_lo}-{row_hi}: albedo bin placement, flat vs. "
                f"info-weighted", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def _solve_one(fpa, row_lo, row_hi, band, absco, wide_inst, geo, solar, albedo,
               wn_hires, ils, anchor_density, bin_centers, surface_positions,
               row_state_interp=None):
    """Minimal hires-only solve for co2_ppm+p_surface_hpa+albedo, mirroring
    gd_joint_block_whole_slit_sweep.py::_solve_window's hires branch exactly,
    with two differences: surface_positions is supplied directly rather than
    computed internally by albedo_positions_for, and row_state_interp lets a
    caller set a per-row interpolation kind -- the two hooks this demo needs
    that the production function doesn't expose."""
    rows_win = np.arange(row_lo, row_hi + 1)
    y_true = band["A"][rows_win, :].ravel()
    noise_model = geocarb_noise_model(fpa)
    sigma = noise_model.sigma([y_true], [None])
    Sy_inv_diag = 1.0 / np.maximum(sigma, _SIGMA_FLOOR) ** 2

    band_label = GEOCARB_BANDS[fpa][0]
    spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)
    spectrum_jac = jac.make_spectrum_jac(absco, wide_inst, geo, solar, albedo)

    a_lo, a_hi = max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD)
    anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / anchor_density)
    anchor_etas = np.sort(_eta_of(fpa, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float)))

    spec = state_spec_from_scene(bin_centers, free=("co2_ppm", "p_surface_hpa", "albedo"),
                                 band_label=band_label, surface_positions=surface_positions,
                                 row_state_interp=row_state_interp)
    fwd = build_forward_state(fpa, rows_win, anchor_etas, spec, spectrum, wn_hires, ils,
                              pad=PAD, state_interp="linear")

    def lin(x):
        return jac.linearize(fpa, rows_win, anchor_etas, spec, spectrum_jac, wn_hires, ils, x,
                             pad=PAD, state_interp="linear")

    x, S_ret = gauss_newton_state(fwd, y_true, spec, Sy_inv_diag, verbose=False,
                                  jacobian_fn=lin, return_cov=True)
    resid = y_true - fwd(x)
    return dict(row_lo=row_lo, row_hi=row_hi,
               hires=spec.snapshot(x, resid=resid, resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                                   jacobian_used="analytic", cov=S_ret))


GD_GRATIO = 3.0  # matches retrieval_defaults.yml's own default; fixed here for a self-contained demo


def _native_error_summary(windows, name, truth_fn, boundary_km, tol_km=15.0):
    stack = stack_windows_along_slit(windows, name, solve="hires")
    x_km = stack.eta * als.SLIT_HALF_KM
    err = stack.values - truth_fn(x_km)
    near = np.abs(x_km - boundary_km) <= tol_km
    far = ~near
    return dict(n=len(x_km), mean_abs_err=float(np.mean(np.abs(err))),
               near_mean_abs_err=float(np.mean(np.abs(err[near]))) if near.any() else float("nan"),
               far_mean_abs_err=float(np.mean(np.abs(err[far]))) if far.any() else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-index", type=int, default=54,
                    help="index into build_window_tiles(FPA)'s own list -- default 54 (rows "
                         "884-924, straddling the desert/grass boundary at 1015km) has enough "
                         "albedo bins (G=40) for a reliable before/after comparison. A narrower "
                         "boundary-straddling window (e.g. --window-index 7, rows 63-71, only "
                         "G=7) is a real, useful negative example of the OPPOSITE problem this "
                         "whole feature is about: too few bins to say anything meaningful either "
                         "way -- confirmed directly against als.SURFACE_PATCHES, not guessed.")
    ap.add_argument("--anchor-density", type=int, default=16)
    ap.add_argument("--shared-grid", action="store_true",
                    help="2026-08-25 extension: instead of comparing flat-vs-info-weighted on "
                         "albedo's OWN separate grid, solve ONCE more with co2_ppm/p_surface_hpa/"
                         "albedo all sharing a single combined_information_weighted_bin_centers "
                         "grid (combining albedo's info-density with a flat CO2 floor, so the "
                         "grid shape is still driven by the boundary proxy), with albedo alone "
                         "set to state_interp='nearest' (co2/p_surface stay 'linear'). Compared "
                         "against the flat-scheme baseline already reported without this flag.")
    args = ap.parse_args()

    fpa = FPA
    print(f"Building realistic-scene FPA{fpa} band...", flush=True)
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
    row_lo, row_hi = tiles[args.window_index]
    print(f"window {args.window_index}: rows {row_lo}-{row_hi}")

    cols = np.arange(1024.0)
    rows_win = np.arange(row_lo, row_hi + 1)
    eta_all = np.stack([_eta_of(fpa, cols, np.full(1024, float(r))) for r in rows_win])
    G = max(2, round(len(rows_win) / GD_GRATIO))
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)

    flat_pos, info_pos = _bin_placements(bin_centers, surface_density=3)

    x_km_center = 0.5 * (bin_centers.min() + bin_centers.max()) * als.SLIT_HALF_KM
    boundary_km = min((p[0] for p in als.SURFACE_PATCHES[1:]), key=lambda b: abs(b - x_km_center))
    print(f"window center ~{x_km_center:.1f}km, nearest patch boundary at {boundary_km:.1f}km")

    plots_dir = REPO_ROOT / "plots" / "joint_block"
    _plot_bin_placement(fpa, row_lo, row_hi, flat_pos, info_pos,
                        plots_dir / f"gd_information_density_bins_r{row_lo}-{row_hi}.png")

    print("\nsolving (flat placement)...", flush=True)
    t0 = time.time()
    w_flat = _solve_one(fpa, row_lo, row_hi, band, absco, wide_inst, geo, solar, albedo,
                        band["wn_hires"], band["ils"], args.anchor_density, bin_centers, flat_pos)
    print(f"  done in {time.time()-t0:.0f}s")

    print("solving (info-weighted placement)...", flush=True)
    t0 = time.time()
    w_info = _solve_one(fpa, row_lo, row_hi, band, absco, wide_inst, geo, solar, albedo,
                        band["wn_hires"], band["ils"], args.anchor_density, bin_centers, info_pos)
    print(f"  done in {time.time()-t0:.0f}s")

    truth_fn = lambda x_km: als.albedo_for_label(x_km, GEOCARB_BANDS[fpa][0])  # noqa: E731
    s_flat = _native_error_summary([w_flat], "albedo", truth_fn, boundary_km)
    s_info = _native_error_summary([w_info], "albedo", truth_fn, boundary_km)

    results = [("flat (separate grid)", s_flat), ("info-weighted (separate grid)", s_info)]

    if args.shared_grid:
        print("\nsolving (shared grid, albedo=nearest)...", flush=True)
        G_shared = len(flat_pos)
        # co2_ppm contributes no real information-density proxy of its own here
        # (none built this pass) -- a flat weight at albedo_info_density's own
        # floor value, so the combined (max) grid is driven entirely by
        # albedo's boundary proxy, exactly like the separate-grid case above,
        # just now applied to ALL rows instead of albedo alone.
        shared_grid = combined_information_weighted_bin_centers(
            bin_centers.min(), bin_centers.max(), G_shared,
            {"albedo": lambda eta: als.albedo_info_density(eta * als.SLIT_HALF_KM),
             "co2_ppm": lambda eta: np.full_like(eta, 0.1)})
        t0 = time.time()
        w_shared = _solve_one(fpa, row_lo, row_hi, band, absco, wide_inst, geo, solar, albedo,
                              band["wn_hires"], band["ils"], args.anchor_density,
                              shared_grid, None, row_state_interp={"albedo": "nearest"})
        print(f"  done in {time.time()-t0:.0f}s")
        s_shared = _native_error_summary([w_shared], "albedo", truth_fn, boundary_km)
        results.append(("shared grid + nearest", s_shared))

    print("\n=== native-resolution albedo error ===")
    print(f"{'':30s} {'n_bins':>8s} {'overall':>10s} {'near (<=15km)':>15s} {'far (>15km)':>12s}")
    for label, s in results:
        print(f"{label:30s} {s['n']:8d} {s['mean_abs_err']:10.4f} "
             f"{s['near_mean_abs_err']:15.4f} {s['far_mean_abs_err']:12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
