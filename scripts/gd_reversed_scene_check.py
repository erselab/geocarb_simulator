#!/usr/bin/env python3
"""Just-for-fun experiment (2026-08-19, user's own idea): mirror the
realistic scene's atm/surface composition along the slit (x_km -> -x_km)
while leaving the REAL, asymmetric keystone/smile geometry untouched, then
render the same bingrid_g<N>_linear_ad<d> config against the mirrored truth.
Composition only (STATE_FIELDS: co2_ppm/ch4_ppb/co_ppb/h2o_surface_vmr/
p_surface_hpa) -- albedo/reflectance is NOT mirrored (it isn't part of
STATE_FIELDS at all -- see along_slit_scene.py's own note on why).

Since the distortion geometry is fixed but the scene's own feature
positions are now mirrored, a hot spot that used to sit near a low-keystone
row now sits near a high-keystone row (and vice versa) -- this isolates
whether/how residual structure comes from WHERE a real feature falls
relative to the (asymmetric) distortion pattern, not just how sharp the
feature itself is. Directly comparable against the un-reversed
bingrid_g<N>_linear_ad<d> config already in
results/realistic_prior/forward_check/, same g_ratio/anchor_density.

Run:  PYTHONPATH=. python3 scripts/gd_reversed_scene_check.py \\
        --g-ratio 3 --anchor-density 16
Output: results/realistic_prior/reversed_scene_check/*.npy
        plots/realistic_prior/forward_check_reversed_scene_fpa<N>.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import GERT_ROOT, band_basics  # noqa: E402
import gd_joint_block_whole_slit_sweep as sw  # noqa: E402
import gd_realistic_prior_forward_check as fc  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402

N_COLS = fc.N_COLS


def _mirror_fn(fn):
    """Wrap a STATE_FIELDS field function so it reads the real function's
    value at the MIRRORED along-slit position -- x_km -> -x_km. Every field
    is wrapped the same way, so the whole scene (not just one quantity)
    reflects about the slit centre."""
    return lambda x: fn(-np.asarray(x, dtype=float))


FIELDS_REVERSED = {name: _mirror_fn(fn) for name, fn in als.STATE_FIELDS.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpa", type=int, default=2)
    ap.add_argument("--g-ratio", type=float, default=sw.G_RATIO)
    ap.add_argument("--anchor-density", type=float, default=16.0)
    ap.add_argument("--n-lookup-samples", type=int, default=5600)
    ap.add_argument("--out-root", type=str,
                    default=str(REPO_ROOT / "results" / "realistic_prior" / "reversed_scene_check"))
    ap.add_argument("--plot-dir", type=str,
                    default=str(REPO_ROOT / "plots" / "realistic_prior"))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[args.fpa]

    # -- mirrored truth: _lookup_sample (build_lookup_radiance's own
    # per-sample worker, inside along_slit_scene.py) calls als.atmosphere_at
    # unqualified and isn't parameterized by `fields=` the way
    # render_prior_image is -- so the only way to mirror the DENSE truth
    # itself is to swap the module-level function out for the duration of
    # this one call, then put it back.
    real_atmosphere_at = als.atmosphere_at

    def _reversed_atmosphere_at(x_km, h2o_scale_height_km=2.0):
        return real_atmosphere_at(-np.asarray(x_km, dtype=float), h2o_scale_height_km)

    als.atmosphere_at = _reversed_atmosphere_at
    print(f"Building MIRRORED-composition FPA{args.fpa} truth "
         f"(n_lookup_samples={args.n_lookup_samples})...", flush=True)
    t0 = time.time()
    try:
        band_rev = gdt._band_setup(args.fpa, atm_center, absco, geo, solar, snr,
                                   args.n_lookup_samples, None, False, False, 32, False, 0, False)
    finally:
        als.atmosphere_at = real_atmosphere_at
    truth_rev = np.asarray(band_rev["A"], dtype=float)
    print(f"mirrored truth done in {time.time() - t0:.0f}s -- A {truth_rev.shape}", flush=True)
    np.save(out_root / f"truth_reversed_fpa{args.fpa}_nls{args.n_lookup_samples}.npy", truth_rev)

    wide_win, wide_inst, albedo = band_basics(args.fpa, atm_center, absco, geo, solar)
    tiles = sw.build_window_tiles(args.fpa)
    label = f"bingrid_g{args.g_ratio:g}_linear_ad{args.anchor_density:g}_reversed"
    print(f"rendering '{label}' (mirrored composition, real geometry, "
         f"g_ratio={args.g_ratio:g}, anchor_density={args.anchor_density:g})...", flush=True)
    A_rev = fc.render_prior_image(args.fpa, band_rev, absco, wide_inst, geo, solar, albedo,
                                  tiles, args.g_ratio, FIELDS_REVERSED, None, label,
                                  native_res=False, rt_density=args.anchor_density,
                                  state_interp="linear")
    resid_rev = truth_rev - A_rev
    np.save(out_root / f"prior_{label}_fpa{args.fpa}.npy", A_rev)
    np.save(out_root / f"resid_{label}_fpa{args.fpa}.npy", resid_rev)
    rms_row_rev = np.sqrt(np.nanmean(resid_rev ** 2, axis=1))
    pct_row_rev = rms_row_rev / np.nanmean(np.abs(truth_rev), axis=1) * 100.0
    print(f"  '{label}': residRMS/mean|A| median {np.nanmedian(pct_row_rev):.3f}%, "
         f"max {np.nanmax(pct_row_rev):.3f}%", flush=True)

    # -- matching UN-reversed config, already on disk from the main
    # forward-check sweep -- same g_ratio/anchor_density, real (unmirrored)
    # scene, for direct comparison.
    fwd_root = REPO_ROOT / "results" / "realistic_prior" / "forward_check"
    orig_label = f"bingrid_g{args.g_ratio:g}_linear_ad{args.anchor_density:g}"
    orig_resid_path = fwd_root / f"resid_{orig_label}_fpa{args.fpa}.npy"
    orig_truth_paths = sorted(fwd_root.glob(f"truth_fpa{args.fpa}_nls*.npy"))
    if not orig_resid_path.exists() or not orig_truth_paths:
        print(f"no matching un-reversed '{orig_label}' found in {fwd_root} -- "
             f"skipping comparison plot (mirrored-only .npy still saved above)")
        return 0
    truth_orig = np.load(orig_truth_paths[-1])
    resid_orig = np.load(orig_resid_path)
    rms_row_orig = np.sqrt(np.nanmean(resid_orig ** 2, axis=1))
    pct_row_orig = rms_row_orig / np.nanmean(np.abs(truth_orig), axis=1) * 100.0

    _plot(truth_orig, resid_orig, pct_row_orig, truth_rev, resid_rev, pct_row_rev,
         orig_label, args.fpa, plot_dir)
    return 0


def _plot(truth_orig, resid_orig, pct_row_orig, truth_rev, resid_rev, pct_row_rev,
         orig_label, fpa, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    cont_orig = np.where(np.abs(truth_orig) > 0, np.abs(truth_orig), np.nan)
    cont_rev = np.where(np.abs(truth_rev) > 0, np.abs(truth_rev), np.nan)
    pct_orig = resid_orig / cont_orig * 100.0
    pct_rev = resid_rev / cont_rev * 100.0

    shared_max = max(float(np.nanmax(np.abs(pct_orig))), float(np.nanmax(np.abs(pct_rev)))) or 1.0
    shared_lin = min(
        float(np.nanmedian(np.abs(pct_orig[np.isfinite(pct_orig)]))) or shared_max * 1e-3,
        float(np.nanmedian(np.abs(pct_rev[np.isfinite(pct_rev)]))) or shared_max * 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, pct, title in ((axes[0], pct_orig, f"{orig_label} (real scene)"),
                           (axes[1], pct_rev, f"{orig_label}_reversed (mirrored composition, "
                                              "real geometry)")):
        im = ax.imshow(pct, aspect="auto", cmap="RdBu_r", origin="upper",
                       norm=SymLogNorm(linthresh=shared_lin, vmin=-shared_max,
                                      vmax=shared_max, base=10))
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("detector column")
    axes[0].set_ylabel("detector row")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="residual / |truth| (%)")
    fig.suptitle(f"FPA{fpa}: same distortion geometry, mirrored scene composition -- "
                f"same colour scale", fontsize=12)
    out2d = plot_dir / f"forward_check_reversed_scene_2d_fpa{fpa}.png"
    fig.savefig(out2d, dpi=150)
    print(f"saved {out2d}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    rows = np.arange(truth_orig.shape[0])
    ax.plot(rows, pct_row_orig, lw=1.3, label=f"{orig_label} (real scene)", color="C0")
    ax.plot(rows, pct_row_rev, lw=1.3, label=f"{orig_label}_reversed (mirrored composition)",
           color="C3")
    ax.plot(1023 - rows, pct_row_orig, lw=1.0, ls=":", color="0.4",
           label=f"{orig_label} (real scene), ROW-MIRRORED for reference")
    ax.set_xlabel("detector row")
    ax.set_ylabel("residRMS / mean|truth| (%)")
    ax.set_yscale("log")
    ax.set_title(f"FPA{fpa}: does mirroring the scene's composition also mirror the "
                f"residual pattern? (it would, if the distortion geometry were symmetric)",
                fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_row = plot_dir / f"forward_check_reversed_scene_by_row_fpa{fpa}.png"
    fig.savefig(out_row, dpi=150)
    print(f"saved {out_row}")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
