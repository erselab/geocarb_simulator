#!/usr/bin/env python3
"""Test the fine-resolution state / PSF-as-ISRF strategy discussed after
the first joint-block diagnostics (docs/JOINT_BIN_RETRIEVAL_ATBD.html
&sect;8.1's quantization finding): instead of the retrieval's own G bin
values being assigned to real pixels by NEAREST-bin lookup (a step
function in eta), linearly interpolate the G-dim retrieval state to a
much finer set of anchors -- one per real detector row -- and evaluate
RT at that resolution before the same spatial PSF blur. This is the
along-slit mirror of what `_diagonal_ils_convolve` already does in the
spectral direction: evaluate on a fine grid, then convolve with the
relevant instrument response.

Both "coarse" (G=15 bins, nearest-bin assignment, already validated) and
"hi-res" (G_eff anchors at native row resolution, linearly interpolated
from the SAME G=15 free parameters, re-solved independently) are run as
full, independent regularized Gauss-Newton solves on the identical row
890-935 FPA2 scene, so the comparison isolates the forward-model
resolution question, not a different truth or a different state
dimensionality -- the retrieval still only has G=15 free parameters in
both cases; what changes is how those G values get mapped onto real
pixels before RT.

Safety property preserved exactly: linear interpolation happens in
STATE space (co2_scale, before RT), never in radiance space -- the
hi-res forward model reuses `nearest_bin_scene` unchanged, just at
G_eff-anchor resolution instead of G-bin resolution, so every predicted
pixel still traces back to exactly one anchor's own independent RT
output, never a blend of two anchors' spectra.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_hires_test.py
Output: plots/gd_joint_block_hires_test_fpa2_row890-935_G15.png
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
import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import (  # noqa: E402
    FPA, GERT_ROOT, _eta_of, band_basics, build_forward,
    gauss_newton_regularized, make_spectrum_fn,
)
from gd_joint_block_diagnostics import (  # noqa: E402
    bin_assign, per_bin_chi2, pixel_density_bin_centers,
)

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, nearest_bin_scene, sample_geometries  # noqa: E402
from geocarb_gert import gd_render  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed  # noqa: E402


def build_forward_hires(fpa, rows_win, anchor_rows, bin_centers, spectrum_for, wn_hires, ils, pad=4):
    """G_eff anchors, one per real detector row in the padded window
    (native row resolution). Each anchor's co2_scale is linearly
    interpolated from the G-dim retrieval state x at bin_centers --
    state-space only, never radiance-space. Reuses nearest_bin_scene
    unchanged at G_eff resolution instead of G, so within-row keystone
    splicing is resolved at native row granularity."""
    anchor_etas = _eta_of(fpa, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float))
    order = np.argsort(anchor_etas)
    anchor_etas_sorted = anchor_etas[order]
    anchor_atms = [als.atmosphere_at(float(e * als.SLIT_HALF_KM)) for e in anchor_etas_sorted]
    G_eff = len(anchor_etas_sorted)

    cache_co2 = np.full(G_eff, np.nan)
    cache_S = [None] * G_eff

    def forward(x):
        x = np.asarray(x, dtype=float)
        co2_anchors = np.interp(anchor_etas_sorted, bin_centers, x)
        for g in range(G_eff):
            if cache_S[g] is None or co2_anchors[g] != cache_co2[g]:
                cache_S[g] = spectrum_for(co2_anchors[g], anchor_atms[g])
                cache_co2[g] = co2_anchors[g]
        radiance = nearest_bin_scene(anchor_etas_sorted, cache_S)
        A = gd_render.predict_neighborhood(fpa, rows_win, wn_hires, radiance, ils, pad=pad)
        return A.ravel()

    return forward, anchor_etas_sorted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-min", type=int, default=890)
    ap.add_argument("--row-max", type=int, default=935)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--sigma-abs", type=float, default=0.10)
    ap.add_argument("--peak-row", type=int, default=910)
    ap.add_argument("--bg-row", type=int, default=880)
    args = ap.parse_args()

    print(f"Building realistic-scene FPA{FPA} band...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400, None,
                           False, False, 32, False, 0)
    print("done.\n", flush=True)

    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    wn_hires, ils = band["wn_hires"], band["ils"]

    rows_win = np.arange(args.row_min, args.row_max + 1)
    anchor_rows = np.arange(args.row_min - args.pad, args.row_max + args.pad + 1)
    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])

    bin_centers = pixel_density_bin_centers(eta_all.ravel(), args.n_bins)

    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

    true_peak = float(band["xtrue_of_row"]["co2"][args.peak_row])
    true_bg = float(band["xtrue_of_row"]["co2"][args.bg_row])

    def bin_at_row(bc, row):
        eta_c = float(_eta_of(FPA, np.array([512.0]), np.array([float(row)]))[0])
        return int(np.argmin(np.abs(bc - eta_c)))

    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)

    results = {}

    # ---- coarse: G=15, nearest-bin (already-validated pixel-density scheme) ----
    print(f"\n=== coarse: G={args.n_bins}, nearest-bin ===", flush=True)
    x_km_bins = bin_centers * als.SLIT_HALF_KM
    prior_atms = [als.atmosphere_at(float(xk)) for xk in x_km_bins]
    prior_co2_ppm_bins = np.array([float(als.xco2_ppm(xk)) for xk in x_km_bins])
    forward_coarse = build_forward(FPA, rows_win, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=args.pad)
    t0 = time.time()
    x_coarse = gauss_newton_regularized(forward_coarse, y_true, x0=np.ones(args.n_bins),
                                        Sy_inv_diag=Sy_inv_diag, gamma=args.gamma,
                                        sigma_abs=args.sigma_abs, label="coarse")
    elapsed_coarse = time.time() - t0
    resid_coarse = y_true - forward_coarse(x_coarse)
    resid_img_coarse = resid_coarse.reshape(len(rows_win), 1024)
    retrieved_ppm_coarse = prior_co2_ppm_bins * x_coarse
    gpk, gbg = bin_at_row(bin_centers, args.peak_row), bin_at_row(bin_centers, args.bg_row)
    capture_coarse = (retrieved_ppm_coarse[gpk] - retrieved_ppm_coarse[gbg]) / (true_peak - true_bg)
    bin_idx_coarse = bin_assign(bin_centers, eta_all.ravel())
    chi2_coarse, counts_coarse = per_bin_chi2(resid_coarse, Sy_inv_diag, bin_idx_coarse, args.n_bins)
    print(f"  ({elapsed_coarse:.0f}s) capture={capture_coarse*100:.1f}%  "
         f"mean chi2={chi2_coarse.mean():.4g}  min count={counts_coarse.min()}", flush=True)

    # ---- hi-res: G_eff anchors at native row resolution, interpolated from the SAME G=15 state ----
    print(f"\n=== hi-res: G_eff={len(anchor_rows)} anchors (native row), interpolated from G={args.n_bins} ===", flush=True)
    forward_hires, anchor_etas = build_forward_hires(FPA, rows_win, anchor_rows, bin_centers,
                                                      spectrum_for, wn_hires, ils, pad=args.pad)
    t0 = time.time()
    x_hires = gauss_newton_regularized(forward_hires, y_true, x0=np.ones(args.n_bins),
                                       Sy_inv_diag=Sy_inv_diag, gamma=args.gamma,
                                       sigma_abs=args.sigma_abs, label="hi-res")
    elapsed_hires = time.time() - t0
    resid_hires = y_true - forward_hires(x_hires)
    resid_img_hires = resid_hires.reshape(len(rows_win), 1024)
    # score capture fraction using the SAME retrieval-parameter bin centers/co2 ppm scale
    retrieved_ppm_hires = prior_co2_ppm_bins * x_hires
    capture_hires = (retrieved_ppm_hires[gpk] - retrieved_ppm_hires[gbg]) / (true_peak - true_bg)
    chi2_hires, counts_hires = per_bin_chi2(resid_hires, Sy_inv_diag, bin_idx_coarse, args.n_bins)
    print(f"  ({elapsed_hires:.0f}s) capture={capture_hires*100:.1f}%  "
         f"mean chi2={chi2_hires.mean():.4g}  min count={counts_hires.min()}", flush=True)

    print(f"\nstate comparison (co2_scale per retrieval bin):")
    for g in range(args.n_bins):
        print(f"  bin {g:2d}  eta={bin_centers[g]:+.4f}  coarse={x_coarse[g]:.5f}  hires={x_hires[g]:.5f}  "
             f"diff={x_hires[g]-x_coarse[g]:+.5f}")

    # ================= profile / bias curves =================
    eta_lo, eta_hi = float(eta_all.min()), float(eta_all.max())
    eta_profile = np.linspace(eta_lo, eta_hi, 800)
    true_profile = als.xco2_ppm(eta_profile * als.SLIT_HALF_KM)

    # coarse posterior: step function (nearest-bin), matches how the coarse
    # forward model actually assigns pixels -- searchsorted against the
    # SAME bin edges nearest_bin_scene itself uses.
    coarse_edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
    coarse_idx = np.searchsorted(coarse_edges, eta_profile)
    coarse_posterior_profile = retrieved_ppm_coarse[coarse_idx]

    # hi-res posterior: piecewise-linear through the SAME G retrieved
    # points -- exactly what build_forward_hires's own np.interp uses.
    hires_posterior_profile = np.interp(eta_profile, bin_centers, retrieved_ppm_hires)

    bias_coarse_profile = coarse_posterior_profile - true_profile
    bias_hires_profile = hires_posterior_profile - true_profile

    print(f"\nretrieval bias (posterior - true), evaluated across the full window:")
    print(f"  coarse:  mean={bias_coarse_profile.mean():+.3f} ppm  "
         f"rms={np.sqrt(np.mean(bias_coarse_profile**2)):.3f} ppm  "
         f"max|bias|={np.max(np.abs(bias_coarse_profile)):.3f} ppm")
    print(f"  hi-res:  mean={bias_hires_profile.mean():+.3f} ppm  "
         f"rms={np.sqrt(np.mean(bias_hires_profile**2)):.3f} ppm  "
         f"max|bias|={np.max(np.abs(bias_hires_profile)):.3f} ppm")

    # ================= figure =================
    def add_row_grid(ax, row_min, row_max):
        for r in range(row_min, row_max + 1):
            ax.axhline(r, color="white", lw=0.25, alpha=0.35, zorder=1)

    def add_boundaries(ax, edges, eta_grid, rows_win, color="white", lw=1.0, alpha=0.9):
        ax.contour(np.arange(1024), rows_win, eta_grid, levels=edges,
                  colors=color, linewidths=lw, alpha=alpha, zorder=3)

    vmax_resid = max(np.abs(resid_img_coarse).max(), np.abs(resid_img_hires).max())

    fig = plt.figure(figsize=(14.5, 16.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.3, 1.0, 1.15, 1.15], hspace=0.48, wspace=0.28)

    # row 1 (spanning): true / prior / posterior CO2 profile
    ax = fig.add_subplot(gs[0, :])
    ax.plot(eta_profile, true_profile, color="black", lw=1.8, label="true", zorder=5)
    ax.plot(bin_centers, prior_co2_ppm_bins, "o", ms=5, color="0.55", mec="white", mew=0.6,
           label=f"prior (G={args.n_bins}, local truth by construction)", zorder=6)
    ax.plot(eta_profile, coarse_posterior_profile, color="tab:orange", lw=1.6,
           label="coarse posterior (step)", zorder=4)
    ax.plot(bin_centers, retrieved_ppm_coarse, "o", ms=4, color="tab:orange", zorder=6)
    ax.plot(eta_profile, hires_posterior_profile, color="tab:blue", lw=1.6,
           label="hi-res posterior (piecewise-linear)", zorder=4)
    ax.plot(bin_centers, retrieved_ppm_hires, "o", ms=4, color="tab:blue", zorder=6)
    ax.set_ylabel("CO2 [ppm]")
    ax.set_xlabel(r"$\eta$")
    ax.set_title(f"true / prior / posterior CO2 profile, G={args.n_bins}", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper left")

    # row 2 (spanning): retrieval bias vs eta
    ax = fig.add_subplot(gs[1, :])
    ax.axhline(0, color="black", lw=0.7)
    ax.plot(eta_profile, bias_coarse_profile, color="tab:orange", lw=1.6,
           label=f"coarse (rms={np.sqrt(np.mean(bias_coarse_profile**2)):.3f} ppm)")
    ax.plot(eta_profile, bias_hires_profile, color="tab:blue", lw=1.6,
           label=f"hi-res (rms={np.sqrt(np.mean(bias_hires_profile**2)):.3f} ppm)")
    for bc in bin_centers:
        ax.axvline(bc, color="0.85", lw=0.5, zorder=0)
    ax.set_ylabel("retrieval bias\n(posterior - true) [ppm]")
    ax.set_xlabel(r"$\eta$")
    ax.set_title("retrieval bias vs. along-slit position (thin gray lines: retrieval bin centers)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    # row 3: coarse / hi-res residual images
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(resid_img_coarse, aspect="auto", cmap="RdBu_r", vmin=-vmax_resid, vmax=vmax_resid,
                   extent=[0, 1024, args.row_max, args.row_min], zorder=0)
    add_row_grid(ax, args.row_min, args.row_max)
    add_boundaries(ax, coarse_edges, eta_all, rows_win, color="black", lw=1.0, alpha=0.6)
    ax.set_title(f"coarse residual -- mean bin chi2={chi2_coarse.mean():.3g}", fontsize=10.5)
    ax.set_xlabel("column"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="resid")

    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(resid_img_hires, aspect="auto", cmap="RdBu_r", vmin=-vmax_resid, vmax=vmax_resid,
                   extent=[0, 1024, args.row_max, args.row_min], zorder=0)
    add_row_grid(ax, args.row_min, args.row_max)
    add_boundaries(ax, coarse_edges, eta_all, rows_win, color="black", lw=1.0, alpha=0.6)
    ax.set_title(f"hi-res residual -- mean bin chi2={chi2_hires.mean():.3g}", fontsize=10.5)
    ax.set_xlabel("column"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="resid")

    # row 4: coarse / hi-res geometry (bin/anchor boundaries + faint row grid)
    hires_edges = 0.5 * (anchor_etas[:-1] + anchor_etas[1:])
    ax = fig.add_subplot(gs[3, 0])
    im = ax.imshow(eta_all, aspect="auto", cmap="cividis", extent=[0, 1024, args.row_max, args.row_min], zorder=0)
    add_row_grid(ax, args.row_min, args.row_max)
    add_boundaries(ax, coarse_edges, eta_all, rows_win)
    ax.set_title(f"coarse geometry: G={args.n_bins} bins", fontsize=10.5)
    ax.set_xlabel("column"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label=r"$\eta$")

    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(eta_all, aspect="auto", cmap="cividis", extent=[0, 1024, args.row_max, args.row_min], zorder=0)
    add_row_grid(ax, args.row_min, args.row_max)
    add_boundaries(ax, hires_edges, eta_all, rows_win, lw=0.35, alpha=0.55)
    ax.set_title(f"hi-res geometry: G_eff={len(anchor_rows)} anchors (1/row)", fontsize=10.5)
    ax.set_xlabel("column"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label=r"$\eta$")

    fig.suptitle(f"Coarse vs. hi-res forward model -- FPA2 rows {args.row_min}-{args.row_max}, "
                f"G={args.n_bins} free parameters", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f"gd_joint_block_hires_test_fpa2_row{args.row_min}-{args.row_max}_G{args.n_bins}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
