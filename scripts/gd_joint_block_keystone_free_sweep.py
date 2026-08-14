#!/usr/bin/env python3
"""Keystone-free ceiling for the whole-slit joint-block sweep
(scripts/gd_joint_block_whole_slit_sweep.py): how much of the joint
block's remaining bias is keystone/smile specifically, versus the
resolution floor (finite G, PSF blur) that would persist even at zero
distortion?

Reuses the real sweep's own window tiles, per-window G, and bin_centers
unchanged (results/gd_joint_block_whole_slit_fpa2.pkl) -- same resolution
choices, so the comparison is apples-to-apples against that sweep's own
coarse/hi-res results, not a re-tuned experiment.

The forward model and the synthetic "truth" both use
predict_neighborhood_no_keystone below instead of gd_render.
predict_neighborhood: every column in a row is assigned that row's own
single center-column eta (removing keystone's within-row spread, same
convention gd_test._undistorted_row already uses for "no keystone"), while
real per-column nu (smile/dispersion), real ILS convolution, AND the real
spatial PSF blur are all still applied. This is the key difference from
the existing "undistorted" pipeline, which drops the PSF blur entirely and
was already shown (prior turn) to be the wrong ceiling for the joint block
-- the joint block's own forward model always keeps the real PSF, so its
ceiling must too.

Nuisance gases are held at the SAME local-truth idealization the real
joint block uses (als.atmosphere_at(x_km_bin), only co2_scale free) --
not native/undistorted's own fully-free joint nuisance retrieval. This
keeps the comparison isolating keystone/smile specifically: both this
ceiling and the real (keystone-present) sweep share the identical nuisance
idealization, window tiling, G, bin placement, forward-model resolution
scheme (coarse/hi-res), and regularization, differing *only* in whether
keystone/smile spread is present in the synthetic measurement.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_keystone_free_sweep.py \\
        [--n-workers N]
Output: results/gd_joint_block_keystone_free_fpa2.pkl
        plots/joint_block/gd_joint_block_keystone_free_fpa2.png
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
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
    FPA, GERT_ROOT, _eta_of, band_basics, gauss_newton_regularized, make_spectrum_fn,
)

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, nearest_bin_scene, sample_geometries  # noqa: E402
from geocarb_gert.focalplane import gaussian_blur_rows  # noqa: E402
from geocarb_gert.gd_polynomials import N_PX, rows_crossed, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import available_cpus  # noqa: E402
from geocarb_gert import gd_render  # noqa: E402

ROW_MAX_IDX = 1023

_SWEEP = {}


def predict_neighborhood_no_keystone(fpa, rows, wn_hires, radiance, ils, pad=4,
                                     spatial_psf_fwhm_px=1.5):
    """gd_render.predict_neighborhood, but with within-row keystone spread
    removed: every column in a row samples that row's own single
    center-column eta (col 512, matching gd_test._undistorted_row's own
    convention) instead of each column's own real, keystone-shifted eta.
    Real per-column nu (smile kept), real ILS convolution, and the real
    spatial PSF blur are all still applied -- unlike _undistorted_row,
    which drops the PSF blur too. See module docstring for why that
    distinction is what makes this the correct ceiling here."""
    rows = np.asarray(rows, dtype=int)
    row_lo, row_hi = int(rows.min()) - pad, int(rows.max()) + pad
    rows_padded = np.arange(max(0, row_lo), min(N_PX, row_hi + 1))
    cols = np.arange(N_PX, dtype=float)

    A_pad = np.empty((len(rows_padded), N_PX), dtype=float)
    for k, i in enumerate(rows_padded):
        lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(N_PX, float(i)))
        nu_row = 1.0e4 / lam_row
        eta_center = float(_eta_of(fpa, np.array([512.0]), np.array([float(i)]))[0])
        S_row = np.asarray(radiance(np.full(N_PX, eta_center)), dtype=float)
        A_pad[k] = gd_render._diagonal_ils_convolve(wn_hires, S_row, nu_row, ils)

    A_pad = gaussian_blur_rows(A_pad, spatial_psf_fwhm_px)
    idx = np.searchsorted(rows_padded, rows)
    return A_pad[idx]


def build_forward_no_keystone(fpa, rows, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=4):
    """gd_joint_block_retrieve.build_forward, rendered with
    predict_neighborhood_no_keystone -- the coarse keystone-free ceiling
    forward model."""
    G = len(bin_centers)
    cache_x = np.full(G, np.nan)
    cache_S = [None] * G

    def forward(x):
        x = np.asarray(x, dtype=float)
        for g in range(G):
            if cache_S[g] is None or x[g] != cache_x[g]:
                cache_S[g] = spectrum_for(x[g], prior_atms[g])
                cache_x[g] = x[g]
        radiance = nearest_bin_scene(bin_centers, cache_S)
        A = predict_neighborhood_no_keystone(fpa, rows, wn_hires, radiance, ils, pad=pad)
        return A.ravel()

    return forward


def build_forward_hires_no_keystone(fpa, rows_win, anchor_rows, bin_centers, spectrum_for,
                                    wn_hires, ils, pad=4, atm_center=None):
    """gd_joint_block_hires_test.build_forward_hires, rendered with
    predict_neighborhood_no_keystone. With keystone removed, every column
    in a row already shares one eta, so the coarse-vs-hires distinction
    here is purely between-row interpolation (step vs piecewise-linear
    across rows), not within-row attribution.

    atm_center : optional shared-atmosphere override for a uniform-scene
    test -- see gd_joint_block_hires_test.build_forward_hires's own note."""
    anchor_etas = _eta_of(fpa, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float))
    order = np.argsort(anchor_etas)
    anchor_etas_sorted = anchor_etas[order]
    G_eff = len(anchor_etas_sorted)
    if atm_center is not None:
        anchor_atms = [atm_center] * G_eff
    else:
        anchor_atms = [als.atmosphere_at(float(e * als.SLIT_HALF_KM)) for e in anchor_etas_sorted]

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
        A = predict_neighborhood_no_keystone(fpa, rows_win, wn_hires, radiance, ils, pad=pad)
        return A.ravel()

    return forward, anchor_etas_sorted


def _solve_window(row_lo: int, row_hi: int, bin_centers: np.ndarray, G: int):
    band = _SWEEP["band"]
    absco, wide_inst, geo, solar, albedo = (_SWEEP["absco"], _SWEEP["wide_inst"],
                                            _SWEEP["geo"], _SWEEP["solar"], _SWEEP["albedo"])
    wn_hires, ils, gamma, sigma_abs = _SWEEP["wn_hires"], _SWEEP["ils"], _SWEEP["gamma"], _SWEEP["sigma_abs"]
    pad = _SWEEP["pad"]
    uniform, atm_center = _SWEEP["uniform"], _SWEEP["atm_center"]

    rows_win = np.arange(row_lo, row_hi + 1)
    if uniform:
        prior_atms = [atm_center] * G
        prior_co2_ppm_bins = np.full(G, float(als.xco2_ppm(0.0)))
    else:
        x_km_bins = bin_centers * als.SLIT_HALF_KM
        prior_atms = [als.atmosphere_at(float(xk)) for xk in x_km_bins]
        prior_co2_ppm_bins = np.array([float(als.xco2_ppm(xk)) for xk in x_km_bins])

    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)

    # keystone-free synthetic "truth" -- real scene, real PSF, no keystone.
    y_true = predict_neighborhood_no_keystone(FPA, rows_win, wn_hires, band["radiance"],
                                              ils, pad=pad).ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

    out = dict(row_lo=row_lo, row_hi=row_hi, width=len(rows_win), G=G, bin_centers=bin_centers,
              prior_co2_ppm_bins=prior_co2_ppm_bins)

    t0 = time.time()
    forward_coarse = build_forward_no_keystone(FPA, rows_win, bin_centers, prior_atms,
                                               spectrum_for, wn_hires, ils, pad=pad)
    x_coarse = gauss_newton_regularized(forward_coarse, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                        gamma=gamma, sigma_abs=sigma_abs, label=f"[{row_lo}-{row_hi}] coarse-nk")
    resid_coarse = y_true - forward_coarse(x_coarse)
    out["x_coarse"] = x_coarse
    out["resid_coarse_rms"] = float(np.sqrt(np.mean(resid_coarse ** 2)))
    out["t_coarse"] = time.time() - t0

    anchor_rows = np.arange(max(0, row_lo - pad), min(ROW_MAX_IDX, row_hi + pad) + 1)
    t0 = time.time()
    forward_hires, anchor_etas = build_forward_hires_no_keystone(FPA, rows_win, anchor_rows, bin_centers,
                                                                  spectrum_for, wn_hires, ils, pad=pad,
                                                                  atm_center=(atm_center if uniform else None))
    x_hires = gauss_newton_regularized(forward_hires, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                       gamma=gamma, sigma_abs=sigma_abs, label=f"[{row_lo}-{row_hi}] hires-nk")
    resid_hires = y_true - forward_hires(x_hires)
    out["x_hires"] = x_hires
    out["G_eff"] = len(anchor_rows)
    out["resid_hires_rms"] = float(np.sqrt(np.mean(resid_hires ** 2)))
    out["t_hires"] = time.time() - t0

    return out


def _worker(task):
    row_lo, row_hi, bin_centers, G = task
    try:
        return _solve_window(row_lo, row_hi, bin_centers, G)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        return dict(row_lo=row_lo, row_hi=row_hi, error=f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--uniform", action="store_true", help="constant-atmosphere scene, "
                    "for isolating implementation bugs from real resolution-floor effects "
                    "-- see gd_joint_block_whole_slit_sweep.py's own --uniform for the "
                    "rationale. Tiles/G/bin_centers are still reused from the REALISTIC-"
                    "scene source pickle regardless (window tiling is purely geometric, "
                    "independent of scene content), only the band and prior atmospheres change.")
    args = ap.parse_args()

    src_path = REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}.pkl"
    with open(src_path, "rb") as f:
        src = pickle.load(f)
    src_windows = sorted(src["results"].values(), key=lambda r: r["row_lo"])
    gamma, sigma_abs, pad = src["gamma"], src["sigma_abs"], src["pad"]
    print(f"reusing {len(src_windows)} windows/G/bin_centers from {src_path.name} "
         f"(gamma={gamma}, sigma_abs={sigma_abs}, pad={pad})", flush=True)

    scene_label = "uniform" if args.uniform else "realistic"
    print(f"Building {scene_label}-scene FPA{FPA} band...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400, None,
                           args.uniform, False, 32, False, 0)
    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    _SWEEP.update(dict(band=band, absco=absco, wide_inst=wide_inst, geo=geo, solar=solar,
                       albedo=albedo, wn_hires=band["wn_hires"], ils=band["ils"],
                       gamma=gamma, sigma_abs=sigma_abs, pad=pad,
                       uniform=args.uniform, atm_center=atm_center))

    tasks = [(w["row_lo"], w["row_hi"], w["bin_centers"], w["G"]) for w in src_windows]

    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    print(f"solving {len(tasks)} windows with {n_workers} workers...", flush=True)

    t0 = time.time()
    results = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for r in pool.imap_unordered(_worker, tasks, chunksize=1):
            key = (r["row_lo"], r["row_hi"])
            results[key] = r
            n_done += 1
            status = "ERROR: " + r["error"] if "error" in r else \
                     f"G={r['G']} t_coarse={r['t_coarse']:.0f}s t_hires={r['t_hires']:.0f}s"
            elapsed = time.time() - t0
            print(f"  {n_done}/{len(tasks)} rows {key[0]}-{key[1]}: {status} "
                 f"({elapsed:.0f}s elapsed)", flush=True)

    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nall done ({time.time()-t0:.0f}s): {n_ok}/{len(tasks)} windows solved successfully", flush=True)

    suffix = "_uniform" if args.uniform else ""
    out_path = REPO_ROOT / "results" / f"gd_joint_block_keystone_free_fpa{FPA}{suffix}.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "fpa": FPA, "uniform": args.uniform, "gamma": gamma,
                    "sigma_abs": sigma_abs, "pad": pad, "source": str(src_path)}, f)
    print(f"saved {out_path}")

    # ================= comparison plot =================
    def stitch(results_dict):
        windows = sorted(results_dict.values(), key=lambda r: r["row_lo"])
        rows_all, true_all, coarse_all, hires_all = [], [], [], []
        for w in windows:
            row_lo, row_hi = w["row_lo"], w["row_hi"]
            rows_win = np.arange(row_lo, row_hi + 1)
            eta_win = _eta_of(FPA, np.full(len(rows_win), 512.0), rows_win.astype(float))
            true_win = (np.full(len(rows_win), float(als.xco2_ppm(0.0))) if args.uniform
                       else als.xco2_ppm(eta_win * als.SLIT_HALF_KM))
            bin_centers = w["bin_centers"]
            retrieved_ppm_coarse = w["prior_co2_ppm_bins"] * w["x_coarse"]
            retrieved_ppm_hires = w["prior_co2_ppm_bins"] * w["x_hires"]
            if len(bin_centers) > 1:
                edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
                idx = np.searchsorted(edges, eta_win)
                coarse_win = retrieved_ppm_coarse[idx]
                hires_win = np.interp(eta_win, bin_centers, retrieved_ppm_hires)
            else:
                coarse_win = np.full(len(rows_win), retrieved_ppm_coarse[0])
                hires_win = np.full(len(rows_win), retrieved_ppm_hires[0])
            rows_all.append(rows_win); true_all.append(true_win)
            coarse_all.append(coarse_win); hires_all.append(hires_win)
        rows_all = np.concatenate(rows_all); true_all = np.concatenate(true_all)
        coarse_all = np.concatenate(coarse_all); hires_all = np.concatenate(hires_all)
        return rows_all, coarse_all - true_all, hires_all - true_all

    rows_nk, bias_coarse_nk, bias_hires_nk = stitch(results)
    if args.uniform:
        ref_path = REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}_uniform.pkl"
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)
        print(f"reference (keystone-present) curve from {ref_path.name}")
    else:
        ref = src
    rows_real, bias_coarse_real, bias_hires_real = stitch(ref["results"])

    def stats(b):
        return dict(mean=float(b.mean()), rms=float(np.sqrt(np.mean(b ** 2))), max=float(np.max(np.abs(b))))

    s_c_nk, s_h_nk = stats(bias_coarse_nk), stats(bias_hires_nk)
    s_c_real, s_h_real = stats(bias_coarse_real), stats(bias_hires_real)
    print(f"\nkeystone-free ceiling: coarse rms={s_c_nk['rms']:.4f} max={s_c_nk['max']:.4f} ppm; "
         f"hires rms={s_h_nk['rms']:.4f} max={s_h_nk['max']:.4f} ppm")
    print(f"real (keystone-present): coarse rms={s_c_real['rms']:.4f} max={s_c_real['max']:.4f} ppm; "
         f"hires rms={s_h_real['rms']:.4f} max={s_h_real['max']:.4f} ppm")

    rc_fine = rows_crossed(FPA, rows_real.astype(float))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.6, 0.9]})

    ax = axes[0]
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(rows_real, bias_coarse_real, color="tab:orange", lw=0.8, alpha=0.9,
           label=f"keystone-present, coarse (rms={s_c_real['rms']:.3f}, max={s_c_real['max']:.3f})")
    ax.plot(rows_real, bias_hires_real, color="tab:blue", lw=0.8, alpha=0.9,
           label=f"keystone-present, hi-res (rms={s_h_real['rms']:.3f}, max={s_h_real['max']:.3f})")
    ax.set_ylabel("retrieval bias\n(posterior - true) [ppm]")
    ax.set_title("real (keystone-present) joint block, whole slit -- for reference", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[1]
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(rows_nk, bias_coarse_nk, color="tab:orange", lw=0.8, ls="--", alpha=0.9,
           label=f"keystone-free ceiling, coarse (rms={s_c_nk['rms']:.3f}, max={s_c_nk['max']:.3f})")
    ax.plot(rows_nk, bias_hires_nk, color="tab:blue", lw=0.8, ls="--", alpha=0.9,
           label=f"keystone-free ceiling, hi-res (rms={s_h_nk['rms']:.3f}, max={s_h_nk['max']:.3f})")
    ax.set_ylabel("retrieval bias\n(posterior - true) [ppm]")
    ax.set_title("keystone-free (real PSF kept, same G/bins/nuisance idealization) -- same y scale as above", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(axes[0].get_ylim())

    ax = axes[2]
    ax.plot(rows_real, rc_fine, color="0.5", lw=1.0)
    ax.set_ylabel("rows_crossed\n(local keystone)")
    ax.set_xlabel("detector row")
    ax.set_title("local keystone amplitude, for reference", fontsize=10)

    fig.suptitle(f"FPA{FPA}: does removing keystone/smile close the joint block's remaining gap?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_joint_block_keystone_free_fpa{FPA}{suffix}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
