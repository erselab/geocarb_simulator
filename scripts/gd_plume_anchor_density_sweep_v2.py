#!/usr/bin/env python3
"""Corrected version of gd_plume_anchor_density_sweep.py, isolating anchor
PLACEMENT from the separate "collapse G and G_eff into one tier" change
that confounded the first attempt (docs/JOINT_BLOCK_MIGRATION_PLAN.md
Sec.9 -- that run showed the new scheme performing WORSE almost
everywhere, traced to G=5 anchors spread over the padded window leaving
only ~2-3 real RT points inside the core window, versus the old scheme's
G interpolated onto a fixed 23-point grid there -- a confound, not
evidence against density-weighted placement itself).

This version keeps the OLD two-tier architecture (G state values
interpolated onto a FIXED, larger G_eff anchor grid, exactly like
`build_forward_hires`) for BOTH schemes, changing ONLY how the G_eff
anchors are placed:
  - OLD: `build_forward_hires`'s own row-uniform anchors (one per row,
    column 512), G_eff = width + 2*PAD, same as every other script in
    this investigation.
  - NEW: `build_forward_hires_density` (below), same G_eff count, but
    positions chosen via `pixel_density_bin_centers` over the padded
    window's full per-pixel eta population -- computed ONCE per window,
    fixed across the whole G sweep (not recomputed per G), so both
    schemes' anchor grids are directly comparable at matched size.

G itself (the retrieval's free state parameters, interpolated onto
whichever fixed anchor grid) is swept from 5 up to G_eff+1 for each
window, deliberately crossing each window's own G_eff -- the point
being to watch the bias/resid curve (and the interpolation matrix M's
own rank) cross from full column rank (G <= G_eff) into the exact null
space proven analytically in the doc (G > G_eff), for BOTH anchor
schemes side by side.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_anchor_density_sweep_v2.py \\
        [--n-workers N]
Output: results/gd_plume_anchor_density_sweep_v2_fpa2.pkl,
        plots/joint_block/gd_plume_anchor_density_sweep_v2_fpa2.png
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
    FPA, GERT_ROOT, _eta_of, band_basics, make_spectrum_fn,
    gauss_newton_regularized,
)
from gd_joint_block_hires_test import build_forward_hires  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
from gd_joint_block_whole_slit_sweep import PAD, ROW_MAX_IDX  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, nearest_bin_scene, sample_geometries  # noqa: E402
from geocarb_gert import gd_render  # noqa: E402
from geocarb_gert.gd_render import available_cpus  # noqa: E402

BASE_GAMMA = 3.0
BASE_SIGMA_ABS = 0.10

# (row_lo, row_hi, label) -- same three windows throughout this
# investigation. G_eff (= width+2*PAD) and the G sweep (5 .. G_eff+1)
# are computed from this at runtime.
WINDOWS = [
    (331, 345, "broad plume peak"),
    (884, 924, "east hot spot (ATBD 890-935 ref)"),
    (108, 116, "west hot spot (worst-case)"),
]

_CTX = {}


def interp_matrix(anchor_etas_sorted, bin_centers):
    G_eff, G = len(anchor_etas_sorted), len(bin_centers)
    M = np.zeros((G_eff, G))
    idx_hi = np.clip(np.searchsorted(bin_centers, anchor_etas_sorted), 1, G - 1)
    idx_lo = idx_hi - 1
    e_lo, e_hi = bin_centers[idx_lo], bin_centers[idx_hi]
    frac = np.clip((anchor_etas_sorted - e_lo) / (e_hi - e_lo), 0.0, 1.0)
    for i in range(G_eff):
        M[i, idx_lo[i]] += 1 - frac[i]
        M[i, idx_hi[i]] += frac[i]
    return M


def build_forward_hires_density(fpa, rows_win, anchor_etas_sorted, bin_centers, spectrum_for,
                                wn_hires, ils, pad=4, atm_center=None):
    """Same architecture as build_forward_hires (G state values -> linear
    interpolation onto a FIXED anchor grid -> one RT call per anchor ->
    nearest_bin_scene pixel assignment) -- the only difference is
    `anchor_etas_sorted` is supplied directly (pixel-density-placed over
    the padded window, computed once by the caller), not derived from
    `_eta_of(col=512)` per row."""
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
        A = gd_render.predict_neighborhood(fpa, rows_win, wn_hires, radiance, ils, pad=pad)
        return A.ravel()

    return forward


def _solve_task(task):
    row_lo, row_hi, G = task
    band, absco, wide_inst, geo, solar, albedo = (
        _CTX["band"], _CTX["absco"], _CTX["wide_inst"], _CTX["geo"], _CTX["solar"], _CTX["albedo"])
    wn_hires, ils = _CTX["wn_hires"], _CTX["ils"]

    rows_win = np.arange(row_lo, row_hi + 1)
    width = len(rows_win)
    cols = np.arange(1024.0)
    eta_core = np.stack([_eta_of(FPA, cols, np.full(1024, float(r))) for r in rows_win])
    bin_centers = pixel_density_bin_centers(eta_core.ravel(), G)

    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)
    eta_win = _eta_of(FPA, np.full(width, 512.0), rows_win.astype(float))
    true_win = als.xco2_ppm(eta_win * als.SLIT_HALF_KM)

    label = f"[{row_lo}-{row_hi} G{G}]"

    # ---- fixed anchor grids, both computed ONCE per window (not per G) ----
    anchor_rows = np.arange(max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD) + 1)
    anchor_etas_old = np.sort(_eta_of(FPA, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float)))
    eta_pad_flat = np.stack([_eta_of(FPA, cols, np.full(1024, float(r))) for r in anchor_rows]).ravel()
    anchor_etas_new = np.sort(pixel_density_bin_centers(eta_pad_flat, len(anchor_rows)))
    G_eff = len(anchor_rows)

    # ---- OLD scheme ----
    forward_old, _ = build_forward_hires(FPA, rows_win, anchor_rows, bin_centers,
                                         spectrum_for, wn_hires, ils, pad=PAD, atm_center=None)
    x_old = gauss_newton_regularized(forward_old, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                     gamma=BASE_GAMMA, sigma_abs=BASE_SIGMA_ABS, label=f"{label} old")
    resid_old = y_true - forward_old(x_old)
    resid_old_rms = float(np.sqrt(np.mean(resid_old ** 2)))
    co2_ppm_bins = np.array([float(als.xco2_ppm(bc * als.SLIT_HALF_KM)) for bc in bin_centers])
    old_win = np.interp(eta_win, bin_centers, co2_ppm_bins * x_old) if G > 1 else np.full(width, co2_ppm_bins[0] * x_old[0])
    bias_old = old_win - true_win
    rank_old = int(np.linalg.matrix_rank(interp_matrix(anchor_etas_old, bin_centers)))

    # ---- NEW scheme (same G_eff, density-placed) ----
    forward_new = build_forward_hires_density(FPA, rows_win, anchor_etas_new, bin_centers,
                                              spectrum_for, wn_hires, ils, pad=PAD, atm_center=None)
    x_new = gauss_newton_regularized(forward_new, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                     gamma=BASE_GAMMA, sigma_abs=BASE_SIGMA_ABS, label=f"{label} new")
    resid_new = y_true - forward_new(x_new)
    resid_new_rms = float(np.sqrt(np.mean(resid_new ** 2)))
    new_win = np.interp(eta_win, bin_centers, co2_ppm_bins * x_new) if G > 1 else np.full(width, co2_ppm_bins[0] * x_new[0])
    bias_new = new_win - true_win
    rank_new = int(np.linalg.matrix_rank(interp_matrix(anchor_etas_new, bin_centers)))

    return dict(row_lo=row_lo, row_hi=row_hi, G=G, G_eff=G_eff, width=width, rows_win=rows_win, true_win=true_win,
               old_win=old_win, new_win=new_win, x_old=x_old, x_new=x_new,
               resid_old=resid_old, resid_new=resid_new,
               bias_old_rms=float(np.sqrt(np.mean(bias_old ** 2))),
               bias_new_rms=float(np.sqrt(np.mean(bias_new ** 2))),
               resid_old_rms=resid_old_rms, resid_new_rms=resid_new_rms,
               rank_old=rank_old, rank_new=rank_new, nullity=max(0, G - min(rank_old, rank_new)))


def _worker(task):
    row_lo, row_hi, G = task
    try:
        return _solve_task(task)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        return dict(row_lo=row_lo, row_hi=row_hi, G=G, error=f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()

    print(f"Building realistic-scene FPA{FPA} band (renders the real 1024x1024 detector image)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400, None, False, False, 32, False, 0)
    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    _CTX.update(dict(band=band, absco=absco, wide_inst=wide_inst, geo=geo, solar=solar,
                     albedo=albedo, wn_hires=band["wn_hires"], ils=band["ils"]))

    window_g_values = {}
    tasks = []
    for row_lo, row_hi, _ in WINDOWS:
        width = row_hi - row_lo + 1
        G_eff = width + 2 * PAD
        g_values = sorted(set([5, round(G_eff * 0.5), round(G_eff * 0.8), G_eff, G_eff + 1]))
        g_values = [g for g in g_values if g >= 2]
        window_g_values[(row_lo, row_hi)] = g_values
        for G in g_values:
            tasks.append((row_lo, row_hi, G))
    print(f"{len(tasks)} (window, G) solves queued, G sweeping 5 -> G_eff+1 per window, "
         f"OLD (row-uniform) vs NEW (pixel-density) anchors both solved at each "
         f"(gamma={BASE_GAMMA:g}, sigma_abs={BASE_SIGMA_ABS:g})", flush=True)
    for (row_lo, row_hi), g_values in window_g_values.items():
        print(f"  {row_lo}-{row_hi}: G_eff={row_hi-row_lo+1+2*PAD}, G values={g_values}", flush=True)

    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    print(f"solving with {n_workers} workers...", flush=True)

    t0 = time.time()
    results = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for r in pool.imap_unordered(_worker, tasks, chunksize=1):
            key = (r["row_lo"], r["row_hi"], r["G"])
            results[key] = r
            n_done += 1
            status = "ERROR: " + r["error"] if "error" in r else \
                     f"old={r['bias_old_rms']:.4f} new={r['bias_new_rms']:.4f} nullity={r['nullity']}"
            print(f"  {n_done}/{len(tasks)} {key}: {status} ({time.time()-t0:.0f}s elapsed)", flush=True)

    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nall done ({time.time()-t0:.0f}s): {n_ok}/{len(tasks)} solves succeeded", flush=True)

    out_path = REPO_ROOT / "results" / f"gd_plume_anchor_density_sweep_v2_fpa{FPA}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "windows": WINDOWS, "window_g_values": window_g_values, "fpa": FPA,
                    "base_gamma": BASE_GAMMA, "base_sigma_abs": BASE_SIGMA_ABS}, f)
    print(f"saved {out_path}")

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(9.5, 3.8 * len(WINDOWS)))
    for k, (row_lo, row_hi, wlabel) in enumerate(WINDOWS):
        g_values = window_g_values[(row_lo, row_hi)]
        G_eff = row_hi - row_lo + 1 + 2 * PAD
        pts = [results[(row_lo, row_hi, G)] for G in g_values
              if (row_lo, row_hi, G) in results and "error" not in results[(row_lo, row_hi, G)]]

        ax = axes[k]
        ax.plot([r["G"] for r in pts], [r["bias_old_rms"] for r in pts],
               "o-", color="tab:blue", label="OLD (row-uniform anchors)")
        ax.plot([r["G"] for r in pts], [r["bias_new_rms"] for r in pts],
               "o-", color="tab:red", label="NEW (pixel-density anchors)")
        ax.axvline(G_eff, color="0.6", lw=1.2, ls="--", label=f"G_eff={G_eff}")
        ax.set_yscale("log")
        ax.set_xlabel("G (state dimension, interpolated onto fixed G_eff anchors)")
        ax.set_ylabel("hi-res bias rms [ppm]")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel}, width={row_hi-row_lo+1}, G_eff={G_eff})", fontsize=10.5)
        if k == 0:
            ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(f"FPA{FPA}: anchor placement (matched G_eff) -- bias vs. G, crossing the null space", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_plume_anchor_density_sweep_v2_fpa{FPA}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")

    print("\nsummary:")
    print(f"{'window':>12s} {'G':>4s} {'G_eff':>6s} {'old_rms':>9s} {'new_rms':>9s} "
         f"{'rank_old':>9s} {'rank_new':>9s} {'nullity':>8s}")
    for row_lo, row_hi, wlabel in WINDOWS:
        for G in window_g_values[(row_lo, row_hi)]:
            r = results.get((row_lo, row_hi, G))
            if r is None or "error" in r:
                continue
            print(f"{row_lo}-{row_hi:>6d} {G:4d} {r['G_eff']:6d} {r['bias_old_rms']:9.4f} {r['bias_new_rms']:9.4f} "
                 f"{r['rank_old']:9d} {r['rank_new']:9d} {r['nullity']:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
