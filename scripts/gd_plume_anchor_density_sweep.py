#!/usr/bin/env python3
"""Direct test of the keystone-aware anchor placement proposal
(docs/JOINT_BLOCK_MIGRATION_PLAN.md Sec.9, "Towards a keystone-aware
anchor selection strategy"), matched exactly against the existing
row-uniform-anchor `build_forward_hires` at the SAME state dimension G,
on the same three windows and the same G values already swept in
`gd_plume_bin_density_sweep.py` -- a direct, apples-to-apples rerun of
that sweep with one thing changed.

Old anchor scheme (`build_forward_hires`, `gd_joint_block_hires_test.
py:63`): G_eff = width + 2*PAD anchors, ONE PER ROW, each anchor's eta
taken at column 512 only -- fixed regardless of G. The retrieval's G-dim
state is then linearly interpolated onto this fixed G_eff grid. When
G > G_eff this interpolation matrix is provably rank-deficient (see the
doc's `G > G_eff` null-space finding) -- part of the state is invisible
to the data. When G < G_eff (every case tested here, since the G values
below never exceed each window's own G_eff), the interpolation itself
is not rank-deficient, but the ANCHOR grid it interpolates onto is still
placed blind to keystone (uniform by row, not by real pixel density).

New anchor scheme (`build_forward_hires_density`, defined below): G
anchors placed directly via `pixel_density_bin_centers` on the FULL
per-pixel eta population of the PADDED window (every column of every
padded row, not just column 512 of each row) -- i.e. G_eff := G by
construction, anchors placed exactly like the coarse model's own bins
already are, just extended to cover the padding. No interpolation
matrix, no null space possible by construction. Each anchor gets its
own independent RT call directly from the G-dim state x (x IS the
anchor's own co2_scale, not interpolated from a separate coarser grid).

Both schemes solved independently (fresh Gauss-Newton solves) at every
(window, G) point already tested in `gd_plume_bin_density_sweep.py`, so
the comparison isolates anchor PLACEMENT strategy at matched G (hence
roughly matched RT cost), not a different G range or different windows.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_anchor_density_sweep.py \\
        [--n-workers N]
Output: results/gd_plume_anchor_density_sweep_fpa2.pkl,
        plots/joint_block/gd_plume_anchor_density_sweep_fpa2.png,
        plots/joint_block/gd_plume_anchor_density_sweep_profiles_fpa2.png
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
    FPA, GERT_ROOT, _eta_of, band_basics, build_forward,
    gauss_newton_regularized, make_spectrum_fn,
)
from gd_joint_block_hires_test import build_forward_hires  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
from gd_joint_block_whole_slit_sweep import G_RATIO, PAD, ROW_MAX_IDX  # noqa: E402
from gd_plume_bin_density_sweep import WINDOWS  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, nearest_bin_scene, sample_geometries  # noqa: E402
from geocarb_gert import gd_render  # noqa: E402
from geocarb_gert.gd_render import available_cpus  # noqa: E402

BASE_GAMMA = 3.0
BASE_SIGMA_ABS = 0.10

_CTX = {}


def build_forward_hires_density(fpa, rows_win, G, spectrum_for, wn_hires, ils, pad=4, atm_center=None):
    """Keystone-aware anchor placement: G anchors placed via pixel-density
    quantiles over the FULL padded (row, col) pixel population -- G_eff :=
    G by construction, so no interpolation matrix and no null space is
    even possible (contrast `build_forward_hires`'s fixed row-uniform,
    G_eff=width+2*PAD grid). x IS each anchor's own co2_scale directly."""
    row_lo, row_hi = int(rows_win.min()), int(rows_win.max())
    pad_rows = np.arange(max(0, row_lo - pad), min(ROW_MAX_IDX, row_hi + pad) + 1)
    cols = np.arange(1024.0)
    eta_pad_flat = np.stack([_eta_of(fpa, cols, np.full(1024, float(r))) for r in pad_rows]).ravel()
    anchor_etas_sorted = np.sort(pixel_density_bin_centers(eta_pad_flat, G))

    if atm_center is not None:
        anchor_atms = [atm_center] * G
    else:
        anchor_atms = [als.atmosphere_at(float(e * als.SLIT_HALF_KM)) for e in anchor_etas_sorted]

    cache_x = np.full(G, np.nan)
    cache_S = [None] * G

    def forward(x):
        x = np.asarray(x, dtype=float)
        for g in range(G):
            if cache_S[g] is None or x[g] != cache_x[g]:
                cache_S[g] = spectrum_for(x[g], anchor_atms[g])
                cache_x[g] = x[g]
        radiance = nearest_bin_scene(anchor_etas_sorted, cache_S)
        A = gd_render.predict_neighborhood(fpa, rows_win, wn_hires, radiance, ils, pad=pad)
        return A.ravel()

    return forward, anchor_etas_sorted


def _solve_task(task):
    row_lo, row_hi, G = task
    band, absco, wide_inst, geo, solar, albedo = (
        _CTX["band"], _CTX["absco"], _CTX["wide_inst"], _CTX["geo"], _CTX["solar"], _CTX["albedo"])
    wn_hires, ils = _CTX["wn_hires"], _CTX["ils"]

    rows_win = np.arange(row_lo, row_hi + 1)
    width = len(rows_win)
    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    x_km_bins = bin_centers * als.SLIT_HALF_KM
    prior_atms = [als.atmosphere_at(float(xk)) for xk in x_km_bins]

    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)
    eta_win = _eta_of(FPA, np.full(width, 512.0), rows_win.astype(float))
    true_win = als.xco2_ppm(eta_win * als.SLIT_HALF_KM)

    label = f"[{row_lo}-{row_hi} G{G}]"

    # ---- OLD: row-uniform anchors, G interpolated onto G_eff=width+2*PAD ----
    anchor_rows = np.arange(max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD) + 1)
    forward_old, anchor_etas_old = build_forward_hires(FPA, rows_win, anchor_rows, bin_centers,
                                                        spectrum_for, wn_hires, ils, pad=PAD, atm_center=None)
    x_old = gauss_newton_regularized(forward_old, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                     gamma=BASE_GAMMA, sigma_abs=BASE_SIGMA_ABS, label=f"{label} old")
    resid_old = y_true - forward_old(x_old)
    resid_old_rms = float(np.sqrt(np.mean(resid_old ** 2)))
    prior_co2_ppm_bins = np.array([float(als.xco2_ppm(xk)) for xk in x_km_bins])
    retrieved_old = prior_co2_ppm_bins * x_old
    old_win = np.interp(eta_win, bin_centers, retrieved_old) if G > 1 else np.full(width, retrieved_old[0])
    bias_old = old_win - true_win

    # ---- NEW: keystone-aware, pixel-density-placed anchors, G_eff := G ----
    forward_new, anchor_etas_new = build_forward_hires_density(FPA, rows_win, G, spectrum_for, wn_hires, ils,
                                                                pad=PAD, atm_center=None)
    x_new = gauss_newton_regularized(forward_new, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                     gamma=BASE_GAMMA, sigma_abs=BASE_SIGMA_ABS, label=f"{label} new")
    resid_new = y_true - forward_new(x_new)
    resid_new_rms = float(np.sqrt(np.mean(resid_new ** 2)))
    anchor_co2_ppm = np.array([float(als.xco2_ppm(e * als.SLIT_HALF_KM)) for e in anchor_etas_new])
    retrieved_new = anchor_co2_ppm * x_new
    new_win = np.interp(eta_win, anchor_etas_new, retrieved_new) if G > 1 else np.full(width, retrieved_new[0])
    bias_new = new_win - true_win

    return dict(row_lo=row_lo, row_hi=row_hi, G=G, width=width, rows_win=rows_win, true_win=true_win,
               old_win=old_win, new_win=new_win, x_old=x_old, x_new=x_new,
               resid_old=resid_old, resid_new=resid_new,
               bias_old_rms=float(np.sqrt(np.mean(bias_old ** 2))),
               bias_old_max=float(np.max(np.abs(bias_old))),
               bias_new_rms=float(np.sqrt(np.mean(bias_new ** 2))),
               bias_new_max=float(np.max(np.abs(bias_new))),
               resid_old_rms=resid_old_rms, resid_new_rms=resid_new_rms)


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

    tasks = []
    for row_lo, row_hi, _, g_values in WINDOWS:
        for G in g_values:
            tasks.append((row_lo, row_hi, G))
    print(f"{len(tasks)} (window, G) solves queued, OLD vs NEW anchor scheme both solved at each "
         f"(gamma={BASE_GAMMA:g}, sigma_abs={BASE_SIGMA_ABS:g})", flush=True)

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
                     f"old={r['bias_old_rms']:.4f} new={r['bias_new_rms']:.4f}"
            print(f"  {n_done}/{len(tasks)} {key}: {status} ({time.time()-t0:.0f}s elapsed)", flush=True)

    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nall done ({time.time()-t0:.0f}s): {n_ok}/{len(tasks)} solves succeeded", flush=True)

    out_path = REPO_ROOT / "results" / f"gd_plume_anchor_density_sweep_fpa{FPA}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "windows": WINDOWS, "fpa": FPA,
                    "base_gamma": BASE_GAMMA, "base_sigma_abs": BASE_SIGMA_ABS}, f)
    print(f"saved {out_path}")

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    # ---- trend plot: bias rms vs. G, old vs. new, per window ----
    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(9, 3.6 * len(WINDOWS)))
    for k, (row_lo, row_hi, wlabel, g_values) in enumerate(WINDOWS):
        pts = [results[(row_lo, row_hi, G)] for G in sorted(g_values)
              if (row_lo, row_hi, G) in results and "error" not in results[(row_lo, row_hi, G)]]
        prod_G = max(2, int(round((row_hi - row_lo + 1) / G_RATIO)))

        ax = axes[k]
        ax.plot([r["G"] for r in pts], [r["bias_old_rms"] for r in pts],
               "o-", color="tab:blue", label="OLD (row-uniform anchors)")
        ax.plot([r["G"] for r in pts], [r["bias_new_rms"] for r in pts],
               "o-", color="tab:red", label="NEW (pixel-density anchors)")
        ax.axvline(prod_G, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("G (= G_eff for the new scheme)")
        ax.set_ylabel("hi-res bias rms [ppm]")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel}, width={row_hi-row_lo+1}, production G={prod_G})", fontsize=10)
        if k == 0:
            ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(f"FPA{FPA}: anchor placement strategy -- row-uniform (old) vs. pixel-density (new), "
                f"matched G", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_plume_anchor_density_sweep_fpa{FPA}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")

    # ---- profile plot: at the finest G tested, old vs new vs true ----
    fig2, axes2 = plt.subplots(len(WINDOWS), 1, figsize=(11, 3.4 * len(WINDOWS)))
    if len(WINDOWS) == 1:
        axes2 = [axes2]
    for k, (row_lo, row_hi, wlabel, g_values) in enumerate(WINDOWS):
        fine_G = max(g_values)
        r = results.get((row_lo, row_hi, fine_G))
        ax = axes2[k]
        if r is not None and "error" not in r:
            ax.plot(r["rows_win"], r["true_win"], color="black", lw=1.4, label="true", zorder=5)
            ax.plot(r["rows_win"], r["old_win"], color="tab:blue", lw=1.1,
                   label=f"OLD, G={fine_G} (rms={r['bias_old_rms']:.3f})")
            ax.plot(r["rows_win"], r["new_win"], color="tab:red", lw=1.1, ls="--",
                   label=f"NEW, G={fine_G} (rms={r['bias_new_rms']:.3f})")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel})", fontsize=10.5)
        ax.set_xlabel("detector row"); ax.set_ylabel("CO2 [ppm]")
        ax.legend(fontsize=8, loc="best")

    fig2.suptitle(f"FPA{FPA}: profiles at the finest G tested, old vs. new anchor scheme", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out_fig2 = plots_dir / f"gd_plume_anchor_density_sweep_profiles_fpa{FPA}.png"
    fig2.savefig(out_fig2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out_fig2}")

    print("\nsummary:")
    print(f"{'window':>12s} {'G':>4s} {'old_rms':>9s} {'new_rms':>9s} {'improvement':>12s} "
         f"{'resid_old':>11s} {'resid_new':>11s}")
    for row_lo, row_hi, wlabel, g_values in WINDOWS:
        for G in sorted(g_values):
            r = results.get((row_lo, row_hi, G))
            if r is None or "error" in r:
                continue
            improvement = (r["bias_old_rms"] - r["bias_new_rms"]) / r["bias_old_rms"] * 100 if r["bias_old_rms"] > 0 else 0.0
            print(f"{row_lo}-{row_hi:>6d} {G:4d} {r['bias_old_rms']:9.4f} {r['bias_new_rms']:9.4f} "
                 f"{improvement:11.1f}% {r['resid_old_rms']:11.2e} {r['resid_new_rms']:11.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
