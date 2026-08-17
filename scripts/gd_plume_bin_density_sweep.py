#!/usr/bin/env python3
"""Direct test of the bin-density (G) resolution-ceiling hypothesis
(docs/JOINT_BLOCK_MIGRATION_PLAN.md Sec.9), the natural follow-up after
`gd_plume_sigma_abs_sweep.py` falsified sigma_abs/gamma as the driver of
the real plume/hot-spot bias on the production windows. There, x_hires
stayed within 0.0006 of 1 regardless of how loose sigma_abs was made --
the retrieval was never being resisted by the prior, so the bias must
come from somewhere else. The leading remaining candidate, argued
directly from the profile plots there: with G held at the production
default, the retrieved curve is a piecewise-linear interpolant between
already-correct (position-dependent "local truth") bin-center values,
and it visibly cuts inside the true curve's own concave stretches --
a resolution ceiling set by how many bins are available to represent
curvature between them, nothing to do with regularization. This script
tests that directly: hold gamma/sigma_abs at the production baseline
(3.0, 0.10) and sweep G (bin count) instead, on the same three real
windows used there.

Windows tested (identical to gd_plume_sigma_abs_sweep.py, see its
docstring for how each was picked from results/gd_joint_block_whole_
slit_fpa2.pkl):
  - rows 331-345 (width=15, production G=5): broad plume peak.
  - rows 884-924 (width=41, production G=14): east hot spot (ATBD
    890-935 ref).
  - rows 108-116 (width=9, production G=3): west hot spot, worst-case
    bias in the whole-slit sweep.

G values per window span from below the production default (G=2, the
coarsest possible) up to roughly one bin per detector row (G=width) or a
bit past it, to see whether the bias keeps shrinking as G grows (resolution
ceiling, expected) or saturates well short of it (something else still
unaccounted for). Sized per-window rather than with one shared G list
since build_forward/build_forward_hires's own cost scales with G
(one extra spectrum_for RT call per Jacobian column per GN iteration) --
the widest window (884-924) is kept to G<=width to bound runtime; the
two narrower windows can afford a wider relative range cheaply.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_bin_density_sweep.py \\
        [--n-workers N]
Output: results/gd_plume_bin_density_sweep_fpa2.pkl,
        plots/joint_block/gd_plume_bin_density_sweep_fpa2.png,
        plots/joint_block/gd_plume_bin_density_sweep_profiles_fpa2.png
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

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.gd_render import available_cpus  # noqa: E402

BASE_GAMMA = 3.0
BASE_SIGMA_ABS = 0.10

# (row_lo, row_hi, label, G values to test) -- production G (width/G_RATIO,
# rounded) is always included so this sweep's baseline point matches
# gd_plume_sigma_abs_sweep.py's own baseline exactly.
WINDOWS = [
    (331, 345, "broad plume peak", [2, 5, 8, 15, 30]),
    (884, 924, "east hot spot (ATBD 890-935 ref)", [2, 7, 14, 28, 41]),
    (108, 116, "west hot spot (worst-case)", [2, 3, 5, 9, 18]),
]

_CTX = {}


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
    prior_co2_ppm_bins = np.array([float(als.xco2_ppm(xk)) for xk in x_km_bins])

    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

    label = f"[{row_lo}-{row_hi} G{G}]"
    forward_coarse = build_forward(FPA, rows_win, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=PAD)
    x_coarse = gauss_newton_regularized(forward_coarse, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                        gamma=BASE_GAMMA, sigma_abs=BASE_SIGMA_ABS, label=f"{label} coarse")
    resid_coarse = y_true - forward_coarse(x_coarse)
    resid_coarse_rms = float(np.sqrt(np.mean(resid_coarse ** 2)))

    anchor_rows = np.arange(max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD) + 1)
    forward_hires, anchor_etas = build_forward_hires(FPA, rows_win, anchor_rows, bin_centers,
                                                      spectrum_for, wn_hires, ils, pad=PAD, atm_center=None)
    x_hires = gauss_newton_regularized(forward_hires, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                       gamma=BASE_GAMMA, sigma_abs=BASE_SIGMA_ABS, label=f"{label} hires")
    resid_hires = y_true - forward_hires(x_hires)
    resid_hires_rms = float(np.sqrt(np.mean(resid_hires ** 2)))

    eta_win = _eta_of(FPA, np.full(width, 512.0), rows_win.astype(float))
    true_win = als.xco2_ppm(eta_win * als.SLIT_HALF_KM)
    retrieved_ppm_coarse = prior_co2_ppm_bins * x_coarse
    retrieved_ppm_hires = prior_co2_ppm_bins * x_hires
    if G > 1:
        coarse_edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
        idx = np.searchsorted(coarse_edges, eta_win)
        coarse_win = retrieved_ppm_coarse[idx]
        hires_win = np.interp(eta_win, bin_centers, retrieved_ppm_hires)
    else:
        coarse_win = np.full(width, retrieved_ppm_coarse[0])
        hires_win = np.full(width, retrieved_ppm_hires[0])

    bias_coarse = coarse_win - true_win
    bias_hires = hires_win - true_win

    return dict(row_lo=row_lo, row_hi=row_hi, G=G, width=width,
               bin_centers=bin_centers, prior_co2_ppm_bins=prior_co2_ppm_bins,
               x_coarse=x_coarse, x_hires=x_hires,
               resid_coarse=resid_coarse, resid_hires=resid_hires,
               rows_win=rows_win, true_win=true_win, coarse_win=coarse_win, hires_win=hires_win,
               bias_coarse_rms=float(np.sqrt(np.mean(bias_coarse ** 2))),
               bias_hires_rms=float(np.sqrt(np.mean(bias_hires ** 2))),
               bias_coarse_max=float(np.max(np.abs(bias_coarse))),
               bias_hires_max=float(np.max(np.abs(bias_hires))),
               max_dev_coarse=float(np.max(np.abs(x_coarse - 1))),
               max_dev_hires=float(np.max(np.abs(x_hires - 1))),
               resid_coarse_rms=resid_coarse_rms, resid_hires_rms=resid_hires_rms)


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
    print(f"{len(tasks)} (window, G) solves queued at the production baseline "
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
                     f"bias_hires_rms={r['bias_hires_rms']:.4f} resid_hires_rms={r['resid_hires_rms']:.2e}"
            print(f"  {n_done}/{len(tasks)} {key}: {status} ({time.time()-t0:.0f}s elapsed)", flush=True)

    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nall done ({time.time()-t0:.0f}s): {n_ok}/{len(tasks)} solves succeeded", flush=True)

    out_path = REPO_ROOT / "results" / f"gd_plume_bin_density_sweep_fpa{FPA}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "windows": WINDOWS, "fpa": FPA,
                    "base_gamma": BASE_GAMMA, "base_sigma_abs": BASE_SIGMA_ABS}, f)
    print(f"saved {out_path}")

    # ---- trend plot: bias rms & resid rms vs. G, per window ----
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(9, 3.6 * len(WINDOWS)))
    for k, (row_lo, row_hi, wlabel, g_values) in enumerate(WINDOWS):
        pts = [results[(row_lo, row_hi, G)] for G in sorted(g_values)
              if (row_lo, row_hi, G) in results and "error" not in results[(row_lo, row_hi, G)]]
        prod_G = max(2, int(round((row_hi - row_lo + 1) / G_RATIO)))

        ax = axes[k]
        ax2 = ax.twinx()
        ax.plot([r["G"] for r in pts], [r["bias_hires_rms"] for r in pts],
               "o-", color="tab:blue", label="hi-res bias rms [ppm]")
        ax.plot([r["G"] for r in pts], [r["bias_coarse_rms"] for r in pts],
               "o-", color="tab:orange", label="coarse bias rms [ppm]")
        ax2.plot([r["G"] for r in pts], [r["resid_hires_rms"] for r in pts],
                "s--", color="0.4", lw=1.0, ms=4, label="hi-res spectral resid rms")
        ax.axvline(prod_G, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("G (bin count)")
        ax.set_ylabel("bias rms [ppm]")
        ax2.set_ylabel("spectral resid rms", color="0.4")
        ax2.set_yscale("log")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel}, width={row_hi-row_lo+1}, production G={prod_G})", fontsize=10)
        if k == 0:
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper right")

    fig.suptitle(f"FPA{FPA}: real plume/hot-spot windows -- sensitivity to bin count G "
                f"(gamma={BASE_GAMMA:g}, sigma_abs={BASE_SIGMA_ABS:g} held at production baseline)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_plume_bin_density_sweep_fpa{FPA}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")

    # ---- profile plot: production G vs. the finest G tested, per window ----
    fig2, axes2 = plt.subplots(len(WINDOWS), 1, figsize=(11, 3.4 * len(WINDOWS)))
    if len(WINDOWS) == 1:
        axes2 = [axes2]
    for k, (row_lo, row_hi, wlabel, g_values) in enumerate(WINDOWS):
        prod_G = max(2, int(round((row_hi - row_lo + 1) / G_RATIO)))
        fine_G = max(g_values)
        base = results.get((row_lo, row_hi, prod_G))
        fine = results.get((row_lo, row_hi, fine_G))
        ax = axes2[k]
        if base is not None and "error" not in base:
            ax.plot(base["rows_win"], base["true_win"], color="black", lw=1.4, label="true", zorder=5)
            ax.plot(base["rows_win"], base["hires_win"], color="tab:blue", lw=1.1,
                   label=f"hi-res, G={prod_G} (production, rms={base['bias_hires_rms']:.3f})")
        if fine is not None and "error" not in fine:
            ax.plot(fine["rows_win"], fine["hires_win"], color="tab:red", lw=1.1, ls="--",
                   label=f"hi-res, G={fine_G} (finest tested, rms={fine['bias_hires_rms']:.3f})")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel})", fontsize=10.5)
        ax.set_xlabel("detector row"); ax.set_ylabel("CO2 [ppm]")
        ax.legend(fontsize=8, loc="best")

    fig2.suptitle(f"FPA{FPA}: real plume/hot-spot profiles, production G vs. finest G tested", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out_fig2 = plots_dir / f"gd_plume_bin_density_sweep_profiles_fpa{FPA}.png"
    fig2.savefig(out_fig2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out_fig2}")

    print("\nsummary:")
    print(f"{'window':>12s} {'G':>4s} {'bias_hires_rms':>15s} {'bias_coarse_rms':>16s} {'resid_hires_rms':>16s}")
    for row_lo, row_hi, wlabel, g_values in WINDOWS:
        for G in sorted(g_values):
            r = results.get((row_lo, row_hi, G))
            if r is None or "error" in r:
                continue
            print(f"{row_lo}-{row_hi:>6d} {G:4d} {r['bias_hires_rms']:15.4f} {r['bias_coarse_rms']:16.4f} {r['resid_hires_rms']:16.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
