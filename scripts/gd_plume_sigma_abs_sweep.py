#!/usr/bin/env python3
"""Direct test of the sigma_abs/gamma regularization-shrinkage hypothesis
(docs/JOINT_BLOCK_MIGRATION_PLAN.md Sec.9) on the REAL plume-region windows
from the production whole-slit sweep (gd_joint_block_whole_slit_sweep.py),
not the controlled synthetic ramp. The ramp test showed sigma_abs=0.10 is
measurably too tight for a shared-prior linear ramp with a ~10 ppm swing;
this script checks whether the same lever moves the real plume/hot-spot
bias, using the real realistic-scene detector image and the production
per-bin "local truth" priors, not an idealized shared prior.

Windows tested (identified from results/gd_joint_block_whole_slit_fpa2.pkl,
the gamma=3.0/sigma_abs=0.10 production baseline already on disk):
  - rows 331-345 (G=5): the broad co-located CO2/CO plume's own peak,
    x_km [-510,-472], true CO2 422.3-423.0 ppm -- a smoothly-varying
    feature, closest in character to the controlled ramp test.
  - rows 884-924 (G=14): the original single-hot-spot ATBD test window
    (row 890-935 in the whole-slit sweep's own docstring), CO2 hot spot
    at x_km=+1050.
  - rows 108-116 (G=3): the worst hi-res bias anywhere in the whole-slit
    sweep (-0.655 ppm at row 110), CO2 hot spot at x_km=-1100 -- the
    narrowest window (only 3 bins) over the sharpest along-slit feature.

Important asymmetry versus the ramp test, worth stating up front: the
production sweep's priors are already POSITION-DEPENDENT "local truth"
(`als.atmosphere_at(x_km)` at each bin's own center), not one shared
`atm_center` prior for the whole window. So a perfect retrieval does not
need x to move away from 1 to match truth AT the bin centers -- only the
piecewise-linear INTERPOLATION between bin centers needs to represent
whatever curvature lies between them. Checked directly against the
baseline pkl: x_hires sits within 0.0003 of 1.0 at every one of these
windows even where the ppm bias is largest (0.66 ppm) -- so unlike the
ramp test, this is not obviously a "prior refuses to let x move" story.
This sweep is exactly what settles it: if loosening sigma_abs lets x move
further from 1 and the bias shrinks, the same shrinkage mechanism found
in the ramp test transfers here too; if x barely moves regardless, the
real bias here is dominated by something else (a genuine sub-bin/sub-PSF
resolution ceiling, most likely) and sigma_abs is not the lever for it.

Sweep design mirrors gd_ramp_self_consistency_test.py's isolation
methodology: vary one regularization constant at a time, holding the
other at the production default (gamma=3.0, sigma_abs=0.10), rather than
a full 2D grid.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plume_sigma_abs_sweep.py \\
        [--n-workers N] [--gamma-values 0.3,3.0,30.0] \\
        [--sigma-abs-values 0.05,0.1,0.3,1.0,3.0,10.0]
Output: results/gd_plume_sigma_abs_sweep_fpa2.pkl,
        plots/joint_block/gd_plume_sigma_abs_sweep_fpa2.png,
        plots/joint_block/gd_plume_sigma_abs_sweep_profiles_fpa2.png
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

# (row_lo, row_hi, short label) -- see module docstring for how each was picked.
WINDOWS = [
    (331, 345, "broad plume peak"),
    (884, 924, "east hot spot (ATBD 890-935 ref)"),
    (108, 116, "west hot spot (worst-case)"),
]

_CTX = {}


def _solve_task(task):
    row_lo, row_hi, gamma, sigma_abs = task
    band, absco, wide_inst, geo, solar, albedo = (
        _CTX["band"], _CTX["absco"], _CTX["wide_inst"], _CTX["geo"], _CTX["solar"], _CTX["albedo"])
    wn_hires, ils = _CTX["wn_hires"], _CTX["ils"]

    rows_win = np.arange(row_lo, row_hi + 1)
    width = len(rows_win)
    G = max(2, int(round(width / G_RATIO)))
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

    label = f"[{row_lo}-{row_hi} g{gamma:g} s{sigma_abs:g}]"
    forward_coarse = build_forward(FPA, rows_win, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=PAD)
    x_coarse = gauss_newton_regularized(forward_coarse, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                        gamma=gamma, sigma_abs=sigma_abs, label=f"{label} coarse")
    resid_coarse = y_true - forward_coarse(x_coarse)
    resid_coarse_rms = float(np.sqrt(np.mean(resid_coarse ** 2)))

    anchor_rows = np.arange(max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD) + 1)
    forward_hires, anchor_etas = build_forward_hires(FPA, rows_win, anchor_rows, bin_centers,
                                                      spectrum_for, wn_hires, ils, pad=PAD, atm_center=None)
    x_hires = gauss_newton_regularized(forward_hires, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                       gamma=gamma, sigma_abs=sigma_abs, label=f"{label} hires")
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

    return dict(row_lo=row_lo, row_hi=row_hi, gamma=gamma, sigma_abs=sigma_abs, G=G, width=width,
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
    row_lo, row_hi, gamma, sigma_abs = task
    try:
        return _solve_task(task)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        return dict(row_lo=row_lo, row_hi=row_hi, gamma=gamma, sigma_abs=sigma_abs,
                   error=f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma-values", type=str, default="0.3,3.0,30.0")
    ap.add_argument("--sigma-abs-values", type=str, default="0.05,0.1,0.3,1.0,3.0,10.0")
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()
    gamma_values = sorted({float(v) for v in args.gamma_values.split(",")})
    sigma_abs_values = sorted({float(v) for v in args.sigma_abs_values.split(",")})

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

    # tasks: gamma sweep at sigma_abs=BASE_SIGMA_ABS, sigma_abs sweep at gamma=BASE_GAMMA,
    # deduplicated so the shared (BASE_GAMMA, BASE_SIGMA_ABS) baseline is solved once.
    tasks = set()
    for row_lo, row_hi, _ in WINDOWS:
        for g in gamma_values:
            tasks.add((row_lo, row_hi, g, BASE_SIGMA_ABS))
        for s in sigma_abs_values:
            tasks.add((row_lo, row_hi, BASE_GAMMA, s))
    tasks = sorted(tasks)
    print(f"{len(tasks)} (window, gamma, sigma_abs) solves queued "
         f"({len(WINDOWS)} windows x ({len(gamma_values)} gamma + {len(sigma_abs_values)} sigma_abs - 1 shared baseline))",
         flush=True)

    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    print(f"solving with {n_workers} workers...", flush=True)

    t0 = time.time()
    results = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for r in pool.imap_unordered(_worker, tasks, chunksize=1):
            key = (r["row_lo"], r["row_hi"], r["gamma"], r["sigma_abs"])
            results[key] = r
            n_done += 1
            status = "ERROR: " + r["error"] if "error" in r else \
                     f"bias_hires_rms={r['bias_hires_rms']:.4f} max_dev_hires={r['max_dev_hires']:.4f}"
            print(f"  {n_done}/{len(tasks)} {key}: {status} ({time.time()-t0:.0f}s elapsed)", flush=True)

    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nall done ({time.time()-t0:.0f}s): {n_ok}/{len(tasks)} solves succeeded", flush=True)

    out_path = REPO_ROOT / "results" / f"gd_plume_sigma_abs_sweep_fpa{FPA}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "windows": WINDOWS, "fpa": FPA,
                    "gamma_values": gamma_values, "sigma_abs_values": sigma_abs_values,
                    "base_gamma": BASE_GAMMA, "base_sigma_abs": BASE_SIGMA_ABS}, f)
    print(f"saved {out_path}")

    # ---- trend plot: bias rms & max|x-1| vs sigma_abs (gamma fixed) and vs gamma (sigma_abs fixed) ----
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(len(WINDOWS), 2, figsize=(13, 3.6 * len(WINDOWS)))
    for k, (row_lo, row_hi, wlabel) in enumerate(WINDOWS):
        s_pts = [results[(row_lo, row_hi, BASE_GAMMA, s)] for s in sigma_abs_values
                if (row_lo, row_hi, BASE_GAMMA, s) in results and "error" not in results[(row_lo, row_hi, BASE_GAMMA, s)]]
        g_pts = [results[(row_lo, row_hi, g, BASE_SIGMA_ABS)] for g in gamma_values
                if (row_lo, row_hi, g, BASE_SIGMA_ABS) in results and "error" not in results[(row_lo, row_hi, g, BASE_SIGMA_ABS)]]

        ax = axes[k, 0]
        ax2 = ax.twinx()
        ax.plot([r["sigma_abs"] for r in s_pts], [r["bias_hires_rms"] for r in s_pts],
               "o-", color="tab:blue", label="hi-res bias rms")
        ax.plot([r["sigma_abs"] for r in s_pts], [r["bias_coarse_rms"] for r in s_pts],
               "o-", color="tab:orange", label="coarse bias rms")
        ax2.plot([r["sigma_abs"] for r in s_pts], [r["max_dev_hires"] for r in s_pts],
                "s--", color="0.4", lw=1.0, ms=4, label="max|x_hires-1|")
        ax.set_xscale("log")
        ax.axvline(BASE_SIGMA_ABS, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("sigma_abs (gamma fixed at %.1f)" % BASE_GAMMA)
        ax.set_ylabel("bias rms [ppm]")
        ax2.set_ylabel("max|x-1|", color="0.4")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel}): vs. sigma_abs", fontsize=10)
        if k == 0:
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="upper right")

        ax = axes[k, 1]
        ax2 = ax.twinx()
        ax.plot([r["gamma"] for r in g_pts], [r["bias_hires_rms"] for r in g_pts],
               "o-", color="tab:blue", label="hi-res bias rms")
        ax.plot([r["gamma"] for r in g_pts], [r["bias_coarse_rms"] for r in g_pts],
               "o-", color="tab:orange", label="coarse bias rms")
        ax2.plot([r["gamma"] for r in g_pts], [r["max_dev_hires"] for r in g_pts],
                "s--", color="0.4", lw=1.0, ms=4, label="max|x_hires-1|")
        ax.set_xscale("log")
        ax.axvline(BASE_GAMMA, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("gamma (sigma_abs fixed at %.2f)" % BASE_SIGMA_ABS)
        ax.set_ylabel("bias rms [ppm]")
        ax2.set_ylabel("max|x-1|", color="0.4")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel}): vs. gamma", fontsize=10)

    fig.suptitle(f"FPA{FPA}: real plume/hot-spot windows -- sensitivity to gamma/sigma_abs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_plume_sigma_abs_sweep_fpa{FPA}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")

    # ---- profile plot: baseline vs. the loosest sigma_abs tested, per window ----
    sigma_abs_loose = max(sigma_abs_values)
    fig2, axes2 = plt.subplots(len(WINDOWS), 1, figsize=(11, 3.4 * len(WINDOWS)))
    if len(WINDOWS) == 1:
        axes2 = [axes2]
    for k, (row_lo, row_hi, wlabel) in enumerate(WINDOWS):
        base = results.get((row_lo, row_hi, BASE_GAMMA, BASE_SIGMA_ABS))
        loose = results.get((row_lo, row_hi, BASE_GAMMA, sigma_abs_loose))
        ax = axes2[k]
        if base is not None and "error" not in base:
            ax.plot(base["rows_win"], base["true_win"], color="black", lw=1.4, label="true", zorder=5)
            ax.plot(base["rows_win"], base["hires_win"], color="tab:blue", lw=1.1,
                   label=f"hi-res, sigma_abs={BASE_SIGMA_ABS:g} (rms={base['bias_hires_rms']:.3f})")
        if loose is not None and "error" not in loose:
            ax.plot(loose["rows_win"], loose["hires_win"], color="tab:red", lw=1.1, ls="--",
                   label=f"hi-res, sigma_abs={sigma_abs_loose:g} (rms={loose['bias_hires_rms']:.3f})")
        ax.set_title(f"rows {row_lo}-{row_hi} ({wlabel})", fontsize=10.5)
        ax.set_xlabel("detector row"); ax.set_ylabel("CO2 [ppm]")
        ax.legend(fontsize=8, loc="best")

    fig2.suptitle(f"FPA{FPA}: real plume/hot-spot profiles, tight vs. loosened sigma_abs", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out_fig2 = plots_dir / f"gd_plume_sigma_abs_sweep_profiles_fpa{FPA}.png"
    fig2.savefig(out_fig2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out_fig2}")

    print("\nsummary (sigma_abs sweep, gamma fixed at %.1f):" % BASE_GAMMA)
    print(f"{'window':>12s} {'sigma_abs':>10s} {'bias_hires_rms':>15s} {'bias_coarse_rms':>16s} {'max|x_hires-1|':>15s}")
    for row_lo, row_hi, wlabel in WINDOWS:
        for s in sigma_abs_values:
            r = results.get((row_lo, row_hi, BASE_GAMMA, s))
            if r is None or "error" in r:
                continue
            print(f"{row_lo}-{row_hi:>6d} {s:10.3f} {r['bias_hires_rms']:15.4f} {r['bias_coarse_rms']:16.4f} {r['max_dev_hires']:15.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
