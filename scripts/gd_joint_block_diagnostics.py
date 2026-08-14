#!/usr/bin/env python3
"""Diagnostics for the joint multi-atmosphere block, requested directly
after the first realistic result (`JOINT_ROW_INVERSION_PLAN.md` §4,
"Phase 1-3 result"): (1) spectral-residual structure, the generalizable
goodness-of-fit diagnostic that survives once real (noisy, truth-unknown)
data replaces this synthetic scene, unlike the capture-fraction number,
which needs known truth; (2) a pixel-density-weighted bin placement,
replacing the uniform `np.linspace` grid `gd_joint_block_retrieve.py`
used, so bin density (bins per unit eta) scales with the ACTUAL density
of real pixels in eta -- which is higher wherever keystone is large (more
rows' column ranges overlap a given eta interval there) -- instead of
being constant across the window. An earlier version of this used
`rows_crossed` as an indirect per-row proxy for that density; this
version places bins at quantiles of the real per-pixel eta distribution
directly, no proxy or floor needed.

Runs the *same* row 890-935 / G=15 case from the first result, once with
uniform bin spacing and once with pixel-density-weighted spacing, and
compares:
- the reshaped (rows x cols) residual image for each,
- per-bin reduced chi-square for each,
- where each scheme's bin centers land relative to rows_crossed(row),
- the resulting peak-enhancement capture fraction for each.

Reuses every piece of `gd_joint_block_retrieve.py` unchanged (band_basics,
make_spectrum_fn, build_forward, gauss_newton_regularized, _eta_of) --
this script only adds the two new pieces of analysis, no new retrieval
machinery.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_diagnostics.py
Output: plots/joint_block/gd_joint_block_diagnostics_fpa2_row890-935_G15.png
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

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed  # noqa: E402

def uniform_bin_centers(eta_lo, eta_hi, G):
    return np.linspace(eta_lo, eta_hi, G)


def pixel_density_bin_centers(eta_flat, G):
    """Bin centers placed at evenly-spaced quantiles of the ACTUAL
    per-pixel eta distribution in the window (`eta_flat` = every real
    (row, col) pixel's own true eta, not a per-row proxy). Supersedes an
    earlier version of this function that used `rows_crossed` as an
    indirect per-row density proxy, routed through each row's own
    center-column eta -- that required an extra floor to handle a near-
    null row's real internal wiggle (rows_crossed's endpoint-difference
    definition reads ~0 there despite real spread, §0's finding), and
    only approximated what's already directly available: wherever
    keystone is large, more rows' column ranges overlap a given eta
    interval, so real pixel DENSITY there is genuinely higher -- no proxy
    or floor needed, `np.quantile` on the real data does this directly."""
    return np.quantile(eta_flat, np.linspace(0.0, 1.0, G))


def bin_assign(bin_centers, eta_flat):
    """Same nearest-bin logic as geocarb_gert.nearest_bin_scene, exposed
    here to also assign real pixels to bins for the per-bin chi2 below."""
    edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
    return np.searchsorted(edges, eta_flat)


def per_bin_chi2(resid, Sy_inv_diag, bin_idx, G):
    chi2 = np.zeros(G)
    counts = np.zeros(G, dtype=int)
    contrib = resid ** 2 * Sy_inv_diag
    for g in range(G):
        mask = bin_idx == g
        counts[g] = int(mask.sum())
        if counts[g] > 0:
            chi2[g] = float(contrib[mask].sum() / counts[g])
    return chi2, counts


def run_scheme(fpa, rows_win, bin_centers, band, absco, wide_inst, geo, solar, albedo,
               y_true, Sy_inv_diag, wn_hires, ils, gamma, sigma_abs, label):
    G = len(bin_centers)
    x_km_bins = bin_centers * als.SLIT_HALF_KM
    prior_atms = [als.atmosphere_at(float(xk)) for xk in x_km_bins]
    prior_co2_ppm_bins = np.array([float(als.xco2_ppm(xk)) for xk in x_km_bins])
    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    forward = build_forward(fpa, rows_win, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=4)

    t0 = time.time()
    x = gauss_newton_regularized(forward, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                 gamma=gamma, sigma_abs=sigma_abs, label=label)
    elapsed = time.time() - t0

    resid = y_true - forward(x)
    resid_img = resid.reshape(len(rows_win), 1024)
    retrieved_ppm = prior_co2_ppm_bins * x
    return dict(bin_centers=bin_centers, x=x, retrieved_ppm=retrieved_ppm,
               prior_co2_ppm_bins=prior_co2_ppm_bins, resid=resid, resid_img=resid_img,
               elapsed=elapsed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-min", type=int, default=890)
    ap.add_argument("--row-max", type=int, default=935)
    ap.add_argument("--n-bins", type=int, default=15)
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
    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    eta_lo, eta_hi = float(eta_all.min()), float(eta_all.max())

    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

    true_peak = float(band["xtrue_of_row"]["co2"][args.peak_row])
    true_bg = float(band["xtrue_of_row"]["co2"][args.bg_row])

    schemes = {
        "uniform": uniform_bin_centers(eta_lo, eta_hi, args.n_bins),
        "pixel-density": pixel_density_bin_centers(eta_all.ravel(), args.n_bins),
    }

    results = {}
    for name, bin_centers in schemes.items():
        print(f"\n=== {name}, G={args.n_bins} ===", flush=True)
        r = run_scheme(FPA, rows_win, bin_centers, band, absco, wide_inst, geo, solar, albedo,
                       y_true, Sy_inv_diag, wn_hires, ils, args.gamma, args.sigma_abs, name)
        bin_idx = bin_assign(bin_centers, eta_all.ravel())
        chi2, counts = per_bin_chi2(r["resid"], Sy_inv_diag, bin_idx, args.n_bins)
        r["chi2"], r["counts"] = chi2, counts

        def bin_at_row(row, bc=bin_centers):
            eta_c = float(_eta_of(FPA, np.array([512.0]), np.array([float(row)]))[0])
            return int(np.argmin(np.abs(bc - eta_c)))

        g_peak, g_bg = bin_at_row(args.peak_row), bin_at_row(args.bg_row)
        capture = ((r["retrieved_ppm"][g_peak] - r["retrieved_ppm"][g_bg])
                  / (true_peak - true_bg))
        r["capture"] = capture
        r["g_peak"], r["g_bg"] = g_peak, g_bg
        print(f"  ({r['elapsed']:.0f}s)  peak-enhancement captured: {capture*100:.1f}%  "
             f"mean per-bin chi2={chi2.mean():.4g}  min pixel count/bin={counts.min()}", flush=True)
        results[name] = r

    # -- figure --
    rows_fine = np.arange(args.row_min, args.row_max + 1)
    rc_fine = rows_crossed(FPA, rows_fine.astype(float))
    eta_fine = _eta_of(FPA, np.full(len(rows_fine), 512.0), rows_fine.astype(float))

    vmax = max(np.abs(results["uniform"]["resid_img"]).max(),
              np.abs(results["pixel-density"]["resid_img"]).max())

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.6, 1.6], hspace=0.4, wspace=0.3)

    for col, name in enumerate(["uniform", "pixel-density"]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(results[name]["resid_img"], aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[0, 1024, args.row_max, args.row_min])
        ax.set_title(f"{name} (G={args.n_bins}): residual image\n"
                    f"capture={results[name]['capture']*100:.1f}%, "
                    f"mean bin chi2={results[name]['chi2'].mean():.3g}", fontsize=10)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="resid [radiance units]")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(rows_fine, rc_fine, "-", color="black", lw=1.2, label="rows_crossed(row)")
    ax.set_ylabel("rows_crossed")
    ax.set_xlabel("row")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])
    for name, marker, color in [("uniform", "o", "tab:orange"), ("pixel-density", "^", "tab:blue")]:
        bc = results[name]["bin_centers"]
        rows_of_bins = np.interp(bc, eta_fine, rows_fine)
        ax.plot(rows_of_bins, np.interp(rows_of_bins, rows_fine, rc_fine), marker,
               color=color, ms=7, label=f"{name} bin centers", alpha=0.8)
    ax.axvline(args.peak_row, color="gold", lw=1, ls="--", label=f"peak row {args.peak_row}")
    ax.axvline(args.bg_row, color="gray", lw=1, ls="--", label=f"bg row {args.bg_row}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("bin center placement vs. local keystone amplitude", fontsize=10)

    ax = fig.add_subplot(gs[2, :])
    for name, color in [("uniform", "tab:orange"), ("pixel-density", "tab:blue")]:
        bc = results[name]["bin_centers"]
        ax.plot(bc, results[name]["chi2"], "o-", color=color, ms=4, label=f"{name}")
    ax.set_yscale("log")
    ax.set_xlabel("bin center (eta)")
    ax.set_ylabel("per-bin reduced chi2 (log scale)")
    ax.set_title("per-bin goodness of fit", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"Joint block diagnostics -- FPA2 rows {args.row_min}-{args.row_max}, "
                f"G={args.n_bins}, gamma={args.gamma}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_joint_block_diagnostics_fpa2_row{args.row_min}-{args.row_max}_G{args.n_bins}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
