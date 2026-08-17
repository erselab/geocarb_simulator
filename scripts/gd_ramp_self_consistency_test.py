#!/usr/bin/env python3
"""Controlled linear-ramp self-consistency test for the joint block's own
machinery, at several different local keystone amplitudes -- the next
debugging step after §8 of docs/JOINT_BLOCK_MIGRATION_PLAN.md (the ILS-
convolution and obs_grid findings). The uniform (constant-atmosphere)
self-consistency test already showed the joint block's own machinery is
exactly correct on a field with zero curvature; the still-open plume-
region anomaly (scripts/gd_joint_block_whole_slit_sweep.py) only appears
with real curvature. This isolates whether *keystone amplitude itself* is
implicated, using a scene simple enough to have an exactly-known correct
answer, rather than the full realistic scene's many simultaneous features.

Scene: CO2 varies exactly linearly in eta (co2_ppm(eta) = co2_ref +
slope*eta); every other gas/pressure held fixed at atm_center (the same
uniform-nuisance idealization the joint block's own local-truth priors
already use elsewhere in this investigation, isolating CO2-ramp recovery
specifically). Self-consistent by construction: truth (predict_neighborhood,
real per-pixel keystone) and the retrieval's own forward model
(build_forward / build_forward_hires) are rendered by the identical code
path, exactly as the uniform test was -- so any deviation from the known-
correct answer is unambiguous, not a reference-tool artifact (see §8).

Two sharp, exactly-known expectations, not just "small residual expected":
- hi-res (piecewise-linear interpolation of the retrieval's own G-dim
  state onto G_eff native-row anchors): a linear function is its own exact
  piecewise-linear interpolant, so hi-res should recover the ramp to
  GN-convergence-tolerance-level noise, regardless of G or keystone level,
  if nothing else is wrong.
- coarse (nearest-bin / step-function attribution): NOT expected to be
  zero -- a step function fit to a slope has a calculable sawtooth bias,
  zero at each bin center and +/- slope*(bin_width/2) at bin edges. Also
  computed and compared directly, since "coarse should have a nonzero but
  *predictable* bias" is itself a testable claim.

Test windows are sized by the same adaptive formula
(scripts/gd_joint_block_whole_slit_sweep.py's own radius = max(MIN_WINDOW,
round(2.2*rows_crossed))) at several centers spanning low to high local
keystone, so each window is appropriately scaled to its own local keystone
the same way the production sweep is -- isolating keystone LEVEL as the
one swept variable, not window/bin sizing convention.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_ramp_self_consistency_test.py \\
        [--n-bins 15] [--slope-ppm 20.0] [--co2-ref-ppm 413.0]
Output: plots/joint_block/gd_ramp_self_consistency_test_fpa2.png
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
from gd_joint_block_hires_test import build_forward_hires  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
from gd_joint_block_whole_slit_sweep import MIN_WINDOW, G_RATIO, PAD  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert import gd_render  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed  # noqa: E402

ROW_MAX_IDX = 1023
# Row centers spanning low (near the row-25 keystone null) to high
# (near the slit edges) local keystone amplitude.
TEST_CENTERS = [25, 275, 525, 775, 1000]


def window_for_center(fpa: int, center: int, min_window: int = MIN_WINDOW) -> tuple[int, int]:
    """Same fixed-point radius formula as build_window_tiles, evaluated at
    one arbitrary center instead of tiling the whole slit."""
    r = min_window
    for _ in range(6):
        c = min(max(center, 0), ROW_MAX_IDX)
        k_c = float(rows_crossed(fpa, np.array([float(c)]))[0])
        r_new = max(min_window, int(round(2.2 * k_c)))
        if r_new == r:
            break
        r = r_new
    row_lo = max(0, center - r)
    row_hi = min(ROW_MAX_IDX, center + r)
    return row_lo, row_hi


def make_ramp_radiance(spectrum_for, atm_center, co2_ref_ppm: float, slope_ppm: float,
                       eta_center: float, eta_lo: float, eta_hi: float, n_samples: int = 41):
    """radiance(eta) -> hi-res spectrum for the known linear ramp
    co2_ppm(eta) = co2_ref_ppm + slope_ppm*(eta - eta_center) -- anchored at
    the WINDOW's own centre, not globally at eta=0. Anchoring globally
    while scaling slope_ppm up for a narrow, far-from-zero window (see
    --swing-ppm) implies an absurd, unphysical CO2 value back at eta=0
    (found by hand-checking: a window near eta=-0.95 with a slope scaled
    for its own tiny span implied ~-200 ppm CO2 at eta=0) -- anchoring
    locally keeps every sampled CO2 value within swing_ppm/2 of co2_ref_ppm
    by construction, regardless of how far the window sits from eta=0 or
    how steep the locally-required slope is.

    Precomputes n_samples real RT evaluations (one per co2_scale sample,
    via the same spectrum_for every joint-block script already uses)
    across [eta_lo, eta_hi], then linearly interpolates the PRECOMPUTED
    HI-RES SPECTRA between neighboring samples for any query eta -- the
    same dense-lookup convention als.build_lookup_radiance already uses
    for truth generation everywhere else in this investigation (not the
    retrieval's own state-space-only rule, which is specifically about
    not blending the RETRIEVAL's widely-spaced bin states; here the
    samples are dense enough, relative to the window, for the
    interpolation error to be negligible against what's being tested)."""
    eta_samples = np.linspace(eta_lo, eta_hi, n_samples)
    co2_scale_samples = (co2_ref_ppm + slope_ppm * (eta_samples - eta_center)) / co2_ref_ppm
    spectra = np.stack([spectrum_for(float(s), atm_center) for s in co2_scale_samples])

    def radiance(eta):
        eta = np.atleast_1d(np.asarray(eta, dtype=float))
        idx_hi = np.clip(np.searchsorted(eta_samples, eta), 1, n_samples - 1)
        idx_lo = idx_hi - 1
        e_lo, e_hi = eta_samples[idx_lo], eta_samples[idx_hi]
        frac = np.clip((eta - e_lo) / (e_hi - e_lo), 0.0, 1.0)
        return spectra[idx_lo] * (1 - frac)[:, None] + spectra[idx_hi] * frac[:, None]

    return radiance


def coarse_sawtooth_prediction(eta_profile, bin_centers, slope_ppm):
    """Closed-form expected coarse bias for a linear ramp under nearest-
    bin (step-function) attribution: within each bin, the step sits at
    the bin CENTER's own true value, so bias(eta) = slope*(eta - center)
    for eta in that bin -- zero at the center, growing linearly to the
    bin edges. Compared directly against the actual retrieved coarse bias
    as a second, sharp, non-zero expectation (not just "some quantization
    expected"). Note this formula is independent of the ramp's own
    eta_center anchor (only differences in eta enter), so no change
    needed there when the ramp's anchor moved to the window's own center."""
    edges = 0.5 * (bin_centers[:-1] + bin_centers[1:]) if len(bin_centers) > 1 else np.array([])
    idx = np.searchsorted(edges, eta_profile)
    nearest_center = bin_centers[idx]
    return slope_ppm * (eta_profile - nearest_center)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bins", type=int, default=None, help="override G "
                    "(default: adaptive, width/G_RATIO, matching production)")
    ap.add_argument("--swing-ppm", type=float, default=10.0, help="total CO2 swing "
                    "[ppm] across EACH window's own eta span (not a fixed global "
                    "ppm-per-eta slope) -- keeps the local signal strength relative to "
                    "the regularization comparable across very different window widths, "
                    "matching how a real hot-spot/plume feature would present at "
                    "different keystone levels. A fixed global slope instead makes "
                    "low-keystone (hence narrow) windows see an unrealistically tiny "
                    "local swing, easily swamped by gamma/sigma_abs tuned for real "
                    "several-ppm features -- found by hand-checking a first attempt.")
    ap.add_argument("--co2-ref-ppm", type=float, default=None, help="ramp baseline "
                    "[ppm] -- default: atm_center's own true CO2 value, so the ramp's "
                    "co2_scale-vs-ppm conversion is EXACTLY consistent with spectrum_for's "
                    "own scaling convention (co2_scale=1 means exactly atm_center's CO2). "
                    "Overriding this to a different constant reintroduces a small but real "
                    "baseline/slope mismatch between what's rendered and what's scored against.")
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--sigma-abs", type=float, default=0.10)
    args = ap.parse_args()

    print(f"Building FPA{FPA} band basics (real geometry/ILS, no realistic-scene truth needed)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    wn_hires, ils = wide_win.wn_hires, wide_win.ils
    if args.co2_ref_ppm is None:
        args.co2_ref_ppm = float(np.mean(atm_center.gases["co2"])) * 1e6
    print(f"ramp baseline (atm_center's own CO2): {args.co2_ref_ppm:.4f} ppm", flush=True)
    print("done.\n", flush=True)

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(len(TEST_CENTERS), 2, figsize=(13.5, 3.4 * len(TEST_CENTERS)))
    summary = []

    print(f"{'center':>6s} {'row_lo':>6s} {'row_hi':>6s} {'width':>5s} {'G':>3s} "
         f"{'rows_crossed':>12s} {'coarse_rms':>10s} {'coarse_vs_pred':>14s} "
         f"{'hires_rms':>10s} {'hires_max':>10s}", flush=True)

    for k, center in enumerate(TEST_CENTERS):
        row_lo, row_hi = window_for_center(FPA, center)
        rows_win = np.arange(row_lo, row_hi + 1)
        width = len(rows_win)
        G = args.n_bins if args.n_bins is not None else max(2, int(round(width / G_RATIO)))
        cols = np.arange(1024.0)
        eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
        bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
        rc_center = float(rows_crossed(FPA, np.array([float(center)]))[0])

        eta_lo, eta_hi = float(eta_all.min()), float(eta_all.max())
        eta_center = 0.5 * (eta_lo + eta_hi)
        # Local slope, not a fixed global one -- see --swing-ppm's own help text.
        slope_ppm = args.swing_ppm / (eta_hi - eta_lo) if eta_hi > eta_lo else 0.0
        # The ramp's own lookup table must cover whatever eta predict_neighborhood
        # actually queries, which includes PAD rows on each side for PSF-blur
        # edge handling -- NOT just the window's own rows. Missing this clamps
        # the padding rows to the boundary sample instead of continuing the
        # ramp, corrupting the PSF-blurred truth for the whole window (found
        # by hand-checking: the padded row range's own eta extent was ~10x
        # wider than a naive 5%-of-window-span margin covered).
        pad_row_lo = max(0, row_lo - PAD)
        pad_row_hi = min(ROW_MAX_IDX, row_hi + PAD)
        eta_padded = np.stack([_eta_of(FPA, cols, np.full(1024, float(i)))
                               for i in range(pad_row_lo, pad_row_hi + 1)])
        pad_eta_lo, pad_eta_hi = float(eta_padded.min()), float(eta_padded.max())
        margin = 0.05 * (eta_hi - eta_lo) if eta_hi > eta_lo else 0.01
        sample_lo = min(eta_lo, pad_eta_lo) - margin
        sample_hi = max(eta_hi, pad_eta_hi) + margin
        radiance_ramp = make_ramp_radiance(spectrum_for, atm_center, args.co2_ref_ppm, slope_ppm,
                                           eta_center, sample_lo, sample_hi, n_samples=61)

        t0 = time.time()
        y_true = gd_render.predict_neighborhood(FPA, rows_win, wn_hires, radiance_ramp, ils, pad=PAD).ravel()

        prior_atms = [atm_center] * G
        # x is a co2_scale FACTOR relative to prior_atms[g]'s own CO2 (atm_center,
        # a single constant here, since prior_atms is the same object for every
        # bin) -- NOT relative to the ramp's own per-bin true value. Converting
        # x back to ppm must use that one constant reference, not a per-bin one.
        atm_co2_ppm = float(np.mean(atm_center.gases["co2"])) * 1e6
        y_scale = float(np.mean(np.abs(y_true)))
        Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

        forward_coarse = build_forward(FPA, rows_win, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=PAD)
        x_coarse = gauss_newton_regularized(forward_coarse, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                            gamma=args.gamma, sigma_abs=args.sigma_abs,
                                            label=f"[c{center}] coarse")

        anchor_rows = np.arange(max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD) + 1)
        forward_hires, anchor_etas = build_forward_hires(FPA, rows_win, anchor_rows, bin_centers,
                                                          spectrum_for, wn_hires, ils, pad=PAD,
                                                          atm_center=atm_center)
        x_hires = gauss_newton_regularized(forward_hires, y_true, x0=np.ones(G), Sy_inv_diag=Sy_inv_diag,
                                           gamma=args.gamma, sigma_abs=args.sigma_abs,
                                           label=f"[c{center}] hires")
        elapsed = time.time() - t0

        retrieved_ppm_coarse = atm_co2_ppm * x_coarse
        retrieved_ppm_hires = atm_co2_ppm * x_hires

        eta_profile = np.linspace(eta_lo, eta_hi, 400)
        true_profile = args.co2_ref_ppm + slope_ppm * (eta_profile - eta_center)
        coarse_edges = 0.5 * (bin_centers[:-1] + bin_centers[1:]) if len(bin_centers) > 1 else np.array([])
        coarse_idx = np.searchsorted(coarse_edges, eta_profile)
        coarse_profile = retrieved_ppm_coarse[coarse_idx]
        hires_profile = np.interp(eta_profile, bin_centers, retrieved_ppm_hires)

        bias_coarse = coarse_profile - true_profile
        bias_hires = hires_profile - true_profile
        pred_coarse = coarse_sawtooth_prediction(eta_profile, bin_centers, slope_ppm)
        coarse_vs_pred_rms = float(np.sqrt(np.mean((bias_coarse - pred_coarse) ** 2)))

        rms_coarse = float(np.sqrt(np.mean(bias_coarse ** 2)))
        rms_hires = float(np.sqrt(np.mean(bias_hires ** 2)))
        max_hires = float(np.max(np.abs(bias_hires)))
        summary.append(dict(center=center, row_lo=row_lo, row_hi=row_hi, width=width, G=G,
                            rows_crossed=rc_center, rms_coarse=rms_coarse,
                            coarse_vs_pred_rms=coarse_vs_pred_rms, rms_hires=rms_hires, max_hires=max_hires))
        print(f"{center:6d} {row_lo:6d} {row_hi:6d} {width:5d} {G:3d} {rc_center:12.4f} "
             f"{rms_coarse:10.4f} {coarse_vs_pred_rms:14.6f} {rms_hires:10.6f} {max_hires:10.6f} "
             f"({elapsed:.0f}s)", flush=True)

        ax = axes[k, 0]
        ax.plot(eta_profile, true_profile, color="black", lw=1.6, label="true ramp", zorder=5)
        ax.plot(eta_profile, coarse_profile, color="tab:orange", lw=1.3, label="coarse posterior")
        ax.plot(eta_profile, hires_profile, color="tab:blue", lw=1.3, label="hi-res posterior")
        for bc in bin_centers:
            ax.axvline(bc, color="0.85", lw=0.4, zorder=0)
        ax.set_title(f"rows {row_lo}-{row_hi} (center={center}, rows_crossed={rc_center:.2f}, "
                    f"G={G}): profile", fontsize=10)
        ax.set_xlabel(r"$\eta$"); ax.set_ylabel("CO2 [ppm]")
        if k == 0:
            ax.legend(fontsize=7.5, loc="upper left")

        ax = axes[k, 1]
        ax.axhline(0, color="black", lw=0.6)
        ax.plot(eta_profile, bias_coarse, color="tab:orange", lw=1.2,
               label=f"coarse (rms={rms_coarse:.4f})")
        ax.plot(eta_profile, pred_coarse, color="tab:orange", lw=0.8, ls=":",
               label=f"coarse, predicted sawtooth (match rms={coarse_vs_pred_rms:.4f})")
        ax.plot(eta_profile, bias_hires, color="tab:blue", lw=1.2,
               label=f"hi-res (rms={rms_hires:.4f}, max={max_hires:.4f})")
        ax.set_title(f"rows {row_lo}-{row_hi}: bias vs. known-exact ramp", fontsize=10)
        ax.set_xlabel(r"$\eta$"); ax.set_ylabel("bias [ppm]")
        ax.legend(fontsize=7, loc="upper left")

    print(f"\nsummary: hi-res rms bias should stay near-zero (GN-tolerance noise) at every "
         f"keystone level if nothing else is wrong; coarse rms should track the predicted "
         f"sawtooth closely (small coarse_vs_pred_rms) at every level too.")

    fig.suptitle(f"FPA{FPA}: linear-ramp self-consistency test across keystone levels "
                f"(local swing={args.swing_ppm:g} ppm across each window)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_ramp_self_consistency_test_fpa{FPA}_gamma{args.gamma:g}_sigmaabs{args.sigma_abs:g}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")

    # ---- trend plot: bias vs. keystone level (rows_crossed) ----
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    rc_vals = [s["rows_crossed"] for s in summary]
    ax.plot(rc_vals, [s["rms_hires"] for s in summary], "o-", color="tab:blue", label="hi-res rms bias")
    ax.plot(rc_vals, [s["rms_coarse"] for s in summary], "o-", color="tab:orange", label="coarse rms bias")
    ax.plot(rc_vals, [s["coarse_vs_pred_rms"] for s in summary], "o--", color="tab:red",
           label="coarse: actual vs. predicted sawtooth")
    ax.set_xlabel("rows_crossed at window center (local keystone amplitude)")
    ax.set_ylabel("RMS bias [ppm]")
    ax.set_title(f"FPA{FPA}: ramp-recovery error vs. local keystone amplitude", fontsize=12)
    ax.legend(fontsize=9)
    fig2.tight_layout()
    out_fig2 = plots_dir / f"gd_ramp_self_consistency_trend_fpa{FPA}_gamma{args.gamma:g}_sigmaabs{args.sigma_abs:g}.png"
    fig2.savefig(out_fig2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out_fig2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
