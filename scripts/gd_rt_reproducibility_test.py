#!/usr/bin/env python3
"""Follow-up unit test to gd_diagonal_ils_convolve_test.py: with the ILS
convolution kernel now proven bit-identical between "truth" and "retrieval"
(gd_render._diagonal_ils_convolve fixed to match gert.instrument.ILS.
convolve exactly), the fixed-nuisance uniform-scene, PSF-off residual
(~0.002 ppm) can no longer be attributed to convolution. This isolates the
next candidate: the RT step itself, since truth and the retrieval each run
their own, separately-instantiated ForwardModel call for what should be
the identical atmosphere/instrument physics.

Truth's own RT call (geocarb_gert.along_slit_scene._lookup_sample, the
function build_no_keystone_psf_image's band['radiance'] is actually built
from):
    ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(),
                solar_spectrum=solar).run(albedo=[alb], albedo_slope=[0.0])
    -- wide_inst wraps a SpectralWindow built with explicit hires_spacing/
    channels_per_fwhm, no obs_grid.

Retrieval's own RT call (scripts/gd_keystone_free_1d_ceiling.py's
_joint_retrieve_fixed_nuisance, reproduced here directly, bypassing
GERTRetrieval so the raw pre-convolution I_hires can be compared):
    ForwardModel(atm, absco, inst, geo, solver=SingleScatterSolver(),
                solar_spectrum=solar).run(albedo=[alb], albedo_slope=[0.0])
    -- inst wraps a SpectralWindow built with obs_grid=nu_row (a real row's
    own wavenumber grid), hires_spacing/channels_per_fwhm left at their
    defaults (confirmed equal to wide_win's explicit values).

wn_hires (the RT computation grid) depends only on wn_min/wn_max/
hires_spacing, all identical between the two windows, and obs_grid only
affects wn_instrument (the *output* grid), not wn_hires -- so if RT is
numerically deterministic given identical atmosphere/absco/geometry/
solver/grid, I_hires should come out bit-identical. This test checks that
directly, then convolves both raw hi-res spectra with the SAME (now-fixed)
_diagonal_ils_convolve at a couple of real rows, to see whether any RT-
level discrepancy survives convolution and could plausibly explain the
remaining ~0.002 ppm.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_rt_reproducibility_test.py
Output: plots/joint_block/gd_rt_reproducibility_test_fpa2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, geocarb_noise_model, sample_geometries  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import _diagonal_ils_convolve  # noqa: E402

from gert.forward_model import ForwardModel  # noqa: E402
from gert.instrument import ILS, SpectralWindow  # noqa: E402
from gert.instrument_config import Instrument  # noqa: E402
from gert.rt_solver import SingleScatterSolver  # noqa: E402

TEST_ROWS = [25, 512, 1000]


def line_depth(spec: np.ndarray) -> tuple[float, float, float]:
    continuum = float(spec.max())
    core = float(spec.min())
    return continuum, core, continuum - core


def main() -> int:
    print(f"Building FPA{FPA} band (uniform scene, same setup as the real check)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400, None,
                           True, False, 32, False, 0)  # uniform=True
    print("done.\n", flush=True)

    # ---- truth's own RT output: band['radiance'] IS the _lookup_sample result ----
    S_hires_truth = np.asarray(band["radiance"](0.0), dtype=float).ravel()
    wn_hires = band["wn_hires"]
    print(f"truth I_hires: {len(S_hires_truth)} pts, range [{S_hires_truth.min():.6g}, "
         f"{S_hires_truth.max():.6g}]", flush=True)

    # ---- retrieval's own RT output: exact construction from
    # _joint_retrieve_fixed_nuisance, bypassing GERTRetrieval to read I_hires
    # directly. obs_grid uses row 512's real grid -- wn_hires itself doesn't
    # depend on obs_grid (only wn_instrument does), so this is a fair,
    # like-for-like RT-only comparison regardless of which row's grid is used.
    cols = np.arange(1024.0)
    lam_512, _ = xy_to_wavelength_slit(FPA, cols, np.full(1024, 512.0))
    nu_512 = 1e4 / lam_512
    win = SpectralWindow(wn_min=band["wn_min"], wn_max=band["wn_max"],
                         ils=ILS(type="gaussian", fwhm=band["fwhm_cm"]),
                         molecules=band["mols"], label=band["label"], obs_grid=nu_512)
    inst = Instrument(windows=[win], noise_model=geocarb_noise_model(FPA))
    fm = ForwardModel(atm_center, absco, inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    res = fm.run(albedo=np.array([band["albedo"]]), albedo_slope=np.zeros(1))
    S_hires_retrieval = np.asarray(res.I_hires[0], dtype=float)
    print(f"retrieval I_hires: {len(S_hires_retrieval)} pts, range [{S_hires_retrieval.min():.6g}, "
         f"{S_hires_retrieval.max():.6g}]\n", flush=True)

    if len(S_hires_truth) != len(S_hires_retrieval):
        print(f"FAIL: hi-res grids have different lengths ({len(S_hires_truth)} vs "
             f"{len(S_hires_retrieval)}) -- cannot compare directly.")
        return 1

    raw_diff = S_hires_retrieval - S_hires_truth
    rel = raw_diff / S_hires_truth
    print(f"raw I_hires (pre-convolution) comparison:")
    print(f"  max|diff|={np.max(np.abs(raw_diff)):.6g}  "
         f"max|rel diff|={np.max(np.abs(rel)):.6g}  "
         f"mean diff={raw_diff.mean():+.6g}")
    raw_identical = np.max(np.abs(raw_diff)) == 0.0
    print(f"  {'bit-identical' if raw_identical else 'NOT bit-identical'}\n", flush=True)

    # ---- convolve BOTH raw hi-res spectra with the SAME (now-fixed)
    # _diagonal_ils_convolve, at a few real rows, to see if any RT-level
    # discrepancy survives convolution ----
    fig, axes = plt.subplots(len(TEST_ROWS), 2, figsize=(13, 3.6 * len(TEST_ROWS)))
    print(f"{'row':>5s} {'source':>10s} {'continuum':>12s} {'core':>12s} {'depth':>12s} "
         f"{'depth_diff':>12s} {'depth_reldiff':>14s} {'max|diff|':>12s}")
    for k, row in enumerate(TEST_ROWS):
        lam_row, _ = xy_to_wavelength_slit(FPA, cols, np.full(1024, float(row)))
        nu_row = 1e4 / lam_row

        y_truth = _diagonal_ils_convolve(
            wn_hires, np.broadcast_to(S_hires_truth, (1024, len(S_hires_truth))), nu_row, win.ils)
        y_retrieval = _diagonal_ils_convolve(
            wn_hires, np.broadcast_to(S_hires_retrieval, (1024, len(S_hires_retrieval))), nu_row, win.ils)

        diff = y_retrieval - y_truth
        t_c, t_core, t_depth = line_depth(y_truth)
        r_c, r_core, r_depth = line_depth(y_retrieval)
        depth_diff = r_depth - t_depth
        depth_reldiff = depth_diff / t_depth if t_depth != 0 else float("nan")
        print(f"{row:5d} {'truth':>10s} {t_c:12.6g} {t_core:12.6g} {t_depth:12.6g} "
             f"{'--':>12s} {'--':>14s} {'--':>12s}")
        print(f"{'':5s} {'retrieval':>10s} {r_c:12.6g} {r_core:12.6g} {r_depth:12.6g} "
             f"{depth_diff:+12.4g} {depth_reldiff:+14.6g} {np.max(np.abs(diff)):12.4g}", flush=True)

        ax = axes[k, 0]
        ax.plot(nu_row, y_truth, color="tab:green", lw=1.3, label="truth RT (build_lookup_radiance)")
        ax.plot(nu_row, y_retrieval, color="tab:purple", lw=1.0, ls="--", label="retrieval RT (fresh ForwardModel)")
        ax.set_title(f"row {row}: same ILS convolution, different RT source", fontsize=10.5)
        ax.set_xlabel("wavenumber [cm-1]"); ax.set_ylabel("radiance")
        ax.legend(fontsize=7.5)

        ax = axes[k, 1]
        ax.axhline(0, color="black", lw=0.6)
        ax.plot(nu_row, diff, color="tab:red", lw=1.0)
        ax.set_title(f"row {row}: retrieval RT - truth RT (max|diff|={np.max(np.abs(diff)):.3g})", fontsize=10.5)
        ax.set_xlabel("wavenumber [cm-1]"); ax.set_ylabel("diff")

    fig.suptitle(f"FPA{FPA}: same ILS kernel, two independently-instantiated RT calls "
                f"-- does I_hires itself match?", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_rt_reproducibility_test_fpa{FPA}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0 if raw_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
