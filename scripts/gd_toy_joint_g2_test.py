#!/usr/bin/env python3
"""Phase 0 -- smallest possible extensibility test for the joint
multi-atmosphere block (`JOINT_ROW_INVERSION_PLAN.md` §4).

GERT's own `GERTRetrieval`/`StateVector` machinery cannot be extended
in-place to a multi-atmosphere joint state (confirmed 2026-08-12 by
directly reading `gert/retrieval.py` -- `StateVector.apply()` maps onto
exactly one `AtmosphericProfile`, and `GERTRetrieval.__init__` takes
exactly one `fm_prior`/`state_vector`). What IS reusable unchanged:
`ForwardModel` and `StateVector.gas_scaling()` + `.apply()`, called once
per atmosphere/bin. This script builds a small standalone finite-
difference Gauss-Newton loop around exactly that reusable core -- no
`gert` modification, no `GERTRetrieval` involved.

Synthetic setup: FPA2, a small window of rows around a center row (default
512), a noise-free hard truth edge in eta (`geocarb_gert.focalplane.
edge_scene`, softness=0 -- the edge blur seen in the rendered image comes
entirely from the real spatial PSF, not from this synthetic scene). Two
true atmospheres (co2_scale A and B) differ from the shared prior. The
rendered "truth" image uses the real `xy_to_wavelength_slit` geometry, the
real per-row ILS convolution, and the real along-slit PSF blur -- i.e.
exactly `gd_render.image()`'s own per-row loop, just restricted to a small
padded row window for speed.

Two solves are run against the identical truth data:
  G=2 -- joint fit of both co2_scale_A and co2_scale_B simultaneously.
  G=1 -- a single shared co2_scale fit to the same pixels (the dilution
         control -- this is structurally what native's per-row-independent
         retrieval does when a row's own footprint straddles the edge).

Success criterion (plan doc §4): G=2 recovers both true values to within
ordinary retrieval noise; G=1 does not (it lands at a diluted value
between the two truths, weighted by pixel fractions on each side).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_joint_g2_test.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GERT_ROOT = Path("/scratch/scrowel3_lab/gert")

import geosat_geometry as gg
import gert
from geocarb_gert import GEOCARB_BANDS, albedo_for, along_slit_scene as als, sample_geometries
from geocarb_gert.focalplane import edge_scene, gaussian_blur_rows
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit
from geocarb_gert.gd_render import _diagonal_ils_convolve, s_max

from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import StateVector
from gert.rt_solver import SingleScatterSolver

FPA = 2
SPATIAL_PSF_FWHM_PX = 1.5   # matches gd_render.image()'s own default


def band_basics(fpa: int, atm_center, absco, geo, solar):
    """Just the cheap pieces of gd_test._band_setup -- wide_win/wide_inst/
    albedo -- without rendering the full 1024x1024 image."""
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[fpa]
    mols = list(mols)
    wn_min, wn_max = real_wavenumber_range(fpa, margin_cm1=10.0)
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=mols, label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=1e9)
    albedo = float(albedo_for(wide_inst, "desert")[0])
    return wide_win, wide_inst, albedo


def make_spectrum_fn(atm_center, absco, wide_inst, geo, solar, albedo):
    """Returns spectrum_for(co2_scale) -> S_hires, reusing StateVector.
    gas_scaling()+.apply() (unmodified) to build a modified atmosphere,
    then a fresh ForwardModel.run() -- the same recipe GERTRetrieval.
    _forward() itself uses, minus the Retrieval-specific bookkeeping."""
    sv = StateVector.gas_scaling(prior_albedo=np.array([albedo]), prior_albedo_slope=np.zeros(1),
                                 gases=["co2"], gas_uncerts={"co2": 0.10})
    idx = sv.names.index("co2_scale")

    def spectrum_for(co2_scale: float) -> np.ndarray:
        sv.values[idx] = co2_scale
        atm, alb, alb_slope, tau_aer, sbp = sv.apply(atm_center, np.array([albedo]), np.zeros(1))
        fm = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
        res = fm.run(albedo=alb, albedo_slope=alb_slope)
        return np.asarray(res.I_hires[0], dtype=float)

    return spectrum_for


def local_render(fpa: int, rows: np.ndarray, radiance, wn_hires: np.ndarray, ils,
                 pad: int = 4, psf_fwhm_px: float = SPATIAL_PSF_FWHM_PX) -> np.ndarray:
    """gd_render.image()'s own per-row loop, restricted to a padded window
    around `rows` so the PSF blur (which mixes neighboring rows) is handled
    correctly without re-rendering the full 1024x1024 image."""
    rows = np.asarray(rows, dtype=int)
    rows_padded = np.arange(rows.min() - pad, rows.max() + pad + 1)
    cols = np.arange(1024.0)
    A_pad = np.empty((len(rows_padded), 1024))
    for k, i in enumerate(rows_padded):
        lam_row, s_row = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(i)))
        nu_row = 1e4 / lam_row
        eta_row = s_row / s_max(fpa)
        S_row = np.asarray(radiance(eta_row), dtype=float)
        A_pad[k] = _diagonal_ils_convolve(wn_hires, S_row, nu_row, ils)
    A_pad = gaussian_blur_rows(A_pad, psf_fwhm_px)
    idx = np.searchsorted(rows_padded, rows)
    return A_pad[idx]


def gauss_newton(forward, y_true: np.ndarray, x0: np.ndarray, step: float = 1e-3,
                 max_iter: int = 12, tol: float = 1e-6, label: str = ""):
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    for it in range(max_iter):
        y0 = forward(x)
        resid = y_true - y0
        K = np.empty((len(y0), n))
        for k in range(n):
            xp = x.copy()
            xp[k] += step
            K[:, k] = (forward(xp) - y0) / step
        dx, *_ = np.linalg.lstsq(K, resid, rcond=None)
        x = x + dx
        rms = float(np.sqrt(np.mean(resid ** 2)))
        print(f"  [{label}] iter {it}: x={np.array2string(x, precision=6)} "
             f"|dx|={np.linalg.norm(dx):.3e} rms_resid={rms:.4g}", flush=True)
        if np.linalg.norm(dx) < tol:
            break
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center-row", type=int, default=512)
    ap.add_argument("--half-window", type=int, default=2, help="rows on each side of center-row (5 rows total by default)")
    ap.add_argument("--pad", type=int, default=4, help="extra padded rows for correct PSF-blur edge handling")
    ap.add_argument("--true-a", type=float, default=0.95, help="true co2_scale on the eta<edge side")
    ap.add_argument("--true-b", type=float, default=1.05, help="true co2_scale on the eta>edge side")
    args = ap.parse_args()

    print(f"Setting up FPA{FPA} shared prior + band basics (no full-image render)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    wn_hires, ils = wide_win.wn_hires, wide_win.ils

    spectrum_for = make_spectrum_fn(atm_center, absco, wide_inst, geo, solar, albedo)

    core_rows = np.arange(args.center_row - args.half_window, args.center_row + args.half_window + 1)
    eta_edge = float((xy_to_wavelength_slit(FPA, np.array([512.0]), np.array([float(args.center_row)]))[1])[0]
                    / s_max(FPA))
    print(f"core rows: {core_rows.tolist()}  eta_edge={eta_edge:.6f}  "
         f"true co2_scale: A={args.true_a} B={args.true_b}", flush=True)

    print("rendering truth (real geometry/ILS/PSF, synthetic hard edge, no noise)...", flush=True)
    S_A_true = spectrum_for(args.true_a)
    S_B_true = spectrum_for(args.true_b)
    radiance_true = edge_scene(S_A_true, S_B_true, edge=eta_edge, softness=0.0)
    A_true = local_render(FPA, core_rows, radiance_true, wn_hires, ils, pad=args.pad)
    y_true = A_true.ravel()
    print(f"truth rendered: {A_true.shape} pixels, {y_true.size} total samples", flush=True)

    def forward_g2(x):
        ca, cb = x
        Sa = spectrum_for(ca)
        Sb = spectrum_for(cb)
        rad = edge_scene(Sa, Sb, edge=eta_edge, softness=0.0)
        A = local_render(FPA, core_rows, rad, wn_hires, ils, pad=args.pad)
        return A.ravel()

    def forward_g1(x):
        (c,) = x
        S = spectrum_for(c)

        def rad(eta):
            eta = np.asarray(eta, dtype=float)
            return np.broadcast_to(S, eta.shape + S.shape)

        A = local_render(FPA, core_rows, rad, wn_hires, ils, pad=args.pad)
        return A.ravel()

    print("\n=== G=2 joint solve ===", flush=True)
    x2 = gauss_newton(forward_g2, y_true, x0=np.array([1.0, 1.0]), label="G=2")
    print(f"G=2 recovered: co2_scale_A={x2[0]:.5f} (true {args.true_a}), "
         f"co2_scale_B={x2[1]:.5f} (true {args.true_b})")
    print(f"  error_A={x2[0]-args.true_a:+.5f}  error_B={x2[1]-args.true_b:+.5f}")

    print("\n=== G=1 control (single shared atmosphere, same pixels) ===", flush=True)
    x1 = gauss_newton(forward_g1, y_true, x0=np.array([1.0]), label="G=1")
    # pixel-fraction-weighted expectation for reference
    eta_rows = np.stack([xy_to_wavelength_slit(FPA, np.arange(1024.0), np.full(1024, float(i)))[1] / s_max(FPA)
                         for i in core_rows])
    frac_b = float((eta_rows > eta_edge).mean())
    expected_dilute = (1 - frac_b) * args.true_a + frac_b * args.true_b
    print(f"G=1 recovered: co2_scale={x1[0]:.5f}  "
         f"(pixel-fraction-weighted dilution expectation ~{expected_dilute:.5f}, "
         f"{frac_b*100:.1f}% of core-window pixels on the B side)")

    print("\n=== verdict ===")
    g2_ok = abs(x2[0] - args.true_a) < 0.01 and abs(x2[1] - args.true_b) < 0.01
    g1_diluted = abs(x1[0] - args.true_a) > 0.005 and abs(x1[0] - args.true_b) > 0.005
    print(f"G=2 recovers both truths to <0.01 co2_scale: {g2_ok}")
    print(f"G=1 lands away from both truths (diluted, as expected): {g1_diluted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
