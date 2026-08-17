#!/usr/bin/env python3
"""Phase 1-3 realistic diagnostic for the joint multi-atmosphere block
(`JOINT_ROW_INVERSION_PLAN.md` §2/§4), re-running exactly §11p's row-910
hot-spot case: instead of one atmosphere per row (native's own
per-row-independent `GERTRetrieval` call, which captures only 44% of the
true peak enhancement there), solve jointly for `G` atmospheres on a grid
of eta bins spanning a row window around the hot spot, using the exact
real, no-noise FPA2 realistic-scene detector image already used to build
`results/gd_joint_fpa2.pkl` and the trace-and-select scripts.

Reuses, unmodified: `geocarb_gert.gd_render.predict_neighborhood` (Phase
1's forward operator, validated 2026-08-12 against `image()` to floating-
point precision) and `geocarb_gert.nearest_bin_scene` (Phase 1's
state-space-only, nearest-bin scene assignment -- no radiance-space
interpolation anywhere, avoiding rectify's own §9l/§9m bias). The GN/OE
solve itself is new, standalone code here (not a `gert` modification, not
`GERTRetrieval` -- confirmed not to generalize this way, see §4's GERT
inspection note), built directly on `StateVector.gas_scaling()`+`.apply()`
and `ForwardModel`, exactly Phase 0's own validated recipe, now with `G`
bins instead of 2.

Two known, flagged simplifications for this first realistic pass (not
hidden, both real open items for a later iteration):
- Only `co2_scale` is solved per bin. Every other gas (h2o/ch4/co,
  surface pressure) is held at its own *local true* value for that bin's
  along-slit position -- `als.atmosphere_at(x_km)`, the same function
  that generated the real image's per-row truth -- rather than retrieved.
  An earlier version of this script held ALL bins at one *shared* prior
  atmosphere (`atm_center = atmosphere_at(0.0)`) instead; that failed
  badly (a 5-row/G=5 trial run converged to a >1000% "capture" fraction,
  nowhere near physical). The residual wasn't CO2 dilution at all -- it
  was the model trying to explain real per-row CH4/H2O/pressure variation
  it had no free parameter for, by distorting CO2 instead. Giving each
  bin its own correct local nuisance-gas truth removes that confound and
  isolates the actual question this test asks: does bin-splicing recover
  CO2's keystone-diluted gradient specifically, independent of whether
  other gases are also jointly retrieved (a separate, later engineering
  question, not this one).
- `Sy_inv` is a simple uniform relative weighting (`1/mean(y)^2`), not the
  real photon-noise model `_joint_retrieve` uses -- reasonable on a
  still-noise-free scene where the fit's limiting factor is genuine model
  mismatch (G discrete bins vs. continuous truth), not weighting-driven
  noise averaging.

`G` bins this large is underdetermined by construction (§3) -- a
first-difference Tikhonov smoothness prior on adjacent bins' co2_scale
(strength `--gamma`) regularizes it, the same honest resolution-limit
tradeoff §3 describes, not a bug to engineer around.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_retrieve.py \\
        --row-min 890 --row-max 935 --n-bins 15 --gamma 3.0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_test as gdt  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import GEOCARB_BANDS, albedo_for, along_slit_scene as als, nearest_bin_scene, sample_geometries  # noqa: E402
from geocarb_gert import gd_render  # noqa: E402
from geocarb_gert import gert_root  # noqa: E402
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

from gert.forward_model import ForwardModel  # noqa: E402
from gert.instrument import ILS, SpectralWindow  # noqa: E402
from gert.instrument_config import Instrument  # noqa: E402
from gert.retrieval import StateVector  # noqa: E402
from gert.rt_solver import SingleScatterSolver  # noqa: E402

# Where gert's input/ tree (absco.h5, solar.h5) lives. Defaults to the HPC
# scratch path every batch script and prior run used, so those are unchanged;
# override with $GERT_ROOT to run the same sweeps on a workstation (needed
# 2026-08-17, when the cluster was unavailable -- nothing else in the
# joint-block chain is HPC-specific, since the sweep already runs standalone
# on a local multiprocessing Pool when --task-id/--n-tasks are omitted).
GERT_ROOT = gert_root()   # geocarb_gert.paths -- shared by every driver
FPA = 2


def _eta_of(fpa, cols, rows):
    _, s = xy_to_wavelength_slit(fpa, cols, rows)
    return s / s_max(fpa)


def band_basics(fpa, atm_center, absco, geo, solar):
    """Cheap wide_win/wide_inst reconstruction -- matches `_band_setup`'s
    own recipe exactly (same molecules/wn range/fwhm/label), but rebuilt
    here since `_band_setup`'s returned `band` dict exposes `wn_hires`/
    `ils`/`albedo` but not the `Instrument` object itself."""
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


def make_spectrum_fn(absco, wide_inst, geo, solar, albedo):
    """spectrum_for(co2_scale, prior_atm) -> S_hires. `prior_atm` is
    per-call (per-bin, per its own local truth), unlike Phase 0's single
    shared `atm_center` -- see module docstring for why."""
    def spectrum_for(co2_scale: float, prior_atm) -> np.ndarray:
        sv = StateVector.gas_scaling(prior_albedo=np.array([albedo]), prior_albedo_slope=np.zeros(1),
                                     gases=["co2"], gas_uncerts={"co2": 0.10})
        sv.values[sv.names.index("co2_scale")] = co2_scale
        atm, alb, alb_slope, tau_aer, sbp = sv.apply(prior_atm, np.array([albedo]), np.zeros(1))
        fm = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
        res = fm.run(albedo=alb, albedo_slope=alb_slope)
        return np.asarray(res.I_hires[0], dtype=float)

    return spectrum_for


def build_forward(fpa, rows, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=4):
    """Returns forward(x) -> predicted raveled sub-image, x = per-bin
    co2_scale (shape (G,)), each bin evaluated against its OWN
    `prior_atms[g]` (that bin's local true nuisance-gas state). Caches
    each bin's own hi-res spectrum and only recomputes the bins that
    actually changed between calls -- the "G+1 ForwardModel calls per
    iteration, not G*(G+1)" optimization."""
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
        A = gd_render.predict_neighborhood(fpa, rows, wn_hires, radiance, ils, pad=pad)
        return A.ravel()

    return forward


def gauss_newton_regularized(forward, y_true, x0, Sy_inv_diag, gamma, sigma_abs,
                             step=1e-3, max_iter=15, tol=1e-5, label=""):
    """Rodgers-form regularized GN update, standalone (JOINT_ROW_INVERSION_
    PLAN.md §4's 'revised first step'): dx = (K^T Sy^-1 K + Sa^-1)^-1
    (K^T Sy^-1 resid - Sa^-1 (x - x_a)). Sa^-1 = gamma * L^T L (first-
    difference smoothness) + (1/sigma_abs^2) I (loose absolute floor,
    keeps the system well-posed against the smoothness term's null space:
    a constant shift of all bins together)."""
    x = np.asarray(x0, dtype=float).copy()
    x_a = x.copy()
    n = len(x)
    L = np.zeros((n - 1, n))
    for k in range(n - 1):
        L[k, k] = -1.0
        L[k, k + 1] = 1.0
    Sa_inv = gamma * (L.T @ L) + np.eye(n) / sigma_abs ** 2

    for it in range(max_iter):
        y0 = forward(x)
        resid = y_true - y0
        K = np.empty((len(y0), n))
        for k in range(n):
            xp = x.copy()
            xp[k] += step
            K[:, k] = (forward(xp) - y0) / step
        KtSyinv = K.T * Sy_inv_diag[None, :]
        A = KtSyinv @ K + Sa_inv
        b = KtSyinv @ resid - Sa_inv @ (x - x_a)
        dx = np.linalg.solve(A, b)
        x = x + dx
        rms = float(np.sqrt(np.mean(resid ** 2)))
        print(f"  [{label}] iter {it}: |dx|={np.linalg.norm(dx):.3e} rms_resid={rms:.4g}", flush=True)
        if np.linalg.norm(dx) < tol:
            break
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-min", type=int, default=890)
    ap.add_argument("--row-max", type=int, default=935)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=3.0, help="smoothness-prior strength")
    ap.add_argument("--sigma-abs", type=float, default=0.10, help="loose absolute-prior floor on co2_scale")
    ap.add_argument("--peak-row", type=int, default=910)
    ap.add_argument("--bg-row", type=int, default=880)
    args = ap.parse_args()

    print(f"Building realistic-scene FPA{FPA} band (renders the real 1024x1024 detector image, "
         f"same underlying scene as results/gd_joint_fpa2.pkl)...", flush=True)
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

    wide_win, wide_inst, albedo_check = band_basics(FPA, atm_center, absco, geo, solar)
    wn_hires = band["wn_hires"]
    ils = band["ils"]
    albedo = band["albedo"]
    assert abs(albedo - albedo_check) < 1e-9, "band_basics albedo must match _band_setup's own"

    rows_win = np.arange(args.row_min, args.row_max + 1)
    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    eta_lo, eta_hi = float(eta_all.min()), float(eta_all.max())
    bin_centers = np.linspace(eta_lo, eta_hi, args.n_bins)
    x_km_bins = bin_centers * als.SLIT_HALF_KM
    prior_atms = [als.atmosphere_at(float(xk)) for xk in x_km_bins]
    prior_co2_ppm_bins = np.array([float(als.xco2_ppm(xk)) for xk in x_km_bins])
    print(f"rows {args.row_min}-{args.row_max} ({len(rows_win)} rows), "
         f"eta range [{eta_lo:.5f}, {eta_hi:.5f}], G={args.n_bins} bins, "
         f"each with its own local true nuisance-gas atmosphere", flush=True)

    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    forward = build_forward(FPA, rows_win, bin_centers, prior_atms, spectrum_for, wn_hires, ils, pad=4)

    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

    true_peak = float(band["xtrue_of_row"]["co2"][args.peak_row])
    true_bg = float(band["xtrue_of_row"]["co2"][args.bg_row])

    def bin_at_row(row):
        eta_c = float(_eta_of(FPA, np.array([512.0]), np.array([float(row)]))[0])
        return int(np.argmin(np.abs(bin_centers - eta_c)))

    g_peak = bin_at_row(args.peak_row)
    g_bg = bin_at_row(args.bg_row)
    print(f"peak row {args.peak_row} -> bin {g_peak} (eta={bin_centers[g_peak]:.5f}); "
         f"bg row {args.bg_row} -> bin {g_bg} (eta={bin_centers[g_bg]:.5f})", flush=True)

    t0 = time.time()
    print(f"\n=== G={args.n_bins} joint block solve (gamma={args.gamma}) ===", flush=True)
    x_g = gauss_newton_regularized(forward, y_true, x0=np.ones(args.n_bins),
                                   Sy_inv_diag=Sy_inv_diag, gamma=args.gamma,
                                   sigma_abs=args.sigma_abs, label=f"G={args.n_bins}")
    print(f"({time.time()-t0:.0f}s)", flush=True)

    retrieved_ppm = prior_co2_ppm_bins * x_g
    print("\nper-bin recovered CO2 [ppm]:")
    for g in range(args.n_bins):
        print(f"  bin {g:2d}  eta={bin_centers[g]:+.4f}  local_true={prior_co2_ppm_bins[g]:.2f}ppm  "
             f"co2_scale={x_g[g]:.5f}  co2={retrieved_ppm[g]:.2f} ppm  "
             f"error={retrieved_ppm[g]-prior_co2_ppm_bins[g]:+.2f} ppm")

    retrieved_peak = retrieved_ppm[g_peak]
    retrieved_bg = retrieved_ppm[g_bg]
    print(f"\ntrue rise: {true_peak - true_bg:+.3f} ppm   "
         f"joint-block retrieved rise: {retrieved_peak - retrieved_bg:+.3f} ppm")
    capture = (retrieved_peak - retrieved_bg) / (true_peak - true_bg)
    print(f"peak-enhancement captured: {capture*100:.1f}%  "
         f"(native=44%, undistorted=99%, trace-and-select[tol=0.5km]=84.9%, "
         f"from KEYSTONE_SMILE_BIAS_PLAN.md §11p / JOINT_ROW_INVERSION_PLAN.md §0a)")

    print(f"\n=== G=1 control (single shared atmosphere, same pixels) ===", flush=True)
    eta_mid = 0.5 * (eta_lo + eta_hi)
    prior_atm_1 = als.atmosphere_at(float(eta_mid * als.SLIT_HALF_KM))
    prior_co2_ppm_1 = float(als.xco2_ppm(eta_mid * als.SLIT_HALF_KM))
    t0 = time.time()
    x_1 = gauss_newton_regularized(forward=build_forward(FPA, rows_win, np.array([eta_mid]), [prior_atm_1],
                                                          spectrum_for, wn_hires, ils, pad=4),
                                   y_true=y_true, x0=np.ones(1), Sy_inv_diag=Sy_inv_diag,
                                   gamma=0.0, sigma_abs=args.sigma_abs, label="G=1")
    print(f"({time.time()-t0:.0f}s)", flush=True)
    retrieved_1 = prior_co2_ppm_1 * x_1[0]
    print(f"G=1 recovered: {retrieved_1:.2f} ppm at both rows (single atmosphere, window-midpoint "
         f"local truth={prior_co2_ppm_1:.2f}ppm) -- true peak={true_peak:.2f}, true bg={true_bg:.2f}, "
         f"native's own dilution capture at this hot spot was 44% (§11p)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
