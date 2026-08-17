#!/usr/bin/env python3
"""Full multi-band (no-aerosol) stress test -- uniform/barcode/realistic
scenes x noise/no-noise x native/rectified/undistorted pipelines -- for an
ordered set of >=1 GeoCarb bands (default FPA0 + FPA2). A single FPA (e.g.
--fpas 0) runs the plain single-band battery that used to be a separate
script (gd_band_stress_test.py, retired 2026-08-10 once this generalized to
N>=1 cleanly -- see the "N ordered bands" note below); >=2 FPAs runs a joint
retrieval sharing one state vector. See KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11g
(first no-aerosol joint result, order=0, uniform-only, 2 bands) and Sec. 11h
(the full 2-band battery this generalizes) for the rationale. Deliberately
no aerosol still (Sec. 11's own motivation would have that nonlinearity
compound with anything found here, and it costs much more compute) -- see
Sec. 11d items 4/6 for when that gets added back in.

Three pipelines, with a specifically cross-band meaning each, generalized
from 2 bands to N ordered bands `fpas = [fpas[0], fpas[1], ...]` (fpas[0]
is always the along-slit position reference for pairing and plots):
  "native"       -- each band's own real per-pixel row, exactly as
                     gd_band_stress_test.py's native pipeline renders it
                     (real projection operator + ILS convolution, real
                     spatial PSF blur), paired across all N bands via
                     geocarb_gert.cross_band.nearest_row_pairing_multi
                     (Sec. 11f -- real slit angle `s`, never normalized
                     `eta`; each individual pairing bounded <=0.5 row
                     mismatch, confirmed 2026-07-27 -- error does not
                     compound as bands are added, only the valid along-slit
                     range, an N-way intersection, can shrink).
  "undistorted"  -- each band's zero-distortion truth spectrum evaluated
                     directly at its own real per-row position (bypasses
                     gd_render.image()/rectify() entirely, same mechanism as
                     gd_band_stress_test.py's undistorted branch), same
                     nearest-row pairing as native. Isolates whether a
                     cross-band retrieval difficulty is fundamental
                     information content vs. something real geometric
                     distortion (keystone/smile) makes worse -- mirroring
                     that pipeline's single-band role. Floats dispersion
                     (order=2, same as native/rectified) despite having no
                     real wavelength-calibration error to correct -- see
                     Sec. 11k: this is a numerical-self-consistency
                     workaround, not a physical correction. Without it,
                     every retrieval (any pipeline) that never floats
                     dispersion is stuck evaluating its forward model with
                     grid-snapped ILS centering (`gert.instrument.ILS.
                     convolve`'s `exact_center=False` default), while this
                     project's own truth-rendering path (`gd_render.
                     _diagonal_ils_convolve`, shared by native and
                     undistorted alike) always uses exact centering --
                     merely having ANY nonzero-sized dispersion vector in
                     the state vector flips `gert.forward_model`'s internal
                     convolution call onto the matching exact-center path
                     (`ForwardModel.run()` passes `wn_centers=win.
                     dispersion_centers(...)` whenever dispersion is
                     present, and `SpectralWindow.convolve()` only requests
                     `exact_center=True` when `wn_centers` is given), so
                     floating it here removes a real, confirmed chi2 floor
                     (~0.05 for FPA0+FPA3, collapsing to <1e-9 with this
                     fix, verified 2026-07-28 via `scripts/
                     gd_joint_undistorted_dispersion_diag.py`) that has
                     nothing to do with atmospheric retrieval quality.
  "rectified"    -- each band rectified independently (gd_render.rectify)
                     onto ONE SHARED cross-band `s_grid` (real degrees,
                     N-way-intersection-clipped to every band's valid
                     coverage) -- the ORIGINAL co-registration mechanism
                     Sec. 11b/11c proposed and then set aside in favor of
                     nearest-row, kept here specifically to test that
                     decision empirically: does shared-grid rectification's
                     already-quantified single-band interpolation bias
                     (Sec. 9l/9m) show up here too, on top of any real
                     cross-band mismatch? Rows are aligned by construction
                     (same `s_grid` index for every band) -- no separate
                     pairing needed for this pipeline. Coverage shrinks
                     (never grows) as more bands are added, since it's an
                     N-way intersection of ranges.

State vector: single shared `p_scale`/`h2o_scale`, each other gas sensitive
only through whichever band(s) have it in their molecule list (confirmed
via gert's own multi-window Jacobian assembly, Sec. 11g) -- no aerosol.
All three pipelines get per-band dispersion order 2 (each band's own
`disp_a{k}_{b}`, `b` = window index, i.e. its position in `fpas`) as of
2026-07-28 (Sec. 11k) -- including undistorted, previously order 0. Its
retrieved dispersion coefficients are expected to sit near zero (there is
no real calibration error for them to explain); they're floated purely to
force `gert`'s exact-center ILS convolution path, matching this project's
truth-rendering convention -- see the "undistorted" bullet above.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_test.py \\
        --fpas 0,2 [--uniform | --barcode [--barcode-bars N]] \\
        [--noise [--snr S] [--noise-seed N]] [--row-step N] [--out-tag TAG]
      (--fpas takes any >=2 comma-separated FPA indices, e.g. 0,1,2,3)
Output: results/gd_joint_<fpas_tag>[_uniform|_barcode][_noise][_<tag>].pkl
        (fpas_tag = "fpa0_fpa2", "fpa0_fpa1_fpa2_fpa3", etc. -- see
        geocarb_gert.cross_band.fpas_tag)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

import geosat_geometry as gg
from geocarb_gert import GEOCARB_BANDS, albedo_for, along_slit_scene as als, sample_geometries
from geocarb_gert import gert_root  # noqa: E402
from geocarb_gert import gd_render, build_geocarb_instrument
from geocarb_gert import (RADIOMETRIC_SPEC_BY_FPA, geocarb_noise_model,
                          geocarb_noise_model_multi)
from geocarb_gert.cross_band import fpas_tag, nearest_row_pairing_multi, real_s_of_row
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit
from geocarb_gert.gd_render import available_cpus, s_max, _diagonal_ils_convolve

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

REPO_ROOT = Path(__file__).resolve().parent.parent
GERT_ROOT = gert_root()   # $GERT_ROOT -> ../../gert -> ../gert -> HPC scratch

WELL_MIXED_GASES = {"o2", "n2o"}
DEFAULT_SNR_BY_FPA = {0: 400.0, 1: 300.0, 2: 300.0, 3: 200.0}

_G = {}


def _band_setup(fpa: int, atm_center, absco, geo, solar, snr: float, n_lookup_samples: int,
                n_workers, uniform: bool, barcode: bool, barcode_bars: int,
                noise: bool, noise_seed: int, realistic_barcode: bool = False,
                vary_albedo: bool = False):
    """Render one band's raw detector image plus everything needed for all
    three pipelines: `A` (native/rectified source), `wn_hires`/`radiance`
    (undistorted source, and the barcode/lookup truth), the nominal per-band
    `wn_grid` (rectified's wavelength axis), and per-row real `s`/`x_km`
    (pairing and truth-scoring). Same mechanism as gd_band_stress_test.py's
    main(), just factored to run once per band here."""
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[fpa]
    mols = list(mols)
    wn_min, wn_max = real_wavenumber_range(fpa, margin_cm1=10.0)
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=mols, label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=snr)
    albedo = float(albedo_for(wide_inst, "desert")[0])

    if barcode:
        from geocarb_gert.focalplane import barcode_scene
        fm_center = ForwardModel(atm_center, absco, wide_inst, geo, solver=SingleScatterSolver(),
                                 solar_spectrum=solar)
        res_center = fm_center.run(albedo=np.array([albedo]), albedo_slope=[0.0])
        S_center = np.asarray(res_center.I_hires[0], dtype=float)
        brightness = np.resize([1.0, 0.2], barcode_bars)
        radiance = barcode_scene(S_center, brightness=brightness, widths=None, softness=0.0)
        wn_hires = wide_win.wn_hires
    elif realistic_barcode:
        # Realistic along-slit truth (genuine per-eta gas/pressure variation,
        # same as the plain realistic scene) with a barcode brightness gain
        # multiplied on top -- combines continuous state-vector variation
        # with a sharp, high-spatial-frequency reflectance pattern, unlike
        # plain --barcode above (which fixes the atmosphere at its center
        # value and only varies brightness). Reuses barcode_scene's own
        # eta->gain bar/boundary math by applying it to a unit "spectrum" (an
        # all-ones array), which broadcasts one shared scalar gain across
        # every hi-res bin; multiplying that gain elementwise onto the real
        # per-eta radiance gives the desired "realistic scene x barcode" scene.
        from geocarb_gert.focalplane import barcode_scene
        wn_hires, radiance_real = als.build_lookup_radiance(
            absco, wide_inst, geo, solar, np.array([albedo]),
            n_samples=n_lookup_samples, n_workers=n_workers, uniform=False,
            vary_albedo=vary_albedo)
        brightness = np.resize([1.0, 0.2], barcode_bars)
        gain_of_eta = barcode_scene(np.ones_like(wn_hires), brightness=brightness, widths=None, softness=0.0)

        def radiance(eta, _gain=gain_of_eta, _real=radiance_real):
            return _gain(eta) * _real(eta)
    else:
        # vary_albedo=True makes surface albedo vary along the slit too
        # (als.albedo_at: land-cover patches + fine-scale variability),
        # instead of the single fixed `albedo` scalar above. Off by default,
        # so pre-2026-08-17 runs reproduce exactly.
        wn_hires, radiance = als.build_lookup_radiance(
            absco, wide_inst, geo, solar, np.array([albedo]),
            n_samples=n_lookup_samples, n_workers=n_workers, uniform=uniform,
            vary_albedo=vary_albedo)

    A = gd_render.image(fpa, wn_hires, radiance, wide_win.ils, spatial_psf_fwhm_px=1.5,
                        n_workers=n_workers)
    # LinearShotNoise, calibrated from real instrument-test data
    # (geocarb_gert.radiometry.RADIOMETRIC_SPEC_BY_FPA) rather than the
    # legacy flat FlatSNR(snr) floor: sigma now scales with each pixel's own
    # rendered radiance, so e.g. a barcode scene's dark bars (5x fainter)
    # correctly get a smaller noise floor than its bright bars, instead of
    # borrowing the bright-bar sigma everywhere.
    band_noise = geocarb_noise_model(fpa)
    noise_n0, noise_n1, noise_i_max = band_noise.N0, band_noise.N1, band_noise.I_max
    sat_mask = np.abs(A) > noise_i_max
    n_sat = int(sat_mask.sum())
    if n_sat:
        print(f"  WARNING: {n_sat}/{A.size} pixels exceed I_max={noise_i_max:g} "
             f"(saturated) in FPA{fpa}'s rendered image", flush=True)
    sigma_map = np.sqrt(noise_n0 ** 2 + noise_n1 * np.abs(A))
    noise_arr = None
    if noise:
        rng = np.random.default_rng(noise_seed)
        noise_arr = rng.normal(0.0, sigma_map)
        A = A + noise_arr

    nominal_inst = build_geocarb_instrument()
    wn_grid = nominal_inst.windows[fpa].wn_instrument

    cols_center = np.full(1024, 512.0)
    _, s_of_row = xy_to_wavelength_slit(fpa, cols_center, np.arange(1024.0))
    sm = s_max(fpa)
    x_km_of_row = (s_of_row / sm) * als.SLIT_HALF_KM
    xtrue_x_km = np.zeros(1024) if (uniform or barcode) else x_km_of_row

    h2o_mean_prior_ppm = float(np.mean(atm_center.gases["h2o"])) * 1e6
    h2o_surface_ratio = als.h2o_surface_vmr(xtrue_x_km) / als.h2o_surface_vmr(0.0)
    xtrue_of_row = {
        "co2": als.xco2_ppm(xtrue_x_km), "ch4": als.xch4_ppb(xtrue_x_km), "co": als.xco_ppb(xtrue_x_km),
        "h2o": h2o_mean_prior_ppm * h2o_surface_ratio,
        "p_surface": als.p_surface_hpa(xtrue_x_km),
    }
    return dict(fpa=fpa, label=label, mols=mols, R=R, wn_min=wn_min, wn_max=wn_max,
               fwhm_cm=fwhm_cm, albedo=albedo, A=A, wn_hires=wn_hires, radiance=radiance,
               ils=wide_win.ils, wn_grid=wn_grid, noise_n0=noise_n0, noise_n1=noise_n1,
               noise_i_max=noise_i_max, sat_mask=sat_mask, noise_arr=noise_arr,
               x_km_of_row=x_km_of_row, xtrue_of_row=xtrue_of_row)


def _shared_s_grid(fpas, n: int = 1024) -> np.ndarray:
    """Real slit-angle grid [deg] spanning the N-way intersection of every
    band's covered range -- the target grid for the "rectified" pipeline.
    Same real-`s` convention as Sec. 11f's nearest_row_pairing, never eta."""
    los, his = [], []
    for fpa in fpas:
        s = real_s_of_row(fpa)
        los.append(min(s.min(), s.max()))
        his.append(max(s.min(), s.max()))
    return np.linspace(max(los), min(his), n)


def _native_row(band: dict, row: int):
    fpa = band["fpa"]
    cols = np.arange(1024.0)
    lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(row)))
    nu_row = 1e4 / lam_row
    y_dist = band["A"][row, :].copy()
    if nu_row[0] < nu_row[-1]:
        nu_row = nu_row[::-1]
        y_dist = y_dist[::-1]
    return nu_row, y_dist


def _undistorted_row(band: dict, row: int):
    fpa = band["fpa"]
    cols = np.arange(1024.0)
    lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(row)))
    nu_row = 1e4 / lam_row
    reverse = nu_row[0] < nu_row[-1]
    if reverse:
        nu_row = nu_row[::-1]
    eta_row = band["x_km_of_row"][row] / als.SLIT_HALF_KM
    S_row = np.asarray(band["radiance"](np.full(len(nu_row), eta_row)), dtype=float)
    y_dist = _diagonal_ils_convolve(band["wn_hires"], S_row, nu_row, band["ils"])
    if band["noise_arr"] is not None:
        noise_row = band["noise_arr"][row, :]
        if reverse:
            noise_row = noise_row[::-1]
        y_dist = y_dist + noise_row
    return nu_row, y_dist


def _rectified_row(band: dict, Rimg: np.ndarray, grid_idx: int):
    row = Rimg[grid_idx, :]
    valid = ~np.isnan(row)
    if valid.sum() < 0.5 * len(row):
        return None, None
    nu_valid = band["wn_grid"][valid]
    y_dist = row[valid][::-1]
    return nu_valid[::-1], y_dist


def _gas_truth_source(gas: str, bands: list, xtrue_rows: list) -> dict:
    """Which band's truth to score a gas bias against: the *last* band (in
    `fpas` order) whose molecule list contains it, falling back to the
    first (reference) band otherwise -- generalizes the original 2-band
    rule (prefer band_b, default band_a) to N bands."""
    for b, xt in zip(reversed(bands), reversed(xtrue_rows)):
        if gas in b["mols"]:
            return xt
    return xtrue_rows[0]


def _joint_retrieve(nus: list, ys: list, order: int, xtrue_rows: list, bands: list):
    g = _G
    atm_center, absco, geo, solar = g["atm"], g["absco"], g["geo"], g["solar"]
    n_bands = len(bands)

    windows = [SpectralWindow(wn_min=b["wn_min"], wn_max=b["wn_max"],
                              ils=ILS(type="gaussian", fwhm=b["fwhm_cm"]),
                              molecules=b["mols"], label=b["label"], obs_grid=nu)
              for b, nu in zip(bands, nus)]
    # Per-band LinearShotNoise, calibrated from real instrument-test data
    # (geocarb_gert.radiometry.RADIOMETRIC_SPEC_BY_FPA, same source
    # _band_setup used to render each band's own noise). sigma is computed
    # from THIS row's own measured spectrum `ys`, not a scene-wide scalar,
    # so Sy_inv is now genuinely per-row -- a dim row (e.g. a barcode dark
    # bar, or a low-radiance edge of a realistic scene) gets a
    # correspondingly tighter noise floor rather than inheriting the
    # scene's brightest pixel's sigma.
    noise_model = geocarb_noise_model_multi([b["fpa"] for b in bands])
    inst = Instrument(windows=windows, noise_model=noise_model)

    y_true = np.concatenate(ys)
    sigma = noise_model.sigma(ys, windows)
    Sy_inv = np.diag(1.0 / sigma ** 2)

    fm = ForwardModel(atm_center, absco, inst, geo, solver=SingleScatterSolver(),
                      solar_spectrum=solar)
    prior_albedo = np.array([b["albedo"] for b in bands])
    gases = sorted({m for b in bands for m in b["mols"]
                    if m != "h2o" and m not in WELL_MIXED_GASES}) + ["h2o"]
    sv = StateVector.gas_scaling(prior_albedo=prior_albedo, prior_albedo_slope=np.zeros(n_bands),
                                 gases=gases, gas_uncerts={"h2o": 0.60},
                                 include_dispersion=(order > 0),
                                 dispersion_order=max(order, 0), dispersion_uncert=2.0)
    ret = GERTRetrieval(fm, y_true, Sy_inv, sv, prior_albedo=prior_albedo,
                        prior_albedo_slope=np.zeros(n_bands), analytical_jacobians=True,
                        max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
    names = sv.names
    nl = {}
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            res = ret.run()
    except (ValueError, np.linalg.LinAlgError) as e:
        nl["_chi2"] = np.nan
        nl["_conv"] = False
        nl["_diverged"] = str(e)
        return nl

    for gas in gases:
        src = _gas_truth_source(gas, bands, xtrue_rows)
        unit = 1e6 if gas not in ("ch4", "co") else 1e9
        prior_val = float(np.mean(atm_center.gases[gas])) * unit
        retrieved_val = prior_val * res.x_ret[names.index(gas + "_scale")]
        nl[gas] = retrieved_val - float(src[gas])
    p_scale_ret = float(res.x_ret[names.index("p_scale")])
    p_surface_prior = float(als.p_surface_hpa(np.array([0.0]))[0])
    nl["p_surface"] = p_surface_prior * p_scale_ret - float(xtrue_rows[0]["p_surface"])

    # Complete raw state vector + posterior uncertainty (every element,
    # all N bands' albedo/dispersion included), independent of the derived
    # gas/pressure bias dict above -- same convention gd_band_stress_test.py
    # uses for the single-band case (added 2026-07-25 there, added here
    # 2026-07-28 after being asked directly whether h2o was even in this
    # state vector -- it always is, `gases` unconditionally appends "h2o").
    # `_state` is a retrieved-minus-prior convenience view for everything
    # NOT already covered by the gas/pressure bias dict above (T_offset,
    # every band's albedo_{b}/albedo_slope_{b}, and -- order>0 only -- every
    # band's own disp_a{k}_{b}) -- exactly the "co-retrieved parameters" a
    # bias-correction analysis would want to regress the gas bias against.
    nl["_state_full"] = {
        "names": list(names),
        "x_ret": res.x_ret.copy(),
        "x_prior": sv.prior.copy(),
        "sigma": res.posterior_sigma(),
    }
    _tracked = {f"{gas}_scale" for gas in gases} | {"p_scale"}
    nl["_state"] = {name: float(res.x_ret[i] - prior)
                    for i, (name, prior) in enumerate(zip(names, sv.prior))
                    if name not in _tracked}

    nl["_chi2"] = float(res.chisq_reduced)
    nl["_conv"] = res.converged
    if res.diverged:
        nl["_diverged"] = "gert: chi2 non-finite"

    # Post-fit residual (y_true - y_ret) and each band's own wavenumber
    # grid, one array per band in `bands`/`fpas` order -- added 2026-07-29,
    # closing the exact gap gd_plot_residual_spectra.py's own
    # docstring flagged (it had to re-render + re-run a representative row
    # subset from scratch because this wasn't saved here). Mirrors
    # gd_band_stress_test.py's convention (residual/nu saved per row since
    # 2026-07-25), just split per band since a joint fit has more than one
    # wavenumber axis. Storage cost is real (roughly doubles per-row size
    # for a 2-band pair, since each band's own nu/residual arrays are
    # ~1024 floats) but cheap relative to retrieval compute, per the
    # standing project rule to always save residuals from the start.
    resid = y_true - res.y_ret
    lens = [len(y) for y in ys]
    nl["_residuals"] = np.split(resid, np.cumsum(lens)[:-1])
    nl["_nus"] = list(nus)
    return nl


def _worker(task):
    pipeline, order, rows = task
    g = _G
    bands = g["bands"]
    xtrue_rows = []
    for b, r in zip(bands, rows):
        xt = {gas: float(b["xtrue_of_row"][gas][r]) for gas in ("co2", "ch4", "co", "h2o")}
        xt["p_surface"] = float(b["xtrue_of_row"]["p_surface"][r])
        xtrue_rows.append(xt)
    try:
        if pipeline == "native":
            nus, ys = zip(*[_native_row(b, r) for b, r in zip(bands, rows)])
        elif pipeline == "undistorted":
            nus, ys = zip(*[_undistorted_row(b, r) for b, r in zip(bands, rows)])
        else:  # rectified -- rows are all the shared s_grid index by construction
            pairs = [_rectified_row(b, Rimg, r) for b, Rimg, r in zip(bands, g["Rimgs"], rows)]
            if any(nu is None for nu, _ in pairs):
                return (pipeline, order, rows), {"_chi2": np.nan, "_conv": False, "off_detector": True}
            nus, ys = zip(*pairs)
        nl = _joint_retrieve(list(nus), list(ys), order, xtrue_rows, bands)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        nl = {"_chi2": np.nan, "_conv": False, "_diverged": f"{type(e).__name__}: {e}"}
    return (pipeline, order, rows), nl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpas", type=str, required=True,
                    help="comma-separated list of >=1 FPA indices (0-3), e.g. "
                         "0 for a single-band run, or 0,2 / 0,1,2,3 for a joint "
                         "one; the first is the along-slit position reference "
                         "used for row-pairing and plots (a no-op for N=1)")
    ap.add_argument("--row-step", type=int, default=1, help="subsample the reference band's native rows")
    ap.add_argument("--n-lookup-samples", type=int, default=400)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--out-tag", type=str, default=None)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--barcode", action="store_true")
    ap.add_argument("--realistic-barcode", action="store_true",
                    help="realistic along-slit truth (genuine gas/pressure "
                         "variation) modulated by a barcode brightness "
                         "pattern -- unlike --barcode, which fixes the "
                         "atmosphere at its center value")
    ap.add_argument("--barcode-bars", type=int, default=32)
    ap.add_argument("--snr", type=float, default=None,
                    help="override every band's SNR (default: the min across "
                         "all bands' own DEFAULT_SNR_BY_FPA, the most "
                         "conservative choice for a joint fit)")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--pipelines", type=str, default="native,rectified,undistorted",
                    help="comma-separated subset of native,rectified,undistorted")
    args = ap.parse_args()
    if sum([args.uniform, args.barcode, args.realistic_barcode]) > 1:
        ap.error("--uniform, --barcode, and --realistic-barcode are mutually exclusive")
    pipelines = args.pipelines.split(",")

    FPAS = [int(x) for x in args.fpas.split(",")]
    if len(FPAS) < 1:
        ap.error("--fpas needs at least 1 band index")
    if len(set(FPAS)) != len(FPAS):
        ap.error("--fpas entries must be unique")
    for fpa in FPAS:
        if fpa not in (0, 1, 2, 3):
            ap.error(f"invalid FPA index {fpa}, must be 0-3")

    snr = args.snr if args.snr is not None else min(DEFAULT_SNR_BY_FPA[fpa] for fpa in FPAS)

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)

    t0 = time.time()
    scene_label = ('realistic_barcode' if args.realistic_barcode else
                  'barcode' if args.barcode else 'uniform' if args.uniform else 'realistic')
    bands = []
    for fpa in FPAS:
        print(f"rendering FPA{fpa} ({scene_label}{', noise' if args.noise else ''})...", flush=True)
        b = _band_setup(fpa, atm_center, absco, geo, solar, snr, args.n_lookup_samples,
                        args.n_workers, args.uniform, args.barcode, args.barcode_bars,
                        args.noise, args.noise_seed, args.realistic_barcode)
        print(f"  done ({time.time()-t0:.0f}s)", flush=True)
        bands.append(b)

    _G.update(dict(bands=bands, atm=atm_center, absco=absco, geo=geo,
                   solar=solar, snr=snr))

    tasks = []

    if "native" in pipelines or "undistorted" in pipelines:
        rows_ref = np.arange(0, 1024, args.row_step, dtype=float)
        pairing = nearest_row_pairing_multi(FPAS, rows_ref)
        n_valid = len(pairing["rows"][FPAS[0]])
        mismatch_summary = ", ".join(
            f"FPA{fpa} mean={np.abs(pairing['mismatch_rows'][fpa]).mean():.3f} "
            f"max={np.abs(pairing['mismatch_rows'][fpa]).max():.3f}"
            for fpa in FPAS[1:])
        print(f"native/undistorted pairing: {n_valid}/{len(rows_ref)} rows "
             f"({pairing['n_dropped']} dropped), mismatch vs FPA{FPAS[0]} (rows): "
             f"{mismatch_summary}", flush=True)
        for i in range(n_valid):
            rows = tuple(int(pairing["rows"][fpa][i]) for fpa in FPAS)
            if "native" in pipelines:
                tasks.append(("native", 2, rows))
            if "undistorted" in pipelines:
                tasks.append(("undistorted", 2, rows))

    if "rectified" in pipelines:
        s_grid_shared = _shared_s_grid(FPAS)
        Rimgs = [gd_render.rectify(fpa, b["A"], s_grid_shared, b["wn_grid"])
                for fpa, b in zip(FPAS, bands)]
        _G["Rimgs"] = Rimgs
        grid_idx = np.arange(0, len(s_grid_shared), max(args.row_step, 1))
        print(f"rectified shared s_grid: {len(s_grid_shared)} pts, "
             f"[{s_grid_shared[0]:.3f},{s_grid_shared[-1]:.3f}] deg, "
             f"{len(grid_idx)} indices tested", flush=True)
        for k in grid_idx:
            tasks.append(("rectified", 2, tuple(int(k) for _ in FPAS)))

    print(f"{len(tasks)} joint retrievals ({pipelines})", flush=True)
    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for key, nl in pool.imap_unordered(_worker, tasks, chunksize=4):
            out[key] = nl
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"all done ({time.time()-t0:.0f}s)", flush=True)

    for pl in pipelines:
        keys = [k for k in out if k[0] == pl]
        n_conv = sum(1 for k in keys if out[k].get("_conv") and not out[k].get("_diverged")
                    and not out[k].get("off_detector"))
        print(f"  {pl}: converged {n_conv}/{len(keys)}", flush=True)

    tag = fpas_tag(FPAS)
    mode_suffix = ("_realistic_barcode" if args.realistic_barcode else
                  "_uniform" if args.uniform else ("_barcode" if args.barcode else ""))
    noise_suffix = "_noise" if args.noise else ""
    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    out_path = (REPO_ROOT / "results" /
               f"gd_joint_{tag}{mode_suffix}{noise_suffix}{tag_suffix}.pkl")
    out_path.parent.mkdir(exist_ok=True)
    any_barcode = args.barcode or args.realistic_barcode
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "fpas": FPAS, "uniform": args.uniform,
                    "barcode": args.barcode, "realistic_barcode": args.realistic_barcode,
                    "barcode_bars": args.barcode_bars if any_barcode else None,
                    "noise": args.noise, "noise_seed": args.noise_seed if args.noise else None,
                    "snr": snr, "pipelines": pipelines}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
