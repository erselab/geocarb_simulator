#!/usr/bin/env python3
"""Single-band composition/pressure stress test: render + rectify + retrieve
a full band using the along-slit-varying truth atmosphere (geocarb_gert.
along_slit_scene) instead of the uniform scene used throughout Sec. 9.

Extends the Sec. 9m/9n dense-sweep pipeline (previously FPA2/CO2_strong
only, uniform composition, albedo-only variability) two ways at once:
1. The truth atmosphere now varies continuously along the slit (realistic
   XCO2/XCH4/XCO/H2O/surface-pressure gradients, plumes, and small hot
   spots -- see geocarb_gert.along_slit_scene and scripts/
   gd_along_slit_atm_profiles.py's plot).
2. Runs on whichever band is requested via --fpa, retrieving that band's
   own appropriate gas set (e.g. FPA1/CO2_weak retrieves co2+h2o;
   FPA3/CH4_CO retrieves ch4+co+h2o) plus p_scale, which every band's
   StateVector already includes by default. Well-mixed gases (o2, n2o --
   see WELL_MIXED_GASES) are deliberately EXCLUDED from the retrieved gas
   list even when present in the band's molecule set: their true column
   never varies along the slit (see xtrue_of_row below), so retrieving a
   separate {gas}_scale for them is nearly degenerate with p_scale (both
   change a well-mixed gas's column via the same total-air-mass path).
   Found 2026-07-24: floating o2_scale alongside p_scale for FPA0/O2_A
   produced near-100% non-convergence (chi2 already good, but dx_norm
   never settling -- the fit wandering along the o2_scale/p_scale ridge).
   FPA0 now retrieves h2o_scale + p_scale only; p_scale alone carries the
   surface-pressure signal, exactly as an operational O2-A retrieval does.

Three pipelines, each retrieved at every row with the dispersion order
matched to what's physically appropriate for it (see pipeline_orders in
main() for the full rationale):
  "native"        (order=2 only) -- each row's own true per-pixel grid
                     (real keystone/smile/clocking baked in via the raw
                     render, real spatial PSF blur across rows).
  "rectified"     (order=2 only) -- rectified onto the shared nominal
                     wn_grid (adds bilinear-interpolation error on top of
                     "native"). Both of these have real wavelength-
                     calibration error for the dispersion polynomial to
                     absorb; order=0 has consistently been the noisy,
                     less-informative case for such pipelines throughout
                     this study, so it's skipped here to save compute.
  "undistorted"   (order=0 only) -- no keystone/smile/clocking/spatial
                     PSF at all: the true spectrum at each row's exact
                     intended slit position, evaluated directly (bypasses
                     gd_render.image()/rectify() entirely). A baseline for
                     whether a retrieval difficulty is a fundamental RT/
                     information-content limit or something geometric
                     distortion makes worse -- added 2026-07-24 to check
                     the H2O/p_scale degeneracy found near the pressure
                     mountain in the first FPA1 run (confirmed: the same
                     dip appears here too, with zero distortion, so it's
                     a real RT/retrieval limit). No real miscalibration
                     for dispersion to correct here, so order=2 is
                     skipped -- floating it would give those parameters
                     nothing to fit and risks spurious degeneracy with
                     h2o_scale/p_scale that would contaminate exactly the
                     question this baseline exists to answer.

Known ABSCO coverage requirement (found 2026-07-24): FPA3 (CH4_CO) needs
the ch4/h2o/co ABSCO blocks extended to ~4400 cm-1 (see
KEYSTONE_SMILE_BIAS_PLAN.md) -- this script will raise a clear ValueError
from gert.absco if that hasn't been done yet. FPA0/FPA1 have no known gaps.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_band_stress_test.py --fpa 1
Output: results/gd_band_stress_test_fpa<N>.pkl
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
from geocarb_gert import gd_render
from geocarb_gert import build_geocarb_instrument
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit
from geocarb_gert.gd_render import available_cpus, s_max
from geocarb_gert.gd_render import _diagonal_ils_convolve

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
REPO_ROOT = Path(__file__).resolve().parent.parent

# Gases whose true column is constant along the slit (see xtrue_of_row in
# main()) -- retrieving a separate {gas}_scale for these is nearly
# degenerate with p_scale (every StateVector floats p_scale by default),
# since both change a well-mixed gas's column via the same total-air-mass
# path. Excluded from `gases` even when present in a band's molecule set;
# p_scale alone carries the surface-pressure signal instead. See module
# docstring.
WELL_MIXED_GASES = {"o2", "n2o"}

# -- globals populated in main() before the Pool is forked --
_G = {}


def _retrieve(y_dist: np.ndarray, obs_grid: np.ndarray, order: int, xtrue_row: dict):
    g = _G
    win = SpectralWindow(wn_min=g["wn_min"], wn_max=g["wn_max"],
                         ils=ILS(type="gaussian", fwhm=g["fwhm_cm"]),
                         molecules=list(g["mols"]), label=g["label"], obs_grid=obs_grid)
    inst = Instrument(windows=[win], snr=300.0)
    # The retrieval's prior/forward-model atmosphere is deliberately the
    # fixed slit-centre state (g["atm"]), NOT the row's true state -- the
    # along-slit deviation from this prior is exactly what each row's
    # retrieval must recover via its gas-scale/p_scale state elements.
    fm = ForwardModel(g["atm"], g["absco"], inst, g["geo"],
                      solver=SingleScatterSolver(), solar_spectrum=g["solar"])
    sigma = np.maximum(0.003 * np.abs(y_dist).max(), 1e-6)
    Sy_inv = np.diag(np.full(len(y_dist), 1.0 / sigma ** 2))
    sv = StateVector.gas_scaling(prior_albedo=g["albedo"], prior_albedo_slope=np.zeros(1),
                                 gases=g["gases"],   # co2/ch4 uncertainties default to 0.10/0.20,
                                 gas_uncerts={"h2o": 0.60},  # default 0.10 is far tighter than this
                                 # scene's true h2o swing (+-56% of prior, i.e. ~5.6 prior-sigma at
                                 # 10%) -- found 2026-07-24 comparing to p_scale's default 10%, which
                                 # is a comfortable ~2.1 prior-sigma match for this scene's true
                                 # p_surface swing. 0.60 is a measured sweet spot, not a guess: FPA0
                                 # smoke tests at 10%/60%/200% gave h2o bias std of 1164/615/904 ppm
                                 # respectively -- looser isn't better past ~60%, since with little
                                 # prior regularization left the fit starts drifting on H2O's real
                                 # but weak sensitivity in this band (same failure flavor as the
                                 # o2_scale/p_scale ridge above, just softer). p_surface bias is
                                 # ~0.24 hPa std at all three settings -- this doesn't perturb that.
                                 include_dispersion=(order > 0),   # everything else 0.10 -- gas_scaling's own defaults
                                 dispersion_order=max(order, 0), dispersion_uncert=2.0)
    ret = GERTRetrieval(fm, y_dist, Sy_inv, sv, prior_albedo=g["albedo"],
                        prior_albedo_slope=np.zeros(1), analytical_jacobians=True,
                        max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
    names = sv.names
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            res = ret.run()
    except (ValueError, np.linalg.LinAlgError) as e:
        nl = {gas: np.nan for gas in g["gases"]}
        nl["p_surface"] = np.nan
        nl["_chi2"] = np.nan
        nl["_conv"] = False
        nl["_diverged"] = str(e)
        return nl, None
    # bias = retrieved column - TRUE column at this row (not the prior) --
    # retrieved = prior_atm_value * scale; true is xtrue_row (already in the
    # reported units, see gas_units in main()).
    nl = {}
    for gas in g["gases"]:
        prior_val = float(np.mean(g["atm"].gases[gas])) * g["gas_units"][gas]
        retrieved_val = prior_val * res.x_ret[names.index(gas + "_scale")]
        nl[gas] = retrieved_val - xtrue_row[gas]
    # p_scale is always in the state vector (StateVector.gas_scaling adds it
    # by default regardless of `gases`) -- track its bias unconditionally so
    # bands that exclude o2/n2o from `gases` (see WELL_MIXED_GASES) still
    # get a surface-pressure bias metric.
    p_scale_ret = float(res.x_ret[names.index("p_scale")])
    nl["p_surface"] = g["p_surface_prior"] * p_scale_ret - xtrue_row["p_surface"]
    nl["_chi2"] = float(res.chisq_reduced)
    nl["_conv"] = res.converged
    if res.diverged:
        nl["_diverged"] = "gert: chi2 non-finite"
        return nl, None
    residual = y_dist - res.y_ret
    return nl, residual


def _worker(task):
    pipeline, order, k = task
    g = _G
    xtrue_row = {gas: float(g["xtrue_of_row"][gas][k]) for gas in g["gases"]}
    xtrue_row["p_surface"] = float(g["xtrue_of_row"]["p_surface"][k])
    try:
        if pipeline == "native":
            cols = np.arange(1024.0)
            lam_row, _ = xy_to_wavelength_slit(g["fpa"], cols, np.full(1024, float(k)))
            nu_row = 1e4 / lam_row
            y_dist = g["A"][k, :]
            # gert's y/y_ret is always in ascending-wavelength (descending
            # wavenumber) order. Column order happens to already match that
            # for FPA2 (its real dispersion runs the other way), which is
            # why no reversal was ever needed there -- but FPA1's real
            # dispersion direction is the opposite (found 2026-07-24: every
            # native-pipeline retrieval failed catastrophically, even at
            # the slit centre where truth == prior, until this was fixed).
            # Check explicitly rather than assume either direction.
            if nu_row[0] < nu_row[-1]:   # ascending wavenumber -> wrong order, flip
                nu_row = nu_row[::-1]
                y_dist = y_dist[::-1]
            nl, resid = _retrieve(y_dist, nu_row, order, xtrue_row)
            nu_out = nu_row
        elif pipeline == "rectified":
            row = g["Rimg"][k, :]
            valid = ~np.isnan(row)
            if valid.sum() < 0.5 * len(row):
                return (pipeline, order, k), {"bias": None, "residual": None, "nu": None, "n_bad": int((~valid).sum())}
            nu_valid = g["wn_grid"][valid]
            y_dist = row[valid][::-1]
            nl, resid = _retrieve(y_dist, nu_valid, order, xtrue_row)
            nu_out = nu_valid[::-1]
            nl["_n_bad"] = int((~valid).sum())
        else:  # "undistorted" -- no keystone/smile/clocking, no spatial PSF:
            # the true hi-res spectrum at this row's exact intended slit
            # position, evaluated directly and ILS-convolved onto the
            # shared nominal wn_grid -- bypasses gd_render.image()/rectify()
            # entirely. Isolates the retrieval's inherent ability to
            # separate state-vector elements (e.g. H2O vs p_scale) given
            # only RT physics + noise + prior, with zero geometric-
            # distortion confound. Same trick used in Sec. 9l to isolate
            # pure rectification-interpolation error from the true spectrum.
            eta_row = g["x_km_of_row"][k] / als.SLIT_HALF_KM
            wn_grid = g["wn_grid"]
            S_row = np.asarray(g["radiance"](np.full(len(wn_grid), eta_row)), dtype=float)
            y_ideal = _diagonal_ils_convolve(g["wn_hires"], S_row, wn_grid, g["ils"])
            y_dist = y_ideal[::-1]   # ascending-wn_grid order -> ascending-wavelength (gert's convention)
            nl, resid = _retrieve(y_dist, wn_grid, order, xtrue_row)
            nu_out = wn_grid[::-1]
    except Exception as e:   # noqa: BLE001 -- keep the sweep alive
        nl = {gas: np.nan for gas in g["gases"]}
        nl["p_surface"] = np.nan
        nl["_chi2"] = np.nan
        nl["_conv"] = False
        nl["_diverged"] = f"{type(e).__name__}: {e}"
        resid, nu_out = None, None
    return (pipeline, order, k), {"bias": nl, "residual": resid, "nu": nu_out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpa", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--row-step", type=int, default=1, help="subsample rows (1 = every row)")
    ap.add_argument("--n-lookup-samples", type=int, default=400,
                    help="along-slit truth-atmosphere lookup-table density (default resolves "
                         "the ~10 km hot spots with ~3 samples across their FWHM)")
    ap.add_argument("--n-workers", type=int, default=None, help="default: all available cores")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="append _<tag> to the output filename (e.g. --out-tag smoketest) so a "
                         "test run never overwrites a real run's results/gd_band_stress_test_fpa<N>.pkl")
    args = ap.parse_args()
    FPA = args.fpa

    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[FPA]
    mols = list(mols)
    wn_min, wn_max = real_wavenumber_range(FPA, margin_cm1=10.0)
    print(f"FPA{FPA} ({label}): molecules={mols}  wn=[{wn_min:.2f},{wn_max:.2f}]", flush=True)

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))

    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=mols, label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=300.0)
    albedo = albedo_for(wide_inst, "desert")

    t0 = time.time()
    wn_hires, radiance = als.build_lookup_radiance(
        absco, wide_inst, geo, solar, albedo,
        n_samples=args.n_lookup_samples, n_workers=args.n_workers)
    print(f"lookup table built ({args.n_lookup_samples} samples, {time.time()-t0:.1f}s)", flush=True)

    A = gd_render.image(FPA, wn_hires, radiance, wide_win.ils, spatial_psf_fwhm_px=1.5,
                        n_workers=args.n_workers)
    print(f"raw render done ({time.time()-t0:.1f}s)", flush=True)

    nominal_inst = build_geocarb_instrument()
    nominal_win = nominal_inst.windows[FPA]
    wn_grid = nominal_win.wn_instrument
    sm = s_max(FPA)
    s_grid = np.linspace(-sm, sm, 1024)
    Rimg = gd_render.rectify(FPA, A, s_grid, wn_grid)
    print(f"rectified {Rimg.shape} ({time.time()-t0:.1f}s)", flush=True)

    # Retrieval prior/forward-model atmosphere: the fixed slit-centre state
    # (x_km=0), used for every row -- deliberately NOT each row's true
    # state (see _retrieve's docstring comment).
    atm_center = als.atmosphere_at(0.0)

    # -- along-slit truth at each row's real slit position, for scoring bias --
    # (row index -> real s -> eta -> x_km -> true gas column at that position)
    cols_center = np.full(1024, 512.0)
    _, s_of_row = xy_to_wavelength_slit(FPA, cols_center, np.arange(1024.0))
    x_km_of_row = (s_of_row / sm) * als.SLIT_HALF_KM
    # h2o's true/prior comparison must be on the same basis _retrieve() uses
    # for every gas (mean of the full vertical profile, since h2o_scale
    # multiplies the whole prior profile shape) -- not the surface VMR
    # directly, which would introduce a spurious constant offset since h2o
    # decays sharply with height. True and prior share the same scale
    # height (atmosphere_at always uses the default), so the ratio of
    # surface VMRs equals the ratio of profile means exactly.
    h2o_mean_prior_ppm = float(np.mean(atm_center.gases["h2o"])) * 1e6
    h2o_surface_ratio = als.h2o_surface_vmr(x_km_of_row) / als.h2o_surface_vmr(0.0)
    xtrue_of_row = {
        "co2": als.xco2_ppm(x_km_of_row),
        "ch4": als.xch4_ppb(x_km_of_row),
        "co": als.xco_ppb(x_km_of_row),
        "h2o": h2o_mean_prior_ppm * h2o_surface_ratio,   # ppm, profile-mean basis (see above)
        "o2": np.full(1024, 0.2095 * 1e6),                # well-mixed, not varied along slit
        "n2o": np.full(1024, 330.0),                      # well-mixed, not varied along slit (ppb)
        "p_surface": als.p_surface_hpa(x_km_of_row),      # hPa -- not a "gas", tracked via p_scale
    }
    gas_units = {"co2": 1e6, "ch4": 1e9, "co": 1e9, "h2o": 1e6, "o2": 1e6, "n2o": 1e9}
    p_surface_prior = float(als.p_surface_hpa(np.array([0.0]))[0])

    # h2o always last (cosmetic only); well-mixed gases (o2, n2o) excluded --
    # see WELL_MIXED_GASES.
    gases = [m for m in mols if m != "h2o" and m not in WELL_MIXED_GASES] + ["h2o"]

    rows = list(range(0, 1024, args.row_step))
    n_workers = args.n_workers if args.n_workers is not None else available_cpus()

    out = {}
    # "undistorted": no keystone/smile/clocking/spatial-PSF -- the true
    # spectrum at each row's exact intended slit position, evaluated
    # directly (see _worker). Isolates whether a retrieval difficulty
    # (e.g. the H2O/p_scale degeneracy near the pressure mountain, found
    # 2026-07-24 in the native/rectified FPA1 run) is a fundamental RT/
    # information-content limit -- present even with a perfect instrument
    # -- or something geometric distortion is making worse.
    #
    # Dispersion order is matched to what's physically appropriate per
    # pipeline rather than run uniformly: native/rectified have real
    # keystone/smile-driven wavelength-calibration error for the
    # dispersion polynomial to absorb (order=2 only -- order=0 has
    # consistently been the noisy, less-informative case throughout this
    # study for pipelines with real distortion, not worth the extra
    # compute). undistorted has zero such error by construction -- the
    # true spectrum is evaluated exactly on the nominal grid -- so
    # floating dispersion there would give those parameters nothing real
    # to fit, risking spurious degeneracy with h2o_scale/p_scale that
    # would contaminate exactly the "fundamental limit vs. distortion
    # artifact" question this baseline exists to answer (order=0 only,
    # keeping the state vector matched to the true generative model).
    pipeline_orders = {"native": [2], "rectified": [2], "undistorted": [0]}
    tasks = [(pipeline, order, k) for pipeline, orders in pipeline_orders.items()
            for order in orders for k in rows]
    print(f"{len(tasks)} retrievals ({len(rows)} rows x "
         f"{sum(len(v) for v in pipeline_orders.values())} pipeline/order combos), "
         f"{n_workers} workers, gases={gases}", flush=True)

    _G.update(dict(fpa=FPA, A=A, Rimg=Rimg, wn_grid=wn_grid, wn_hires=wn_hires, radiance=radiance,
                   ils=wide_win.ils, wn_min=wn_min, wn_max=wn_max,
                   fwhm_cm=fwhm_cm, mols=mols, label=label, atm=atm_center, absco=absco,
                   geo=geo, solar=solar, albedo=albedo, gases=gases, p_surface_prior=p_surface_prior,
                   xtrue_of_row=xtrue_of_row, gas_units=gas_units, x_km_of_row=x_km_of_row))

    ctx = mp.get_context("fork")
    n_done = 0
    with ctx.Pool(n_workers) as pool:
        for key, val in pool.imap_unordered(_worker, tasks, chunksize=4):
            out[key] = val
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"all done ({time.time()-t0:.0f}s)", flush=True)
    n_diverged = sum(1 for v in out.values() if v["bias"] is not None and v["bias"].get("_diverged"))
    n_offdet = sum(1 for v in out.values() if v["bias"] is None)
    print(f"diverged: {n_diverged}, off-detector (rectified only): {n_offdet}, total: {len(out)}")

    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    out_path = REPO_ROOT / "results" / f"gd_band_stress_test_fpa{FPA}{tag_suffix}.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "rows": rows, "wn_grid": wn_grid, "s_grid": s_grid,
                    "A_raw": A, "R_rectified": Rimg, "FPA": FPA, "gases": gases,
                    "xtrue_of_row": xtrue_of_row, "x_km_of_row": x_km_of_row}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
