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
   own appropriate gas set (e.g. FPA0/O2_A retrieves o2+h2o -- o2 is the
   real surface-pressure-sensing channel; FPA1/CO2_weak retrieves co2+h2o;
   FPA3/CH4_CO retrieves ch4+co+h2o) plus p_scale, which every band's
   StateVector already includes by default.

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

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
REPO_ROOT = Path(__file__).resolve().parent.parent

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
        else:  # rectified
            row = g["Rimg"][k, :]
            valid = ~np.isnan(row)
            if valid.sum() < 0.5 * len(row):
                return (pipeline, order, k), {"bias": None, "residual": None, "nu": None, "n_bad": int((~valid).sum())}
            nu_valid = g["wn_grid"][valid]
            y_dist = row[valid][::-1]
            nl, resid = _retrieve(y_dist, nu_valid, order, xtrue_row)
            nu_out = nu_valid[::-1]
            nl["_n_bad"] = int((~valid).sum())
    except Exception as e:   # noqa: BLE001 -- keep the sweep alive
        nl = {gas: np.nan for gas in g["gases"]}
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
    }
    gas_units = {"co2": 1e6, "ch4": 1e9, "co": 1e9, "h2o": 1e6, "o2": 1e6, "n2o": 1e9}

    gases = [m for m in mols if m != "h2o"] + ["h2o"]   # h2o always last, cosmetic only

    rows = list(range(0, 1024, args.row_step))
    n_workers = args.n_workers if args.n_workers is not None else available_cpus()

    out = {}
    tasks = [(pipeline, order, k) for pipeline in ("native", "rectified")
            for order in (0, 2) for k in rows]
    print(f"{len(tasks)} retrievals ({len(rows)} rows x 2 pipelines x 2 orders), "
         f"{n_workers} workers, gases={gases}", flush=True)

    _G.update(dict(fpa=FPA, A=A, Rimg=Rimg, wn_grid=wn_grid, wn_min=wn_min, wn_max=wn_max,
                   fwhm_cm=fwhm_cm, mols=mols, label=label, atm=atm_center, absco=absco,
                   geo=geo, solar=solar, albedo=albedo, gases=gases,
                   xtrue_of_row=xtrue_of_row, gas_units=gas_units))

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

    out_path = REPO_ROOT / "results" / f"gd_band_stress_test_fpa{FPA}.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "rows": rows, "wn_grid": wn_grid, "s_grid": s_grid,
                    "A_raw": A, "R_rectified": Rimg, "FPA": FPA, "gases": gases,
                    "xtrue_of_row": xtrue_of_row, "x_km_of_row": x_km_of_row}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
