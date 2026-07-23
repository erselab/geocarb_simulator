#!/usr/bin/env python3
"""Rectify a real-GD-curve rendered image, then retrieve on the rectified grid.

This is the "verification" test proposed in conversation on 2026-07-23: does
the gd_render simulator, run through the *same two-stage processing* as the
real ground-test analysis (rectify raw detector data onto a regular grid,
then retrieve on the rectified spectra -- see KEYSTONE_SMILE_BIAS_PLAN.md
Sec. 9, `keystone_report.pdf` Sec. 4.8), produce the same qualitative
signatures (large bias, poor convergence away from the sweet row, line-
correlated residual striping) that were seen in the real data? If so, that's
evidence the infrastructure is realistic enough for the planned OSSEs.

Two things this does differently from every gd_render retrieval test so far
in this study (which all used a per-row *native* obs_grid, exactly matching
each row's own true channel centres, specifically to remove the grid-
mismatch confound and isolate the underlying distortion mechanisms):

1. Rectifies the raw image onto a *regular* (slit, wavenumber) grid via the
   inverse polynomial mapping (gd_render.rectify, using wavelength_slit_to_xy
   / the C-D pair) -- the same direction a real L1B pipeline uses.
2. Retrieves every rectified row against the *same shared* nominal
   instrument grid (build_geocarb_instrument()'s own FPA2 window) instead of
   a per-row custom grid -- because after rectification there's no reason
   for row-specific grids any more; any remaining bias is now specifically
   *rectification error*, not a raw grid-mismatch artifact.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_rectify_retrieve.py
Output: results/gd_rectify_retrieve.pkl, plots/gd_raw_vs_rectified_fpa2.png
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

import geosat_geometry as gg
from geocarb_gert import (GEOCARB_BANDS, albedo_for, build_geocarb_instrument,
                          reference_atmosphere, sample_geometries)
from geocarb_gert import gd_render
from geocarb_gert.focalplane import uniform_scene
from geocarb_gert.gd_polynomials import real_wavenumber_range, rows_crossed
from geocarb_gert.gd_render import s_max

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
REPO_ROOT = Path(__file__).resolve().parent.parent
FPA = 2
GASES = ["co2", "ch4", "co", "h2o"]


def main() -> int:
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[FPA]
    wn_min, wn_max = real_wavenumber_range(FPA, margin_cm1=10.0)

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    atm = reference_atmosphere()
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))

    # -- render: real curves, uniform desert scene (same truth as Sec. 9h) --
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=list(mols), label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=300.0)
    fm_wide = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    albedo_desert = albedo_for(wide_inst, "desert")
    res0 = fm_wide.run(albedo=list(albedo_desert), albedo_slope=[0.0])
    wn_hires, S_desert = res0.wn_band_hires[0], res0.I_hires[0]
    radiance = uniform_scene(S_desert)

    t0 = time.time()
    A = gd_render.image(FPA, wn_hires, radiance, wide_win.ils, spatial_psf_fwhm_px=1.5)
    print(f"raw render done ({time.time()-t0:.1f}s)", flush=True)

    # -- rectify onto a regular grid: 1024 slit positions x the STANDARD --
    # -- nominal FPA2 instrument wavenumber grid (build_geocarb_instrument) --
    nominal_inst = build_geocarb_instrument()
    nominal_win = nominal_inst.windows[FPA]
    wn_grid = nominal_win.wn_instrument
    sm = s_max(FPA)
    s_grid = np.linspace(-sm, sm, 1024)

    Rimg = gd_render.rectify(FPA, A, s_grid, wn_grid)
    print(f"rectified ({Rimg.shape}) ({time.time()-t0:.1f}s)", flush=True)

    # -- retrieve on rectified rows, nominal grid restricted to that row's
    # -- valid (on-detector) channels, with/without dispersion --
    xtrue = {g: atm.column_xgas(g) * (1e6 if g == "co2" else 1e9) for g in ("co2", "ch4", "co")}

    def retrieve_rectified_row(k, order):
        row = Rimg[k, :]
        valid = ~np.isnan(row)
        n_bad = (~valid).sum()
        if valid.sum() < 0.5 * len(row):        # essentially the whole row off-detector
            return None, None, n_bad
        # Rimg's columns follow wn_grid ascending (wavenumber order). GERT's
        # ForwardModel always returns y in *wavelength* order (SpectralWindow.
        # wl_instrument = wn_instrument[::-1] -- see gert/forward_model.py's
        # "Reverses to wavelength order" step) regardless of what order
        # obs_grid is given in (it's sorted ascending internally). So the
        # measured vector must be reversed to line up index-for-index with
        # the model's y / y_ret -- otherwise every channel is compared
        # against the wrong model channel, which was silently scrambling
        # every rectified-grid retrieval in this script (caught 2026-07-23:
        # iter-0 chi2 ~12700 collapsed to a sane value once reversed).
        y_dist = row[valid][::-1]
        nu_valid = wn_grid[valid]
        win_row = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                                 molecules=list(mols), label=label, obs_grid=nu_valid)
        inst_row = Instrument(windows=[win_row], snr=300.0)
        fm_row = ForwardModel(atm, absco, inst_row, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
        sigma = np.maximum(0.003 * np.abs(y_dist).max(), 1e-6)
        Sy_inv = np.diag(np.full(len(y_dist), 1.0 / sigma ** 2))
        sv = StateVector.gas_scaling(prior_albedo=albedo_desert, prior_albedo_slope=np.zeros(1),
                                     gases=GASES, include_dispersion=(order > 0),
                                     dispersion_order=max(order, 0), dispersion_uncert=2.0)
        ret = GERTRetrieval(fm_row, y_dist, Sy_inv, sv, prior_albedo=albedo_desert,
                            prior_albedo_slope=np.zeros(1), analytical_jacobians=True,
                            max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
        names = sv.names
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                res = ret.run()
        except (ValueError, np.linalg.LinAlgError) as e:
            # Gauss-Newton diverged into an unphysical atmospheric state --
            # without dispersion to absorb a large mismatch this can overshoot
            # rather than just fail to converge (KEYSTONE_SMILE_BIAS_PLAN.md
            # Sec. 9j hit the same failure mode). Real ground-test retrievals
            # away from the sweet row reportedly failed to converge too --
            # recording this as a diverged case is truer to that than letting
            # the whole sweep crash on one bad row.
            nl = {g: np.nan for g in ("co2", "ch4", "co")}
            nl["_chi2"] = np.nan
            nl["_conv"] = False
            nl["_n_bad"] = int(n_bad)
            nl["_diverged"] = str(e)
            return nl, None, nu_valid[::-1]
        nl = {g: xtrue[g] * (res.x_ret[names.index(g + "_scale")] - 1.0) for g in ("co2", "ch4", "co")}
        nl["_chi2"] = float(res.chisq_reduced)
        nl["_conv"] = res.converged
        nl["_n_bad"] = int(n_bad)
        residual = y_dist - res.y_ret
        return nl, residual, nu_valid[::-1]

    test_k = [25, 100, 300, 512, 700, 900, 950]   # same rows used throughout Sec. 9h for comparability
    print("rows_crossed at test rows:", dict(zip(test_k, np.round(rows_crossed(FPA, test_k), 2))), flush=True)
    out = {}
    for order in (0, 2):
        for k in test_k:
            nl, resid, nu_valid = retrieve_rectified_row(k, order)
            out[(order, k)] = {"bias": nl, "residual": resid, "nu_valid": nu_valid}
            if nl is None:
                print(f"order={order} row={k:4d}  <no data (off detector)>  ({time.time()-t0:.1f}s)", flush=True)
            elif nl.get("_diverged"):
                print(f"order={order} row={k:4d}  <DIVERGED: {nl['_diverged'][:60]}>  ({time.time()-t0:.1f}s)", flush=True)
            else:
                print(f"order={order} row={k:4d}  chi2={nl['_chi2']:8.4f} conv={nl['_conv']!s:5s} "
                     f"n_bad={nl['_n_bad']:3d}  co2={nl['co2']:+8.4f}  ({time.time()-t0:.1f}s)", flush=True)

    out_path = REPO_ROOT / "results" / "gd_rectify_retrieve.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"A_raw": A, "R_rectified": Rimg, "s_grid": s_grid, "wn_grid": wn_grid,
                    "out": out, "test_k": test_k, "FPA": FPA}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
