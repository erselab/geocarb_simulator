#!/usr/bin/env python3
"""Spectral residuals for a real-GD-curve keystone-transition retrieval test.

Renders a desert/water transition scene (the highest-contrast surface pair)
through geocarb_gert.gd_render at FPA2 (strong CO2), with scene boundaries
placed at the same 6 keystone-spanning rows used in the systematic
12-transition test (KEYSTONE_SMILE_BIAS_PLAN.md Sec. 9j), then retrieves at
rows +/-10 from the two highest-keystone boundaries (700, 850) -- with and
without dispersion floated -- saving both the aggregate bias/chi2 and the
full post-fit spectral residual (y_dist - y_ret) for each case.

This is a companion to the 9j systematic test, which only recorded the
aggregate bias/chi2 (a clean null result there) -- this script exists to look
at the *residual shape*, not just whether it moved the retrieved gas amount.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_transition_residuals.py
Output: results/gd_transitions_residuals.pkl
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

import geosat_geometry as gg
from geocarb_gert import (GEOCARB_BANDS, SCENE_TYPES, albedo_for,
                          hires_spectra_for, reference_atmosphere,
                          sample_geometries)
from geocarb_gert import gd_render
from geocarb_gert.focalplane import _segment_blend
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
REPO_ROOT = Path(__file__).resolve().parent.parent
FPA = 2
BOUNDARY_ROWS = [100, 250, 400, 550, 700, 850]
TEST_ROWS = [690, 710, 840, 860]      # boundary +/-10 at the two highest-keystone boundaries
GASES = ["co2", "ch4", "co", "h2o"]


def transition_scene(S_A, S_B, boundaries, softness=0.0):
    seg_matrix = np.stack([S_A if k % 2 == 0 else S_B for k in range(len(boundaries) + 1)])
    return _segment_blend(boundaries, seg_matrix, softness)


def main() -> int:
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[FPA]
    wn_min, wn_max = real_wavenumber_range(FPA, margin_cm1=10.0)

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    atm = reference_atmosphere()
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))

    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=list(mols), label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=300.0)
    fm_wide = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    albedo_desert = albedo_for(wide_inst, "desert")
    ils_render = wide_win.ils

    hires = hires_spectra_for(fm_wide)
    wn_hires = hires["desert"].wn_band_hires[0]
    S = {s: hires[s].I_hires[0] for s in SCENE_TYPES}

    xtrue = {g: atm.column_xgas(g) * (1e6 if g == "co2" else 1e9) for g in ("co2", "ch4", "co")}
    cols = np.arange(1024.0)
    sm = gd_render.s_max(FPA)
    _, s_b = xy_to_wavelength_slit(FPA, np.full(len(BOUNDARY_ROWS), 512.0),
                                   np.array(BOUNDARY_ROWS, dtype=float))
    boundary_eta = np.array(s_b) / sm

    def retrieve_row(A, i, order):
        lam_row, _ = xy_to_wavelength_slit(FPA, cols, np.full(1024, float(i)))
        nu_row = 1e4 / lam_row
        y_dist = A[i, :]
        sigma = np.maximum(0.003 * np.abs(y_dist).max(), 1e-6)
        Sy_inv = np.diag(np.full(1024, 1.0 / sigma ** 2))
        win_row = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                                 molecules=list(mols), label=label, obs_grid=nu_row)
        inst_row = Instrument(windows=[win_row], snr=300.0)
        fm_row = ForwardModel(atm, absco, inst_row, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
        sv = StateVector.gas_scaling(prior_albedo=albedo_desert, prior_albedo_slope=np.zeros(1),
                                     gases=GASES, include_dispersion=(order > 0),
                                     dispersion_order=max(order, 0), dispersion_uncert=2.0)
        ret = GERTRetrieval(fm_row, y_dist, Sy_inv, sv, prior_albedo=albedo_desert,
                            prior_albedo_slope=np.zeros(1), analytical_jacobians=True,
                            max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
        res = ret.run()
        names = sv.names
        nl = {g: xtrue[g] * (res.x_ret[names.index(g + "_scale")] - 1.0) for g in ("co2", "ch4", "co")}
        nl["_chi2"] = float(res.chisq_reduced)
        nl["_conv"] = res.converged
        residual = y_dist - res.y_ret
        return nl, residual, nu_row

    t0 = time.time()
    radiance = transition_scene(S["desert"], S["water"], boundary_eta, softness=0.0)
    Aimg = gd_render.image(FPA, wn_hires, radiance, ils_render, spatial_psf_fwhm_px=1.5)
    print(f"render done ({time.time()-t0:.1f}s)", flush=True)

    out = {}
    for order in (0, 2):
        for i in TEST_ROWS:
            nl, resid, nu_row = retrieve_row(Aimg, i, order)
            out[(order, i)] = {"bias": nl, "residual": resid, "nu_row": nu_row}
            print(f"order={order} row={i} chi2={nl['_chi2']:.4f} co2={nl['co2']:+.4f} "
                 f"({time.time()-t0:.1f}s)", flush=True)

    out_path = REPO_ROOT / "results" / "gd_transitions_residuals.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "test_rows": TEST_ROWS, "boundary_rows": BOUNDARY_ROWS,
                    "pair": ("desert", "water"), "FPA": FPA}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
