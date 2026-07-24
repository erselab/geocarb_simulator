#!/usr/bin/env python3
"""Dense along-slit sweep: retrieve at (nearly) every row, with and without
rectification, to look for systematic residual patterns behind the
rectification-interpolation bias found in gd_rectify_retrieve.py
(KEYSTONE_SMILE_BIAS_PLAN.md Sec. 9l).

Two independent retrieval pipelines, run at the same dense set of row
indices, on the *same* underlying uniform-desert render (results/
gd_rectify_retrieve.pkl -- reused, not re-rendered):

  "native"     -- retrieve directly on row k's own true per-pixel grid
                  (A[k, :], obs_grid = that row's real per-column
                  wavelengths). This is the §9h/9j pipeline: never showed
                  more than a few ppm of bias.
  "rectified"  -- retrieve on rectified row k (Rimg[k, :], shared nominal
                  wn_grid). This is the §9l pipeline: showed a robust
                  -50 to -65 ppm bias, roughly constant across the slit.

Rows are independent retrievals, so this is embarrassingly parallel --
uses multiprocessing with the 'fork' start method so all worker processes
share the ~2.2GB ABSCO table via copy-on-write (loaded once in the parent,
before the Pool is created) rather than each worker reloading it.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_dense_sweep.py
Output: results/gd_dense_sweep.pkl
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

import geosat_geometry as gg
from geocarb_gert import GEOCARB_BANDS, albedo_for, reference_atmosphere, sample_geometries
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit
from geocarb_gert.gd_render import available_cpus

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
ROW_STEP = 1                 # every row; set >1 to subsample
N_WORKERS = available_cpus()

# -- globals populated in main() before the Pool is forked, so every worker
# -- inherits them via copy-on-write instead of re-loading/re-pickling --
_G = {}


def _retrieve(y_dist: np.ndarray, obs_grid: np.ndarray, order: int):
    g = _G
    win = SpectralWindow(wn_min=g["wn_min"], wn_max=g["wn_max"],
                         ils=ILS(type="gaussian", fwhm=g["fwhm_cm"]),
                         molecules=list(g["mols"]), label=g["label"], obs_grid=obs_grid)
    inst = Instrument(windows=[win], snr=300.0)
    fm = ForwardModel(g["atm"], g["absco"], inst, g["geo"],
                      solver=SingleScatterSolver(), solar_spectrum=g["solar"])
    sigma = np.maximum(0.003 * np.abs(y_dist).max(), 1e-6)
    Sy_inv = np.diag(np.full(len(y_dist), 1.0 / sigma ** 2))
    sv = StateVector.gas_scaling(prior_albedo=g["albedo_desert"], prior_albedo_slope=np.zeros(1),
                                 gases=GASES, include_dispersion=(order > 0),
                                 dispersion_order=max(order, 0), dispersion_uncert=2.0)
    ret = GERTRetrieval(fm, y_dist, Sy_inv, sv, prior_albedo=g["albedo_desert"],
                        prior_albedo_slope=np.zeros(1), analytical_jacobians=True,
                        max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
    names = sv.names
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            res = ret.run()
    except (ValueError, np.linalg.LinAlgError) as e:
        nl = {gas: np.nan for gas in ("co2", "ch4", "co")}
        nl["_chi2"] = np.nan
        nl["_conv"] = False
        nl["_diverged"] = str(e)
        return nl, None
    nl = {gas: g["xtrue"][gas] * (res.x_ret[names.index(gas + "_scale")] - 1.0)
         for gas in ("co2", "ch4", "co")}
    nl["_chi2"] = float(res.chisq_reduced)
    nl["_conv"] = res.converged
    if res.diverged:   # gert-reported: chi2 went non-finite, not just an exception
        nl["_diverged"] = "gert: chi2 non-finite"
        return nl, None
    residual = y_dist - res.y_ret
    return nl, residual


def _worker(task):
    pipeline, order, k = task
    g = _G
    try:
        if pipeline == "native":
            cols = np.arange(1024.0)
            lam_row, _ = xy_to_wavelength_slit(FPA, cols, np.full(1024, float(k)))
            nu_row = 1e4 / lam_row          # descending wn == ascending wavelength, matches gert's y order
            y_dist = g["A"][k, :]
            nl, resid = _retrieve(y_dist, nu_row, order)
            nu_out = nu_row
        else:  # rectified
            row = g["Rimg"][k, :]
            valid = ~np.isnan(row)
            if valid.sum() < 0.5 * len(row):
                return (pipeline, order, k), {"bias": None, "residual": None, "nu": None, "n_bad": int((~valid).sum())}
            nu_valid = g["wn_grid"][valid]
            y_dist = row[valid][::-1]       # reverse: Rimg columns are ascending-wn, gert's y is ascending-wavelength
            nl, resid = _retrieve(y_dist, nu_valid, order)
            nu_out = nu_valid[::-1]
            nl["_n_bad"] = int((~valid).sum())
    except Exception as e:   # noqa: BLE001 -- keep the sweep alive; row-25-style failures are the point of this test
        nl = {"co2": np.nan, "ch4": np.nan, "co": np.nan, "_chi2": np.nan, "_conv": False,
             "_diverged": f"{type(e).__name__}: {e}"}
        resid, nu_out = None, None
    return (pipeline, order, k), {"bias": nl, "residual": resid, "nu": nu_out}


def main() -> int:
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[FPA]
    wn_min, wn_max = real_wavenumber_range(FPA, margin_cm1=10.0)

    with open(REPO_ROOT / "results" / "gd_rectify_retrieve.pkl", "rb") as f:
        d = pickle.load(f)
    A, Rimg, wn_grid = d["A_raw"], d["R_rectified"], d["wn_grid"]

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
    albedo_desert = albedo_for(wide_inst, "desert")
    xtrue = {g: atm.column_xgas(g) * (1e6 if g == "co2" else 1e9) for g in ("co2", "ch4", "co")}

    _G.update(dict(A=A, Rimg=Rimg, wn_grid=wn_grid, wn_min=wn_min, wn_max=wn_max,
                   fwhm_cm=fwhm_cm, mols=mols, label=label, atm=atm, absco=absco,
                   geo=geo, solar=solar, albedo_desert=albedo_desert, xtrue=xtrue))

    rows = list(range(0, 1024, ROW_STEP))
    tasks = [(pipeline, order, k) for pipeline in ("native", "rectified")
            for order in (0, 2) for k in rows]
    print(f"{len(tasks)} retrievals ({len(rows)} rows x 2 pipelines x 2 orders), "
         f"{N_WORKERS} workers", flush=True)

    t0 = time.time()
    ctx = mp.get_context("fork")
    out = {}
    n_done = 0
    with ctx.Pool(N_WORKERS) as pool:
        for key, val in pool.imap_unordered(_worker, tasks, chunksize=4):
            out[key] = val
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"all done ({time.time()-t0:.0f}s)", flush=True)

    n_diverged = sum(1 for v in out.values() if v["bias"] is not None and v["bias"].get("_diverged"))
    n_offdet = sum(1 for v in out.values() if v["bias"] is None)
    print(f"diverged: {n_diverged}, off-detector (rectified only): {n_offdet}, total: {len(out)}")

    out_path = REPO_ROOT / "results" / "gd_dense_sweep.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "rows": rows, "wn_grid": wn_grid, "FPA": FPA,
                    "row_step": ROW_STEP}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
