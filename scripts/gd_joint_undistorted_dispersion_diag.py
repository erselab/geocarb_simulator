#!/usr/bin/env python3
"""Diagnostic (not part of the regular battery): does floating dispersion
(order=2) for the "undistorted" pipeline collapse its chi2/gas-bias scatter
down toward native's level?

Follows up the 2026-07-28 investigation into why FPA0+FPA3's undistorted
pipeline showed much larger scatter than native for uniform/no-noise, even
though both pipelines' measurement and forward-model residual at the TRUE
prior state were confirmed near-identical (std ~0.073 FPA0, ~0.011 FPA3,
verified directly, no optimizer involved) -- i.e. a small, pre-existing
forward-model-vs-truth-generation mismatch (inherited from `als.
build_lookup_radiance`/`_band_setup`, predating today's N-band work) that
native's 6 floating dispersion nuisance parameters (order=2) happen to
absorb, while undistorted (order=0 by convention -- "no genuine calibration
error to correct") has nothing to absorb it with, leaving the full floor
exposed as chi2~0.05 and correspondingly large gas-bias scatter.

This script tests that explanation directly: run undistorted with
order=2 (dispersion fitting ON, same 6 extra state elements native gets)
alongside the regular native (order=2) and undistorted (order=0), for a
representative subset of rows across 3 scenes (uniform/no-noise, uniform/
noise, realistic/no-noise), and compare chi2/gas-bias distributions. If the
explanation is right, undistorted-with-dispersion's chi2 and scatter should
collapse toward native's, not stay at the order=0 undistorted level.

Ad hoc diagnostic, not a battery script: prints comparison stats directly,
does not save a results pkl (nothing here is meant to be replotted later by
gd_joint_band_plot.py/gd_joint_band_plot_residual_spectra.py).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python \\
        scripts/gd_joint_undistorted_dispersion_diag.py
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geosat_geometry as gg
from geocarb_gert import along_slit_scene as als, sample_geometries
from geocarb_gert.cross_band import nearest_row_pairing_multi
import gert

from gd_joint_band_test import (
    _band_setup, _native_row, _undistorted_row, _joint_retrieve,
    DEFAULT_SNR_BY_FPA, _G,
)

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
FPAS = [0, 3]
SCENES = [("uniform", False), ("uniform", True), ("realistic", False)]
ROW_STEP = 40  # ~25 rows per scene, enough to see a distribution cheaply


def _worker(task):
    variant, order, rows = task
    g = _G
    bands = g["bands"]
    xtrue_rows = []
    for b, r in zip(bands, rows):
        xt = {gas: float(b["xtrue_of_row"][gas][r]) for gas in ("co2", "ch4", "co", "h2o")}
        xt["p_surface"] = float(b["xtrue_of_row"]["p_surface"][r])
        xtrue_rows.append(xt)
    rowfn = _native_row if variant == "native" else _undistorted_row
    nus, ys = zip(*[rowfn(b, r) for b, r in zip(bands, rows)])
    try:
        nl = _joint_retrieve(list(nus), list(ys), order, xtrue_rows, bands)
    except Exception as e:  # noqa: BLE001
        nl = {"_chi2": np.nan, "_conv": False, "_diverged": f"{type(e).__name__}: {e}"}
    return (variant, order, rows), nl


def _gas_list(out):
    gases = set()
    for nl in out.values():
        if not nl.get("_conv") or nl.get("_diverged"):
            continue
        for k in nl.keys():
            if k == "p_surface" or k.startswith("_"):
                continue
            gases.add(k)
    return sorted(gases)


def run_scene(scene: str, noise: bool):
    uniform = scene == "uniform"
    barcode = scene == "barcode"
    snr = min(DEFAULT_SNR_BY_FPA[fpa] for fpa in FPAS)

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)

    bands = [_band_setup(fpa, atm_center, absco, geo, solar, snr, 400, None,
                         uniform, barcode, 32, noise, 0)
             for fpa in FPAS]
    _G.update(dict(bands=bands, atm=atm_center, absco=absco, geo=geo, solar=solar, snr=snr))

    rows_ref = np.arange(0, 1024, ROW_STEP, dtype=float)
    pairing = nearest_row_pairing_multi(FPAS, rows_ref)
    n_valid = len(pairing["rows"][FPAS[0]])
    row_tuples = [tuple(int(pairing["rows"][fpa][i]) for fpa in FPAS) for i in range(n_valid)]

    tasks = []
    for rows in row_tuples:
        tasks.append(("native", 2, rows))
        tasks.append(("undistorted", 0, rows))
        tasks.append(("undistorted", 2, rows))  # the test: dispersion ON for undistorted

    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(min(len(tasks), 16)) as pool:
        for key, nl in pool.imap_unordered(_worker, tasks, chunksize=2):
            out[key] = nl

    gas_list = _gas_list(out)
    label = f"{scene}{' noise' if noise else ''}"
    print(f"\n=== FPA0+FPA3 -- {label} ({n_valid} rows) ===")
    for variant, order in [("native", 2), ("undistorted", 0), ("undistorted", 2)]:
        keys = [k for k in out if k[0] == variant and k[1] == order]
        chi2 = np.array([out[k]["_chi2"] for k in keys
                         if out[k].get("_conv") and not out[k].get("_diverged")])
        n_conv = len(chi2)
        tag = f"{variant} (order={order})"
        if n_conv == 0:
            print(f"  {tag:28s}: 0/{len(keys)} converged")
            continue
        gas_str = ""
        for g in gas_list:
            vals = np.array([out[k].get(g, np.nan) for k in keys
                             if out[k].get("_conv") and not out[k].get("_diverged")])
            gas_str += f" {g} std={np.std(vals):.3f}"
        print(f"  {tag:28s}: {n_conv}/{len(keys)} converged  chi2 median={np.median(chi2):.4g} "
             f"p95={np.percentile(chi2, 95):.4g} max={np.max(chi2):.4g} |{gas_str}")


def main() -> int:
    t0 = time.time()
    for scene, noise in SCENES:
        run_scene(scene, noise)
    print(f"\ndone ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
