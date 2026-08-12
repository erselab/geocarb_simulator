#!/usr/bin/env python3
"""Full along-slit sweep of the trace-and-select retrieval
(`JOINT_ROW_INVERSION_PLAN.md` §0/§0a), for direct row-by-row comparison
against native/undistorted's already-existing `results/gd_joint_fpa2.pkl`.

For every row (or every `--row-step`'th row), builds the trace-and-select
`obs_grid` -- real, un-interpolated pixels within `--tolerance-km` of that
row's own eta0 (col=512 convention), scattered across a keystone-sized
neighborhood, exactly as `gd_toy_trace_retrieve.py`'s single-row go/no-go
test does -- and retrieves against it using the same `gd_test.
_joint_retrieve()` call native/undistorted/rectified already use. No new
solver, no interpolation of any real pixel value; just a different
`obs_grid` per row than native's own 1024 columns.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_trace_retrieve_sweep.py \\
        --tolerance-km 0.5 [--row-step 1] [--n-workers N]
Output: results/gd_trace_fpa2_tol{tolerance_km}.pkl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_test as gdt  # noqa: E402
from gd_toy_trace_retrieve import retrieve_at  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.gd_render import available_cpus  # noqa: E402

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
FPA = 2

_SWEEP = {}


def _worker(row: int):
    band = _SWEEP["band"]
    tol = _SWEEP["tolerance_km"]
    order = _SWEEP["order"]
    try:
        nl, n_pixels, rows_used = retrieve_at(FPA, band, row, tol, order)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        nl = {"_chi2": np.nan, "_conv": False, "_diverged": f"{type(e).__name__}: {e}"}
        n_pixels, rows_used = 0, []
    return row, nl, n_pixels, rows_used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance-km", type=float, default=0.5)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--row-step", type=int, default=1)
    ap.add_argument("--row-min", type=int, default=0)
    ap.add_argument("--row-max", type=int, default=1023)
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()

    print(f"Building realistic-scene FPA{FPA} band (renders the real 1024x1024 detector image)...", flush=True)
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

    rows = np.arange(args.row_min, args.row_max + 1, args.row_step)
    _SWEEP.update(dict(band=band, tolerance_km=args.tolerance_km, order=args.order))

    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    print(f"{len(rows)} rows, tolerance={args.tolerance_km:g} km, {n_workers} workers", flush=True)

    t0 = time.time()
    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for row, nl, n_pixels, rows_used in pool.imap_unordered(_worker, rows, chunksize=2):
            out[int(row)] = {"nl": nl, "n_pixels": n_pixels, "rows_used": rows_used}
            n_done += 1
            if n_done % 20 == 0 or n_done == len(rows):
                elapsed = time.time() - t0
                rate = elapsed / n_done
                eta = rate * (len(rows) - n_done)
                print(f"  {n_done}/{len(rows)} done ({elapsed:.0f}s, {rate:.2f}s/row, "
                     f"~{eta:.0f}s remaining)", flush=True)

    n_conv = sum(1 for v in out.values() if v["nl"].get("_conv") and not v["nl"].get("_diverged"))
    print(f"\nall done ({time.time()-t0:.0f}s): converged {n_conv}/{len(rows)}", flush=True)

    out_path = REPO_ROOT / "results" / f"gd_trace_fpa{FPA}_tol{args.tolerance_km:g}.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "fpa": FPA, "tolerance_km": args.tolerance_km,
                    "order": args.order, "row_step": args.row_step}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
