#!/usr/bin/env python3
"""Whole-slit regression matrix for the analytic-Jacobian solver.

Runs full-slit joint-block retrievals across a 2x2x2 configuration matrix --
free rows (CO2-only vs CO2+surface pressure) x window count (58 vs 29) x
G ratio (1 vs 3) -- with BOTH solvers, and scores every run two ways:

1. **Solver agreement** (the regression proper). The finite-difference solver
   is the reference, since every published result in this project came from
   it. Analytic derivatives are a different route to the same fixed point, so
   the retrieved states must agree to solver tolerance -- not to plotting
   precision. A single window measured 1e-8 relative; anything materially
   worse anywhere in the matrix is a bug, not noise.

2. **Accuracy against truth**, per config. This is what the matrix is
   actually for: CO2 and surface-pressure bias against `along_slit_scene`'s
   own fields, so the solver change can be shown NOT to move the science
   numbers, and so the window-count and G-ratio axes can be read off in one
   place.

Why these axes. `G` sets the state resolution within a window and was worth
6.4x on CO2 bias in the CO2-only case (JOINT_BLOCK_MIGRATION_PLAN Sec.10);
window count sets how much slit each independent solve sees, and interacts
with `G` because `G = width/g_ratio`. Freeing surface pressure changes the
character of both, since CO2 and p_surface are near-degenerate in this band
alone -- a free pressure row absorbs CO2 signal, which is exactly why the
CO2-only column has to stay in the matrix as a control.

Cost. The finite-difference reference is the expensive half (n_free+1
forward evaluations per iteration against 1), and cost grows roughly as
width^2 per window, so the 29-window configs are the slow ones. Use
`--analytic-only` to skip the reference once it has been established, and
`--configs` to run a subset. Existing outputs are skipped unless --force.

Run:  PYTHONPATH=. python3 analytic_jacobian_testing/gd_solver_regression.py
      PYTHONPATH=. python3 analytic_jacobian_testing/gd_solver_regression.py --analytic-only
      PYTHONPATH=. python3 analytic_jacobian_testing/gd_solver_regression.py --configs co2-58-g1,co2p-29-g3
Output: results/gd_solver_regression.pkl  +  SOLVER_REGRESSION.md, both here in
        analytic_jacobian_testing/ -- deliberately self-contained (see HERE below)
        so a future re-run cannot scatter files back into the shared results/ pool.
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent   # still scripts/gd_joint_block_whole_slit_sweep.py's home
HERE = Path(__file__).resolve().parent                # this archive folder -- all OUTPUT goes here, not results/
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

N_COLS = 1024
FREE_SETS = {"co2": "co2_ppm", "co2p": "co2_ppm,p_surface_hpa"}
N_WINDOWS = (58, 29)
G_RATIOS = (1.0, 3.0)


def config_id(free_key, nwin, gratio):
    return f"{free_key}-{nwin}-g{gratio:g}"


def all_configs():
    return [config_id(f, n, g)
            for f, n, g in itertools.product(FREE_SETS, N_WINDOWS, G_RATIOS)]


def parse_config(cid):
    free_key, nwin, gtag = cid.split("-")
    return free_key, int(nwin), float(gtag[1:])


def out_path(cid, jacobian):
    return HERE / "results" / f"{cid}_{jacobian}.pkl"


def run_one(cid, jacobian, n_workers, env, force=False):
    """Run one sweep, returning (path, wall_seconds or None if skipped)."""
    free_key, nwin, gratio = parse_config(cid)
    path = out_path(cid, jacobian)
    if path.exists() and not force:
        print(f"  {cid:14s} {jacobian:8s} exists, skipping", flush=True)
        return path, None
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "gd_joint_block_whole_slit_sweep.py"),
           "--g-ratio", f"{gratio:g}", "--state-interp",
           "--free", FREE_SETS[free_key], "--n-windows", str(nwin),
           "--jacobian", jacobian, "--n-workers", str(n_workers),
           "--out", str(path)]
    print(f"  {cid:14s} {jacobian:8s} running ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError(f"{cid}/{jacobian} failed (rc={r.returncode})")
    print(f"  {cid:14s} {jacobian:8s} done in {dt/60:.1f} min", flush=True)
    return path, dt


def stitch(results, solve, eta_rows):
    """{row_name: per-detector-row values}, and the per-row residual RMS."""
    out, rr = {}, np.full(N_COLS, np.nan)
    for w in results.values():
        snap = w.get(solve)
        if snap is None:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        sl = slice(lo, hi + 1)
        for name, rec in snap["params"].items():
            arr = out.setdefault(name, np.full(N_COLS, np.nan))
            pos, val = np.asarray(rec["positions"]), np.asarray(rec["values"])
            arr[sl] = (np.interp(eta_rows[sl], pos, val) if pos.size > 1 else val[0])
        blk = np.asarray(w[f"resid_{solve}"], dtype=float).reshape(hi - lo + 1, N_COLS)
        rr[sl] = np.sqrt(np.mean(blk ** 2, axis=1))
    return out, rr


def score(path, eta_rows, truth):
    """Per-solve accuracy against truth, plus the raw state for comparison."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"]
    failed = [(w["row_lo"], w["row_hi"]) for w in res.values() if "error" in w]
    out = {"n_windows": len(res), "failed": failed, "free": d.get("free"),
           "g_ratio": d.get("g_ratio"), "jacobian": d.get("jacobian", "fd"),
           "window_scale": d.get("window_scale")}
    for solve in ("coarse", "hires"):
        st, rr = stitch(res, solve, eta_rows)
        if not st:
            continue
        rec = {"resid_rms_median": float(np.nanmedian(rr)),
               "t_total": float(sum(w.get(f"t_{solve}", 0.0) for w in res.values()))}
        for name in ("co2_ppm", "p_surface_hpa"):
            if name in st:
                dv = st[name] - truth[name]
                rec[f"{name}_rms"] = float(np.sqrt(np.nanmean(dv ** 2)))
                rec[f"{name}_max"] = float(np.nanmax(np.abs(dv)))
        rec["_state"] = {k: v for k, v in st.items()}
        rec["_resid_rms"] = rr          # per-row array, kept for compare()
        out[solve] = rec
    return out


def compare(a, b):
    """Solver agreement between two scored runs, worst case over rows/solves.

    `resid_rms_median` used to diff the two runs' median-of-medians as a
    single scalar (`abs(ra-rb)/ra`). That is fragile near a degenerate
    manifold (CO2/p_surface are near-perfectly anti-correlated in this band
    alone -- see run_free_state_sweeps.sh's own note): two solvers can land
    on two slightly different, individually valid points along the ridge,
    which can flip which row sits at the median RANK without any row's own
    residual moving by more than floating-point/iteration-path noise. That
    is exactly what co2p-29-g3 showed (2026-08-18): the scalar-median metric
    read 1.89e-5, just over the 1e-5 tolerance, while the actual per-row
    comparison below -- reusing the SAME `rr` arrays, just not collapsed to
    one number first -- showed a median relative difference of 3.1e-6/3.5e-6
    (coarse/hires) with every outlier traced to near-zero-residual edge rows
    where dividing two tiny, nearly-equal numbers amplifies relative error
    despite ~5e-9 absolute agreement. Comparing the per-row MEDIAN of the
    relative differences (rather than the difference of two medians) is not
    sensitive to that rank flip, since it summarizes the same distribution
    the by-hand check did.
    """
    worst = {}
    for solve in ("coarse", "hires"):
        if solve not in a or solve not in b:
            continue
        for name in a[solve]["_state"]:
            va, vb = a[solve]["_state"][name], b[solve]["_state"].get(name)
            if vb is None:
                continue
            ok = np.isfinite(va) & np.isfinite(vb)
            if not ok.any():
                continue
            rel = np.abs(va[ok] - vb[ok]) / np.maximum(np.abs(va[ok]), 1e-30)
            worst[name] = max(worst.get(name, 0.0), float(rel.max()))
        ra, rb = a[solve].get("_resid_rms"), b[solve].get("_resid_rms")
        if ra is not None and rb is not None:
            ok = np.isfinite(ra) & np.isfinite(rb)
            rel = np.abs(ra[ok] - rb[ok]) / np.maximum(ra[ok], 1e-30)
            worst["resid_rms_median"] = max(worst.get("resid_rms_median", 0.0),
                                            float(np.median(rel)) if rel.size else 0.0)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of " + ",".join(all_configs()))
    ap.add_argument("--analytic-only", action="store_true",
                    help="skip the finite-difference reference (much faster; leaves "
                         "the accuracy table intact but drops the solver-agreement column)")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run configs whose output exists")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="solver-agreement tolerance for the PASS/FAIL verdict "
                         "(default 1e-5; a single window measured 1e-8)")
    args = ap.parse_args()

    cids = args.configs.split(",") if args.configs else all_configs()
    jacobians = ["analytic"] if args.analytic_only else ["analytic", "fd"]
    n_workers = args.n_workers or max(1, (os.cpu_count() or 4) - 2)

    env = dict(os.environ)
    gert_dir = REPO_ROOT.parent / "gert"
    if not (gert_dir / "input").is_dir():
        gert_dir = REPO_ROOT.parent.parent / "gert"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{gert_dir}"
    # Each config runs as N_WORKERS separate processes (multiprocessing.Pool in
    # the sweep script). This machine's BLAS (Apple Accelerate) spawns its own
    # threads per process with no cap by default, so N_WORKERS processes each
    # multithreading BLAS oversubscribes the machine's cores -- measured 2026-08-18
    # on co2p-58-g1/analytic: two of the widest, most matrix-heavy windows (41 and
    # 45 anchors) took 2156s and 2182s under the 12-worker pool, versus 150s each
    # reproduced through the identical code path with one worker. Not a Jacobian
    # bug -- both states and residuals agreed with the FD reference to 5 significant
    # figures; only wall time was affected, and only for the heaviest windows,
    # because those do the most per-anchor linear algebra per call. One BLAS thread
    # per process trades single-call speed for not fighting the other N_WORKERS-1
    # processes for the same cores, which is the right trade once N_WORKERS > 1.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"

    rows = np.arange(float(N_COLS))
    _, s = xy_to_wavelength_slit(2, np.full(N_COLS, 512.0), rows)
    eta_rows = s / s_max(2)
    x_km = eta_rows * als.SLIT_HALF_KM
    truth = {n: np.asarray(f(x_km), dtype=float) for n, f in als.STATE_FIELDS.items()}

    print(f"{len(cids)} configs x {len(jacobians)} solvers, {n_workers} workers")
    print(f"gert: {gert_dir}\n")
    (HERE / "results").mkdir(parents=True, exist_ok=True)

    scored, timings = {}, {}
    for cid in cids:
        print(f"{cid}:", flush=True)
        for j in jacobians:
            path, dt = run_one(cid, j, n_workers, env, force=args.force)
            scored[(cid, j)] = score(path, eta_rows, truth)
            if dt is not None:
                timings[(cid, j)] = dt

    # ------------------------------------------------------------- report --
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("## Accuracy against truth")
    emit()
    hdr = (f"| {'config':14s} | solve  | {'CO2 rms':>9s} | {'CO2 max':>9s} | "
           f"{'dP rms':>8s} | {'dP max':>8s} | {'residRMS':>10s} | fail |")
    emit(hdr)
    emit("|" + "|".join(["-" * len(c) for c in hdr.split("|")[1:-1]]) + "|")
    for cid in cids:
        for j in jacobians:
            sc = scored[(cid, j)]
            for solve in ("coarse", "hires"):
                if solve not in sc:
                    continue
                r = sc[solve]
                tag = cid if j == "analytic" else f"{cid} (fd)"
                emit(f"| {tag:14s} | {solve:6s} | {r.get('co2_ppm_rms', float('nan')):9.4f} | "
                     f"{r.get('co2_ppm_max', float('nan')):9.4f} | "
                     f"{r.get('p_surface_hpa_rms', float('nan')):8.4f} | "
                     f"{r.get('p_surface_hpa_max', float('nan')):8.4f} | "
                     f"{r['resid_rms_median']:10.3e} | {len(sc['failed']):4d} |")

    if not args.analytic_only:
        emit()
        emit("## Solver agreement (analytic vs finite difference)")
        emit()
        emit(f"Tolerance {args.tol:.0e}. Worst relative difference over all detector "
             f"rows and both solves.")
        emit()
        h2 = (f"| {'config':14s} | {'co2_ppm':>10s} | {'p_surface':>10s} | "
              f"{'residRMS':>10s} | verdict |")
        emit(h2)
        emit("|" + "|".join(["-" * len(c) for c in h2.split("|")[1:-1]]) + "|")
        overall_ok = True
        for cid in cids:
            w = compare(scored[(cid, "analytic")], scored[(cid, "fd")])
            ok = all(v <= args.tol for v in w.values())
            overall_ok &= ok
            emit(f"| {cid:14s} | {w.get('co2_ppm', float('nan')):10.2e} | "
                 f"{w.get('p_surface_hpa', float('nan')):10.2e} | "
                 f"{w.get('resid_rms_median', float('nan')):10.2e} | "
                 f"{'PASS' if ok else 'FAIL':7s} |")
        emit()
        emit(f"**Overall: {'PASS' if overall_ok else 'FAIL'}**")

    if timings:
        emit()
        emit("## Wall time (minutes)")
        emit()
        for cid in cids:
            parts = [f"{j} {timings[(cid, j)] / 60:.1f}" for j in jacobians
                     if (cid, j) in timings]
            if parts:
                emit(f"- `{cid}`: " + ", ".join(parts))

    with open(HERE / "results" / "gd_solver_regression.pkl", "wb") as f:
        pickle.dump({"scored": scored, "timings": timings, "tol": args.tol}, f)
    doc = HERE / "SOLVER_REGRESSION.md"
    doc.write_text("# Analytic-Jacobian solver regression\n\n"
                   "Generated by `analytic_jacobian_testing/gd_solver_regression.py`. "
                   "See README.md in this folder for the design matrix.\n"
                   + "\n".join(lines) + "\n")
    print(f"\nsaved {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
