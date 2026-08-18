#!/usr/bin/env python3
"""Generic N-way config-matrix runner for `gd_joint_block_whole_slit_sweep.py`.

Generalizes `analytic_jacobian_testing/gd_solver_regression.py` (now closed
-- see that folder's README) from a fixed 2x2x2 matrix into one where every
axis is a CLI-toggleable list: free rows, window count, `g_ratio`,
`anchor_density`, `state_interp`, and solver (`jacobian`). Each invocation
is namespaced by `--tag`, so different sweep sessions (an anchor-density
check, a state-interp check, ...) never collide with each other or with the
closed regression's own files.

Why generic rather than another fixed matrix: the closed regression proved
the analytic solver agrees with FD across free-rows x windows x G, always at
`anchor_density=1`, `state_interp=True` -- two axes that were never varied
at all (`analytic_jacobian_testing/README.md` Sec.5's explicit gap). The
next questions (does more anchors help enough to be worth the RT cost, does
`state_interp=False` change anything) are each a small, targeted slice of a
much bigger space than is worth hard-coding -- hence one tool with knobs,
not a second fixed script per question.

Cost-aware by construction, not just by convention:
  * `--hires-only` skips coarse entirely -- correct to do whenever only
    `anchor_density` is being swept, since coarse renders at `bin_centers`
    directly and never touches `anchor_etas`, so it cannot depend on
    `anchor_density` at all. Re-running it would be pure waste.
  * A config whose axis values are ALL at the closed regression's own
    defaults (`anchor_density=1`, `state_interp=True`) is looked up in
    `analytic_jacobian_testing/results/` before running anything fresh --
    that matrix already paid for those points once. Reused configs are
    marked as such in the report rather than silently re-timed as if fresh.

Solver agreement is scored PER SOLVE from each snapshot's own
`jacobian_used` (falls back to the sweep's requested `--jacobian` for
archives that predate that field), not assumed to equal the requested
solver -- required for correctness now that this tool can request
`state_interp=False`, where `gauss_newton_state`'s per-solve gating makes
hires silently fall back to FD even when `--jacobian analytic` was asked
for (coarse does not fall back -- see `gd_joint_block_whole_slit_sweep.py`'s
own note on `use_analytic_hires`).

Run (defaults to ONE cheap config x 2 solvers, both already archived --
a safe, fast no-op smoke test):
    PYTHONPATH=. python3 scripts/gd_joint_block_matrix.py --tag smoketest

The "targeted 4" anchor-density check this was built for (run 2026-08-19,
result: anchor_density=4 cut CO2/p_surface rms 70-80% at both window counts
-- see results/config_matrix/anchor_density_v1/REPORT.md):
    PYTHONPATH=. python3 scripts/gd_joint_block_matrix.py \\
        --tag anchor_density_v1 --free co2p --n-windows 58,29 --g-ratio 1 \\
        --anchor-density 1,4 --hires-only --jacobian analytic,fd

`--jacobian` now DEFAULTS to analytic-only (2026-08-19) -- FD agreement is
already established across every axis this tool sweeps (see `--jacobian`'s
own help). Pass `--jacobian analytic,fd` explicitly, as the example above
still does, only when actually re-validating -- e.g. the barcode scene
(new state-priors code path) is exactly that kind of case, worth one FD
cross-check before treating it as routine too.

Output: results/config_matrix/<tag>/results/<config>_<jacobian>.pkl
        results/config_matrix/<tag>/results/summary.pkl
        results/config_matrix/<tag>/REPORT.md
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARCHIVE = REPO_ROOT / "analytic_jacobian_testing" / "results"

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

N_COLS = 1024
FREE_SETS = {"co2": "co2_ppm", "co2p": "co2_ppm,p_surface_hpa"}


def parse_list(s, cast):
    return [cast(x.strip()) for x in s.split(",")]


def parse_bool_list(s):
    return [x.strip().lower() in ("1", "true", "t", "yes") for x in s.split(",")]


# ------------------------------------------------------------ config identity --

#: scene name -> gd_joint_block_whole_slit_sweep.py flags it needs, and
#: whether the true composition is spatially flat (drives which `truth`
#: dict score() must compare against -- see `truth_for`).
SCENES = {
    "realistic": {"uniform": False, "barcode": False, "realistic_barcode": False, "flat": False},
    "uniform": {"uniform": True, "barcode": False, "realistic_barcode": False, "flat": True},
    "barcode": {"uniform": False, "barcode": True, "realistic_barcode": False, "flat": True},
    "realistic-barcode": {"uniform": False, "barcode": False, "realistic_barcode": True, "flat": False},
}


def config_id(free_key, nwin, gratio, adens, si, scene="realistic", barcode_bars=32):
    """`{free}-{nwin}-g{gratio}[-ad{adens}][-nosi][-{scene}[bars]]`.

    Elides every non-default piece (`adens==1`, `si=True`,
    `scene="realistic"`) so a config at the closed regression's own defaults
    gets EXACTLY that regression's own id (`co2p-58-g1`, no suffix) --
    which is what makes the archive-reuse lookup in `resolve_existing` a
    simple path check rather than a translation table.
    """
    cid = f"{free_key}-{nwin}-g{gratio:g}"
    if adens != 1:
        cid += f"-ad{adens}"
    if not si:
        cid += "-nosi"
    if scene != "realistic":
        tag = scene.replace("-", "")
        if scene in ("barcode", "realistic-barcode"):
            tag += str(barcode_bars)
        cid += f"-{tag}"
    return cid


def parse_config(cid):
    """Inverse of `config_id`, best-effort (used only for display)."""
    parts = cid.split("-")
    free_key, nwin, gtag = parts[0], parts[1], parts[2]
    adens, si, scene, barcode_bars = 1, True, "realistic", 32
    for p in parts[3:]:
        if p.startswith("ad"):
            adens = int(p[2:])
        elif p == "nosi":
            si = False
        elif p.startswith("uniform"):
            scene = "uniform"
        elif p.startswith("realisticbarcode"):
            scene, barcode_bars = "realistic-barcode", int(p[len("realisticbarcode"):])
        elif p.startswith("barcode"):
            scene, barcode_bars = "barcode", int(p[len("barcode"):])
    return free_key, int(nwin), float(gtag[1:]), adens, si, scene, barcode_bars


def is_bare(cid: str) -> bool:
    """True iff this id matches the closed regression's own naming exactly
    (anchor_density=1, state_interp=True, scene=realistic) -- the only case
    an archive lookup is meaningful (the archive has no barcode/uniform
    runs at all)."""
    return "-ad" not in cid and "-nosi" not in cid and parse_config(cid)[5] == "realistic"


# ---------------------------------------------------------------- running --

def resolve_existing(cid, jacobian, tag_dir):
    """(path, source) if this exact config/solver already has output
    somewhere -- this tag's own folder first, then (for bare ids only) the
    closed regression's archive. `source` is "own" or "archive"/None."""
    own = tag_dir / f"{cid}_{jacobian}.pkl"
    if own.exists():
        return own, "own"
    if is_bare(cid):
        arch = ARCHIVE / f"{cid}_{jacobian}.pkl"
        if arch.exists():
            return arch, "archive"
    return None, None


def run_one(cid, jacobian, n_workers, env, tag_dir, force=False):
    free_key, nwin, gratio, adens, si, scene, barcode_bars = parse_config(cid)
    path, source = (None, None) if force else resolve_existing(cid, jacobian, tag_dir)
    if path is not None:
        tag = "archive (already validated)" if source == "archive" else "exists"
        print(f"  {cid:26s} {jacobian:8s} {tag}, skipping", flush=True)
        return path, None, source

    out_path = tag_dir / f"{cid}_{jacobian}.pkl"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "gd_joint_block_whole_slit_sweep.py"),
          "--g-ratio", f"{gratio:g}", "--free", FREE_SETS[free_key],
          "--n-windows", str(nwin), "--anchor-density", str(adens),
          "--jacobian", jacobian, "--n-workers", str(n_workers),
          "--out", str(out_path)]
    if si:
        cmd.append("--state-interp")
    sc = SCENES[scene]
    if sc["uniform"]:
        cmd.append("--uniform")
    if sc["barcode"]:
        cmd += ["--barcode", "--barcode-bars", str(barcode_bars)]
    if sc["realistic_barcode"]:
        cmd += ["--realistic-barcode", "--barcode-bars", str(barcode_bars)]
    if env.get("_HIRES_ONLY") == "1":
        cmd.append("--hires-only")
    print(f"  {cid:26s} {jacobian:8s} running ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, env={k: v for k, v in env.items() if k != "_HIRES_ONLY"},
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError(f"{cid}/{jacobian} failed (rc={r.returncode})")
    print(f"  {cid:26s} {jacobian:8s} done in {dt/60:.1f} min", flush=True)
    return out_path, dt, "fresh"


# --------------------------------------------------------------- scoring --

def stitch(results, solve, eta_rows):
    out, rr, jac_used = {}, np.full(N_COLS, np.nan), set()
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
        jac_used.add(snap.get("jacobian_used"))
    return out, rr, jac_used


def score(path, eta_rows, truth):
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"]
    failed = [(w["row_lo"], w["row_hi"]) for w in res.values() if "error" in w]
    out = {"n_windows": len(res), "failed": failed, "requested_jacobian": d.get("jacobian", "fd")}
    for solve in ("coarse", "hires"):
        st, rr, jac_used = stitch(res, solve, eta_rows)
        if not st:
            continue
        # jacobian_used is per-solve truth; falls back to the requested flag
        # only for pre-gating archives that never recorded it at all.
        jac_used.discard(None)
        rec = {"resid_rms_median": float(np.nanmedian(rr)), "_resid_rms": rr,
              "jacobian_used": (jac_used.pop() if len(jac_used) == 1
                                else "mixed" if jac_used else out["requested_jacobian"]),
              "t_total": float(sum(w.get(f"t_{solve}", 0.0) for w in res.values()))}
        for name in ("co2_ppm", "p_surface_hpa"):
            if name in st:
                dv = st[name] - truth[name]
                rec[f"{name}_rms"] = float(np.sqrt(np.nanmean(dv ** 2)))
                rec[f"{name}_max"] = float(np.nanmax(np.abs(dv)))
        rec["_state"] = dict(st)
        out[solve] = rec
    return out


def compare(a, b):
    """Solver agreement between two scored runs -- median of per-row
    relative differences (see the closed regression's own compare() for why
    not a difference of two medians)."""
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


# ------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default",
                    help="namespaces this run's output under results/config_matrix/<tag>/ "
                         "so different sweep sessions never collide (default: 'default')")
    ap.add_argument("--free", default="co2p", help="comma list from {co2,co2p} (default: co2p)")
    ap.add_argument("--n-windows", default="58", help="comma list of ints (default: 58)")
    ap.add_argument("--g-ratio", default="1", help="comma list of floats (default: 1)")
    ap.add_argument("--anchor-density", default="1", help="comma list of ints (default: 1)")
    ap.add_argument("--state-interp", default="true",
                    help="comma list of true/false (default: true)")
    ap.add_argument("--jacobian", default="analytic",
                    help="comma list from {analytic,fd} (default: analytic only -- FD's "
                         "own agreement with it is already established: PASS on all 16 "
                         "closed-regression configs plus both anchor_density=4 configs, "
                         "state agreement 1e-7..1e-8 everywhere. Pass 'analytic,fd' or "
                         "'fd' explicitly to re-run the comparison, e.g. after a real "
                         "change to geocarb_gert/jacobians.py or joint_state.py, or to "
                         "exercise a state target (dispersion, albedo) neither of those "
                         "matrices covered.")
    ap.add_argument("--scene", default="realistic",
                    help="comma list from {realistic,uniform,barcode,realistic-barcode} "
                         "(default: realistic). 'barcode': fixed atmosphere, reflectance-"
                         "only bars -- isolated test of a sharp reflectance boundary. "
                         "'realistic-barcode': real along-slit composition PLUS the same "
                         "reflectance bars on top. 'uniform'/'barcode' both score against "
                         "a FLAT truth (their composition genuinely is uniform); "
                         "'realistic'/'realistic-barcode' score against the normal "
                         "along-slit truth.")
    ap.add_argument("--barcode-bars", type=int, default=32,
                    help="bars for barcode/realistic-barcode scenes (default 32)")
    ap.add_argument("--hires-only", action="store_true",
                    help="skip coarse -- correct whenever only anchor_density is swept, "
                         "since coarse cannot depend on it at all")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run configs even if found in this tag's folder OR the archive")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    frees = parse_list(args.free, str)
    nwins = parse_list(args.n_windows, int)
    gratios = parse_list(args.g_ratio, float)
    adenss = parse_list(args.anchor_density, int)
    sis = parse_bool_list(args.state_interp)
    jacobians = parse_list(args.jacobian, str)
    for f in frees:
        if f not in FREE_SETS:
            ap.error(f"--free: {f!r} not in {list(FREE_SETS)}")
    for j in jacobians:
        if j not in ("analytic", "fd"):
            ap.error(f"--jacobian: {j!r} not in ('analytic','fd')")
    scenes = parse_list(args.scene, str)
    for sc in scenes:
        if sc not in SCENES:
            ap.error(f"--scene: {sc!r} not in {list(SCENES)}")

    cids = [config_id(f, n, g, a, s, scene, args.barcode_bars)
           for f, n, g, a, s, scene
           in itertools.product(frees, nwins, gratios, adenss, sis, scenes)]
    n_workers = args.n_workers or max(1, (os.cpu_count() or 4) - 2)

    tag_dir = REPO_ROOT / "results" / "config_matrix" / args.tag
    (tag_dir / "results").mkdir(parents=True, exist_ok=True)
    tag_results = tag_dir / "results"

    env = dict(os.environ)
    gert_dir = REPO_ROOT.parent / "gert"
    if not (gert_dir / "input").is_dir():
        gert_dir = REPO_ROOT.parent.parent / "gert"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{gert_dir}"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"  # see closed regression's own note: avoids BLAS oversubscription
    if args.hires_only:
        env["_HIRES_ONLY"] = "1"

    rows = np.arange(float(N_COLS))
    _, s = xy_to_wavelength_slit(2, np.full(N_COLS, 512.0), rows)
    eta_rows = s / s_max(2)
    x_km = eta_rows * als.SLIT_HALF_KM
    # Two truth dicts, picked per-config by SCENES[scene]["flat"]: uniform/
    # barcode's true composition really is spatially constant (matching
    # state_spec_from_scene(uniform=True)'s own priors), so scoring those
    # against the along-slit-varying truth would just measure the ambient
    # gradient, not retrieval error. realistic/realistic-barcode keep the
    # normal along-slit truth -- realistic-barcode's composition genuinely
    # varies, only reflectance gets a pattern on top.
    truth_varying = {n: np.asarray(f(x_km), dtype=float) for n, f in als.STATE_FIELDS.items()}
    truth_flat = {n: np.full(N_COLS, float(f(np.zeros(1))[0])) for n, f in als.STATE_FIELDS.items()}

    print(f"tag: {args.tag}   {len(cids)} configs x {len(jacobians)} solvers, "
         f"{n_workers} workers, hires_only={args.hires_only}")
    print(f"axes: free={frees} n_windows={nwins} g_ratio={gratios} "
         f"anchor_density={adenss} state_interp={sis} scene={scenes}")
    print(f"gert: {gert_dir}\n")

    scored, timings, sources = {}, {}, {}
    for cid in cids:
        print(f"{cid}:", flush=True)
        cfg_scene = parse_config(cid)[5]
        truth = truth_flat if SCENES[cfg_scene]["flat"] else truth_varying
        for j in jacobians:
            path, dt, source = run_one(cid, j, n_workers, env, tag_results, force=args.force)
            scored[(cid, j)] = score(path, eta_rows, truth)
            sources[(cid, j)] = source
            if dt is not None:
                timings[(cid, j)] = dt

    # ------------------------------------------------------------- report --
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Config matrix -- tag `{args.tag}`\n")
    emit(f"Axes: free={frees}, n_windows={nwins}, g_ratio={gratios}, "
        f"anchor_density={adenss}, state_interp={sis}, jacobian={jacobians}, "
        f"scene={scenes}, hires_only={args.hires_only}\n")
    emit("## Accuracy against truth\n")
    hdr = (f"| {'config':26s} | jac(req) | solve  | jac(used) | {'CO2 rms':>9s} | "
          f"{'CO2 max':>9s} | {'dP rms':>8s} | {'dP max':>8s} | {'residRMS':>10s} | src |")
    emit(hdr)
    emit("|" + "|".join(["-" * len(c) for c in hdr.split("|")[1:-1]]) + "|")
    for cid in cids:
        for j in jacobians:
            sc = scored[(cid, j)]
            for solve in ("coarse", "hires"):
                if solve not in sc:
                    continue
                r = sc[solve]
                emit(f"| {cid:26s} | {j:8s} | {solve:6s} | {r['jacobian_used']:9s} | "
                    f"{r.get('co2_ppm_rms', float('nan')):9.4f} | "
                    f"{r.get('co2_ppm_max', float('nan')):9.4f} | "
                    f"{r.get('p_surface_hpa_rms', float('nan')):8.4f} | "
                    f"{r.get('p_surface_hpa_max', float('nan')):8.4f} | "
                    f"{r['resid_rms_median']:10.3e} | {sources[(cid,j)]:7s} |")

    if "analytic" in jacobians and "fd" in jacobians:
        emit("\n## Solver agreement (analytic vs finite difference)\n")
        emit(f"Tolerance {args.tol:.0e}.\n")
        h2 = (f"| {'config':26s} | {'co2_ppm':>10s} | {'p_surface':>10s} | "
             f"{'residRMS':>10s} | verdict |")
        emit(h2)
        emit("|" + "|".join(["-" * len(c) for c in h2.split("|")[1:-1]]) + "|")
        for cid in cids:
            w = compare(scored[(cid, "analytic")], scored[(cid, "fd")])
            ok = all(v <= args.tol for v in w.values())
            emit(f"| {cid:26s} | {w.get('co2_ppm', float('nan')):10.2e} | "
                f"{w.get('p_surface_hpa', float('nan')):10.2e} | "
                f"{w.get('resid_rms_median', float('nan')):10.2e} | "
                f"{'PASS' if ok else 'FAIL':7s} |")

    if timings:
        emit("\n## Wall time (minutes) -- fresh runs only, reused configs show nothing\n")
        for cid in cids:
            parts = [f"{j} {timings[(cid, j)] / 60:.1f}" for j in jacobians
                    if (cid, j) in timings]
            if parts:
                emit(f"- `{cid}`: " + ", ".join(parts))

    with open(tag_results / "summary.pkl", "wb") as f:
        pickle.dump({"scored": scored, "timings": timings, "sources": sources,
                    "tol": args.tol, "args": vars(args)}, f)
    doc = tag_dir / "REPORT.md"
    doc.write_text("\n".join(lines) + "\n")
    print(f"\nsaved {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
