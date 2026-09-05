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
`anchor_density=1`, `state_interp="linear"` -- two axes that were never
varied at all (`analytic_jacobian_testing/README.md` Sec.5's explicit gap).
The next questions (does more anchors help enough to be worth the RT cost,
does `state_interp="nearest"` downscaling change anything) are each a small,
targeted slice of a much bigger space than is worth hard-coding -- hence one
tool with knobs, not a second fixed script per question.

Cost-aware by construction, not just by convention:
  * `--hires-only` skips coarse entirely -- correct to do whenever only
    `anchor_density` is being swept, since coarse renders at `bin_centers`
    directly and never touches `anchor_etas`, so it cannot depend on
    `anchor_density` at all. Re-running it would be pure waste.
  * A config whose axis values are ALL at the closed regression's own
    defaults (`anchor_density=1`, `state_interp="linear"`) is looked up in
    `analytic_jacobian_testing/results/` before running anything fresh --
    that matrix already paid for those points once. Reused configs are
    marked as such in the report rather than silently re-timed as if fresh.

Solver agreement is scored PER SOLVE from each snapshot's own
`jacobian_used` (falls back to the sweep's requested `--jacobian` for
archives that predate that field), not assumed to equal the requested
solver. Found 2026-08-19 (user): `state_interp`'s old boolean form and
`build_forward_state`'s exact-truth bypass at `False` are both retired --
every row, free or frozen, always goes through the same interpolation now,
`"linear"` or `"nearest"`, and analytic Jacobians are unconditionally
available for hires (no more falling back to FD there -- see
`gd_joint_block_whole_slit_sweep.py`'s own note on `use_analytic_hires`).

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
from geocarb_gert.mission_config import RetrievalDefaults  # noqa: E402

N_COLS = 1024
FREE_SETS = {"co2": "co2_ppm", "co2p": "co2_ppm,p_surface_hpa",
            "co2p_albedo": "co2_ppm,p_surface_hpa,albedo"}


def parse_list(s, cast):
    return [cast(x.strip()) for x in s.split(",")]


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


def config_id(free_key, nwin, gratio, adens, si, scene="realistic", barcode_bars=32,
              prior_fields="exact"):
    """`{free}-{nwin}-g{gratio}[-ad{adens}][-{si}][-{scene}[bars]][-pf-{prior_fields}]`.

    Elides every non-default piece (`adens==1`, `si="linear"`,
    `scene="realistic"`, `prior_fields="exact"`) so a config at the closed
    regression's own defaults gets EXACTLY that regression's own id
    (`co2p-58-g1`, no suffix) -- which is what makes the archive-reuse
    lookup in `resolve_existing` a simple path check rather than a
    translation table.

    `si` is the `state_interp` kind string (`"linear"`/`"nearest"`) --
    found 2026-08-19 (user): the boolean `state_interp` this used to encode
    (`True`/`False`, tagging `False` as `-nosi`) is retired along with
    `build_forward_state`'s old exact-truth bypass. The closed regression
    only ever archived `state_interp=True` (today's `"linear"`), so archive
    reuse is unaffected -- `-nosi` configs were never actually archived.

    `prior_fields` names a key in `als.PRIOR_FIELD_SETS` (2026-08-20) --
    what the PRIOR knows, orthogonal to `scene` (what the TRUTH is). Always
    appended last and with its own `-pf-` marker (rather than folded into
    the bare `-{name}` scheme `scene` uses) since prior-field names contain
    underscores themselves (`co2_plus1pct_psurf_minus1pct`) and would be
    ambiguous to split back out of a plain hyphen-joined id otherwise.
    """
    cid = f"{free_key}-{nwin}-g{gratio:g}"
    if adens != 1:
        cid += f"-ad{adens}"
    if si != "linear":
        cid += f"-{si}"
    if scene != "realistic":
        tag = scene.replace("-", "")
        if scene in ("barcode", "realistic-barcode"):
            tag += str(barcode_bars)
        cid += f"-{tag}"
    if prior_fields != "exact":
        cid += f"-pf-{prior_fields}"
    return cid


def parse_config(cid):
    """Inverse of `config_id`, best-effort (used only for display)."""
    prior_fields = "exact"
    if "-pf-" in cid:
        cid, prior_fields = cid.split("-pf-", 1)
    parts = cid.split("-")
    free_key, nwin, gtag = parts[0], parts[1], parts[2]
    adens, si, scene, barcode_bars = 1, "linear", "realistic", 32
    for p in parts[3:]:
        if p.startswith("ad"):
            adens = int(p[2:])
        elif p == "nearest":
            si = "nearest"
        elif p.startswith("uniform"):
            scene = "uniform"
        elif p.startswith("realisticbarcode"):
            scene, barcode_bars = "realistic-barcode", int(p[len("realisticbarcode"):])
        elif p.startswith("barcode"):
            scene, barcode_bars = "barcode", int(p[len("barcode"):])
    return free_key, int(nwin), float(gtag[1:]), adens, si, scene, barcode_bars, prior_fields


def is_bare(cid: str) -> bool:
    """True iff this id matches the closed regression's own naming exactly
    (anchor_density=1, state_interp="linear", scene=realistic,
    prior_fields="exact") -- the only case an archive lookup is meaningful
    (the archive has no barcode/uniform/imperfect-prior runs at all)."""
    parsed = parse_config(cid)
    return ("-ad" not in cid and parsed[4] == "linear" and parsed[5] == "realistic"
           and parsed[7] == "exact")


# ---------------------------------------------------------------- running --

def _payload_flat_sy_inv(path) -> bool:
    """Best-effort read of a cached pkl's own flat_sy_inv flag -- pre-Phase-D
    pkls (or any pkl from before this field existed) have no such key, and
    are treated as flat_sy_inv=False (they predate the real-noise-model
    default entirely, so that's what they in fact are)."""
    try:
        with open(path, "rb") as f:
            return bool(pickle.load(f).get("flat_sy_inv", False))
    except Exception:
        return False


def _payload_overlap(path) -> int:
    """Best-effort read of a cached pkl's own `overlap` -- pkls from before
    window-overlap tiling existed have no such key, and are treated as 0
    (they predate the feature entirely, so that's what they in fact are)."""
    try:
        with open(path, "rb") as f:
            return int(pickle.load(f).get("overlap", 0))
    except Exception:
        return 0


def _payload_prior_form(path) -> str:
    """Best-effort read of a cached pkl's own `prior_form` -- pkls from
    before this field existed default to `"exponential"`, matching this
    project's own current default (`input/retrieval_defaults.yml`)."""
    try:
        with open(path, "rb") as f:
            return str(pickle.load(f).get("prior_form", "exponential"))
    except Exception:
        return "exponential"


def _payload_vary_albedo(path) -> bool:
    """Best-effort read of a cached pkl's own `vary_albedo` -- pkls from
    before this feature existed have no such key, and are treated as False
    (they predate it entirely, so that's what they in fact are)."""
    try:
        with open(path, "rb") as f:
            return bool(pickle.load(f).get("vary_albedo", False))
    except Exception:
        return False


def resolve_existing(cid, jacobian, tag_dir, flat_sy_inv=False, overlap=0, prior_form="exponential",
                     vary_albedo=False):
    """(path, source) if this exact config/solver already has output
    somewhere -- this tag's own folder first, then (for bare ids only) the
    closed regression's archive. `source` is "own" or "archive"/None.

    A file that exists at the expected path but whose own flat_sy_inv/
    overlap/prior_form metadata does NOT match what was requested is
    treated as NOT found (forces a fresh run) rather than silently
    returned -- config_id() has no axis for any of these (they're meant to
    be paired with their own --tag, not mixed into an existing tag's own
    configs), so this is the guard against e.g. a --overlap 2 run silently
    reusing a same-path overlap=0 result, or vice versa. Same pattern as
    the original flat_sy_inv check."""
    own = tag_dir / f"{cid}_{jacobian}.pkl"
    if (own.exists() and _payload_flat_sy_inv(own) == flat_sy_inv
           and _payload_overlap(own) == overlap and _payload_prior_form(own) == prior_form
           and _payload_vary_albedo(own) == vary_albedo):
        return own, "own"
    if (is_bare(cid) and not flat_sy_inv and overlap == 0 and prior_form == "exponential"
           and not vary_albedo):
        arch = ARCHIVE / f"{cid}_{jacobian}.pkl"
        if arch.exists():
            return arch, "archive"
    return None, None


def run_one(cid, jacobian, n_workers, env, tag_dir, force=False):
    free_key, nwin, gratio, adens, si, scene, barcode_bars, prior_fields = parse_config(cid)
    flat_sy_inv = env.get("_FLAT_SY_INV") == "1"
    overlap = int(env.get("_OVERLAP", "0"))
    prior_form = env.get("_PRIOR_FORM", "exponential")
    vary_albedo = env.get("_VARY_ALBEDO") == "1"
    path, source = (None, None) if force else resolve_existing(
        cid, jacobian, tag_dir, flat_sy_inv, overlap, prior_form, vary_albedo)
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
    if si != "linear":
        cmd += ["--state-interp", si]
    sc = SCENES[scene]
    if sc["uniform"]:
        cmd.append("--uniform")
    if sc["barcode"]:
        cmd += ["--barcode", "--barcode-bars", str(barcode_bars)]
    if sc["realistic_barcode"]:
        cmd += ["--realistic-barcode", "--barcode-bars", str(barcode_bars)]
    if prior_fields != "exact":
        cmd += ["--prior-fields", prior_fields]
    if env.get("_HIRES_ONLY") == "1":
        cmd.append("--hires-only")
    if env.get("_CONFIG_PATH"):
        cmd += ["--config", env["_CONFIG_PATH"]]
    if env.get("_NO_TRUTH_CACHE") == "1":
        cmd.append("--no-truth-cache")
    if env.get("_FLAT_SY_INV") == "1":
        cmd.append("--flat-sy-inv")
    if overlap:
        cmd += ["--overlap", str(overlap)]
    if prior_form != "exponential":
        cmd += ["--prior-form", prior_form]
    if vary_albedo:
        cmd.append("--vary-albedo")
    print(f"  {cid:26s} {jacobian:8s} running ...", flush=True)
    t0 = time.time()
    _internal_keys = ("_HIRES_ONLY", "_CONFIG_PATH", "_NO_TRUTH_CACHE", "_FLAT_SY_INV",
                      "_OVERLAP", "_PRIOR_FORM", "_VARY_ALBEDO")
    r = subprocess.run(cmd, cwd=REPO_ROOT, env={k: v for k, v in env.items() if k not in _internal_keys},
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
        # dof_frac: mean over windows of trace(AVK)/n_free -- the degrees-
        # of-freedom-for-signal fraction (0 = fully prior-dominated, 1 =
        # fully data-dominated), averaged across this config's own windows.
        # Absent (nan) for any pkl from before gauss_newton_state's
        # return_avk existed -- docs/PROJECT_STATUS.md Sec.7.10.
        dof_fracs = [w[solve]["dof"] / w[solve]["n_free"] for w in res.values()
                    if solve in w and "dof" in w[solve] and w[solve].get("n_free", 0) > 0]
        rec = {"resid_rms_median": float(np.nanmedian(rr)), "_resid_rms": rr,
              "jacobian_used": (jac_used.pop() if len(jac_used) == 1
                                else "mixed" if jac_used else out["requested_jacobian"]),
              "t_total": float(sum(w.get(f"t_{solve}", 0.0) for w in res.values())),
              "dof_frac": float(np.mean(dof_fracs)) if dof_fracs else float("nan")}
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
    # Parse just --config before building the real parser below, so its
    # own default values (anchor_density/state_interp/jacobian/prior_fields)
    # can reflect it -- same two-pass pattern gd_joint_block_whole_slit_
    # sweep.py's own _resolve_cli_defaults() uses, for the same reason.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=None)
    _pre_args, _ = _pre.parse_known_args()
    _cfg = RetrievalDefaults.from_yaml(_pre_args.config)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None,
                    help="path to retrieval_defaults.yml, forwarded as --config to every "
                         "shelled-out gd_joint_block_whole_slit_sweep.py invocation (default: "
                         "the checked-in input/retrieval_defaults.yml). Only --anchor-density/"
                         "--state-interp/--jacobian/--prior-fields's own defaults below read "
                         "from it directly -- --free and --g-ratio keep this script's own "
                         "historical defaults ('co2p', '1'), which deliberately differ from "
                         "the single-run sweep script's defaults ('co2_ppm', '3') and would "
                         "silently change if sourced from the same config value.")
    ap.add_argument("--tag", default="default",
                    help="namespaces this run's output under results/config_matrix/<tag>/ "
                         "so different sweep sessions never collide (default: 'default')")
    ap.add_argument("--free", default="co2p", help="comma list from {co2,co2p} (default: co2p)")
    ap.add_argument("--n-windows", default="58", help="comma list of ints (default: 58)")
    ap.add_argument("--g-ratio", default="1", help="comma list of floats (default: 1)")
    ap.add_argument("--anchor-density", default=str(_cfg.anchor_density),
                    help="comma list of ints (default: input/retrieval_defaults.yml's "
                         "solve.anchor_density)")
    ap.add_argument("--state-interp", default=_cfg.state_interp,
                    help="comma list of state_interp kinds, linear/nearest "
                         "(default: input/retrieval_defaults.yml's solve.state_interp)")
    ap.add_argument("--jacobian", default=_cfg.jacobian,
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
    ap.add_argument("--prior-fields", default=_cfg.prior_fields,
                    help=f"comma list of named prior field-sets from als.PRIOR_FIELD_SETS "
                         f"(default: exact, prior=truth) -- {sorted(als.PRIOR_FIELD_SETS)}. "
                         f"Orthogonal to --scene: --scene controls what the TRUTH is, "
                         f"--prior-fields controls what the PRIOR knows about it.")
    ap.add_argument("--hires-only", action="store_true",
                    help="skip coarse -- correct whenever only anchor_density is swept, "
                         "since coarse cannot depend on it at all")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--no-truth-cache", action="store_true",
                    help="forwarded to every shelled-out sweep invocation -- always re-render "
                         "the truth scene instead of reusing results/truth_cache/ (see "
                         "geocarb_gert.truth_cache). Caching is on by default and is exactly "
                         "what this tool benefits from most: every config here that shares a "
                         "scene renders it once. Pass this after a rendering-path code change "
                         "that hasn't bumped TRUTH_CACHE_VERSION yet.")
    ap.add_argument("--flat-sy-inv", action="store_true",
                    help="forwarded to every shelled-out sweep invocation -- reproduces the "
                         "pre-Phase-D flat-scalar Sy_inv weighting instead of the real noise "
                         "model (the default). NOT a production option, solely for deliberately "
                         "reproducing an old comparison point (docs/PROJECT_STATUS.md Sec.6). "
                         "ALWAYS pair this with a --tag no corrected-weighting run has ever "
                         "used -- resolve_existing() checks each cached file's own flat_sy_inv "
                         "metadata and refuses to reuse a mismatched one (forces a fresh run "
                         "instead of silently returning the wrong-weighting result), but a "
                         "dedicated tag is still the intended way to keep the two comparable "
                         "runs' outputs cleanly separated on disk.")
    ap.add_argument("--overlap", type=int, default=0,
                    help="forwarded to every shelled-out sweep invocation -- rows of symmetric "
                         "overlap between adjacent windows (see build_window_tiles/along_slit_"
                         "query.query_state, docs/PROJECT_STATUS.md Sec.6). Default 0 (today's "
                         "exact tiling). Part of resolve_existing()'s own metadata check, same "
                         "reasoning as --flat-sy-inv: use a --tag no other overlap value has "
                         "used, or reuse will correctly force a fresh run rather than silently "
                         "returning a mismatched-overlap result.")
    ap.add_argument("--prior-form", default="exponential", choices=("exponential", "tikhonov"),
                    help="forwarded to every shelled-out sweep invocation -- ParamSpec's own "
                         "prior_form (default: exponential, matching input/retrieval_defaults."
                         "yml). Also part of resolve_existing()'s metadata check.")
    ap.add_argument("--vary-albedo", action="store_true",
                    help="forwarded to every shelled-out sweep invocation -- render the truth "
                         "scene with real along-slit surface heterogeneity instead of a "
                         "constant scalar albedo (geocarb_gert.along_slit_scene.albedo_at). "
                         "Required by the sweep script itself whenever 'albedo' is in --free. "
                         "Also part of resolve_existing()'s metadata check.")
    ap.add_argument("--force", action="store_true",
                    help="re-run configs even if found in this tag's folder OR the archive")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    frees = parse_list(args.free, str)
    nwins = parse_list(args.n_windows, int)
    gratios = parse_list(args.g_ratio, float)
    adenss = parse_list(args.anchor_density, int)
    sis = parse_list(args.state_interp, str)
    for si in sis:
        if si not in ("linear", "nearest"):
            ap.error(f"--state-interp: {si!r} not in ('linear','nearest')")
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
    prior_fields_list = parse_list(args.prior_fields, str)
    for pf in prior_fields_list:
        if pf not in als.PRIOR_FIELD_SETS:
            ap.error(f"--prior-fields: {pf!r} not in {sorted(als.PRIOR_FIELD_SETS)}")

    cids = [config_id(f, n, g, a, s, scene, args.barcode_bars, pf)
           for f, n, g, a, s, scene, pf
           in itertools.product(frees, nwins, gratios, adenss, sis, scenes, prior_fields_list)]
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
    if args.config:
        env["_CONFIG_PATH"] = args.config
    if args.no_truth_cache:
        env["_NO_TRUTH_CACHE"] = "1"
    if args.flat_sy_inv:
        env["_FLAT_SY_INV"] = "1"
    if args.overlap:
        env["_OVERLAP"] = str(args.overlap)
    if args.prior_form != "exponential":
        env["_PRIOR_FORM"] = args.prior_form
    if args.vary_albedo:
        env["_VARY_ALBEDO"] = "1"

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
         f"anchor_density={adenss} state_interp={sis} scene={scenes} "
         f"prior_fields={prior_fields_list}")
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
        f"scene={scenes}, prior_fields={prior_fields_list}, hires_only={args.hires_only}\n")
    emit("## Accuracy against truth\n")
    hdr = (f"| {'config':26s} | jac(req) | solve  | jac(used) | {'CO2 rms':>9s} | "
          f"{'CO2 max':>9s} | {'dP rms':>8s} | {'dP max':>8s} | {'residRMS':>10s} | "
          f"{'DOF frac':>8s} | src |")
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
                    f"{r['resid_rms_median']:10.3e} | "
                    f"{r.get('dof_frac', float('nan')):8.3f} | {sources[(cid,j)]:7s} |")

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
