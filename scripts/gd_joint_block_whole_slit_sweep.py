#!/usr/bin/env python3
"""Whole-slit joint-block sweep, extending the row 890-935 hot-spot test
(docs/JOINT_BIN_RETRIEVAL_ATBD.html) across all of FPA2.

The slit is tiled into non-overlapping windows sized off local keystone
amplitude: window radius = max(MIN_WINDOW, round(2.2 * rows_crossed(fpa,
center))), the same formula gd_toy_trace_retrieve.py already validated for
sizing a single target's own neighborhood. Windows are non-overlapping by
construction (each window starts exactly where the previous one ended),
not a fixed stride -- a fixed stride would either overlap in high-keystone
regions or leave gaps in low-keystone ones, since window width varies by
more than 5x across the slit. Non-overlapping tiling also means every row
belongs to exactly one window's own independent solve, so there is never
an ambiguous choice between two overlapping windows' estimates at a given
row.

Within each window, G = max(2, round(width/3)) bins are placed by pixel
density (geocarb_gert not involved -- reuses scripts/gd_joint_block_
diagnostics.py's own pixel_density_bin_centers unchanged), and BOTH the
coarse (nearest-bin) and hi-res (state interpolated to native row
resolution) forward models are solved independently, exactly as in the
row 890-935 example -- same regularization, same gamma, same local-truth
nuisance-gas idealization. Because the forward model's own Jacobian and
the regularization's own Laplacian are both local/banded (a bin's own
influence never reaches past the PSF's few-row footprint or past its
immediate neighbors), solving each window independently is not an
approximation to one giant whole-slit joint solve -- it gives essentially
the same answer at a fraction of the cost, as long as the padding used
here (PAD rows on each side, feeding predict_neighborhood) correctly
covers the PSF's own reach, which it does by the same margin already
validated for the single hot-spot window.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_whole_slit_sweep.py \\
        [--n-workers N] [--gamma 3.0] [--g-ratio 3]
Output: results/gd_joint_block_whole_slit_fpa2.pkl
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
from gd_joint_block_retrieve import (  # noqa: E402
    FPA, GERT_ROOT, _eta_of, band_basics, build_forward,
    gauss_newton_regularized, make_spectrum_fn,
)
from gd_joint_block_hires_test import build_forward_hires  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.joint_state import (build_forward_state, gauss_newton_state,  # noqa: E402
                                      state_spec_from_scene)
from gert.forward_model import ForwardModel  # noqa: E402
from gert.rt_solver import SingleScatterSolver  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed  # noqa: E402
from geocarb_gert.gd_render import available_cpus  # noqa: E402

MIN_WINDOW = 4
PAD = 4
G_RATIO = 3
ROW_MAX_IDX = 1023

_SWEEP = {}


def build_window_tiles(fpa: int, row_min: int = 0, row_max: int = ROW_MAX_IDX,
                       min_window: int = MIN_WINDOW) -> list:
    """Non-overlapping tiling of [row_min, row_max]. Each window's own
    radius is a small fixed-point solve (window width depends on
    rows_crossed at the window's own center, which depends on width) --
    converges in a couple of iterations since rows_crossed varies slowly."""
    tiles = []
    row_start = row_min
    while row_start <= row_max:
        r = min_window
        for _ in range(6):
            center = min(row_start + r, row_max)
            k_c = float(rows_crossed(fpa, np.array([float(center)]))[0])
            r_new = max(min_window, int(round(2.2 * k_c)))
            if r_new == r:
                break
            r = r_new
        row_end = min(row_start + 2 * r, row_max)
        tiles.append((row_start, row_end))
        row_start = row_end + 1
    return tiles


def _make_state_spectrum(absco, wide_inst, geo, solar, albedo):
    """spectrum(params_dict) -> hi-res radiance, built straight from
    `als.atmosphere_from_params`. Deliberately NOT gert's
    StateVector.gas_scaling, which only knows how to scale gases and would
    reintroduce the CO2-is-special asymmetry. Albedo stays fixed: it is not
    part of AtmosphericProfile (see als.STATE_FIELDS)."""
    def spectrum(params: dict):
        atm = als.atmosphere_from_params(**params)
        fm = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(),
                          solar_spectrum=solar)
        res = fm.run(albedo=np.array([albedo]), albedo_slope=np.zeros(1))
        return np.asarray(res.I_hires[0], dtype=float)
    return spectrum


def _solve_window(row_lo: int, row_hi: int):
    band = _SWEEP["band"]
    absco, wide_inst, geo, solar, albedo = (_SWEEP["absco"], _SWEEP["wide_inst"],
                                            _SWEEP["geo"], _SWEEP["solar"], _SWEEP["albedo"])
    wn_hires, ils, gamma, sigma_abs = _SWEEP["wn_hires"], _SWEEP["ils"], _SWEEP["gamma"], _SWEEP["sigma_abs"]
    uniform, atm_center = _SWEEP["uniform"], _SWEEP["atm_center"]
    g_ratio = _SWEEP.get("g_ratio", G_RATIO)
    hires_only = _SWEEP.get("hires_only", False)
    anchor_density = int(_SWEEP.get("anchor_density", 1))
    state_interp = bool(_SWEEP.get("state_interp", False))

    rows_win = np.arange(row_lo, row_hi + 1)
    width = len(rows_win)
    G = max(2, int(round(width / g_ratio)))
    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    if uniform:
        # true scene really is constant everywhere -- every bin's own
        # "local truth" prior is the SAME shared atm_center, not a
        # position-dependent one, so there is no legitimate resolution-
        # floor/quantization effect possible: any nonzero bias below is
        # unambiguously a bug, not a real physical ceiling.
        prior_atms = [atm_center] * G
        prior_co2_ppm_bins = np.full(G, float(als.xco2_ppm(0.0)))
    else:
        x_km_bins = bin_centers * als.SLIT_HALF_KM
        prior_atms = [als.atmosphere_at(float(xk)) for xk in x_km_bins]
        prior_co2_ppm_bins = np.array([float(als.xco2_ppm(xk)) for xk in x_km_bins])

    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

    out = dict(row_lo=row_lo, row_hi=row_hi, width=width, G=G, bin_centers=bin_centers,
              prior_co2_ppm_bins=prior_co2_ppm_bins)

    free = tuple(_SWEEP.get("free", ("co2_ppm",)))
    corr_length = _SWEEP.get("corr_length")          # None -> per-row physical defaults
    prior_form = _SWEEP.get("prior_form", "exponential")
    spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)

    # COARSE: scene evaluated at the state's own bin centres, so
    # interp_to() is the identity -- --state-interp has no content here.
    if not hires_only:
        t0 = time.time()
        spec_c = state_spec_from_scene(bin_centers, free=free, corr_length=corr_length,
                                       prior_form=prior_form)
        # coarse: scene positions ARE the state positions, so interpolation is
        # the identity either way -- state_interp is genuinely a no-op here.
        fwd_c = build_forward_state(FPA, rows_win, bin_centers, spec_c, spectrum,
                                    wn_hires, ils, pad=PAD, state_interp=True)
        x_c = gauss_newton_state(fwd_c, y_true, spec_c, Sy_inv_diag,
                                 label=f"[{row_lo}-{row_hi}] coarse", verbose=False)
        resid_c = y_true - fwd_c(x_c)
        # Standing rule: save the ENTIRE state vector (free AND frozen) and
        # the FULL residual field, never just summary scalars.
        out["coarse"] = spec_c.snapshot(x_c, resid=resid_c,
                                        resid_rms=float(np.sqrt(np.mean(resid_c ** 2))))
        out["x_coarse"] = x_c                      # back-compat with existing plotters
        out["resid_coarse"] = resid_c
        out["resid_coarse_rms"] = float(np.sqrt(np.mean(resid_c ** 2)))
        out["t_coarse"] = time.time() - t0

    a_lo, a_hi = max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD)
    anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / anchor_density)
    anchor_etas = np.sort(_eta_of(FPA, np.full(len(anchor_rows), 512.0),
                                  anchor_rows.astype(float)))
    t0 = time.time()
    # HI-RES: scene on the finer anchor grid, every row interpolated there.
    spec_h = state_spec_from_scene(bin_centers, free=free, corr_length=corr_length,
                                   prior_form=prior_form)
    fwd_h = build_forward_state(FPA, rows_win, anchor_etas, spec_h, spectrum,
                                wn_hires, ils, pad=PAD, state_interp=state_interp)
    x_h = gauss_newton_state(fwd_h, y_true, spec_h, Sy_inv_diag,
                             label=f"[{row_lo}-{row_hi}] hires", verbose=False)
    resid_h = y_true - fwd_h(x_h)
    out["hires"] = spec_h.snapshot(x_h, resid=resid_h,
                                   resid_rms=float(np.sqrt(np.mean(resid_h ** 2))))
    out["x_hires"] = x_h
    out["resid_hires"] = resid_h
    out["anchor_etas"] = anchor_etas
    out["G_eff"] = len(anchor_rows)
    out["resid_hires_rms"] = float(np.sqrt(np.mean(resid_h ** 2)))
    out["t_hires"] = time.time() - t0

    return out


def _worker(tile):
    row_lo, row_hi = tile
    try:
        return _solve_window(row_lo, row_hi)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        return dict(row_lo=row_lo, row_hi=row_hi, error=f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--sigma-abs", type=float, default=0.10)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--uniform", action="store_true", help="constant-atmosphere scene "
                    "(no real along-slit variation at all) -- a debugging aid: with a "
                    "genuinely uniform truth, correct bias is EXACTLY zero everywhere "
                    "(no bin-quantization/resolution-floor effect is even possible), so "
                    "any nonzero structure here is unambiguously implementation, not physics.")
    ap.add_argument("--g-ratio", type=float, default=G_RATIO, help="G = max(2, round(width/"
                    "g_ratio)) per window -- default matches the production convention "
                    "(G_RATIO=3). Pass 1.0 for 'one bin per row' (ground-footprint-tied "
                    "resolution, docs/JOINT_BLOCK_MIGRATION_PLAN.md Sec.9).")
    ap.add_argument("--task-id", type=int, default=None, help="SLURM-array-friendly "
                    "partitioning: process only tiles[task_id::n_tasks] (round-robin, so "
                    "wide/narrow windows are spread across tasks rather than grouped) and "
                    "write a per-task output file instead of the combined one. Requires "
                    "--n-tasks. Merge outputs afterward with gd_joint_block_whole_slit_merge.py.")
    ap.add_argument("--n-tasks", type=int, default=None, help="total number of tasks for "
                    "--task-id round-robin partitioning (e.g. SLURM_ARRAY_TASK_COUNT).")
    ap.add_argument("--hires-only", action="store_true",
                    help="skip the coarse solve (unchanged by --anchor-density/"
                         "--state-interp, so re-running it would just reproduce the baseline)")
    ap.add_argument("--anchor-density", type=int, default=1,
                    help="anchors per detector row for the hi-res forward model "
                         "(default 1 = the original one-per-row). >1 places anchors at "
                         "fractional row positions, shrinking the nearest-anchor step "
                         "error ~(1/4)|f'|h proportionally. Cost scales with it.")
    ap.add_argument("--state-interp", action="store_true",
                    help="build each hi-res anchor's atmosphere from state parameters "
                         "linearly interpolated from the G bin centres (CO2, CH4, CO, H2O, "
                         "p_surface alike) instead of exact truth at the anchor's own eta. "
                         "State-space interpolation + fresh RT per anchor; no spectrum blending.")
    ap.add_argument("--free", type=str, default="co2_ppm",
                    help="comma-separated state rows to retrieve; everything else is "
                         "frozen at local truth. Names from als.STATE_FIELDS, e.g. "
                         "co2_ppm,p_surface_hpa. CO2 is an ordinary row and may be frozen.")
    ap.add_argument("--corr-length", type=float, default=None,
                    help="prior correlation length in eta applied to EVERY row. Default "
                         "(unset) uses joint_state.DEFAULT_CORR_LENGTH_ETA -- each row's "
                         "own physical scale (co2 ~10km hot-spot, p_surface ~140km "
                         "topography, etc). A single shared value is rarely right, since "
                         "surface pressure and a CO2 hot spot do not share a scale.")
    ap.add_argument("--prior-form", type=str, default="exponential",
                    choices=["exponential", "tikhonov"],
                    help="'exponential' (default): sigma^2 exp(-|d_eta|/corr_length) in "
                         "PHYSICAL eta, correct under non-uniform bin spacing. "
                         "'tikhonov': the original gamma*(L^T L)+I/sigma^2 in bin index, "
                         "bit-identical to pre-2026-08-17 results.")
    args = ap.parse_args()
    if (args.task_id is None) != (args.n_tasks is None):
        ap.error("--task-id and --n-tasks must be given together")

    scene_label = "uniform" if args.uniform else "realistic"
    print(f"Building {scene_label}-scene FPA{FPA} band (renders the real 1024x1024 detector image)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400, None,
                           args.uniform, False, 32, False, 0)
    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    all_tiles = build_window_tiles(FPA)
    widths = [hi - lo + 1 for lo, hi in all_tiles]
    print(f"{len(all_tiles)} windows total, widths min={min(widths)} max={max(widths)} "
         f"mean={np.mean(widths):.1f}, total rows={sum(widths)}", flush=True)

    if args.task_id is not None:
        tiles = all_tiles[args.task_id::args.n_tasks]
        print(f"task {args.task_id}/{args.n_tasks}: {len(tiles)} windows assigned "
             f"(round-robin, tiles[{args.task_id}::{args.n_tasks}])", flush=True)
    else:
        tiles = all_tiles

    _SWEEP.update(dict(band=band, absco=absco, wide_inst=wide_inst, geo=geo, solar=solar,
                       albedo=albedo, wn_hires=band["wn_hires"], ils=band["ils"],
                       gamma=args.gamma, sigma_abs=args.sigma_abs, g_ratio=args.g_ratio,
                       uniform=args.uniform, atm_center=atm_center,
                       hires_only=args.hires_only, anchor_density=args.anchor_density,
                       state_interp=args.state_interp,
                       free=tuple(x.strip() for x in args.free.split(',')),
                       corr_length=args.corr_length, prior_form=args.prior_form))

    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    print(f"solving with {n_workers} workers...", flush=True)

    t0 = time.time()
    results = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for r in pool.imap_unordered(_worker, tiles, chunksize=1):
            key = (r["row_lo"], r["row_hi"])
            results[key] = r
            n_done += 1
            status = "ERROR: " + r["error"] if "error" in r else \
                     (f"G={r['G']} " + (f"t_coarse={r['t_coarse']:.0f}s "
                                        if "t_coarse" in r else "")
                      + f"t_hires={r['t_hires']:.0f}s")
            elapsed = time.time() - t0
            print(f"  {n_done}/{len(tiles)} rows {key[0]}-{key[1]}: {status} "
                 f"({elapsed:.0f}s elapsed)", flush=True)

    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nall done ({time.time()-t0:.0f}s): {n_ok}/{len(tiles)} windows solved successfully", flush=True)

    suffix = "_uniform" if args.uniform else ""
    suffix += f"_gratio{args.g_ratio:g}"
    if args.anchor_density != 1:
        suffix += f"_adens{args.anchor_density}"
    if args.state_interp:
        suffix += "_stateinterp"
    free_t = tuple(x.strip() for x in args.free.split(","))
    if free_t != ("co2_ppm",):
        suffix += "_free-" + "-".join(n.split("_")[0] for n in free_t)
    payload = {"results": results, "tiles": all_tiles, "fpa": FPA, "uniform": args.uniform,
              "gamma": args.gamma, "sigma_abs": args.sigma_abs,
              "g_ratio": args.g_ratio, "min_window": MIN_WINDOW, "pad": PAD,
              "hires_only": args.hires_only, "anchor_density": args.anchor_density,
              "state_interp": args.state_interp, "free": free_t,
              "corr_length": args.corr_length, "prior_form": args.prior_form}
    if args.task_id is not None:
        payload.update(task_id=args.task_id, n_tasks=args.n_tasks)
        out_dir = REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}{suffix}_parts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"task{args.task_id:03d}of{args.n_tasks}.pkl"
    else:
        out_path = REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}{suffix}.pkl"
        out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
