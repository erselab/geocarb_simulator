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
from gd_joint_block_retrieve import FPA, GERT_ROOT, _eta_of, band_basics  # noqa: E402
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
from geocarb_gert import jacobians as jac  # noqa: E402

MIN_WINDOW = 4
PAD = 4
G_RATIO = 3
ROW_MAX_IDX = 1023

_SWEEP = {}


def scale_for_window_count(fpa: int, target: int, min_window: int = MIN_WINDOW) -> float:
    """The `window_scale` whose tiling yields `target` windows.

    Window width is set by local keystone, so the count cannot be requested
    directly -- but it falls monotonically with the scale factor, so a coarse
    scan finds it. Returns the smallest scale hitting `target` exactly, or
    raises with the counts that ARE reachable, since not every target is
    (widths are integers, so the count moves in jumps).
    """
    seen = {}
    for s in np.arange(0.5, 6.001, 0.05):
        n = len(build_window_tiles(fpa, min_window=min_window, window_scale=float(s)))
        seen.setdefault(n, float(s))
        if n == target:
            return float(s)
    raise ValueError(f"no window_scale in [0.5, 6] gives {target} windows at "
                     f"min_window={min_window}; reachable counts near it: "
                     f"{sorted(k for k in seen if abs(k - target) <= 6)}")


def build_window_tiles(fpa: int, row_min: int = 0, row_max: int = ROW_MAX_IDX,
                       min_window: int = MIN_WINDOW,
                       window_scale: float = 1.0) -> list:
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
            r_new = max(min_window, int(round(window_scale * 2.2 * k_c)))
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
    # `uniform_priors` -- NOT the same as the sweep's own `--uniform` flag by
    # itself: it is set whenever the TRUE composition is spatially constant,
    # which is also the case under `--barcode` (fixed atmosphere, only
    # reflectance varies). `--realistic-barcode` does NOT set it -- its
    # composition genuinely varies along the slit, same as the plain
    # realistic scene, only with a reflectance pattern multiplied on top.
    # See `state_spec_from_scene`'s own `uniform` docstring for why this
    # has to reach the StateSpec priors, not just the rendered image.
    uniform_priors = _SWEEP["uniform_priors"]
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
    # kept only for gd_joint_block_whole_slit_plot.py's own backward-compat
    # reconstruction (prior_co2_ppm_bins * x_coarse); the real solve below
    # gets its priors from state_spec_from_scene(..., uniform=uniform_priors)
    prior_co2_ppm_bins = (np.full(G, float(als.xco2_ppm(0.0))) if uniform_priors
                          else np.array([float(als.xco2_ppm(xk))
                                        for xk in bin_centers * als.SLIT_HALF_KM]))

    y_true = band["A"][rows_win, :].ravel()
    y_scale = float(np.mean(np.abs(y_true)))
    Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)

    out = dict(row_lo=row_lo, row_hi=row_hi, width=width, G=G, bin_centers=bin_centers,
              prior_co2_ppm_bins=prior_co2_ppm_bins)

    free = tuple(_SWEEP.get("free", ("co2_ppm",)))
    corr_length = _SWEEP.get("corr_length")          # None -> per-row physical defaults
    prior_form = _SWEEP.get("prior_form", "exponential")
    spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)

    # Analytic Jacobians: derivatives from gert's per-layer arrays composed with
    # the detector operator, rather than n_free+1 forward evaluations.
    #
    # `linearize` unconditionally does state-space interpolation for every row
    # (`spec.interp_to`), with no equivalent of `build_forward_state`'s
    # `state_interp=False` "exact truth at anchor" override for frozen rows.
    # That is fine for COARSE regardless of the sweep's own `state_interp`
    # flag -- `fwd_c` below is itself always built with `state_interp=True`,
    # since `scene_etas = bin_centers` there makes interpolation the identity
    # (see that call's own comment) -- but it is a REAL mismatch for HIRES
    # whenever `state_interp=False`: `linearize` would silently interpolate a
    # frozen row that `fwd_h` evaluates at exact truth, changing what the
    # `adens4`-style configs are actually testing. So analytic is used for
    # hires only when the sweep also requested `state_interp=True`; otherwise
    # hires falls back to FD and `jacobian_used` records that per solve below,
    # rather than the top-level `jacobian` flag silently overstating it.
    use_analytic = _SWEEP.get("jacobian", "fd") == "analytic"
    use_analytic_hires = use_analytic and state_interp
    spectrum_jac = (jac.make_spectrum_jac(absco, wide_inst, geo, solar, albedo)
                    if use_analytic else None)

    def _linearizer(spec_, scene_etas_, enabled):
        if not enabled:
            return None
        def lin(x):
            return jac.linearize(FPA, rows_win, scene_etas_, spec_, spectrum_jac,
                                 wn_hires, ils, x, pad=PAD)
        return lin

    # COARSE: scene evaluated at the state's own bin centres, so
    # interp_to() is the identity -- --state-interp has no content here.
    if not hires_only:
        t0 = time.time()
        spec_c = state_spec_from_scene(bin_centers, free=free, corr_length=corr_length,
                                       prior_form=prior_form, uniform=uniform_priors)
        # coarse: scene positions ARE the state positions, so interpolation is
        # the identity either way -- state_interp is genuinely a no-op here.
        fwd_c = build_forward_state(FPA, rows_win, bin_centers, spec_c, spectrum,
                                    wn_hires, ils, pad=PAD, state_interp=True)
        x_c = gauss_newton_state(fwd_c, y_true, spec_c, Sy_inv_diag,
                                 label=f"[{row_lo}-{row_hi}] coarse", verbose=False,
                                 jacobian_fn=_linearizer(spec_c, bin_centers, use_analytic))
        resid_c = y_true - fwd_c(x_c)
        # Standing rule: save the ENTIRE state vector (free AND frozen) and
        # the FULL residual field, never just summary scalars. `jacobian_used`
        # is the solve's OWN record, not the requested `--jacobian` flag --
        # see the note above on why hires can silently differ from coarse.
        out["coarse"] = spec_c.snapshot(x_c, resid=resid_c,
                                        resid_rms=float(np.sqrt(np.mean(resid_c ** 2))),
                                        jacobian_used=("analytic" if use_analytic else "fd"))
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
                                   prior_form=prior_form, uniform=uniform_priors)
    fwd_h = build_forward_state(FPA, rows_win, anchor_etas, spec_h, spectrum,
                                wn_hires, ils, pad=PAD, state_interp=state_interp)
    x_h = gauss_newton_state(fwd_h, y_true, spec_h, Sy_inv_diag,
                             label=f"[{row_lo}-{row_hi}] hires", verbose=False,
                             jacobian_fn=_linearizer(spec_h, anchor_etas, use_analytic_hires))
    resid_h = y_true - fwd_h(x_h)
    out["hires"] = spec_h.snapshot(x_h, resid=resid_h,
                                   resid_rms=float(np.sqrt(np.mean(resid_h ** 2))),
                                   jacobian_used=("analytic" if use_analytic_hires else "fd"))
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
    ap.add_argument("--barcode", action="store_true",
                    help="fixed-atmosphere scene with a barcode REFLECTANCE pattern (sharp "
                         "brightness bars alternating along the slit, composition held at "
                         "the single center atmosphere) -- an isolated test of whether a "
                         "sharp reflectance-only boundary corrupts the retrieval via "
                         "keystone/PSF cross-row mixing, with no real composition signal "
                         "also in play. Implies uniform=True state priors automatically "
                         "(the true composition IS uniform here), independent of whether "
                         "--uniform was also passed. Mutually exclusive with "
                         "--realistic-barcode.")
    ap.add_argument("--realistic-barcode", action="store_true",
                    help="the normal realistic along-slit composition scene (real "
                         "als.atmosphere_at variation, same priors as a plain run) with a "
                         "barcode reflectance pattern multiplied on top -- tests whether a "
                         "sharp reflectance boundary makes bias WORSE on top of the "
                         "mechanisms already present in the realistic scene, rather than in "
                         "isolation. Do not combine with --uniform: this scene's true "
                         "composition is deliberately non-uniform, so uniform priors would "
                         "be a real prior/truth mismatch, not a simplification. Mutually "
                         "exclusive with --barcode.")
    ap.add_argument("--barcode-bars", type=int, default=32,
                    help="number of alternating-brightness bars across the slit for "
                         "--barcode/--realistic-barcode (default 32, matching gd_test.py's "
                         "own default). Ignored otherwise.")
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
    ap.add_argument("--jacobian", type=str, default="analytic", choices=["fd", "analytic"],
                    help="'analytic' (default since 2026-08-19) uses "
                         "geocarb_gert.jacobians.linearize -- derivatives assembled from "
                         "gert's per-layer arrays, one evaluation per iteration, no step "
                         "size. Agreement with 'fd' is established (16/16 PASS in the "
                         "closed regression, plus both anchor_density=4 configs; state "
                         "agreement 1e-7..1e-8 throughout), so this is no longer opt-in. "
                         "Coarse always gets analytic regardless of --state-interp; hires "
                         "only does when --state-interp is also passed (otherwise it "
                         "falls back to 'fd', printed as a NOTE below and recorded per-"
                         "solve in jacobian_used -- validated bit-identical to a plain "
                         "'fd' run of the same config). 'fd' finite-differences the "
                         "forward model instead, n_free+1 evaluations per iteration -- "
                         "pass it explicitly to re-validate after a real change to "
                         "geocarb_gert/jacobians.py or joint_state.py, or to exercise a "
                         "state target (dispersion, albedo) neither matrix covered.")
    ap.add_argument("--n-windows", type=int, default=None,
                    help="target number of slit windows; solved for via "
                         "scale_for_window_count. Default (None) keeps the historical "
                         "keystone-only tiling, which gives 58 on FPA2.")
    ap.add_argument("--window-scale", type=float, default=1.0,
                    help="multiplier on the keystone window-radius formula; larger "
                         "means fewer, wider windows. Ignored when --n-windows is given.")
    ap.add_argument("--min-window", type=int, default=MIN_WINDOW,
                    help=f"minimum window radius (default {MIN_WINDOW})")
    ap.add_argument("--out", type=str, default=None,
                    help="output pickle path (default: derived from the run settings)")
    ap.add_argument("--prior-form", type=str, default="exponential",
                    choices=["exponential", "tikhonov"],
                    help="'exponential' (default): sigma^2 exp(-|d_eta|/corr_length) in "
                         "PHYSICAL eta, correct under non-uniform bin spacing. "
                         "'tikhonov': the original gamma*(L^T L)+I/sigma^2 in bin index, "
                         "bit-identical to pre-2026-08-17 results.")
    args = ap.parse_args()
    if (args.task_id is None) != (args.n_tasks is None):
        ap.error("--task-id and --n-tasks must be given together")
    if args.barcode and args.realistic_barcode:
        ap.error("--barcode and --realistic-barcode are mutually exclusive "
                 "(gd_test._band_setup's own if/elif -- fixed atmosphere+brightness only, "
                 "vs. real composition variation+brightness on top; pick one)")
    if args.realistic_barcode and args.uniform:
        ap.error("--realistic-barcode's whole point is non-uniform composition; "
                 "--uniform would force flat state priors against a truth that "
                 "deliberately is not flat -- a real prior/truth mismatch, not a "
                 "simplification. Drop --uniform.")
    # NOT just args.uniform: --barcode's true composition is also spatially constant
    # (only reflectance varies), so it needs the same flat StateSpec priors --
    # see state_spec_from_scene's own `uniform` docstring for why this has to reach
    # the actual solve, not just the rendered image.
    uniform_priors = args.uniform or args.barcode

    scene_label = ("barcode" if args.barcode else
                  "realistic-barcode" if args.realistic_barcode else
                  "uniform" if args.uniform else "realistic")
    print(f"Building {scene_label}-scene FPA{FPA} band (renders the real 1024x1024 detector image)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400, None,
                           args.uniform, args.barcode, args.barcode_bars, False, 0,
                           args.realistic_barcode)
    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    if args.n_windows is not None:
        window_scale = scale_for_window_count(FPA, args.n_windows, args.min_window)
        print(f"--n-windows {args.n_windows} -> window_scale {window_scale:.2f}", flush=True)
    else:
        window_scale = args.window_scale
    all_tiles = build_window_tiles(FPA, min_window=args.min_window,
                                   window_scale=window_scale)
    widths = [hi - lo + 1 for lo, hi in all_tiles]
    print(f"{len(all_tiles)} windows total, widths min={min(widths)} max={max(widths)} "
         f"mean={np.mean(widths):.1f}, total rows={sum(widths)}", flush=True)
    if args.jacobian == "analytic" and not args.state_interp:
        print("NOTE: --jacobian analytic without --state-interp -- coarse uses analytic, "
             "hires falls back to fd (linearize has no state_interp=False equivalent for "
             "frozen rows). Each solve's snapshot records its own jacobian_used.", flush=True)

    if args.task_id is not None:
        tiles = all_tiles[args.task_id::args.n_tasks]
        print(f"task {args.task_id}/{args.n_tasks}: {len(tiles)} windows assigned "
             f"(round-robin, tiles[{args.task_id}::{args.n_tasks}])", flush=True)
    else:
        tiles = all_tiles

    _SWEEP.update(dict(band=band, absco=absco, wide_inst=wide_inst, geo=geo, solar=solar,
                       albedo=albedo, wn_hires=band["wn_hires"], ils=band["ils"],
                       gamma=args.gamma, sigma_abs=args.sigma_abs, g_ratio=args.g_ratio,
                       uniform=args.uniform, uniform_priors=uniform_priors, atm_center=atm_center,
                       hires_only=args.hires_only, anchor_density=args.anchor_density,
                       state_interp=args.state_interp,
                       free=tuple(x.strip() for x in args.free.split(',')),
                       corr_length=args.corr_length, prior_form=args.prior_form,
                       jacobian=args.jacobian))

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
    if args.barcode:
        suffix += f"_barcode{args.barcode_bars}"
    elif args.realistic_barcode:
        suffix += f"_realisticbarcode{args.barcode_bars}"
    suffix += f"_gratio{args.g_ratio:g}"
    if args.anchor_density != 1:
        suffix += f"_adens{args.anchor_density}"
    if args.state_interp:
        suffix += "_stateinterp"
    free_t = tuple(x.strip() for x in args.free.split(","))
    if free_t != ("co2_ppm",):
        suffix += "_free-" + "-".join(n.split("_")[0] for n in free_t)
    if len(all_tiles) != 58:
        suffix += f"_nwin{len(all_tiles)}"
    if args.jacobian != "fd":
        suffix += f"_{args.jacobian}"
    payload = {"results": results, "tiles": all_tiles, "fpa": FPA, "uniform": args.uniform,
              "barcode": args.barcode, "realistic_barcode": args.realistic_barcode,
              "barcode_bars": args.barcode_bars if (args.barcode or args.realistic_barcode) else None,
              "uniform_priors": uniform_priors,
              "gamma": args.gamma, "sigma_abs": args.sigma_abs,
              "g_ratio": args.g_ratio, "min_window": MIN_WINDOW, "pad": PAD,
              "hires_only": args.hires_only, "anchor_density": args.anchor_density,
              "state_interp": args.state_interp, "free": free_t,
              "corr_length": args.corr_length, "prior_form": args.prior_form,
              "jacobian": args.jacobian, "n_windows": len(all_tiles),
              "window_scale": window_scale}
    if args.task_id is not None:
        payload.update(task_id=args.task_id, n_tasks=args.n_tasks)
        out_dir = REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}{suffix}_parts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"task{args.task_id:03d}of{args.n_tasks}.pkl"
    else:
        out_path = (Path(args.out) if args.out else
                    REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}{suffix}.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
