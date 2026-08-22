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
from geocarb_gert.mission_config import RetrievalDefaults  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402

# Defensive floor on the real noise model's sigma (Phase D of the
# config-consolidation plan), radiance units (W/m^2/sr/um, same as `A`).
# Not currently load-bearing -- linear_shot_noise_params solves N0 > 0 for
# all four calibrated bands in RADIOMETRIC_SPEC_BY_FPA today -- but guards
# against a future band whose N0 ~ 0 producing inf in Sy_inv_diag at a
# literal-zero-radiance pixel.
_SIGMA_FLOOR = 1e-6

# Sourced from input/retrieval_defaults.yml's tiling: block at import time
# (Phase C of the config-consolidation plan) -- were typed-inline literals.
# Always reflect the CHECKED-IN default YAML, not whatever --config resolves
# to at runtime: these are baked into build_window_tiles/scale_for_window_
# count's own function-signature defaults at import time, before argparse
# ever runs, so a per-run --config choice cannot reach them this way -- see
# main()'s own --config pre-pass for the CLI-flag-level mechanism that DOES
# honor --config (the actual `--min-window`/`--g-ratio` values used for any
# given run always come from args.min_window/args.g_ratio, not these bare
# module constants).
_defaults = RetrievalDefaults.from_yaml()
MIN_WINDOW = _defaults.min_window
PAD = _defaults.pad
G_RATIO = _defaults.g_ratio
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
    # Shadows the module-level `FPA` import for the rest of this function:
    # every bare `FPA` reference below (_eta_of, jac.linearize) now resolves
    # to the active run's --fpa choice (Phase C of the config-consolidation
    # plan) rather than the fixed gd_joint_block_retrieve.FPA constant.
    FPA = _SWEEP["fpa"]
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
    state_interp = _SWEEP.get("state_interp", "linear")  # interp1d kind: "linear" or "nearest"
    prior_fields = _SWEEP.get("prior_fields")            # an als.PRIOR_FIELD_SETS[...] dict, always explicit (never None)
    prior_anchor_density = _SWEEP.get("prior_anchor_density")  # None -> exact-per-bin (today's default)

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
    # Real per-pixel noise (Phase D of the config-consolidation plan),
    # replacing the old flat-scalar Sy_inv_diag = 1/mean(|signal|)^2 for
    # the whole window. geocarb_noise_model(FPA) is the same LinearShotNoise
    # gd_test.py::_band_setup already computes (and previously discarded
    # after the saturation check) from RADIOMETRIC_SPEC_BY_FPA's real
    # per-band calibration: sigma(I)^2 = N0^2 + N1*|I|, so a dim pixel
    # (window edge, dark scene) gets a correspondingly tighter noise floor
    # instead of inheriting the window's own mean-|signal| scale. Routed
    # through the NoiseModel.sigma() interface rather than inlining the
    # formula, so any future change to the noise model's own definition is
    # picked up here automatically. `windows` is accepted but unused by
    # LinearShotNoise.sigma() (only reads R/N0/N1), so `[None]` is a valid
    # placeholder, not a real per-window object.
    #
    # THIS CHANGES RETRIEVED NUMBERS relative to every run before this
    # commit, including the Phase 1 bingrid sweep and the imperfect-prior
    # sweep from this session -- deliberate, not opt-in as a PRODUCTION
    # default (config-consolidation plan Phase D). --flat-sy-inv below is
    # NOT a reversal of that decision -- it exists solely so a deliberate,
    # labeled A/B comparison against the old behavior stays reproducible
    # without reverting real code; the default here is unchanged.
    if _SWEEP.get("flat_sy_inv", False):
        # Reproduces the pre-Phase-D formula exactly, for comparison only
        # -- see docs/PROJECT_STATUS.md Sec.6 and --flat-sy-inv's own help.
        y_scale = float(np.mean(np.abs(y_true)))
        Sy_inv_diag = np.full(y_true.size, 1.0 / y_scale ** 2)
    else:
        noise_model = geocarb_noise_model(FPA)
        sigma = noise_model.sigma([y_true], [None])
        Sy_inv_diag = 1.0 / np.maximum(sigma, _SIGMA_FLOOR) ** 2

    out = dict(row_lo=row_lo, row_hi=row_hi, width=width, G=G, bin_centers=bin_centers,
              prior_co2_ppm_bins=prior_co2_ppm_bins)

    free = tuple(_SWEEP.get("free", ("co2_ppm",)))
    corr_length = _SWEEP.get("corr_length")          # None -> per-row physical defaults
    prior_form = _SWEEP.get("prior_form", "exponential")
    spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)

    # Analytic Jacobians: derivatives from gert's per-layer arrays composed with
    # the detector operator, rather than n_free+1 forward evaluations.
    #
    # `linearize` and `build_forward_state` both take the same `state_interp`
    # kind and both apply it to EVERY row, free or frozen, with no bypass --
    # the old asymmetry (frozen rows silently exact-at-anchor while free rows
    # interpolated, which `linearize` had no equivalent for) is gone along
    # with `state_interp=False`'s old exact-truth override. So analytic is
    # available for hires unconditionally now, the same as coarse always was.
    use_analytic = _SWEEP.get("jacobian", "fd") == "analytic"
    use_analytic_hires = use_analytic
    spectrum_jac = (jac.make_spectrum_jac(absco, wide_inst, geo, solar, albedo)
                    if use_analytic else None)

    def _linearizer(spec_, scene_etas_, enabled, interp_kind):
        if not enabled:
            return None
        def lin(x):
            return jac.linearize(FPA, rows_win, scene_etas_, spec_, spectrum_jac,
                                 wn_hires, ils, x, pad=PAD, state_interp=interp_kind)
        return lin

    # COARSE: scene evaluated at the state's own bin centres, so
    # interp_to() is the identity -- state_interp's kind has no content here.
    if not hires_only:
        t0 = time.time()
        spec_c = state_spec_from_scene(bin_centers, free=free, corr_length=corr_length,
                                       prior_form=prior_form, uniform=uniform_priors,
                                       fields=prior_fields,
                                       prior_anchor_density=prior_anchor_density)
        # coarse: scene positions ARE the state positions, so interpolation is
        # the identity regardless of kind -- state_interp is genuinely a no-op here.
        fwd_c = build_forward_state(FPA, rows_win, bin_centers, spec_c, spectrum,
                                    wn_hires, ils, pad=PAD, state_interp="linear")
        x_c = gauss_newton_state(fwd_c, y_true, spec_c, Sy_inv_diag,
                                 label=f"[{row_lo}-{row_hi}] coarse", verbose=False,
                                 jacobian_fn=_linearizer(spec_c, bin_centers, use_analytic, "linear"))
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
                                   prior_form=prior_form, uniform=uniform_priors,
                                   fields=prior_fields,
                                   prior_anchor_density=prior_anchor_density)
    fwd_h = build_forward_state(FPA, rows_win, anchor_etas, spec_h, spectrum,
                                wn_hires, ils, pad=PAD, state_interp=state_interp)
    x_h = gauss_newton_state(fwd_h, y_true, spec_h, Sy_inv_diag,
                             label=f"[{row_lo}-{row_hi}] hires", verbose=False,
                             jacobian_fn=_linearizer(spec_h, anchor_etas, use_analytic_hires,
                                                     state_interp))
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


def _resolve_cli_defaults() -> RetrievalDefaults:
    """Parse just --config (if given) before building the real argparse
    parser below, so every other flag's own `default=` can reflect it.
    argparse has no mechanism for one add_argument's default to depend on
    an earlier flag's value within a single parse, so this is a small,
    deliberate two-pass parse -- everything downstream (args.gamma,
    args.g_ratio, ...) still comes from the real, single ap.parse_args()
    call in main(); this only decides what its defaults are."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    return RetrievalDefaults.from_yaml(pre_args.config)


def main() -> int:
    cfg = _resolve_cli_defaults()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None,
                    help="path to retrieval_defaults.yml, overriding every flag's own "
                         "default below (default: the checked-in input/retrieval_defaults.yml)")
    ap.add_argument("--gamma", type=float, default=cfg.gamma)
    ap.add_argument("--sigma-abs", type=float, default=cfg.sigma_abs)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--no-truth-cache", action="store_true",
                    help="always re-render the truth detector image instead of reusing a "
                         "cached one from results/truth_cache/ (see geocarb_gert.truth_cache). "
                         "Caching is on by default -- every config in a matrix sweep that "
                         "shares a scene (fpa, --uniform/--barcode/--realistic-barcode, "
                         "--barcode-bars, n_lookup_samples) hits the same cache entry, so "
                         "only the first config in a sweep pays the render cost. Pass this "
                         "to force a fresh render, e.g. after a change to the rendering code "
                         "itself that hasn't bumped truth_cache.TRUTH_CACHE_VERSION yet.")
    ap.add_argument("--flat-sy-inv", action="store_true",
                    help="reproduce the pre-Phase-D Sy_inv weighting exactly (Sy_inv_diag = "
                         "1/mean(|signal|)^2, uniform across the whole window) instead of the "
                         "real per-pixel noise model (the default). NOT a production option -- "
                         "exists solely to reproduce old comparison points deliberately (e.g. "
                         "the imperfect-prior 'first try' results computed before Phase D of "
                         "the config-consolidation plan landed) without checking out old code. "
                         "Adds a _flatsyinv suffix to the output filename so it can never "
                         "collide with a corrected-weighting run of the same config. See "
                         "docs/PROJECT_STATUS.md Sec.6.")
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
    ap.add_argument("--g-ratio", type=float, default=cfg.g_ratio, help="G = max(2, round(width/"
                    "g_ratio)) per window -- default from input/retrieval_defaults.yml "
                    "(g_ratio=3 in the checked-in config). Pass 1.0 for 'one bin per row' "
                    "(ground-footprint-tied resolution, docs/JOINT_BLOCK_MIGRATION_PLAN.md Sec.9).")
    ap.add_argument("--fpa", type=str, default=",".join(str(f) for f in cfg.default_fpa),
                    help="comma-separated GeoCarb band index/indices (0=O2_A, 1=CO2_weak, "
                         "2=CO2_strong, 3=CH4_CO). Default from input/retrieval_defaults.yml's "
                         "band.default_fpa. Currently must resolve to exactly one band -- "
                         "multi-band joint retrieval is a later phase of the "
                         "config-consolidation plan, not yet implemented in this script.")
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
    ap.add_argument("--anchor-density", type=int, default=cfg.anchor_density,
                    help="anchors per detector row for the hi-res forward model "
                         "(default 1 = the original one-per-row). >1 places anchors at "
                         "fractional row positions, shrinking the nearest-anchor step "
                         "error ~(1/4)|f'|h proportionally. Cost scales with it.")
    ap.add_argument("--state-interp", type=str, default=cfg.state_interp,
                    choices=["linear", "nearest"],
                    help="how each hi-res anchor's atmosphere is built from the G bin "
                         "centres (CO2, CH4, CO, H2O, p_surface alike) -- a "
                         "scipy.interpolate.interp1d kind. 'linear' (default): the usual "
                         "state-space blend between the two nearest bins, fresh RT per "
                         "anchor, no spectrum ever blended. 'nearest': piecewise-constant, "
                         "each anchor takes its single nearest bin's own value -- the other "
                         "natural downscaling scheme, now a real forward-model choice with "
                         "a matching analytic Jacobian (geocarb_gert.jacobians.linearize), "
                         "not just a diagnostic. Found 2026-08-19 (user): there is "
                         "deliberately no 'exact truth at anchor' option any more -- every "
                         "row, free or frozen, always goes through this same interpolation; "
                         "a row that should be as-good-as-truth gets that via the PRIOR "
                         "(--free, or a densely-constructed fields=) instead.")
    ap.add_argument("--free", type=str, default=",".join(cfg.free),
                    help="comma-separated state rows to retrieve; everything else is "
                         "frozen at local truth. Names from als.STATE_FIELDS, e.g. "
                         "co2_ppm,p_surface_hpa. CO2 is an ordinary row and may be frozen.")
    ap.add_argument("--corr-length", type=float, default=None,
                    help="prior correlation length in eta applied to EVERY row. Default "
                         "(unset) uses joint_state.DEFAULT_CORR_LENGTH_ETA -- each row's "
                         "own physical scale (co2 ~10km hot-spot, p_surface ~140km "
                         "topography, etc). A single shared value is rarely right, since "
                         "surface pressure and a CO2 hot spot do not share a scale.")
    ap.add_argument("--jacobian", type=str, default=cfg.jacobian, choices=["fd", "analytic"],
                    help="'analytic' (default since 2026-08-19) uses "
                         "geocarb_gert.jacobians.linearize -- derivatives assembled from "
                         "gert's per-layer arrays, one evaluation per iteration, no step "
                         "size. Agreement with 'fd' is established (16/16 PASS in the "
                         "closed regression, plus both anchor_density=4 configs; state "
                         "agreement 1e-7..1e-8 throughout) for --state-interp linear, so "
                         "this is no longer opt-in. Coarse and hires both always get "
                         "analytic now, for either --state-interp kind -- 'linear' and "
                         "'nearest' both have matching analytic Jacobians (no more free/"
                         "frozen asymmetry to fall back to FD for). 'fd' finite-differences "
                         "the forward model instead, n_free+1 evaluations per iteration -- "
                         "pass it explicitly to re-validate after a real change to "
                         "geocarb_gert/jacobians.py or joint_state.py, to validate the "
                         "'nearest' kind's own analytic Jacobian (see "
                         "scripts/gd_jacobian_validate.py --interp-kind nearest), or to "
                         "exercise a state target (dispersion, albedo) neither matrix "
                         "covered.")
    ap.add_argument("--n-windows", type=int, default=None,
                    help="target number of slit windows; solved for via "
                         "scale_for_window_count. Default (None) keeps the historical "
                         "keystone-only tiling, which gives 58 on FPA2.")
    ap.add_argument("--window-scale", type=float, default=1.0,
                    help="multiplier on the keystone window-radius formula; larger "
                         "means fewer, wider windows. Ignored when --n-windows is given.")
    ap.add_argument("--min-window", type=int, default=cfg.min_window,
                    help=f"minimum window radius (default {cfg.min_window})")
    ap.add_argument("--out", type=str, default=None,
                    help="output pickle path (default: derived from the run settings)")
    ap.add_argument("--prior-form", type=str, default=cfg.prior_form,
                    choices=["exponential", "tikhonov"],
                    help="'exponential' (default): sigma^2 exp(-|d_eta|/corr_length) in "
                         "PHYSICAL eta, correct under non-uniform bin spacing. "
                         "'tikhonov': the original gamma*(L^T L)+I/sigma^2 in bin index, "
                         "bit-identical to pre-2026-08-17 results.")
    ap.add_argument("--prior-fields", type=str, default=cfg.prior_fields,
                    choices=sorted(als.PRIOR_FIELD_SETS),
                    help="named prior field-set from als.PRIOR_FIELD_SETS (default: "
                         "'exact', prior=truth). 'structural': als.STATE_FIELDS_PRIOR "
                         "(background/topography-aware, never localized plume/hot-spot/"
                         "synoptic content -- the retired --realistic-prior=True case). "
                         "'co2_plus1pct'/'co2_plus1pct_psurf_minus1pct': uniform "
                         "multiplicative-bias priors -- full truth structure kept, only "
                         "the absolute level is wrong, isolating that failure mode from "
                         "structural's 'missing detail' one. Orthogonal to --uniform/"
                         "--barcode/--realistic-barcode, which flatten the TRUTH scene -- "
                         "this only changes what the prior knows, truth stays fully "
                         "realistic. Any non-'exact' choice defaults --n-lookup-samples "
                         "to 5600 and --out-root to results/realistic_prior unless "
                         "overridden (same convention every prior in this experiment "
                         "class shares, for comparability).")
    ap.add_argument("--n-lookup-samples", type=int, default=None,
                    help="along-slit samples for build_lookup_radiance (default: 400, or "
                         "5600 under a non-'exact' --prior-fields). Governs how well the "
                         "rendered TRUTH image resolves the ~10km-wide hot spots (400 -> "
                         "7km spacing, ~3 samples/FWHM; 5600 -> 0.5km spacing, ~20 "
                         "samples/FWHM). One-time forward-rendering cost, not paid per-solve.")
    ap.add_argument("--out-root", type=str, default=None,
                    help="root directory for output files, replacing 'results' (default: "
                         "'results', or 'results/realistic_prior' under a non-'exact' "
                         "--prior-fields).")
    ap.add_argument("--prior-anchor-density", type=float, default=None,
                    help="resolution knob for the prior, independent of --prior-fields: "
                         "None (default) samples the prior fields exactly at each bin's own "
                         "position (today's mechanism). 1.0 is an explicit way to ask for "
                         "the same thing ('just bin centers'). <1.0 builds the prior from "
                         "fewer anchor points than bins, spread evenly across the window and "
                         "linearly interpolated -- a genuinely coarser prior that smooths "
                         "away sub-anchor-spacing structure even if the underlying field is "
                         "exact truth. >1.0 oversamples. See "
                         "geocarb_gert.joint_state.state_spec_from_scene's own docstring.")
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
    if args.prior_fields != "exact" and (args.uniform or args.barcode or args.realistic_barcode):
        ap.error("--prior-fields changes what the PRIOR knows; --uniform/--barcode/"
                 "--realistic-barcode flatten the TRUTH scene -- combining them mixes two "
                 "different kinds of idealization/de-idealization in one run. Drop one.")
    fpa_list = tuple(int(x.strip()) for x in args.fpa.split(","))
    if len(fpa_list) != 1:
        ap.error(f"--fpa currently supports exactly one band (got {args.fpa!r}); multi-band "
                 "joint retrieval is a later phase of the config-consolidation plan, not yet "
                 "implemented in this script")
    fpa = fpa_list[0]
    n_lookup_samples = (args.n_lookup_samples if args.n_lookup_samples is not None
                        else 5600 if args.prior_fields != "exact" else 400)
    out_root = Path(args.out_root if args.out_root is not None
                    else "results/realistic_prior" if args.prior_fields != "exact" else "results")
    # NOT just args.uniform: --barcode's true composition is also spatially constant
    # (only reflectance varies), so it needs the same flat StateSpec priors --
    # see state_spec_from_scene's own `uniform` docstring for why this has to reach
    # the actual solve, not just the rendered image.
    uniform_priors = args.uniform or args.barcode

    scene_label = ("barcode" if args.barcode else
                  "realistic-barcode" if args.realistic_barcode else
                  "uniform" if args.uniform else "realistic")
    print(f"Building {scene_label}-scene FPA{fpa} band (renders the real 1024x1024 detector image)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
    band = gdt._band_setup_cached(fpa, atm_center, absco, geo, solar, snr, n_lookup_samples, None,
                                  args.uniform, args.barcode, args.barcode_bars, False, 0,
                                  args.realistic_barcode, use_cache=not args.no_truth_cache)
    wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    if args.n_windows is not None:
        window_scale = scale_for_window_count(fpa, args.n_windows, args.min_window)
        print(f"--n-windows {args.n_windows} -> window_scale {window_scale:.2f}", flush=True)
    else:
        window_scale = args.window_scale
    all_tiles = build_window_tiles(fpa, min_window=args.min_window,
                                   window_scale=window_scale)
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
                       albedo=albedo, wn_hires=band["wn_hires"], ils=band["ils"], fpa=fpa,
                       gamma=args.gamma, sigma_abs=args.sigma_abs, g_ratio=args.g_ratio,
                       uniform=args.uniform, uniform_priors=uniform_priors, atm_center=atm_center,
                       hires_only=args.hires_only, anchor_density=args.anchor_density,
                       state_interp=args.state_interp,
                       free=tuple(x.strip() for x in args.free.split(',')),
                       corr_length=args.corr_length, prior_form=args.prior_form,
                       jacobian=args.jacobian,
                       prior_fields=als.PRIOR_FIELD_SETS[args.prior_fields],
                       prior_anchor_density=args.prior_anchor_density,
                       flat_sy_inv=args.flat_sy_inv))

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
    if args.state_interp != "linear":
        suffix += f"_{args.state_interp}"
    free_t = tuple(x.strip() for x in args.free.split(","))
    if free_t != ("co2_ppm",):
        suffix += "_free-" + "-".join(n.split("_")[0] for n in free_t)
    if len(all_tiles) != 58:
        suffix += f"_nwin{len(all_tiles)}"
    if args.jacobian != "fd":
        suffix += f"_{args.jacobian}"
    if args.prior_fields != "exact":
        suffix += f"_prior-{args.prior_fields}"
    if args.prior_anchor_density is not None:
        suffix += f"_prad{args.prior_anchor_density:g}"
    if args.flat_sy_inv:
        suffix += "_flatsyinv"
    payload = {"results": results, "tiles": all_tiles, "fpa": fpa, "uniform": args.uniform,
              "barcode": args.barcode, "realistic_barcode": args.realistic_barcode,
              "barcode_bars": args.barcode_bars if (args.barcode or args.realistic_barcode) else None,
              "uniform_priors": uniform_priors,
              "gamma": args.gamma, "sigma_abs": args.sigma_abs,
              # min_window: args.min_window (the value actually used to tile),
              # not the bare MIN_WINDOW default constant -- fixed alongside the
              # Phase C config wiring (pre-existing: previously always recorded
              # the constant even when --min-window overrode it).
              "g_ratio": args.g_ratio, "min_window": args.min_window, "pad": PAD,
              "hires_only": args.hires_only, "anchor_density": args.anchor_density,
              "state_interp": args.state_interp, "free": free_t,
              "corr_length": args.corr_length, "prior_form": args.prior_form,
              "jacobian": args.jacobian, "n_windows": len(all_tiles),
              "window_scale": window_scale, "prior_fields": args.prior_fields,
              "n_lookup_samples": n_lookup_samples,
              "prior_anchor_density": args.prior_anchor_density,
              "flat_sy_inv": args.flat_sy_inv}
    if args.task_id is not None:
        payload.update(task_id=args.task_id, n_tasks=args.n_tasks)
        out_dir = REPO_ROOT / out_root / f"gd_joint_block_whole_slit_fpa{fpa}{suffix}_parts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"task{args.task_id:03d}of{args.n_tasks}.pkl"
    else:
        out_path = (Path(args.out) if args.out else
                    REPO_ROOT / out_root / f"gd_joint_block_whole_slit_fpa{fpa}{suffix}.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
