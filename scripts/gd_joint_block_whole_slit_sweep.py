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
import functools
import hashlib
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
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.joint_state import (build_forward_state, gauss_newton_state,  # noqa: E402
                                      state_spec_from_scene, default_pad_for_psf)
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
                       window_scale: float = 1.0, overlap: int = 0) -> list:
    """Non-overlapping tiling of [row_min, row_max] by default (overlap=0,
    unchanged from before). Each window's own radius is a small
    fixed-point solve (window width depends on rows_crossed at the
    window's own center, which depends on width) -- converges in a couple
    of iterations since rows_crossed varies slowly.

    `overlap`: rows of symmetric overlap ADDED to each internal boundary
    AFTER the base non-overlapping tiling above is computed -- window i's
    own row_hi extends by `overlap` past its natural boundary, and window
    i+1's own row_lo retreats by `overlap` past the same boundary, so the
    two windows share `2*overlap` real rows (clamped at the slit's own two
    ends, which have no neighbor to overlap into). Purely additive:
    `overlap=0` (the default) reproduces today's exact tiling, unchanged,
    for every existing caller. Lets `geocarb_gert.along_slit_query.
    query_state` blend two independent windows' own estimates in the
    shared rows instead of either window's own boundary value being used
    un-blended -- see docs/PROJECT_STATUS.md Sec.6's window-boundary
    overshoot/undershoot finding for why that matters (it's a genuine
    sign-reversing bias, which averaging directly cancels). The windows'
    own retrieved bins feed `geocarb_gert.along_slit_state.
    stack_windows_along_slit` first, which is what `query_state` reads.

    Keep `overlap` small relative to the narrowest window (`2*min_window+1`
    rows, e.g. 9 at the default MIN_WINDOW=4) -- this function does not
    guard against `overlap` large enough that a window ends up overlapping
    BOTH of its neighbors at once (a 3-way overlap merge_windows_along_
    slit does not attempt to reconstruct correctly beyond simple N-way
    inverse-variance combination, which is still valid, just not the
    2-window case this was designed and validated against).
    """
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
    if overlap > 0 and len(tiles) > 1:
        widened = []
        for i, (lo, hi) in enumerate(tiles):
            new_lo = lo - overlap if i > 0 else lo
            new_hi = hi + overlap if i < len(tiles) - 1 else hi
            widened.append((max(row_min, new_lo), min(row_max, new_hi)))
        tiles = widened
    return tiles


def _make_state_spectrum(absco, wide_inst, geo, solar, albedo):
    """spectrum(params_dict[, surface_dict]) -> hi-res radiance, built
    straight from `als.atmosphere_from_params`. Deliberately NOT gert's
    StateVector.gas_scaling, which only knows how to scale gases and would
    reintroduce the CO2-is-special asymmetry.

    `albedo` (the module-level scalar from `_SWEEP["albedo"]`) is the
    fallback used whenever there is no free/frozen `surface`-target
    `albedo` row in the state at all -- every pre-2026-08-25 caller, and
    any current one that doesn't free "albedo". When `build_forward_state`
    DOES have a surface row (see its own docstring: "called instead as
    spectrum(params, surface)"), `surface["albedo"]` -- the state's own
    per-position value, not the fixed scalar -- is what actually reaches
    the forward model, matching `state_spec_from_scene`'s whole point in
    adding that row in the first place.

    `surface.get("tau_aerosol")`/`.get("height_aerosol")` (2026-09-08,
    `None` when neither row is present -- every existing caller, and any
    current one that doesn't free/freeze them) forward straight to `fm.
    run`'s own `tau_aerosol`/`height_aerosol` kwargs, which already treat
    `None` as "aerosol term omitted" -- zero behavior change by default,
    the same guarantee `t_offset_k=0.0`'s own default gave.
    """
    def spectrum(params: dict, surface: dict | None = None):
        atm = als.atmosphere_from_params(**params)
        fm = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(),
                          solar_spectrum=solar)
        px_albedo = surface["albedo"] if surface is not None else albedo
        px_tau_aer = surface.get("tau_aerosol") if surface is not None else None
        px_height_aer = surface.get("height_aerosol") if surface is not None else None
        n_wn = len(wide_inst.windows[0].wn_hires)
        # P_aerosol (2026-09-08): the Henyey-Greenstein phase function at
        # this geometry's own real scattering angle -- REQUIRED, not
        # optional, for I_scatter to be nonzero at all (see `als.
        # aerosol_phase_hg`'s own docstring on the bug this fixes).
        p_aer_val = als.aerosol_phase_hg(als.AEROSOL_G, np.cos(geo.scattering_angle))
        res = fm.run(albedo=np.array([px_albedo]), albedo_slope=np.zeros(1),
                     tau_aerosol=px_tau_aer, height_aerosol=px_height_aer,
                     aerosol_profile_shape="gaussian",
                     thickness_aerosol=als.AEROSOL_THICKNESS_PA,
                     ssa_aerosol=[np.full(n_wn, als.AEROSOL_SSA)],
                     g_aerosol=[als.AEROSOL_G],
                     qext_aerosol=[np.full(n_wn, als.AEROSOL_QEXT_NORM)],
                     P_aerosol=[np.full(n_wn, p_aer_val)])
        return np.asarray(res.I_hires[0], dtype=float)
    return spectrum


def _oracle_g_truth(eta, fpa):
    """`sub_bin_anomaly`'s exact-truth oracle `g` (2026-08-28,
    docs/PROJECT_STATUS.md Sec.10): the true fine-scale albedo texture.
    Module-level (not a lambda) so `functools.partial(_oracle_g_truth,
    fpa=fpa)` is picklable across `multiprocessing`'s task queue -- same
    reasoning as `gd_information_density_bins_demo.py`'s own `_oracle_g`,
    duplicated here rather than cross-imported since this is the
    production sweep, not the demo."""
    return als.albedo_for_label(eta * als.SLIT_HALF_KM, GEOCARB_BANDS[fpa][0], include_fine=True)


def _oracle_g_prior(eta, fpa):
    """`sub_bin_anomaly`'s "real, imperfect product" oracle `g`: an
    independent noise realization with the true field's own statistics
    but not its phase -- `als.albedo_for_label_prior_fine`."""
    return als.albedo_for_label_prior_fine(eta * als.SLIT_HALF_KM, GEOCARB_BANDS[fpa][0])


#: `--sub-bin-anomaly` choice -> the oracle function it wires into
#: `row_sub_bin_anomaly={"albedo": {"g_fn": ...}}`. "none" (default) means
#: no `sub_bin_anomaly` at all -- a byte-identical no-op, not an entry
#: pointing at a trivial g.
SUB_BIN_ANOMALY_ORACLES = {"truth": _oracle_g_truth, "prior": _oracle_g_prior}

#: Row name -> ParamSpec `kind` override, passed to every `state_spec_
#: from_scene` call (2026-09-07, the `t_offset_k` state row; extended
#: 2026-09-08 for `tau_aerosol`/`height_aerosol`). Every row not listed
#: here keeps `state_spec_from_scene`'s own default ("scale"), so this
#: dict is harmless for any run that doesn't free/freeze any of these
#: rows at all. All three use "absolute" -- the retrieved number IS the
#: physical value (Kelvin offset; AOD; Pa) rather than a multiplier on a
#: prior that could be zero -- only valid with `--jacobian analytic`
#: (`gauss_newton_state` raises otherwise; see its own docstring on why
#: kind="scale" is required for the finite-difference step).
ROW_KINDS = {"t_offset_k": "absolute", "tau_aerosol": "absolute",
            "height_aerosol": "absolute"}


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
    surface_fields = _SWEEP.get("surface_fields")        # an als.SURFACE_PRIOR_FIELD_SETS[...] dict, always explicit (never None)
    prior_anchor_density = _SWEEP.get("prior_anchor_density")  # None -> exact-per-bin (today's default)
    row_sub_bin_anomaly = _SWEEP.get("row_sub_bin_anomaly")  # None -> no sub_bin_anomaly at all (default)
    anchor_workers = _SWEEP.get("anchor_workers", 1)      # ANCHOR-level parallelism within build_forward_state's own forward() (2026-09-02); see --anchor-workers' own help
    surface_positions_mode = _SWEEP.get("surface_positions_mode", "shared")  # "shared" (default) or "anchor" -- see --surface-positions' own help
    frozen_atmosphere_positions_mode = _SWEEP.get("frozen_atmosphere_positions_mode", "shared")  # "shared" (default) or "anchor" -- see --frozen-atmosphere-positions' own help
    retrieval_psf_fwhm_px = _SWEEP.get("retrieval_psf_fwhm_px", 1.5)  # PSF the retrieval's OWN forward model assumes -- see --retrieval-psf-fwhm-px' own help
    retrieval_pad = default_pad_for_psf(retrieval_psf_fwhm_px)  # auto-scaled to avoid truncating a wider PSF's own kernel at window edges

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
    # `vary_albedo`, not just `"albedo" in free`, decides whether a surface
    # row exists at all (2026-08-29 fix): before this, freezing albedo
    # (leaving it out of --free) while --vary-albedo rendered a spatially-
    # varying TRUTH always meant band_label stayed None -- no surface row
    # got added to the StateSpec at all, so build_forward_state fell back
    # to _SWEEP["albedo"]'s single fixed scalar for EVERY anchor, silently
    # mismatching a truth scene that genuinely varies along the slit. That
    # was never exercised before -- every prior sweep either froze albedo
    # AND rendered it constant (`--free` without `--vary-albedo`) or freed
    # albedo AND varied it (`--free ...,albedo --vary-albedo`, enforced by
    # main()'s own validation) -- the "freeze albedo at the real per-
    # position truth" case this fix enables was simply unreachable. Now:
    # whenever `--vary-albedo` renders a real truth field, a surface row is
    # always added (frozen unless "albedo" is also in --free), so a frozen
    # row's own prior IS the true per-bin-center value (surface_fields
    # defaults to als.SURFACE_FIELDS, exact truth) -- "albedo held fixed at
    # truth" becomes an actual, correctly-wired configuration.
    vary_albedo = _SWEEP.get("vary_albedo", False)
    band_label = GEOCARB_BANDS[FPA][0] if ("albedo" in free or vary_albedo) else None
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
                                 wn_hires, ils, x, pad=PAD, state_interp=interp_kind,
                                 n_workers=anchor_workers)
        return lin

    # COARSE: scene evaluated at the state's own bin centres, so
    # interp_to() is the identity -- state_interp's kind has no content here.
    if not hires_only:
        t0 = time.time()
        spec_c = state_spec_from_scene(bin_centers, free=free, corr_length=corr_length,
                                       prior_form=prior_form, uniform=uniform_priors,
                                       fields=prior_fields, band_label=band_label,
                                       surface_fields=surface_fields,
                                       prior_anchor_density=prior_anchor_density,
                                       row_sub_bin_anomaly=row_sub_bin_anomaly,
                                       kinds=ROW_KINDS)
        # coarse: scene positions ARE the state positions, so interpolation is
        # the identity regardless of kind -- state_interp is genuinely a no-op here.
        fwd_c = build_forward_state(FPA, rows_win, bin_centers, spec_c, spectrum,
                                    wn_hires, ils, pad=retrieval_pad, state_interp="linear",
                                    n_workers=anchor_workers,
                                    spatial_psf_fwhm_px=retrieval_psf_fwhm_px)
        x_c, S_ret_c, avk_c = gauss_newton_state(fwd_c, y_true, spec_c, Sy_inv_diag,
                                                 label=f"[{row_lo}-{row_hi}] coarse", verbose=False,
                                                 jacobian_fn=_linearizer(spec_c, bin_centers, use_analytic, "linear"),
                                                 return_cov=True, return_avk=True)
        resid_c = y_true - fwd_c(x_c)
        # Standing rule: save the ENTIRE state vector (free AND frozen) and
        # the FULL residual field, never just summary scalars. `jacobian_used`
        # is the solve's OWN record, not the requested `--jacobian` flag --
        # see the note above on why hires can silently differ from coarse.
        # `cov` is the packed posterior covariance in SCALE units, over
        # exactly the free elements `slices`/`x` describe -- see
        # StateSpec.cov_for/project_cov to get one row's own physical-units
        # block (or a projection onto arbitrary rows). `avk` is the Rodgers
        # averaging kernel over the same packed order (see
        # gauss_newton_state's own docstring); `dof` = trace(avk), the
        # degrees-of-freedom-for-signal for this window's whole solve --
        # docs/PROJECT_STATUS.md Sec.7.10/7.3's DOF-collapse hypothesis.
        out["coarse"] = spec_c.snapshot(x_c, resid=resid_c,
                                        resid_rms=float(np.sqrt(np.mean(resid_c ** 2))),
                                        jacobian_used=("analytic" if use_analytic else "fd"),
                                        cov=S_ret_c, avk=avk_c, dof=float(np.trace(avk_c)))
        out["x_coarse"] = x_c                      # back-compat with existing plotters
        out["resid_coarse"] = resid_c
        out["resid_coarse_rms"] = float(np.sqrt(np.mean(resid_c ** 2)))
        out["t_coarse"] = time.time() - t0

    anchor_ext = max(PAD, retrieval_pad)
    a_lo, a_hi = max(0, row_lo - anchor_ext), min(ROW_MAX_IDX, row_hi + anchor_ext)
    anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / anchor_density)
    anchor_etas = np.sort(_eta_of(FPA, np.full(len(anchor_rows), 512.0),
                                  anchor_rows.astype(float)))
    t0 = time.time()
    # HI-RES: scene on the finer anchor grid, every row interpolated there.
    surface_positions = anchor_etas if surface_positions_mode == "anchor" else None
    row_positions = None
    if frozen_atmosphere_positions_mode == "anchor":
        row_positions = {name: anchor_etas for name in prior_fields if name not in free}
    spec_h = state_spec_from_scene(bin_centers, free=free, corr_length=corr_length,
                                   prior_form=prior_form, uniform=uniform_priors,
                                   fields=prior_fields, band_label=band_label,
                                   surface_fields=surface_fields,
                                   surface_positions=surface_positions,
                                   row_positions=row_positions,
                                   prior_anchor_density=prior_anchor_density,
                                   row_sub_bin_anomaly=row_sub_bin_anomaly,
                                   kinds=ROW_KINDS)
    fwd_h = build_forward_state(FPA, rows_win, anchor_etas, spec_h, spectrum,
                                wn_hires, ils, pad=retrieval_pad, state_interp=state_interp,
                                n_workers=anchor_workers,
                                spatial_psf_fwhm_px=retrieval_psf_fwhm_px)
    x_h, S_ret_h, avk_h = gauss_newton_state(fwd_h, y_true, spec_h, Sy_inv_diag,
                                             label=f"[{row_lo}-{row_hi}] hires", verbose=False,
                                             jacobian_fn=_linearizer(spec_h, anchor_etas, use_analytic_hires,
                                                                     state_interp),
                                             return_cov=True, return_avk=True)
    resid_h = y_true - fwd_h(x_h)
    out["hires"] = spec_h.snapshot(x_h, resid=resid_h,
                                   resid_rms=float(np.sqrt(np.mean(resid_h ** 2))),
                                   jacobian_used=("analytic" if use_analytic_hires else "fd"),
                                   cov=S_ret_h, avk=avk_h, dof=float(np.trace(avk_h)))
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
    ap.add_argument("--n-workers", type=int, default=None,
                    help="WINDOW-level parallelism: an in-process Pool distributing the "
                         "58 (or --n-windows) independent windows across this many local "
                         "cores. Default: every available CPU.")
    ap.add_argument("--anchor-workers", type=int, default=1,
                    help="ANCHOR-level parallelism WITHIN a single window's own forward-"
                         "model calls (build_forward_state's own n_workers, 2026-09-02) -- "
                         "a real, separate cost driver from window-level parallelism above: "
                         "a wide window at a fine --anchor-density can have hundreds of "
                         "anchors, each a fresh RT call, run single-process by default. "
                         "Default 1 (unchanged): a window-level Pool worker (--n-workers>1, "
                         "in-process) CANNOT itself spawn a nested pool, so this is silently "
                         "forced back to 1 whenever this process is itself such a worker, "
                         "regardless of what's passed here -- set this >1 only when windows "
                         "are distributed as separate top-level processes instead "
                         "(--task-id/--n-tasks, one SLURM job-array task per subset of "
                         "windows), where the two levels genuinely compose rather than nest.")
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
    ap.add_argument("--overlap", type=int, default=0,
                    help="rows of symmetric overlap added to each internal window "
                         "boundary (default 0, today's exact non-overlapping tiling, "
                         "unchanged). >0 lets geocarb_gert.along_slit_query.query_state "
                         "blend two independent windows' own estimates in their shared "
                         "rows instead of using either window's boundary value "
                         "un-blended -- see docs/PROJECT_STATUS.md Sec.6's window-"
                         "boundary overshoot/undershoot finding. Keep small relative to the "
                         "narrowest window (2*min_window+1 rows) -- see "
                         "build_window_tiles's own docstring.")
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
    ap.add_argument("--vary-albedo", action="store_true",
                    help="render the TRUTH scene with real along-slit surface heterogeneity "
                         "(als.SURFACE_PATCHES' 4 land-cover archetypes + fine-scale texture, "
                         "als.albedo_at) instead of a single constant scalar albedo per band. "
                         "REQUIRED whenever 'albedo' is in --free (enforced below) -- fitting "
                         "a free albedo row against a constant-albedo truth was exactly the "
                         "2026-08-18 CAVEAT this flag closes (see along_slit_scene.py's "
                         "SURFACE_FIELDS docstring). Safe to pass without freeing albedo too "
                         "(tests whether OTHER rows are robust to unmodeled surface "
                         "heterogeneity) -- only the reverse (free, not rendered) is blocked.")
    ap.add_argument("--surface-positions", type=str, default="shared",
                    choices=["shared", "anchor"],
                    help="positions the surface (albedo) row lives at, independent of "
                         "atmosphere-row bin_centers. 'shared' (default): albedo shares the "
                         "same bin_centers grid every atmosphere row uses (today's mechanism, "
                         "state_spec_from_scene's own default). 'anchor': albedo instead gets "
                         "one value per ANCHOR (this window's own anchor_etas, the same fine "
                         "grid the forward model already renders from) -- for a row frozen at "
                         "the exact truth, this removes the piecewise-linear-between-bin-"
                         "centers representability gap for albedo specifically (2026-09-05, "
                         "user: give a frozen row the true value at every anchor, not just at "
                         "bin centers, to isolate whether albedo's own sub-bin texture is what "
                         "drove Sec.16.2's frozen-row blowup under Mode-1 dense truth). Only "
                         "affects the hires solve (state_spec_from_scene's surface_positions "
                         "parameter) -- coarse mode is unaffected. Meaningless without "
                         "--vary-albedo (no surface row exists at all otherwise).")
    ap.add_argument("--frozen-atmosphere-positions", type=str, default="shared",
                    choices=["shared", "anchor"],
                    help="positions FROZEN atmosphere rows (any row named in --free is "
                         "unaffected and stays on bin_centers) live at. 'shared' (default): "
                         "every atmosphere row shares the same bin_centers grid (today's "
                         "mechanism). 'anchor': frozen atmosphere rows instead get one value "
                         "per ANCHOR, the atmosphere-row analogue of --surface-positions "
                         "anchor (2026-09-05) -- lets a frozen row's exact-truth value be "
                         "reconstructed at the same fine resolution the forward model already "
                         "renders from, instead of piecewise-linear between (coarser) bin "
                         "centers. Only affects the hires solve; coarse mode is unaffected.")
    ap.add_argument("--retrieval-psf-fwhm-px", type=float, default=1.5,
                    help="along-slit (N/S) PSF FWHM [detector pixels] the RETRIEVAL's own "
                         "forward model assumes (2026-09-06, user: defocus experiments -- "
                         "the telescope has a focal-adjustment mechanism that can shift focus "
                         "without changing spectral resolution, purely spatial blurring). "
                         "Default 1.5 matches the real ground-test-measured nominal value and "
                         "every prior run's behavior exactly. Independent of whatever PSF the "
                         "TRUTH image was actually rendered with (see scratch_work's "
                         "defocus truth-build driver, and --prior-fields' own injected-truth "
                         "mechanism) -- passing a value here that DIFFERS from the truth's own "
                         "PSF simulates an uncorrected/uncalibrated defocus event; passing the "
                         "SAME value simulates a known, modeled one. Affects both the coarse "
                         "and hires forward models (build_forward_state); pad auto-scales via "
                         "geocarb_gert.joint_state.default_pad_for_psf to avoid truncating a "
                         "wider PSF's own Gaussian kernel at window edges.")
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
    ap.add_argument("--sub-bin-anomaly", type=str, default="none",
                    choices=["none", "truth", "prior"],
                    help="2026-08-28 (docs/PROJECT_STATUS.md Sec.10): set albedo's "
                         "ParamSpec.sub_bin_anomaly to an oracle g, riding ADDITIVELY on top "
                         "of albedo's own retrieved bin values (reduces exactly to plain "
                         "linear interpolation wherever g is itself linear between bin "
                         "centers). 'none' (default): no sub_bin_anomaly at all, a byte-"
                         "identical no-op. 'truth': als.albedo_for_label(..., include_fine="
                         "True), the EXACT true fine-scale texture. 'prior': als.albedo_for_"
                         "label_prior_fine, an INDEPENDENT noise realization with the same "
                         "statistics (ALBEDO_COV, ALBEDO_CORR_KM) but not the true phase -- "
                         "what a real, imperfect external product (MODIS-like) would supply. "
                         "Requires 'albedo' in --free (same reasoning as --vary-albedo).")
    ap.add_argument("--resolution-matched-anchor-density", type=int, default=None,
                    help="2026-08-29 (docs/PROJECT_STATUS.md, resolution-matched truth "
                         "images): render the TRUTH scene from geocarb_gert.along_slit_"
                         "scene.resolution_matched_fields/resolution_matched_albedo_fn "
                         "instead of the raw continuous truth -- band-limited to EXACTLY "
                         "the whole-slit anchor grid this anchor_density would use (built "
                         "by scripts/gd_build_resolution_matched_truth.py, must already be "
                         "cached -- this flag reconstructs the SAME anchor grid/resolution_"
                         "tag to hit that cache, not re-render). Removes the 'genuinely-"
                         "unresolvable sub-anchor truth structure' confound: any remaining "
                         "retrieval error against THIS truth is real model/retrieval error. "
                         "Implies --vary-albedo and n_lookup_samples=20000 (the value the "
                         "images were built at -- overrides --n-lookup-samples if both are "
                         "given). A mismatched --anchor-density (the RETRIEVAL's own anchor "
                         "grid) is allowed but warned about -- matching it is the direct "
                         "apples-to-apples test; a deliberate mismatch tests under/over-"
                         "sampling a truth of known resolution.")
    ap.add_argument("--resolution-matched-g-ratio-bins", type=float, default=None,
                    help="2026-08-29 (user): like --resolution-matched-anchor-density, but "
                         "the truth is band-limited to the retrieval's own STATE BIN grid "
                         "(whole_slit_bin_centers(fpa, this g_ratio), gd_build_resolution_"
                         "matched_truth.py) instead of the anchor grid -- removes the bin-"
                         "grid-vs-anchor-grid representability gap entirely: the retrieval's "
                         "bin-to-bin piecewise-linear reconstruction becomes bit-for-bit the "
                         "SAME function the truth was built from, everywhere, not just at "
                         "bin centers. A genuine ceiling test, not just 'as fine as the "
                         "forward model can sample.' Must already be cached by gd_build_"
                         "resolution_matched_truth.py --g-ratio-bins. Pass a fine --anchor-"
                         "density (e.g. 16) so the forward model can still resolve this "
                         "truth without an anchor-level stepping-bias confound -- this g_"
                         "ratio should normally equal --g-ratio (the RETRIEVAL's own bin "
                         "grid), or the truth and the retrieval are matched to two DIFFERENT "
                         "bin grids, defeating the point. Mutually exclusive with "
                         "--resolution-matched-anchor-density.")
    args = ap.parse_args()
    if (args.resolution_matched_anchor_density is not None
            and args.resolution_matched_g_ratio_bins is not None):
        ap.error("--resolution-matched-anchor-density and --resolution-matched-g-ratio-bins "
                 "are mutually exclusive -- pick one truth-matching mode.")
    if (args.resolution_matched_g_ratio_bins is not None
            and args.resolution_matched_g_ratio_bins != args.g_ratio):
        print(f"WARNING: --resolution-matched-g-ratio-bins "
             f"{args.resolution_matched_g_ratio_bins} != --g-ratio {args.g_ratio} -- the "
             f"truth and the retrieval are matched to DIFFERENT bin grids, which defeats "
             f"the point of this mode (a ceiling test needs them identical). Allowed, but "
             f"almost certainly not what you want unless deliberate.", flush=True)
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
    free_check = tuple(x.strip() for x in args.free.split(","))
    if "albedo" in free_check and not args.vary_albedo:
        ap.error("'albedo' is in --free but --vary-albedo was not passed -- this fits a free "
                 "albedo row against a truth scene rendered with a single CONSTANT albedo, "
                 "exactly the 2026-08-18 CAVEAT documented in along_slit_scene.py's "
                 "SURFACE_FIELDS. Add --vary-albedo.")
    if args.sub_bin_anomaly != "none" and "albedo" not in free_check:
        ap.error("--sub-bin-anomaly only has an effect on a free 'albedo' row -- add "
                 "'albedo' to --free, or drop --sub-bin-anomaly.")
    if args.resolution_matched_anchor_density is not None and args.anchor_density != args.resolution_matched_anchor_density:
        print(f"WARNING: --resolution-matched-anchor-density "
             f"{args.resolution_matched_anchor_density} != --anchor-density "
             f"{args.anchor_density} -- the retrieval's own anchor grid will NOT match "
             f"the truth's own band-limiting grid. This is allowed (tests under/over-"
             f"sampling a truth of known resolution) but is not the direct apples-to-"
             f"apples config unless deliberate.", flush=True)
    fpa_list = tuple(int(x.strip()) for x in args.fpa.split(","))
    if len(fpa_list) != 1:
        ap.error(f"--fpa currently supports exactly one band (got {args.fpa!r}); multi-band "
                 "joint retrieval is a later phase of the config-consolidation plan, not yet "
                 "implemented in this script")
    fpa = fpa_list[0]
    resolution_matched_active = (args.resolution_matched_anchor_density is not None
                                 or args.resolution_matched_g_ratio_bins is not None)
    n_lookup_samples = (20000 if resolution_matched_active else
                        args.n_lookup_samples if args.n_lookup_samples is not None
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
    # Computed here (moved ahead of the resolution-matched truth-building
    # block below, 2026-09-01 bug fix) so whole_slit_anchor_etas/whole_
    # slit_bin_centers can be handed the SAME tiling (window_scale/overlap/
    # min_window) this run will actually solve against -- previously they
    # silently used build_window_tiles's own defaults (window_scale=1.0,
    # overlap=0) regardless of --n-windows/--overlap/--min-window, which
    # shifts every window's own row_lo/row_hi under a non-default overlap
    # and drifts the density-weighted anchor/bin placement away from what
    # _solve_window computes, concentrated at window boundaries -- see
    # docs/PROJECT_STATUS.md Sec.12.8.
    if args.n_windows is not None:
        window_scale = scale_for_window_count(fpa, args.n_windows, args.min_window)
        print(f"--n-windows {args.n_windows} -> window_scale {window_scale:.2f}", flush=True)
    else:
        window_scale = args.window_scale
    if resolution_matched_active:
        # Reconstruct the EXACT grid/resolution_tag gd_build_resolution_matched_
        # truth.py used to build+cache this image -- must match bit-for-bit or
        # this call is a cache MISS (a silent, expensive re-render) rather than
        # hitting the already-built image. Two modes: anchor-grid-matched
        # (whole_slit_anchor_etas) or bin-grid-matched (whole_slit_bin_centers,
        # 2026-08-29 -- removes the representability gap entirely, see that
        # flag's own help text).
        from gd_build_resolution_matched_truth import whole_slit_anchor_etas, whole_slit_bin_centers
        rm_band_label = GEOCARB_BANDS[fpa][0]
        if args.resolution_matched_g_ratio_bins is not None:
            rm_g = args.resolution_matched_g_ratio_bins
            match_etas_rm = whole_slit_bin_centers(fpa, rm_g, min_window=args.min_window,
                                                    window_scale=window_scale, overlap=args.overlap)
            rm_tag_prefix = f"gr{rm_g:g}bins"
            rm_desc = f"g_ratio_bins={rm_g}"
        else:
            rm_ad = args.resolution_matched_anchor_density
            match_etas_rm = whole_slit_anchor_etas(fpa, rm_ad, min_window=args.min_window,
                                                    window_scale=window_scale, overlap=args.overlap)
            rm_tag_prefix = f"ad{rm_ad}"
            rm_desc = f"anchor_density={rm_ad}"
        match_x_km_rm = match_etas_rm * als.SLIT_HALF_KM
        rm_fields = als.resolution_matched_fields(match_x_km_rm)
        rm_surface_fields = {rm_band_label: als.resolution_matched_albedo_fn(match_x_km_rm, rm_band_label)}
        rm_hash = hashlib.sha256(match_x_km_rm.tobytes()).hexdigest()[:16]
        rm_tag = f"{rm_tag_prefix}-{rm_hash}"
        print(f"resolution-matched truth: {rm_desc}, "
             f"{len(match_x_km_rm)} points, resolution_tag={rm_tag}", flush=True)
        band = gdt._band_setup_cached(fpa, atm_center, absco, geo, solar, snr, n_lookup_samples, None,
                                      args.uniform, args.barcode, args.barcode_bars, False, 0,
                                      args.realistic_barcode, vary_albedo=True,
                                      use_cache=not args.no_truth_cache,
                                      fields=rm_fields, surface_fields=rm_surface_fields,
                                      resolution_tag=rm_tag)
    else:
        band = gdt._band_setup_cached(fpa, atm_center, absco, geo, solar, snr, n_lookup_samples, None,
                                      args.uniform, args.barcode, args.barcode_bars, False, 0,
                                      args.realistic_barcode, vary_albedo=args.vary_albedo,
                                      use_cache=not args.no_truth_cache)
    wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    all_tiles = build_window_tiles(fpa, min_window=args.min_window,
                                   window_scale=window_scale, overlap=args.overlap)
    widths = [hi - lo + 1 for lo, hi in all_tiles]
    print(f"{len(all_tiles)} windows total, widths min={min(widths)} max={max(widths)} "
         f"mean={np.mean(widths):.1f}, total rows={sum(widths)}"
         + (f", overlap={args.overlap}" if args.overlap else ""), flush=True)
    if args.task_id is not None:
        tiles = all_tiles[args.task_id::args.n_tasks]
        print(f"task {args.task_id}/{args.n_tasks}: {len(tiles)} windows assigned "
             f"(round-robin, tiles[{args.task_id}::{args.n_tasks}])", flush=True)
    else:
        tiles = all_tiles

    row_sub_bin_anomaly = None
    if args.sub_bin_anomaly != "none":
        oracle = SUB_BIN_ANOMALY_ORACLES[args.sub_bin_anomaly]
        row_sub_bin_anomaly = {"albedo": {"g_fn": functools.partial(oracle, fpa=fpa)}}

    # Retrieval PRIOR field-set. Under --resolution-matched-anchor-density,
    # "exact" (prior == truth) must mean exactly the RESOLUTION-MATCHED
    # truth the image was actually rendered from -- not the raw continuous
    # STATE_FIELDS/SURFACE_FIELDS -- otherwise a FROZEN row (whose value IS
    # its prior, verbatim) would silently disagree with what generated
    # y_true, a real, avoidable truth/prior mismatch for exactly the rows
    # meant to be held fixed at the true value. 'structural' (or any other
    # non-exact choice) is a DELIBERATE imperfect prior and is untouched by
    # this -- resolution-matching the truth doesn't change what "imperfect"
    # means for it.
    if resolution_matched_active and args.prior_fields == "exact":
        prior_fields_resolved = rm_fields
        # state_spec_from_scene's own surface_fields convention differs from
        # build_lookup_radiance's: keyed by ROW NAME ("albedo"), called as
        # fn(x_km, band_label) -- two positional args (see state_spec_from_
        # scene's _row/_eval closure) -- vs rm_surface_fields above, keyed
        # by band LABEL, fn(x_km) only one arg. Wrap rather than reuse
        # directly (found via a smoke test: reusing rm_surface_fields as-is
        # raised "takes 1 positional argument but 2 were given").
        _rm_albedo_fn = rm_surface_fields[rm_band_label]
        surface_fields_resolved = {"albedo": lambda x_km, label, _fn=_rm_albedo_fn: _fn(x_km)}
    else:
        prior_fields_resolved = als.PRIOR_FIELD_SETS[args.prior_fields]
        surface_fields_resolved = als.SURFACE_PRIOR_FIELD_SETS.get(
            args.prior_fields, als.SURFACE_FIELDS)

    _SWEEP.update(dict(band=band, absco=absco, wide_inst=wide_inst, geo=geo, solar=solar,
                       albedo=albedo, wn_hires=band["wn_hires"], ils=band["ils"], fpa=fpa,
                       gamma=args.gamma, sigma_abs=args.sigma_abs, g_ratio=args.g_ratio,
                       uniform=args.uniform, uniform_priors=uniform_priors, atm_center=atm_center,
                       hires_only=args.hires_only, anchor_density=args.anchor_density,
                       state_interp=args.state_interp,
                       free=tuple(x.strip() for x in args.free.split(',')),
                       corr_length=args.corr_length, prior_form=args.prior_form,
                       jacobian=args.jacobian,
                       prior_fields=prior_fields_resolved,
                       surface_fields=surface_fields_resolved,
                       prior_anchor_density=args.prior_anchor_density,
                       flat_sy_inv=args.flat_sy_inv, vary_albedo=args.vary_albedo,
                       row_sub_bin_anomaly=row_sub_bin_anomaly,
                       surface_positions_mode=args.surface_positions,
                       frozen_atmosphere_positions_mode=args.frozen_atmosphere_positions,
                       anchor_workers=args.anchor_workers,
                       retrieval_psf_fwhm_px=args.retrieval_psf_fwhm_px))

    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    if args.anchor_workers > 1 and n_workers > 1 and args.task_id is None:
        print(f"  NOTE: --anchor-workers {args.anchor_workers} has no effect here -- "
             f"windows are being distributed via an in-process Pool (--n-workers "
             f"{n_workers}), whose daemon workers cannot themselves spawn a nested pool "
             f"(build_forward_state's own daemon check forces anchor-level parallelism "
             f"back to 1 there). Use --task-id/--n-tasks instead to distribute windows as "
             f"separate top-level processes if you want both levels active at once.",
             flush=True)
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
    if args.overlap:
        suffix += f"_ovlp{args.overlap}"
    if args.vary_albedo:
        suffix += "_valb"
    if args.sub_bin_anomaly != "none":
        suffix += f"_anomaly-{args.sub_bin_anomaly}"
    if args.retrieval_psf_fwhm_px != 1.5:
        # 2026-09-06 (defocus experiments): without this, two configs that
        # differ ONLY in --retrieval-psf-fwhm-px (e.g. matched vs mismatched
        # defocus against the SAME injected truth/prior_fields name) would
        # silently collide in the SAME _parts directory -- caught the hard
        # way once already this session (see docs/PROJECT_STATUS.md's
        # standing naming-collision note).
        suffix += f"_retrpsf{args.retrieval_psf_fwhm_px:g}"
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
              "flat_sy_inv": args.flat_sy_inv, "overlap": args.overlap,
              "vary_albedo": args.vary_albedo, "sub_bin_anomaly": args.sub_bin_anomaly,
              "anchor_workers": args.anchor_workers,
              "retrieval_psf_fwhm_px": args.retrieval_psf_fwhm_px}
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
