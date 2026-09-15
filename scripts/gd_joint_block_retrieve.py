#!/usr/bin/env python3
"""Joint multi-atmosphere block retrieval: solve jointly for `G`
atmospheres on a grid of eta bins, rather than one atmosphere per
detector row (the per-row approach's own keystone dilution -- see
`docs/JOINT_BIN_RETRIEVAL_ATBD.html`). Defaults to tiling the ENTIRE
slit into non-overlapping windows sized off local keystone amplitude;
`--row-min`/`--row-max` restricts tiling to a sub-range instead (e.g.
`--row-min 890 --row-max 935` reproduces the original single-window
hot-spot case -- formerly a separate script, `gd_joint_block_retrieve.py`
under the OLD, simpler `StateVector.gas_scaling()`-based mechanism,
retired 2026-09-09 once this flag subsumed its role and every other
importer's `FPA`/`GERT_ROOT`/`_eta_of`/`band_basics` moved here).

Window radius = max(MIN_WINDOW, round(2.2 * rows_crossed(fpa, center))),
the same formula gd_toy_trace_retrieve.py already validated for sizing a
single target's own neighborhood. Windows are non-overlapping by
construction (each window starts exactly where the previous one ended),
not a fixed stride -- a fixed stride would either overlap in high-keystone
regions or leave gaps in low-keystone ones, since window width varies by
more than 5x across the slit. Non-overlapping tiling also means every row
belongs to exactly one window's own independent solve, so there is never
an ambiguous choice between two overlapping windows' estimates at a given
row.

Within each window, G = max(2, round(width/3)) bins are placed by pixel
density by default (`--bin-scheme`; `geocarb_gert.joint_state.pixel_
density_bin_centers`), and BOTH the coarse (nearest-bin) and hi-res
(state interpolated to native row resolution) forward models are solved
independently -- same regularization, same gamma, same local-truth
nuisance-gas idealization every window shares. Because the forward
model's own Jacobian and the regularization's own Laplacian are both
local/banded (a bin's own influence never reaches past the PSF's few-row
footprint or past its immediate neighbors), solving each window
independently is not an approximation to one giant whole-slit joint
solve -- it gives essentially the same answer at a fraction of the cost,
as long as the padding used here (PAD rows on each side, feeding
predict_neighborhood) correctly covers the PSF's own reach, which it
does by the same margin already validated for the single hot-spot
window.

`--merge`/`--plot` are two independent post-processing modes, folded in
2026-09-09 from the formerly-separate `gd_joint_block_whole_slit_merge.py`/
`_plot.py`: `--merge <parts_dir>` combines a SLURM-array run's per-task
output files into one pickle; `--plot <pkl>` compares a (merged or
single-process) run's output against truth. See `merge_parts`/
`plot_sweep`'s own docstrings.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_retrieve.py \\
        [--row-min 890 --row-max 935] [--n-workers N] [--gamma 3.0] [--g-ratio 3]
      PYTHONPATH=. ... scripts/gd_joint_block_retrieve.py --merge results/..._parts
      PYTHONPATH=. ... scripts/gd_joint_block_retrieve.py --plot results/...pkl
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_per_row_retrieve as gpr  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import GEOCARB_BANDS, albedo_for, along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert import gert_root  # noqa: E402
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.joint_state import (build_forward_state, gauss_newton_state,  # noqa: E402
                                      pixel_density_bin_centers, state_spec_from_scene,
                                      default_pad_for_psf)
from geocarb_gert.spectrum import simulate_spectrum  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed  # noqa: E402
from geocarb_gert.gd_render import available_cpus, s_max  # noqa: E402
from geocarb_gert import jacobians as jac  # noqa: E402
from geocarb_gert.mission_config import RetrievalDefaults  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402

from gert.instrument import ILS, SpectralWindow  # noqa: E402
from gert.instrument_config import Instrument  # noqa: E402

# Where gert's input/ tree (absco.h5, solar.h5) lives. Defaults to the HPC
# scratch path every batch script and prior run used, so those are unchanged;
# override with $GERT_ROOT to run the same sweeps on a workstation (needed
# 2026-08-17, when the cluster was unavailable). Formerly defined in the
# standalone gd_joint_block_retrieve.py demo script (retired 2026-09-09 once
# --row-min/--row-max subsumed its single-window role -- see docs/
# PROJECT_STATUS.md Sec.13); moved here since this is now THE joint-block
# retrieval script every other one imports FPA/GERT_ROOT/_eta_of/band_basics
# from, unchanged by that rename (same names, same import statements
# elsewhere -- only what file they resolve to changed).
GERT_ROOT = gert_root()   # geocarb_gert.paths -- shared by every driver
# Sourced from input/retrieval_defaults.yml's band.default_fpa[0] at import
# time (Phase C of the config-consolidation plan) -- was a bare literal `2`.
# This module-level constant is still what every other script importing
# `FPA` from here gets; it is NOT itself multi-band-aware (still exactly one
# int) and does not reflect any --config/--fpa CLI override a caller may
# have passed -- THIS script's own main() adds a real --fpa flag and
# resolves its OWN active band from that at runtime, independent of this
# constant. See geocarb_gert/mission_config.py.
FPA = RetrievalDefaults.from_yaml().default_fpa[0]


def _eta_of(fpa, cols, rows):
    _, s = xy_to_wavelength_slit(fpa, cols, rows)
    return s / s_max(fpa)


def band_basics(fpa, atm_center, absco, geo, solar):
    """Cheap wide_win/wide_inst reconstruction -- matches `_band_setup`'s
    own recipe exactly (same molecules/wn range/fwhm/label), but rebuilt
    here since `_band_setup`'s returned `band` dict exposes `wn_hires`/
    `ils`/`albedo` but not the `Instrument` object itself."""
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[fpa]
    mols = list(mols)
    wn_min, wn_max = real_wavenumber_range(fpa, margin_cm1=10.0)
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=mols, label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=1e9)
    albedo = float(albedo_for(wide_inst, "desert")[0])
    return wide_win, wide_inst, albedo

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


def _make_state_spectrum(absco, wide_inst, geo, solar, albedo, solver: str = "single_scatter"):
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

    `surface.get("amplitude_aerosol")`/`.get("height_aerosol")`/
    `.get("thickness_aerosol")` (2026-09-08, updated 2026-09-15 when
    `tau_aerosol` was retired as a directly-retrievable row in favor of
    the Gaussian shape parameters -- see `along_slit_scene.amplitude_
    aerosol`'s own docstring) -- `None` when no aerosol rows are present
    (every existing caller, and any current one that doesn't free/freeze
    them) forward through `simulate_spectrum`'s own `_build_aerosol_kwargs`,
    which converts amplitude/thickness to the column `tau_aerosol` `fm.run`
    still wants and treats `None` as "aerosol term omitted" -- zero
    behavior change by default, the same guarantee `t_offset_k=0.0`'s own
    default gave.
    """
    def spectrum(params: dict, surface: dict | None = None):
        # 2026-09-09 consolidation (geocarb_gert.spectrum) -- was its own
        # independent inline ForwardModel/fm.run duplicate; see that
        # module's docstring for why that pattern was a real liability.
        px_albedo = surface["albedo"] if surface is not None else albedo
        sfc = dict(surface or {})
        sfc["albedo"] = px_albedo
        return simulate_spectrum(params, sfc, absco, wide_inst, geo, solar,
                                 solver=solver).I_hires
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
ROW_KINDS = {"t_offset_k": "absolute", "height_aerosol": "absolute",
            "amplitude_aerosol": "absolute", "thickness_aerosol": "absolute"}


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
    gn_verbose = _SWEEP.get("gn_verbose", False)
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
    bin_scheme = _SWEEP.get("bin_scheme", "pixel-density")
    if bin_scheme == "uniform":
        # The original (pre-pixel-density) placement -- np.linspace over the
        # window's own eta range, no pixel-density weighting. Kept for the
        # diagnostics.py-style uniform-vs-pixel-density comparison; not the
        # production default.
        bin_centers = np.linspace(float(eta_all.min()), float(eta_all.max()), G)
    else:
        bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    # kept only for gd_joint_block_whole_slit_plot.py's own backward-compat
    # reconstruction (prior_co2_ppm_bins * x_coarse); the real solve below
    # gets its priors from state_spec_from_scene(..., uniform=uniform_priors)
    prior_co2_ppm_bins = (np.full(G, float(als.xco2_ppm(0.0))) if uniform_priors
                          else np.array([float(als.xco2_ppm(xk))
                                        for xk in bin_centers * als.SLIT_HALF_KM]))

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
    with_aerosol = _SWEEP.get("with_aerosol", False)
    # `with_aerosol` also forces a surface row to exist (2026-09-09): the
    # frozen tau_aerosol/height_aerosol rows must be in the StateSpec for
    # build_forward_state to thread them into the forward model, or the
    # retrieval would omit the aerosol the truth render now includes --
    # the exact mismatch Sec.14 traced. surface_fields (from _SWEEP) still
    # carries the aerosol fns in this case; it is stripped of them when
    # --aerosol is off.
    band_label = (GEOCARB_BANDS[FPA][0]
                  if ("albedo" in free or vary_albedo or with_aerosol) else None)
    corr_length = _SWEEP.get("corr_length")          # None -> per-row physical defaults
    prior_form = _SWEEP.get("prior_form", "exponential")
    rt_solver = _SWEEP.get("solver", "single_scatter")
    spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo, solver=rt_solver)

    anchor_ext = max(PAD, retrieval_pad)
    a_lo, a_hi = max(0, row_lo - anchor_ext), min(ROW_MAX_IDX, row_hi + anchor_ext)
    anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / anchor_density)
    anchor_etas = np.sort(_eta_of(FPA, np.full(len(anchor_rows), 512.0),
                                  anchor_rows.astype(float)))

    # 2026-09-13 (user: "I would like to modify the truth generation
    # pipeline so that we first specify a truth grid and then generate the
    # observation image through the retrieval pipeline. That will allow us
    # to match exactly" -- "The retrieval forward model" -- "Please also
    # make that method of generation of truth the default"): `band["A"]`
    # (the OLD default) is rendered by `_band_setup`'s OWN independent
    # anchor grid (a fixed dx_km=0.5, or --resolution-matched-*'s own
    # n_lookup_samples-driven one) -- a genuinely DIFFERENT discretization
    # of `render_at_anchors` than the retrieval's own `build_forward_state`
    # uses, confirmed directly (docs/PROJECT_STATUS.md's resolution-matched
    # diagnostic): even with field VALUES matched exactly, residual never
    # quite reached zero, shrinking as the two grids converged but never
    # closing, because they were never actually the SAME grid. Building
    # truth by calling `build_forward_state` itself -- the retrieval's own
    # forward model -- at a fully-frozen (`free=()`) StateSpec whose rows
    # sit on `anchor_etas` (this window's own anchor grid, so `interp_to`
    # is the identity there, not a band-limiting re-interpolation) removes
    # that gap entirely: `y_true` becomes bit-for-bit what `forward(x)`
    # would compute at `x` = the true value, since it IS that computation.
    # `--legacy-truth` reverts to the old `band["A"]` slice for direct
    # comparison against every pre-2026-09-13 result, or for scripts that
    # rely on the whole-image truth cache (`band["A"]` still gets built
    # either way -- this only changes which array `y_true` reads from).
    if _SWEEP.get("legacy_truth", False):
        y_true = band["A"][rows_win, :].ravel()
    else:
        truth_surface_fields = als.SURFACE_FIELDS if with_aerosol else {
            k: v for k, v in als.SURFACE_FIELDS.items()
            if k not in ("amplitude_aerosol", "height_aerosol", "thickness_aerosol")}
        truth_spec = state_spec_from_scene(anchor_etas, free=(), fields=als.STATE_FIELDS,
                                           band_label=band_label,
                                           surface_fields=truth_surface_fields,
                                           surface_positions=anchor_etas,
                                           kinds=ROW_KINDS)
        fwd_truth = build_forward_state(FPA, rows_win, anchor_etas, truth_spec, spectrum,
                                        wn_hires, ils, pad=retrieval_pad, state_interp=state_interp,
                                        n_workers=anchor_workers,
                                        spatial_psf_fwhm_px=retrieval_psf_fwhm_px)
        y_true = fwd_truth(np.array([]))
    # Real per-pixel noise (Phase D of the config-consolidation plan),
    # replacing the old flat-scalar Sy_inv_diag = 1/mean(|signal|)^2 for
    # the whole window. geocarb_noise_model(FPA) is the same LinearShotNoise
    # gd_per_row_retrieve.py::_band_setup already computes (and previously discarded
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
    spectrum_jac = (jac.make_spectrum_jac(absco, wide_inst, geo, solar, albedo, solver=rt_solver)
                    if use_analytic else None)

    def _linearizer(spec_, scene_etas_, enabled, interp_kind, pool=None):
        if not enabled:
            return None
        def lin(x):
            return jac.linearize(FPA, rows_win, scene_etas_, spec_, spectrum_jac,
                                 wn_hires, ils, x, pad=PAD, state_interp=interp_kind,
                                 n_workers=anchor_workers, pool=pool)
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
                                       kinds=ROW_KINDS, gamma=gamma)
        # coarse: scene positions ARE the state positions, so interpolation is
        # the identity regardless of kind -- state_interp is genuinely a no-op here.
        fwd_c = build_forward_state(FPA, rows_win, bin_centers, spec_c, spectrum,
                                    wn_hires, ils, pad=retrieval_pad, state_interp="linear",
                                    n_workers=anchor_workers,
                                    spatial_psf_fwhm_px=retrieval_psf_fwhm_px)
        # 2026-09-12 (user: "let's make that change" -- LinearizePool, see
        # jacobians.py): one persistent pool for the whole coarse GN solve
        # instead of jac.linearize() forking a fresh one every iteration.
        lin_pool_c = (jac.LinearizePool(anchor_workers, spectrum_jac)
                     if use_analytic and anchor_workers > 1 else None)
        try:
            x_c, S_ret_c, avk_c = gauss_newton_state(fwd_c, y_true, spec_c, Sy_inv_diag,
                                                     label=f"[{row_lo}-{row_hi}] coarse", verbose=gn_verbose,
                                                     jacobian_fn=_linearizer(spec_c, bin_centers, use_analytic,
                                                                             "linear", pool=lin_pool_c),
                                                     return_cov=True, return_avk=True)
        finally:
            if lin_pool_c is not None:
                lin_pool_c.close()
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

    t0 = time.time()
    # HI-RES: scene on the finer anchor grid, every row interpolated there.
    surface_positions = anchor_etas if surface_positions_mode == "anchor" else None
    row_positions = None
    if frozen_atmosphere_positions_mode == "anchor":
        row_positions = {name: anchor_etas for name in prior_fields if name not in free}
    # 2026-09-12 (user: "[tau_aerosol/height_aerosol's] behavior is much
    # more like that of a gas than the surface -- correlation length
    # scales... are much more like trace gases"): input/retrieval_
    # defaults.yml's own corr_length_eta backs this up directly --
    # tau_aerosol ~100km (matching ch4_ppb/co_ppb's broad-plume width),
    # height_aerosol ~500km (matching t_offset_k/h2o_surface_vmr's own
    # broad synoptic scale) -- neither anywhere near albedo's much
    # shorter ~30km. Both were still riding on albedo's own fine anchor
    # grid by construction (state_spec_from_scene's surface-row loop had
    # no per-row override until this session's fix), spending the large
    # majority of a c5 window's memory resolving along-slit structure
    # the prior itself says isn't there. Put them on the same coarse
    # bin_centers grid the gas rows already use, independent of
    # --surface-positions (which stays governing albedo only) --
    # `state_spec_from_scene` silently ignores a `row_positions` entry
    # for a row that isn't in `surface_fields` this call, so this is a
    # no-op whenever --aerosol is off.
    row_positions = dict(row_positions or {})
    row_positions.setdefault("amplitude_aerosol", bin_centers)
    row_positions.setdefault("height_aerosol", bin_centers)
    row_positions.setdefault("thickness_aerosol", bin_centers)
    spec_h = state_spec_from_scene(bin_centers, free=free, corr_length=corr_length,
                                   prior_form=prior_form, uniform=uniform_priors,
                                   fields=prior_fields, band_label=band_label,
                                   surface_fields=surface_fields,
                                   surface_positions=surface_positions,
                                   row_positions=row_positions,
                                   prior_anchor_density=prior_anchor_density,
                                   row_sub_bin_anomaly=row_sub_bin_anomaly,
                                   kinds=ROW_KINDS, gamma=gamma)
    fwd_h = build_forward_state(FPA, rows_win, anchor_etas, spec_h, spectrum,
                                wn_hires, ils, pad=retrieval_pad, state_interp=state_interp,
                                n_workers=anchor_workers,
                                spatial_psf_fwhm_px=retrieval_psf_fwhm_px)
    # 2026-09-12 (user: "let's make that change" -- LinearizePool, see
    # jacobians.py): one persistent pool for the whole hires GN solve
    # instead of jac.linearize() forking a fresh one every iteration --
    # the dominant driver of the impprior_ws OOMs (Sec.17's pool.map
    # pickling fix alone wasn't enough; see docs/PROJECT_STATUS.md).
    lin_pool_h = (jac.LinearizePool(anchor_workers, spectrum_jac)
                 if use_analytic_hires and anchor_workers > 1 else None)
    try:
        x_h, S_ret_h, avk_h = gauss_newton_state(fwd_h, y_true, spec_h, Sy_inv_diag,
                                                 label=f"[{row_lo}-{row_hi}] hires", verbose=gn_verbose,
                                                 jacobian_fn=_linearizer(spec_h, anchor_etas, use_analytic_hires,
                                                                         state_interp, pool=lin_pool_h),
                                                 return_cov=True, return_avk=True)
    finally:
        if lin_pool_h is not None:
            lin_pool_h.close()
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
    # chi2 in the retrieval's own noise-weighted metric (Sy_inv_diag is the
    # per-pixel 1/sigma^2 from the real GeoCarb noise model, computed above).
    # chi2_reduced normalises by DOF = n_data - n_free; ~1 is a good fit,
    # >>1 means the forward model can't reproduce the data at the noise
    # level (an unrepresentable-scene or model-error signature), <<1 means
    # over-fitting / an over-generous noise model.
    chi2_h = float(np.sum((resid_h ** 2) * Sy_inv_diag))
    out["chi2_hires"] = chi2_h
    out["n_data_hires"] = int(resid_h.size)
    out["n_free_hires"] = int(np.asarray(x_h).size)
    out["chi2_hires_reduced"] = chi2_h / max(resid_h.size - np.asarray(x_h).size, 1)
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


def merge_parts(parts_dir: Path, out: "Path | None" = None) -> int:
    """Merge the per-task output files from a SLURM-array run (--task-id/
    --n-tasks) into a single combined pickle, in exactly the same format
    this script's own non-array (single-process) mode produces -- so
    `plot_sweep` and every other downstream consumer need zero changes.

    No shared state during the run itself (each array task only ever
    wrote its own file); this is the one, deliberately simple, place any
    cross-task I/O happens, run once after the array job completes.
    Formerly the standalone `gd_joint_block_whole_slit_merge.py` (folded
    in 2026-09-09 -- see `--merge`'s own CLI help).
    """
    part_files = sorted(parts_dir.glob("task*of*.pkl"))
    if not part_files:
        print(f"no task*.pkl files found in {parts_dir}")
        return 1

    results = {}
    meta = None
    n_tasks_seen = set()
    task_ids_seen = set()
    all_tiles = None
    for pf in part_files:
        with open(pf, "rb") as f:
            d = pickle.load(f)
        n_tasks_seen.add(d["n_tasks"])
        task_ids_seen.add(d["task_id"])
        if all_tiles is None:
            all_tiles = d["tiles"]
        elif d["tiles"] != all_tiles:
            print(f"FAIL: {pf} has a different `tiles` list than earlier parts -- "
                 f"these are not outputs from the same run.")
            return 1
        if meta is None:
            # window_scale/overlap added 2026-09-13 (user: "only show the
            # values that are at bin centers strictly inside each window's
            # eta bounds") -- plot_sweep needs both to reconstruct the RUN's
            # own natural (overlap=0) tiling and identify each window's core
            # (non-overlap) bins; previously missing here meant a merged
            # pickle's own d.get("overlap", 0)/d.get("window_scale", 1.0)
            # silently fell back to the wrong defaults for any run that
            # used --overlap or --window-scale.
            meta = {k: d[k] for k in ("fpa", "uniform", "gamma", "sigma_abs", "g_ratio",
                                      "min_window", "pad", "window_scale", "overlap")}
            # solver added 2026-09-15 -- default "single_scatter" for any
            # part file written before this field existed (every run
            # before today), matching window_scale/overlap's own
            # backward-compat precedent above.
            meta["solver"] = d.get("solver", "single_scatter")
        else:
            d_solver = d.get("solver", "single_scatter")
            mismatches = {k: (meta[k], (d_solver if k == "solver" else d[k]))
                         for k in meta if meta[k] != (d_solver if k == "solver" else d[k])}
            if mismatches:
                print(f"FAIL: {pf} has different run parameters than earlier parts: {mismatches}")
                return 1
        overlap = set(d["results"]) & set(results)
        if overlap:
            print(f"FAIL: {pf} re-solves windows already covered by another part: {sorted(overlap)}")
            return 1
        results.update(d["results"])

    if len(n_tasks_seen) != 1:
        print(f"FAIL: parts disagree on n_tasks: {n_tasks_seen}")
        return 1
    n_tasks = n_tasks_seen.pop()
    missing_tasks = set(range(n_tasks)) - task_ids_seen
    if missing_tasks:
        print(f"WARNING: {len(missing_tasks)}/{n_tasks} task IDs never wrote a part file "
             f"(likely still running, or failed): {sorted(missing_tasks)}")

    expected_keys = {(lo, hi) for lo, hi in all_tiles}
    missing_windows = expected_keys - set(results)
    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"{len(part_files)} part files, {len(results)}/{len(expected_keys)} windows present "
         f"({n_ok} solved OK, {len(results)-n_ok} errored, {len(missing_windows)} missing)")
    if missing_windows:
        print(f"  missing: {sorted(missing_windows)[:10]}{' ...' if len(missing_windows) > 10 else ''}")

    # 2026-09-13 (user: "create a unique folder for each run that has the
    # pkl and plots"): one directory per run, named for the run itself
    # (parts_dir's own name minus "_parts" -- already the FULL
    # distinguishing config: gratio/adens/free rows/aero/valb/spos/fapos/
    # ovlp/etc., unlike the old default_out's own suffix, which only ever
    # covered uniform/g_ratio and so collided across different --free
    # configs sharing everything else -- e.g. every one of c1-c5 at the
    # same gratio/adens would previously have written the SAME default
    # path). plot_sweep saves alongside whatever pkl it's given, so a run
    # merged into this folder gets its plot placed here too, automatically.
    run_name = parts_dir.name
    if run_name.endswith("_parts"):
        run_name = run_name[: -len("_parts")]
    run_dir = parts_dir.parent / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    default_out = run_dir / f"{run_name}.pkl"
    out_path = out if out else default_out
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "tiles": all_tiles, **meta}, f)
    print(f"saved {out_path}")
    return 0 if not missing_tasks and not missing_windows else 2


#: Display label per state row for `plot_sweep` -- everything it knows how
#: to plot. `als.STATE_FIELDS`'s own 5 atmosphere rows plus `albedo` (the
#: one `surface`-target row `state_spec_from_scene` ever adds).
PLOT_ROW_LABELS = {
    "co2_ppm": "CO2 [ppm]", "ch4_ppb": "CH4 [ppb]", "co_ppb": "CO [ppb]",
    "h2o_surface_vmr": "H2O surface VMR", "p_surface_hpa": "p_surface [hPa]",
    "albedo": "albedo",
}


def _plot_truth_fn_for(row_name: str, truth: str, match_x_km, band_label: str):
    """`fn(x_km) -> value` for one state row, evaluated ONLY at the
    state's own discrete positions -- see `plot_sweep`'s own docstring
    for why `raw` and `bin` coincide there, and why `anchor` alone still
    needs the resolution-matched reconstruction."""
    if row_name == "albedo":
        if truth == "anchor":
            return als.resolution_matched_albedo_fn(match_x_km, band_label)
        return lambda x_km: als.albedo_for_label(x_km, band_label)  # noqa: E731
    if row_name in ("amplitude_aerosol", "height_aerosol", "thickness_aerosol"):
        # 2026-09-15: first successful full c5 (--aerosol) merge exposed
        # this gap (originally for tau_aerosol/height_aerosol, before
        # tau_aerosol was retired as a free row the same day) -- these
        # rows live in als.SURFACE_FIELDS (same
        # (x_km, band_label) signature as albedo), not als.STATE_FIELDS,
        # so the plain STATE_FIELDS[row_name] lookup below raised
        # KeyError. No resolution-matched ("anchor") truth variant exists
        # for either row (unlike albedo) -- both are on the coarse
        # bin_centers grid already (Sec.19), not the fine anchor grid, so
        # there's no sub-anchor structure a resolution-matched
        # reconstruction would need to recover.
        return lambda x_km: als.SURFACE_FIELDS[row_name](x_km, band_label)  # noqa: E731
    if truth == "anchor":
        return als.resolution_matched_fields(match_x_km)[row_name]
    return als.STATE_FIELDS[row_name]


def plot_sweep(in_path: Path, truth: str = "raw", truth_anchor_density: int = 4,
              rows: "list[str] | None" = None, truth_image: "str | None" = None) -> int:
    """Compare the independent windows of a sweep run against truth, for
    EVERY state row saved in the sweep's own snapshots -- `co2_ppm`/
    `ch4_ppb`/`co_ppb`/`h2o_surface_vmr`/`p_surface_hpa`/`albedo`, free or
    frozen. Every quantity is reconstructed from the sweep's own saved
    StateSpec snapshots (`w["hires"]["params"][name]`, already-scaled
    physical values per window) -- no re-solving. Formerly the standalone
    `gd_joint_block_whole_slit_plot.py` (folded in 2026-09-09 -- see
    `--plot`'s own CLI help).

    2026-09-02 (user): "the prior and posterior value should be compared
    to the truth value AT THOSE POINTS. I don't want to see along-slit
    interpolation in these results." Every retrieved quantity lives only
    at its own state's discrete positions (bin centers); the truth
    reference is evaluated EXACTLY at those same positions, never
    interpolated onto an intervening detector row. Each window is
    plotted as its own disconnected '-o' segment (marker + line WITHIN a
    window, no line bridging one window's last point to the next
    window's first) with grey dashed vertical lines at window
    boundaries -- no whole-slit polyline, which used to (a) require
    interpolating the state between bin centers onto every detector row,
    and (b) for `truth="bin"` specifically, silently compare against a
    WHOLE-SLIT truth reference containing other windows' own bin centers
    interleaved in eta-space with this window's (real, even at
    overlap=0, since eta depends on column too -- keystone -- so
    row-disjoint windows are not eta-disjoint) -- a scoring bug, not a
    real retrieval error, found and fixed by removing interpolation
    entirely rather than tracking down every place it could reappear.

    `truth`: `raw` (default) and `bin` are now IDENTICAL when evaluated
    only at bin centers (the whole point of a `bin`-mode truth is that it
    equals the raw continuous truth exactly there, by construction) --
    `bin` is kept as an explicit, self-documenting alias rather than
    removed. `anchor` (resolution-matched to a whole-slit ANCHOR grid,
    genuinely different from raw AT a bin center since bin centers
    generally fall between anchor nodes) still needs the anchor-grid
    reconstruction -- but that one IS a single, whole-slit-coherent
    function (no per-window locality issue: `whole_slit_anchor_etas`
    truth images were rendered as one global scene, unlike a `bin`-mode
    truth, which is built per-window).

    `rows`: which state rows to plot -- default (`None`) ALL rows present
    in the sweep's own saved snapshot (frozen rows included: a frozen
    row's own profile/bias panel is a real, cheap check that it was
    actually held at what the caller intended).
    """
    with open(in_path, "rb") as f:
        d = pickle.load(f)
    results = d["results"]
    # 2026-09-15: an errored window's own snapshot is just {row_lo, row_hi,
    # error} (_worker's own except-branch, see its docstring) -- no
    # "hires" key, so plotting it raised KeyError. First hit on an
    # incomplete/partial merge (some windows never solved OK) rather than
    # a fully-successful one, where this never came up. Skip errored
    # windows the same way merge_parts's own summary line already does,
    # rather than trying to plot a result that was never produced.
    errored = [r for r in results.values() if "error" in r]
    if errored:
        print(f"WARNING: {len(errored)}/{len(results)} windows errored during solve "
             f"(rows {[(r['row_lo'], r['row_hi']) for r in errored]}) -- excluded from plot.")
    windows = sorted((r for r in results.values() if "error" not in r), key=lambda r: r["row_lo"])
    fpa = d.get("fpa", FPA)
    band_label = GEOCARB_BANDS[fpa][0]

    row_names = rows if rows else list(windows[0]["hires"]["params"].keys())

    if truth == "anchor":
        from gd_build_resolution_matched_truth import whole_slit_anchor_etas
        tiling_kw = dict(min_window=d.get("min_window"), window_scale=d.get("window_scale", 1.0),
                         overlap=d.get("overlap", 0))
        match_etas = whole_slit_anchor_etas(fpa, truth_anchor_density, **tiling_kw)
        match_x_km = match_etas * als.SLIT_HALF_KM
        truth_desc = f"resolution-matched truth (anchor_density={truth_anchor_density})"
    else:
        match_x_km = None
        truth_desc = "raw continuous truth (== bin-mode truth exactly at bin centers)"
    print(f"truth reference: {truth_desc}")
    print(f"rows: {row_names}")

    truth_fns = {name: _plot_truth_fn_for(name, truth, match_x_km, band_label) for name in row_names}

    # eta -> approximate detector row, for x-axis PLACEMENT only (never used
    # to reconstruct a value -- a nearest-row lookup, not an interpolation
    # of any physical quantity).
    row_lut = np.arange(1024.0)
    eta_lut = _eta_of(fpa, np.full(1024, 512.0), row_lut)
    order_lut = np.argsort(eta_lut)
    eta_lut_sorted, row_lut_sorted = eta_lut[order_lut], row_lut[order_lut]

    def eta_to_row(eta):
        idx = np.searchsorted(eta_lut_sorted, eta)
        idx = np.clip(idx, 0, len(eta_lut_sorted) - 1)
        return row_lut_sorted[idx]

    # --hires-only sweeps (--anchor-density / --state-interp variants) carry no
    # x_coarse/resid_coarse_rms at all. Detect that once here.
    has_coarse = all("x_coarse" in w for w in windows)
    if not has_coarse:
        print("no x_coarse in this pickle (hi-res-only sweep) -- plotting hi-res only")

    # 2026-09-13 (user: "only show the values that are at bin centers
    # strictly inside each window's eta bounds"): under --overlap, each
    # window's own render extends past its natural (overlap=0) boundary
    # so a later stage (query_state) can blend two windows' independent
    # estimates in the shared rows -- but that also means a bin in the
    # shared margin is covered by BOTH this window and its neighbor, each
    # solved independently, so plotting every window's full range
    # double-plots the overlap margin and can make an otherwise-clean fit
    # look noisier than it is. Restrict each window's own plotted points
    # to strictly inside its natural (overlap=0) tile's eta bounds --
    # boundary_rows/window count below still reflect the real (possibly
    # wider) tiling this run actually used; only which points get DRAWN
    # changes. No-op at overlap=0 (natural bounds == actual bounds).
    overlap_used = d.get("overlap", 0)
    core_eta_bounds = None
    if overlap_used:
        natural_tiles = build_window_tiles(fpa, min_window=d.get("min_window"),
                                           window_scale=d.get("window_scale", 1.0), overlap=0)
        if len(natural_tiles) == len(windows):
            core_eta_bounds = []
            for lo, hi in natural_tiles:
                eta_lo = float(_eta_of(fpa, np.array([512.0]), np.array([float(lo)]))[0])
                eta_hi = float(_eta_of(fpa, np.array([512.0]), np.array([float(hi)]))[0])
                core_eta_bounds.append((min(eta_lo, eta_hi), max(eta_lo, eta_hi)))
        else:
            print(f"WARNING: natural (overlap=0) tiling has {len(natural_tiles)} windows, "
                 f"this run has {len(windows)} -- can't map core bounds by index, "
                 f"plotting every bin (including overlap margins) as-is.")

    # Per window, per row: bin-center rows (x-axis) and true/prior/hires
    # VALUES AT THOSE SAME BIN CENTERS -- no interpolation anywhere.
    per_row = {name: [] for name in row_names}  # each entry: one window's own dict
    resid_c_all, resid_h_all, width_all, G_all, row_mid_all, boundary_rows = [], [], [], [], [], []
    chi2r_h_all = []  # per-window reduced chi2 (hi-res), when present in the snapshot

    for i, w in enumerate(windows):
        row_lo, row_hi = w["row_lo"], w["row_hi"]
        eta_bounds = core_eta_bounds[i] if core_eta_bounds is not None else None
        for name in row_names:
            p = w["hires"]["params"][name]
            positions = np.asarray(p["positions"])
            x_km = positions * als.SLIT_HALF_KM
            true_vals = np.asarray(truth_fns[name](x_km), dtype=float)
            hires_vals = np.asarray(p["values"], dtype=float)
            prior_vals = np.asarray(p["prior"], dtype=float)
            coarse_vals = (np.asarray(w["coarse"]["params"][name]["values"], dtype=float)
                          if has_coarse else None)
            if eta_bounds is not None:
                core_mask = (positions >= eta_bounds[0]) & (positions <= eta_bounds[1])
                positions, true_vals = positions[core_mask], true_vals[core_mask]
                hires_vals, prior_vals = hires_vals[core_mask], prior_vals[core_mask]
                if coarse_vals is not None:
                    coarse_vals = coarse_vals[core_mask]
            per_row[name].append(dict(
                rows=eta_to_row(positions), true=true_vals, hires=hires_vals,
                prior=prior_vals, coarse=coarse_vals))
        width_all.append(w["width"])
        G_all.append(w["G"])
        if has_coarse:
            resid_c_all.append(w["resid_coarse_rms"])
        resid_h_all.append(w["resid_hires_rms"])
        chi2r_h_all.append(w.get("chi2_hires_reduced", np.nan))
        row_mid_all.append(0.5 * (row_lo + row_hi))
        boundary_rows.append(row_lo - 0.5)
    boundary_rows.append(windows[-1]["row_hi"] + 0.5)

    all_rows_seen = np.concatenate([wd["rows"] for name in row_names for wd in per_row[name]])
    rc_fine = rows_crossed(fpa, np.arange(all_rows_seen.min(), all_rows_seen.max() + 1, dtype=float))
    rc_rows = np.arange(all_rows_seen.min(), all_rows_seen.max() + 1)

    # Whole-run summary stats (concatenating every window's own bin-center
    # points -- a plain array op, not a reconstruction).
    prior_differs = d.get("prior_fields", "exact") != "exact"
    for name in row_names:
        wds = per_row[name]
        bh = np.concatenate([wd["hires"] - wd["true"] for wd in wds])
        bp = np.concatenate([wd["prior"] - wd["true"] for wd in wds])
        rows_cat = np.concatenate([wd["rows"] for wd in wds])
        print(f"\n[{name}] prior:  mean={bp.mean():+.4g} rms={np.sqrt(np.mean(bp**2)):.4g} "
             f"max|bias|={np.max(np.abs(bp)):.4g}")
        if has_coarse:
            bc = np.concatenate([wd["coarse"] - wd["true"] for wd in wds])
            print(f"[{name}] coarse: mean={bc.mean():+.4g} rms={np.sqrt(np.mean(bc**2)):.4g} "
                 f"max|bias|={np.max(np.abs(bc)):.4g}")
        print(f"[{name}] hires:  mean={bh.mean():+.4g} rms={np.sqrt(np.mean(bh**2)):.4g} "
             f"max|bias|={np.max(np.abs(bh)):.4g}  (worst near row {rows_cat[np.argmax(np.abs(bh))]})")

    snr_row = None
    if truth_image is not None:
        with open(truth_image, "rb") as f:
            truth_obj = pickle.load(f)
        if isinstance(truth_obj, dict):
            A_img = truth_obj.get("representative", truth_obj.get("dense"))
            if A_img is None:
                A_img = next(v for v in truth_obj.values() if isinstance(v, np.ndarray))
        else:
            A_img = truth_obj
        A_img = np.asarray(A_img, dtype=float)
        noise_model = geocarb_noise_model(fpa)
        continuum = np.max(A_img, axis=1)
        sigma_continuum = np.sqrt(noise_model.N0 ** 2 + noise_model.N1 * np.abs(continuum))
        snr_row = continuum / np.maximum(sigma_continuum, 1e-30)
        snr_rows_axis = np.arange(A_img.shape[0])
        print(f"continuum SNR: min={snr_row.min():.4g} median={np.median(snr_row):.4g} "
             f"max={snr_row.max():.4g}  (from {truth_image})")

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    n_rows = len(row_names)
    have_chi2 = bool(np.isfinite(np.asarray(chi2r_h_all, dtype=float)).any())
    n_extra = 2 + (1 if have_chi2 else 0) + (1 if snr_row is not None else 0)
    height_ratios = ([2.2, 1.6] * n_rows + [1.0]
                     + ([1.0] if have_chi2 else [])
                     + [1.0]
                     + ([1.0] if snr_row is not None else []))
    fig, axes = plt.subplots(2 * n_rows + n_extra, 1,
                             figsize=(13, 3.6 * n_rows + 4 + (1.4 if have_chi2 else 0)
                                      + (1.6 if snr_row is not None else 0)),
                             sharex=True, gridspec_kw={"height_ratios": height_ratios})

    def _boundaries(ax):
        for b in boundary_rows:
            ax.axvline(b, color="0.75", lw=0.6, ls="--", zorder=0)

    ms = 2.5
    for i, name in enumerate(row_names):
        wds = per_row[name]
        label = PLOT_ROW_LABELS.get(name, name)
        free = windows[0]["hires"]["params"][name].get("free", True)
        frozen_tag = "" if free else " [FROZEN]"

        ax = axes[2 * i]
        _boundaries(ax)
        for j, wd in enumerate(wds):
            kw = dict(label=None if j else "true")
            ax.plot(wd["rows"], wd["true"], "-o", color="black", lw=1.1, ms=ms, zorder=5, **kw)
            if prior_differs:
                ax.plot(wd["rows"], wd["prior"], "-o", color="0.5", lw=0.9, ms=ms, ls="--", zorder=4,
                        label=None if j else "prior")
            if has_coarse:
                ax.plot(wd["rows"], wd["coarse"], "-o", color="tab:orange", lw=0.9, ms=ms, alpha=0.85,
                        label=None if j else "coarse posterior")
            ax.plot(wd["rows"], wd["hires"], "-o", color="tab:blue", lw=0.9, ms=ms, alpha=0.85,
                    label=None if j else "hi-res posterior")
        ax.set_ylabel(label)
        title = f"true vs. retrieved {name}{frozen_tag} -- per window, no along-slit interpolation"
        if i == 0:
            title = (f"FPA{fpa} whole-slit joint block sweep, {len(windows)} independent "
                    f"windows ({truth_desc})\n{title}")
        ax.set_title(title, fontsize=11 if i else 12)
        ax.legend(fontsize=8, loc="upper right", markerscale=2)

        ax = axes[2 * i + 1]
        _boundaries(ax)
        ax.axhline(0, color="black", lw=0.6)
        bh_all = np.concatenate([wd["hires"] - wd["true"] for wd in wds])
        bp_all = np.concatenate([wd["prior"] - wd["true"] for wd in wds])
        for j, wd in enumerate(wds):
            bh_j = wd["hires"] - wd["true"]
            if prior_differs:
                bp_j = wd["prior"] - wd["true"]
                ax.plot(wd["rows"], bp_j, "-o", color="0.5", lw=0.8, ms=ms, ls="--",
                        label=(None if j else f"prior (rms={np.sqrt(np.mean(bp_all**2)):.3g}, "
                                             f"max={np.max(np.abs(bp_all)):.3g})"))
            if has_coarse:
                bc_j = wd["coarse"] - wd["true"]
                bc_all = np.concatenate([wd2["coarse"] - wd2["true"] for wd2 in wds])
                ax.plot(wd["rows"], bc_j, "-o", color="tab:orange", lw=0.8, ms=ms,
                        label=(None if j else f"coarse (rms={np.sqrt(np.mean(bc_all**2)):.3g}, "
                                             f"max={np.max(np.abs(bc_all)):.3g})"))
            ax.plot(wd["rows"], bh_j, "-o", color="tab:blue", lw=0.8, ms=ms,
                    label=(None if j else f"hi-res (rms={np.sqrt(np.mean(bh_all**2)):.3g}, "
                                         f"max={np.max(np.abs(bh_all)):.3g})"))
        ax.set_ylabel("bias (-true)" if prior_differs else f"{name}\nbias (posterior-true)")
        ax.set_title(f"{name} bias, per window (dashed grey = window boundary)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    _pi = 2 * n_rows
    ax = axes[_pi]
    _boundaries(ax)
    if has_coarse:
        ax.step(row_mid_all, resid_c_all, where="mid", color="tab:orange", lw=1.2, label="coarse")
    ax.step(row_mid_all, resid_h_all, where="mid", color="tab:blue", lw=1.2, label="hi-res")
    ax.set_yscale("log")
    ax.set_ylabel("per-window\nresid RMS")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("per-window residual RMS (fit quality)", fontsize=10.5)

    if have_chi2:
        _pi += 1
        ax = axes[_pi]
        _boundaries(ax)
        ax.axhline(1.0, color="black", lw=0.6, ls=":")
        ax.step(row_mid_all, chi2r_h_all, where="mid", color="tab:red", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("per-window\nreduced $\\chi^2$")
        _c = np.asarray(chi2r_h_all, dtype=float)
        _c = _c[np.isfinite(_c)]
        ax.set_title(f"per-window reduced $\\chi^2$ (=1 dotted; noise-weighted fit quality, "
                     f"DOF = n_data - n_free)   median={np.median(_c):.3g}", fontsize=10.5)

    _pi += 1
    ax = axes[_pi]
    _boundaries(ax)
    ax2 = ax.twinx()
    ax.step(row_mid_all, width_all, where="mid", color="0.4", lw=1.2, label="window width [rows]")
    ax2.step(row_mid_all, G_all, where="mid", color="#A0631B", lw=1.2, label="G [bins]")
    ax.plot(rc_rows, rc_fine * 4, color="0.75", lw=0.8, ls="--", label="rows_crossed x4 (scale ref.)")
    ax.set_ylabel("window width [rows]")
    ax2.set_ylabel("G [bins]", color="#A0631B")
    if snr_row is None:
        ax.set_xlabel("detector row")
    ax.set_title("adaptive window size and bin count (tied to local keystone)", fontsize=10.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    if snr_row is not None:
        _pi += 1
        ax = axes[_pi]
        _boundaries(ax)
        ax.step(snr_rows_axis, snr_row, where="mid", color="tab:green", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("continuum SNR")
        ax.set_xlabel("detector row")
        ax.set_title("continuum SNR vs. along-slit position (real GeoCarb noise model; "
                     "continuum = each row's own brightest column)", fontsize=10.5)

    fig.tight_layout()
    # 2026-09-13 (user: "create a unique folder for each run that has the
    # pkl and plots"): save alongside whatever pkl this plot was built
    # from, so a run merged into its own directory (merge_parts's own
    # per-run folder, see its docstring) gets its plot placed right next
    # to the data it came from, not off in a separate global tree. Was
    # briefly REPO_ROOT/"plots"/"joint_block" (a fixed global location,
    # the standing "figures go to plots/, never scratchpad" rule) -- this
    # supersedes that for this workflow specifically: `results/` is not
    # scratchpad, and co-locating pkl+plot per run is more useful here
    # than a single shared plots/ tree. A caller pointing --plot at a
    # bare pkl outside any run folder still gets a sensible answer (saves
    # next to that pkl, wherever it is) rather than erroring.
    plots_dir = in_path.parent
    plots_dir.mkdir(parents=True, exist_ok=True)
    truth_suffix = "" if truth != "anchor" else f"_truth-ad{truth_anchor_density}"
    rows_suffix = "" if rows is None else "_" + "-".join(n.split("_")[0] for n in row_names)
    out_path = plots_dir / f"{in_path.stem}{truth_suffix}{rows_suffix}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


def _merge_cli(argv) -> int:
    ap = argparse.ArgumentParser(prog="gd_joint_block_retrieve.py --merge",
                                 description=merge_parts.__doc__)
    ap.add_argument("parts_dir", type=str, help="directory of taskNNNofM.pkl files "
                    "(the *_parts/ directory the array job wrote into)")
    ap.add_argument("--out", type=str, default=None, help="output path (default: a new "
                    "directory named for the run -- parts_dir with the trailing _parts "
                    "stripped -- containing <run_name>.pkl; --plot then saves its figure "
                    "alongside it in that same directory)")
    args = ap.parse_args(argv)
    return merge_parts(Path(args.parts_dir), Path(args.out) if args.out else None)


def _plot_cli(argv) -> int:
    ap = argparse.ArgumentParser(prog="gd_joint_block_retrieve.py --plot",
                                 description=plot_sweep.__doc__)
    ap.add_argument("input", type=str, nargs="?",
                    default=str(REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}.pkl"),
                    help="path to this script's own (or --merge's) output pickle")
    ap.add_argument("--truth", choices=["raw", "anchor", "bin"], default="raw",
                    help="'raw' (default) and 'bin': the raw continuous truth, evaluated exactly "
                         "at each bin center -- identical to each other there by construction. "
                         "'anchor': resolution-matched to a whole-slit ANCHOR grid "
                         "(--truth-anchor-density), genuinely different from raw at a bin center.")
    ap.add_argument("--truth-anchor-density", type=int, default=4,
                    help="anchor_density for --truth anchor (default 4, matching Sec.10/11's "
                         "own convention)")
    ap.add_argument("--rows", type=str, default=None,
                    help="comma-separated state rows to plot (default: every row present in "
                         f"the sweep's own saved snapshot). Known rows: {sorted(PLOT_ROW_LABELS)}.")
    ap.add_argument("--truth-image", type=str, default=None,
                    help="path to a pickle holding the rendered whole-slit truth image this "
                         "sweep's y_true actually came from (e.g. scratch_work/"
                         "whole_slit_truth_ad4.pkl) -- accepts either a bare (1024,1024) array "
                         "or a dict with a 'representative'/'dense' key. Adds a continuum-SNR-"
                         "vs-row panel (real GeoCarb noise model, continuum = each row's own "
                         "brightest/least-absorbed column). Omitted by default since this "
                         "script's own output pickle doesn't carry the rendered image (only "
                         "derived state/residual quantities do).")
    args = ap.parse_args(argv)
    rows = [r.strip() for r in args.rows.split(",")] if args.rows else None
    return plot_sweep(Path(args.input), truth=args.truth,
                      truth_anchor_density=args.truth_anchor_density,
                      rows=rows, truth_image=args.truth_image)


def main() -> int:
    # --merge/--plot are two independent, self-contained modes (formerly
    # separate scripts -- gd_joint_block_whole_slit_merge.py/_plot.py,
    # folded in 2026-09-09) with their OWN flag sets (a positional
    # parts_dir/pkl path, not this script's own --gamma/--fpa/etc.), so
    # they get their own dedicated parsers rather than trying to coexist
    # with the sweep parser's defaults below -- checked first, consuming
    # the rest of argv themselves, before any sweep-only flag is touched.
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        return _merge_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        return _plot_cli(sys.argv[2:])

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
                         "--barcode/--realistic-barcode (default 32, matching gd_per_row_retrieve.py's "
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
    ap.add_argument("--gn-verbose", action="store_true",
                    help="print gauss_newton_state's own per-iteration line (|dx|, "
                         "rms_resid, J, lam, accepted) to stdout as the solve runs, "
                         "instead of only finding out how far it got after the fact -- "
                         "the function already supports this (its own verbose=True), "
                         "just never wired to a flag before. Off by default: noisy for "
                         "quick/smoke-test invocations, on by default in "
                         "submit_impprior_ws.sbatch's own production runs.")
    ap.add_argument("--legacy-truth", action="store_true",
                    help="2026-09-13 (user: match the retrieval's forward model "
                         "exactly): the DEFAULT truth generation now calls "
                         "build_forward_state itself (the retrieval's own forward "
                         "model) on a fully-frozen StateSpec at this window's own "
                         "anchor grid, instead of slicing band['A'] -- the whole-"
                         "image truth _band_setup/render_at_anchors rendered "
                         "independently, on its OWN (possibly different) anchor "
                         "grid. That independence used to leave a real, never-"
                         "quite-zero residual even with field VALUES matched "
                         "exactly (docs/PROJECT_STATUS.md's resolution-matched "
                         "diagnostic). Pass this flag to revert to the old "
                         "band['A']-slice behavior -- for reproducing any "
                         "pre-2026-09-13 result, or comparing against it directly.")
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
    ap.add_argument("--solver", type=str, default="single_scatter",
                    choices=["single_scatter", "xrtm"],
                    help="which gert RTSolver both truth generation and the retrieval's "
                         "own forward model use (2026-09-15, Phase 2 of the XRTM "
                         "integration plan). 'single_scatter' (default, unchanged "
                         "behavior) is gert.rt_solver.SingleScatterSolver -- Beer-Lambert "
                         "+ single-scatter aerosol, no multiple scattering. 'xrtm' is "
                         "gert.rt_solver.XRTMSolver(method='eig_add', n_streams=2) -- real "
                         "multiple scattering, and (only under this solver) genuinely "
                         "smooth analytic height_aerosol/thickness_aerosol Jacobians (see "
                         "geocarb_gert/jacobians.py's height_aerosol_dI_dparam_xrtm/"
                         "thickness_aerosol_dI_dparam_xrtm) -- SingleScatterSolver's own "
                         "K_ssa_lay==0 makes that same composition degenerate. 'xrtm' is "
                         "NOT yet benchmarked at this project's own scale (Phase 5 of the "
                         "plan) -- materially more expensive per RT call than "
                         "single_scatter, opt in deliberately, not as a silent default.")
    ap.add_argument("--n-windows", type=int, default=None,
                    help="target number of slit windows; solved for via "
                         "scale_for_window_count. Default (None) keeps the historical "
                         "keystone-only tiling, which gives 58 on FPA2.")
    ap.add_argument("--window-scale", type=float, default=1.0,
                    help="multiplier on the keystone window-radius formula; larger "
                         "means fewer, wider windows. Ignored when --n-windows is given.")
    ap.add_argument("--row-min", type=int, default=0,
                    help="restrict tiling to [row-min, row-max] instead of the whole "
                         "slit (default: the whole slit, 0-1023) -- e.g. --row-min 890 "
                         "--row-max 935 reproduces the original single-window joint-bin "
                         "case (formerly a separate script, gd_joint_block_retrieve.py, "
                         "retired 2026-09-09 once this flag subsumed its role)")
    ap.add_argument("--row-max", type=int, default=ROW_MAX_IDX)
    ap.add_argument("--bin-scheme", type=str, default="pixel-density",
                    choices=["pixel-density", "uniform"],
                    help="how each window's G bin centers are placed in eta -- "
                         "'pixel-density' (default, unchanged behavior): quantiles of "
                         "the real per-pixel eta distribution, so bin density tracks "
                         "local keystone amplitude. 'uniform': np.linspace(eta_lo, "
                         "eta_hi, G), the original placement pixel-density superseded -- "
                         "kept for the diagnostics.py-style comparison, not recommended "
                         "for production runs.")
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
    ap.add_argument("--aerosol", action="store_true",
                    help="render the TRUTH scene with background aerosol (als.SURFACE_FIELDS' "
                         "amplitude_aerosol/height_aerosol/thickness_aerosol -- AOD "
                         "(=amplitude*thickness*sqrt(2pi)) 0.05 + a haze feature; a mid-BL "
                         "layer height + synoptic drift; a fixed-width Gaussian + its own "
                         "modest drift) AND carry a matching frozen amplitude_aerosol/"
                         "height_aerosol/thickness_aerosol row in the retrieval state. OFF by "
                         "default (2026-09-09, Sec.14): without this, both the truth render "
                         "and the forward model are aerosol-free -- the pre-Sec.11 behaviour. "
                         "Implied automatically when any aerosol row is in "
                         "--free. NOTE: joint aerosol retrieval does not yet converge "
                         "(PROJECT_STATUS Sec.14) -- this flag exists to reproduce that, and "
                         "to keep every OTHER config genuinely aerosol-free.")
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
                 "(gd_per_row_retrieve._band_setup's own if/elif -- fixed atmosphere+brightness only, "
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
    with_aerosol = (args.aerosol or "amplitude_aerosol" in free_check
                    or "height_aerosol" in free_check
                    or "thickness_aerosol" in free_check)
    if with_aerosol and not args.aerosol:
        print("NOTE: --aerosol implied (an aerosol row is in --free).", flush=True)
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
    if resolution_matched_active and not args.vary_albedo:
        # The resolution-matched truth is ALWAYS rendered with a spatially
        # varying albedo (`vary_albedo=True` is hardcoded at the
        # `_band_setup_cached` call below -- `resolution_matched_albedo_fn`
        # only makes sense as a varying field). Without `--vary-albedo` the
        # retrieval would carry NO surface row and fall back to a single
        # constant scalar for the whole window -- a ~0.37 forward residual
        # against the varying truth, with co2 driven tens of ppm off
        # (2026-09-10, docs/PROJECT_STATUS.md Sec.15). Force it on rather
        # than error: a resolution-matched run with a constant-albedo
        # retrieval is never what the caller wants.
        print("NOTE: --vary-albedo forced on (--resolution-matched-* always renders a "
              "varying-albedo truth; a constant-albedo retrieval cannot match it).",
              flush=True)
        args.vary_albedo = True
    if with_aerosol and resolution_matched_active:
        ap.error("--aerosol is not supported with --resolution-matched-* -- the "
                 "resolution-matched truth builder (gd_build_resolution_matched_truth.py) "
                 "has no aerosol fields on its band-limited grid. Run the plain realistic "
                 "scene, or add aerosol to that builder first.")
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
    gpr._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gpr.DEFAULT_SNR_BY_FPA[fpa]
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
        band = gpr._band_setup_cached(fpa, atm_center, absco, geo, solar, snr, n_lookup_samples, None,
                                      args.uniform, args.barcode, args.barcode_bars, False, 0,
                                      args.realistic_barcode, vary_albedo=True,
                                      use_cache=not args.no_truth_cache,
                                      fields=rm_fields, surface_fields=rm_surface_fields,
                                      resolution_tag=rm_tag)
    else:
        band = gpr._band_setup_cached(fpa, atm_center, absco, geo, solar, snr, n_lookup_samples, None,
                                      args.uniform, args.barcode, args.barcode_bars, False, 0,
                                      args.realistic_barcode, vary_albedo=args.vary_albedo,
                                      use_cache=not args.no_truth_cache,
                                      with_aerosol=with_aerosol)
    wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
    print("done.\n", flush=True)

    all_tiles = build_window_tiles(fpa, row_min=args.row_min, row_max=args.row_max,
                                   min_window=args.min_window,
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
    # meant to be held fixed at the true value.
    #
    # 2026-09-13 (user, correcting this section's own former framing: "It's
    # my intention to verify the retrieval works properly by giving it less
    # and less information with each experiment. Everything not free in
    # these experiments should be exactly equal to the truth"): a
    # non-"exact" --prior-fields (e.g. "structural") used to apply
    # UNIFORMLY to every row regardless of free/frozen status -- a FROZEN
    # row's value IS its prior verbatim, so this injected permanent,
    # uncorrectable bias into whatever wasn't under active study, not just
    # the row(s) being tested. Measured directly on t_offset_k: its
    # "structural" prior is flat/zero while truth is a full ±3K synoptic
    # sinusoid -- freezing it under the old code meant carrying ~100% of
    # the true signal as bias, not a small representability gap. That's a
    # confound for exactly the experiment this sweep exists to run
    # ("how much can freeing MORE rows correct for), so an imperfect prior
    # now only ever applies to a row that's actually free; every frozen row
    # uses the real STATE_FIELDS/SURFACE_FIELDS value, mixed in per-row
    # below. Also applies to `--vary-albedo`-only runs where "albedo" is
    # never in --free (still gets a real state row, frozen) -- previously
    # frozen at SURFACE_FIELDS_PRIOR's patch-layout-only approximation
    # instead of the real per-position truth.
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
        _imperfect_fields = als.PRIOR_FIELD_SETS[args.prior_fields]
        _imperfect_surface_fields = als.SURFACE_PRIOR_FIELD_SETS.get(
            args.prior_fields, als.SURFACE_FIELDS)
        prior_fields_resolved = {name: (fn if name in free_check else als.STATE_FIELDS[name])
                                 for name, fn in _imperfect_fields.items()}
        surface_fields_resolved = {name: (fn if name in free_check else als.SURFACE_FIELDS[name])
                                   for name, fn in _imperfect_surface_fields.items()}
    if not with_aerosol:
        # Drop the aerosol rows from the retrieval-side surface registry too,
        # so state_spec_from_scene never adds an amplitude_aerosol/
        # height_aerosol/thickness_aerosol row -- keeping the retrieval
        # forward aerosol-free to match the (now also aerosol-free) truth
        # render. With --aerosol they stay, frozen at their prior/exact
        # value unless also in --free.
        surface_fields_resolved = {k: v for k, v in surface_fields_resolved.items()
                                   if k not in ("amplitude_aerosol", "height_aerosol",
                                                "thickness_aerosol")}

    _SWEEP.update(dict(band=band, absco=absco, wide_inst=wide_inst, geo=geo, solar=solar,
                       albedo=albedo, wn_hires=band["wn_hires"], ils=band["ils"], fpa=fpa,
                       gamma=args.gamma, sigma_abs=args.sigma_abs, g_ratio=args.g_ratio,
                       uniform=args.uniform, uniform_priors=uniform_priors, atm_center=atm_center,
                       hires_only=args.hires_only, gn_verbose=args.gn_verbose,
                       legacy_truth=args.legacy_truth,
                       anchor_density=args.anchor_density,
                       state_interp=args.state_interp,
                       free=tuple(x.strip() for x in args.free.split(',')),
                       corr_length=args.corr_length, prior_form=args.prior_form,
                       jacobian=args.jacobian, solver=args.solver,
                       prior_fields=prior_fields_resolved,
                       surface_fields=surface_fields_resolved,
                       prior_anchor_density=args.prior_anchor_density,
                       flat_sy_inv=args.flat_sy_inv, vary_albedo=args.vary_albedo,
                       row_sub_bin_anomaly=row_sub_bin_anomaly,
                       surface_positions_mode=args.surface_positions,
                       frozen_atmosphere_positions_mode=args.frozen_atmosphere_positions,
                       anchor_workers=args.anchor_workers,
                       retrieval_psf_fwhm_px=args.retrieval_psf_fwhm_px,
                       bin_scheme=args.bin_scheme, with_aerosol=with_aerosol))

    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    if args.anchor_workers > 1 and n_workers > 1 and args.task_id is None:
        print(f"  NOTE: --anchor-workers {args.anchor_workers} has no effect here -- "
             f"windows are being distributed via an in-process Pool (--n-workers "
             f"{n_workers}), whose daemon workers cannot themselves spawn a nested pool "
             f"(build_forward_state's own daemon check forces anchor-level parallelism "
             f"back to 1 there). Use --task-id/--n-tasks instead to distribute windows as "
             f"separate top-level processes if you want both levels active at once.",
             flush=True)

    # 2026-09-11 (user: "find the cost driver" for the impprior_ws c3/c4/c5
    # SLURM-array TIMEOUTs): a single-tile task (the normal --task-id case
    # -- every array task owns exactly one window) still went through
    # ctx.Pool(n_workers) below -- Pool() unconditionally pre-forks
    # n_workers DAEMON processes even with only one job to hand out, and
    # _worker/_solve_window then ran inside one of them.
    # anchor_spectra_and_derivs's and linearize's own nested-pool guards
    # (`if mp.current_process().daemon: n_workers = 1`) silently forced
    # --anchor-workers down to 1 as a result -- confirmed directly
    # (mp.current_process().daemon printed True from inside _worker in
    # this exact configuration). That serialized every anchor RT call and
    # every Jacobian L()-projection column: measured on the rows 346-362
    # c4 window that was timing out at 12h, one linearize() call took
    # 1736s single-threaded vs 16.6s for one forward(); with max_iter=15
    # outer GN iterations each needing one linearize(), that alone
    # approaches the wall-clock limit before counting LM retries or a
    # genuinely hard-to-converge window. The warning above never caught
    # this because it only fires when `args.task_id is None`, exactly
    # backwards from where the array sweep lives. Skipping the outer Pool
    # entirely for the single-tile case lets anchor_workers actually
    # parallelize the one window a --task-id task owns.
    single_tile = len(tiles) == 1 and args.task_id is not None
    if single_tile:
        print(f"solving 1 window in-process (--anchor-workers {args.anchor_workers} "
             f"active -- single-tile task, no outer Pool)...", flush=True)
    else:
        print(f"solving with {n_workers} workers...", flush=True)

    t0 = time.time()
    results = {}
    if single_tile:
        tile_results = [_worker(tiles[0])]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            tile_results = list(pool.imap_unordered(_worker, tiles, chunksize=1))
    n_done = 0
    for r in tile_results:
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
    # 2026-09-10: without these two, a run with frozen rows on the anchor
    # grid silently shares a _parts directory with the (materially
    # different) shared-bin-grid run of the same --free set -- the same
    # naming-collision class flagged in docs/PROJECT_STATUS.md Sec.1/9.
    if args.surface_positions != "shared":
        suffix += f"_spos-{args.surface_positions}"
    if args.frozen_atmosphere_positions != "shared":
        suffix += f"_fapos-{args.frozen_atmosphere_positions}"
    if args.resolution_matched_anchor_density is not None:
        suffix += f"_rmad{args.resolution_matched_anchor_density}"
    if args.resolution_matched_g_ratio_bins is not None:
        suffix += f"_rmgr{args.resolution_matched_g_ratio_bins:g}"
    if with_aerosol:
        suffix += "_aero"
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
              "jacobian": args.jacobian, "solver": args.solver, "n_windows": len(all_tiles),
              "window_scale": window_scale, "prior_fields": args.prior_fields,
              "n_lookup_samples": n_lookup_samples,
              "prior_anchor_density": args.prior_anchor_density,
              "flat_sy_inv": args.flat_sy_inv, "overlap": args.overlap,
              "vary_albedo": args.vary_albedo, "sub_bin_anomaly": args.sub_bin_anomaly,
              "anchor_workers": args.anchor_workers,
              "retrieval_psf_fwhm_px": args.retrieval_psf_fwhm_px,
              "with_aerosol": with_aerosol}
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
