#!/usr/bin/env python3
"""Build and cache a collection of resolution-matched TRUTH images.

2026-08-29 (user): removes a confound that kept surfacing in Sec.10/11's
investigations -- when a retrieval underperforms, it's genuinely ambiguous
whether that's the retrieval mechanism falling short, or the TRUTH scene
having real structure below any resolution the model could ever represent.
This builds one truth image per `--anchor-density` (default {1,4,16}),
whose underlying atmosphere (`co2_ppm`, `ch4_ppb`, `co_ppb`, `h2o_surface_
vmr`, `p_surface_hpa`) and surface (`albedo`) fields are deliberately
band-limited -- piecewise-linear at EXACTLY the anchor grid that anchor
density would use for a real whole-slit retrieval, with zero structure
below that spacing (`geocarb_gert.along_slit_scene.resolution_matched_
fields`/`resolution_matched_albedo_fn`). Retrieving against a truth the
model can, in principle, perfectly represent isolates real retrieval/model
error from truth the anchor grid could never have captured.

The anchor grid itself is the exact union (sorted, deduplicated) of every
window's own anchor grid from `gd_joint_block_retrieve.py::_solve_
window`'s own formula (`a_lo, a_hi = max(0, row_lo-PAD), min(ROW_MAX_IDX,
row_hi+PAD)`, `anchor_rows = arange(a_lo, a_hi, 1/anchor_density)`), over
the standard 58-window tiling -- the SAME grid a real whole-slit retrieval
at that anchor_density actually uses, not an approximation of it.

Caching: routed through `gd_per_row_retrieve._band_setup_cached`'s existing
`truth_cache` mechanism, with a `resolution_tag` (an anchor-grid hash, not
just the anchor_density label) so a silently-different tiling/PAD from a
future code change can never collide with a stale cache entry, and so a
call WITHOUT this feature (every existing caller) keeps hitting exactly
the cache entries it always has -- verified directly, see docs/
PROJECT_STATUS.md's own write-up of this feature.

For a later comparison run: scoring must ALSO use `resolution_matched_
fields`/`resolution_matched_albedo_fn` as the truth reference (not
`als.xco2_ppm`/`als.albedo_for_label` directly) -- reconstruct with the
SAME anchor grid this script prints/saves in the manifest, deterministic
given `--anchor-density` and the standard tiling, so no need to persist
the anchor VALUES separately.

Run:  PYTHONPATH=. python3 scripts/gd_build_resolution_matched_truth.py
        [--anchor-density 1 4 16] [--n-lookup-samples 20000]
Output: one cached truth image per anchor_density (results/truth_cache/),
        plus results/resolution_matched_truth/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import gd_per_row_retrieve as gpr  # noqa: E402
from gd_joint_block_retrieve import (FPA, GERT_ROOT, _eta_of,  # noqa: E402
                                     build_window_tiles, PAD, ROW_MAX_IDX,
                                     _make_state_spectrum)
from geocarb_gert.joint_state import pixel_density_bin_centers  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from geocarb_gert.joint_state import render_at_anchors, default_pad_for_psf  # noqa: E402


def render_dense_truth_window(fpa: int, row_lo: int, row_hi: int, dx_km: float,
                              spectrum, wn_hires, ils, band_label: str, pad: int | None = None,
                              fields=None, surface_fields=None,
                              spatial_psf_fwhm_px: float = 1.5) -> np.ndarray:
    """Mode 1 -- "highest-resolution truth": anchors spaced ``dx_km`` apart
    (default 500m, user), UNIFORMLY across this window's own PAD-extended
    η range, sampling the FULL CONTINUOUS truth (`fields`/`surface_fields`
    default to the raw, un-band-limited `als.STATE_FIELDS`/`SURFACE_
    FIELDS`) -- as-true-as-affordable, not tied to any retrieval grid.
    Goes through :func:`geocarb_gert.joint_state.render_at_anchors`, the
    SAME forward model every other caller (Mode 2 below, the retrieval's
    own `build_forward_state`) uses -- fresh RT per anchor, footprint-
    integrated into pixels, never spectral interpolation (2026-09-02,
    user). Returns this window's own ``(row_hi-row_lo+1, 1024)`` sub-image.

    ``spatial_psf_fwhm_px`` (2026-09-06, user: defocus experiments) is the
    along-slit PSF FWHM this TRUTH is rendered with -- a wider value
    simulates the telescope's focal-adjustment mechanism defocused, purely
    spatial (does not touch `ils`/spectral resolution). Default 1.5
    reproduces prior behavior exactly. ``pad`` defaults to
    :func:`geocarb_gert.joint_state.default_pad_for_psf` of whatever
    ``spatial_psf_fwhm_px`` is given (4 at the nominal 1.5px, matching the
    old hardcoded default byte-for-byte) rather than a fixed ``4`` --
    otherwise a wide-PSF render would truncate its own Gaussian kernel at
    the window edges. The anchor-grid extension margin (``a_lo``/``a_hi``
    below) is widened to match whenever ``pad`` exceeds the module's own
    ``PAD``, since `render_at_anchors` requires anchor coverage to be at
    least as wide as its own render pad.
    """
    if pad is None:
        pad = default_pad_for_psf(spatial_psf_fwhm_px)
    fields = als.STATE_FIELDS if fields is None else fields
    surface_fields = als.SURFACE_FIELDS if surface_fields is None else surface_fields
    rows_win = np.arange(row_lo, row_hi + 1)
    ext = max(PAD, pad)
    a_lo, a_hi = max(0, row_lo - ext), min(ROW_MAX_IDX, row_hi + ext)
    # dx_km in ROW units at this window's own local eta/row scale --
    # dispersed uniformly in ROW index (matching every other anchor grid
    # in this codebase, which is built uniform-in-row not uniform-in-km;
    # keystone's own row->eta nonlinearity then makes the actual km
    # spacing vary slightly across the window, same as anchor_density's
    # own convention elsewhere).
    row_span_km = (a_hi - a_lo) * als.SLIT_HALF_KM * 2 / (2 * ROW_MAX_IDX)  # rough row->km scale
    dx_rows = max(dx_km / max(row_span_km / max(a_hi - a_lo, 1e-9), 1e-9), 1e-6)
    anchor_rows = np.arange(a_lo, a_hi + 1e-9, dx_rows)
    anchor_etas = _eta_of(fpa, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float))
    anchor_x_km = anchor_etas * als.SLIT_HALF_KM

    atm_params = {name: np.asarray(fn(anchor_x_km), dtype=float) for name, fn in fields.items()}
    surf_params = ({"albedo": np.asarray(surface_fields["albedo"](anchor_x_km, band_label), dtype=float)}
                   if band_label is not None else {})
    return render_at_anchors(fpa, rows_win, anchor_etas, atm_params, surf_params,
                             spectrum, wn_hires, ils, pad=pad,
                             spatial_psf_fwhm_px=spatial_psf_fwhm_px)


def render_representative_truth_window(fpa: int, row_lo: int, row_hi: int, bin_centers_eta,
                                       anchor_density: int, spectrum, wn_hires, ils,
                                       band_label: str, pad: int | None = None,
                                       state_interp: str = "linear",
                                       spatial_psf_fwhm_px: float = 1.5) -> np.ndarray:
    """Mode 2 -- "representative truth": the truth the retrieval's OWN
    forward code, with THESE bin centers and THIS anchor density, can
    reproduce EXACTLY (2026-09-02, user). Not a separate approximation of
    that claim -- it literally reuses `state_spec_from_scene` +
    `build_forward_state` (the retrieval's own machinery) with the state
    fixed at `resolution_matched_fields(bin_centers)`'s own prior (exact
    at every bin center, `prior_anchor_density=None`) and evaluates
    ``forward(x0())`` -- the SAME code path the retrieval calls every GN
    iteration, at the prior. Representability is therefore zero BY
    CONSTRUCTION (same function, same inputs), not by two independently-
    written pipelines happening to agree closely.

    ``spatial_psf_fwhm_px``/``pad`` follow `render_dense_truth_window`'s
    own convention exactly (2026-09-06, user: defocus experiments) -- see
    that function's docstring.

    Returns this window's own ``(row_hi-row_lo+1, 1024)`` sub-image.
    """
    from gd_joint_block_retrieve import state_spec_from_scene
    from geocarb_gert.joint_state import build_forward_state

    if pad is None:
        pad = default_pad_for_psf(spatial_psf_fwhm_px)
    bin_centers_eta = np.asarray(bin_centers_eta, dtype=float)
    bin_centers_x_km = bin_centers_eta * als.SLIT_HALF_KM
    fields = als.resolution_matched_fields(bin_centers_x_km)
    # state_spec_from_scene's own surface_fields convention is
    # {"albedo": fn(x_km, band_label)} -- keyed by ROW NAME (2 positional
    # args), NOT {band_label: fn(x_km)} (a mismatch already found and
    # fixed once before for gd_joint_block_retrieve.py's own
    # wiring -- resolution_matched_albedo_fn itself returns a 1-arg
    # fn(x_km), so it needs the same wrap here).
    surface_fields = ({"albedo": lambda x_km, label, _fn=als.resolution_matched_albedo_fn(
                          bin_centers_x_km, band_label): _fn(x_km)}
                      if band_label is not None else None)
    all_names = list(fields.keys()) + (["albedo"] if band_label is not None else [])
    spec_true = state_spec_from_scene(bin_centers_eta, free=all_names, uniform=False,
                                      fields=fields, band_label=band_label,
                                      surface_fields=surface_fields, prior_anchor_density=None)

    rows_win = np.arange(row_lo, row_hi + 1)
    ext = max(PAD, pad)
    a_lo, a_hi = max(0, row_lo - ext), min(ROW_MAX_IDX, row_hi + ext)
    anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / anchor_density)
    anchor_etas = _eta_of(fpa, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float))

    fwd = build_forward_state(fpa, rows_win, anchor_etas, spec_true, spectrum,
                              wn_hires, ils, pad=pad, state_interp=state_interp,
                              spatial_psf_fwhm_px=spatial_psf_fwhm_px)
    # build_forward_state's own forward() ravels its output (matching the
    # retrieval's own y_true/y0 convention) -- reshaped back to
    # (n_rows, 1024) here so this returns the SAME shape
    # render_dense_truth_window/render_at_anchors do, for consistent
    # whole-slit stitching regardless of which mode built a given window.
    return fwd(spec_true.x0()).reshape(len(rows_win), -1)


def whole_slit_anchor_etas(fpa: int, anchor_density: int, *, min_window: int = None,
                           window_scale: float = 1.0, overlap: int = 0) -> np.ndarray:
    """The exact union of every window's own anchor grid, standard 58-
    window tiling -- the SAME grid a real whole-slit retrieval at this
    anchor_density uses (`gd_joint_block_retrieve.py::_solve_
    window`'s own formula), not a from-scratch uniform approximation of it.

    2026-09-01 (bug fix): `min_window`/`window_scale`/`overlap` MUST match
    whatever tiling the retrieval sweep this truth is being matched to
    actually used (`args.min_window`/the `window_scale` implied by
    `--n-windows`/`args.overlap`) -- `build_window_tiles` defaults
    (`window_scale=1.0, overlap=0`) silently tile DIFFERENTLY from a sweep
    run with `--overlap > 0` (or a non-default `--n-windows`/`--min-
    window`), which shifts every window's own `row_lo`/`row_hi` and hence
    the density-weighted anchor/bin placement near every window boundary
    -- exactly the "frozen rows aren't bit-exact at the truth" artifact
    found and traced in docs/PROJECT_STATUS.md Sec.12.8. Defaulting these
    to `build_window_tiles`'s own defaults keeps every pre-fix standalone
    caller (a manifest built with the standard 58-window, zero-overlap
    tiling) reproducing the exact same grid as before.
    """
    kw = dict(overlap=overlap, window_scale=window_scale)
    if min_window is not None:
        kw["min_window"] = min_window
    tiles = build_window_tiles(fpa, **kw)
    parts = []
    for row_lo, row_hi in tiles:
        a_lo, a_hi = max(0, row_lo - PAD), min(ROW_MAX_IDX, row_hi + PAD)
        anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / anchor_density)
        parts.append(_eta_of(fpa, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float)))
    return np.unique(np.concatenate(parts))


def whole_slit_bin_centers(fpa: int, g_ratio: float, *, min_window: int = None,
                           window_scale: float = 1.0, overlap: int = 0) -> np.ndarray:
    """The exact union of every window's own STATE BIN grid at this
    g_ratio, standard 58-window tiling -- `gd_joint_block_whole_slit_
    sweep.py::_solve_window`'s own `G`/`bin_centers` formula
    (`G = max(2, round(width/g_ratio))`, `pixel_density_bin_centers`),
    replicated exactly, not approximated.

    2026-08-29 (user): a truth built from THIS grid (rather than the
    anchor grid `whole_slit_anchor_etas` uses) removes the representability
    gap entirely -- the retrieval's own bin-to-bin piecewise-linear
    reconstruction becomes bit-for-bit the same function the truth was
    built from, everywhere, not just at the bin centers. Combined with a
    fine anchor_density for the actual forward-model sampling (so no
    anchor-level stepping-bias confound gets reintroduced), this is a
    genuine ceiling test: with the state fully, exactly representable by
    the model, the observation-minus-model and prior-minus-model
    residuals are zero at the prior by construction, not just close.

    2026-09-01 (bug fix): see `whole_slit_anchor_etas`'s own note --
    `min_window`/`window_scale`/`overlap` must match the retrieval
    sweep's actual tiling, or `row_lo`/`row_hi` per window (and hence
    `G`/`bin_centers`, density-weighted over that exact row range) drift
    from what `_solve_window` itself computes, concentrated at window
    boundaries where `overlap` shifts them the most.
    """
    kw = dict(overlap=overlap, window_scale=window_scale)
    if min_window is not None:
        kw["min_window"] = min_window
    tiles = build_window_tiles(fpa, **kw)
    cols = np.arange(1024.0)
    parts = []
    for row_lo, row_hi in tiles:
        rows_win = np.arange(row_lo, row_hi + 1)
        width = len(rows_win)
        G = max(2, int(round(width / g_ratio)))
        eta_all = np.stack([_eta_of(fpa, cols, np.full(1024, float(i))) for i in rows_win])
        parts.append(pixel_density_bin_centers(eta_all.ravel(), G))
    return np.unique(np.concatenate(parts))


_WHOLE_SLIT_TRUTH_G: dict = {}


def _render_one_window(tile):
    """Module-level (picklable, fork-inherited -- same pattern as
    `along_slit_scene._lookup_sample`/`geocarb_gert.joint_state._render_
    one_anchor`) so `build_whole_slit_truth` parallelizes across WINDOWS,
    the same granularity `gd_joint_block_retrieve.py`'s own sweep
    already parallelizes at -- each window-level worker then falls back
    to a single-process anchor loop internally (render_at_anchors/
    build_forward_state detect the daemon worker process and refuse to
    nest a second pool), avoiding nested-pool errors entirely rather than
    needing to divide `n_workers` between two parallelism levels.
    2026-09-02 (user): fixed after this ran fully single-threaded on a
    16-core interactive session.
    """
    G = _WHOLE_SLIT_TRUTH_G
    row_lo, row_hi = tile
    psf = G.get("spatial_psf_fwhm_px", 1.5)
    if G["mode"] == "dense":
        sub = render_dense_truth_window(G["fpa"], row_lo, row_hi, G["dx_km"], G["spectrum"],
                                        G["wn_hires"], G["ils"], G["band_label"], pad=G["pad"],
                                        spatial_psf_fwhm_px=psf)
    else:
        cols = np.arange(1024.0)
        rows_win = np.arange(row_lo, row_hi + 1)
        width = len(rows_win)
        Gn = max(2, int(round(width / G["g_ratio"])))
        eta_all = np.stack([_eta_of(G["fpa"], cols, np.full(1024, float(i))) for i in rows_win])
        bin_centers_eta = pixel_density_bin_centers(eta_all.ravel(), Gn)
        sub = render_representative_truth_window(
            G["fpa"], row_lo, row_hi, bin_centers_eta, G["anchor_density"], G["spectrum"],
            G["wn_hires"], G["ils"], G["band_label"], pad=G["pad"], state_interp=G["state_interp"],
            spatial_psf_fwhm_px=psf)
    return row_lo, row_hi, sub


def build_whole_slit_truth(fpa: int, mode: str, spectrum, wn_hires, ils, band_label: str,
                           *, dx_km: float = 0.5, g_ratio: float = 1.0, anchor_density: int = 16,
                           min_window: int = None, window_scale: float = 1.0, overlap: int = 0,
                           pad: int | None = None, state_interp: str = "linear",
                           n_workers: int | None = None, stitch: bool = True,
                           spatial_psf_fwhm_px: float = 1.5):
    """Every window's own truth sub-image, either STITCHED into one full
    ``(1024, 1024)`` detector array (``stitch=True``, the default) or kept
    as a per-window COLLECTION (``stitch=False``) -- ``{(row_lo, row_hi):
    sub_image}``, one entry per window, never merged.

    ``stitch=True`` requires ``overlap=0`` (raises otherwise): disjoint
    tiling means every row belongs to EXACTLY one window, so a single
    array is unambiguous and every row is exactly representable by
    construction (mode="representative") or as-true-as-affordable
    (mode="dense"). At ``overlap>0`` two adjacent windows' own
    independently-computed local anchor grids generally disagree in their
    shared rows, so there is no single array that is simultaneously
    exactly representable by BOTH windows there -- unresolved, deferred
    (2026-09-02, user).

    ``stitch=False`` sidesteps that entirely (2026-09-02, user: "generate
    individual window-specific truth images ... for debugging and sanity
    checks") -- works at ANY `overlap`, since nothing gets merged; each
    window's own entry is exactly what THAT window's own retrieval
    forward code can reproduce (mode="representative") or the best
    available reference (mode="dense"), looked up by that window's own
    ``(row_lo, row_hi)`` key rather than by detector row. This is the
    right tool for testing one specific window at `--overlap 2` (this
    session's own production convention) without first having to resolve
    the whole-slit stitching-conflict question above.

    Parameters
    ----------
    mode : "dense" or "representative"
        "dense" -- Mode 1 (`render_dense_truth_window`): anchors every
        `dx_km`, sampling the full continuous truth.
        "representative" -- Mode 2 (`render_representative_truth_window`):
        the retrieval's own bin-grid-band-limited truth, exactly
        reproducible by the retrieval's own forward code at `g_ratio`/
        `anchor_density`.
    n_workers : int, optional
        Windows are independent -- ``None`` (default) uses every
        available CPU; ``1`` forces the old single-process loop.
    spatial_psf_fwhm_px : float
        Along-slit PSF FWHM [px] this whole-slit truth is rendered with
        (2026-09-06, user: defocus experiments) -- forwarded to every
        window's own `render_dense_truth_window`/`render_representative_
        truth_window` call. Default 1.5 reproduces prior behavior
        exactly; ``pad`` (left ``None``) auto-scales via `geocarb_gert.
        joint_state.default_pad_for_psf` to match.

    Returns
    -------
    ndarray, shape (1024, 1024) if ``stitch=True``;
    dict[(int, int), ndarray] (one entry per window) if ``stitch=False``.
    """
    import multiprocessing as mp
    from geocarb_gert.gd_render import available_cpus

    if pad is None:
        pad = default_pad_for_psf(spatial_psf_fwhm_px)
    if mode not in ("dense", "representative"):
        raise ValueError(f"mode must be 'dense' or 'representative', got {mode!r}")
    if stitch and overlap != 0:
        raise ValueError("stitch=True requires overlap=0 (a single array cannot be "
                         "simultaneously exact for two overlapping windows' own "
                         "independent local anchor grids -- pass stitch=False for "
                         "a per-window collection at overlap>0 instead)")
    tiles = build_window_tiles(fpa, overlap=overlap,
                               **({"min_window": min_window} if min_window is not None else {}),
                               window_scale=window_scale)

    if n_workers is None:
        n_workers = available_cpus()
    if mp.current_process().daemon:
        n_workers = 1

    _WHOLE_SLIT_TRUTH_G.update(dict(
        fpa=fpa, mode=mode, spectrum=spectrum, wn_hires=wn_hires, ils=ils,
        band_label=band_label, dx_km=dx_km, g_ratio=g_ratio, anchor_density=anchor_density,
        pad=pad, state_interp=state_interp, spatial_psf_fwhm_px=spatial_psf_fwhm_px))
    if n_workers <= 1:
        results = [_render_one_window(tile) for tile in tiles]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(n_workers, len(tiles))) as pool:
            results = pool.map(_render_one_window, tiles)

    if not stitch:
        return {(row_lo, row_hi): sub for row_lo, row_hi, sub in results}

    A = np.full((1024, 1024), np.nan)
    for row_lo, row_hi, sub in results:
        A[row_lo:row_hi + 1, :] = sub
    if np.any(np.isnan(A)):
        missing = np.where(np.isnan(A).any(axis=1))[0]
        raise RuntimeError(f"whole-slit truth has {len(missing)} unfilled rows "
                           f"(tiling gap, e.g. rows {missing[:5]}...) -- overlap=0 tiling "
                           f"should cover every row exactly once")
    return A


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-density", type=int, nargs="*", default=None,
                    help="build truth image(s) matched to the ANCHOR grid at these "
                         "densities (default [1,4,16] if neither this nor --g-ratio-bins "
                         "is given).")
    ap.add_argument("--g-ratio-bins", type=float, nargs="*", default=None,
                    help="2026-08-29: build truth image(s) matched to the retrieval's own "
                         "STATE BIN grid at these g_ratio values, instead of the anchor "
                         "grid -- removes the bin-grid-vs-anchor-grid representability gap "
                         "entirely (whole_slit_bin_centers), a genuine ceiling test rather "
                         "than 'as fine as the forward model samples.' Pair with a fine "
                         "--anchor-density on the RETRIEVAL side (not this script) so the "
                         "forward model can still resolve this now-coarser-but-EXACT truth "
                         "without reintroducing an anchor-level stepping-bias confound.")
    ap.add_argument("--n-lookup-samples", type=int, default=20000,
                    help="build_lookup_radiance's own uniform sample spacing (default "
                         "20000 -> ~140m over the 2800km slit) -- must stay safely below "
                         "the finest anchor spacing tested (anchor_density=16 -> ~375m), "
                         "or the render's own radiance-interpolation adds a SECOND, "
                         "unintended source of band-limiting on top of the deliberate one.")
    ap.add_argument("--fpa", type=int, default=FPA)
    args = ap.parse_args()

    fpa = args.fpa
    band_label = GEOCARB_BANDS[fpa][0]

    print(f"Loading absco/solar from {GERT_ROOT} ...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gpr._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gpr.DEFAULT_SNR_BY_FPA[fpa]
    print("done.\n", flush=True)

    anchor_densities = args.anchor_density
    if anchor_densities is None and args.g_ratio_bins is None:
        anchor_densities = [1, 4, 16]
    elif anchor_densities is None:
        anchor_densities = []

    manifest = []
    for anchor_density in anchor_densities:
        print(f"=== anchor_density={anchor_density} ===", flush=True)
        anchor_etas = whole_slit_anchor_etas(fpa, anchor_density)
        match_x_km = anchor_etas * als.SLIT_HALF_KM
        print(f"  {len(match_x_km)} anchors across the whole slit "
             f"(min spacing {np.min(np.diff(match_x_km)):.3f}km)", flush=True)

        fields = als.resolution_matched_fields(match_x_km)
        surface_fields = {band_label: als.resolution_matched_albedo_fn(match_x_km, band_label)}
        match_hash = hashlib.sha256(match_x_km.tobytes()).hexdigest()[:16]
        resolution_tag = f"ad{anchor_density}-{match_hash}"

        t0 = time.time()
        band = gpr._band_setup_cached(
            fpa, atm_center, absco, geo, solar, snr, args.n_lookup_samples, None,
            False, False, None, False, 0, False, vary_albedo=True,
            fields=fields, surface_fields=surface_fields, resolution_tag=resolution_tag)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

        manifest.append(dict(match="anchor_density", anchor_density=anchor_density,
                             n_points=len(match_x_km), resolution_tag=resolution_tag, fpa=fpa,
                             n_lookup_samples=args.n_lookup_samples,
                             image_shape=list(band["A"].shape)))

    for g_ratio in (args.g_ratio_bins or []):
        print(f"=== g_ratio_bins={g_ratio} ===", flush=True)
        bin_etas = whole_slit_bin_centers(fpa, g_ratio)
        match_x_km = bin_etas * als.SLIT_HALF_KM
        print(f"  {len(match_x_km)} bin centers across the whole slit "
             f"(min spacing {np.min(np.diff(match_x_km)):.3f}km)", flush=True)

        fields = als.resolution_matched_fields(match_x_km)
        surface_fields = {band_label: als.resolution_matched_albedo_fn(match_x_km, band_label)}
        match_hash = hashlib.sha256(match_x_km.tobytes()).hexdigest()[:16]
        resolution_tag = f"gr{g_ratio:g}bins-{match_hash}"

        t0 = time.time()
        band = gpr._band_setup_cached(
            fpa, atm_center, absco, geo, solar, snr, args.n_lookup_samples, None,
            False, False, None, False, 0, False, vary_albedo=True,
            fields=fields, surface_fields=surface_fields, resolution_tag=resolution_tag)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

        manifest.append(dict(match="g_ratio_bins", g_ratio=g_ratio,
                             n_points=len(match_x_km), resolution_tag=resolution_tag, fpa=fpa,
                             n_lookup_samples=args.n_lookup_samples,
                             image_shape=list(band["A"].shape)))

    out_dir = REPO_ROOT / "results" / "resolution_matched_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    # Merge with any existing manifest (by resolution_tag) rather than
    # overwrite -- a --g-ratio-bins-only run must not silently discard the
    # --anchor-density entries an earlier run already built, and vice versa.
    existing = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
    by_tag = {e["resolution_tag"]: e for e in existing}
    for e in manifest:
        by_tag[e["resolution_tag"]] = e
    with open(manifest_path, "w") as f:
        json.dump(list(by_tag.values()), f, indent=2)
    print(f"\nsaved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
