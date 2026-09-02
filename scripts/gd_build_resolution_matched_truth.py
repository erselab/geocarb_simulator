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
window's own anchor grid from `gd_joint_block_whole_slit_sweep.py::_solve_
window`'s own formula (`a_lo, a_hi = max(0, row_lo-PAD), min(ROW_MAX_IDX,
row_hi+PAD)`, `anchor_rows = arange(a_lo, a_hi, 1/anchor_density)`), over
the standard 58-window tiling -- the SAME grid a real whole-slit retrieval
at that anchor_density actually uses, not an approximation of it.

Caching: routed through `gd_test._band_setup_cached`'s existing
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

import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, _eta_of  # noqa: E402
from gd_joint_block_whole_slit_sweep import build_window_tiles, PAD, ROW_MAX_IDX  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402


def whole_slit_anchor_etas(fpa: int, anchor_density: int, *, min_window: int = None,
                           window_scale: float = 1.0, overlap: int = 0) -> np.ndarray:
    """The exact union of every window's own anchor grid, standard 58-
    window tiling -- the SAME grid a real whole-slit retrieval at this
    anchor_density uses (`gd_joint_block_whole_slit_sweep.py::_solve_
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
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
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
        band = gdt._band_setup_cached(
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
        band = gdt._band_setup_cached(
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
