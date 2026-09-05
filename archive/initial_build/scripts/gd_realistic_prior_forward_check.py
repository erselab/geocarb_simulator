#!/usr/bin/env python3
"""Forward-only truth-vs-prior comparison for the realistic-prior experiment
class (see docs/JOINT_BLOCK_MIGRATION_PLAN.md and the along_slit_scene.py
STATE_FIELDS_PRIOR / joint_state.py prior_anchor_density additions).

Renders the dense-truth FPA image once, then for each named prior config
evaluates the forward model AT THE PRIOR ITSELF -- `forward(spec.x0())`,
never `gauss_newton_state` -- window by window, and stitches the results
into a full 1024x1024 predicted image. No retrieval happens here; this is
purely "how wrong is the prior, by itself, before any solve gets a chance
to correct it", which is exactly what needs to be understood before
committing to an expensive full retrieval sweep.

Configs compared:
  bingrid_g<N>_<kind>_ad<d> (--g-ratio-sweep) the G-bin state (g_ratio=<N>,
               prior=truth exactly at each bin) downscaled onto the native
               anchor grid at density <d> via state_interp=<kind> ("linear"
               or "nearest"). Answers "how coarse can the bin grid be, and
               does better reconstruction (linear vs. nearest) recover any
               of that shortfall", at a FIXED (deep) anchor density -- see
               forward_check_g_ratio_sweep_fpa<N>.png.
  nativegrid_ad<d> (--rt-sweep) native_res=True at anchor density <d> -- prior=
               truth exactly, state built DIRECTLY on the anchor grid, no
               G-bin layer at all (d=1 is one exact-truth value per detector
               row). Answers "how far do we need to push RT/state resolution
               before it stops helping", independent of any retrieval, and
               is the ceiling the bingrid sweep should converge onto as its
               g_ratio shrinks -- see forward_check_rt_sweep_fpa<N>.png and
               docs/REALISTIC_PRIOR_EXPERIMENT_CLASS.md.

Run:  PYTHONPATH=. python3 scripts/gd_realistic_prior_forward_check.py
      PYTHONPATH=. python3 scripts/gd_realistic_prior_forward_check.py \\
        --g-ratio-sweep 0.125,0.25,0.5,1,3,6,12 \\
        --g-ratio-sweep-nearest-check 0.125,1,12 --g-ratio-sweep-anchor-density 16
      PYTHONPATH=. python3 scripts/gd_realistic_prior_forward_check.py \\
        --rt-sweep 0.25,0.5,1,2,4,8,16
Output: results/realistic_prior/forward_check/*.npy (truth + one per config)
        plots/realistic_prior/forward_check_*.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers, bin_assign  # noqa: E402
import gd_joint_block_whole_slit_sweep as sw  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.joint_state import build_forward_state, state_spec_from_scene  # noqa: E402

N_COLS = 1024
KM_PER_ROW = 2.0 * als.SLIT_HALF_KM / 1024.0  # ~2.73 km/row on FPA2's 2800km slit

# Approximate row bands read off forward_check_residual_by_row_fpa2.png:
# the two CO2/CO hot-spot locations (sharp oscillating spikes in coarse-prior
# configs) vs. a clearly smooth stretch away from both the hot spots and the
# p_surface "mountain" transition. Illustrative, not precise feature edges.
HOTSPOT_ROW_BANDS = [(95, 125), (885, 925)]
SMOOTH_ROW_BAND = (550, 650)


def _rt_density_from_label(label):
    """Inverse of the rt-sweep config labeling: 'nativegrid_ad<d>' is density
    <d> (including d=1, the former 'highres' point -- folded into this same
    naming 2026-08-19), anything else isn't part of the sweep. 'native_ad<d>'
    (the pre-2026-08-19 name, before "native grid" terminology was settled
    on -- see docs/PROJECT_STATUS.md) is still recognized so existing
    on-disk results from before the rename stay readable."""
    if label.startswith("nativegrid_ad"):
        return float(label[len("nativegrid_ad"):])
    if label.startswith("native_ad"):
        return float(label[len("native_ad"):])
    return None


def render_prior_image(fpa, band, absco, wide_inst, geo, solar, albedo, tiles,
                       g_ratio, fields, prior_anchor_density, label, native_res=False,
                       rt_density=1.0, state_interp="linear"):
    """forward(x0()) window-by-window, stitched into a full 1024x1024 image
    -- reproduces gd_joint_block_whole_slit_sweep.py's own hi-res forward
    setup (native-row anchors) minus the GN solve.

    `native_res=True` builds the STATE ITSELF on the anchor grid (one exact-
    truth value per anchor, not per coarse G bin) -- genuinely more degrees
    of freedom than the usual G-bin state, unlike `prior_anchor_density > 1`
    (which only changes how accurately the EXISTING G bin values are
    computed from `fields`, and so can only ever approach the
    `prior_anchor_density=None` exact-per-bin result from below, never
    exceed it -- there is no way to give a G-bin state more resolution than
    "every one of its G values is already exact"). With `native_res=True`,
    `scene_etas == spec`'s own positions, so `state_interp`'s kind is a
    no-op (same as the coarse solve's own convention: "scene positions ARE
    the state positions").

    `rt_density` sets anchor spacing as a multiple of the native per-row
    grid (1.0 = one anchor per row, matching `--anchor-density 1` in the
    real sweep; >1 oversamples). Only matters when `native_res=True` -- for
    the ordinary G-bin configs the state's own resolution is set by `G`
    (`g_ratio`), and RT already always runs at one-per-row regardless.

    `state_interp` is the `interp1d` kind used to downscale the G-bin state
    onto the native anchor grid for G-bin configs -- `"linear"` (default) or
    `"nearest"` (piecewise-constant, each anchor takes its single nearest
    bin's own value). No exact-truth bypass exists any more; every row,
    free or frozen, always goes through this same interpolation.
    """
    wn_hires, ils = band["wn_hires"], band["ils"]
    spectrum = sw._make_state_spectrum(absco, wide_inst, geo, solar, albedo)
    A = np.full((1024, N_COLS), np.nan)
    t0 = time.time()
    for i, (row_lo, row_hi) in enumerate(tiles):
        rows_win = np.arange(row_lo, row_hi + 1)
        a_lo, a_hi = max(0, row_lo - sw.PAD), min(sw.ROW_MAX_IDX, row_hi + sw.PAD)
        anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / rt_density)
        anchor_etas = np.sort(sw._eta_of(fpa, np.full(len(anchor_rows), 512.0),
                                         anchor_rows.astype(float)))
        if native_res:
            bin_centers = anchor_etas
        else:
            cols = np.arange(float(N_COLS))
            eta_all = np.stack([sw._eta_of(fpa, cols, np.full(N_COLS, float(r)))
                                for r in rows_win])
            G = max(2, int(round(len(rows_win) / g_ratio)))
            bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
        spec = state_spec_from_scene(bin_centers, free=(), fields=fields,
                                     prior_anchor_density=prior_anchor_density)
        fwd = build_forward_state(fpa, rows_win, anchor_etas, spec, spectrum,
                                  wn_hires, ils, pad=sw.PAD, state_interp=state_interp)
        y = fwd(spec.x0())
        A[row_lo:row_hi + 1, :] = y.reshape(len(rows_win), N_COLS)
        elapsed = time.time() - t0
        print(f"  [{label}] {i + 1}/{len(tiles)} rows {row_lo}-{row_hi} "
             f"({elapsed:.0f}s elapsed)", flush=True)
    return A


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpa", type=int, default=2)
    ap.add_argument("--n-lookup-samples", type=int, default=5600,
                    help="along-slit truth-rendering density (default 5600 -- "
                         "0.5km spacing, ~20 samples/hot-spot-FWHM)")
    ap.add_argument("--g-ratio", type=float, default=sw.G_RATIO)
    ap.add_argument("--n-windows", type=int, default=None,
                    help="target window count (default: historical keystone-only "
                         "tiling, 58 on FPA2)")
    ap.add_argument("--g-ratio-sweep", type=str, default="",
                    help="bin-grid convergence-to-truth sweep: comma list of g_ratio "
                         "values (coarse to fine, e.g. '12,6,3,1,0.5,0.25,0.125'), each "
                         "rendered at a FIXED --g-ratio-sweep-anchor-density and "
                         "state_interp='linear' -- so any residual change across the "
                         "sweep is attributable to G alone, not a second moving axis. "
                         "As g_ratio shrinks, G grows toward the anchor count, and the "
                         "sweep should converge onto the already-measured "
                         "nativegrid_ad<N> residual at that same density (its own "
                         "ceiling). Empty (default) skips this sweep entirely.")
    ap.add_argument("--g-ratio-sweep-anchor-density", type=float, default=16.0,
                    help="anchor density held fixed across --g-ratio-sweep (default 16, "
                         "the deepest density already characterized by --rt-sweep, so "
                         "results are directly comparable to that same nativegrid_ad16 "
                         "point).")
    ap.add_argument("--g-ratio-sweep-nearest-check", type=str, default="",
                    help="comma list of a SUBSET of --g-ratio-sweep's own g_ratio values "
                         "to also render with state_interp='nearest' -- a smaller side "
                         "check of whether the linear/nearest gap shrinks as G grows, not "
                         "a full second sweep. Empty (default) skips it.")
    ap.add_argument("--rt-sweep", type=str, default="0.25,0.5,1,2,4,8,16",
                    help="RT-resolution convergence sweep: comma list of anchor "
                         "densities (1.0 = one anchor/row = 2.73km spacing on FPA2), "
                         "each rendered with native_res=True (state built directly on "
                         "the anchor grid, prior=truth exactly, no G-bin layer at all). "
                         "Answers 'how far do we need to push anchor resolution before "
                         "it stops helping', independent of any retrieval, and is the "
                         "ceiling the g_ratio sweep converges onto -- see "
                         "docs/REALISTIC_PRIOR_EXPERIMENT_CLASS.md.")
    ap.add_argument("--no-rt-sweep", dest="rt_sweep", action="store_const", const="")
    ap.add_argument("--anchor-schematic", dest="anchor_schematic", action="store_true",
                    default=True,
                    help="also render a zoomed-in figure showing WHERE each config's "
                         "actual sampling/state points fall (row positions), next to a "
                         "crop of the truth image -- see --schematic-row. Cheap (no RT), "
                         "default on. The displayed row range is derived from the real "
                         "production window containing --schematic-row (+ PAD), not "
                         "requested directly, so it's never mostly-empty.")
    ap.add_argument("--no-anchor-schematic", dest="anchor_schematic", action="store_false")
    ap.add_argument("--schematic-row", type=int, default=110,
                    help="detector row whose containing window --anchor-schematic zooms "
                         "into -- default falls inside the west CO2/CO hot spot (see "
                         "HOTSPOT_ROW_BANDS).")
    ap.add_argument("--bin-sample-counts", dest="bin_sample_counts", action="store_true",
                    default=True,
                    help="also render a figure of real-pixel population per G bin along "
                         "the whole slit, for --bin-count-g-ratios -- answers whether "
                         "mean pixels/bin (~N_COLS*g_ratio) is conserved along the slit "
                         "regardless of window width/keystone. Cheap (no RT), default on.")
    ap.add_argument("--no-bin-sample-counts", dest="bin_sample_counts", action="store_false")
    ap.add_argument("--bin-count-g-ratios", type=str, default="1,3,6",
                    help="comma list of g_ratio values to compare in --bin-sample-counts "
                         "(default matches G_RATIO=3, the production default, plus 1 and "
                         "6 for contrast).")
    ap.add_argument("--locus-detail", dest="locus_detail", action="store_true", default=True,
                    help="also render real-pixel nearest-anchor assignment (green=own row, "
                         "red=borrowed from a neighbouring row) for a column subset at the "
                         "long-wavelength end, where keystone/smile are largest -- see "
                         "--locus-detail-row. Cheap (no RT), default on.")
    ap.add_argument("--no-locus-detail", dest="locus_detail", action="store_false")
    ap.add_argument("--locus-detail-row", type=int, default=950,
                    help="detector row whose containing window --locus-detail zooms into "
                         "(default 950: near a slit edge, where keystone/smile are large "
                         "enough to show real cross-row reassignment -- row 110 or row 50, "
                         "closer to FPA2's keystone-null region, show little to none).")
    ap.add_argument("--bin-loci", dest="bin_loci", action="store_true", default=True,
                    help="also render the true (row, col) locus of every G bin's own "
                         "computed spectrum (one colour per bin, grey elsewhere), plus a "
                         "companion figure with native anchors coloured by nearest bin -- "
                         "see --bin-loci-row/--bin-vs-native-densities. Cheap, default on.")
    ap.add_argument("--no-bin-loci", dest="bin_loci", action="store_false")
    ap.add_argument("--bin-loci-row", type=int, default=110,
                    help="detector row whose containing window --bin-loci zooms into "
                         "(default 110, matching --schematic-row for continuity).")
    ap.add_argument("--bin-vs-native-densities", type=str, default="1,4",
                    help="comma list of native-anchor densities to compare against the bin "
                         "grid in the --bin-loci companion figure (default 1,4).")
    ap.add_argument("--out-root", type=str,
                    default=str(REPO_ROOT / "results" / "realistic_prior" / "forward_check"))
    ap.add_argument("--plot-dir", type=str,
                    default=str(REPO_ROOT / "plots" / "realistic_prior"))
    ap.add_argument("--plot-only", action="store_true",
                    help="skip all rendering -- reload truth_fpa<N>_nls*.npy and "
                         "resid_<label>_fpa<N>.npy from --out-root (as already saved by a "
                         "previous run) and regenerate plots only.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        truth_paths = sorted(out_root.glob(f"truth_fpa{args.fpa}_nls*.npy"))
        if not truth_paths:
            raise SystemExit(f"--plot-only: no truth_fpa{args.fpa}_nls*.npy found in {out_root}")
        truth_A = np.load(truth_paths[-1])
        residual_rms_by_row = {}
        resid_by_label = {}
        for p in sorted(out_root.glob(f"resid_*_fpa{args.fpa}.npy")):
            label = p.stem[len("resid_"):-len(f"_fpa{args.fpa}")]
            resid = np.load(p)
            resid_by_label[label] = resid
            rms_row = np.sqrt(np.nanmean(resid ** 2, axis=1))
            residual_rms_by_row[label] = rms_row / np.nanmean(np.abs(truth_A), axis=1) * 100.0
        _plot(truth_A, residual_rms_by_row, args.fpa, plot_dir)
        _plot_2d(truth_A, resid_by_label, args.fpa, plot_dir)
        rt_densities_all = sorted(_rt_density_from_label(lb) for lb in resid_by_label
                                  if _rt_density_from_label(lb) is not None)
        if rt_densities_all:
            _plot_rt_sweep(residual_rms_by_row, rt_densities_all, args.fpa, plot_dir,
                          snr=gdt.DEFAULT_SNR_BY_FPA.get(args.fpa))
        _plot_g_ratio_sweep(residual_rms_by_row, args.fpa, plot_dir,
                           snr=gdt.DEFAULT_SNR_BY_FPA.get(args.fpa))
        if args.anchor_schematic:
            _plot_anchor_schematic(truth_A, args.fpa, args.schematic_row, args.g_ratio, plot_dir)
        if args.bin_sample_counts:
            g_ratios = [float(x) for x in args.bin_count_g_ratios.split(",")]
            _plot_bin_sample_counts(args.fpa, g_ratios, plot_dir)
        if args.locus_detail:
            _plot_locus_detail(truth_A, args.fpa, args.locus_detail_row, plot_dir=plot_dir)
        if args.bin_loci:
            densities = [float(x) for x in args.bin_vs_native_densities.split(",")]
            _plot_bin_spectrum_loci(args.fpa, args.bin_loci_row, args.g_ratio, plot_dir)
            _plot_bin_vs_native_map(args.fpa, args.bin_loci_row, args.g_ratio, densities, plot_dir)
        return 0

    print(f"Building realistic-scene FPA{args.fpa} truth band "
         f"(n_lookup_samples={args.n_lookup_samples})...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[args.fpa]
    t0 = time.time()
    band = gdt._band_setup(args.fpa, atm_center, absco, geo, solar, snr,
                           args.n_lookup_samples, None, False, False, 32, False, 0, False)
    wide_win, wide_inst, albedo = band_basics(args.fpa, atm_center, absco, geo, solar)
    truth_A = np.asarray(band["A"], dtype=float)
    print(f"truth done in {time.time() - t0:.0f}s -- A {truth_A.shape}, "
         f"range [{truth_A.min():.4g}, {truth_A.max():.4g}] W/m2/sr/um", flush=True)
    np.save(out_root / f"truth_fpa{args.fpa}_nls{args.n_lookup_samples}.npy", truth_A)

    if args.n_windows is not None:
        window_scale = sw.scale_for_window_count(args.fpa, args.n_windows)
    else:
        window_scale = 1.0
    tiles = sw.build_window_tiles(args.fpa, window_scale=window_scale)
    print(f"{len(tiles)} windows, g_ratio={args.g_ratio:g}\n", flush=True)

    configs = []
    rt_sweep_densities = ([] if not args.rt_sweep else
                          [float(d) for d in args.rt_sweep.split(",")])
    for d in rt_sweep_densities:
        configs.append((f"nativegrid_ad{d:g}", als.STATE_FIELDS, None, True, d, "linear",
                        args.g_ratio))

    # bin-grid convergence-to-truth sweep: G varied broadly at FIXED anchor
    # density and (primarily) fixed downscaling kind, so any change in
    # residual is attributable to G alone -- see the conversation this was
    # designed in. state_interp="nearest" gets a smaller side-check at a
    # subset of the same g_ratio values, not a full second sweep.
    if args.g_ratio_sweep:
        g_ratios = [float(g) for g in args.g_ratio_sweep.split(",")]
        ad = args.g_ratio_sweep_anchor_density
        for g in g_ratios:
            configs.append((f"bingrid_g{g:g}_linear_ad{ad:g}", als.STATE_FIELDS, None, False,
                            ad, "linear", g))
        if args.g_ratio_sweep_nearest_check:
            check_ratios = [float(g) for g in args.g_ratio_sweep_nearest_check.split(",")]
            for g in check_ratios:
                configs.append((f"bingrid_g{g:g}_nearest_ad{ad:g}", als.STATE_FIELDS, None, False,
                                ad, "nearest", g))

    residual_rms_by_row = {}
    resid_by_label = {}
    for label, fields, density, native_res, rt_density, state_interp, g_ratio in configs:
        print(f"rendering prior config '{label}' "
             f"(fields={'STATE_FIELDS_PRIOR' if fields is als.STATE_FIELDS_PRIOR else 'STATE_FIELDS'}, "
             f"prior_anchor_density={density}, native_res={native_res}, "
             f"rt_density={rt_density}, state_interp={state_interp!r}, "
             f"g_ratio={g_ratio:g})...", flush=True)
        A = render_prior_image(args.fpa, band, absco, wide_inst, geo, solar, albedo,
                               tiles, g_ratio, fields, density, label,
                               native_res=native_res, rt_density=rt_density,
                               state_interp=state_interp)
        np.save(out_root / f"prior_{label}_fpa{args.fpa}.npy", A)
        resid = truth_A - A
        np.save(out_root / f"resid_{label}_fpa{args.fpa}.npy", resid)
        resid_by_label[label] = resid
        rms_row = np.sqrt(np.nanmean(resid ** 2, axis=1))
        pct_row = rms_row / np.nanmean(np.abs(truth_A), axis=1) * 100.0
        residual_rms_by_row[label] = pct_row
        print(f"  '{label}': residRMS/mean|A| median {np.nanmedian(pct_row):.3f}%, "
             f"max {np.nanmax(pct_row):.3f}%\n", flush=True)

    _plot(truth_A, residual_rms_by_row, args.fpa, plot_dir)
    _plot_2d(truth_A, resid_by_label, args.fpa, plot_dir)
    rt_densities_all = sorted(set(rt_sweep_densities))
    if rt_densities_all:
        _plot_rt_sweep(residual_rms_by_row, rt_densities_all, args.fpa, plot_dir,
                      snr=gdt.DEFAULT_SNR_BY_FPA.get(args.fpa))
    _plot_g_ratio_sweep(residual_rms_by_row, args.fpa, plot_dir,
                       snr=gdt.DEFAULT_SNR_BY_FPA.get(args.fpa))
    if args.anchor_schematic:
        _plot_anchor_schematic(truth_A, args.fpa, args.schematic_row, args.g_ratio, plot_dir)
    if args.bin_sample_counts:
        g_ratios = [float(x) for x in args.bin_count_g_ratios.split(",")]
        _plot_bin_sample_counts(args.fpa, g_ratios, plot_dir)
    if args.locus_detail:
        _plot_locus_detail(truth_A, args.fpa, args.locus_detail_row, plot_dir=plot_dir)
    if args.bin_loci:
        densities = [float(x) for x in args.bin_vs_native_densities.split(",")]
        _plot_bin_spectrum_loci(args.fpa, args.bin_loci_row, args.g_ratio, plot_dir)
        _plot_bin_vs_native_map(args.fpa, args.bin_loci_row, args.g_ratio, densities, plot_dir)
    return 0


def _plot(truth_A, residual_rms_by_row, fpa, plot_dir):
    """One curve per config. Colour is grouped by MECHANISM, not just an
    arbitrary per-line cycle -- with >10 configs the default matplotlib
    cycle wraps and reuses colours across genuinely different families,
    which is actively misleading here since the two families have a real,
    different shape: bingrid_* (G-bin state, piecewise-reconstructed onto
    the anchor grid) rings/oscillates from the coarse interpolation between
    bin centres; nativegrid_ad<d>/native_ad<d> (state built directly on a
    fine anchor grid, no bin layer) are much smoother. Blues = bingrid
    family (solid=linear downscale, dashed=nearest, darker=smaller g_ratio
    i.e. finer/more bins); oranges = native family (dotted, darker=denser
    anchor grid). Anything else falls back to the default cycle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    rows = np.arange(truth_A.shape[0])

    bingrid, native, other = {}, {}, []
    for label in residual_rms_by_row:
        parsed = _bingrid_config_from_label(label)
        if parsed is not None:
            g_ratio, kind, _density = parsed
            bingrid[label] = (g_ratio, kind)
            continue
        d = _rt_density_from_label(label)
        if d is not None:
            native[label] = d
            continue
        other.append(label)

    def _log_frac(x, lo, hi):
        return 0.0 if hi == lo else (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))

    if bingrid:
        g_ratios = [g for g, _ in bingrid.values()]
        g_lo, g_hi = min(g_ratios), max(g_ratios)
        for label, (g_ratio, kind) in sorted(bingrid.items(), key=lambda kv: -kv[1][0]):
            # smaller g_ratio (more bins, finer) -> darker
            shade = 0.85 - 0.55 * (1.0 - _log_frac(g_ratio, g_lo, g_hi))
            ax.plot(rows, residual_rms_by_row[label], color=plt.cm.Blues(shade),
                    ls="-" if kind == "linear" else "--", lw=1.1, label=label)

    if native:
        densities = list(native.values())
        d_lo, d_hi = min(densities), max(densities)
        for label, d in sorted(native.items(), key=lambda kv: kv[1]):
            # denser anchor grid -> darker
            shade = 0.3 + 0.6 * _log_frac(d, d_lo, d_hi)
            ax.plot(rows, residual_rms_by_row[label], color=plt.cm.Oranges(shade),
                    ls=":", lw=1.3, label=label)

    for label in other:
        ax.plot(rows, residual_rms_by_row[label], lw=1.2, label=label)

    ax.set_xlabel("detector row")
    ax.set_ylabel("residRMS / mean|truth| (%)")
    ax.set_yscale("log")
    ax.set_title(f"Prior-only forward residual by row, FPA{fpa} (no solve)\n"
                "blue = bingrid (solid=linear, dashed=nearest downscale, darker=finer g_ratio)"
                "  |  orange dotted = nativegrid/native (darker=denser anchor grid)",
                fontsize=10.5)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = plot_dir / f"forward_check_residual_by_row_fpa{fpa}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _plot_rt_sweep(residual_rms_by_row, densities, fpa, plot_dir, snr=None):
    """RT-resolution convergence: %residual vs. anchor spacing, for the
    native_res sweep (prior=truth exactly, state built directly on the
    anchor grid -- see render_prior_image's own docstring). Three summary
    curves per density: global median, median within the two known hot-spot
    row bands, median within a smooth reference band.

    There is NO plateau in this error vs. resolution curve within any
    practically reachable density -- it is first-order in anchor spacing
    (`nearest_bin_scene`'s own hard pixel-to-anchor assignment, per its own
    docstring, plus keystone spreading a single row's pixels across
    multiple true eta positions column-to-column -- neither is smoothed
    away by the row-mixing PSF, which operates on a different axis). So
    "diminishing returns" isn't a kink to find in this curve; it's whatever
    density first drops representation error safely below the instrument's
    OWN noise floor (1/SNR), since finer resolution than that cannot be
    distinguished from noise in a real fit. `snr` (e.g.
    `gd_test.DEFAULT_SNR_BY_FPA[fpa]`) draws that floor as a reference line
    when given.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spacing_km, med_all, max_all, med_hot, med_smooth = [], [], [], [], []
    for d in densities:
        label = f"nativegrid_ad{d:g}"
        if label not in residual_rms_by_row:
            label = f"native_ad{d:g}"  # pre-rename on-disk data, still readable
        if label not in residual_rms_by_row:
            continue
        pct_row = residual_rms_by_row[label]
        spacing_km.append(KM_PER_ROW / d)
        med_all.append(np.nanmedian(pct_row))
        max_all.append(np.nanmax(pct_row))
        hot_rows = np.concatenate([np.arange(lo, hi + 1) for lo, hi in HOTSPOT_ROW_BANDS])
        med_hot.append(np.nanmedian(pct_row[hot_rows]))
        lo, hi = SMOOTH_ROW_BAND
        med_smooth.append(np.nanmedian(pct_row[lo:hi + 1]))

    if not spacing_km:
        print("_plot_rt_sweep: no matching densities found, skipping")
        return

    order = np.argsort(spacing_km)[::-1]  # coarse (large spacing) first
    spacing_km = np.array(spacing_km)[order]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(spacing_km, np.array(med_all)[order], "o-", label="global median", lw=1.6)
    ax.plot(spacing_km, np.array(max_all)[order], "o--", label="global max", lw=1.2, alpha=0.6)
    ax.plot(spacing_km, np.array(med_hot)[order], "s-",
           label=f"hot-spot rows median {HOTSPOT_ROW_BANDS}", lw=1.6)
    ax.plot(spacing_km, np.array(med_smooth)[order], "^-",
           label=f"smooth rows median {SMOOTH_ROW_BAND}", lw=1.6)
    if snr:
        ax.axhline(100.0 / snr, color="0.3", ls=":", lw=1.4,
                  label=f"instrument noise floor (1/SNR={snr:g})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # finer resolution (smaller spacing) to the right
    ax.set_xlabel("anchor spacing (km, log scale -- finer resolution to the right)")
    ax.set_ylabel("residRMS / mean|truth| (%)")
    ax.set_title(f"RT-resolution convergence, FPA{fpa} (prior=truth exactly, no solve)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = plot_dir / f"forward_check_rt_sweep_fpa{fpa}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _bingrid_config_from_label(label):
    """Parse 'bingrid_g<ratio>_<linear|nearest>_ad<density>' back into
    (g_ratio, interp_kind, anchor_density), or None if `label` isn't one of
    these g_ratio-sweep configs."""
    import re
    m = re.match(r"^bingrid_g([0-9.]+)_(linear|nearest)_ad([0-9.]+)$", label)
    if not m:
        return None
    return float(m.group(1)), m.group(2), float(m.group(3))


def _plot_g_ratio_sweep(residual_rms_by_row, fpa, plot_dir, snr=None):
    """Bin-grid convergence-to-truth: residual vs. g_ratio (equivalently,
    vs. G, since G = width/g_ratio) at whatever anchor density(ies) the
    'bingrid_g<ratio>_<kind>_ad<density>' labels in `residual_rms_by_row`
    were rendered at. One line per (interp kind, anchor density) found.
    Overlays the matching native-grid ceiling (`nativegrid_ad<density>`) at
    each density present, as a horizontal reference -- what the bin-grid
    sweep should converge onto as g_ratio shrinks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_group = {}  # (kind, density) -> [(g_ratio, med, mx), ...]
    for label, pct_row in residual_rms_by_row.items():
        parsed = _bingrid_config_from_label(label)
        if parsed is None:
            continue
        g_ratio, kind, density = parsed
        by_group.setdefault((kind, density), []).append(
            (g_ratio, float(np.nanmedian(pct_row)), float(np.nanmax(pct_row))))

    if not by_group:
        print("_plot_g_ratio_sweep: no bingrid_g<ratio>_<kind>_ad<density> labels found, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"linear": "C0", "nearest": "C1"}
    for (kind, density), rows in sorted(by_group.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.sort(key=lambda r: -r[0])  # coarse (large g_ratio) first
        g_ratios = [r[0] for r in rows]
        meds = [r[1] for r in rows]
        maxs = [r[2] for r in rows]
        c = colors.get(kind, None)
        ax.plot(g_ratios, meds, "o-", color=c, label=f"{kind} ad{density:g}: median", lw=1.6)
        ax.plot(g_ratios, maxs, "o--", color=c, alpha=0.6, label=f"{kind} ad{density:g}: max", lw=1.2)

        ceiling_label = f"nativegrid_ad{density:g}"
        if ceiling_label not in residual_rms_by_row:
            ceiling_label = f"native_ad{density:g}"  # pre-rename on-disk fallback
        if ceiling_label in residual_rms_by_row:
            ceil_pct = residual_rms_by_row[ceiling_label]
            ax.axhline(float(np.nanmax(ceil_pct)), color=c, ls=":", lw=1.2, alpha=0.8,
                      label=f"{ceiling_label} (native-grid ceiling): max")

    if snr:
        ax.axhline(100.0 / snr, color="0.3", ls=":", lw=1.4,
                  label=f"instrument noise floor (1/SNR={snr:g})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # smaller g_ratio (more bins, finer) to the right
    ax.set_xlabel("g_ratio (log scale -- more bins / finer to the right)")
    ax.set_ylabel("residRMS / mean|truth| (%)")
    ax.set_title(f"Bin-grid convergence to truth, FPA{fpa} (prior=truth exactly, no solve)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = plot_dir / f"forward_check_g_ratio_sweep_fpa{fpa}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _find_window(fpa, target_row, window_scale=1.0):
    """The real production window (as gd_joint_block_whole_slit_sweep.py's
    own tiling would build it) containing `target_row`, so a schematic uses
    the same G/PAD/anchor-grid extents a real run actually would."""
    for lo, hi in sw.build_window_tiles(fpa, window_scale=window_scale):
        if lo <= target_row <= hi:
            return lo, hi
    raise ValueError(f"no window contains row {target_row}")


def _bin_sample_counts(fpa, g_ratio, window_scale=1.0):
    """For every production window: G (pixel_density_bin_centers's bin
    count), and each of those G bins' actual real-pixel population (every
    (row, col) pixel in the window assigned to its nearest bin, same
    nearest-neighbor rule `nearest_bin_scene` uses at render time) -- no RT,
    cheap. `G = max(2, round(width/g_ratio))` and total pixels in a window
    = width*N_COLS, so mean pixels/bin = width*N_COLS/G =~ N_COLS*g_ratio,
    independent of width -- the same g_ratio should give the same average
    bin population everywhere along the slit regardless of how keystone
    varies window to window; this is what this function measures directly
    rather than assumes.

    Returns list of dicts, one per window: row_lo, row_hi, width, G, counts.
    """
    out = []
    for row_lo, row_hi in sw.build_window_tiles(fpa, window_scale=window_scale):
        rows_win = np.arange(row_lo, row_hi + 1)
        cols = np.arange(float(N_COLS))
        eta_all = np.stack([sw._eta_of(fpa, cols, np.full(N_COLS, float(r)))
                            for r in rows_win])
        width = len(rows_win)
        G = max(2, int(round(width / g_ratio)))
        bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
        idx = bin_assign(bin_centers, eta_all.ravel())
        counts = np.bincount(idx, minlength=G)
        out.append(dict(row_lo=row_lo, row_hi=row_hi, width=width, G=G, counts=counts))
    return out


def _plot_bin_sample_counts(fpa, g_ratios, plot_dir):
    """Two panels along the whole slit: how G (bin count) tracks window
    width/keystone, and whether mean pixels/bin stays flat at N_COLS*g_ratio
    regardless -- i.e. whether that product really is conserved along the
    slit, for each g_ratio, rather than only in one window's own arithmetic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_g, ax_n) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, g_ratio in enumerate(g_ratios):
        data = _bin_sample_counts(fpa, g_ratio)
        rows_c = np.array([0.5 * (d["row_lo"] + d["row_hi"]) for d in data])
        Gs = np.array([d["G"] for d in data])
        color = f"C{i}"
        ax_g.plot(rows_c, Gs, "o-", ms=3, lw=1, color=color, label=f"g_ratio={g_ratio:g}")

        xs = np.concatenate([np.full(d["G"], 0.5 * (d["row_lo"] + d["row_hi"])) for d in data])
        ys = np.concatenate([d["counts"] for d in data])
        ax_n.scatter(xs, ys, s=5, alpha=0.35, color=color, label=f"g_ratio={g_ratio:g} (per bin)")
        theory = N_COLS * g_ratio
        ax_n.axhline(theory, color=color, ls="--", lw=1.3,
                    label=f"g_ratio={g_ratio:g}: N_COLS*g_ratio={theory:g}")

    ax_g.set_ylabel("G (bins per window)")
    ax_g.set_title("Bin count G tracks window width/keystone along the slit", fontsize=10.5)
    ax_g.legend(fontsize=8, ncol=len(g_ratios))
    ax_g.grid(alpha=0.3)

    ax_n.set_ylabel("real pixels assigned to each bin")
    ax_n.set_xlabel("detector row (window centre)")
    ax_n.set_title("Per-bin pixel population -- scatter = every individual bin, "
                   "dashed = N_COLS*g_ratio", fontsize=10.5)
    ax_n.grid(alpha=0.3)
    handles, labels = ax_n.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_n.legend(by_label.values(), by_label.keys(), fontsize=7.5, ncol=len(g_ratios))

    fig.suptitle(f"FPA{fpa}: is mean pixels/bin conserved along the slit? "
                f"(pixel_density_bin_centers, {len(sw.build_window_tiles(fpa))} windows)",
                fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = plot_dir / f"forward_check_bin_sample_counts_fpa{fpa}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _plot_locus_detail(truth_A, fpa, target_row, n_cols_shown=18, plot_dir=None):
    """Per-pixel nearest-anchor assignment, real (row, col) pixels only, for
    a subset of columns at the long-wavelength end -- makes concrete what
    the previous schematic's flat row-lines glossed over: a fixed anchor's
    OWN eta (evaluated once, at col=512) is only the truth at that single
    reference pixel. At every OTHER column, `nearest_bin_scene` assigns
    whichever anchor is closest to THIS pixel's own true eta (from the real
    2D keystone/smile geometry, `sw._eta_of`) -- which is not always the
    anchor sharing this pixel's own row index, once keystone/smile shift the
    row<->eta mapping enough at that column.

    Marker color: green = this pixel's nearest anchor IS its own row (the
    naive no-keystone expectation holds here); orange/red = it's a
    NEIGHBOURING row's anchor instead -- that anchor's spectrum, computed
    for a position `abs(assigned_row - row)` rows away, is what this pixel
    actually renders with. Marker size scales with that offset.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_lo, row_hi = _find_window(fpa, target_row)
    a_lo, a_hi = max(0, row_lo - sw.PAD), min(sw.ROW_MAX_IDX, row_hi + sw.PAD)
    anchor_rows = np.arange(a_lo, a_hi + 1)
    anchor_etas = sw._eta_of(fpa, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float))
    order = np.argsort(anchor_etas)
    anchor_rows_sorted = anchor_rows[order]
    anchor_etas_sorted = anchor_etas[order]
    edges = 0.5 * (anchor_etas_sorted[:-1] + anchor_etas_sorted[1:])

    cols_shown = np.linspace(N_COLS - 150, N_COLS - 1, n_cols_shown).round().astype(int)
    cols_shown = np.unique(cols_shown)

    fig, ax = plt.subplots(figsize=(10, 8))
    if truth_A is not None:
        crop = truth_A[a_lo:a_hi + 1, cols_shown.min():cols_shown.max() + 1]
        ax.imshow(crop, aspect="auto", cmap="gray", origin="upper",
                 extent=[cols_shown.min(), cols_shown.max() + 1, a_hi + 1, a_lo], alpha=0.6)

    n_same, n_diff = 0, 0
    for c in cols_shown:
        etas = sw._eta_of(fpa, np.full(len(anchor_rows), float(c)), anchor_rows.astype(float))
        idx = np.searchsorted(edges, etas)
        assigned_row = anchor_rows_sorted[idx]
        offset = assigned_row - anchor_rows
        same = offset == 0
        n_same += int(same.sum())
        n_diff += int((~same).sum())
        ax.scatter(np.full(same.sum(), c), anchor_rows[same], color="tab:green",
                  s=14, zorder=3, marker="s")
        if (~same).any():
            sizes = 14 + 10 * np.abs(offset[~same])
            ax.scatter(np.full((~same).sum(), c), anchor_rows[~same], color="tab:red",
                      s=sizes, zorder=3, marker="s", alpha=0.85)

    ax.set_xlabel("detector column (long-wavelength end)")
    ax.set_ylabel("detector row")
    ax.set_ylim(a_hi + 1, a_lo)
    frac_diff = 100.0 * n_diff / max(1, n_same + n_diff)
    ax.set_title(f"FPA{fpa} window {row_lo}-{row_hi}: which anchor each real pixel actually "
                f"renders with\n(green = own-row anchor, red = a neighbouring row's anchor "
                f"instead, size ~ offset -- {frac_diff:.0f}% reassigned here)", fontsize=11)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:green", markersize=9,
              label="nearest anchor = this pixel's own row"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:red", markersize=9,
              label="nearest anchor = a NEIGHBOURING row (spectrum borrowed)"),
    ]
    ax.legend(handles=legend_elems, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out = plot_dir / f"forward_check_locus_detail_fpa{fpa}_row{target_row}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _add_pixel_row_grid(ax, row_lo, row_hi):
    """Overlay real detector pixel ROW boundaries (integer +/- 0.5) as thin
    reference lines -- for the schematic figures only (bin loci / bin-vs-
    native), never the residual/accuracy plots. Column boundaries need no
    equivalent: every schematic here already evaluates every real column
    (`cols = np.arange(N_COLS)`), so the column axis is already at native
    detector resolution -- 1 column of data per real pixel column,
    regardless of the anchor density being shown. Only the ROW axis can be
    finer than a real pixel (native_res anchors at density>1 sit at
    fractional rows), so only rows need this overlay.
    """
    for r in np.arange(np.floor(row_lo) - 0.5, np.ceil(row_hi) + 1.5, 1.0):
        ax.axhline(r, color="k", lw=0.4, alpha=0.35, zorder=5)


def _bin_loci(fpa, bin_centers, cols, search_rows):
    """For each bin centre and each column, the single discrete row (from
    `search_rows`) whose true eta is closest to that bin's own eta -- the
    literal (row, col) pixel where that bin's one computed spectrum exactly
    represents the true physical state at that column, up to pixel
    discretisation (this is the same constant-eta locus traced numerically
    earlier in this conversation, now for every bin and every column).
    Returns an array shape (len(bin_centers), len(cols)) of row values.
    """
    out = np.empty((len(bin_centers), len(cols)), dtype=float)
    for ci, c in enumerate(cols):
        etas = sw._eta_of(fpa, np.full(len(search_rows), float(c)), search_rows.astype(float))
        for bi, bc in enumerate(bin_centers):
            out[bi, ci] = search_rows[np.argmin(np.abs(etas - bc))]
    return out


def _bin_setup(fpa, target_row, g_ratio):
    """Shared setup for the two figures below: the real window, its G bin
    centres, and a comfortably wide row-search range for tracing loci."""
    row_lo, row_hi = _find_window(fpa, target_row)
    rows_win = np.arange(row_lo, row_hi + 1)
    cols_all = np.arange(float(N_COLS))
    eta_all = np.stack([sw._eta_of(fpa, cols_all, np.full(N_COLS, float(r)))
                        for r in rows_win])
    G = max(2, int(round(len(rows_win) / g_ratio)))
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    a_lo, a_hi = max(0, row_lo - sw.PAD), min(sw.ROW_MAX_IDX, row_hi + sw.PAD)
    search_rows = np.arange(max(0, a_lo - sw.PAD), min(sw.ROW_MAX_IDX, a_hi + sw.PAD) + 1)
    return row_lo, row_hi, a_lo, a_hi, bin_centers, search_rows


def _plot_bin_spectrum_loci(fpa, target_row, g_ratio, plot_dir):
    """Figure 1: for the G bins in the window containing `target_row`, the
    actual (row, col) pixels where each bin's ONE computed spectrum exactly
    represents the true state -- one distinctly-coloured curve per bin,
    traced across every column, on a plain grey background (everything
    else). This is the full, true version of the flat row-lines the first
    schematic (`_plot_anchor_schematic`) approximated with a single
    reference column -- see the constant-eta-locus check earlier in this
    conversation for the numbers behind why it curves.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_lo, row_hi, a_lo, a_hi, bin_centers, search_rows = _bin_setup(fpa, target_row, g_ratio)
    cols = np.arange(N_COLS)
    loci = _bin_loci(fpa, bin_centers, cols, search_rows)

    colors = [plt.cm.tab10(i % 10) for i in range(len(bin_centers))]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("0.85")
    for bi in range(len(bin_centers)):
        # connected line, not scatter -- avoids marker-size gap artifacts
        # (see the bin-vs-native figure's own note on this)
        ax.plot(cols, loci[bi], color=colors[bi], lw=2.2,
               label=f"bin {bi} (row~{np.median(loci[bi]):.0f} at col 512)")
    ax.set_xlim(0, N_COLS)
    ax.set_ylim(loci.max() + 3, loci.min() - 3)
    _add_pixel_row_grid(ax, int(np.floor(loci.min())) - 1, int(np.ceil(loci.max())) + 1)
    ax.set_xlabel("detector column")
    ax.set_ylabel("detector row")
    ax.set_title(f"FPA{fpa} window {row_lo}-{row_hi}: where each G bin's ONE computed "
                f"spectrum is exactly true (grey = every other pixel)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", markerscale=2.5)
    fig.tight_layout()
    out = plot_dir / f"forward_check_bin_loci_fpa{fpa}_row{target_row}_gratio{g_ratio:g}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _plot_bin_vs_native_map(fpa, target_row, g_ratio, densities, plot_dir):
    """Figure 2: the same G bin loci as light grey reference outlines, with
    the native/anchor grid for a couple of densities drawn on top -- each
    anchor row coloured by whichever bin its TRUE eta is nearest to AT EACH
    COLUMN (not a single col=512 snapshot -- an anchor's own row is fixed,
    but the real pixel at (anchor_row, col) has a different true eta at
    every column, exactly like the bin loci in figure 1, so its nearest
    bin can and does change across the row). Found 2026-08-19 (user): an
    earlier version coloured each whole anchor row ONE colour from its
    col=512 eta alone, which stayed flat regardless of how much the true
    bin boundary moved underneath it across columns -- wrong, since a real
    retrieval's own per-pixel rendering (`nearest_bin_scene`) uses each
    pixel's own true eta, not a col=512 stand-in.

    Answers directly whether native anchors are a 'downscaled' version of
    the bin grid: same-coloured anchors clustering tightly within their
    bin's own grey outline, WITH the same colour transitions as figure 1's
    staircase, says yes; anything flatter than figure 1's own staircase
    means this figure -- like the row-only schematic before it -- is still
    hiding real column dependence.

    Filled with `pcolormesh` over the full (row, col) grid at each density's
    own resolution, not scatter markers -- found 2026-08-19 (user): a
    monotonic eta(row) at fixed column (verified directly -- see the
    conversation) guarantees the true bin assignment has NO holes at any
    column, but small scatter markers can visually fake gaps between
    same-colour neighbours purely from marker-size/spacing at render time.
    `pcolormesh` fills every grid cell exactly once, so it can't misrepresent
    contiguity the way marker scatter can. The bin loci overlay is now a
    connected `plot` line for the same reason, not scatter points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    row_lo, row_hi, a_lo, a_hi, bin_centers, search_rows = _bin_setup(fpa, target_row, g_ratio)
    cols = np.arange(N_COLS)
    loci = _bin_loci(fpa, bin_centers, cols, search_rows)
    edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
    G = len(bin_centers)
    bin_colors = [plt.cm.tab10(i % 10) for i in range(G)]
    cmap = ListedColormap(bin_colors)
    norm = BoundaryNorm(np.arange(G + 1) - 0.5, cmap.N)

    fig, axes = plt.subplots(1, len(densities), figsize=(6.5 * len(densities), 7),
                             sharey=True)
    if len(densities) == 1:
        axes = [axes]

    col_edges = np.arange(N_COLS + 1) - 0.5

    for ax, density in zip(axes, densities):
        anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / density)
        step = 0.5 / density

        # true eta of the REAL pixel at (row=ar, col=c), every column -- not
        # a single col=512 stand-in -- for every anchor row at this density
        bin_map = np.empty((len(anchor_rows), N_COLS), dtype=int)
        for i, ar in enumerate(anchor_rows):
            etas = sw._eta_of(fpa, cols.astype(float), np.full(N_COLS, ar))
            bin_map[i] = np.searchsorted(edges, etas)

        row_edges = np.concatenate([anchor_rows - step, [anchor_rows[-1] + step]])
        ax.pcolormesh(col_edges, row_edges, bin_map, cmap=cmap, norm=norm,
                     shading="flat")

        # bin loci overlay, on top, as a connected line (not scatter) so it
        # reads as continuous regardless of marker size
        for bi in range(G):
            ax.plot(cols, loci[bi], color="0.3", lw=1.1, alpha=0.9)

        _add_pixel_row_grid(ax, a_lo, a_hi)

        label = f"nativegrid_ad{density:g}"
        ax.set_xlim(0, N_COLS)
        ax.set_xlabel("detector column")
        ax.set_title(f"{label}\n({len(anchor_rows)} anchors = {density:g} pts/pixel row, "
                    f"coloured by nearest bin AT EACH COLUMN)", fontsize=10.5)

    axes[0].set_ylabel("detector row")
    axes[0].set_ylim(loci.max() + 3, loci.min() - 3)
    fig.suptitle(f"FPA{fpa} window {row_lo}-{row_hi}: native anchors coloured by the bin "
                f"a retrieval would draw them from (grey lines = bin loci from fig. 1)",
                fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = plot_dir / f"forward_check_bin_vs_native_fpa{fpa}_row{target_row}_gratio{g_ratio:g}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


#: (short lane label, long description) -- the long form goes in the
#: figure's caption text, not the lane title, to keep lanes narrow.
_SCHEMATIC_LANES_LEGEND = [
    ("G bins", "the bingrid_g<N>_<kind>_ad<d> configs all solve for values at "
              "these G bin positions (g_ratio sets G); they differ only in "
              "how each bin's own value gets downscaled onto the native "
              "anchor grid (state_interp kind), not in how many bins exist "
              "or where they sit."),
    ("nativegrid_adD", "no G-bin layer at all -- the STATE ITSELF sits at these "
                       "positions, D anchors per detector row (D=1 is one "
                       "exact-truth value per detector row)."),
]


def _sampling_positions(fpa, target_row, g_ratio, display_pad=6,
                        rt_densities=(0.25, 1.0, 2.0, 4.0, 8.0, 16.0)):
    """Row-space positions of every config's actual sampling/anchor points,
    for the real production window containing `target_row` -- no RT, cheap
    (positions only). The display range is DERIVED from that window's own
    extent (+ PAD + `display_pad`), not requested directly, so the figure is
    never mostly-empty regardless of how narrow or wide that window is.

    Returns (row_lo, row_hi, disp_lo, disp_hi, {label: row_positions_array}).
    """
    row_lo, row_hi = _find_window(fpa, target_row)
    rows_win = np.arange(row_lo, row_hi + 1)
    cols = np.arange(float(N_COLS))
    eta_all = np.stack([sw._eta_of(fpa, cols, np.full(N_COLS, float(r))) for r in rows_win])
    G = max(2, int(round(len(rows_win) / g_ratio)))
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)

    a_lo, a_hi = max(0, row_lo - sw.PAD), min(sw.ROW_MAX_IDX, row_hi + sw.PAD)
    disp_lo, disp_hi = max(0, a_lo - display_pad), min(sw.ROW_MAX_IDX, a_hi + display_pad)

    # row<->eta reference at col=512 (same reference column render_prior_image
    # itself uses to build anchor_etas), spanning past the display range for
    # safe interpolation. Bin centers mix ALL columns' etas (pixel_density_
    # bin_centers), so mapping them back to a single "row" via this one-
    # column reference is an approximation -- fine for a schematic, not exact.
    row_ref = np.arange(max(0, disp_lo - 2), min(sw.ROW_MAX_IDX, disp_hi + 2) + 1, dtype=float)
    eta_ref = sw._eta_of(fpa, np.full(len(row_ref), 512.0), row_ref)
    order = np.argsort(eta_ref)

    def row_of_eta(eta):
        return np.interp(eta, eta_ref[order], row_ref[order])

    def clip(rows):
        rows = np.asarray(rows, dtype=float)
        return rows[(rows >= disp_lo) & (rows <= disp_hi)]

    positions = {"G bins": clip(row_of_eta(bin_centers))}

    for d in rt_densities:
        anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / d)
        positions[f"nativegrid_ad{d:g}"] = clip(anchor_rows)

    return row_lo, row_hi, disp_lo, disp_hi, positions


def _plot_anchor_schematic(truth_A, fpa, target_row, g_ratio, plot_dir):
    """Zoomed-in view of where every config's actual sampling/state points
    fall, lane by lane, next to a crop of the truth image for context --
    answers 'where are the anchors' directly rather than by description."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_lo, row_hi, disp_lo, disp_hi, positions = \
        _sampling_positions(fpa, target_row, g_ratio)

    n_lanes = len(positions)
    fig, axes = plt.subplots(1, 1 + n_lanes, figsize=(2.6 + 1.05 * n_lanes, 7.5),
                             sharey=True, gridspec_kw=dict(wspace=0.08))

    ax0 = axes[0]
    if truth_A is not None:
        crop = truth_A[disp_lo:disp_hi + 1, :]
        ax0.imshow(crop, aspect="auto", cmap="gray", origin="upper",
                  extent=[0, N_COLS, disp_hi + 1, disp_lo])
    ax0.set_ylabel("detector row")
    ax0.set_xlabel("column")
    ax0.set_title("truth", fontsize=9)

    for ax, (label, rows) in zip(axes[1:], positions.items()):
        for r in rows:
            ax.axhline(r, color="C0", lw=1.1, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(f"{label}\n(n={len(rows)})", fontsize=8.5)
        ax.grid(axis="y", alpha=0.15)
    for lo, hi in (HOTSPOT_ROW_BANDS if fpa == 2 else []):
        if lo <= disp_hi and hi >= disp_lo:
            for ax in axes:
                ax.axhspan(max(lo, disp_lo), min(hi, disp_hi), color="orange", alpha=0.08, zorder=0)

    axes[0].set_ylim(disp_hi + 1, disp_lo)
    caption = "  |  ".join(f"{short}: {long}" for short, long in _SCHEMATIC_LANES_LEGEND)
    fig.suptitle(f"FPA{fpa}: where each config's sampling points actually fall "
                f"(window {row_lo}-{row_hi}, PAD={sw.PAD} -> shown {disp_lo}-{disp_hi})",
                fontsize=12, y=0.99)
    fig.text(0.5, 0.005, "\n".join(_wrap(caption, 130)), ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out = plot_dir / f"forward_check_anchor_schematic_fpa{fpa}_row{target_row}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width)


def _plot_2d(truth_A, resid_by_label, fpa, plot_dir):
    """Full 1024x1024 FPA residual images, one per prior config -- raw
    (W/m2/sr/um, symlog diverging) and as a percent of the local truth
    continuum, matching gd_joint_block_residual_plot.py's own
    full_detector_figure() convention (SymLogNorm + RdBu_r) so these read
    the same way as every other residual figure in this project."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    cont = np.where(np.abs(truth_A) > 0, np.abs(truth_A), np.nan)

    # -- one figure per config: raw + pct side by side --
    for label, resid in resid_by_label.items():
        pct = resid / cont * 100.0
        amax = float(np.nanmax(np.abs(resid))) or 1.0
        lin = float(np.nanmedian(np.abs(resid[np.isfinite(resid)]))) or amax * 1e-3
        pmax = float(np.nanmax(np.abs(pct))) or 1.0
        plin = float(np.nanmedian(np.abs(pct[np.isfinite(pct)]))) or pmax * 1e-3

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        im0 = axes[0].imshow(resid, aspect="auto", cmap="RdBu_r", origin="upper",
                             norm=SymLogNorm(linthresh=lin, vmin=-amax, vmax=amax, base=10))
        axes[0].set_title(f"raw residual (symlog, linthresh={lin:.2g})", fontsize=10.5)
        axes[0].set_xlabel("detector column")
        axes[0].set_ylabel("detector row")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, label="W/m2/sr/um")

        im1 = axes[1].imshow(pct, aspect="auto", cmap="RdBu_r", origin="upper",
                             norm=SymLogNorm(linthresh=plin, vmin=-pmax, vmax=pmax, base=10))
        axes[1].set_title(f"residual / |truth| (symlog, linthresh={plin:.2g}%)", fontsize=10.5)
        axes[1].set_xlabel("detector column")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03, label="%")

        fig.suptitle(f"FPA{fpa} prior-only forward residual, config '{label}' (no solve)",
                    fontsize=13)
        fig.tight_layout()
        out = plot_dir / f"forward_check_2d_{label}_fpa{fpa}.png"
        fig.savefig(out, dpi=150)
        print(f"saved {out}")
        plt.close(fig)

    # -- comparison grid: bingrid (left column) vs. native-grid (right
    # column) pct residual, on ONE shared colour scale, so amplitudes are
    # directly comparable both within and across the two families --
    # 'nearest' downscaled bingrid configs excluded (rejected 2026-08-19,
    # see forward_check_g_ratio_sweep_fpa<N>.png -- 'linear' only from here
    # on). Each column sorted coarse->fine (bingrid: descending g_ratio;
    # native: ascending density), row-aligned by that ordering, NOT by
    # matched density -- the two families' own density axes don't correspond
    # 1:1, this is just "coarsest at top" for each column independently.
    labels_bingrid = sorted(
        (lb for lb in resid_by_label if (parsed := _bingrid_config_from_label(lb)) is not None
         and parsed[1] == "linear"),
        key=lambda lb: -_bingrid_config_from_label(lb)[0])
    labels_native = sorted(
        (lb for lb in resid_by_label if _rt_density_from_label(lb) is not None),
        key=_rt_density_from_label)
    if not labels_bingrid and not labels_native:
        print("_plot_2d: no bingrid_*(linear)/native(grid)_ad* labels present, skipping "
             "comparison grid")
        return
    pct_all = {lb: resid_by_label[lb] / cont * 100.0 for lb in labels_bingrid + labels_native}
    shared_max = max(float(np.nanmax(np.abs(p))) for p in pct_all.values()) or 1.0
    shared_lin = min(float(np.nanmedian(np.abs(p[np.isfinite(p)]))) or shared_max * 1e-3
                     for p in pct_all.values())

    nrows = max(len(labels_bingrid), len(labels_native))
    fig, axes = plt.subplots(nrows, 2, figsize=(10.4, 5.2 * nrows), squeeze=False)
    im = None
    for col, col_labels in enumerate((labels_bingrid, labels_native)):
        for row in range(nrows):
            ax = axes[row][col]
            if row >= len(col_labels):
                ax.axis("off")
                continue
            label = col_labels[row]
            im = ax.imshow(pct_all[label], aspect="auto", cmap="RdBu_r", origin="upper",
                           norm=SymLogNorm(linthresh=shared_lin, vmin=-shared_max,
                                          vmax=shared_max, base=10))
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("detector column")
            ax.set_ylabel("detector row")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="residual / |truth| (%)")
    fig.suptitle(f"FPA{fpa} prior-only forward residual, all configs (shared colour scale, "
                f"no solve)", fontsize=13)
    out = plot_dir / f"forward_check_2d_comparison_fpa{fpa}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
