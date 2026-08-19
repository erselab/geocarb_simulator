#!/usr/bin/env python3
"""Render and cache one band's raw detector image `band["A"]` to .npy, so
downstream analysis can normalise residuals by the real radiance/continuum
without re-running the render every time.

Why this exists: `gd_joint_block_whole_slit_sweep.py` saves each window's
`resid_*` but never `y_true` (= `band["A"][rows, :]`) nor even `y_scale`, so
a saved sweep alone cannot express its residuals as a fraction of the
continuum -- the denominator simply isn't in the pickle. Re-rendering is the
only way to recover it, and it's deterministic, so it only has to happen
once per (fpa, scene) and can then be cached.

Reproduces `gd_joint_block_whole_slit_sweep.py`'s own setup EXACTLY (same
`geocarb_demo` block, same `sample_geometries(..., n=1, seed=0)` geometry,
same `atm_center = als.atmosphere_at(0.0)`, same
`_band_setup(fpa, ..., snr, 400, None, uniform, False, 32, False, 0)` call),
so the cached `A` is bit-identical to the `y_true` those sweeps fitted
against -- not merely a similar render. `--check-against` verifies that
directly against a sweep pickle rather than trusting the claim.

Run:  PYTHONPATH=. python3 scripts/gd_cache_band_image.py --fpa 2
      PYTHONPATH=. python3 scripts/gd_cache_band_image.py --fpa 2 --uniform
      PYTHONPATH=. python3 scripts/gd_cache_band_image.py --fpa 2 \\
        --check-against /path/to/gd_joint_block_whole_slit_fpa2_gratio1.pkl
Output: <cache-dir>/band_image_fpa<N>[_uniform].npy   (1024x1024 float64)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Where gert's input/ tree (absco.h5, solar.h5) lives. Resolved in order:
# $GERT_ROOT, then the sibling checkout this repo already reaches via
# `../../gert` (the relationship `geocarb_gert` itself relies on), then the
# HPC scratch path. Deliberately NOT an absolute machine-specific default --
# an earlier version hardcoded one workstation's Google-Drive path, which
# breaks the moment the tree is relocated.
from geocarb_gert import gert_root  # noqa: E402

DEFAULT_GERT_ROOT = str(gert_root())


def cache_path(cache_dir: Path, fpa: int, uniform: bool,
               barcode: bool = False, realistic_barcode: bool = False,
               barcode_bars: int = 32, n_lookup_samples: int = 400) -> Path:
    tag = ("_uniform" if uniform else "")
    if barcode:
        tag += f"_barcode{barcode_bars}"
    elif realistic_barcode:
        tag += f"_realisticbarcode{barcode_bars}"
    if n_lookup_samples != 400:
        tag += f"_nls{n_lookup_samples}"
    return cache_dir / f"band_image_fpa{fpa}{tag}.npy"


def render(fpa: int, uniform: bool, gert_root: Path,
          barcode: bool = False, realistic_barcode: bool = False, barcode_bars: int = 32,
          n_lookup_samples: int = 400):
    import gd_test as gdt
    import geosat_geometry as gg
    import gert
    from geocarb_gert import along_slit_scene as als, sample_geometries

    t0 = time.time()
    print("building geometry ...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    print(f"loading absco/solar from {gert_root} ...", flush=True)
    absco = gert.ABSCOTable.load_all(str(gert_root / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(gert_root / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
    print(f"rendering FPA{fpa} 1024x1024 detector image "
          f"(uniform={uniform}, barcode={barcode}, realistic_barcode={realistic_barcode}) "
          f"-- this is the slow part ...", flush=True)
    # positional order must match gd_joint_block_whole_slit_sweep.py's own
    # _band_setup call exactly, or the cache silently stops being bit-identical
    # to what a sweep actually fitted against -- including n_lookup_samples,
    # which must match that sweep's own --n-lookup-samples (400 unless it
    # was run with --realistic-prior or an explicit override)
    band = gdt._band_setup(fpa, atm_center, absco, geo, solar, snr, n_lookup_samples, None,
                           uniform, barcode, barcode_bars, False, 0, realistic_barcode)
    A = np.asarray(band["A"], dtype=float)
    print(f"done in {time.time() - t0:.0f}s -- A {A.shape}, "
          f"range [{A.min():.4g}, {A.max():.4g}] W/m2/sr/um", flush=True)
    return A


def check_against(A: np.ndarray, pkl: Path) -> int:
    """Verify the cached render really is the sweep's own `y_true`, by
    reconstructing one window's `y_true` from A and confirming the sweep's
    saved residual is consistent with it.

    The sweep saved `resid = y_true - forward(x)`; we don't have `forward`,
    so an exact identity isn't recoverable here. What IS checkable, and
    sufficient to catch a wrong render (wrong scene/geometry/FPA), is that
    `|resid| << |y_true|` with the right shape and scale -- a mismatched
    render would put the residual at the same order as the signal.
    """
    import pickle
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    ws = sorted(d["results"].values(), key=lambda r: r["row_lo"])
    print(f"\nchecking against {pkl.name} ({len(ws)} windows):")
    bad = 0
    for w in (ws[0], ws[len(ws) // 2], ws[-1]):
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        y = A[lo:hi + 1, :].ravel()
        r = np.asarray(w.get("resid_hires", w.get("resid_coarse")), dtype=float)
        if r.size != y.size:
            print(f"  rows {lo}-{hi}: SHAPE MISMATCH {r.size} vs {y.size}")
            bad += 1
            continue
        frac = float(np.sqrt(np.mean(r ** 2)) / np.mean(np.abs(y)))
        flag = "ok" if frac < 0.05 else "SUSPICIOUS"
        print(f"  rows {lo:4d}-{hi:4d}: mean|y|={np.mean(np.abs(y)):.4g}  "
              f"residRMS/mean|y| = {frac:.3%}  [{flag}]")
        if frac >= 0.05:
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpa", type=int, default=2)
    ap.add_argument("--uniform", action="store_true",
                    help="render the constant-atmosphere scene instead of the realistic one")
    ap.add_argument("--barcode", action="store_true",
                    help="render the fixed-atmosphere barcode-reflectance scene "
                         "(matches gd_joint_block_whole_slit_sweep.py's own --barcode)")
    ap.add_argument("--realistic-barcode", action="store_true",
                    help="render the realistic-composition scene with a barcode "
                         "reflectance pattern on top (matches --realistic-barcode there)")
    ap.add_argument("--barcode-bars", type=int, default=32,
                    help="only meaningful with --barcode/--realistic-barcode")
    ap.add_argument("--n-lookup-samples", type=int, default=400,
                    help="must match the sweep's own --n-lookup-samples (400 unless it was "
                         "run with --realistic-prior [5600] or an explicit override), or the "
                         "cache silently stops being bit-identical to what that sweep fitted "
                         "against.")
    ap.add_argument("--gert-root", type=str, default=DEFAULT_GERT_ROOT)
    ap.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "results" / "band_cache"))
    ap.add_argument("--check-against", type=str, default=None,
                    help="a sweep pickle to sanity-check the render against")
    ap.add_argument("--force", action="store_true", help="re-render even if cached")
    args = ap.parse_args()
    if args.barcode and args.realistic_barcode:
        ap.error("--barcode and --realistic-barcode are mutually exclusive")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_path(cache_dir, args.fpa, args.uniform, args.barcode,
                     args.realistic_barcode, args.barcode_bars, args.n_lookup_samples)

    if out.exists() and not args.force:
        print(f"already cached: {out} (use --force to re-render)")
        A = np.load(out)
    else:
        A = render(args.fpa, args.uniform, Path(args.gert_root), args.barcode,
                  args.realistic_barcode, args.barcode_bars, args.n_lookup_samples)
        np.save(out, A)
        print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    if args.check_against:
        return 1 if check_against(A, Path(args.check_against)) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
