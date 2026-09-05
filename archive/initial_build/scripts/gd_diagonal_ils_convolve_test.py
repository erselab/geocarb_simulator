#!/usr/bin/env python3
"""Unit test for geocarb_gert.gd_render._diagonal_ils_convolve, the custom
ILS-convolution routine written for the joint block's own renderer
(predict_neighborhood / predict_neighborhood_no_keystone). Its own
docstring claims "same truncated-Gaussian kernel and normalisation as
ILS.convolve(..., exact_center=True)" -- reading the two implementations
side by side turns up two concrete differences from gert.instrument.ILS's
own reference convolution:

1. Window half-width: ILS.convolve's internal _window() uses
   `half_idx = max(1, ceil(half/spacing) + 1)`; _diagonal_ils_convolve uses
   `half_idx = max(1, ceil(half/spacing))` -- one fewer hi-res point of
   margin on each side.
2. ILS.convolve applies a second, exact hard cutoff `mask = |delta| <= half`
   after slicing the window; _diagonal_ils_convolve does not -- it uses
   every point in the sliced window as-is, which (given the ceil rounding
   in half_idx) can include points marginally past the true truncation
   radius that ILS.convolve would discard.

This script checks directly whether that produces a measurable difference
in the convolved spectrum -- specifically in absorption line depth, since
this was motivated by a small (~5e-6 relative) constant CO2 bias found in
scripts/gd_keystone_free_1d_ceiling.py's own uniform-scene, PSF-off check,
where truth (built via _diagonal_ils_convolve) and the GERTRetrieval-based
retrieval (built via ILS.convolve internally) are two independently-coded
convolution paths fed the identical hi-res spectrum.

Builds one real hi-res CO2-bearing spectrum (atm_center, co2_scale=1, the
same recipe scripts/gd_joint_block_retrieve.py uses throughout this
investigation) and convolves it with both routines at a few real rows'
own true wavenumber grids (real smile, spanning low/mid/high keystone
amplitude), broadcasting the single spectrum across all 1024 columns so
only the convolution kernel itself is under test -- no attribution, no
RT-model mismatch, no retrieval.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_diagonal_ils_convolve_test.py
Output: plots/joint_block/gd_diagonal_ils_convolve_test_fpa2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics, make_spectrum_fn  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import _diagonal_ils_convolve  # noqa: E402

# Rows spanning low/mid/high real keystone amplitude (see rows_crossed(FPA,.)
# grown from the row-25 null): a couple in the spirit requested, plus the
# array centre for context.
TEST_ROWS = [25, 512, 1000]


def line_depth(nu: np.ndarray, spec: np.ndarray) -> tuple[float, float, float]:
    """Continuum estimate (max over the window), line-core value (min), and
    depth (continuum - core) -- a simple, robust proxy for "how deep is the
    absorption feature" without needing to identify a specific line by
    wavenumber."""
    continuum = float(spec.max())
    core = float(spec.min())
    return continuum, core, continuum - core


def main() -> int:
    print(f"Building real hi-res CO2-bearing spectrum, FPA{FPA}, atm_center, co2_scale=1...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    wide_win, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    spectrum_for = make_spectrum_fn(absco, wide_inst, geo, solar, albedo)
    S_hires = spectrum_for(1.0, atm_center)
    wn_hires = wide_win.wn_hires
    ils = wide_win.ils
    print(f"done. wn_hires: {len(wn_hires)} pts, [{wn_hires[0]:.3f},{wn_hires[-1]:.3f}] cm-1, "
         f"spacing={wn_hires[1]-wn_hires[0]:.4f} cm-1\n", flush=True)

    cols = np.arange(1024.0)
    fig, axes = plt.subplots(len(TEST_ROWS), 2, figsize=(13, 3.6 * len(TEST_ROWS)))

    print(f"{'row':>5s} {'method':>10s} {'continuum':>12s} {'core':>12s} {'depth':>12s} "
         f"{'depth_diff':>12s} {'depth_reldiff':>14s} {'max|diff|':>12s}")
    all_ok = True
    for k, row in enumerate(TEST_ROWS):
        lam_row, _ = xy_to_wavelength_slit(FPA, cols, np.full(1024, float(row)))
        nu_row = 1e4 / lam_row

        # _diagonal_ils_convolve: (n_cols, n_hires) per-column spectra --
        # broadcast the single spectrum so only the kernel is under test.
        S_row_bcast = np.broadcast_to(S_hires, (1024, len(S_hires)))
        y_custom = _diagonal_ils_convolve(wn_hires, S_row_bcast, nu_row, ils)

        # ILS.convolve: one shared spectrum, many centres -- the reference.
        y_gert = ils.convolve(wn_hires, S_hires, nu_row, exact_center=True)

        diff = y_custom - y_gert
        c_c, c_core, c_depth = line_depth(nu_row, y_custom)
        g_c, g_core, g_depth = line_depth(nu_row, y_gert)
        depth_diff = c_depth - g_depth
        depth_reldiff = depth_diff / g_depth if g_depth != 0 else float("nan")
        ok = abs(depth_reldiff) < 1e-4
        all_ok &= ok
        print(f"{row:5d} {'custom':>10s} {c_c:12.6g} {c_core:12.6g} {c_depth:12.6g} "
             f"{'--':>12s} {'--':>14s} {'--':>12s}")
        print(f"{'':5s} {'gert':>10s} {g_c:12.6g} {g_core:12.6g} {g_depth:12.6g} "
             f"{depth_diff:+12.4g} {depth_reldiff:+14.6g} {np.max(np.abs(diff)):12.4g}"
             f"  {'OK' if ok else 'FAIL (>1e-4 rel)'}", flush=True)

        ax = axes[k, 0]
        ax.plot(nu_row, y_custom, color="tab:orange", lw=1.3, label="_diagonal_ils_convolve (custom)")
        ax.plot(nu_row, y_gert, color="tab:blue", lw=1.0, ls="--", label="ILS.convolve (gert reference)")
        ax.set_title(f"row {row}: convolved spectrum", fontsize=10.5)
        ax.set_xlabel("wavenumber [cm-1]"); ax.set_ylabel("radiance")
        ax.legend(fontsize=7.5)

        ax = axes[k, 1]
        ax.axhline(0, color="black", lw=0.6)
        ax.plot(nu_row, diff, color="tab:red", lw=1.0)
        ax.set_title(f"row {row}: custom - gert (max|diff|={np.max(np.abs(diff)):.3g})", fontsize=10.5)
        ax.set_xlabel("wavenumber [cm-1]"); ax.set_ylabel("diff")

    print(f"\noverall: {'ALL PASS' if all_ok else 'AT LEAST ONE FAIL'} (tolerance: "
         f"|line-depth relative difference| < 1e-4)")

    fig.suptitle(f"FPA{FPA}: _diagonal_ils_convolve vs. gert.instrument.ILS.convolve "
                f"(exact_center=True) -- same real spectrum, real per-row grid", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_diagonal_ils_convolve_test_fpa{FPA}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
