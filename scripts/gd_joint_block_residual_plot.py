#!/usr/bin/env python3
"""The joint block's post-fit residuals, in the two coordinate systems that
answer different questions about them: DETECTOR space (row x column, where
the residual actually lives) and FITTING space (residual vs. wavenumber
within a single eta bin, the space the GN solve actually sees).

Nothing here re-solves or re-renders. `gd_joint_block_whole_slit_sweep.py`
saves each window's full `resid_coarse`/`resid_hires` vector (not just the
scalar RMS that `gd_joint_block_whole_slit_plot.py` uses), and `y_true =
band["A"][rows_win, :].ravel()`, so `resid.reshape(width, 1024)` recovers
the residual image exactly. Every other quantity needed -- each pixel's own
eta, its own wavenumber, and its bin assignment -- is pure GD-polynomial
geometry, recomputable from the saved `row_lo`/`row_hi`/`bin_centers`
alone. Same pickle-read + matplotlib convention as every other plotting
script here.

Why two spaces, and why the second one is not redundant
-------------------------------------------------------
A single eta bin collects pixels from SEVERAL detector rows (that's the
whole point of the joint block -- keystone spreads one along-slit position
across a range of rows/columns), and each row carries its own nu(col)
mapping. So within one bin, the same wavenumber is sampled repeatedly, by
different rows. That splits the residual into two physically distinct
components which are NOT separable in the detector image:

  - coherent structure vs. nu (the median across pixels at the same nu):
    genuine spectral model error for that bin's own retrieved atmosphere --
    a wrong co2_scale, or unmodelled nuisance-gas/albedo structure.
  - scatter about that median at fixed nu: cross-row inconsistency --
    attribution/keystone/PSF error. Pixels the model claims see the SAME
    atmosphere disagreeing with each other.

The right-hand summary panel reports both as separate numbers per bin
(`coherent` = RMS of the per-nu median curve, `scatter` = RMS about it), so
"this bin fits badly" can be attributed to one or the other rather than
left as a single chi2.

Bin edges are drawn as contours over the detector image, so the same
boundaries are visible in both spaces -- residual structure that follows
the bin edges is a quantization/attribution signature, structure that runs
down columns is spectral.

Units: residuals are in W m-2 sr-1 um-1, inherited from `band["A"]` (see
RAD_UNITS below for the full provenance chain). That makes them directly
comparable to GeoCarb's own per-pixel noise: for FPA2 (SCO2),
`geocarb_gert.radiometry`'s real instrument-test calibration gives sigma =
0.011 at I=2 and 0.017 at I=5 W m-2 sr-1 um-1, so a residual RMS of a few
1e-3 is a *sub-noise* model error on a single pixel, even though these
noise-free sweeps resolve it exactly.

Caveat: the sweep does not save `y_scale` (`mean(|y_true|)`, the uniform
`Sy_inv_diag` normalisation), so per-bin fit quality is reported as residual
RMS in physical radiance rather than as a reduced chi2. Under the sweep's
uniform weighting the two differ only by that one shared constant, so
relative comparison between bins is unaffected.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_residual_plot.py \\
        [results/gd_joint_block_whole_slit_fpa2.pkl] [--solve hires] [--n-windows 6]
      PYTHONPATH=. .../python scripts/gd_joint_block_residual_plot.py \\
        --windows 884-924,108-116 --solve both
Output: plots/joint_block/<stem>_residual_detector.png
        plots/joint_block/<stem>_residual_fitting.png
        plots/joint_block/<stem>_residual.pdf  (both, all windows)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

N_COLS = 1024
SOLVE_COLOR = {"coarse": "tab:orange", "hires": "tab:blue"}

# Residuals inherit `band["A"]`'s own units, which are raw rendered radiance:
# `gd_test._band_setup` builds `A = gd_render.image(...)` (noise, when
# enabled at all, is added separately afterwards, and these sweeps are
# noise-free), `gd_render` renders `ForwardModel.run()`'s `I_hires`, and
# gert documents that as W m-2 sr-1 um-1 (`gert/forward_model.py`, "Radiance
# I : W m-2 sr-1 um-1"). `geocarb_gert.radiometry`'s own
# RADIOMETRIC_SPEC_BY_FPA agrees independently ("all radiances in
# W/m^2/sr/um"), which is what makes the per-pixel noise sigma below
# directly comparable to these residuals.
RAD_UNITS = r"W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$"
RAD_UNITS_LABEL = f"resid [{RAD_UNITS}]"

PLOT_STYLE = {
    "font.family": "serif", "font.size": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8,
}


# ---------------------------------------------------------------- geometry --

def pixel_eta_nu(fpa: int, rows: np.ndarray):
    """(eta, nu) for every pixel of a row window, each shape (n_rows, 1024).

    Evaluated per pixel at its own true (wavelength, slit position) -- the
    same convention `gd_render.image`/`predict_neighborhood` render on, not
    a per-row centre-column proxy. `nu` in cm-1 (the polynomials return
    microns; `gd_render` converts identically at its own line 212).
    """
    cols = np.arange(N_COLS, dtype=float)
    eta = np.empty((len(rows), N_COLS))
    nu = np.empty((len(rows), N_COLS))
    for i, r in enumerate(rows):
        lam_row, s_row = xy_to_wavelength_slit(fpa, cols, np.full(N_COLS, float(r)))
        eta[i] = s_row / s_max(fpa)
        nu[i] = 1.0e4 / lam_row
    return eta, nu


def bin_assign(bin_centers: np.ndarray, eta_flat: np.ndarray) -> np.ndarray:
    """Nearest-bin assignment -- identical logic to
    `geocarb_gert.nearest_bin_scene` (and to
    `gd_joint_block_diagnostics.bin_assign`), so the pixels grouped here are
    exactly the pixels the forward model attributed to each bin."""
    edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
    return np.searchsorted(edges, eta_flat)


def bin_edges_of(bin_centers: np.ndarray) -> np.ndarray:
    return 0.5 * (bin_centers[:-1] + bin_centers[1:])


# ------------------------------------------------------------- continuum --

def load_band_image(fpa: int, uniform: bool = False, cache_dir: Path | None = None):
    """The sweep's own rendered detector image `band["A"]`, from the .npy
    cache written by `scripts/gd_cache_band_image.py`, or None if absent.

    The sweep pickles save `resid_*` but never `y_true`, so the continuum
    that residuals are normalised against simply is not recoverable from a
    sweep alone -- it has to come from re-rendering, which that script does
    once and caches. Verified there (via `--check-against`) to reproduce the
    sweep's own `y_true`: residRMS/mean|y| came out 0.000-0.003% on the
    gratio1 windows, i.e. the render matches what the sweep actually fitted.
    """
    cache_dir = cache_dir or (REPO_ROOT / "results" / "band_cache")
    p = Path(cache_dir) / f"band_image_fpa{fpa}{'_uniform' if uniform else ''}.npy"
    if not p.exists():
        return None, p
    return np.load(p), p


def continuum_of(A: np.ndarray, block: int = 64, pct: float = 98.0) -> np.ndarray:
    """Per-row continuum (upper envelope) of a detector image, same shape.

    Estimated as a high percentile of the radiance within sliding blocks of
    `block` columns, then linearly interpolated back to full column
    resolution. `block=64` spans roughly 3-4 CO2 line spacings at this
    band's dispersion (~0.1 cm-1/column against ~1.5-2 cm-1 line spacing),
    which is wide enough that every block contains genuine between-line
    continuum, but narrow enough to still follow the real continuum's own
    slow wavelength dependence.

    Row-by-row rather than one shared spectrum, so this stays correct if the
    scene is later given along-slit albedo variation (it currently has none
    -- `along_slit_scene` holds albedo fixed and varies only gases and
    surface pressure -- but then the continuum genuinely would vary by row).
    """
    n_rows, n_cols = A.shape
    n_blk = n_cols // block
    trimmed = A[:, :n_blk * block].reshape(n_rows, n_blk, block)
    env = np.percentile(trimmed, pct, axis=2)              # (n_rows, n_blk)
    centers = np.arange(n_blk) * block + (block - 1) / 2.0
    cols = np.arange(n_cols, dtype=float)
    out = np.empty_like(A)
    for i in range(n_rows):
        out[i] = np.interp(cols, centers, env[i])
    return out


# ------------------------------------------------------------ decomposition --

def coherent_scatter_split(nu_flat, resid_flat, n_nu=200):
    """Split a bin's residuals into a coherent-vs-nu curve and the scatter
    about it.

    Bins the pixels by wavenumber and takes the median in each -- median,
    not mean, so a few outlying rows can't drag the "coherent" component.
    Returns (nu_centers, median_curve, coherent_rms, scatter_rms), with the
    curve NaN in any wavenumber bin holding no pixels.
    """
    if len(nu_flat) == 0:
        return np.zeros(0), np.zeros(0), np.nan, np.nan
    lo, hi = float(nu_flat.min()), float(nu_flat.max())
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros(0), np.zeros(0), np.nan, np.nan
    edges = np.linspace(lo, hi, n_nu + 1)
    idx = np.clip(np.searchsorted(edges, nu_flat, side="right") - 1, 0, n_nu - 1)
    curve = np.full(n_nu, np.nan)
    for k in range(n_nu):
        m = idx == k
        if m.any():
            curve[k] = np.median(resid_flat[m])
    centers = 0.5 * (edges[:-1] + edges[1:])
    coherent = curve[idx]
    ok = np.isfinite(coherent)
    coherent_rms = float(np.sqrt(np.nanmean(curve ** 2))) if np.isfinite(curve).any() else np.nan
    scatter_rms = (float(np.sqrt(np.mean((resid_flat[ok] - coherent[ok]) ** 2)))
                   if ok.any() else np.nan)
    return centers, curve, coherent_rms, scatter_rms


# ------------------------------------------------------------- window prep --

def window_payload(w: dict, fpa: int, solve: str, norm: np.ndarray | None = None):
    """Everything both figures need for one window, or None if this window
    has no saved residual for `solve` (older pickles predate residual
    saving, and `_merge.py` output may be mixed).

    `norm`, when given, is a full-detector (1024, 1024) multiplier applied to
    the residual -- `100 / continuum` for percent-of-continuum units. The
    per-window RMS is then recomputed from the scaled image rather than
    reusing the pickle's own `resid_*_rms`, which is in raw radiance.
    """
    key = f"resid_{solve}"
    if key not in w:
        return None
    lo, hi = int(w["row_lo"]), int(w["row_hi"])
    rows = np.arange(lo, hi + 1)
    resid = np.asarray(w[key], dtype=float)
    if resid.size != len(rows) * N_COLS:
        print(f"  window {lo}-{hi}: {key} has {resid.size} elements, "
              f"expected {len(rows) * N_COLS} -- skipping")
        return None
    img = resid.reshape(len(rows), N_COLS)
    if norm is not None:
        img = img * norm[lo:hi + 1, :]
    eta, nu = pixel_eta_nu(fpa, rows)
    bin_centers = np.asarray(w["bin_centers"], dtype=float)
    return dict(
        rows=rows, resid_img=img, eta=eta, nu=nu,
        bin_centers=bin_centers, bin_idx=bin_assign(bin_centers, eta.ravel()),
        row_lo=lo, row_hi=hi, G=int(w["G"]),
        rms=float(np.sqrt(np.mean(img ** 2))),
    )


# --------------------------------------------------------------- figure 1 --

def detector_figure(payloads: dict, solve: str, fpa: int, unit_label: str = RAD_UNITS_LABEL):
    """Residual images, one column per window, with bin edges overlaid.

    Row 1: the residual image itself. Row 2: its column-mean (spectral
    marginal) and row-mean (along-slit marginal), which separate a residual
    that is uniform down the slit from one that tracks bin structure.
    """
    n = len(payloads)
    fig, axes = plt.subplots(3, n, figsize=(4.8 * n, 10.5), squeeze=False,
                             gridspec_kw={"height_ratios": [2.4, 1.0, 1.0]})

    for j, (name, p) in enumerate(payloads.items()):
        img = p["resid_img"]
        vmax = float(np.nanmax(np.abs(img))) or 1.0

        ax = axes[0, j]
        im = ax.imshow(img, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[0, N_COLS, p["row_hi"] + 0.5, p["row_lo"] - 0.5])
        # Bin edges are contours of constant eta -- drawn in the SAME
        # detector coordinates as the image, so residual structure can be
        # read as either following the bins (attribution/quantization) or
        # running down columns (spectral).
        edges = bin_edges_of(p["bin_centers"])
        if len(edges):
            ax.contour(np.arange(N_COLS), p["rows"], p["eta"], levels=edges,
                       colors="0.25", linewidths=0.6, linestyles="-", alpha=0.75)
        ax.set_title(f"rows {p['row_lo']}-{p['row_hi']}  (G={p['G']}, "
                     f"RMS={p['rms']:.3g})", fontsize=10)
        ax.set_xlabel("detector column")
        if j == 0:
            ax.set_ylabel("detector row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit_label)

        ax = axes[1, j]
        ax.axhline(0, color="black", lw=0.6)
        ax.plot(np.arange(N_COLS), img.mean(axis=0), lw=0.7, color=SOLVE_COLOR[solve])
        ax.set_xlabel("detector column")
        if j == 0:
            ax.set_ylabel("column mean\n(spectral marginal)")

        ax = axes[2, j]
        ax.axhline(0, color="black", lw=0.6)
        ax.plot(p["rows"], img.mean(axis=1), lw=0.9, color=SOLVE_COLOR[solve])
        for e in bin_edges_of(p["bin_centers"]):
            # each bin edge's own row, at the window's centre column
            r_of_e = np.interp(e, p["eta"][:, N_COLS // 2], p["rows"],
                               left=np.nan, right=np.nan)
            if np.isfinite(r_of_e):
                ax.axvline(r_of_e, color="0.7", lw=0.5)
        ax.set_xlabel("detector row")
        if j == 0:
            ax.set_ylabel("row mean\n(along-slit marginal)")

    fig.suptitle(f"FPA{fpa} joint block -- post-fit residual in DETECTOR space "
                 f"({solve}; grey lines = eta bin edges)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


# ---------------------------------------------------------- figure 1-full --

def full_detector_figure(results: dict, fpa: int, solve: str,
                         norm: np.ndarray | None = None,
                         unit_label: str = RAD_UNITS_LABEL):
    """The whole 1024x1024 FPA at once, every window stitched into one image.

    Two renderings, because one cannot serve both purposes: per-window
    residual RMS spans ~2700x across the slit, so a single linear scale
    shows only the mid-slit windows and leaves everything else flat white.

      left  -- raw residual on a symmetric-log colour scale, so the real
               amplitude ordering between windows stays readable.
      right -- each window divided by its OWN RMS, which discards the
               amplitude ordering entirely but makes the residual's spatial
               STRUCTURE visible in every window at once, including the
               near-null ones at low row.

    Rows no window covered stay NaN (drawn as the axes background), so gaps
    are visible rather than silently closed up.
    """
    from matplotlib.colors import SymLogNorm

    n_rows_full = 1024
    img = np.full((n_rows_full, N_COLS), np.nan)
    img_n = np.full((n_rows_full, N_COLS), np.nan)
    row_rms = np.full(n_rows_full, np.nan)
    covered = 0

    for w in sorted(results.values(), key=lambda r: r["row_lo"]):
        key = f"resid_{solve}"
        if key not in w:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        nr = hi - lo + 1
        r = np.asarray(w[key], dtype=float)
        if r.size != nr * N_COLS or hi >= n_rows_full:
            continue
        blk = r.reshape(nr, N_COLS)
        if norm is not None:
            blk = blk * norm[lo:hi + 1, :]
        img[lo:hi + 1] = blk
        rms = float(np.sqrt(np.mean(blk ** 2))) or 1.0
        img_n[lo:hi + 1] = blk / rms
        row_rms[lo:hi + 1] = np.sqrt(np.mean(blk ** 2, axis=1))
        covered += nr

    finite = np.isfinite(img)
    if not finite.any():
        return None, 0
    # Window boundaries: the row right after each window's own row_hi -- the
    # seam between two independently-regularized solves. Derived from
    # `results` itself (every window's own row_lo/row_hi), not a separate
    # `tiles` argument, so this never drifts out of sync with what was
    # actually stitched into `img` above.
    lo_hi = sorted((int(w["row_lo"]), int(w["row_hi"])) for w in results.values())
    win_bounds = [hi + 0.5 for _, hi in lo_hi[:-1]]
    amax = float(np.nanmax(np.abs(img)))
    # linear threshold at the median per-row RMS: below this the scale is
    # linear, above it logarithmic, so both the near-null low-row windows
    # and the mid-slit peak are legible on one colour bar.
    lin = float(np.nanmedian(row_rms))
    lin = lin if np.isfinite(lin) and lin > 0 else amax * 1e-3

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.0], hspace=0.22, wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(img, aspect="auto", cmap="RdBu_r", origin="upper",
                   extent=[0, N_COLS, n_rows_full, 0],
                   norm=SymLogNorm(linthresh=lin, vmin=-amax, vmax=amax, base=10))
    ax.set_title(f"raw residual (symlog, linthresh={lin:.2g})", fontsize=10.5)
    ax.set_xlabel("detector column")
    ax.set_ylabel("detector row")
    for b in win_bounds:
        ax.axhline(b, color="0.3", lw=0.3, alpha=0.4, zorder=3)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=unit_label)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(img_n, aspect="auto", cmap="RdBu_r", origin="upper",
                   extent=[0, N_COLS, n_rows_full, 0], vmin=-3, vmax=3)
    ax.set_title("each window normalised by its own RMS "
                 "(structure, not amplitude)", fontsize=10.5)
    ax.set_xlabel("detector column")
    for b in win_bounds:
        ax.axhline(b, color="0.2", lw=0.3, alpha=0.5, zorder=3)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="resid / window RMS")

    ax = fig.add_subplot(gs[1, :])
    ax.semilogy(np.arange(n_rows_full), row_rms, lw=0.8, color=SOLVE_COLOR[solve])
    for b in win_bounds:
        ax.axvline(b, color="0.6", lw=0.4, alpha=0.5, zorder=0.5)
    ax.set_xlim(0, n_rows_full)
    ax.set_xlabel("detector row")
    ax.set_ylabel("per-row residual RMS")
    ax.set_title("per-row residual RMS across the full slit (log scale)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(f"FPA{fpa} joint block -- FULL-DETECTOR post-fit residual ({solve}), "
                 f"{len(results)} windows stitched, {covered}/{n_rows_full} rows covered, "
                 f"{len(win_bounds)} window boundaries marked",
                 fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.93, bottom=0.07)
    return fig, covered


# --------------------------------------------------------------- figure 2 --

def fitting_figure(payload: dict, solve: str, fpa: int, bins_to_show=6, n_nu=200,
                   unit_label: str = RAD_UNITS_LABEL, unit_short: str = RAD_UNITS):
    """Residual vs. wavenumber WITHIN individual bins, plus a per-bin
    coherent/scatter summary for every bin in the window.

    Bins shown: evenly spaced across the window, plus the worst bin by total
    residual RMS (labelled), since that is the one worth reading.
    """
    G = payload["G"]
    resid_flat = payload["resid_img"].ravel()
    nu_flat = payload["nu"].ravel()
    row_flat = np.repeat(payload["rows"], N_COLS)
    bin_idx = payload["bin_idx"]

    stats = []
    for g in range(G):
        m = bin_idx == g
        if not m.any():
            stats.append((g, 0, np.nan, np.nan, np.nan))
            continue
        _, _, coh, sca = coherent_scatter_split(nu_flat[m], resid_flat[m], n_nu=n_nu)
        stats.append((g, int(m.sum()), float(np.sqrt(np.mean(resid_flat[m] ** 2))), coh, sca))

    rms_per_bin = np.array([s[2] for s in stats], dtype=float)
    worst = int(np.nanargmax(rms_per_bin)) if np.isfinite(rms_per_bin).any() else 0
    chosen = sorted(set(np.linspace(0, G - 1, min(bins_to_show, G)).astype(int)) | {worst})

    n = len(chosen)
    fig = plt.figure(figsize=(15, 2.5 * n + 2.2))
    gs = fig.add_gridspec(n, 2, width_ratios=[2.3, 1.0], hspace=0.55, wspace=0.22)

    for i, g in enumerate(chosen):
        ax = fig.add_subplot(gs[i, 0])
        m = bin_idx == g
        ax.axhline(0, color="black", lw=0.6)
        if m.any():
            # colour by detector row: within one bin the model asserts every
            # one of these pixels sees the SAME atmosphere, so any
            # row-ordered structure here is attribution error, not spectral.
            sc = ax.scatter(nu_flat[m], resid_flat[m], c=row_flat[m], s=1.5,
                            cmap="viridis", alpha=0.55, linewidths=0)
            cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
            cb.set_label("detector row", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            centers, curve, coh, sca = coherent_scatter_split(nu_flat[m], resid_flat[m], n_nu=n_nu)
            ax.plot(centers, curve, color="crimson", lw=1.0,
                    label=f"per-nu median (coherent RMS={coh:.3g})")
            ax.plot([], [], " ", label=f"scatter about it RMS={sca:.3g}")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no pixels in this bin", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=9)
        eta_c = payload["bin_centers"][g]
        flag = "  <-- worst" if g == worst else ""
        ax.set_title(f"bin {g}  (eta={eta_c:+.5f}, x={eta_c * als.SLIT_HALF_KM:+.0f} km, "
                     f"{stats[g][1]} px){flag}", fontsize=9.5)
        ax.set_ylabel("resid")
        if i == n - 1:
            ax.set_xlabel("wavenumber [cm$^{-1}$]")

    ax = fig.add_subplot(gs[:, 1])
    bc = payload["bin_centers"]
    ax.plot(bc, [s[2] for s in stats], "o-", color="0.25", ms=4, label="total RMS")
    ax.plot(bc, [s[3] for s in stats], "s-", color="crimson", ms=4,
            label="coherent (spectral model error)")
    ax.plot(bc, [s[4] for s in stats], "^-", color="tab:blue", ms=4,
            label="scatter (cross-row inconsistency)")
    ax.set_yscale("log")
    ax.set_xlabel("bin center (eta)")
    ax.set_ylabel(f"residual RMS [{unit_short}]")
    ax.set_title("per-bin fit quality, decomposed", fontsize=10)
    ax.legend(fontsize=8)
    for g in chosen:
        ax.axvline(bc[g], color="0.85", lw=0.5, zorder=0)

    fig.suptitle(f"FPA{fpa} joint block rows {payload['row_lo']}-{payload['row_hi']} "
                 f"-- residual in FITTING space ({solve}): within-bin residual vs. wavenumber",
                 fontsize=12.5)
    # subplots_adjust, not tight_layout: the per-bin colorbars are added
    # into this gridspec's own axes, which tight_layout can't account for
    # (it warns and mislays the panels).
    fig.subplots_adjust(left=0.06, right=0.97, top=0.955, bottom=0.05)
    return fig


# -------------------------------------------------------------------- main --

def select_windows(results: dict, solve: str, n_windows: int, explicit=None):
    """Windows to plot: either explicit `row_lo-row_hi` names, or the worst
    window by saved residual RMS plus `n_windows-1` spanning the slit --
    same best/worst-plus-spanning idea as
    `gd_plot_residual_spectra._select_representative`, but keyed on the
    joint block's own per-window RMS rather than a per-row chi2."""
    ws = sorted(results.values(), key=lambda r: r["row_lo"])
    ws = [w for w in ws if f"resid_{solve}" in w]
    if not ws:
        return []
    if explicit:
        want = set(explicit)
        chosen = [w for w in ws if f"{w['row_lo']}-{w['row_hi']}" in want]
        missing = want - {f"{w['row_lo']}-{w['row_hi']}" for w in chosen}
        for mname in sorted(missing):
            print(f"  requested window {mname} not in results (or has no saved "
                  f"resid_{solve}) -- skipping")
        return chosen
    rms = np.array([w.get(f"resid_{solve}_rms", np.nan) for w in ws], dtype=float)
    worst = int(np.nanargmax(rms)) if np.isfinite(rms).any() else 0
    span = set(np.linspace(0, len(ws) - 1, min(max(n_windows - 1, 1), len(ws))).astype(int))
    return [ws[i] for i in sorted(span | {worst})]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=str, nargs="?",
                    default=str(REPO_ROOT / "results" / "gd_joint_block_whole_slit_fpa2.pkl"),
                    help="a gd_joint_block_whole_slit_sweep.py (or _merge.py) output pickle")
    ap.add_argument("--fpa", type=int, default=None,
                    help="FPA the sweep was run on (default: read from the pickle's own "
                         "'fpa' key, which the sweep saves)")
    ap.add_argument("--solve", type=str, default="hires", choices=["coarse", "hires", "both"],
                    help="which solve's residual to plot (default: hires)")
    ap.add_argument("--n-windows", type=int, default=4,
                    help="how many windows in the detector-space figure "
                         "(worst by RMS + the rest spanning the slit)")
    ap.add_argument("--windows", type=str, default=None,
                    help="explicit comma-separated 'row_lo-row_hi' list, overriding --n-windows")
    ap.add_argument("--bins", type=int, default=6,
                    help="bins shown per window in the fitting-space figure "
                         "(evenly spaced, plus the worst)")
    ap.add_argument("--units", type=str, default="percent", choices=["percent", "radiance"],
                    help="'percent' (default) normalises by the per-row continuum from "
                         "the cached band image (scripts/gd_cache_band_image.py); "
                         "'radiance' leaves residuals in W/m2/sr/um. Falls back to "
                         "radiance with a warning if the cache is missing.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"MISSING: {in_path}")
        return 1
    with open(in_path, "rb") as f:
        d = pickle.load(f)
    if "results" not in d:
        print(f"{in_path.name} has no 'results' key (keys: {sorted(d)[:8]}) -- this script "
              f"reads gd_joint_block_whole_slit_sweep.py output, not the gd_test.py battery "
              f"(use scripts/gd_plot_residual_spectra.py for that).")
        return 1
    results = d["results"]
    fpa = args.fpa if args.fpa is not None else int(d.get("fpa", 2))
    print(f"{in_path.name}: {len(results)} windows, FPA{fpa}, "
          f"g_ratio={d.get('g_ratio')}, gamma={d.get('gamma')}, "
          f"sigma_abs={d.get('sigma_abs')}, uniform={d.get('uniform')}")
    explicit = [s.strip() for s in args.windows.split(",")] if args.windows else None
    solves = ["coarse", "hires"] if args.solve == "both" else [args.solve]

    # Percent-of-continuum needs the rendered radiance, which no sweep pickle
    # carries -- see load_band_image. Fall back loudly rather than silently
    # mislabelling raw radiance as a percentage.
    norm, unit_label, unit_short = None, RAD_UNITS_LABEL, RAD_UNITS
    if args.units == "percent":
        A, cache_p = load_band_image(fpa, uniform=bool(d.get("uniform", False)))
        if A is None:
            print(f"WARNING: no cached band image at {cache_p} -- falling back to raw "
                  f"radiance units. Generate it with:\n"
                  f"  PYTHONPATH=. python3 scripts/gd_cache_band_image.py --fpa {fpa}"
                  f"{' --uniform' if d.get('uniform') else ''}")
        else:
            cont = continuum_of(A)
            norm = 100.0 / cont
            unit_short = "% of continuum"
            unit_label = "resid [% of continuum]"
            print(f"continuum from {cache_p.name}: "
                  f"median {np.median(cont):.4g}, range [{cont.min():.4g}, {cont.max():.4g}] "
                  f"W/m2/sr/um -- residuals plotted as % of it")

    plt.rcParams.update(PLOT_STYLE)
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    usuffix = "" if norm is None else "_pct"
    pdf_path = plots_dir / f"{in_path.stem}_residual{usuffix}.pdf"

    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for solve in solves:
            windows = select_windows(results, solve, args.n_windows, explicit)
            if not windows:
                print(f"no windows with a saved resid_{solve} in {in_path.name} -- "
                      f"this sweep predates residual saving, or was merged from runs that did.")
                continue

            fig, covered = full_detector_figure(results, fpa, solve, norm=norm,
                                               unit_label=unit_label)
            if fig is not None:
                out = plots_dir / f"{in_path.stem}_residual_full_{solve}{usuffix}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {out.name}  ({covered}/1024 rows covered)", flush=True)

            payloads = {}
            for w in windows:
                p = window_payload(w, fpa, solve, norm=norm)
                if p is not None:
                    payloads[f"{p['row_lo']}-{p['row_hi']}"] = p
            if not payloads:
                continue
            print(f"[{solve}] {len(payloads)} windows: {', '.join(payloads)}", flush=True)

            fig = detector_figure(payloads, solve, fpa, unit_label=unit_label)
            out = plots_dir / f"{in_path.stem}_residual_detector_{solve}{usuffix}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            pdf.savefig(fig)
            plt.close(fig)
            n_saved += 1
            print(f"  saved {out.name}", flush=True)

            for name, p in payloads.items():
                fig = fitting_figure(p, solve, fpa, bins_to_show=args.bins,
                                     unit_label=unit_label, unit_short=unit_short)
                out = plots_dir / f"{in_path.stem}_residual_fitting_{solve}_{name}{usuffix}.png"
                fig.savefig(out, dpi=140, bbox_inches="tight")
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {out.name}", flush=True)

    if n_saved == 0:
        pdf_path.unlink(missing_ok=True)
        print("nothing plotted.")
        return 1
    print(f"\nsaved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
