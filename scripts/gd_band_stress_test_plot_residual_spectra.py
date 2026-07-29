#!/usr/bin/env python3
"""Representative post-fit spectral residual shapes (residual vs. wavenumber,
not just the RMS-vs-position summary in gd_band_stress_test_plot_grid.py)
across the full along-slit stress-test sweep (2026-07-25).

For each case, 12 representative rows are selected using native's chi2 (the
most reliably-converged pipeline, so a full spanning set is almost always
available even in cases where rectified diverges heavily): the best
(lowest) chi2 row, the worst (highest) chi2 row, and 10 more rows evenly
spaced across the slit (by along-slit position, from the remaining
in-family rows). All three pipelines' residual spectra are then plotted at
those SAME 12 row positions -- not each pipeline's own best/worst, since a
shared row set is what makes the panels directly comparable across
pipelines. A pipeline missing at a given row (diverged/off-detector/not
in-family there) is simply omitted from that panel's legend.

Same chi2-outlier filtering as gd_band_stress_test_plot.py /
_plot_grid.py (robust MAD-based -- excludes rows that reported
converged=True over an exploded/unphysical state).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_band_stress_test_plot_residual_spectra.py
Output: plots/gd_band_stress_test_fpa<N>[_uniform|_barcode][_noise]_residual_spectra.png (up to 24)
        plots/gd_band_stress_test_all_residual_spectra.pdf (every available case, one page each)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ORDER = {"native": 2, "rectified": 2, "undistorted": 2}  # undistorted floats dispersion too, since 2026-07-28 (Sec. 11k) -- a numerical workaround, not a physical correction
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}

FPAS = (0, 1, 2, 3)
SCENES = ("realistic", "uniform", "barcode")
NOISES = (False, True)
N_SPAN = 10   # + best + worst = 12 panels total


def _mode_suffix(scene: str) -> str:
    return {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}[scene]


def _result_path(fpa: int, scene: str, noise: bool) -> Path:
    suffix = _mode_suffix(scene) + ("_noise" if noise else "")
    return REPO_ROOT / "results" / f"gd_band_stress_test_fpa{fpa}{suffix}.pkl"


def _chi2_outlier_mask(chi2: np.ndarray) -> np.ndarray:
    chi2 = np.asarray(chi2, dtype=float)
    if len(chi2) == 0:
        return np.zeros(0, dtype=bool)
    log_chi2 = np.log10(np.maximum(chi2, 1e-300))
    med = np.median(log_chi2)
    mad = np.median(np.abs(log_chi2 - med))
    if mad < 1e-12:
        return chi2 > 1e6
    return log_chi2 > med + 8.0 * 1.4826 * mad


def _in_family_rows(out, rows, pipeline: str):
    order = PIPELINE_ORDER[pipeline]
    ks, chi2s = [], []
    for k in rows:
        b = out[(pipeline, order, int(k))]["bias"]
        if b is not None and not b.get("_diverged") and b.get("_conv"):
            ks.append(int(k)); chi2s.append(b["_chi2"])
    ks = np.array(ks); chi2s = np.array(chi2s, dtype=float)
    outlier = _chi2_outlier_mask(chi2s)
    return ks[~outlier], chi2s[~outlier]


def _select_representative_rows(out, rows, x_km_of_row, ref_pipeline="native", n_span=N_SPAN):
    ks, chi2s = _in_family_rows(out, rows, ref_pipeline)
    if len(ks) == 0:
        return []
    order_by_chi2 = np.argsort(chi2s)
    best_k = int(ks[order_by_chi2[0]])
    worst_k = int(ks[order_by_chi2[-1]])
    chosen = {best_k: f"best (chi2={chi2s[order_by_chi2[0]]:.3g})",
             worst_k: f"worst (chi2={chi2s[order_by_chi2[-1]]:.3g})"}

    remaining_mask = ~np.isin(ks, [best_k, worst_k])
    rem_ks = ks[remaining_mask]
    rem_x = np.array([x_km_of_row[k] for k in rem_ks])
    order_by_x = np.argsort(rem_x)
    rem_ks_sorted = rem_ks[order_by_x]
    if len(rem_ks_sorted) > 0:
        span_idx = np.linspace(0, len(rem_ks_sorted) - 1, min(n_span, len(rem_ks_sorted))).astype(int)
        span_idx = np.unique(span_idx)
        for i in span_idx:
            k = int(rem_ks_sorted[i])
            chosen[k] = f"x={x_km_of_row[k]:.0f}km"

    # order the final panel list by along-slit position for a readable layout
    items = [(k, label) for k, label in chosen.items()]
    items.sort(key=lambda kv: x_km_of_row[kv[0]])
    return items


def make_case_figure(fpa: int, scene: str, noise: bool):
    path = _result_path(fpa, scene, noise)
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out, rows = d["out"], np.array(d["rows"])
    x_km_of_row = d["x_km_of_row"]

    reps = _select_representative_rows(out, rows, x_km_of_row)
    if not reps:
        print(f"  no in-family native rows for {path.name}, skipping")
        return None

    # Per-pipeline in-family row sets (converged, not diverged, not a
    # chi2 outlier for THAT pipeline specifically) -- row selection above
    # only used native's, so a row native is fine at can still be stalled/
    # diverged/outlier for rectified or undistorted (found 2026-07-25).
    in_family_ks = {pl: set(_in_family_rows(out, rows, pl)[0].tolist())
                    for pl in ("native", "rectified", "undistorted")}

    n = len(reps)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (k, label) in zip(axes, reps):
        any_data = False
        for pipeline in ("native", "rectified", "undistorted"):
            # Row selection above is based on native's chi2 only -- a row
            # native converges cleanly at is not guaranteed to be converged
            # (or in-family) for rectified/undistorted too (found
            # 2026-07-25: FPA1 row 796 plotted rectified's residual with
            # conv=False, chi2=3.7e8, peak residual ~17000 -- a stalled,
            # meaningless fit, dwarfing the other two pipelines' genuine
            # sub-1 residuals in the same panel). Require this pipeline's
            # own in-family status at this row before plotting it.
            if k not in in_family_ks[pipeline]:
                continue
            order = PIPELINE_ORDER[pipeline]
            rec = out[(pipeline, order, k)]
            nu, resid = rec.get("nu"), rec.get("residual")
            if nu is None or resid is None:
                continue
            any_data = True
            ax.plot(nu, resid, lw=0.6, color=PIPELINE_COLOR[pipeline], label=pipeline, alpha=0.8)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"row {k}: {label}", fontsize=8)
        ax.tick_params(labelsize=6)
        if any_data:
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                   transform=ax.transAxes, color="gray", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    noise_label = "noise" if noise else "no noise"
    fig.suptitle(f"FPA{fpa} -- {scene}, {noise_label}  "
                f"(residual spectra: best/worst native chi2 + {N_SPAN} spanning rows)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> int:
    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    pdf_path = plots_dir / "gd_band_stress_test_all_residual_spectra.pdf"
    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for fpa in FPAS:
            for scene in SCENES:
                for noise in NOISES:
                    label = f"FPA{fpa} {scene} noise={int(noise)}"
                    print(f"{label} ...", flush=True)
                    fig = make_case_figure(fpa, scene, noise)
                    if fig is None:
                        continue
                    suffix = _mode_suffix(scene) + ("_noise" if noise else "")
                    png_path = plots_dir / f"gd_band_stress_test_fpa{fpa}{suffix}_residual_spectra.png"
                    fig.savefig(png_path, dpi=120)
                    pdf.savefig(fig)
                    plt.close(fig)
                    n_saved += 1
                    print(f"  saved {png_path.name}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
