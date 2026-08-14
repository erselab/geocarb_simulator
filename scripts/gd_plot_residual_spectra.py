#!/usr/bin/env python3
"""Representative post-fit spectral residual shapes, by band, for the joint
multi-band battery (gd_test.py / KEYSTONE_SMILE_BIAS_PLAN.md
Sec. 11h) -- the joint-retrieval counterpart of
gd_band_stress_test_plot_residual_spectra.py.

Reads residuals directly from the saved .pkl (`_residuals`/`_nus`, one
array per band, added 2026-07-29 to gd_test.py's
`_joint_retrieve` -- see KEYSTONE_SMILE_BIAS_PLAN.md's residual-capture
note). Earlier versions of this script had to re-render each case and
re-run ~36 retrievals from scratch because that data didn't exist yet;
that workaround is gone now that the battery itself saves it, so this is
pure pickle-read + matplotlib, same convention as
gd_band_stress_test_plot_residual_spectra.py.

Each figure panel shows all N bands' residuals side by side (they're on
different wavenumber axes and can't be overlaid), for whichever of
native/rectified/undistorted actually has an in-family fit at that row
(rectified's own representative row is its nearest shared-`s_grid` index to
the same along-slit position, since it isn't indexed the same way as
native/undistorted's row tuples -- computed here from the saved geometry
only, no rendering).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_residual_spectra.py --fpas 0,2
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_residual_spectra.py --fpas 0,2 \\
        --pipelines native,undistorted
      (drops rectified from every panel; axes autoscale to whatever's
      actually plotted, so native/undistorted's own much smaller residual
      amplitude is no longer squashed by rectified's -- no separate
      rescale step needed)
Output: plots/gd_joint_<fpas_tag>[_uniform|_barcode][_noise]_residual_spectra.png (up to 6)
        plots/gd_joint_<fpas_tag>_all_residual_spectra.pdf
        (filenames get a trailing _<pipelines> tag whenever --pipelines is
        not the full default set, so filtered and unfiltered runs never
        collide)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

from geocarb_gert import along_slit_scene as als
from geocarb_gert.cross_band import fpas_tag, real_s_of_row
from geocarb_gert.gd_render import s_max

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gd_test import _shared_s_grid  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}
ALL_PIPELINES = ("native", "rectified", "undistorted")

SCENES = ("realistic", "uniform", "barcode")
NOISES = (False, True)
N_SPAN = 10


def _mode_suffix(scene: str) -> str:
    return {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}[scene]


def _result_path(fpas, scene: str, noise: bool) -> Path:
    suffix = _mode_suffix(scene) + ("_noise" if noise else "")
    return REPO_ROOT / "results" / f"gd_joint_{fpas_tag(fpas)}{suffix}.pkl"


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


def _select_representative(out, fpas, n_span=N_SPAN):
    """Select (rows_tuple, label) pairs via native's chi2 -- best, worst, and
    n_span spanning the slit (by the reference band's along-slit position)
    -- same method as the single-band script, generalized from a (ka, kb)
    pair to an N-band `rows` tuple."""
    rows_all, chi2_all = [], []
    for key, nl in out.items():
        if key[0] != "native":
            continue
        if nl.get("off_detector") or nl.get("_diverged") or not nl.get("_conv"):
            continue
        rows_all.append(key[2])
        chi2_all.append(nl["_chi2"])
    if not rows_all:
        return []
    chi2_all = np.array(chi2_all, dtype=float)
    outlier = _chi2_outlier_mask(chi2_all)
    rows_all = [r for r, o in zip(rows_all, outlier) if not o]
    chi2_all = chi2_all[~outlier]
    if not rows_all:
        return []

    ref_rows = np.array([r[0] for r in rows_all])
    s_ref = real_s_of_row(fpas[0])
    x_km = (s_ref[ref_rows] / s_max(fpas[0])) * als.SLIT_HALF_KM

    order_by_chi2 = np.argsort(chi2_all)
    best_i, worst_i = int(order_by_chi2[0]), int(order_by_chi2[-1])
    chosen = {rows_all[best_i]: f"best (chi2={chi2_all[best_i]:.3g})",
             rows_all[worst_i]: f"worst (chi2={chi2_all[worst_i]:.3g})"}

    remaining = np.ones(len(rows_all), dtype=bool)
    remaining[[best_i, worst_i]] = False
    rem_idx = np.where(remaining)[0]
    if len(rem_idx) > 0:
        rem_x = x_km[rem_idx]
        order_by_x = np.argsort(rem_x)
        span_idx = np.linspace(0, len(order_by_x) - 1, min(n_span, len(order_by_x))).astype(int)
        span_idx = np.unique(span_idx)
        for oi in order_by_x[span_idx]:
            gi = rem_idx[oi]
            chosen[rows_all[gi]] = f"x={x_km[gi]:.0f}km"

    x_of = dict(zip(rows_all, x_km.tolist()))
    items = list(chosen.items())
    items.sort(key=lambda t: x_of[t[0]])
    return items


def make_case_figure(fpas, scene: str, noise: bool, pipelines_filter=ALL_PIPELINES):
    path = _result_path(fpas, scene, noise)
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]

    reps = _select_representative(out, fpas)
    if not reps:
        print(f"  no in-family native rows for {path.name}, skipping")
        return None

    s_grid_shared = _shared_s_grid(fpas)
    n_bands = len(fpas)
    n = len(reps)
    fig, axes = plt.subplots(n, n_bands, figsize=(5.5 * n_bands, 2.6 * n), squeeze=False)

    for row_i, (rows, label) in enumerate(reps):
        s_here = real_s_of_row(fpas[0], np.array([float(rows[0])]))[0]
        k_rect = int(np.argmin(np.abs(s_grid_shared - s_here)))
        rect_rows = tuple(k_rect for _ in fpas)

        candidates = {"native": rows, "undistorted": rows, "rectified": rect_rows}
        results = {}
        for pl in pipelines_filter:
            key_rows = candidates[pl]
            nl = out.get((pl, 2, key_rows))
            if nl is None or not nl.get("_conv") or nl.get("_diverged") or nl.get("off_detector"):
                continue
            if not nl.get("_residuals"):
                continue
            results[pl] = nl

        any_data = False
        for pl, nl in results.items():
            any_data = True
            for j in range(n_bands):
                axes[row_i, j].plot(nl["_nus"][j], nl["_residuals"][j], lw=0.6,
                                    color=PIPELINE_COLOR[pl], label=pl, alpha=0.8)
        for j, fpa in enumerate(fpas):
            ax = axes[row_i, j]
            ax.axhline(0, color="k", lw=0.4)
            ax.tick_params(labelsize=6)
            row_desc = f"row {rows[j]}" if j == 0 else f"grid/row match"
            ax.set_title(f"{row_desc}: {label}  (FPA{fpa})", fontsize=8)
        if any_data:
            axes[row_i, 0].legend(fontsize=6)
        else:
            axes[row_i, 0].text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=axes[row_i, 0].transAxes, color="gray", fontsize=8)

    band_label = "+".join(f"FPA{f}" for f in fpas)
    noise_label = "noise" if noise else "no noise"
    fig.suptitle(f"Joint {band_label} -- {scene}, {noise_label} "
                f"(residual spectra by band: best/worst native chi2 + {N_SPAN} spanning rows)",
                fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpas", type=str, default="0,2",
                    help="comma-separated list of >=2 FPA indices, matching "
                         "the gd_test.py run to analyze")
    ap.add_argument("--pipelines", type=str, default=",".join(ALL_PIPELINES),
                    help="comma-separated subset of native,rectified,undistorted "
                         "to plot (default: all three). e.g. native,undistorted "
                         "to drop rectified -- axes autoscale to the remaining "
                         "data, no separate rescale step needed")
    args = ap.parse_args()
    fpas = [int(x) for x in args.fpas.split(",")]
    pipelines_filter = [p.strip() for p in args.pipelines.split(",")]
    pipe_suffix = "" if sorted(pipelines_filter) == sorted(ALL_PIPELINES) else "_" + "_".join(sorted(pipelines_filter))

    tag = fpas_tag(fpas)
    plots_dir = REPO_ROOT / "plots" / "band_stress_test" / tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plots_dir / f"gd_joint_{tag}{pipe_suffix}_all_residual_spectra.pdf"
    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for scene in SCENES:
            for noise in NOISES:
                label = f"{'+'.join(f'FPA{f}' for f in fpas)} {scene} noise={int(noise)}"
                print(f"{label} ...", flush=True)
                fig = make_case_figure(fpas, scene, noise, pipelines_filter)
                if fig is None:
                    continue
                suffix = _mode_suffix(scene) + ("_noise" if noise else "")
                png_path = plots_dir / f"gd_joint_{tag}{suffix}{pipe_suffix}_residual_spectra.png"
                fig.savefig(png_path, dpi=120)
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {png_path.name}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
