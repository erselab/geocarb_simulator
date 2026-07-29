#!/usr/bin/env python3
"""Comprehensive state-variable + uncertainty + residual diagnostic across
the full along-slit stress-test sweep (all FPA x scene x noise
combinations, 2026-07-25). One comparable multi-panel figure per case:
every retrieved state-vector element (gas scales, p_scale, T_offset,
albedo, dispersion coefficients where present) plotted vs. along-slit
position with a shaded +/-1sigma posterior-uncertainty band (from
_state_full -- raw retrieved values, not the bias-vs-truth view the main
gd_band_stress_test_plot.py shows), plus a residual-RMS-vs-slit-position
panel (one scalar per row: sqrt(mean(residual**2)), the same per-row
post-fit spectral residual gd_band_stress_test.py already saves). Same
chi2-outlier filtering as gd_band_stress_test_plot.py (robust MAD-based,
excludes rows that reported converged=True over an exploded/unphysical
state -- see that script's module docstring for why this check exists).

Panel layout is fixed (same slots in the same order for every case, "no
data" where a case doesn't have that element -- e.g. dispersion panels for
a pre-2026-07-28 undistorted .pkl, back when it was order=0) so all 24
figures are directly visually comparable.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_band_stress_test_plot_grid.py
Output: plots/gd_band_stress_test_fpa<N>[_uniform|_barcode][_noise]_grid.png (up to 24)
        plots/gd_band_stress_test_all_grid.pdf (every available case, one page each)
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


def _mode_suffix(scene: str) -> str:
    return {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}[scene]


def _result_path(fpa: int, scene: str, noise: bool) -> Path:
    suffix = _mode_suffix(scene) + ("_noise" if noise else "")
    return REPO_ROOT / "results" / f"gd_band_stress_test_fpa{fpa}{suffix}.pkl"


def _chi2_outlier_mask(chi2: np.ndarray) -> np.ndarray:
    """See gd_band_stress_test_plot.py's identical function for the full
    rationale -- catches rows that reported converged=True over a chi2
    that's finite but astronomically, unphysically large."""
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
    ks, biases, chi2s = [], [], []
    for k in rows:
        b = out[(pipeline, order, int(k))]["bias"]
        if b is not None and not b.get("_diverged") and b.get("_conv"):
            ks.append(int(k)); biases.append(b); chi2s.append(b["_chi2"])
    chi2s = np.array(chi2s, dtype=float)
    outlier = _chi2_outlier_mask(chi2s)
    return np.array(ks), biases, outlier


def _state_value_series(out, rows, x_km_of_row, pipeline: str, key: str):
    """Raw retrieved value + 1-sigma posterior uncertainty for one state
    element, vs. x_km -- from _state_full (added 2026-07-25), not the
    bias-vs-prior/truth views elsewhere."""
    ks, biases, outlier = _in_family_rows(out, rows, pipeline)
    xs, vals, sigmas = [], [], []
    for k, b, is_out in zip(ks, biases, outlier):
        if is_out:
            continue
        sf = b.get("_state_full") or {}
        names = sf.get("names")
        if not names or key not in names:
            continue
        i = names.index(key)
        v = float(sf["x_ret"][i])
        s = float(sf["sigma"][i])
        if np.isfinite(v):
            xs.append(x_km_of_row[k]); vals.append(v); sigmas.append(s)
    order_idx = np.argsort(xs)
    return np.array(xs)[order_idx], np.array(vals)[order_idx], np.array(sigmas)[order_idx]


def _residual_rms_series(out, rows, x_km_of_row, pipeline: str):
    order = PIPELINE_ORDER[pipeline]
    ks, _, outlier = _in_family_rows(out, rows, pipeline)
    xs, rms = [], []
    for k, is_out in zip(ks, outlier):
        if is_out:
            continue
        resid = out[(pipeline, order, int(k))]["residual"]
        if resid is None:
            continue
        xs.append(x_km_of_row[k]); rms.append(float(np.sqrt(np.mean(np.asarray(resid) ** 2))))
    order_idx = np.argsort(xs)
    return np.array(xs)[order_idx], np.array(rms)[order_idx]


def make_case_figure(fpa: int, scene: str, noise: bool):
    path = _result_path(fpa, scene, noise)
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out, rows, gases = d["out"], np.array(d["rows"]), d["gases"]
    x_km_of_row = d["x_km_of_row"]

    non_h2o = [g for g in gases if g != "h2o"]
    gas1 = non_h2o[0] if non_h2o else None
    gas2 = non_h2o[1] if len(non_h2o) > 1 else None

    panel_keys, panel_labels = [], []
    if gas1:
        panel_keys.append(f"{gas1}_scale"); panel_labels.append(f"{gas1}_scale")
    if gas2:
        panel_keys.append(f"{gas2}_scale"); panel_labels.append(f"{gas2}_scale")
    panel_keys += ["h2o_scale", "p_scale", "T_offset", "albedo_0", "albedo_slope_0",
                   "disp_a0_0", "disp_a1_0", "disp_a2_0"]
    panel_labels += ["h2o_scale", "p_scale", "T_offset [K]", "albedo_0", "albedo_slope_0",
                      "disp_a0_0 [cm-1]", "disp_a1_0 [cm-1]", "disp_a2_0 [cm-1]"]
    panel_keys.append("residual_rms"); panel_labels.append("residual RMS")

    n = len(panel_keys)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key, label in zip(axes, panel_keys, panel_labels):
        any_data = False
        if key == "residual_rms":
            for pipeline in ("native", "rectified", "undistorted"):
                xs, rms = _residual_rms_series(out, rows, x_km_of_row, pipeline)
                if len(xs):
                    any_data = True
                    ax.plot(xs, rms, ".", ms=2, color=PIPELINE_COLOR[pipeline],
                           label=pipeline, alpha=0.6)
            ax.set_yscale("log")
        else:
            for pipeline in ("native", "rectified", "undistorted"):
                xs, vals, sigmas = _state_value_series(out, rows, x_km_of_row, pipeline, key)
                if len(xs):
                    any_data = True
                    ax.plot(xs, vals, ".", ms=2, color=PIPELINE_COLOR[pipeline],
                           label=pipeline, alpha=0.6)
                    ax.fill_between(xs, vals - sigmas, vals + sigmas,
                                    color=PIPELINE_COLOR[pipeline], alpha=0.15, linewidth=0)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("x [km]", fontsize=8)
        ax.tick_params(labelsize=7)
        if any_data:
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                   transform=ax.transAxes, color="gray", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    noise_label = "noise" if noise else "no noise"
    fig.suptitle(f"FPA{fpa} -- {scene}, {noise_label}  "
                f"(retrieved state +/-1sigma and residual RMS vs. slit position)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> int:
    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    pdf_path = plots_dir / "gd_band_stress_test_all_grid.pdf"
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
                    png_path = plots_dir / f"gd_band_stress_test_fpa{fpa}{suffix}_grid.png"
                    fig.savefig(png_path, dpi=120)
                    pdf.savefig(fig)
                    plt.close(fig)
                    n_saved += 1
                    print(f"  saved {png_path.name}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
