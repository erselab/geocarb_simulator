#!/usr/bin/env python3
"""Comprehensive state-variable + uncertainty + residual diagnostic across
the full multi-band battery (gd_test.py), for any FPA count. One
comparable multi-panel figure per case: every retrieved state-vector
element (gas scales, p_scale, T_offset, and per-band albedo/albedo_slope/
dispersion) plotted vs. along-slit position with a shaded +/-1sigma
posterior-uncertainty band (from _state_full -- raw retrieved values, not
a bias-vs-truth view), plus a residual-RMS-vs-slit-position panel (one
scalar per row: sqrt(mean(residual**2)) over all bands' residuals
concatenated together).

Migrated 2026-08-10 from gd_band_stress_test_plot_grid.py's single-band-
only version to gd_test.py's flat, N-band-generic pkl schema: rows/x_km/
gases aren't stored at the top level of a gd_test.py pkl (unlike the old
gd_band_stress_test.py format), so they're derived here instead --
row-tuples straight from `out`'s own keys, x_km via the same real-geometry
helpers gd_test.py itself uses (fpas[0] is the along-slit reference for
native/undistorted; rectified indexes the shared grid instead, a
genuinely different integer space -- see `_x_km_maps`), and gases by
inspecting one converged entry's own keys.

Panel layout: gas1_scale, gas2_scale (if a second non-h2o gas is present),
h2o_scale, p_scale, T_offset, then one group of five panels (albedo_i,
albedo_slope_i, disp_a0_i, disp_a1_i, disp_a2_i) per band in fpas order,
then residual RMS. Panel count scales with band count, so it's fixed (and
directly comparable) for a given N, but not across different N.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_grid.py --fpas 0
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_grid.py --fpas 0,2
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_grid.py --fpas 0,2 \\
        --pipelines native,undistorted
      (drops rectified from every panel; axes autoscale to whatever's
      actually plotted, so native/undistorted's own much smaller range is
      no longer squashed by rectified's -- no separate rescale step needed)
Output: plots/gd_joint_<fpas_tag>[_uniform|_barcode][_noise]_grid.png (up to 6)
        plots/gd_joint_<fpas_tag>_all_grid.pdf
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
from geocarb_gert import chi2_outlier_mask
from geocarb_gert.cross_band import fpas_tag
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit
from geocarb_gert.gd_render import s_max

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gd_test import _shared_s_grid  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ORDER = {"native": 2, "rectified": 2, "undistorted": 2}
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}
ALL_PIPELINES = ("native", "rectified", "undistorted")

SCENES = ("realistic", "uniform", "barcode")
NOISES = (False, True)

_SPECIAL_KEYS = {"_state", "_state_full", "_chi2", "_conv", "_diverged", "_residuals", "_nus"}


def _mode_suffix(scene: str) -> str:
    return {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}[scene]


def _result_path(fpas, scene: str, noise: bool) -> Path:
    suffix = _mode_suffix(scene) + ("_noise" if noise else "")
    return REPO_ROOT / "results" / f"gd_joint_{fpas_tag(fpas)}{suffix}.pkl"


def _x_km_maps(fpas):
    """Two lookup arrays, index-by-row: native/undistorted use fpas[0]'s
    real per-pixel geometry directly; rectified indexes the shared grid
    (_shared_s_grid) instead -- a different integer space entirely, not
    interchangeable with native row indices."""
    sm = s_max(fpas[0])
    cols = np.full(1024, 512.0)
    _, s_native = xy_to_wavelength_slit(fpas[0], cols, np.arange(1024.0))
    x_km_native = (s_native / sm) * als.SLIT_HALF_KM

    s_grid = _shared_s_grid(fpas)
    x_km_rect = (s_grid / sm) * als.SLIT_HALF_KM
    return x_km_native, x_km_rect


def _row0_to_xkm(pipeline, row0, x_km_native, x_km_rect):
    arr = x_km_rect if pipeline == "rectified" else x_km_native
    return float(arr[int(row0)])


def _chi2_outlier_mask(chi2: np.ndarray) -> np.ndarray:
    return chi2_outlier_mask(chi2, n_mad=8.0)


def _infer_gases(out: dict) -> list:
    for (pl, o, r), v in out.items():
        if pl != "native" or v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        keys = [k for k in v if k not in _SPECIAL_KEYS and k != "p_surface"]
        if keys:
            return sorted(keys)
    return []


def _in_family_rows(out, pipeline: str):
    order = PIPELINE_ORDER[pipeline]
    row_tuples = sorted(set(k[2] for k in out if k[0] == pipeline))
    ks, vals, chi2s = [], [], []
    for rt in row_tuples:
        v = out.get((pipeline, order, rt))
        if v is not None and not v.get("_diverged") and v.get("_conv"):
            ks.append(rt); vals.append(v); chi2s.append(v["_chi2"])
    chi2s = np.array(chi2s, dtype=float)
    outlier = _chi2_outlier_mask(chi2s)
    return ks, vals, outlier


def _state_value_series(out, pipeline: str, key: str, x_km_native, x_km_rect):
    ks, vals_l, outlier = _in_family_rows(out, pipeline)
    xs, vals, sigmas = [], [], []
    for rt, v, is_out in zip(ks, vals_l, outlier):
        if is_out:
            continue
        sf = v.get("_state_full") or {}
        names = sf.get("names")
        if not names or key not in names:
            continue
        i = names.index(key)
        val = float(sf["x_ret"][i])
        sig = float(sf["sigma"][i])
        if np.isfinite(val):
            xs.append(_row0_to_xkm(pipeline, rt[0], x_km_native, x_km_rect))
            vals.append(val); sigmas.append(sig)
    order_idx = np.argsort(xs)
    return np.array(xs)[order_idx], np.array(vals)[order_idx], np.array(sigmas)[order_idx]


def _residual_rms_series(out, pipeline: str, x_km_native, x_km_rect):
    ks, vals_l, outlier = _in_family_rows(out, pipeline)
    xs, rms = [], []
    for rt, v, is_out in zip(ks, vals_l, outlier):
        if is_out:
            continue
        resids = v.get("_residuals")
        if not resids:
            continue
        all_resid = np.concatenate([np.asarray(r) for r in resids])
        xs.append(_row0_to_xkm(pipeline, rt[0], x_km_native, x_km_rect))
        rms.append(float(np.sqrt(np.mean(all_resid ** 2))))
    order_idx = np.argsort(xs)
    return np.array(xs)[order_idx], np.array(rms)[order_idx]


def make_case_figure(fpas, scene: str, noise: bool, pipelines_filter=ALL_PIPELINES):
    path = _result_path(fpas, scene, noise)
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    x_km_native, x_km_rect = _x_km_maps(fpas)

    gases = _infer_gases(out)
    non_h2o = [g for g in gases if g != "h2o"]
    gas1 = non_h2o[0] if non_h2o else None
    gas2 = non_h2o[1] if len(non_h2o) > 1 else None

    panel_keys, panel_labels = [], []
    if gas1:
        panel_keys.append(f"{gas1}_scale"); panel_labels.append(f"{gas1}_scale")
    if gas2:
        panel_keys.append(f"{gas2}_scale"); panel_labels.append(f"{gas2}_scale")
    panel_keys += ["h2o_scale", "p_scale", "T_offset"]
    panel_labels += ["h2o_scale", "p_scale", "T_offset [K]"]
    for i, fpa in enumerate(fpas):
        panel_keys += [f"albedo_{i}", f"albedo_slope_{i}", f"disp_a0_{i}", f"disp_a1_{i}", f"disp_a2_{i}"]
        panel_labels += [f"albedo_{i} (FPA{fpa})", f"albedo_slope_{i} (FPA{fpa})",
                         f"disp_a0_{i} (FPA{fpa}) [cm-1]", f"disp_a1_{i} (FPA{fpa}) [cm-1]",
                         f"disp_a2_{i} (FPA{fpa}) [cm-1]"]
    panel_keys.append("residual_rms"); panel_labels.append("residual RMS (all bands)")

    n = len(panel_keys)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key, label in zip(axes, panel_keys, panel_labels):
        any_data = False
        if key == "residual_rms":
            for pipeline in pipelines_filter:
                xs, rms = _residual_rms_series(out, pipeline, x_km_native, x_km_rect)
                if len(xs):
                    any_data = True
                    ax.plot(xs, rms, ".", ms=2, color=PIPELINE_COLOR[pipeline], label=pipeline, alpha=0.6)
            ax.set_yscale("log")
        else:
            for pipeline in pipelines_filter:
                xs, vals, sigmas = _state_value_series(out, pipeline, key, x_km_native, x_km_rect)
                if len(xs):
                    any_data = True
                    ax.plot(xs, vals, ".", ms=2, color=PIPELINE_COLOR[pipeline], label=pipeline, alpha=0.6)
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

    band_label = "+".join(f"FPA{f}" for f in fpas)
    noise_label = "noise" if noise else "no noise"
    fig.suptitle(f"{band_label} -- {scene}, {noise_label}  "
                f"(retrieved state +/-1sigma and residual RMS vs. slit position)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpas", type=str, default="0,2",
                    help="comma-separated FPA indices, e.g. 0 for a single-band case or 0,2 for joint")
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
    pdf_path = plots_dir / f"gd_joint_{tag}{pipe_suffix}_all_grid.pdf"
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
                png_path = plots_dir / f"gd_joint_{tag}{suffix}{pipe_suffix}_grid.png"
                fig.savefig(png_path, dpi=120)
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {png_path.name}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
