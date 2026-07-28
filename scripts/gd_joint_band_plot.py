#!/usr/bin/env python3
"""Summary diagnostics for the joint two-band battery (gd_joint_band_test.py
/ KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11h) -- the joint-retrieval counterpart of
gd_band_stress_test_plot.py: bias vs. along-slit position for co2/h2o/
p_surface, chi2 vs. position, and a failure-location panel, for all three
pipelines (native/rectified/undistorted) at once, for each of the 6 scene x
noise combinations.

Position axis: real slit angle `s`, converted to km via FPA_A's own s_max
(one consistent reference for the whole plot; native/undistorted/rectified
differ by at most a fraction of a row in real `s` at a given nominal
position, per Sec. 11f/11h, so this is a display-scaling choice, not a
source of error). native/undistorted rows come from
geocarb_gert.cross_band.real_s_of_row at each row pair's `ka`; rectified's
position is the shared `s_grid` value at its own row index (rebuilt fresh
here, deterministic and cheap -- see gd_joint_band_test.py's
`_shared_s_grid`).

Same chi2-outlier filtering as gd_band_stress_test_plot.py (robust
MAD-based, excludes rows reporting converged=True over an exploded state).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_band_plot.py
Output: plots/gd_joint_fpa<A>_fpa<B>[_uniform|_barcode][_noise]_summary.png (up to 6)
        plots/gd_joint_all_summary.pdf (all available cases)
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
from geocarb_gert.cross_band import real_s_of_row
from geocarb_gert.gd_render import s_max

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ORDER = {"native": 2, "rectified": 2, "undistorted": 0}
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}

SCENES = ("realistic", "uniform", "barcode")
NOISES = (False, True)


def _mode_suffix(scene: str) -> str:
    return {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}[scene]


def _result_path(fpa_a: int, fpa_b: int, scene: str, noise: bool) -> Path:
    suffix = _mode_suffix(scene) + ("_noise" if noise else "")
    return REPO_ROOT / "results" / f"gd_joint_fpa{fpa_a}_fpa{fpa_b}{suffix}.pkl"


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


def _shared_s_grid(fpa_a: int, fpa_b: int, n: int = 1024) -> np.ndarray:
    s_a = real_s_of_row(fpa_a)
    s_b = real_s_of_row(fpa_b)
    lo = max(min(s_a.min(), s_a.max()), min(s_b.min(), s_b.max()))
    hi = min(max(s_a.min(), s_a.max()), max(s_b.min(), s_b.max()))
    return np.linspace(lo, hi, n)


def _x_km(s_deg: np.ndarray, fpa_ref: int) -> np.ndarray:
    return (s_deg / s_max(fpa_ref)) * als.SLIT_HALF_KM


def make_case_figure(fpa_a: int, fpa_b: int, scene: str, noise: bool):
    path = _result_path(fpa_a, fpa_b, scene, noise)
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    pipelines = d.get("pipelines", ["native", "rectified", "undistorted"])

    s_grid_shared = _shared_s_grid(fpa_a, fpa_b)
    s_native_a = real_s_of_row(fpa_a)  # index by row number for native/undistorted

    # per-pipeline, per-row: x_km, biases, chi2
    series = {}
    for pl in pipelines:
        order = PIPELINE_ORDER[pl]
        xs, co2s, h2os, ps, chi2s, ks = [], [], [], [], [], []
        n_diverged = n_offdet = n_stalled = 0
        for key, nl in out.items():
            if key[0] != pl:
                continue
            _, _, ka, kb = key
            if nl.get("off_detector"):
                n_offdet += 1
                continue
            if nl.get("_diverged"):
                n_diverged += 1
                continue
            if not nl.get("_conv"):
                n_stalled += 1
                continue
            if pl == "rectified":
                s_here = s_grid_shared[ka]
            else:
                s_here = s_native_a[ka]
            xs.append(float(_x_km(np.array([s_here]), fpa_a)[0]))
            co2s.append(nl.get("co2", np.nan)); h2os.append(nl.get("h2o", np.nan))
            ps.append(nl.get("p_surface", np.nan)); chi2s.append(nl["_chi2"]); ks.append(ka)
        xs = np.array(xs); co2s = np.array(co2s); h2os = np.array(h2os)
        ps = np.array(ps); chi2s = np.array(chi2s); ks = np.array(ks)
        outlier = _chi2_outlier_mask(chi2s)
        order_idx = np.argsort(xs[~outlier])
        series[pl] = dict(
            xs=xs[~outlier][order_idx], co2=co2s[~outlier][order_idx],
            h2o=h2os[~outlier][order_idx], p_surface=ps[~outlier][order_idx],
            chi2=chi2s[~outlier][order_idx],
            n_outlier=int(outlier.sum()), n_diverged=n_diverged,
            n_stalled=n_stalled, n_offdet=n_offdet,
            xs_fail_diverged=[], xs_fail_stalled=[],
        )
        # failure-location x positions (recompute cheaply for the panel)
        for key, nl in out.items():
            if key[0] != pl:
                continue
            _, _, ka, kb = key
            if nl.get("off_detector") or nl.get("_diverged") or not nl.get("_conv"):
                s_here = s_grid_shared[ka] if pl == "rectified" else s_native_a[ka]
                x_here = float(_x_km(np.array([s_here]), fpa_a)[0])
                if nl.get("_diverged"):
                    series[pl]["xs_fail_diverged"].append(x_here)
                elif not nl.get("_conv") and not nl.get("off_detector"):
                    series[pl]["xs_fail_stalled"].append(x_here)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for pl in pipelines:
        s = series[pl]
        if len(s["xs"]):
            ax.plot(s["xs"], s["co2"], ".", ms=2, color=PIPELINE_COLOR[pl], label=pl, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("co2 bias vs along-slit position"); ax.set_xlabel("x [km]"); ax.set_ylabel("co2 bias [ppm]")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for pl in pipelines:
        s = series[pl]
        if len(s["xs"]):
            ax.plot(s["xs"], s["p_surface"], ".", ms=2, color=PIPELINE_COLOR[pl], label=pl, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("p_surface bias vs along-slit position"); ax.set_xlabel("x [km]"); ax.set_ylabel("p_surface bias [hPa]")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for pl in pipelines:
        s = series[pl]
        if len(s["xs"]):
            ax.plot(s["xs"], s["chi2"], ".", ms=2, color=PIPELINE_COLOR[pl], label=pl, alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("chi2_reduced vs along-slit position (outliers excluded)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("chi2 (log)"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    y0 = 0
    for pl in pipelines:
        s = series[pl]
        if s["xs_fail_diverged"]:
            ax.plot(s["xs_fail_diverged"], [y0] * len(s["xs_fail_diverged"]), "x",
                   color=PIPELINE_COLOR[pl], label=f"{pl} diverged (n={s['n_diverged']})")
        if s["xs_fail_stalled"]:
            ax.plot(s["xs_fail_stalled"], [y0 - 0.3] * len(s["xs_fail_stalled"]), "+",
                   color=PIPELINE_COLOR[pl], label=f"{pl} stalled (n={s['n_stalled']})")
        if s["n_offdet"]:
            ax.plot([], [], " ", label=f"{pl} off-detector (n={s['n_offdet']})")
        if s["n_outlier"]:
            ax.plot([], [], " ", label=f"{pl} chi2-outlier (n={s['n_outlier']})")
        y0 -= 1
    ax.set_title("Where each pipeline fails, vs along-slit position")
    ax.set_xlabel("x [km]"); ax.set_yticks([])
    ax.legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))

    noise_label = "noise" if noise else "no noise"
    fig.suptitle(f"Joint FPA{fpa_a}+FPA{fpa_b} -- {scene}, {noise_label} "
                f"(native/rectified/undistorted)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    return fig, series


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpa-a", type=int, default=0)
    ap.add_argument("--fpa-b", type=int, default=2)
    args = ap.parse_args()
    fpa_a, fpa_b = args.fpa_a, args.fpa_b

    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    pdf_path = plots_dir / f"gd_joint_fpa{fpa_a}_fpa{fpa_b}_all_summary.pdf"
    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for scene in SCENES:
            for noise in NOISES:
                label = f"FPA{fpa_a}+FPA{fpa_b} {scene} noise={int(noise)}"
                print(f"{label} ...", flush=True)
                result = make_case_figure(fpa_a, fpa_b, scene, noise)
                if result is None:
                    continue
                fig, series = result
                suffix = _mode_suffix(scene) + ("_noise" if noise else "")
                png_path = plots_dir / f"gd_joint_fpa{fpa_a}_fpa{fpa_b}{suffix}_summary.png"
                fig.savefig(png_path, dpi=120)
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {png_path.name}", flush=True)
                for pl, s in series.items():
                    if len(s["co2"]):
                        print(f"    {pl}: n={len(s['co2'])} co2 mean={np.mean(s['co2']):+.3f} "
                             f"std={np.std(s['co2']):.3f}  p_surf mean={np.mean(s['p_surface']):+.4f} "
                             f"std={np.std(s['p_surface']):.4f}  chi2 median={np.median(s['chi2']):.4g}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
