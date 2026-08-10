#!/usr/bin/env python3
"""Summary diagnostics for the joint multi-band battery (gd_test.py
/ KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11h) -- the joint-retrieval counterpart of
gd_band_stress_test_plot.py: bias vs. along-slit position for whichever
gas(es) the given band pair actually retrieves (read directly off the
result dict's own bias keys, e.g. co2 for FPA0+FPA2, but ch4+co -- no co2
at all -- for FPA0+FPA3) plus p_surface, chi2 vs. position, and a
failure-location panel, for all three pipelines (native/rectified/
undistorted) at once, for each of the 6 scene x noise combinations, for an
ordered set of >=2 GeoCarb bands.

Position axis: real slit angle `s`, converted to km via the reference band's
(fpas[0]'s) own s_max (one consistent reference for the whole plot; every
band's own pipeline differs by at most a fraction of a row in real `s` at a
given nominal position, per Sec. 11f/11h, so this is a display-scaling
choice, not a source of error). native/undistorted rows come from
geocarb_gert.cross_band.real_s_of_row at each row tuple's reference-band
entry; rectified's position is the shared `s_grid` value at its own row
index (rebuilt fresh here, deterministic and cheap -- see
gd_test.py's `_shared_s_grid`).

Same chi2-outlier filtering as gd_band_stress_test_plot.py (robust
MAD-based, excludes rows reporting converged=True over an exploded state).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot.py --fpas 0,2
Output: plots/gd_joint_<fpas_tag>[_uniform|_barcode][_noise]_summary.png (up to 6)
        plots/gd_joint_<fpas_tag>_all_summary.pdf (all available cases)
        (fpas_tag = "fpa0_fpa2", "fpa0_fpa1_fpa2_fpa3", etc. -- see
        geocarb_gert.cross_band.fpas_tag)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ORDER = {"native": 2, "rectified": 2, "undistorted": 2}  # undistorted floats dispersion too, since 2026-07-28 (Sec. 11k) -- a numerical workaround, not a physical correction
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}

SCENES = ("realistic", "uniform", "barcode")
NOISES = (False, True)


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


def _shared_s_grid(fpas, n: int = 1024) -> np.ndarray:
    los, his = [], []
    for fpa in fpas:
        s = real_s_of_row(fpa)
        los.append(min(s.min(), s.max()))
        his.append(max(s.min(), s.max()))
    return np.linspace(max(los), min(his), n)


def _x_km(s_deg: np.ndarray, fpa_ref: int) -> np.ndarray:
    return (s_deg / s_max(fpa_ref)) * als.SLIT_HALF_KM


def _gas_list(out) -> list:
    """Which gases this band pair actually retrieves -- e.g. co2 for
    FPA0+FPA2, but ch4+co (no co2 at all) for FPA0+FPA3. Read directly off
    the result dicts' own bias keys (set unconditionally by
    gd_test.py's _joint_retrieve for every gas it retrieves)
    rather than assumed, since which gas(es) a pair produces depends on
    which bands are paired."""
    gases = set()
    for nl in out.values():
        # only genuine converged retrievals set gas-bias keys; failed/
        # off-detector rows short-circuit in _worker/_joint_retrieve with a
        # status-only dict (e.g. {"off_detector": True}) that would
        # otherwise leak "off_detector" itself into the gas list
        if not nl.get("_conv") or nl.get("off_detector") or nl.get("_diverged"):
            continue
        for k in nl.keys():
            if k == "p_surface" or k.startswith("_"):
                continue
            gases.add(k)
    return sorted(gases)


def _gas_unit(gas: str) -> str:
    return "ppb" if gas in ("ch4", "co") else "ppm"


def make_case_figure(fpas, scene: str, noise: bool):
    path = _result_path(fpas, scene, noise)
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    pipelines = d.get("pipelines", ["native", "rectified", "undistorted"])
    fpa_ref = fpas[0]

    s_grid_shared = _shared_s_grid(fpas)
    s_native_ref = real_s_of_row(fpa_ref)  # index by row number for native/undistorted
    gas_list = _gas_list(out)

    # per-pipeline, per-row: x_km, biases, chi2
    series = {}
    for pl in pipelines:
        xs, ps, chi2s, ks = [], [], [], []
        gas_vals = {g: [] for g in gas_list}
        n_diverged = n_offdet = n_stalled = 0
        for key, nl in out.items():
            if key[0] != pl:
                continue
            _, _, rows = key
            ka = rows[0]
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
                s_here = s_native_ref[ka]
            xs.append(float(_x_km(np.array([s_here]), fpa_ref)[0]))
            for g in gas_list:
                gas_vals[g].append(nl.get(g, np.nan))
            ps.append(nl.get("p_surface", np.nan)); chi2s.append(nl["_chi2"]); ks.append(ka)
        xs = np.array(xs); ps = np.array(ps); chi2s = np.array(chi2s); ks = np.array(ks)
        gas_arrs = {g: np.array(v) for g, v in gas_vals.items()}
        outlier = _chi2_outlier_mask(chi2s)
        order_idx = np.argsort(xs[~outlier])
        series[pl] = dict(
            xs=xs[~outlier][order_idx],
            gases={g: arr[~outlier][order_idx] for g, arr in gas_arrs.items()},
            p_surface=ps[~outlier][order_idx],
            chi2=chi2s[~outlier][order_idx],
            n_outlier=int(outlier.sum()), n_diverged=n_diverged,
            n_stalled=n_stalled, n_offdet=n_offdet,
            xs_fail_diverged=[], xs_fail_stalled=[],
        )
        # failure-location x positions (recompute cheaply for the panel)
        for key, nl in out.items():
            if key[0] != pl:
                continue
            _, _, rows = key
            ka = rows[0]
            if nl.get("off_detector") or nl.get("_diverged") or not nl.get("_conv"):
                s_here = s_grid_shared[ka] if pl == "rectified" else s_native_ref[ka]
                x_here = float(_x_km(np.array([s_here]), fpa_ref)[0])
                if nl.get("_diverged"):
                    series[pl]["xs_fail_diverged"].append(x_here)
                elif not nl.get("_conv") and not nl.get("off_detector"):
                    series[pl]["xs_fail_stalled"].append(x_here)

    # n_gas bias panels + p_surface + chi2 + failure-location, 2 columns --
    # reduces to the original fixed 2x2 layout whenever n_gas==1 (co2-only
    # pairs like FPA0+FPA1/FPA0+FPA2), grows for pairs retrieving more than
    # one gas (e.g. FPA0+FPA3 retrieves ch4+co, no co2 at all).
    n_panels = len(gas_list) + 3
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.ravel()
    panel_i = 0

    for g in gas_list:
        ax = axes_flat[panel_i]; panel_i += 1
        for pl in pipelines:
            s = series[pl]
            if len(s["xs"]):
                ax.plot(s["xs"], s["gases"][g], ".", ms=2, color=PIPELINE_COLOR[pl], label=pl, alpha=0.6)
        ax.axhline(0, color="k", lw=0.5)
        unit = _gas_unit(g)
        ax.set_title(f"{g} bias vs along-slit position"); ax.set_xlabel("x [km]"); ax.set_ylabel(f"{g} bias [{unit}]")
        ax.legend(fontsize=7)

    ax = axes_flat[panel_i]; panel_i += 1
    for pl in pipelines:
        s = series[pl]
        if len(s["xs"]):
            ax.plot(s["xs"], s["p_surface"], ".", ms=2, color=PIPELINE_COLOR[pl], label=pl, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("p_surface bias vs along-slit position"); ax.set_xlabel("x [km]"); ax.set_ylabel("p_surface bias [hPa]")
    ax.legend(fontsize=7)

    ax = axes_flat[panel_i]; panel_i += 1
    for pl in pipelines:
        s = series[pl]
        if len(s["xs"]):
            ax.plot(s["xs"], s["chi2"], ".", ms=2, color=PIPELINE_COLOR[pl], label=pl, alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("chi2_reduced vs along-slit position (outliers excluded)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("chi2 (log)"); ax.legend(fontsize=7)

    ax = axes_flat[panel_i]; panel_i += 1
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

    for j in range(panel_i, len(axes_flat)):
        axes_flat[j].set_visible(False)

    band_label = "+".join(f"FPA{f}" for f in fpas)
    noise_label = "noise" if noise else "no noise"
    fig.suptitle(f"Joint {band_label} -- {scene}, {noise_label} "
                f"(native/rectified/undistorted)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    return fig, series


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpas", type=str, default="0,2",
                    help="comma-separated list of >=2 FPA indices, matching "
                         "the gd_test.py run to analyze")
    args = ap.parse_args()
    fpas = [int(x) for x in args.fpas.split(",")]

    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    tag = fpas_tag(fpas)
    pdf_path = plots_dir / f"gd_joint_{tag}_all_summary.pdf"
    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for scene in SCENES:
            for noise in NOISES:
                label = f"{'+'.join(f'FPA{f}' for f in fpas)} {scene} noise={int(noise)}"
                print(f"{label} ...", flush=True)
                result = make_case_figure(fpas, scene, noise)
                if result is None:
                    continue
                fig, series = result
                suffix = _mode_suffix(scene) + ("_noise" if noise else "")
                png_path = plots_dir / f"gd_joint_{tag}{suffix}_summary.png"
                fig.savefig(png_path, dpi=120)
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {png_path.name}", flush=True)
                for pl, s in series.items():
                    if len(s["xs"]):
                        gas_str = "  ".join(f"{g} mean={np.mean(s['gases'][g]):+.3f} "
                                            f"std={np.std(s['gases'][g]):.3f}" for g in s["gases"])
                        print(f"    {pl}: n={len(s['xs'])}  {gas_str}  p_surf mean={np.mean(s['p_surface']):+.4f} "
                             f"std={np.std(s['p_surface']):.4f}  chi2 median={np.median(s['chi2']):.4g}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
