#!/usr/bin/env python3
"""Representative post-fit spectral residual shapes, by band, for the joint
multi-band battery (gd_joint_band_test.py / KEYSTONE_SMILE_BIAS_PLAN.md
Sec. 11h) -- the joint-retrieval counterpart of
gd_band_stress_test_plot_residual_spectra.py.

gd_joint_band_test.py's saved output only kept the scalar bias/chi2 per
row, not the residual spectrum itself (an oversight found when this script
was requested after the battery had already completed) -- so this script
re-renders each case (same setup: fpas, scene, noise, snr, seed, all read
back from the saved .pkl's own metadata) and re-runs the joint retrieval
fresh, but ONLY for a small set of representative rows, not the full
sweep: 12 positions selected via native's chi2 (best, worst, and 10
spanning the slit among the remaining in-family rows), same method as the
single-band script. Cheap -- a few dozen retrievals per case, not ~3000.

Each figure panel shows all N bands' residuals side by side (they're on
different wavenumber axes and can't be overlaid), for whichever of
native/rectified/undistorted actually has an in-family fit at that row
(rectified's own representative row is its nearest shared-`s_grid` index to
the same along-slit position, since it isn't indexed the same way as
native/undistorted's row tuples).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_band_plot_residual_spectra.py --fpas 0,2
Output: plots/gd_joint_<fpas_tag>[_uniform|_barcode][_noise]_residual_spectra.png (up to 6)
        plots/gd_joint_<fpas_tag>_all_residual_spectra.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

import geosat_geometry as gg
from geocarb_gert import along_slit_scene as als, sample_geometries
from geocarb_gert.cross_band import fpas_tag, real_s_of_row
from geocarb_gert.gd_render import s_max

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gd_joint_band_test import (  # noqa: E402
    _band_setup, _shared_s_grid, _native_row, _undistorted_row, _rectified_row,
    WELL_MIXED_GASES, DEFAULT_SNR_BY_FPA,
)
from geocarb_gert import gd_render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
PIPELINE_ORDER = {"native": 2, "rectified": 2, "undistorted": 2}  # undistorted floats dispersion too, since 2026-07-28 (Sec. 11k) -- a numerical workaround, not a physical correction
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}

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


def _joint_retrieve_with_residual(bands, atm_center, absco, geo, solar, snr, nus, ys, order):
    n_bands = len(bands)
    windows = [SpectralWindow(wn_min=b["wn_min"], wn_max=b["wn_max"],
                              ils=ILS(type="gaussian", fwhm=b["fwhm_cm"]),
                              molecules=b["mols"], label=b["label"], obs_grid=nu)
              for b, nu in zip(bands, nus)]
    inst = Instrument(windows=windows, snr=snr)
    y_true = np.concatenate(ys)
    sigma = np.concatenate([np.full(len(y), b["sigma_band"]) for y, b in zip(ys, bands)])
    Sy_inv = np.diag(1.0 / sigma ** 2)
    fm = ForwardModel(atm_center, absco, inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    prior_albedo = np.array([b["albedo"] for b in bands])
    gases = sorted({m for b in bands for m in b["mols"]
                    if m != "h2o" and m not in WELL_MIXED_GASES}) + ["h2o"]
    sv = StateVector.gas_scaling(prior_albedo=prior_albedo, prior_albedo_slope=np.zeros(n_bands),
                                 gases=gases, gas_uncerts={"h2o": 0.60},
                                 include_dispersion=(order > 0),
                                 dispersion_order=max(order, 0), dispersion_uncert=2.0)
    ret = GERTRetrieval(fm, y_true, Sy_inv, sv, prior_albedo=prior_albedo,
                        prior_albedo_slope=np.zeros(n_bands), analytical_jacobians=True,
                        max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            res = ret.run()
    except (ValueError, np.linalg.LinAlgError):
        return None
    resid = y_true - res.y_ret
    lens = [len(y) for y in ys]
    resids = np.split(resid, np.cumsum(lens)[:-1])
    return dict(nus=list(nus), resids=resids, chi2=float(res.chisq_reduced), conv=bool(res.converged))


def make_case_figure(fpas, scene: str, noise: bool):
    path = _result_path(fpas, scene, noise)
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    uniform, barcode = d["uniform"], d["barcode"]
    barcode_bars = d.get("barcode_bars") or 32
    noise_flag, noise_seed = d["noise"], d.get("noise_seed") or 0
    snr = d["snr"]

    reps = _select_representative(out, fpas)
    if not reps:
        print(f"  no in-family native rows for {path.name}, skipping")
        return None

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)

    bands = [_band_setup(fpa, atm_center, absco, geo, solar, snr, 400, None,
                        uniform, barcode, barcode_bars, noise_flag, noise_seed)
            for fpa in fpas]

    s_grid_shared = _shared_s_grid(fpas)
    Rimgs = [gd_render.rectify(fpa, b["A"], s_grid_shared, b["wn_grid"])
            for fpa, b in zip(fpas, bands)]

    n_bands = len(fpas)
    n = len(reps)
    fig, axes = plt.subplots(n, n_bands, figsize=(5.5 * n_bands, 2.6 * n), squeeze=False)

    for row_i, (rows, label) in enumerate(reps):
        s_here = real_s_of_row(fpas[0], np.array([float(rows[0])]))[0]
        k_rect = int(np.argmin(np.abs(s_grid_shared - s_here)))

        results = {}
        nus, ys = zip(*[_native_row(b, r) for b, r in zip(bands, rows)])
        r = _joint_retrieve_with_residual(bands, atm_center, absco, geo, solar, snr,
                                          list(nus), list(ys), 2)
        if r is not None:
            results["native"] = r

        nus, ys = zip(*[_undistorted_row(b, r) for b, r in zip(bands, rows)])
        r = _joint_retrieve_with_residual(bands, atm_center, absco, geo, solar, snr,
                                          list(nus), list(ys), 2)
        if r is not None:
            results["undistorted"] = r

        pairs = [_rectified_row(b, Rimg, k_rect) for b, Rimg in zip(bands, Rimgs)]
        if all(nu is not None for nu, _ in pairs):
            nus_r, ys_r = zip(*pairs)
            r = _joint_retrieve_with_residual(bands, atm_center, absco, geo, solar, snr,
                                              list(nus_r), list(ys_r), 2)
            if r is not None:
                results["rectified"] = r

        any_data = False
        for pl, r in results.items():
            if not r["conv"]:
                continue
            any_data = True
            for j in range(n_bands):
                axes[row_i, j].plot(r["nus"][j], r["resids"][j], lw=0.6,
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
    noise_label = "noise" if noise_flag else "no noise"
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
                         "the gd_joint_band_test.py run to analyze")
    args = ap.parse_args()
    fpas = [int(x) for x in args.fpas.split(",")]

    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    tag = fpas_tag(fpas)
    pdf_path = plots_dir / f"gd_joint_{tag}_all_residual_spectra.pdf"
    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for scene in SCENES:
            for noise in NOISES:
                label = f"{'+'.join(f'FPA{f}' for f in fpas)} {scene} noise={int(noise)}"
                print(f"{label} ...", flush=True)
                fig = make_case_figure(fpas, scene, noise)
                if fig is None:
                    continue
                suffix = _mode_suffix(scene) + ("_noise" if noise else "")
                png_path = plots_dir / f"gd_joint_{tag}{suffix}_residual_spectra.png"
                fig.savefig(png_path, dpi=120)
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {png_path.name}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
