#!/usr/bin/env python3
"""Representative post-fit spectral residual shapes, by band, for the joint
two-band battery (gd_joint_band_test.py / KEYSTONE_SMILE_BIAS_PLAN.md
Sec. 11h) -- the joint-retrieval counterpart of
gd_band_stress_test_plot_residual_spectra.py.

gd_joint_band_test.py's saved output only kept the scalar bias/chi2 per
row, not the residual spectrum itself (an oversight found when this script
was requested after the battery had already completed) -- so this script
re-renders each case (same setup: fpa_a/fpa_b, scene, noise, snr, seed, all
read back from the saved .pkl's own metadata) and re-runs the joint
retrieval fresh, but ONLY for a small set of representative rows, not the
full sweep: 12 positions selected via native's chi2 (best, worst, and 10
spanning the slit among the remaining in-family rows), same method as the
single-band script. Cheap -- a few dozen retrievals per case, not ~3000.

Each figure panel shows BOTH bands' residual side by side (they're on
different wavenumber axes and can't be overlaid), for whichever of
native/rectified/undistorted actually has an in-family fit at that row
(rectified's own representative row is its nearest shared-`s_grid` index to
the same along-slit position, since it isn't indexed the same way as
native/undistorted's row pairs).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_band_plot_residual_spectra.py
Output: plots/gd_joint_fpa<A>_fpa<B>[_uniform|_barcode][_noise]_residual_spectra.png (up to 6)
        plots/gd_joint_all_residual_spectra.pdf
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
from geocarb_gert.cross_band import real_s_of_row
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
from geocarb_gert.cross_band import nearest_row_pairing  # noqa: E402
from geocarb_gert import gd_render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
PIPELINE_ORDER = {"native": 2, "rectified": 2, "undistorted": 0}
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}

SCENES = ("realistic", "uniform", "barcode")
NOISES = (False, True)
N_SPAN = 10


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


def _select_representative(out, fpa_a, fpa_b, n_span=N_SPAN):
    """Select (ka, kb, x_km) triples via native's chi2 -- best, worst, and
    n_span spanning the slit -- same method as the single-band script."""
    ks, chi2s, kbs = [], [], []
    for key, nl in out.items():
        if key[0] != "native":
            continue
        if nl.get("off_detector") or nl.get("_diverged") or not nl.get("_conv"):
            continue
        ks.append(key[2]); kbs.append(key[3]); chi2s.append(nl["_chi2"])
    ks = np.array(ks); kbs = np.array(kbs); chi2s = np.array(chi2s, dtype=float)
    outlier = _chi2_outlier_mask(chi2s)
    ks, kbs, chi2s = ks[~outlier], kbs[~outlier], chi2s[~outlier]
    if len(ks) == 0:
        return []

    s_a = real_s_of_row(fpa_a)
    x_km = (s_a[ks] / s_max(fpa_a)) * als.SLIT_HALF_KM

    order_by_chi2 = np.argsort(chi2s)
    best_i, worst_i = order_by_chi2[0], order_by_chi2[-1]
    chosen = {int(ks[best_i]): (int(kbs[best_i]), f"best (chi2={chi2s[best_i]:.3g})")}
    chosen[int(ks[worst_i])] = (int(kbs[worst_i]), f"worst (chi2={chi2s[worst_i]:.3g})")

    remaining = np.ones(len(ks), dtype=bool)
    remaining[[best_i, worst_i]] = False
    rem_ks, rem_kbs, rem_x = ks[remaining], kbs[remaining], x_km[remaining]
    order_by_x = np.argsort(rem_x)
    if len(order_by_x) > 0:
        span_idx = np.linspace(0, len(order_by_x) - 1, min(n_span, len(order_by_x))).astype(int)
        span_idx = np.unique(span_idx)
        for i in order_by_x[span_idx]:
            k = int(rem_ks[i])
            chosen[k] = (int(rem_kbs[i]), f"x={rem_x[i]:.0f}km")

    items = [(ka, kb, label) for ka, (kb, label) in chosen.items()]
    x_of = dict(zip(ks.tolist(), x_km.tolist()))
    items.sort(key=lambda t: x_of[t[0]])
    return items


def _joint_retrieve_with_residual(band_a, band_b, atm_center, absco, geo, solar, snr,
                                  nu_a, y_a, nu_b, y_b, order):
    win_a = SpectralWindow(wn_min=band_a["wn_min"], wn_max=band_a["wn_max"],
                           ils=ILS(type="gaussian", fwhm=band_a["fwhm_cm"]),
                           molecules=band_a["mols"], label=band_a["label"], obs_grid=nu_a)
    win_b = SpectralWindow(wn_min=band_b["wn_min"], wn_max=band_b["wn_max"],
                           ils=ILS(type="gaussian", fwhm=band_b["fwhm_cm"]),
                           molecules=band_b["mols"], label=band_b["label"], obs_grid=nu_b)
    inst = Instrument(windows=[win_a, win_b], snr=snr)
    y_true = np.concatenate([y_a, y_b])
    sigma = np.concatenate([np.full(len(y_a), band_a["sigma_band"]),
                            np.full(len(y_b), band_b["sigma_band"])])
    Sy_inv = np.diag(1.0 / sigma ** 2)
    fm = ForwardModel(atm_center, absco, inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    prior_albedo = np.array([band_a["albedo"], band_b["albedo"]])
    gases = sorted({m for m in (band_a["mols"] + band_b["mols"])
                    if m != "h2o" and m not in WELL_MIXED_GASES}) + ["h2o"]
    sv = StateVector.gas_scaling(prior_albedo=prior_albedo, prior_albedo_slope=np.zeros(2),
                                 gases=gases, gas_uncerts={"h2o": 0.60},
                                 include_dispersion=(order > 0),
                                 dispersion_order=max(order, 0), dispersion_uncert=2.0)
    ret = GERTRetrieval(fm, y_true, Sy_inv, sv, prior_albedo=prior_albedo,
                        prior_albedo_slope=np.zeros(2), analytical_jacobians=True,
                        max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            res = ret.run()
    except (ValueError, np.linalg.LinAlgError):
        return None
    resid = y_true - res.y_ret
    n_a = len(y_a)
    return dict(nu_a=nu_a, resid_a=resid[:n_a], nu_b=nu_b, resid_b=resid[n_a:],
               chi2=float(res.chisq_reduced), conv=bool(res.converged))


def make_case_figure(fpa_a: int, fpa_b: int, scene: str, noise: bool):
    path = _result_path(fpa_a, fpa_b, scene, noise)
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

    reps = _select_representative(out, fpa_a, fpa_b)
    if not reps:
        print(f"  no in-family native rows for {path.name}, skipping")
        return None

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)

    band_a = _band_setup(fpa_a, atm_center, absco, geo, solar, snr, 400, None,
                         uniform, barcode, barcode_bars, noise_flag, noise_seed)
    band_b = _band_setup(fpa_b, atm_center, absco, geo, solar, snr, 400, None,
                         uniform, barcode, barcode_bars, noise_flag, noise_seed)

    s_grid_shared = _shared_s_grid(fpa_a, fpa_b)
    Rimg_a = gd_render.rectify(fpa_a, band_a["A"], s_grid_shared, band_a["wn_grid"])
    Rimg_b = gd_render.rectify(fpa_b, band_b["A"], s_grid_shared, band_b["wn_grid"])

    n = len(reps)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), squeeze=False)

    for row_i, (ka, kb, label) in enumerate(reps):
        s_here = real_s_of_row(fpa_a, np.array([float(ka)]))[0]
        k_rect = int(np.argmin(np.abs(s_grid_shared - s_here)))

        results = {}
        nu_a, y_a = _native_row(band_a, ka)
        nu_b, y_b = _native_row(band_b, kb)
        r = _joint_retrieve_with_residual(band_a, band_b, atm_center, absco, geo, solar, snr,
                                          nu_a, y_a, nu_b, y_b, 2)
        if r is not None:
            results["native"] = r

        nu_a, y_a = _undistorted_row(band_a, ka)
        nu_b, y_b = _undistorted_row(band_b, kb)
        r = _joint_retrieve_with_residual(band_a, band_b, atm_center, absco, geo, solar, snr,
                                          nu_a, y_a, nu_b, y_b, 0)
        if r is not None:
            results["undistorted"] = r

        nu_a, y_a = _rectified_row(band_a, Rimg_a, k_rect)
        nu_b, y_b = _rectified_row(band_b, Rimg_b, k_rect)
        if nu_a is not None and nu_b is not None:
            r = _joint_retrieve_with_residual(band_a, band_b, atm_center, absco, geo, solar, snr,
                                              nu_a, y_a, nu_b, y_b, 2)
            if r is not None:
                results["rectified"] = r

        ax_a, ax_b = axes[row_i, 0], axes[row_i, 1]
        any_data = False
        for pl, r in results.items():
            if not r["conv"]:
                continue
            any_data = True
            ax_a.plot(r["nu_a"], r["resid_a"], lw=0.6, color=PIPELINE_COLOR[pl], label=pl, alpha=0.8)
            ax_b.plot(r["nu_b"], r["resid_b"], lw=0.6, color=PIPELINE_COLOR[pl], label=pl, alpha=0.8)
        for ax, fpa in ((ax_a, fpa_a), (ax_b, fpa_b)):
            ax.axhline(0, color="k", lw=0.4)
            ax.tick_params(labelsize=6)
        ax_a.set_title(f"row {ka}: {label}  (FPA{fpa_a})", fontsize=8)
        ax_b.set_title(f"grid/row match: {label}  (FPA{fpa_b})", fontsize=8)
        if any_data:
            ax_a.legend(fontsize=6)
        else:
            ax_a.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_a.transAxes,
                     color="gray", fontsize=8)

    noise_label = "noise" if noise_flag else "no noise"
    fig.suptitle(f"Joint FPA{fpa_a}+FPA{fpa_b} -- {scene}, {noise_label} "
                f"(residual spectra by band: best/worst native chi2 + {N_SPAN} spanning rows)",
                fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpa-a", type=int, default=0)
    ap.add_argument("--fpa-b", type=int, default=2)
    args = ap.parse_args()
    fpa_a, fpa_b = args.fpa_a, args.fpa_b

    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    pdf_path = plots_dir / f"gd_joint_fpa{fpa_a}_fpa{fpa_b}_all_residual_spectra.pdf"
    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for scene in SCENES:
            for noise in NOISES:
                label = f"FPA{fpa_a}+FPA{fpa_b} {scene} noise={int(noise)}"
                print(f"{label} ...", flush=True)
                fig = make_case_figure(fpa_a, fpa_b, scene, noise)
                if fig is None:
                    continue
                suffix = _mode_suffix(scene) + ("_noise" if noise else "")
                png_path = plots_dir / f"gd_joint_fpa{fpa_a}_fpa{fpa_b}{suffix}_residual_spectra.png"
                fig.savefig(png_path, dpi=120)
                pdf.savefig(fig)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {png_path.name}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
