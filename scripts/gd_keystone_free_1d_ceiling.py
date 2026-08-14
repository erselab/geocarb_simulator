#!/usr/bin/env python3
"""Independent, non-joint-block check of the "keystone-free, PSF-kept"
ceiling idea, proposed after the joint-block-based keystone-free ceiling
(gd_joint_block_keystone_free_sweep.py) came back showing essentially the
same bias-vs-row excursions with and without keystone -- a result that's
either a genuine, honest finding (the joint block's own forward model
already resolves per-pixel eta correctly regardless of keystone, so its
remaining bias is purely resolution floor) or a sign that something in
the joint block's own machinery (attribution, GN solver, regularization)
is broken in a way that happens to look the same either way. The only way
to tell those apart is to ask the same physical question with a
completely different, already-validated code path.

This script does exactly that, and touches no joint-block code at all:

1. Build a plain (1024, 1024) array of per-row spectra -- one eta per
   row (that row's own center-column value, the same "no keystone"
   convention gd_test._undistorted_row already uses), broadcast across
   all 1024 real per-column wavenumbers (smile/dispersion kept), ILS-
   convolved with the real ILS. Unlike _undistorted_row, which stops
   here, the real spatial PSF blur is then applied once across the whole
   assembled array (gaussian_blur_rows), matching what the joint block's
   own forward model always keeps.
2. Retrieve CO2 with the existing, already-validated single-atmosphere-
   per-row machinery (gd_test._joint_retrieve -- the exact same
   GERTRetrieval/StateVector/ForwardModel call native and undistorted
   already use), one full retrieval per row, native 1024-row resolution
   -- no G bins, no attribution, no custom Gauss-Newton solver, no
   nearest_bin_scene, nothing from geocarb_gert/joint_block or the
   scripts/gd_joint_block_* family anywhere in this path.

Two retrievals are run per row, against the identical (nu, y) measurement:

- "free": _joint_retrieve's own full free-nuisance-parameter state vector
  (H2O/CH4/CO/p_surface/albedo/dispersion all retrieved, exactly as
  native/undistorted do). Kept as a reference -- this is what an
  already-trusted, unmodified pipeline call sees.
- "fixed": a new state vector, built directly from GERTRetrieval/
  StateVector/ForwardModel (same GERT classes _joint_retrieve itself
  uses, no new solver code), with every nuisance element FROZEN at this
  row's own local truth and only co2_scale free -- matching the joint
  block's own local-truth nuisance idealization exactly. The forward
  model's own atmosphere is als.atmosphere_at(x_km_row) (that row's real
  truth, not the single shared atm_center _joint_retrieve normally uses),
  so freezing H2O/CH4/CO/p_scale/T_offset at their prior values is
  freezing them at the truth, not at some other row's or the scene
  average's atmosphere. Dispersion is still INCLUDED (order=2) but frozen
  at 0, not omitted -- gd_test.py's own module docstring documents that
  omitting dispersion entirely reintroduces a real, already-diagnosed
  grid-snapped-ILS chi2 floor (~0.05, see gd_joint_undistorted_dispersion_
  diag.py) unrelated to atmospheric retrieval quality; freezing rather
  than omitting keeps the numerical fix while still holding the state at
  truth.

This exists to separate two possible explanations for why the joint
block's own keystone-present and keystone-free results looked suspiciously
identical: either (a) the joint block's forward model already resolves
per-pixel eta correctly regardless of keystone (an honest finding), or (b)
something in its own machinery is broken in a way that looks the same
either way. Comparing "free" vs "fixed" here additionally separates a
third possibility: that nuisance-parameter cross-talk in a full free
retrieval, not keystone or a joint-block bug, is what's driving any
excursions -- since "fixed" removes that channel entirely while "free"
still has it.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_keystone_free_1d_ceiling.py \\
        [--row-step 1] [--order 2] [--n-workers N]
Output: results/gd_keystone_free_1d_ceiling_fpa2.pkl
        plots/keystone_free_1d_ceiling/gd_keystone_free_1d_ceiling_fpa2.png (default psf_fwhm)
        plots/joint_block/gd_keystone_free_1d_ceiling_fpa2*.png (non-default psf_fwhm --
        these runs exist specifically as joint-block validation counterparts)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, _eta_of  # noqa: E402 -- FPA/GERT_ROOT/_eta_of only, no forward-model code

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, geocarb_noise_model, sample_geometries  # noqa: E402
from geocarb_gert.focalplane import gaussian_blur_rows  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import _diagonal_ils_convolve, available_cpus  # noqa: E402

from gert.forward_model import ForwardModel  # noqa: E402
from gert.instrument import ILS, SpectralWindow  # noqa: E402
from gert.instrument_config import Instrument  # noqa: E402
from gert.retrieval import GERTRetrieval, StateVector  # noqa: E402
from gert.rt_solver import SingleScatterSolver  # noqa: E402

ROW_MAX_IDX = 1023
_CTX = {}


def _joint_retrieve_fixed_nuisance(band: dict, row: int, nu: np.ndarray, y: np.ndarray,
                                   atm_row, absco, geo, solar, xtrue_row: dict, order: int = 2):
    """Single-band GERTRetrieval built directly from GERT's own classes
    (same ones gd_test._joint_retrieve uses -- ForwardModel, StateVector,
    GERTRetrieval -- no new solver code), with every nuisance element
    frozen at ITS OWN local truth (atm_row = als.atmosphere_at(x_km_row)
    supplies the true H2O/CH4/CO/p_surface directly; T_offset=0/p_scale=1/
    gas_scale=1 frozen at prior leaves atm_row entirely unperturbed for
    those), only co2_scale free -- the joint block's own idealization,
    reproduced here through the unmodified single-row retrieval machinery
    instead of the joint block's own forward model/solver."""
    fpa = band["fpa"]
    win = SpectralWindow(wn_min=band["wn_min"], wn_max=band["wn_max"],
                         ils=ILS(type="gaussian", fwhm=band["fwhm_cm"]),
                         molecules=band["mols"], label=band["label"], obs_grid=nu)
    noise_model = geocarb_noise_model(fpa)
    inst = Instrument(windows=[win], noise_model=noise_model)
    sigma = noise_model.sigma([y], [win])
    Sy_inv = np.diag(1.0 / sigma ** 2)

    fm = ForwardModel(atm_row, absco, inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    prior_albedo = np.array([band["albedo"]])
    gases = sorted({m for m in band["mols"] if m != "h2o" and m not in gdt.WELL_MIXED_GASES}) + ["h2o"]
    sv = StateVector.gas_scaling(prior_albedo=prior_albedo, prior_albedo_slope=np.zeros(1),
                                 gases=gases, gas_uncerts={"co2": 0.10, "h2o": 0.60},
                                 include_dispersion=True, dispersion_order=order, dispersion_uncert=2.0)
    freeze_names = [f"{g}_scale" for g in gases if g != "co2"] + ["T_offset", "p_scale",
                   "albedo_0", "albedo_slope_0"] + [e.name for e in sv.elements if e.name.startswith("disp_a")]
    sv.freeze(*freeze_names)

    ret = GERTRetrieval(fm, y, Sy_inv, sv, prior_albedo=prior_albedo,
                        prior_albedo_slope=np.zeros(1), analytical_jacobians=True,
                        max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
    with np.errstate(over="ignore", invalid="ignore"):
        res = ret.run()
    names = sv.names
    prior_val = float(np.mean(atm_row.gases["co2"])) * 1e6
    retrieved_val = prior_val * res.x_ret[names.index("co2_scale")]
    return {"co2": retrieved_val - float(xtrue_row["co2"]), "_chi2": float(res.chisq_reduced),
           "_conv": res.converged, "_diverged": ("gert: chi2 non-finite" if res.diverged else None)}


def build_no_keystone_psf_image(band: dict, spatial_psf_fwhm_px: float = 1.5) -> np.ndarray:
    """Plain (1024, 1024) array in native pixel-column order (NOT
    frequency-sorted, so the real spatial PSF blur below is applied at
    fixed physical column across rows, matching gd_render.image()'s own
    convention). Row i's spectrum is band['radiance'] evaluated at ONE
    eta (band['x_km_of_row'][i]/SLIT_HALF_KM, the same value gd_test.
    _undistorted_row already uses) broadcast across all 1024 columns'
    real wavenumbers -- exactly _undistorted_row's own per-row math,
    just assembled into a full array before -- not instead of -- the PSF
    blur. Depends only on als/gd_polynomials/gd_render primitives and
    gd_test's own band dict; no joint-block module is imported."""
    fpa = band["fpa"]
    cols = np.arange(1024.0)
    A = np.empty((1024, 1024), dtype=float)
    for row in range(1024):
        lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(row)))
        nu_row = 1e4 / lam_row
        eta_row = band["x_km_of_row"][row] / als.SLIT_HALF_KM
        S_row = np.asarray(band["radiance"](np.full(1024, eta_row)), dtype=float)
        A[row] = _diagonal_ils_convolve(band["wn_hires"], S_row, nu_row, band["ils"])
    return gaussian_blur_rows(A, spatial_psf_fwhm_px)


def _no_keystone_psf_row(band: dict, A_nk: np.ndarray, row: int):
    """Same extraction/frequency-ordering convention as gd_test._native_row
    / _undistorted_row (ascending-nu order, reversing if needed), reading
    from the precomputed no-keystone+PSF array instead of band['A']."""
    fpa = band["fpa"]
    cols = np.arange(1024.0)
    lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(row)))
    nu_row = 1e4 / lam_row
    y_dist = A_nk[row, :].copy()
    if nu_row[0] < nu_row[-1]:
        nu_row = nu_row[::-1]
        y_dist = y_dist[::-1]
    return nu_row, y_dist


def _worker_row(row: int):
    band, A_nk, order = _CTX["band"], _CTX["A_nk"], _CTX["order"]
    absco, geo, solar = _CTX["absco"], _CTX["geo"], _CTX["solar"]
    uniform, atm_center = _CTX["uniform"], _CTX["atm_center"]
    xt = {gas: float(band["xtrue_of_row"][gas][row]) for gas in ("co2", "ch4", "co", "h2o")}
    xt["p_surface"] = float(band["xtrue_of_row"]["p_surface"][row])
    nu, y = _no_keystone_psf_row(band, A_nk, row)

    try:
        nl_free = gdt._joint_retrieve([nu], [y], order, [xt], [band])
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        nl_free = {"_chi2": np.nan, "_conv": False, "_diverged": f"{type(e).__name__}: {e}"}

    try:
        # uniform scene: the true atmosphere really is atm_center
        # everywhere, not a position-dependent als.atmosphere_at(x_km_row)
        # -- using the latter here would itself inject a fake along-slit
        # variation the true scene doesn't have.
        atm_row = atm_center if uniform else als.atmosphere_at(float(band["x_km_of_row"][row]))
        nl_fixed = _joint_retrieve_fixed_nuisance(band, row, nu, y, atm_row, absco, geo, solar, xt, order)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        nl_fixed = {"_chi2": np.nan, "_conv": False, "_diverged": f"{type(e).__name__}: {e}"}

    return row, nl_free, nl_fixed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-step", type=int, default=1)
    ap.add_argument("--order", type=int, default=2, help="dispersion polynomial order -- 2 matches "
                    "the native/undistorted convention already used to build results/gd_joint_fpa2.pkl")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--uniform", action="store_true", help="constant-atmosphere scene, "
                    "for isolating implementation bugs from real resolution-floor effects "
                    "-- with a genuinely uniform truth, correct bias is exactly zero "
                    "everywhere, so any nonzero structure is unambiguous.")
    ap.add_argument("--psf-fwhm", type=float, default=1.5, help="spatial PSF FWHM [px] used "
                    "when building the no-keystone truth image (build_no_keystone_psf_image); "
                    "real per-row dispersion/smile (nu_row) is always kept regardless of this "
                    "flag -- only the cross-row PSF blur is controlled here. Set 0 to test "
                    "whether the single-row retrieval's inability to represent PSF blur is "
                    "what's driving the uniform-scene bowl (gert.retrieval.GERTRetrieval has "
                    "no cross-row concept at all, so it can't reproduce PSF-blurred truth).")
    args = ap.parse_args()

    scene_label = "uniform" if args.uniform else "realistic"
    print(f"Building {scene_label}-scene FPA{FPA} band...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400, None,
                           args.uniform, False, 32, False, 0)
    print("done.\n", flush=True)

    print(f"Building independent keystone-free 2D array (no joint-block code), "
         f"psf_fwhm={args.psf_fwhm}px...", flush=True)
    t0 = time.time()
    A_nk = build_no_keystone_psf_image(band, spatial_psf_fwhm_px=args.psf_fwhm)
    print(f"done ({time.time()-t0:.0f}s).\n", flush=True)

    _CTX.update(dict(band=band, A_nk=A_nk, order=args.order, absco=absco, geo=geo, solar=solar,
                     uniform=args.uniform, atm_center=atm_center))

    rows = list(range(0, 1024, args.row_step))
    print(f"{len(rows)} independent single-row retrievals (order={args.order})...", flush=True)
    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    t0 = time.time()
    out_free, out_fixed = {}, {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for row, nl_free, nl_fixed in pool.imap_unordered(_worker_row, rows, chunksize=4):
            out_free[row] = nl_free
            out_fixed[row] = nl_fixed
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(rows)} done ({time.time()-t0:.0f}s)", flush=True)
    n_conv_free = sum(1 for nl in out_free.values() if nl.get("_conv") and not nl.get("_diverged"))
    n_conv_fixed = sum(1 for nl in out_fixed.values() if nl.get("_conv") and not nl.get("_diverged"))
    print(f"all done ({time.time()-t0:.0f}s): free {n_conv_free}/{len(rows)}, "
         f"fixed {n_conv_fixed}/{len(rows)} converged", flush=True)

    suffix = ("_uniform" if args.uniform else "") + ("" if args.psf_fwhm == 1.5 else f"_psf{args.psf_fwhm:g}")
    out_path = REPO_ROOT / "results" / f"gd_keystone_free_1d_ceiling_fpa{FPA}{suffix}.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"out_free": out_free, "out_fixed": out_fixed, "fpa": FPA, "uniform": args.uniform,
                    "psf_fwhm": args.psf_fwhm, "order": args.order, "row_step": args.row_step}, f)
    print(f"saved {out_path}")

    # ================= comparison plot =================
    def _extract(out_dict):
        rows_ok = np.array(sorted(r for r, nl in out_dict.items()
                                  if nl.get("_conv") and not nl.get("_diverged") and "co2" in nl))
        return rows_ok, np.array([out_dict[r]["co2"] for r in rows_ok])

    rows_free, bias_free = _extract(out_free)
    rows_fixed, bias_fixed = _extract(out_fixed)

    def _joint_series(pipeline: str):
        with open(REPO_ROOT / "results" / f"gd_joint_fpa{FPA}.pkl", "rb") as f:
            d = pickle.load(f)
        d_out = d["out"]
        rows_, vals_ = [], []
        for (pl, order, rt), v in d_out.items():
            if pl != pipeline or v is None or not v.get("_conv") or v.get("_diverged"):
                continue
            rows_.append(rt[0]); vals_.append(v["co2"])
        rows_ = np.array(rows_); vals_ = np.array(vals_)
        order_ = np.argsort(rows_)
        return rows_[order_], vals_[order_]

    def _stitch(results_dict):
        windows = sorted(results_dict.values(), key=lambda r: r["row_lo"])
        rows_all, bias_c, bias_h = [], [], []
        for w in windows:
            row_lo, row_hi = w["row_lo"], w["row_hi"]
            rows_win = np.arange(row_lo, row_hi + 1)
            eta_win = _eta_of(FPA, np.full(len(rows_win), 512.0), rows_win.astype(float))
            true_win = (np.full(len(rows_win), float(als.xco2_ppm(0.0))) if args.uniform
                       else als.xco2_ppm(eta_win * als.SLIT_HALF_KM))
            bin_centers = w["bin_centers"]
            ret_c = w["prior_co2_ppm_bins"] * w["x_coarse"]
            ret_h = w["prior_co2_ppm_bins"] * w["x_hires"]
            if len(bin_centers) > 1:
                edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
                idx = np.searchsorted(edges, eta_win)
                c_win = ret_c[idx]; h_win = np.interp(eta_win, bin_centers, ret_h)
            else:
                c_win = np.full(len(rows_win), ret_c[0]); h_win = np.full(len(rows_win), ret_h[0])
            rows_all.append(rows_win); bias_c.append(c_win - true_win); bias_h.append(h_win - true_win)
        return np.concatenate(rows_all), np.concatenate(bias_c), np.concatenate(bias_h)

    if args.uniform:
        r_native, v_native = np.array([]), np.array([])
        r_undist, v_undist = np.array([]), np.array([])
    else:
        r_native, v_native = _joint_series("native")
        r_undist, v_undist = _joint_series("undistorted")
    # joint block reference pickles only ever exist per --uniform (they have no
    # psf-fwhm variant of their own), so their own suffix must not pick up ours.
    jb_suffix = "_uniform" if args.uniform else ""
    with open(REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}{jb_suffix}.pkl", "rb") as f:
        d_present = pickle.load(f)
    with open(REPO_ROOT / "results" / f"gd_joint_block_keystone_free_fpa{FPA}{jb_suffix}.pkl", "rb") as f:
        d_free = pickle.load(f)
    r_jb_present, c_jb_present, h_jb_present = _stitch(d_present["results"])
    r_jb_free, c_jb_free, h_jb_free = _stitch(d_free["results"])

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.3, 1.0]})

    ax = axes[0]
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(r_jb_free, c_jb_free, color="tab:orange", lw=0.7, ls="--", alpha=0.55,
           label="joint block, keystone-free ceiling, coarse")
    ax.plot(r_jb_free, h_jb_free, color="tab:blue", lw=0.7, ls="--", alpha=0.55,
           label="joint block, keystone-free ceiling, hi-res")
    ax.plot(rows_fixed, bias_fixed, ".", ms=3, color="tab:purple",
           label=f"independent 1D ceiling, FIXED nuisance (n={len(rows_fixed)})")
    ax.set_ylabel("CO2 bias\n(posterior - true) [ppm]")
    ax.set_title("Apples-to-apples: fixed-nuisance independent retrieval vs. the joint "
                "block's own keystone-free ceiling -- same excursions?", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper right", markerscale=2)

    ax = axes[1]
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(rows_free, bias_free, ".", ms=2.5, color="tab:green", alpha=0.7,
           label=f"independent 1D ceiling, FREE nuisance (n={len(rows_free)})")
    ax.plot(rows_fixed, bias_fixed, ".", ms=2.5, color="tab:purple", alpha=0.8,
           label=f"independent 1D ceiling, FIXED nuisance (n={len(rows_fixed)})")
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_title("Same independent pipeline, free vs. fixed nuisance parameters -- "
                "isolates nuisance cross-talk from anything keystone/joint-block related", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[2]
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(r_native, v_native, ".", ms=2, color="tab:red", alpha=0.5, label="native")
    ax.plot(r_undist, v_undist, ".", ms=2, color="tab:gray", alpha=0.5, label="undistorted")
    ax.plot(r_jb_present, c_jb_present, color="tab:orange", lw=0.6, alpha=0.7,
           label="joint block, keystone-present, coarse")
    ax.plot(r_jb_present, h_jb_present, color="tab:blue", lw=0.6, alpha=0.7,
           label="joint block, keystone-present, hi-res")
    ax.plot(rows_free, bias_free, ".", ms=2, color="tab:green", alpha=0.6, label="1D ceiling, free")
    ax.plot(rows_fixed, bias_fixed, ".", ms=2, color="tab:purple", alpha=0.8, label="1D ceiling, fixed")
    ax.set_ylabel("CO2 bias [ppm]")
    ax.set_xlabel("detector row")
    ax.set_title("full context: every pipeline computed so far", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    fig.suptitle(f"FPA{FPA} ({scene_label} scene, psf_fwhm={args.psf_fwhm}px): independent "
                f"(non-joint-block) check of the keystone-free ceiling", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # non-default psf_fwhm runs exist specifically as joint-block validation
    # counterparts (see module docstring), so they belong alongside the
    # joint-block figures rather than the plain standalone 1D-ceiling checks.
    subfolder = "joint_block" if args.psf_fwhm != 1.5 else "keystone_free_1d_ceiling"
    plots_dir = REPO_ROOT / "plots" / subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_fig = plots_dir / f"gd_keystone_free_1d_ceiling_fpa{FPA}{suffix}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
