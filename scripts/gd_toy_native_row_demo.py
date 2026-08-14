#!/usr/bin/env python3
"""Toy, single-row walkthrough of what the "native" pipeline actually does,
built from the real code paths (not a simplified stand-in) but restricted
to one detector row and its immediate neighborhood so it runs in seconds
and produces one legible figure -- written to support conceptualizing
Sec. 1.1/1.2 of NATIVE_VS_UNDISTORTED_ARTIFACT.html for a write-up.

The four panels are the four real steps, in order:

  A. Per-pixel geometric distortion. Every column of a given detector row
     samples its own true (wavelength, slit-position) pair via the real
     ground-test polynomial (geocarb_gert.gd_polynomials.
     xy_to_wavelength_slit) -- not a shared per-row wavelength axis. This
     panel compares the target row's true wavenumber-per-column curve
     against the slit-centre row's (row 512) as a "what a naive constant
     dispersion relation would assume" reference, and plots their
     difference -- keystone/smile made visible as a per-pixel wavenumber
     shift that grows away from slit centre.

  B. Scene heterogeneity across the neighborhood. The along-slit truth
     genuinely differs from one physical row to the next under a realistic
     (non-flat) scene. This overlays the true hi-res spectrum at a few rows
     spanning the PSF's support around the target row, evaluated via the
     same `radiance(eta)` callable gd_render.image() itself calls.

  C. Render: per-row ILS convolution, then the one genuine cross-row step.
     Each row in the neighborhood is rendered independently at its own true
     per-pixel geometry (geocarb_gert.gd_render._diagonal_ils_convolve, the
     exact function gd_render.image() uses internally) -- no cross-row
     borrowing at this stage. Only the final along-slit (N/S) Gaussian PSF
     blur (geocarb_gert.focalplane.gaussian_blur_rows, FWHM 1.5 px) mixes
     rows together. This panel shows the target row before vs. after that
     blur -- the same "pre" and "post" arrays gd_render.image() computes
     internally, just for a small local neighborhood instead of the full
     1024-row detector (so this script doesn't need the expensive full-
     image render).

  D. Forward-model comparison. The measured row (post-blur, from panel C)
     is what a real retrieval is handed as `y_true` on that row's own true
     wavenumber grid (this row's own curve from panel A, not a shared
     nominal grid) -- the defining trick of "native": the forward model is
     evaluated on the SAME per-pixel grid the scene was rendered on, so a
     perfect atmospheric-state guess would reproduce the measurement
     exactly. Top: the measured row vs. the forward model evaluated at the
     PRIOR atmospheric state (gert.forward_model.ForwardModel.run(), no
     fitting yet) -- this mismatch is exactly what Gauss-Newton minimizes.
     Bottom: the real post-fit residual (true - retrieved), obtained by
     calling gd_test.py's own `_joint_retrieve` unmodified on this single
     row, so the "after fitting" curve is the genuine retrieval output, not
     a stand-in.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_native_row_demo.py \\
        --fpa 2 --row 200
Output: plots/gd_toy_native_row_demo_fpa<fpa>_row<row>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geosat_geometry as gg
from geocarb_gert import GEOCARB_BANDS, albedo_for, along_slit_scene as als, sample_geometries
from geocarb_gert import geocarb_noise_model
from geocarb_gert.focalplane import gaussian_blur_rows
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit
from geocarb_gert.gd_render import s_max, _diagonal_ils_convolve

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.rt_solver import SingleScatterSolver

REPO_ROOT = Path(__file__).resolve().parent.parent
GERT_ROOT = Path("/scratch/scrowel3_lab/gert")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_test as gdt   # reuse the real retrieval code, unmodified
from gd_test import _joint_retrieve, WELL_MIXED_GASES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpa", type=int, default=2, help="FPA index 0-3 (default 2, CO2 strong)")
    ap.add_argument("--row", type=int, default=200, help="target detector row, 0-1023")
    ap.add_argument("--neighborhood", type=int, default=10,
                    help="rows on each side of --row to render for the PSF blur "
                         "(default 10; the blur's own support is ~3-4 rows at "
                         "1.5px FWHM, this just pads comfortably)")
    args = ap.parse_args()
    fpa, target_row, K = args.fpa, args.row, args.neighborhood

    print(f"FPA{fpa}, row {target_row} -- setting up shared environment (atm/absco/solar)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)

    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[fpa]
    mols = list(mols)
    wn_min, wn_max = real_wavenumber_range(fpa, margin_cm1=10.0)
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=mols, label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=300.0)
    albedo = float(albedo_for(wide_inst, "desert")[0])
    print("building the along-slit true-radiance lookup...", flush=True)
    wn_hires, radiance = als.build_lookup_radiance(
        absco, wide_inst, geo, solar, np.array([albedo]), n_samples=400, n_workers=None, uniform=False)
    ils = wide_win.ils
    sm = s_max(fpa)

    cols = np.arange(1024.0)
    cols_center = np.full(1024, 512.0)
    _, s_of_row = xy_to_wavelength_slit(fpa, cols_center, np.arange(1024.0))
    x_km_of_row = (s_of_row / sm) * als.SLIT_HALF_KM

    # === Panel A: this row's true per-pixel wavenumber vs. the slit-centre row's ===
    lam_row, s_row = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(target_row)))
    nu_row_true = 1e4 / lam_row
    lam_ref, _ = xy_to_wavelength_slit(fpa, cols, np.full(1024, 512.0))
    nu_ref = 1e4 / lam_ref

    # === Panel B: true hi-res spectrum at a few rows spanning a wider span
    #     than the PSF neighborhood (panel C's K) -- this is showing scene
    #     heterogeneity along the slit, a separate concept from the PSF's own
    #     few-row support, so it needs its own (wider) span to actually be
    #     visible: at K-row spacing the realistic scene's broad along-slit
    #     gradient is usually too gentle over 1-2 PSF-widths to see. ===
    B = max(3 * K, 30)
    show_rows = sorted(set(r for r in
                           [target_row - 2 * B, target_row - B, target_row, target_row + B, target_row + 2 * B]
                           if 0 <= r < 1024))
    eta_show = x_km_of_row[show_rows] / als.SLIT_HALF_KM
    S_show = np.asarray(radiance(eta_show), dtype=float)

    # === Panel C: local render, pre- vs. post-PSF-blur (the exact functions
    #     gd_render.image() itself calls, just for a small row window) ===
    print(f"rendering a {2*K+1}-row local neighborhood around row {target_row}...", flush=True)
    rows_local = np.arange(target_row - K, target_row + K + 1)
    rows_local = rows_local[(rows_local >= 0) & (rows_local < 1024)]
    A_pre = np.empty((len(rows_local), 1024))
    for li, r in enumerate(rows_local):
        lam_r, s_r = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(r)))
        nu_r = 1e4 / lam_r
        eta_r_true = s_r / sm
        S_r = np.asarray(radiance(eta_r_true), dtype=float)
        A_pre[li] = _diagonal_ils_convolve(wn_hires, S_r, nu_r, ils)
    A_post = gaussian_blur_rows(A_pre, 1.5)
    li_target = int(np.where(rows_local == target_row)[0][0])
    row_pre = A_pre[li_target].copy()
    row_post = A_post[li_target].copy()

    # native's own ascending-wavenumber convention (gd_test._native_row)
    nu_native = nu_row_true.copy()
    y_pre = row_pre.copy()
    y_post = row_post.copy()
    if nu_native[0] < nu_native[-1]:
        nu_native = nu_native[::-1]
        y_pre = y_pre[::-1]
        y_post = y_post[::-1]

    # === Panel D: forward model at the prior vs. the measured (post-blur) row,
    #     then the real post-fit residual via gd_test.py's own retrieval code ===
    print("evaluating the forward model at the prior state...", flush=True)
    window_native = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                                   molecules=mols, label=label, obs_grid=nu_native)
    noise_model = geocarb_noise_model(fpa)
    inst_native = Instrument(windows=[window_native], noise_model=noise_model)
    fm = ForwardModel(atm_center, absco, inst_native, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    y_model_prior = fm.run(albedo=np.array([albedo]), albedo_slope=[0.0]).y

    print("running the real single-row retrieval for the post-fit residual...", flush=True)
    xtrue_x_km = float(x_km_of_row[target_row])
    h2o_mean_prior_ppm = float(np.mean(atm_center.gases["h2o"])) * 1e6
    h2o_ratio = float(als.h2o_surface_vmr(np.array([xtrue_x_km]))[0] / als.h2o_surface_vmr(0.0))
    xtrue_row = dict(
        co2=float(als.xco2_ppm(np.array([xtrue_x_km]))[0]),
        ch4=float(als.xch4_ppb(np.array([xtrue_x_km]))[0]),
        co=float(als.xco_ppb(np.array([xtrue_x_km]))[0]),
        h2o=h2o_mean_prior_ppm * h2o_ratio,
        p_surface=float(als.p_surface_hpa(np.array([xtrue_x_km]))[0]),
    )
    band_dict = dict(fpa=fpa, label=label, mols=mols, R=R, wn_min=wn_min, wn_max=wn_max,
                     fwhm_cm=fwhm_cm, albedo=albedo)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    nl = _joint_retrieve([nu_native], [y_post], 2, [xtrue_row], [band_dict])
    y_postfit_resid = nl["_residuals"][0]
    co2_bias = nl.get("co2")
    p_bias = nl.get("p_surface")
    chi2 = nl.get("_chi2")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.7], hspace=0.45, wspace=0.28)

    axA = fig.add_subplot(gs[0, 0])
    axA.plot(cols, nu_ref, lw=1.2, color="tab:gray", label="row 512 (slit-centre reference)")
    axA.plot(cols, nu_row_true, lw=1.2, color="tab:red", label=f"row {target_row} (true)")
    axA.set_xlabel("detector column"); axA.set_ylabel("true wavenumber [cm$^{-1}$]")
    axA.set_title("A. Per-pixel distortion: true dispersion, this row vs. slit centre", fontsize=10)
    axA.legend(fontsize=7)
    axA2 = axA.twinx()
    axA2.plot(cols, nu_row_true - nu_ref, lw=0.8, color="tab:blue", alpha=0.6)
    axA2.set_ylabel("shift [cm$^{-1}$]", color="tab:blue", fontsize=8)
    axA2.tick_params(axis="y", labelsize=7, colors="tab:blue")

    axB = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("coolwarm")
    ti = show_rows.index(target_row)
    axB.axhline(0, color="k", lw=0.6)
    for i, r in enumerate(show_rows):
        if r == target_row:
            continue
        color = cmap(i / max(len(show_rows) - 1, 1))
        axB.plot(wn_hires, S_show[i] - S_show[ti], lw=1.0, color=color, label=f"row {r} $-$ row {target_row}")
    axB.set_xlabel("wavenumber [cm$^{-1}$]"); axB.set_ylabel("true radiance difference")
    axB.set_title(f"B. Truth differs row to row (realistic scene; shown as\n"
                  f"$\\Delta$ from target row, $\\pm${2*B}-row span -- raw overlay is visually identical)", fontsize=9)
    axB.legend(fontsize=6)

    axC = fig.add_subplot(gs[1, :])
    axC.plot(nu_native, y_pre, lw=1.0, color="tab:gray", label="pre-PSF (this row alone, no cross-row mixing)")
    axC.plot(nu_native, y_post, lw=1.2, color="tab:red", label="post-PSF (real N/S blur, 1.5px FWHM)")
    axC.set_xlabel("wavenumber [cm$^{-1}$]"); axC.set_ylabel("radiance")
    axC.set_title("C. Render: independent per-row ILS convolution, then the one real cross-row step (PSF blur)", fontsize=10)
    axC.legend(fontsize=7)

    axD = fig.add_subplot(gs[2, :])
    axD.plot(nu_native, y_post, lw=1.3, color="black", label="measured row (native $y_{true}$, post-PSF)")
    axD.plot(nu_native, y_model_prior, lw=1.1, color="tab:orange", ls="--",
            label="forward model @ prior state (before fitting)")
    axD.set_xlabel("wavenumber [cm$^{-1}$]"); axD.set_ylabel("radiance")
    title_extra = f"  (post-fit: co2 bias={co2_bias:+.3f} ppm, p_surface bias={p_bias:+.3f} hPa, $\\chi^2$={chi2:.3g})" \
        if co2_bias is not None else ""
    axD.set_title("D. Native retrieval: measurement vs. forward model, same row's true grid" + title_extra, fontsize=10)
    axD.legend(fontsize=7)

    axE = fig.add_subplot(gs[3, :])
    axE.axhline(0, color="k", lw=0.5)
    axE.plot(nu_native, y_post - y_model_prior, lw=0.9, color="tab:orange",
            label="pre-fit residual (true - model@prior)")
    axE.plot(nu_native, y_postfit_resid, lw=0.9, color="tab:green",
            label="post-fit residual (true - model@retrieved) -- real gd_test.py output")
    axE.set_xlabel("wavenumber [cm$^{-1}$]"); axE.set_ylabel("residual")
    axE.set_title("Gauss-Newton drives the orange curve down to the green curve", fontsize=9)
    axE.legend(fontsize=7)

    fig.suptitle(f"Native pipeline, one row: FPA{fpa} ({label}), row {target_row} "
                f"(x $\\approx$ {xtrue_x_km:+.0f} km along slit)", fontsize=13)

    plots_dir = REPO_ROOT / "plots" / "toy_diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_toy_native_row_demo_fpa{fpa}_row{target_row}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
