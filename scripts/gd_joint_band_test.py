#!/usr/bin/env python3
"""Full joint two-band (no-aerosol) stress test: the co-located-scene
counterpart of gd_band_stress_test.py's single-band battery --
uniform/barcode/realistic scenes x noise/no-noise x native/rectified/
undistorted pipelines -- for a paired GeoCarb band pair (default FPA0 +
FPA2). See KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11g (first no-aerosol joint
result, order=0, uniform-only) and Sec. 11h (this full battery) for the
rationale. Deliberately no aerosol still (Sec. 11's own motivation would
have that nonlinearity compound with anything found here, and it costs much
more compute) -- see Sec. 11d items 4/6 for when that gets added back in.

Three pipelines, with a specifically cross-band meaning each:
  "native"       -- each band's own real per-pixel row, exactly as
                     gd_band_stress_test.py's native pipeline renders it
                     (real projection operator + ILS convolution, real
                     spatial PSF blur), paired across bands via
                     geocarb_gert.cross_band.nearest_row_pairing (Sec. 11f
                     -- real slit angle `s`, never normalized `eta`;
                     bounded <=0.5 row mismatch, confirmed 2026-07-27).
  "undistorted"  -- each band's zero-distortion truth spectrum evaluated
                     directly at its own real per-row position (bypasses
                     gd_render.image()/rectify() entirely, same mechanism as
                     gd_band_stress_test.py's undistorted branch), same
                     nearest-row pairing as native. Isolates whether a
                     cross-band retrieval difficulty is fundamental
                     information content vs. something real geometric
                     distortion (keystone/smile) makes worse -- mirroring
                     that pipeline's single-band role.
  "rectified"    -- each band rectified independently (gd_render.rectify)
                     onto ONE SHARED cross-band `s_grid` (real degrees,
                     intersection-clipped to both bands' valid coverage) --
                     the ORIGINAL co-registration mechanism Sec. 11b/11c
                     proposed and then set aside in favor of nearest-row,
                     kept here specifically to test that decision
                     empirically: does shared-grid rectification's already-
                     quantified single-band interpolation bias (Sec. 9l/9m)
                     show up here too, on top of any real cross-band
                     mismatch? Rows are aligned by construction (same
                     `s_grid` index for both bands) -- no separate pairing
                     needed for this pipeline.

State vector: single shared `p_scale`/`h2o_scale`, `co2_scale` sensitive
only through whichever band has co2 in its molecule list (confirmed via
gert's own multi-window Jacobian assembly, Sec. 11g) -- no aerosol.
native/rectified get per-band dispersion order 2 (each band's own
`disp_a{k}_{b}`, `b` = window index); undistorted gets order 0 (no genuine
wavelength-calibration error for it to correct, matching the single-band
convention's rationale exactly).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_band_test.py \\
        --fpa-a 0 --fpa-b 2 [--uniform | --barcode [--barcode-bars N]] \\
        [--noise [--snr S] [--noise-seed N]] [--row-step N] [--out-tag TAG]
Output: results/gd_joint_fpa<A>_fpa<B>[_uniform|_barcode][_noise][_<tag>].pkl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

import geosat_geometry as gg
from geocarb_gert import GEOCARB_BANDS, albedo_for, along_slit_scene as als, sample_geometries
from geocarb_gert import gd_render, build_geocarb_instrument
from geocarb_gert.cross_band import nearest_row_pairing, real_s_of_row
from geocarb_gert.gd_polynomials import real_wavenumber_range, xy_to_wavelength_slit
from geocarb_gert.gd_render import available_cpus, s_max, _diagonal_ils_convolve

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

REPO_ROOT = Path(__file__).resolve().parent.parent
GERT_ROOT = Path("/scratch/scrowel3_lab/gert")

WELL_MIXED_GASES = {"o2", "n2o"}
DEFAULT_SNR_BY_FPA = {0: 400.0, 1: 300.0, 2: 300.0, 3: 200.0}

_G = {}


def _band_setup(fpa: int, atm_center, absco, geo, solar, snr: float, n_lookup_samples: int,
                n_workers, uniform: bool, barcode: bool, barcode_bars: int,
                noise: bool, noise_seed: int):
    """Render one band's raw detector image plus everything needed for all
    three pipelines: `A` (native/rectified source), `wn_hires`/`radiance`
    (undistorted source, and the barcode/lookup truth), the nominal per-band
    `wn_grid` (rectified's wavelength axis), and per-row real `s`/`x_km`
    (pairing and truth-scoring). Same mechanism as gd_band_stress_test.py's
    main(), just factored to run once per band here."""
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[fpa]
    mols = list(mols)
    wn_min, wn_max = real_wavenumber_range(fpa, margin_cm1=10.0)
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=mols, label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=snr)
    albedo = float(albedo_for(wide_inst, "desert")[0])

    if barcode:
        from geocarb_gert.focalplane import barcode_scene
        fm_center = ForwardModel(atm_center, absco, wide_inst, geo, solver=SingleScatterSolver(),
                                 solar_spectrum=solar)
        res_center = fm_center.run(albedo=np.array([albedo]), albedo_slope=[0.0])
        S_center = np.asarray(res_center.I_hires[0], dtype=float)
        brightness = np.resize([1.0, 0.2], barcode_bars)
        radiance = barcode_scene(S_center, brightness=brightness, widths=None, softness=0.0)
        wn_hires = wide_win.wn_hires
    else:
        wn_hires, radiance = als.build_lookup_radiance(
            absco, wide_inst, geo, solar, np.array([albedo]),
            n_samples=n_lookup_samples, n_workers=n_workers, uniform=uniform)

    A = gd_render.image(fpa, wn_hires, radiance, wide_win.ils, spatial_psf_fwhm_px=1.5,
                        n_workers=n_workers)
    sigma_band = float(np.max(np.abs(A))) / snr
    noise_arr = None
    if noise:
        rng = np.random.default_rng(noise_seed)
        noise_arr = rng.normal(0.0, sigma_band, size=A.shape)
        A = A + noise_arr

    nominal_inst = build_geocarb_instrument()
    wn_grid = nominal_inst.windows[fpa].wn_instrument

    cols_center = np.full(1024, 512.0)
    _, s_of_row = xy_to_wavelength_slit(fpa, cols_center, np.arange(1024.0))
    sm = s_max(fpa)
    x_km_of_row = (s_of_row / sm) * als.SLIT_HALF_KM
    xtrue_x_km = np.zeros(1024) if (uniform or barcode) else x_km_of_row

    h2o_mean_prior_ppm = float(np.mean(atm_center.gases["h2o"])) * 1e6
    h2o_surface_ratio = als.h2o_surface_vmr(xtrue_x_km) / als.h2o_surface_vmr(0.0)
    xtrue_of_row = {
        "co2": als.xco2_ppm(xtrue_x_km), "ch4": als.xch4_ppb(xtrue_x_km), "co": als.xco_ppb(xtrue_x_km),
        "h2o": h2o_mean_prior_ppm * h2o_surface_ratio,
        "p_surface": als.p_surface_hpa(xtrue_x_km),
    }
    return dict(fpa=fpa, label=label, mols=mols, R=R, wn_min=wn_min, wn_max=wn_max,
               fwhm_cm=fwhm_cm, albedo=albedo, A=A, wn_hires=wn_hires, radiance=radiance,
               ils=wide_win.ils, wn_grid=wn_grid, sigma_band=sigma_band, noise_arr=noise_arr,
               x_km_of_row=x_km_of_row, xtrue_of_row=xtrue_of_row)


def _shared_s_grid(fpa_a: int, fpa_b: int, n: int = 1024) -> np.ndarray:
    """Real slit-angle grid [deg] spanning the intersection of both bands'
    covered range -- the target grid for the "rectified" pipeline. Same
    real-`s` convention as Sec. 11f's nearest_row_pairing, never eta."""
    s_a = real_s_of_row(fpa_a)
    s_b = real_s_of_row(fpa_b)
    lo = max(min(s_a.min(), s_a.max()), min(s_b.min(), s_b.max()))
    hi = min(max(s_a.min(), s_a.max()), max(s_b.min(), s_b.max()))
    return np.linspace(lo, hi, n)


def _native_row(band: dict, row: int):
    fpa = band["fpa"]
    cols = np.arange(1024.0)
    lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(row)))
    nu_row = 1e4 / lam_row
    y_dist = band["A"][row, :].copy()
    if nu_row[0] < nu_row[-1]:
        nu_row = nu_row[::-1]
        y_dist = y_dist[::-1]
    return nu_row, y_dist


def _undistorted_row(band: dict, row: int):
    fpa = band["fpa"]
    cols = np.arange(1024.0)
    lam_row, _ = xy_to_wavelength_slit(fpa, cols, np.full(1024, float(row)))
    nu_row = 1e4 / lam_row
    reverse = nu_row[0] < nu_row[-1]
    if reverse:
        nu_row = nu_row[::-1]
    eta_row = band["x_km_of_row"][row] / als.SLIT_HALF_KM
    S_row = np.asarray(band["radiance"](np.full(len(nu_row), eta_row)), dtype=float)
    y_dist = _diagonal_ils_convolve(band["wn_hires"], S_row, nu_row, band["ils"])
    if band["noise_arr"] is not None:
        noise_row = band["noise_arr"][row, :]
        if reverse:
            noise_row = noise_row[::-1]
        y_dist = y_dist + noise_row
    return nu_row, y_dist


def _rectified_row(band: dict, Rimg: np.ndarray, grid_idx: int):
    row = Rimg[grid_idx, :]
    valid = ~np.isnan(row)
    if valid.sum() < 0.5 * len(row):
        return None, None
    nu_valid = band["wn_grid"][valid]
    y_dist = row[valid][::-1]
    return nu_valid[::-1], y_dist


def _joint_retrieve(nu_a, y_a, nu_b, y_b, order: int, xtrue_row_a: dict, xtrue_row_b: dict):
    g = _G
    band_a, band_b = g["band_a"], g["band_b"]
    atm_center, absco, geo, solar, snr = g["atm"], g["absco"], g["geo"], g["solar"], g["snr"]

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

    fm = ForwardModel(atm_center, absco, inst, geo, solver=SingleScatterSolver(),
                      solar_spectrum=solar)
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
    names = sv.names
    nl = {}
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            res = ret.run()
    except (ValueError, np.linalg.LinAlgError) as e:
        nl["_chi2"] = np.nan
        nl["_conv"] = False
        nl["_diverged"] = str(e)
        return nl

    for gas in gases:
        in_b = gas in band_b["mols"]
        src = xtrue_row_b if in_b else xtrue_row_a
        unit = 1e6 if gas not in ("ch4", "co") else 1e9
        prior_val = float(np.mean(atm_center.gases[gas])) * unit
        retrieved_val = prior_val * res.x_ret[names.index(gas + "_scale")]
        nl[gas] = retrieved_val - float(src[gas])
    p_scale_ret = float(res.x_ret[names.index("p_scale")])
    p_surface_prior = float(als.p_surface_hpa(np.array([0.0]))[0])
    nl["p_surface"] = p_surface_prior * p_scale_ret - float(xtrue_row_a["p_surface"])

    # Complete raw state vector + posterior uncertainty (every element,
    # both bands' albedo/dispersion included), independent of the derived
    # gas/pressure bias dict above -- same convention gd_band_stress_test.py
    # uses for the single-band case (added 2026-07-25 there, added here
    # 2026-07-28 after being asked directly whether h2o was even in this
    # state vector -- it always is, `gases` unconditionally appends "h2o").
    # `_state` is a retrieved-minus-prior convenience view for everything
    # NOT already covered by the gas/pressure bias dict above (T_offset,
    # both bands' albedo_{b}/albedo_slope_{b}, and -- order>0 only -- both
    # bands' own disp_a{k}_{b}) -- exactly the "co-retrieved parameters" a
    # bias-correction analysis would want to regress the gas bias against.
    nl["_state_full"] = {
        "names": list(names),
        "x_ret": res.x_ret.copy(),
        "x_prior": sv.prior.copy(),
        "sigma": res.posterior_sigma(),
    }
    _tracked = {f"{gas}_scale" for gas in gases} | {"p_scale"}
    nl["_state"] = {name: float(res.x_ret[i] - prior)
                    for i, (name, prior) in enumerate(zip(names, sv.prior))
                    if name not in _tracked}

    nl["_chi2"] = float(res.chisq_reduced)
    nl["_conv"] = res.converged
    if res.diverged:
        nl["_diverged"] = "gert: chi2 non-finite"
    return nl


def _worker(task):
    pipeline, order, ka, kb = task
    g = _G
    band_a, band_b = g["band_a"], g["band_b"]
    xtrue_row_a = {gas: float(band_a["xtrue_of_row"][gas][ka]) for gas in ("co2", "ch4", "co", "h2o")}
    xtrue_row_a["p_surface"] = float(band_a["xtrue_of_row"]["p_surface"][ka])
    xtrue_row_b = {gas: float(band_b["xtrue_of_row"][gas][kb]) for gas in ("co2", "ch4", "co", "h2o")}
    xtrue_row_b["p_surface"] = float(band_b["xtrue_of_row"]["p_surface"][kb])
    try:
        if pipeline == "native":
            nu_a, y_a = _native_row(band_a, ka)
            nu_b, y_b = _native_row(band_b, kb)
        elif pipeline == "undistorted":
            nu_a, y_a = _undistorted_row(band_a, ka)
            nu_b, y_b = _undistorted_row(band_b, kb)
        else:  # rectified -- ka == kb == shared s_grid index by construction
            nu_a, y_a = _rectified_row(band_a, g["Rimg_a"], ka)
            nu_b, y_b = _rectified_row(band_b, g["Rimg_b"], kb)
            if nu_a is None or nu_b is None:
                return (pipeline, order, ka, kb), {"_chi2": np.nan, "_conv": False, "off_detector": True}
        nl = _joint_retrieve(nu_a, y_a, nu_b, y_b, order, xtrue_row_a, xtrue_row_b)
    except Exception as e:  # noqa: BLE001 -- keep the sweep alive
        nl = {"_chi2": np.nan, "_conv": False, "_diverged": f"{type(e).__name__}: {e}"}
    return (pipeline, order, ka, kb), nl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpa-a", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--fpa-b", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--row-step", type=int, default=1, help="subsample fpa-a's native rows")
    ap.add_argument("--n-lookup-samples", type=int, default=400)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--out-tag", type=str, default=None)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--barcode", action="store_true")
    ap.add_argument("--barcode-bars", type=int, default=32)
    ap.add_argument("--snr", type=float, default=None,
                    help="override both bands' SNR (default: min of each band's own "
                         "DEFAULT_SNR_BY_FPA, the more conservative choice for a joint fit)")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--pipelines", type=str, default="native,rectified,undistorted",
                    help="comma-separated subset of native,rectified,undistorted")
    args = ap.parse_args()
    if args.uniform and args.barcode:
        ap.error("--uniform and --barcode are mutually exclusive")
    pipelines = args.pipelines.split(",")

    FPA_A, FPA_B = args.fpa_a, args.fpa_b
    snr = args.snr if args.snr is not None else min(DEFAULT_SNR_BY_FPA[FPA_A], DEFAULT_SNR_BY_FPA[FPA_B])

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)

    t0 = time.time()
    print(f"rendering FPA{FPA_A} ({'barcode' if args.barcode else 'uniform' if args.uniform else 'realistic'}"
         f"{', noise' if args.noise else ''})...", flush=True)
    band_a = _band_setup(FPA_A, atm_center, absco, geo, solar, snr, args.n_lookup_samples,
                         args.n_workers, args.uniform, args.barcode, args.barcode_bars,
                         args.noise, args.noise_seed)
    print(f"  done ({time.time()-t0:.0f}s). rendering FPA{FPA_B}...", flush=True)
    band_b = _band_setup(FPA_B, atm_center, absco, geo, solar, snr, args.n_lookup_samples,
                         args.n_workers, args.uniform, args.barcode, args.barcode_bars,
                         args.noise, args.noise_seed)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    _G.update(dict(band_a=band_a, band_b=band_b, atm=atm_center, absco=absco, geo=geo,
                   solar=solar, snr=snr))

    tasks = []

    if "native" in pipelines or "undistorted" in pipelines:
        rows_a = np.arange(0, 1024, args.row_step, dtype=float)
        pairing = nearest_row_pairing(FPA_A, FPA_B, rows_a)
        print(f"native/undistorted pairing: {len(pairing['rows_a'])}/{len(rows_a)} rows "
             f"({pairing['n_dropped']} dropped), mismatch mean="
             f"{np.abs(pairing['mismatch_rows']).mean():.3f} "
             f"max={np.abs(pairing['mismatch_rows']).max():.3f} rows", flush=True)
        for ra, rb in zip(pairing["rows_a"], pairing["rows_b"]):
            if "native" in pipelines:
                tasks.append(("native", 2, int(ra), int(rb)))
            if "undistorted" in pipelines:
                tasks.append(("undistorted", 0, int(ra), int(rb)))

    if "rectified" in pipelines:
        s_grid_shared = _shared_s_grid(FPA_A, FPA_B)
        Rimg_a = gd_render.rectify(FPA_A, band_a["A"], s_grid_shared, band_a["wn_grid"])
        Rimg_b = gd_render.rectify(FPA_B, band_b["A"], s_grid_shared, band_b["wn_grid"])
        _G["Rimg_a"], _G["Rimg_b"] = Rimg_a, Rimg_b
        grid_idx = np.arange(0, len(s_grid_shared), max(args.row_step, 1))
        print(f"rectified shared s_grid: {len(s_grid_shared)} pts, "
             f"[{s_grid_shared[0]:.3f},{s_grid_shared[-1]:.3f}] deg, "
             f"{len(grid_idx)} indices tested", flush=True)
        for k in grid_idx:
            tasks.append(("rectified", 2, int(k), int(k)))

    print(f"{len(tasks)} joint retrievals ({pipelines})", flush=True)
    n_workers = args.n_workers if args.n_workers is not None else available_cpus()
    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        n_done = 0
        for key, nl in pool.imap_unordered(_worker, tasks, chunksize=4):
            out[key] = nl
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"all done ({time.time()-t0:.0f}s)", flush=True)

    for pl in pipelines:
        keys = [k for k in out if k[0] == pl]
        n_conv = sum(1 for k in keys if out[k].get("_conv") and not out[k].get("_diverged")
                    and not out[k].get("off_detector"))
        print(f"  {pl}: converged {n_conv}/{len(keys)}", flush=True)

    mode_suffix = "_uniform" if args.uniform else ("_barcode" if args.barcode else "")
    noise_suffix = "_noise" if args.noise else ""
    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    out_path = (REPO_ROOT / "results" /
               f"gd_joint_fpa{FPA_A}_fpa{FPA_B}{mode_suffix}{noise_suffix}{tag_suffix}.pkl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "fpa_a": FPA_A, "fpa_b": FPA_B, "uniform": args.uniform,
                    "barcode": args.barcode, "barcode_bars": args.barcode_bars if args.barcode else None,
                    "noise": args.noise, "noise_seed": args.noise_seed if args.noise else None,
                    "snr": snr, "pipelines": pipelines}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
