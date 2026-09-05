#!/usr/bin/env python3
"""§0a go/no-go test from `JOINT_ROW_INVERSION_PLAN.md`: does a single-
atmosphere retrieval built from trace-and-select pixels (real,
un-interpolated pixels within `--tolerance-km` of a target eta0, scattered
across multiple rows -- see `scripts/gd_toy_trace_pixels.py`) recover a
real point source better than native's per-row-independent retrieval
does?

Reuses `gd_test.py`'s own realistic-scene rendering (`_band_setup`) and
retrieval machinery (`_joint_retrieve`, `GERTRetrieval`) completely
unmodified -- this test does NOT need a new solver, Jacobian-gather
structure, or `G`-atmosphere state; it just hands `GERTRetrieval` a
different, real, un-interpolated set of pixels than native's own
1024-column row.

Repeats §11p's row-910 hot-spot test (FPA2, realistic scene, no noise)
exactly: retrieves CO2 bias at the true peak (row 910) and at a
background row (row 880) using trace-and-select pixels, then computes the
same peak-enhancement-capture fraction §11p/`gd_toy_hotspot_dilution.py`
used -- directly comparable to native's 44% and undistorted's 99%
(already known from §11p, not rerun here).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_trace_retrieve.py --tolerance-km 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_test as gdt  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert import gert_root  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

GERT_ROOT = gert_root()   # $GERT_ROOT -> ../../gert -> ../gert -> HPC scratch
MIN_WINDOW = 4   # rows on each side, floor for near-null-keystone targets


def _eta_of(fpa: int, cols, rows) -> np.ndarray:
    _, s = xy_to_wavelength_slit(fpa, cols, rows)
    return s / s_max(fpa)


def traced_obs(fpa: int, band: dict, target_row: int, tolerance_km: float):
    """Every real (row, col) pixel within tolerance_km of target_row's own
    eta0 (col=512 convention), as (nu, y) arrays sorted descending in nu
    -- matching gd_test.py's own _native_row convention (ForwardModel
    always returns y in ascending-wavelength / descending-wavenumber
    order). No interpolation anywhere: every returned point is a real,
    unmodified pixel from `band["A"]` at its own true (eta, nu)."""
    k_c = rows_crossed(fpa, target_row)
    window = max(MIN_WINDOW, int(round(2.2 * k_c)))
    rows_win = np.arange(max(0, target_row - window), min(1024, target_row + window + 1))
    cols_full = np.arange(1024.0)
    eta0 = _eta_of(fpa, 512.0, float(target_row))
    km_per_eta = als.SLIT_HALF_KM

    nu_parts, y_parts, rows_used = [], [], []
    for i in rows_win:
        eta_row = _eta_of(fpa, cols_full, float(i))
        d_km = (eta_row - eta0) * km_per_eta
        mask = np.abs(d_km) <= tolerance_km
        if not mask.any():
            continue
        cols_i = cols_full[mask]
        lam_i, _ = xy_to_wavelength_slit(fpa, cols_i, np.full(int(mask.sum()), float(i)))
        nu_parts.append(1e4 / lam_i)
        y_parts.append(band["A"][int(i), mask])
        rows_used.append(int(i))

    nu = np.concatenate(nu_parts)
    y = np.concatenate(y_parts)
    order = np.argsort(-nu)   # descending nu, matching _native_row's own reversal convention
    return nu[order], y[order], len(nu), rows_used


def retrieve_at(fpa: int, band: dict, row: int, tolerance_km: float, order: int):
    nu, y, n_pixels, rows_used = traced_obs(fpa, band, row, tolerance_km)
    xt = {gas: float(band["xtrue_of_row"][gas][row]) for gas in ("co2", "ch4", "co", "h2o")}
    xt["p_surface"] = float(band["xtrue_of_row"]["p_surface"][row])
    nl = gdt._joint_retrieve([nu], [y], order, [xt], [band])
    return nl, n_pixels, rows_used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpa", type=int, default=2)
    ap.add_argument("--peak-row", type=int, default=910, help="§11p's real CO2 hot-spot row")
    ap.add_argument("--bg-row", type=int, default=880, help="§11p's background reference row")
    ap.add_argument("--tolerance-km", type=float, default=0.5)
    ap.add_argument("--order", type=int, default=2, help="dispersion order, matching §11k's convention")
    args = ap.parse_args()
    fpa = args.fpa

    print(f"Building realistic-scene FPA{fpa} band (renders the real 1024x1024 detector image)...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))

    snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
    band = gdt._band_setup(fpa, atm_center, absco, geo, solar, snr, 400, None,
                           False, False, 32, False, 0)
    print("done.\n", flush=True)

    true_co2 = als.xco2_ppm(band["x_km_of_row"])

    print(f"retrieving at peak row {args.peak_row} (tolerance={args.tolerance_km:g} km)...", flush=True)
    nl_peak, n_peak, rows_peak = retrieve_at(fpa, band, args.peak_row, args.tolerance_km, args.order)
    print(f"retrieving at background row {args.bg_row} (tolerance={args.tolerance_km:g} km)...", flush=True)
    nl_bg, n_bg, rows_bg = retrieve_at(fpa, band, args.bg_row, args.tolerance_km, args.order)

    print("\n=== results ===")
    print(f"peak row {args.peak_row}: {n_peak} pixels from {len(rows_peak)} rows "
         f"({min(rows_peak)}-{max(rows_peak)})")
    print(f"  converged={nl_peak.get('_conv')}  chi2={nl_peak.get('_chi2'):.4g}  "
         f"co2_bias={nl_peak.get('co2'):+.3f} ppm")
    print(f"bg row {args.bg_row}: {n_bg} pixels from {len(rows_bg)} rows "
         f"({min(rows_bg)}-{max(rows_bg)})")
    print(f"  converged={nl_bg.get('_conv')}  chi2={nl_bg.get('_chi2'):.4g}  "
         f"co2_bias={nl_bg.get('co2'):+.3f} ppm")

    if nl_peak.get("_conv") and nl_bg.get("_conv"):
        true_peak = true_co2[args.peak_row]
        true_bg = true_co2[args.bg_row]
        retrieved_peak = true_peak + nl_peak["co2"]
        retrieved_bg = true_bg + nl_bg["co2"]
        capture = (retrieved_peak - retrieved_bg) / (true_peak - true_bg)
        print(f"\ntrue rise: {true_peak - true_bg:+.3f} ppm   "
             f"trace-and-select retrieved rise: {retrieved_peak - retrieved_bg:+.3f} ppm")
        print(f"peak-enhancement captured: {capture * 100:.1f}%  "
             f"(native=44%, undistorted=99%, from KEYSTONE_SMILE_BIAS_PLAN.md §11p)")
    else:
        print("\nnot both rows converged -- capture fraction not computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
