#!/usr/bin/env python3
"""GeoCarb instrument-design sweep: σ(XCO₂) and DOF vs (GSD × dwell).

Demonstrates the full seam:
  geosat_geometry (mission: orbit, slit, scan, per-pixel geometry)
      → geocarb_gert (adapter + mission-side GSD/dwell → noise model)
          → gert (radiance, Jacobian, posterior)

The (GSD × dwell) grid only rescales ``Sy``; ``K`` is held fixed, so the whole
grid costs a single forward model + Jacobian.

Run:  PYTHONPATH=. python scripts/run_design_sweep.py --gert /path/to/gert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import geosat_geometry as gg
from geocarb_gert import (build_geocarb_instrument, base_noise_model,
                          staring_sweep_models, sample_geometries, gsd_km,
                          scene_from_profile, reference_atmosphere, GEOCARB_REF)

import gert
from gert.osse import simulate, StateSpec, run_design_sweep
from gert.rt_solver import VectorDOSolver
from gert.radiometry import DetectorSpec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gert", required=True, help="gert repo path (absco/solar tables)")
    ap.add_argument("--snr-ref", type=float, default=300.0)
    args = ap.parse_args()
    G = Path(args.gert)

    # ── Mission: build a GeoCarb scan block and pick a representative pixel ──
    sat = gg.LongSlitGeoSatellite(sat_lon_deg=-85.0)
    demo = gg.geocarb_demo(verbose=False)
    block = demo["blocks"][0]
    picks = sample_geometries(block, n=1, seed=0)
    row, col, geo = picks[0]
    print(f"[mission] ScanBlock {block.shape}  pixel (row={row}, col={col})")
    print(f"[mission] GSD={gsd_km(sat):.1f} km  t_int={sat.integration_time_s:.0f} s "
          f"(STARING: dwell independent of GSD)")
    print(f"[mission] SZA={geo.sza:.2f}  VZA={geo.vza:.2f}  RAA={geo.raa:.2f}")

    # ── Scene: the MISSION supplies its own atmosphere (gert never owns one).
    #    Model-driven scenes come from model_sampler.sample_field_along_rays.
    atm = reference_atmosphere()

    instrument = build_geocarb_instrument()
    n_bands = len(instrument.windows)
    print(f"[gert] bands: {[w.label for w in instrument.windows]}  "
          f"channels={sum(w.n_channels for w in instrument.windows)}")

    absco = gert.ABSCOTable.load_all(str(G / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(G / "input/solar/solar.h5"))
    scene = scene_from_profile(atm, albedo=np.full(n_bands, 0.25))

    # Truth radiance once (noise model does not affect R_band).
    obs = simulate(scene, instrument, geo, absco=absco, solar_spectrum=solar,
                   solver=VectorDOSolver(), snr=args.snr_ref, add_noise=True, seed=0)
    print(f"[gert] truth radiance: {obs.y_truth.size} channels")

    # ── Mission-side noise models over (GSD × dwell) ────────────────────────
    # Calibrate the reference design from the quoted SNR, at the CO2_weak band.
    w = instrument.windows[1]
    wl_ref = float(np.mean(w.wl_instrument))
    dlam_ref = float(np.mean(np.abs(np.gradient(w.wl_instrument))))
    R_ref = float(np.max(np.abs(obs.R_band[1])))
    det = DetectorSpec(read_noise_e=30.0, dark_current_e_s=1e3,
                       full_well_e=1.0e6, bit_depth=14)
    base = base_noise_model(args.snr_ref, R_ref, wl_ref, dlam_ref,
                            t_int_s=GEOCARB_REF["t_int_s"], detector=det)

    gsds = [3.0, 6.0, 12.0]           # km
    dwells = [2.5, 10.0, 40.0]        # s
    models = staring_sweep_models(base, gsds, dwells)

    # CO2 (1.61/2.06 µm) + CH4 and CO (both from the 2.32 µm band).
    spec = StateSpec(label="gas+albedo", gases=("co2", "ch4", "co", "h2o"))
    print(f"\n[sweep] {len(models)} design points, K computed once")
    pts = run_design_sweep(obs, atm, spec, models, solver_kind="ms", verbose=False)

    by = {p.label: p for p in pts}
    # One σ(Xgas) table per retrieved gas (CO2 ppm; CH4/CO ppb).
    UNIT = {"co2": "ppm", "ch4": "ppb", "co": "ppb"}
    for gas in ("co2", "ch4", "co"):
        print(f"\n  σ(X{gas.upper()}) [{UNIT[gas]}]" + " " * 6
              + "".join(f"{t:>10g} s" for t in dwells))
        for g in gsds:
            row_s = "".join(
                f"{by[f'{g:g} km / {t:g} s'].xgas_uncert[gas]:>12.3g}" for t in dwells)
            print(f"    GSD {g:>5g} km   {row_s}")
    print(f"\n  DOF                " + "".join(f"{t:>10g} s" for t in dwells))
    for g in gsds:
        row_s = "".join(f"{by[f'{g:g} km / {t:g} s'].dof:>12.2f}" for t in dwells)
        print(f"    GSD {g:>5g} km   {row_s}")

    # Saturated design points are physically invalid: the well overflows within
    # t_int, so their sigma is meaningless (the noise model reports what the
    # detector *would* do if it did not clip).  Exclude them from the fits.
    print(f"\n  saturated channels " + "".join(f"{t:>10g} s" for t in dwells))
    for g in gsds:
        row_s = "".join(f"{by[f'{g:g} km / {t:g} s'].saturated_frac:>11.1%} " for t in dwells)
        print(f"    GSD {g:>5g} km   {row_s}")

    def _ok(g, t):
        return by[f"{g:g} km / {t:g} s"].saturated_frac == 0.0

    def _slope(xs, ys, label, expect):
        if len(xs) < 2:
            print(f"  σ(XCO₂) slope vs {label}: too few unsaturated points"); return
        s = np.polyfit(np.log(xs), np.log(ys), 1)[0]
        print(f"  σ(XCO₂) slope vs {label:6s} = {s:+.3f}   ({expect})")

    # Verify the staring scalings on *unsaturated* points only.
    t_ref, g_ref = GEOCARB_REF["t_int_s"], GEOCARB_REF["gsd_km"]
    gs = [g for g in gsds if _ok(g, t_ref)]
    ts = [t for t in dwells if _ok(g_ref, t)]
    print()
    _slope(gs, [by[f"{g:g} km / {t_ref:g} s"].xco2_uncert_ppm for g in gs],
           "GSD", "shot-limited staring: -1; dark/read-limited: -2")
    _slope(ts, [by[f"{g_ref:g} km / {t:g} s"].xco2_uncert_ppm for t in ts],
           "dwell", "-0.5 in BOTH shot- and dark-limited regimes")

    n_sat = sum(1 for p in pts if p.saturated_frac > 0)
    print(f"\n  {n_sat}/{len(pts)} design points saturate and are excluded above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())