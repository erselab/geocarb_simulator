#!/usr/bin/env python3
"""Every analytic Jacobian column against central finite differences.

The standing check that makes `geocarb_gert.jacobians` trustworthy. Analytic
derivatives are exactly the kind of code that is confidently wrong -- a
transposed axis or a missing chain-rule factor produces a Jacobian that is
smooth, plausible, and points in the wrong direction, and Gauss-Newton will
happily converge to the wrong answer with a clean-looking residual. So each
column is compared elementwise against a central difference of the SAME
forward model the sweep uses (`build_forward_state`), not a reimplementation.

Read the numbers this way. Central differences have their own truncation
error ~h^2 plus roundoff ~eps/h, so agreement improves with h down to an
optimum and then degrades. A column that is genuinely correct shows the
relative disagreement FALLING as h falls (until roundoff takes over); a
column with a real analytic error shows a floor that h cannot reduce. The
`--h-scan` mode prints the whole curve so the difference is visible rather
than inferred from one h.

The metric is the relative L2 difference over the whole column,
``||K_ana - K_fd|| / ||K_fd||``, plus the cosine between them -- because a
column can have the right SHAPE and the wrong SCALE (a missing constant
factor) or vice versa, and the two failures need different fixes.

Run:  PYTHONPATH=. python3 scripts/gd_jacobian_validate.py
      PYTHONPATH=. python3 scripts/gd_jacobian_validate.py --rows 890 935 --h-scan
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert import jacobians as jac  # noqa: E402
from geocarb_gert.joint_state import (build_forward_state,  # noqa: E402
                                      state_spec_from_scene)
from gd_joint_block_retrieve import FPA, GERT_ROOT, _eta_of, band_basics  # noqa: E402
from gd_joint_block_whole_slit_sweep import ROW_KINDS  # noqa: E402

PAD = 4


def fd_column(forward, x, k, h):
    """Central difference of the packed forward model in element k."""
    xp, xm = x.copy(), x.copy()
    xp[k] += h
    xm[k] -= h
    return (forward(xp) - forward(xm)) / (2.0 * h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs=2, type=int, default=[890, 900],
                    metavar=("LO", "HI"),
                    help="detector row window (default: a short 890-900 slice of the "
                         "east hot-spot window, small enough that the FD reference "
                         "stays affordable -- every column costs 2 full forward runs)")
    ap.add_argument("--free", default="co2_ppm,h2o_surface_vmr",
                    help="rows to differentiate (default: co2_ppm,h2o_surface_vmr)")
    ap.add_argument("--g-ratio", type=float, default=3.0,
                    help="G = max(2, round(width/g_ratio)); default 3 keeps the "
                         "column count modest for a validation run")
    ap.add_argument("--h", type=float, default=1e-4, help="FD step (default 1e-4)")
    ap.add_argument("--h-scan", action="store_true",
                    help="repeat over several h to show whether disagreement is FD "
                         "truncation error (falls with h) or a real analytic error "
                         "(floors out)")
    ap.add_argument("--surface-density", type=int, default=3,
                    help="albedo bins per gas bin (its correlation length is ~30 km "
                         "against 100-500 km for the gases, so it needs a denser grid)")
    ap.add_argument("--max-columns", type=int, default=8,
                    help="cap the number of FD columns (they are the expensive part)")
    ap.add_argument("--interp-kind", type=str, default="linear", choices=["linear", "nearest"],
                    help="state_interp kind passed to BOTH build_forward_state and "
                         "linearize -- default 'linear' (today's validated path); pass "
                         "'nearest' to validate the piecewise-constant downscaling's own "
                         "analytic Jacobian against FD the same way.")
    args = ap.parse_args()

    row_lo, row_hi = args.rows
    rows_win = np.arange(row_lo, row_hi + 1)
    width = len(rows_win)
    G = max(2, int(round(width / args.g_ratio)))
    free = tuple(s.strip() for s in args.free.split(","))

    print(f"loading absco/solar from {GERT_ROOT} ...", flush=True)
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    atm0 = als.atmosphere_at(0.0)
    wide_win, wide_inst, albedo = band_basics(FPA, atm0, absco, geo, solar)
    wn_hires, ils = wide_win.wn_hires, wide_win.ils

    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    bin_centers = np.linspace(eta_all.min(), eta_all.max(), G)
    anchor_rows = np.arange(max(0, row_lo - PAD), min(1023, row_hi + PAD) + 1e-9, 1.0)
    anchor_etas = np.sort(_eta_of(FPA, np.full(len(anchor_rows), 512.0),
                                  anchor_rows.astype(float)))

    # band_label adds the `surface` rows (albedo) on their own denser grid;
    # harmless when albedo is not among `free`, since frozen rows cost nothing
    spec = state_spec_from_scene(bin_centers, free=free, band_label=wide_win.label,
                                 surface_density=args.surface_density, kinds=ROW_KINDS)
    x0 = spec.x0()
    print(f"FPA{FPA} rows {row_lo}-{row_hi} (width {width}), G={G}, "
          f"{len(anchor_etas)} anchors, free={free} -> {spec.n_free} elements")
    print(f"band molecules: {wide_win.molecules}   label: {wide_win.label}")
    print("rows: " + ", ".join(f"{p.name}[{p.n}]/{p.target}{'' if p.free else ' frozen'}"
                               for p in spec.params))

    # analytic
    spectrum_jac = jac.make_spectrum_jac(absco, wide_inst, geo, solar, albedo)
    t0 = time.time()
    y_ana, K_ana, _K_g = jac.linearize(FPA, rows_win, anchor_etas, spec, spectrum_jac,
                                       wn_hires, ils, x0, pad=PAD, state_interp=args.interp_kind)
    t_ana = time.time() - t0
    print(f"analytic: y {y_ana.shape}, K {K_ana.shape}  ({t_ana:.1f}s)")

    # finite-difference reference, through the sweep's own forward model
    def make_spectrum():
        from gert.forward_model import ForwardModel
        from gert.rt_solver import SingleScatterSolver

        def spectrum(params: dict, surface: dict | None = None):
            atm = als.atmosphere_from_params(**params)
            alb = float((surface or {}).get("albedo", albedo))
            slope = float((surface or {}).get("albedo_slope", 0.0))
            # tau_aerosol/height_aerosol (2026-09-08) -- must match
            # _make_state_spectrum's own threading exactly, or this
            # "FD reference" silently omits aerosol physics the analytic
            # path (spectrum_jac) DOES include whenever either row is
            # free/frozen -- caught directly (co2_ppm's own forward
            # agreement broke, not just the aerosol rows', the first time
            # this drifted out of sync).
            tau_aer = (surface or {}).get("tau_aerosol")
            height_aer = (surface or {}).get("height_aerosol")
            n_wn = len(wide_inst.windows[0].wn_hires)
            p_aer_val = als.aerosol_phase_hg(als.AEROSOL_G, np.cos(geo.scattering_angle))
            fm = ForwardModel(atm, absco, wide_inst, geo,
                              solver=SingleScatterSolver(), solar_spectrum=solar)
            res = fm.run(albedo=np.array([alb]), albedo_slope=np.array([slope]),
                         tau_aerosol=tau_aer, height_aerosol=height_aer,
                         aerosol_profile_shape="gaussian",
                         thickness_aerosol=als.AEROSOL_THICKNESS_PA,
                         ssa_aerosol=[np.full(n_wn, als.AEROSOL_SSA)],
                         g_aerosol=[als.AEROSOL_G],
                         qext_aerosol=[np.full(n_wn, als.AEROSOL_QEXT_NORM)],
                         P_aerosol=[np.full(n_wn, p_aer_val)])
            return np.asarray(res.I_hires[0], dtype=float)
        return spectrum

    forward = build_forward_state(FPA, rows_win, anchor_etas, spec, make_spectrum(),
                                  wn_hires, ils, pad=PAD, state_interp=args.interp_kind)
    y_fd = forward(x0)
    dy = np.linalg.norm(y_ana - y_fd) / np.linalg.norm(y_fd)
    print(f"forward agreement (analytic path vs sweep path): rel L2 {dy:.3e}"
          f"   {'OK' if dy < 1e-12 else 'MISMATCH -- the two paths differ!'}")

    slices = spec.slices()
    cols_to_check = []
    for name in free:
        sl = slices[name]
        for k in range(min(sl.stop - sl.start, args.max_columns)):
            cols_to_check.append((name, k, sl.start + k))
    if len(cols_to_check) > args.max_columns:
        step = max(1, len(cols_to_check) // args.max_columns)
        cols_to_check = cols_to_check[::step][:args.max_columns]

    hs = [1e-3, 1e-4, 1e-5, 1e-6] if args.h_scan else [args.h]
    print(f"\nchecking {len(cols_to_check)} columns, h in {hs}"
          f"   ({2 * len(cols_to_check) * len(hs)} forward runs)\n")
    hdr = f"{'row':18s} {'k':>3s} {'h':>8s} {'rel L2':>11s} {'cosine':>12s} {'||K_ana||':>11s}"
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for name, k, col in cols_to_check:
        ka = K_ana[:, col]
        for h in hs:
            kf = fd_column(forward, x0, col, h)
            nf = np.linalg.norm(kf)
            na = np.linalg.norm(ka)
            rel = np.linalg.norm(ka - kf) / nf if nf > 0 else np.nan
            cos = (ka @ kf / (na * nf)) if na > 0 and nf > 0 else np.nan
            if h == hs[-1]:
                worst = max(worst, rel if np.isfinite(rel) else 0.0)
            print(f"{name:18s} {k:3d} {h:8.0e} {rel:11.3e} {cos:12.9f} {na:11.3e}")
        print()

    print(f"worst relative L2 at the finest h: {worst:.3e}")
    print("Expect FD truncation error to dominate: with --h-scan the disagreement "
          "should FALL with h. A floor that h cannot reduce is a real analytic error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
