"""Standalone analytic-vs-FD validation for the XRTM-path aerosol Jacobians
(height_aerosol_dI_dparam_xrtm, thickness_aerosol_dI_dparam_xrtm,
amplitude_aerosol_dI_dparam) -- Phase 4 of the XRTM integration plan
(2026-09-15). Mirrors gd_jacobian_validate.py's own setup pattern, but
scoped to a single anchor (no window solve, no GN loop) since XRTM's
per-wavenumber Python loop makes even one forward call materially more
expensive than SingleScatterSolver's closed form.

Also the direct test of the whole exercise's own point: confirms
height_aerosol's Jacobian is now smooth as height crosses a pressure-layer
boundary, unlike SingleScatterSolver's tau_abv-based version.

Run:  PYTHONPATH=.:/scratch/scrowel3_lab/gert python3 scripts/gd_xrtm_aerosol_jacobian_validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.spectrum import simulate_spectrum, spectrum_and_jacobian  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics  # noqa: E402


def main():
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    atm0 = als.atmosphere_at(0.0)
    wide_win, wide_inst, albedo = band_basics(FPA, atm0, absco, geo, solar)

    atm_params = {name: float(fn(np.array([0.0]))[0]) for name, fn in als.STATE_FIELDS.items()}
    height0, thickness0 = 85000.0, 1.0e4
    amp0 = 0.05 / (thickness0 * np.sqrt(2 * np.pi))
    surface = dict(albedo=albedo, amplitude_aerosol=amp0, height_aerosol=height0,
                  thickness_aerosol=thickness0)

    print("--- forward radiance check (XRTM vs SingleScatterSolver) ---")
    for solver in ("single_scatter", "xrtm"):
        sr = simulate_spectrum(atm_params, surface, absco, wide_inst, geo, solar, solver=solver)
        print(f"{solver}: I_hires[:3]={sr.I_hires[:3]}")

    print("\n--- analytic vs FD, solver=xrtm ---")
    rows = ["amplitude_aerosol", "height_aerosol", "thickness_aerosol"]
    S0, d = spectrum_and_jacobian(atm_params, rows, absco, wide_inst, geo, solar,
                                  surface=surface, solver="xrtm")

    def fd(row, h):
        sfc_p = dict(surface); sfc_p[row] = surface[row] + h
        sfc_m = dict(surface); sfc_m[row] = surface[row] - h
        Sp = simulate_spectrum(atm_params, sfc_p, absco, wide_inst, geo, solar, solver="xrtm").I_hires
        Sm = simulate_spectrum(atm_params, sfc_m, absco, wide_inst, geo, solar, solver="xrtm").I_hires
        return (Sp - Sm) / (2 * h)

    steps = {"amplitude_aerosol": amp0 * 1e-3, "height_aerosol": 50.0, "thickness_aerosol": 50.0}
    for row in rows:
        fd_val = fd(row, steps[row])
        an_val = d[row]
        num = np.linalg.norm(an_val - fd_val)
        den = max(np.linalg.norm(fd_val), 1e-30)
        cos = float(np.dot(an_val, fd_val) / (np.linalg.norm(an_val) * den + 1e-30))
        print(f"{row:20s} rel_L2={num/den:.4e}  cos={cos:.6f}")

    print("\n--- smoothness check: height_aerosol Jacobian across a pressure-layer boundary ---")
    p_layers = np.asarray(als.atmosphere_from_params(**atm_params).p_layers, dtype=float)
    boundary = float(p_layers[len(p_layers) // 2])
    for solver in ("single_scatter", "xrtm"):
        vals = []
        for h in (200.0, 20.0, 2.0):
            _, dp = spectrum_and_jacobian(atm_params, ["height_aerosol"], absco, wide_inst, geo, solar,
                                          surface={**surface, "height_aerosol": boundary + h},
                                          solver=solver)
            _, dm = spectrum_and_jacobian(atm_params, ["height_aerosol"], absco, wide_inst, geo, solar,
                                          surface={**surface, "height_aerosol": boundary - h},
                                          solver=solver)
            vals.append(float(np.linalg.norm(dp["height_aerosol"] - dm["height_aerosol"])))
        print(f"{solver}: |K(boundary+h) - K(boundary-h)| as h shrinks (200,20,2 Pa): {vals}")


if __name__ == "__main__":
    main()
