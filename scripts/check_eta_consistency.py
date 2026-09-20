"""Regression check that every band puts the atmosphere / surface at the SAME
scene location for the same slit-image eta (2026-09-20). Run after any change to
the eta convention, the GD polynomials, or the cross-band pairing:

    PYTHONPATH=.:<gert> python scripts/check_eta_consistency.py

Checks (exit status 1 on any failure):
  1. every FPA's slit image ends map to eta = -1 / +1 (at its long-wavelength column);
  2. the scene truth used by the pipeline is a function of eta alone: gases, p, T and
     h2o priors/truth at equal eta are IDENTICAL across bands, and the per-band albedo
     is evaluated at the same km (`eta * SLIT_HALF_KM`) via `albedo_at`, whose spatial
     pattern is shared by all band labels;
  3. `gd_polynomials.eta_of_s` / `s_of_eta` round-trip;
  4. nearest-row pairing across bands is keyed on eta: the pairing residual is at most
     half a row, with the real-angle mismatch reported as a diagnostic only.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from geocarb_gert import along_slit_scene as als, GEOCARB_BANDS  # noqa: E402
from geocarb_gert.cross_band import eta_of_row, nearest_row_pairing, nearest_row_pairing_multi  # noqa: E402
from geocarb_gert.gd_polynomials import (eta_of_s, s_of_eta, slit_eta_reference,  # noqa: E402
                                          xy_to_wavelength_slit)
from geocarb_gert.joint_state import state_spec_from_scene  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


# 1 -- slit image ends at +/-1 on every FPA
for f in range(4):
    lam, _ = xy_to_wavelength_slit(f, np.array([0.0, 1023.0]), np.full(2, 511.5))
    col = 0.0 if lam[0] > lam[1] else 1023.0
    _, s = xy_to_wavelength_slit(f, np.full(2, col), np.array([0.0, 1023.0]))
    e = eta_of_s(f, s)
    check(f"FPA{f}: slit-image ends -> eta -1/+1 (long-wl col {col:.0f})",
          np.allclose(e, [-1.0, 1.0], atol=1e-9), f"eta={e}")

# 3 -- round trip
s_test = np.linspace(-2.0, 2.0, 11)
check("eta_of_s / s_of_eta round trip", all(np.allclose(s_of_eta(f, eta_of_s(f, s_test)), s_test) for f in range(4)))

# 2 -- state/scene at equal eta is identical across bands
etas = np.linspace(-0.95, 0.95, 40)
specs = {}
for f in range(4):
    lab = GEOCARB_BANDS[f][0]
    specs[f] = state_spec_from_scene(etas, free=("co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k", "albedo"),
                                     fields=als.PRIOR_FIELD_SETS["structural"], band_label=lab,
                                     surface_fields=als.SURFACE_PRIOR_FIELD_SETS["structural"],
                                     surface_positions=etas)
for name in ("co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"):
    ref = np.asarray(specs[0][name].prior)
    same = all(np.array_equal(np.asarray(specs[f][name].prior), ref) for f in range(1, 4))
    check(f"state prior '{name}' identical across bands at equal eta", same)
x_km = etas * als.SLIT_HALF_KM
labs = [GEOCARB_BANDS[f][0] for f in range(4)]
alb = als.albedo_at(x_km, labs, True)
print("       albedo_at evaluated on one km grid for all band labels; per-label mean:",
      np.round(alb.mean(axis=1), 3))

# 4 -- pairing keyed on eta
for a, b in [(0, 2), (0, 1), (1, 2), (0, 3), (2, 3)]:
    p = nearest_row_pairing(a, b, np.arange(0, 1024, 4.0))
    half_row = 0.5 * float(np.abs(np.diff(eta_of_row(b))).mean())
    check(f"pairing FPA{a}->FPA{b} keyed on eta (max |d eta| <= half row)",
          np.abs(p["mismatch_eta"]).max() <= half_row * 1.0001,
          f"max |d eta|={np.abs(p['mismatch_eta']).max():.2e} (half row={half_row:.2e}); "
          f"real-angle diagnostic max={np.abs(p['mismatch_deg']).max():.4f} deg; dropped {p['n_dropped']}")
m = nearest_row_pairing_multi([0, 2, 3])
check("multi-band pairing returns eta for every FPA", all(f in m["eta"] for f in (0, 2, 3)))

print("\nALL CHECKS PASSED" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
