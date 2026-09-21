"""The "realistic" prior (als.PRIOR_FIELD_SETS["realistic"], SURFACE_PRIOR_FIELD_SETS["realistic"]) must not equal
the truth ANYWHERE along the slit (user, 2026-09-21: an exact prior makes 'the retrieval is worse than the prior'
findings meaningless). For every state/surface field: evaluate truth and prior on a dense grid and require a
minimum |error| (absolute for gases/T/p/height, relative for h2o/aerosol/albedo). The old 'structural' prior is
shown alongside for reference. Exit status 1 on failure.

    PYTHONPATH=.:<gert> python scripts/check_realistic_prior.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocarb_gert import along_slit_scene as als  # noqa: E402

x = np.linspace(-1400.0, 1400.0, 5601)       # 0.5 km grid, both ends included
# (name, floor, kind): kind "abs" -> |prior - truth| >= floor [units]; "rel" -> |prior/truth - 1| >= floor
CHECKS = [("co2_ppm", 0.15, "abs", "ppm"), ("ch4_ppb", 5.0, "abs", "ppb"), ("co_ppb", 3.0, "abs", "ppb"),
          ("t_offset_k", 0.30, "abs", "K"), ("p_surface_hpa", 0.50, "abs", "hPa"), ("h2o_surface_vmr", 0.04, "rel", ""),
          ("amplitude_aerosol", 0.15, "rel", ""), ("height_aerosol", 2000.0, "abs", "Pa"),
          ("thickness_aerosol", 0.05, "rel", ""), ("albedo:O2_A", 0.04, "rel~", ""), ("albedo:CO2_strong", 0.04, "rel~", "")]
# "rel~": a relative floor that can only be met on ALL but a small fraction (<= ALBEDO_MAX_FRAC) of the grid --
# the truth's random fine-scale albedo texture makes a strict floor impossible without an unrealistic bias.
ALBEDO_MAX_FRAC = 0.03
fails = []


def fields(name, prior_key):
    surf = name.split(":")[0] in als.SURFACE_FIELDS
    label = name.split(":")[1] if ":" in name else None
    base = name.split(":")[0]
    if surf:
        tr = als.SURFACE_FIELDS[base]
        pr = als.SURFACE_PRIOR_FIELD_SETS[prior_key][base]
        return np.asarray(tr(x, label)), np.asarray(pr(x, label))
    return np.asarray(als.STATE_FIELDS[base](x)), np.asarray(als.PRIOR_FIELD_SETS[prior_key][base](x))


print(f"{'field':20s} {'kind':4s} {'floor':>8s} | realistic: min|err|  max|err|  rms  | structural: min|err|  (grid points with |err| < floor)")
for name, floor, kind, unit in CHECKS:
    row = []
    for key in ("realistic", "structural"):
        t, p = fields(name, key)
        e = np.abs(p - t) if kind == "abs" else np.abs(p / np.where(t == 0, np.nan, t) - 1.0)
        row.append((np.nanmin(e), np.nanmax(e), np.sqrt(np.nanmean(e ** 2)), int(np.sum(e < floor)), np.isnan(p).any()))
    (mn, mx, rms, nbad, nan), (smn, smx, srms, snbad, snan) = row
    ok = (nbad <= ALBEDO_MAX_FRAC * x.size if kind == "rel~" else nbad == 0) and not nan
    if not ok:
        fails.append(name)
    print(f"{name:20s} {kind:4s} {floor:8.3g} | {mn:9.3g} {mx:9.3g} {rms:8.3g} {'ok ' if ok else 'FAIL'} ({nbad}/{x.size} below floor) | {smn:9.3g}  ({snbad}/{x.size})")
# derived AOD (= amplitude * thickness * sqrt(2 pi)): the PRODUCT of two individually-offset rows can still equal the truth
for key in ("realistic", "structural"):
    tr = np.asarray(als.tau_aerosol(x))
    pr = (np.asarray(als.SURFACE_PRIOR_FIELD_SETS[key]["amplitude_aerosol"](x)) *
          np.asarray(als.SURFACE_PRIOR_FIELD_SETS[key]["thickness_aerosol"](x)) * np.sqrt(2 * np.pi))
    e = np.abs(pr / tr - 1.0)
    ok = key != "realistic" or bool(np.all(e >= 0.10))
    print(f"AOD (amplitude*thickness*sqrt(2pi)), {key:10s} prior: min |prior/truth - 1| = {e.min():.3f}, max {e.max():.3f}"
          f"{'  (floor 0.10) ' + ('ok' if ok else 'FAIL') if key == 'realistic' else ''}")
    if not ok:
        fails.append("AOD product floor")
# features: the gas priors must carry no plume / hot spot -- they are pure low-frequency curves
for name in ("co2_ppm", "ch4_ppb", "co_ppb"):
    p = np.asarray(als.PRIOR_FIELD_SETS["realistic"][name](x))
    d2 = np.abs(np.diff(p, 2)).max()
    print(f"{name}: realistic prior max |second difference| per 0.5 km step = {d2:.2e} (smooth; no localized feature)")
    if d2 > 1e-3 * max(np.ptp(p), 1.0):
        fails.append(name + " smoothness")
# aerosol layer centroid >= 100 m above the surface (user, 2026-09-21): truth and every prior set
print()
for label, hfn, pfn in (("truth", als.height_aerosol, als.p_surface_hpa),
                        ("structural prior", als.height_aerosol_prior, als.p_surface_hpa_prior),
                        ("realistic prior", als.height_aerosol_prior_realistic, als.p_surface_hpa_prior_realistic)):
    h, ps = np.asarray(hfn(x)), np.asarray(pfn(x)) * 100.0
    margin_m = -als._AIR_SCALE_HEIGHT_M * np.log(h / ps)            # height above the surface implied by the pressure ratio
    ok = bool(np.all(margin_m >= als.AEROSOL_MIN_HEIGHT_ABOVE_SURFACE_M - 1e-6))
    print(f"aerosol height above the surface, {label:17s}: min {margin_m.min():8.1f} m ({(margin_m < 1e-6).sum()} points at/below ground) -> {'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"aerosol height cap ({label})")
print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
