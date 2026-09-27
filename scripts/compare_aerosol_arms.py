"""FPA0+FPA2 aerosol arm: earlier default-damping run vs the --lm-fast rerun (2026-09-26), same legacy registry_smoke properties.

Pooled boundary-trimmed rms error per state variable (33 tiles), the no-aerosol arm for reference, per-tile solve time, and the
along-slit surface-pressure error of the three arms. Saves plots/aerosol_arm_lmfast_vs_old.png.
    PYTHONPATH=.:<gert> python scripts/compare_aerosol_arms.py
"""
import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402

H = als.SLIT_HALF_KM
MB = REPO / "results/realistic_prior/multiband"
VARS = ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k", "amplitude_aerosol", "height_aerosol"]


def truth(n, x):
    return np.asarray(als.STATE_FIELDS[n](x)) if n in als.STATE_FIELDS else np.asarray(als.SURFACE_FIELDS[n](x, "x"))


def arm(pattern):
    out = {}
    for f in sorted(glob.glob(str(MB / pattern))):
        d = pickle.load(open(f, "rb"))
        out[tuple(d["rows_by_fpa"][2])] = d
    return out


old = arm("mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo-amplitude-height_cover_g1.0_etaslit_prior-realistic_aero.pkl")
new = arm("mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo-amplitude-height_cover_g1.0_etaslit_prior-realistic_aero_registry_smoke_lmfast.pkl")
noa = arm("mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo_cover_g1.0_etaslit_prior-realistic.pkl")
print(f"tiles: old {len(old)}, new {len(new)}, no-aerosol {len(noa)}")
keys = sorted(set(old) & set(new))


def errs(arm_, n, key):
    p = arm_[key]["joint"]["params"][n]
    x = np.asarray(p["positions"]) * H
    return x[1:-1], (np.asarray(p["values"]) - truth(n, x))[1:-1], (np.asarray(p["prior"]) - truth(n, x))[1:-1]


print(f"\n{'variable':18s} {'prior':>10s} {'no-aerosol':>11s} {'old aerosol':>12s} {'new (lm-fast)':>14s}")
pool = {}
for n in VARS:
    row = {}
    for name, a in (("old", old), ("new", new), ("noa", noa)):
        if n not in a[keys[0]]["joint"]["params"]:
            row[name] = np.nan
            continue
        row[name] = np.sqrt(np.mean(np.concatenate([errs(a, n, k)[1] for k in keys if k in a]) ** 2))
    pe = np.sqrt(np.mean(np.concatenate([errs(old, n, k)[2] for k in keys]) ** 2))
    pool[n] = row
    print(f"{n:18s} {pe:10.4g} {row['noa']:11.4g} {row['old']:12.4g} {row['new']:14.4g}")

print("\nper-tile solve time [s] old -> new, and p_surface rms error old -> new (tiles where it changes most):")
rows = []
for i, k in enumerate(keys):
    eo, en = errs(old, "p_surface_hpa", k)[1], errs(new, "p_surface_hpa", k)[1]
    rows.append((i, old[k]["t_solve"], new[k]["t_solve"], np.sqrt(np.mean(eo ** 2)), np.sqrt(np.mean(en ** 2))))
for i, to, tn, po, pn in sorted(rows, key=lambda r: -abs(r[3] - r[4]))[:8]:
    print(f"  tile {i:2d}: solve {to:6.0f} -> {tn:6.0f} s   p_surface rms {po:.3f} -> {pn:.3f} hPa")
print(f"  total solve time: old {sum(r[1] for r in rows) / 3600:.1f} h, new {sum(r[2] for r in rows) / 3600:.1f} h (task-hours)")

fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
for a_, n, lab in ((ax[0], "p_surface_hpa", "surface pressure error [hPa]"), (ax[1], "height_aerosol", "aerosol height error [Pa]")):
    for arm_, col, name in ((old, "#c2185b", "earlier aerosol arm"), (new, "#1baf7a", "aerosol arm, --lm-fast"), (noa, "#8a93a1", "no aerosol")):
        if n not in arm_[keys[0]]["joint"]["params"]:
            continue
        X = np.concatenate([errs(arm_, n, k)[0] for k in keys if k in arm_])
        E = np.concatenate([errs(arm_, n, k)[1] for k in keys if k in arm_])
        o = np.argsort(X)
        a_.plot(X[o], E[o], ".", ms=2.5, color=col, alpha=0.7, label=f"{name} (rms {np.sqrt(np.mean(E ** 2)):.3g})")
    a_.axhline(0, color="#c3c2b7", lw=0.8)
    a_.set_ylabel(lab)
    a_.legend(frameon=False, fontsize=8, markerscale=4)
    a_.grid(True, color="#e6e5e0", lw=0.8)
ax[1].set_xlabel("along-slit position [km]")
fig.suptitle("FPA0+FPA2 with aerosol free: earlier default-damping run vs --lm-fast rerun (legacy registry_smoke, no noise)", fontsize=11)
fig.savefig(REPO / "plots/aerosol_arm_lmfast_vs_old.png", dpi=120, bbox_inches="tight")
print("saved plots/aerosol_arm_lmfast_vs_old.png")
