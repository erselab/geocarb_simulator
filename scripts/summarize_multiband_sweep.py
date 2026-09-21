"""Per-tile summary and recovered-vs-truth errors for the multi-band (FPA0+FPA2, no aerosol) sweep.

Reads results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo_cover_g1.0_etaslit.pkl (one per
tile), matches each to build_window_tiles_multiband((0,2)) by its FPA0 rows, and prints a per-tile table plus rms
errors (retrieved and prior) by width tier for p, h2o, T, CO2 and both albedo rows.

    PYTHONPATH=.:<gert> python scripts/summarize_multiband_sweep.py
"""
import glob
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.multiband_geometry import build_window_tiles_multiband  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402

ROWS = [("p_surface_hpa", None, "p [hPa]"), ("h2o_surface_vmr", None, "h2o (vmr)"), ("t_offset_k", None, "T [K]"),
        ("co2_ppm", None, "CO2 [ppm]"), ("albedo_O2_A", "O2_A", "albedo O2_A"),
        ("albedo_CO2_strong", "CO2_strong", "albedo CO2_str")]

tiles = build_window_tiles_multiband((0, 2), gjr.MIN_WINDOW, 1.0, 2)
by_rows = {tuple(t.rows[0]): i for i, t in enumerate(tiles)}
files = sorted(glob.glob(str(REPO / "results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo_cover_g1.0_etaslit.pkl")))
res = {}
for f in files:
    d = pickle.load(open(f, "rb"))
    k = by_rows.get(tuple(d["rows_by_fpa"][0]))
    if k is not None:
        res[k] = d
missing = [i for i in range(len(tiles)) if i not in res]
print(f"{len(res)}/{len(tiles)} tiles present; missing: {missing}\n")

def errs(d, name, label):
    p = d["joint"]["params"][name]
    x = np.asarray(p["positions"]) * als.SLIT_HALF_KM
    tr = np.asarray(als.SURFACE_FIELDS["albedo"](x, label)) if label else np.asarray(als.STATE_FIELDS[name](x))
    return np.asarray(p["values"]) - tr, np.asarray(p["prior"]) - tr

print("tile  width  n_free  t_solve[s]  rms_resid  FPA0      FPA2")
for i in sorted(res):
    d = res[i]
    w = tiles[i].rows[0][1] - tiles[i].rows[0][0] + 1
    print(f"{i:4d}  {w:4d}  {d['n_free']:6d}  {d['t_solve']:9.0f}  {d['resid_rms']:.2e}  {d['resid_rms_by_band'][0]:.2e}  {d['resid_rms_by_band'][2]:.2e}")

tiers = {"A (width >= 35)": lambda w: w >= 35, "B (27-35)": lambda w: 27 <= w < 35, "C (<= 27)": lambda w: w < 27}
print("\nrms error, retrieved / prior, by width tier (all bins pooled)")
print(f"{'':16s}" + "".join(f"{lab:>24s}" for _, _, lab in ROWS))
for tn, cond in list(tiers.items()) + [("ALL", lambda w: True)]:
    ids = [i for i in res if cond(tiles[i].rows[0][1] - tiles[i].rows[0][0] + 1)]
    line = f"{tn + f' [{len(ids)}]':16s}"
    for name, label, _ in ROWS:
        e = np.concatenate([errs(res[i], name, label)[0] for i in ids]) if ids else np.zeros(1)
        pr = np.concatenate([errs(res[i], name, label)[1] for i in ids]) if ids else np.zeros(1)
        line += f"{np.sqrt(np.mean(e ** 2)):>12.3g}/{np.sqrt(np.mean(pr ** 2)):<11.3g}"
    print(line)
worse = {name: 0 for name, _, _ in ROWS}
for i in res:
    for name, label, _ in ROWS:
        e, pr = errs(res[i], name, label)
        worse[name] += np.sqrt(np.mean(e ** 2)) > np.sqrt(np.mean(pr ** 2))
print("\ntiles where the retrieved error exceeds the prior error:", {k: f"{v}/{len(res)}" for k, v in worse.items()})
