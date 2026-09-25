"""Does the single-band vs multi-band albedo gap come from the anchor grid? (2026-09-25)

Three arms per band pairing (no noise, realistic prior, no aerosol, identical tiles and gas/p/H2O/T bins):
  SB-driver : gd_joint_block_retrieve.py single-band sweep (its own anchor grid)
  SB-mbcode : the SAME single band through gd_multiband_window.py (--fpas <n>; "cover" anchors, as the multi-band runs)
  MB        : FPA0 + FPA<n> through gd_multiband_window.py
Prints pooled boundary-trimmed rms error of the band's albedo and of the shared state rows for each arm.
    PYTHONPATH=.:<gert> python scripts/compare_albedo_codepaths.py
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402

H = als.SLIT_HALF_KM
LAB = {1: "CO2_weak", 2: "CO2_strong", 3: "CH4_CO"}
CFG = {2: ("co2-p-h2o-t-albedo", 33, "geometry_config_noaero_realistic.json", ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"]),
       1: ("co2-ch4-co-p-h2o-t-albedo", 60, "geometry_config_fpa0-1_realistic.json", ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"]),
       3: ("co2-ch4-co-p-h2o-t-albedo", 47, "geometry_config_fpa0-3_realistic.json", ["ch4_ppb", "co_ppb", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"])}


def truth(n, x, lab):
    return np.asarray(als.SURFACE_FIELDS["albedo"](x, lab)) if n == "albedo" else np.asarray(als.STATE_FIELDS[n](x))


def err(params, name, pname, lab):
    x = np.asarray(params[pname]["positions"]) * H
    return (np.asarray(params[pname]["values"]) - truth(name, x, lab))[1:-1]


for pair, (tag, nw, geom, rows) in CFG.items():
    tiles = json.load(open(REPO / geom))["tiles"]
    sbd_dir = f"results/realistic_prior/gd_joint_block_whole_slit_fpa{pair}_gratio1_adens4_free-{tag}_nwin{nw}_analytic_prior-realistic_valb_spos-anchor_fapos-anchor_etaslit"
    sbd = pickle.load(open(glob.glob(str(REPO / sbd_dir / "*.pkl"))[0], "rb"))["results"]
    e = {a: {n: [] for n in rows + ["albedo"]} for a in ("SB-driver", "SB-mbcode", "MB")}
    for t in tiles:
        key = tuple(t["rows_by_fpa"][str(pair)])
        r0 = t["rows_by_fpa"]["0"]
        f1 = glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa{pair}_r{pair}-{key[0]}-{key[1]}_free-{tag}_cover_g1.0_etaslit_prior-realistic_geomcfg.pkl"))
        f2 = glob.glob(str(REPO / f"results/realistic_prior/multiband/mb_fpa0-{pair}_r0-*_r{pair}-{key[0]}-{key[1]}_free-{tag}_cover_g1.0_etaslit_prior-realistic.pkl"))
        if not f1 or not f2 or key not in sbd:
            continue
        P = {"SB-driver": sbd[key]["hires"]["params"],
             "SB-mbcode": pickle.load(open(f1[0], "rb"))["joint"]["params"],
             "MB": pickle.load(open(f2[0], "rb"))["joint"]["params"]}
        for a, p in P.items():
            for n in rows:
                e[a][n].append(err(p, n, n, LAB[pair]))
            e[a]["albedo"].append(err(p, "albedo", "albedo" if a == "SB-driver" else f"albedo_{LAB[pair]}", LAB[pair]))
    n_t = len(e["MB"]["albedo"])
    print(f"\nFPA{pair} vs FPA0+FPA{pair}: {n_t}/{len(tiles)} tiles with all three arms")
    print(f"{'':16s} " + " ".join(f"{a:>12s}" for a in e))
    for n in rows + ["albedo"]:
        vals = [np.sqrt(np.mean(np.concatenate(e[a][n]) ** 2)) if e[a][n] else np.nan for a in e]
        print(f"{n:16s} " + " ".join(f"{v:12.4g}" for v in vals))
    alb_n = {a: int(sum(x.size for x in e[a]["albedo"])) for a in e}
    print(f"{'albedo points':16s} " + " ".join(f"{alb_n[a]:12d}" for a in e))
