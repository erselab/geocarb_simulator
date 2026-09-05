#!/usr/bin/env python3
"""Cross-parameter leakage in a resolution-matched-truth sweep, and how
much of it a simple linear post-correction could remove from CO2 -- the
true science product.

Extends Sec.7.8's window-mean cross-parameter correlation to NATIVE
(bin-level) resolution, now against a resolution-matched truth (so the
correlation isn't muddied by genuinely-unresolvable sub-anchor structure),
and distinguishes two different questions:

  (a) corr(co2_err, albedo_err) -- the idealized case IF we somehow knew
      albedo's own true error (we don't, in real data, without an
      independent reference).
  (b) corr(co2_err, albedo_dev_from_prior) -- the REALIZABLE diagnostic:
      albedo_dev_from_prior = retrieved_albedo - albedo's own PRIOR value,
      available from the retrieval alone, no truth needed. If this
      correlates with co2_err, a real post-correction is possible: fit
      co2_err ~ a + b*albedo_dev_from_prior on a held-out/training set,
      then subtract the prediction from a CO2 estimate -- a real, usable
      bias-correction step, not just a diagnostic.

Both co2_ppm and albedo positions are identical per window (co2_ppm/
p_surface_hpa/albedo share one state grid, this session's own convention)
so no interpolation is needed to pair them up.

Run:  PYTHONPATH=. python3 scripts/gd_resolution_matched_leakage.py <pkl> [--anchor-density 4]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from gd_build_resolution_matched_truth import whole_slit_anchor_etas  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", type=str)
    ap.add_argument("--anchor-density", type=int, default=4,
                    help="which resolution-matched truth this pkl was scored against "
                         "(must match what --resolution-matched-anchor-density used)")
    ap.add_argument("--tol-km", type=float, default=15.0)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    fpa = payload["fpa"]
    band_label = GEOCARB_BANDS[fpa][0]

    anchor_etas = whole_slit_anchor_etas(fpa, args.anchor_density)
    anchor_x_km = anchor_etas * als.SLIT_HALF_KM
    rm_fields = als.resolution_matched_fields(anchor_x_km)
    rm_albedo_fn = als.resolution_matched_albedo_fn(anchor_x_km, band_label)

    x_km_all, co2_err_all, co2_dev_all = [], [], []
    alb_err_all, alb_dev_all = [], []
    # per-WINDOW mean SIGNED error (Sec.7.8's own convention -- a different
    # statistic from the bin-level one: cancels within-window noise, reveals
    # a shared "some windows are just harder" driver instead).
    win_x_km, win_co2_err, win_alb_err, win_co2_dev, win_alb_dev = [], [], [], [], []
    for r in payload["results"].values():
        params = r["hires"]["params"]
        pos = params["co2_ppm"]["positions"]
        assert np.array_equal(pos, params["albedo"]["positions"])
        x_km = pos * als.SLIT_HALF_KM
        co2_v = params["co2_ppm"]["values"]
        alb_v = params["albedo"]["values"]
        co2_err = co2_v - rm_fields["co2_ppm"](x_km)
        alb_err = alb_v - rm_albedo_fn(x_km)
        co2_dev = co2_v - params["co2_ppm"]["prior"]     # realizable: no truth needed
        alb_dev = alb_v - params["albedo"]["prior"]        # realizable: no truth needed
        x_km_all.append(x_km); co2_err_all.append(co2_err); alb_err_all.append(alb_err)
        co2_dev_all.append(co2_dev); alb_dev_all.append(alb_dev)
        win_x_km.append(np.mean(x_km))
        win_co2_err.append(np.mean(co2_err)); win_alb_err.append(np.mean(alb_err))
        win_co2_dev.append(np.mean(co2_dev)); win_alb_dev.append(np.mean(alb_dev))

    x_km = np.concatenate(x_km_all)
    co2_err = np.concatenate(co2_err_all)
    alb_err = np.concatenate(alb_err_all)
    win_x_km = np.array(win_x_km)
    win_co2_err, win_alb_err = np.array(win_co2_err), np.array(win_alb_err)
    win_co2_dev, win_alb_dev = np.array(win_co2_dev), np.array(win_alb_dev)
    co2_dev = np.concatenate(co2_dev_all)
    alb_dev = np.concatenate(alb_dev_all)

    boundaries = np.array([pp[0] for pp in als.SURFACE_PATCHES[1:]])
    dist = np.min(np.abs(x_km[:, None] - boundaries[None, :]), axis=1)
    near = dist <= args.tol_km

    def report(mask, label, co2_e, alb_e, alb_d):
        n = mask.sum()
        if n < 3:
            print(f"  {label}: n={n}, too few to correlate")
            return
        corr_idealized = np.corrcoef(co2_e[mask], alb_e[mask])[0, 1]
        corr_realizable = np.corrcoef(co2_e[mask], alb_d[mask])[0, 1]
        # simple linear fit: co2_err ~ a + b*alb_dev (in-sample, no held-out split --
        # a first look at whether the relationship is even worth pursuing properly)
        b, a = np.polyfit(alb_d[mask], co2_e[mask], 1)
        residual = co2_e[mask] - (a + b * alb_d[mask])
        mae_before = np.mean(np.abs(co2_e[mask]))
        mae_after = np.mean(np.abs(residual))
        print(f"  {label}: n={n}")
        print(f"    corr(co2_err, albedo_err) [idealized, needs truth]:      {corr_idealized:+.3f}")
        print(f"    corr(co2_err, albedo_dev_from_prior) [realizable]:       {corr_realizable:+.3f}")
        print(f"    linear post-correction (co2_err ~ {a:.3f} + {b:.3f}*albedo_dev):")
        print(f"      MAE before: {mae_before:.4f} ppm   MAE after: {mae_after:.4f} ppm   "
             f"({'IMPROVED' if mae_after < mae_before else 'no improvement'}, "
             f"{(1-mae_after/mae_before)*100:+.1f}%)")

    print(f"=== leakage / post-correction analysis: {args.pkl} ===")
    print("### NATIVE BIN-LEVEL (410 bins) ###")
    print("-- whole slit --")
    report(np.ones_like(x_km, dtype=bool), "all", co2_err, alb_err, alb_dev)
    print("-- near boundary (<=%gkm) --" % args.tol_km)
    report(near, "near", co2_err, alb_err, alb_dev)
    print("-- far from boundary --")
    report(~near, "far", co2_err, alb_err, alb_dev)

    win_dist = np.min(np.abs(win_x_km[:, None] - boundaries[None, :]), axis=1)
    win_near = win_dist <= args.tol_km
    print(f"\n### PER-WINDOW MEAN SIGNED ERROR ({len(win_x_km)} windows) "
         f"-- Sec.7.8's own convention ###")
    print("-- whole slit --")
    report(np.ones_like(win_x_km, dtype=bool), "all", win_co2_err, win_alb_err, win_alb_dev)
    print("-- near boundary (<=%gkm) --" % args.tol_km)
    report(win_near, "near", win_co2_err, win_alb_err, win_alb_dev)
    print("-- far from boundary --")
    report(~win_near, "far", win_co2_err, win_alb_err, win_alb_dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
