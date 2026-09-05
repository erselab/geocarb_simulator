#!/usr/bin/env python3
"""Score submit_sub_bin_anomaly_v1.sbatch's 9 whole-slit outputs.

Generalizes gd_information_density_bins_demo.py::_native_error_summary's
single-window near/far-boundary split to the WHOLE slit: "near" is within
`--tol-km` of ANY `als.SURFACE_PATCHES` internal boundary (12 of them across
FPA2's slit), not just the one boundary a single demo window straddles.
Same scoring convention otherwise -- raw retrieved bin values
(StateSpec.snapshot's own "values") against truth sampled at each bin's own
center point (stack_windows_along_slit's own eta), matching every number
already reported in docs/PROJECT_STATUS.md Sec.10 for the single-window
comparisons.

Scores `--row albedo` (default, untouched by sub_bin_anomaly -- how well
the mechanism itself does) and `--row co2_ppm` (2026-08-28: `sub_bin_
anomaly` was only ever set on albedo's row, but co2_ppm is jointly free/
solved in every config, so this shows the SIDE EFFECT on the CO2-albedo
cross-talk documented in Sec.7.8 -- see Sec.10.4) -- `--row both` (default)
prints both in one run.

Run:  PYTHONPATH=. python3 scripts/gd_sub_bin_anomaly_sweep_score.py
      PYTHONPATH=. python3 scripts/gd_sub_bin_anomaly_sweep_score.py --row co2_ppm
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
from geocarb_gert.along_slit_state import stack_windows_along_slit  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results/realistic_prior/config_matrix/sub_bin_anomaly_v1/results"

CONFIGS = [
    ("none",  1,  "co2p_albedo-58-g3-pf-structural_analytic.pkl"),
    ("none",  4,  "co2p_albedo-58-g3-ad4-pf-structural_analytic.pkl"),
    ("none",  16, "co2p_albedo-58-g3-ad16-pf-structural_analytic.pkl"),
    ("truth", 1,  "co2p_albedo-58-g3-pf-structural-anomaly-truth_analytic.pkl"),
    ("truth", 4,  "co2p_albedo-58-g3-ad4-pf-structural-anomaly-truth_analytic.pkl"),
    ("truth", 16, "co2p_albedo-58-g3-ad16-pf-structural-anomaly-truth_analytic.pkl"),
    ("prior", 1,  "co2p_albedo-58-g3-pf-structural-anomaly-prior_analytic.pkl"),
    ("prior", 4,  "co2p_albedo-58-g3-ad4-pf-structural-anomaly-prior_analytic.pkl"),
    ("prior", 16, "co2p_albedo-58-g3-ad16-pf-structural-anomaly-prior_analytic.pkl"),
]

#: `row_name` -> `truth_fn(x_km, fpa) -> ndarray`. `albedo` is per-band
#: (needs fpa's own GEOCARB_BANDS label); `co2_ppm` is band-independent.
TRUTH_FNS = {
    "albedo": lambda x_km, fpa: als.albedo_for_label(x_km, GEOCARB_BANDS[fpa][0]),
    "co2_ppm": lambda x_km, fpa: als.xco2_ppm(x_km),
}


def _whole_slit_summary(pkl_path, row_name, fpa, tol_km=15.0):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    windows = list(payload["results"].values())
    stack = stack_windows_along_slit(windows, row_name, solve="hires")
    x_km = stack.eta * als.SLIT_HALF_KM
    truth = TRUTH_FNS[row_name](x_km, fpa)
    err = stack.values - truth
    boundaries = np.array([p[0] for p in als.SURFACE_PATCHES[1:]])
    dist = np.min(np.abs(x_km[:, None] - boundaries[None, :]), axis=1)
    near = dist <= tol_km
    far = ~near
    return dict(n=len(x_km), mean_abs_err=float(np.mean(np.abs(err))),
               near_mean_abs_err=float(np.mean(np.abs(err[near]))) if near.any() else float("nan"),
               far_mean_abs_err=float(np.mean(np.abs(err[far]))) if far.any() else float("nan"),
               rel_err=float(np.mean(np.abs(err)) / np.mean(truth)),
               n_near=int(near.sum()), n_far=int(far.sum()))


def _score_row(row_name, tol_km):
    print(f"\n=== {row_name} ===")
    print(f"{'setup':8s} {'ad':>3s} {'n_bins':>8s} {'overall':>10s} "
         f"{'near(<=%gkm)' % tol_km:>15s} {'far':>10s} {'rel':>8s}")
    for setup, ad, fname in CONFIGS:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"{setup:8s} {ad:3d}  MISSING: {path}")
            continue
        with open(path, "rb") as f:
            fpa = pickle.load(f)["fpa"]
        s = _whole_slit_summary(path, row_name, fpa, tol_km=tol_km)
        print(f"{setup:8s} {ad:3d} {s['n']:8d} {s['mean_abs_err']:10.4f} "
             f"{s['near_mean_abs_err']:15.4f} {s['far_mean_abs_err']:10.4f} "
             f"{s['rel_err']*100:7.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol-km", type=float, default=15.0)
    ap.add_argument("--row", choices=["albedo", "co2_ppm", "both"], default="both",
                    help="which state row to score -- 'albedo' (sub_bin_anomaly's own "
                         "target), 'co2_ppm' (jointly-solved, never given an anomaly -- "
                         "shows the cross-talk side effect, docs/PROJECT_STATUS.md "
                         "Sec.10.4), or 'both' (default).")
    args = ap.parse_args()

    rows = ["albedo", "co2_ppm"] if args.row == "both" else [args.row]
    for row_name in rows:
        _score_row(row_name, args.tol_km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
