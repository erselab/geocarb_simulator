#!/usr/bin/env python3
"""Score submit_sub_bin_anomaly_v1.sbatch's 9 whole-slit outputs.

Generalizes gd_information_density_bins_demo.py::_native_error_summary's
single-window near/far-boundary split to the WHOLE slit: "near" is within
`--tol-km` of ANY `als.SURFACE_PATCHES` internal boundary (12 of them across
FPA2's slit), not just the one boundary a single demo window straddles.
Same scoring convention otherwise -- raw retrieved albedo bin values
(StateSpec.snapshot's own "values", untouched by sub_bin_anomaly) against
truth sampled at each bin's own center point (stack_windows_along_slit's
own eta), matching every number already reported in docs/PROJECT_STATUS.md
Sec.10 for the single-window comparisons.

Run:  PYTHONPATH=. python3 scripts/gd_sub_bin_anomaly_sweep_score.py
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


def _whole_slit_summary(pkl_path, fpa, tol_km=15.0):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    windows = list(payload["results"].values())
    stack = stack_windows_along_slit(windows, "albedo", solve="hires")
    x_km = stack.eta * als.SLIT_HALF_KM
    truth = als.albedo_for_label(x_km, GEOCARB_BANDS[fpa][0])  # include_fine=True default
    err = stack.values - truth
    boundaries = np.array([p[0] for p in als.SURFACE_PATCHES[1:]])
    dist = np.min(np.abs(x_km[:, None] - boundaries[None, :]), axis=1)
    near = dist <= tol_km
    far = ~near
    return dict(n=len(x_km), mean_abs_err=float(np.mean(np.abs(err))),
               near_mean_abs_err=float(np.mean(np.abs(err[near]))) if near.any() else float("nan"),
               far_mean_abs_err=float(np.mean(np.abs(err[far]))) if far.any() else float("nan"),
               n_near=int(near.sum()), n_far=int(far.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol-km", type=float, default=15.0)
    args = ap.parse_args()

    print(f"{'setup':8s} {'ad':>3s} {'n_bins':>8s} {'overall':>10s} "
         f"{'near(<=%gkm)' % args.tol_km:>15s} {'far':>10s} {'n_near':>7s} {'n_far':>7s}")
    rows = []
    for setup, ad, fname in CONFIGS:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"{setup:8s} {ad:3d}  MISSING: {path}")
            continue
        with open(path, "rb") as f:
            fpa = pickle.load(f)["fpa"]
        s = _whole_slit_summary(path, fpa, tol_km=args.tol_km)
        rows.append((setup, ad, s))
        print(f"{setup:8s} {ad:3d} {s['n']:8d} {s['mean_abs_err']:10.4f} "
             f"{s['near_mean_abs_err']:15.4f} {s['far_mean_abs_err']:10.4f} "
             f"{s['n_near']:7d} {s['n_far']:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
