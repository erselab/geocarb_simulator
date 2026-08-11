#!/usr/bin/env python3
"""Plain AND robust (MAD-outlier-resistant) bias summary statistics across
the full multi-band battery (gd_test.py), for any set of FPA combinations.

Formalizes the ad hoc aggregation used to build the summary tables in
NATIVE_VS_UNDISTORTED_ARTIFACT.html into a real, committed, rerunnable
script -- and adds the robust half that was missing there: those tables
reported plain `np.mean`/`np.std` straight off every converged row, with
no outlier handling at all (unlike gd_plot.py/gd_plot_grid.py's own
along-slit *plots*, which already MAD-filter chi2 outliers before
plotting). A handful of catastrophic barcode-edge rows can dominate a
plain std while barely nudging a robust one -- e.g. FPA0+FPA3 ch4,
barcode, no noise: plain std ~132 ppb, largely a few-row artifact once
those rows are down-weighted (see --n-mad to check how few).

Two outlier layers, reported side by side per pipeline/quantity:
  - chi2-filtered: same MAD-on-log10(reduced chi2) filtering gd_plot.py/
    gd_plot_grid.py already apply before plotting (rows with a
    catastrophic chi2 dropped first), ANDed with an absolute chi2 > 1
    floor -- a reduced chi2 at or below 1 is a statistically consistent
    fit by definition and is never flagged, however far it sits from the
    bulk in MAD terms. Without that floor this over-flags real bimodal
    "good" populations: undistorted's own barcode-scene chi2 splits
    cleanly into bright-bar rows near float precision (~1e-9) and
    dark-bar rows at a still-tiny ~1e-3 -- both excellent fits, but a
    plain log-MAD filter flagged roughly half of them as "outliers"
    purely because the bright-bar mode's near-zero spread makes the
    small bright/dark gap look huge in robust-sigma units.
  - value-robust: median and MAD-sigma of the quantity itself (not chi2),
    computed on the chi2-filtered set -- catches the different failure
    mode of a row with an unremarkable chi2 but a wildly biased gas/
    pressure retrieval anyway.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_summary_stats.py
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_summary_stats.py \\
        --configs "0,2 0,1,2" --scenes barcode --csv-out plots/barcode_stats.csv
Output: a table to stdout; optionally the same data as CSV via --csv-out.
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import pickle
from pathlib import Path

import numpy as np

from geocarb_gert import chi2_outlier_mask, robust_mean_std
from geocarb_gert.cross_band import fpas_tag

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_PIPELINES = ("native", "rectified", "undistorted")
SCENES = ("realistic", "uniform", "barcode")
NOISES = (False, True)
DEFAULT_CONFIGS = ["0", "1", "2", "3", "0,1", "0,2", "0,3", "0,1,2"]
_SPECIAL_KEYS = {"_state", "_state_full", "_chi2", "_conv", "_diverged", "_residuals", "_nus"}


def _mode_suffix(scene: str) -> str:
    return {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}[scene]


def _result_path(fpas, scene: str, noise: bool) -> Path:
    suffix = _mode_suffix(scene) + ("_noise" if noise else "")
    return REPO_ROOT / "results" / f"gd_joint_{fpas_tag(fpas)}{suffix}.pkl"


def _quantities(out: dict) -> list:
    """Which bias quantities this combination actually retrieves -- read
    directly off a converged native row's own keys (same convention as
    gd_plot.py's _gas_list / gd_plot_grid.py's _infer_gases), so this
    works unmodified for any future FPA combination."""
    keys = set()
    for (pl, order, rows), v in out.items():
        if pl != "native" or v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        for k in v:
            if k not in _SPECIAL_KEYS:
                keys.add(k)
    return sorted(keys)


def _values_for(out: dict, pipeline: str, quantity: str, n_mad: float):
    """chi2-filtered values for (pipeline, quantity), plus how many rows
    the chi2 filter itself dropped."""
    chi2s, vals = [], []
    for (pl, order, rows), v in out.items():
        if pl != pipeline or v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        if quantity not in v:
            continue
        chi2s.append(v["_chi2"])
        vals.append(v[quantity])
    chi2s = np.array(chi2s, dtype=float)
    vals = np.array(vals, dtype=float)
    if len(chi2s) == 0:
        return vals, 0
    outlier = chi2_outlier_mask(chi2s, n_mad=n_mad)
    return vals[~outlier], int(outlier.sum())


def compute_rows(fpas, scene: str, noise: bool, n_mad: float):
    path = _result_path(fpas, scene, noise)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    quantities = _quantities(out)
    label = "+".join(f"FPA{f}" for f in fpas)
    rows = []
    for q in quantities:
        for pl in ALL_PIPELINES:
            vals, n_chi2_outlier = _values_for(out, pl, q, n_mad)
            if len(vals) == 0:
                continue
            r = robust_mean_std(vals, n_mad=n_mad)
            rows.append(dict(
                combo=label, scene=scene, noise=noise, quantity=q, pipeline=pl,
                n=r["n"], mean=r["mean"], std=r["std"],
                median=r["median"], mad_sigma=r["mad_sigma"],
                n_value_outliers=r["n_outliers"], n_chi2_outliers=n_chi2_outlier,
                trimmed_mean=r["trimmed_mean"], trimmed_std=r["trimmed_std"],
            ))
    return rows


def print_table(rows):
    if not rows:
        print("(no data)")
        return
    hdr = (f"{'combo':16s} {'scene':10s} {'noise':5s} {'quantity':10s} {'pipeline':11s} "
          f"{'n':>5s} {'mean':>11s} {'std':>11s} {'median':>11s} {'MAD-sig':>11s} "
          f"{'n_out(chi2/val)':>16s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['combo']:16s} {r['scene']:10s} {str(int(r['noise'])):5s} {r['quantity']:10s} "
             f"{r['pipeline']:11s} {r['n']:5d} {r['mean']:+11.4g} {r['std']:11.4g} "
             f"{r['median']:+11.4g} {r['mad_sigma']:11.4g} "
             f"{r['n_chi2_outliers']:>7d}/{r['n_value_outliers']:<7d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", type=str, default=" ".join(DEFAULT_CONFIGS),
                    help="space-separated FPA combos, each comma-separated FPA "
                         "indices, e.g. '0 1,2' for FPA0 alone and FPA1+FPA2 "
                         "jointly (default: the 8 standard configs)")
    ap.add_argument("--scenes", type=str, default=",".join(SCENES),
                    help="comma-separated subset of realistic,uniform,barcode")
    ap.add_argument("--noise", type=str, default="0,1",
                    help="comma-separated subset of 0,1 (no-noise, with-noise)")
    ap.add_argument("--n-mad", type=float, default=8.0,
                    help="outlier threshold in robust-sigma units (default: 8.0, "
                         "matching gd_plot.py/gd_plot_grid.py's own chi2 filtering)")
    ap.add_argument("--csv-out", type=str, default=None,
                    help="optional path to also write the table as CSV")
    args = ap.parse_args()

    configs = [[int(x) for x in c.split(",")] for c in args.configs.split()]
    scenes = [s.strip() for s in args.scenes.split(",")]
    noises = [bool(int(x)) for x in args.noise.split(",")]

    all_rows = []
    for fpas in configs:
        for scene in scenes:
            for noise in noises:
                rows = compute_rows(fpas, scene, noise, args.n_mad)
                if rows is None:
                    print(f"MISSING: {_result_path(fpas, scene, noise).name}")
                    continue
                all_rows.extend(rows)

    print_table(all_rows)

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            fieldnames = ["combo", "scene", "noise", "quantity", "pipeline", "n",
                         "mean", "std", "median", "mad_sigma", "n_value_outliers",
                         "n_chi2_outliers", "trimmed_mean", "trimmed_std"]
            w = csv_mod.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
