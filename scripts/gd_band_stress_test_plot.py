#!/usr/bin/env python3
"""Diagnostics for the along-slit composition/pressure stress test
(gd_band_stress_test.py). Mirrors gd_dense_sweep_plot.py's approach but
adds the along-slit truth profile for context and a failure-location panel
tied to physical position (x_km), since this scene's failures may cluster
near the pressure mountain / plumes rather than just the slit edges.

Each pipeline only has one dispersion order available (native/rectified:
order=2 only; undistorted: order=0 only -- see gd_band_stress_test.py's
pipeline_orders for why), so every panel here plots each pipeline at its
own single available order rather than assuming both exist everywhere.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_band_stress_test_plot.py --fpa 1
Output: plots/gd_band_stress_test_fpa<N>.png
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geocarb_gert import along_slit_scene as als

REPO_ROOT = Path(__file__).resolve().parent.parent

# pipeline -> its only available dispersion order (see gd_band_stress_test.py)
PIPELINE_ORDER = {"native": 2, "rectified": 2, "undistorted": 0}
PIPELINE_COLOR = {"native": "tab:blue", "rectified": "tab:orange", "undistorted": "tab:green"}


def _is_good(b) -> bool:
    return (b is not None and not b.get("_diverged") and b.get("_conv"))


def _bias_series(out, rows, x_km_of_row, pipeline, gas):
    order = PIPELINE_ORDER[pipeline]
    xs, biases = [], []
    for k in rows:
        b = out[(pipeline, order, int(k))]["bias"]
        if _is_good(b) and np.isfinite(b.get(gas, np.nan)):
            xs.append(x_km_of_row[k]); biases.append(b[gas])
    return np.array(xs), np.array(biases)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpa", type=int, required=True)
    ap.add_argument("--in-tag", type=str, default=None,
                    help="read results/gd_band_stress_test_fpa<N>_<tag>.pkl instead of the "
                         "untagged file (matches gd_band_stress_test.py's --out-tag)")
    args = ap.parse_args()
    FPA = args.fpa

    tag_suffix = f"_{args.in_tag}" if args.in_tag else ""
    with open(REPO_ROOT / "results" / f"gd_band_stress_test_fpa{FPA}{tag_suffix}.pkl", "rb") as f:
        d = pickle.load(f)
    out, rows, gases = d["out"], np.array(d["rows"]), d["gases"]
    x_km_of_row = d["x_km_of_row"]
    xtrue_of_row = d["xtrue_of_row"]
    # Bands that exclude well-mixed gases (o2, n2o -- see WELL_MIXED_GASES
    # in gd_band_stress_test.py) from `gases` retrieve only h2o_scale +
    # p_scale; p_scale (always tracked as a "p_surface" bias in hPa, see
    # _retrieve()) is the primary metric there instead of a gas column.
    _non_h2o = [g for g in gases if g != "h2o"]
    primary_gas = _non_h2o[0] if _non_h2o else "p_surface"
    primary_unit = "hPa" if primary_gas == "p_surface" else ""

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # -- Panel: primary-gas bias vs x_km, each pipeline at its own order --
    ax = axes[0, 0]
    for pipeline in ("native", "rectified", "undistorted"):
        xs, biases = _bias_series(out, rows, x_km_of_row, pipeline, primary_gas)
        ax.plot(xs, biases, ".", ms=2, label=f"{pipeline} (order={PIPELINE_ORDER[pipeline]})",
               color=PIPELINE_COLOR[pipeline], alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{primary_gas} bias vs along-slit position")
    ax.set_xlabel("x [km]"); ax.set_ylabel(f"{primary_gas} bias" + (f" [{primary_unit}]" if primary_unit else ""))
    ax.legend(fontsize=8)

    # -- Panel: h2o bias vs x_km, each pipeline at its own order --
    ax = axes[0, 1]
    for pipeline in ("native", "rectified", "undistorted"):
        xs, biases = _bias_series(out, rows, x_km_of_row, pipeline, "h2o")
        ax.plot(xs, biases, ".", ms=2, label=f"{pipeline} (order={PIPELINE_ORDER[pipeline]})",
               color=PIPELINE_COLOR[pipeline], alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("h2o bias vs along-slit position")
    ax.set_xlabel("x [km]"); ax.set_ylabel("h2o bias [ppm]")
    ax.legend(fontsize=8)

    # -- Panel: true primary-gas + pressure profile for context --
    ax = axes[1, 0]
    order_idx = np.argsort(x_km_of_row)
    ax.plot(x_km_of_row[order_idx], xtrue_of_row[primary_gas][order_idx], color="tab:red")
    ax.set_ylabel(f"true {primary_gas}" + (f" [{primary_unit}]" if primary_unit else ""), color="tab:red")
    ax.set_xlabel("x [km]")
    ax.set_title("Truth profile for context")
    if primary_gas != "p_surface":
        ax2 = ax.twinx()
        ax2.plot(x_km_of_row[order_idx], als.p_surface_hpa(x_km_of_row[order_idx]), color="tab:green", alpha=0.5)
        ax2.set_ylabel("true p_surface [hPa]", color="tab:green")

    # -- Panel: failure location map, each pipeline at its own order --
    ax = axes[1, 1]
    for pipeline, color, yoff in (("native", "tab:blue", 0.3), ("rectified", "tab:orange", 0.15),
                                  ("undistorted", "tab:green", 0.0)):
        order = PIPELINE_ORDER[pipeline]
        xs = [x_km_of_row[k] for k in rows if (b := out[(pipeline, order, int(k))]["bias"]) is not None and b.get("_diverged")]
        ax.plot(xs, [yoff] * len(xs), "x", color=color, ms=4, label=f"{pipeline} diverged (n={len(xs)})")
    for pipeline, color, yoff in (("native", "tab:brown", -0.15), ("rectified", "tab:pink", -0.3)):
        order = PIPELINE_ORDER[pipeline]
        xs = [x_km_of_row[k] for k in rows if (b := out[(pipeline, order, int(k))]["bias"]) is not None
             and not b.get("_diverged") and not b.get("_conv")]
        ax.plot(xs, [yoff] * len(xs), "+", color=color, ms=5, label=f"{pipeline} stalled (n={len(xs)})")
    offdet_xs = [x_km_of_row[k] for k in rows if out[("rectified", 2, int(k))]["bias"] is None]
    ax.plot(offdet_xs, [-0.45] * len(offdet_xs), "o", color="gray", ms=3, label=f"rectified off-detector (n={len(offdet_xs)})")
    ax.set_title("Where each pipeline fails, vs along-slit position")
    ax.set_xlabel("x [km]"); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper center", ncol=2)

    # -- Panel: chi2 vs x_km, each pipeline at its own order --
    ax = axes[2, 0]
    for pipeline in ("native", "rectified", "undistorted"):
        order = PIPELINE_ORDER[pipeline]
        xs, chi2s = [], []
        for k in rows:
            b = out[(pipeline, order, int(k))]["bias"]
            if _is_good(b):
                xs.append(x_km_of_row[k]); chi2s.append(b["_chi2"])
        ax.semilogy(xs, chi2s, ".", ms=2, label=f"{pipeline} (order={order})",
                   color=PIPELINE_COLOR[pipeline], alpha=0.6)
    ax.set_title("chi2_reduced vs along-slit position")
    ax.set_xlabel("x [km]"); ax.set_ylabel("chi2 (log)")
    ax.legend(fontsize=8)

    # -- Panel: h2o bias vs true p_surface -- the degeneracy, directly --
    ax = axes[2, 1]
    for pipeline in ("native", "rectified", "undistorted"):
        order = PIPELINE_ORDER[pipeline]
        ps, biases = [], []
        for k in rows:
            b = out[(pipeline, order, int(k))]["bias"]
            if _is_good(b) and np.isfinite(b.get("h2o", np.nan)):
                ps.append(als.p_surface_hpa(x_km_of_row[k])); biases.append(b["h2o"])
        ax.plot(ps, biases, ".", ms=2, label=pipeline, color=PIPELINE_COLOR[pipeline], alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("h2o bias vs true surface pressure\n(the H2O/p_scale degeneracy, directly)")
    ax.set_xlabel("true p_surface [hPa]"); ax.set_ylabel("h2o bias [ppm]")
    ax.legend(fontsize=8)

    fig.suptitle(f"Along-slit composition/pressure stress test -- FPA{FPA}, {len(rows)} rows", fontsize=13)
    fig.tight_layout()
    out_path = REPO_ROOT / "plots" / f"gd_band_stress_test_fpa{FPA}{tag_suffix}.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
