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

Chi2-outlier filtering (added 2026-07-25): `gert`'s own divergence check
only catches non-finite (NaN/Inf) chi2 -- a Gauss-Newton step that
overshoots into an unphysical state can still leave chi2 astronomically
large but technically *finite*, in which case `dx_norm` (which measures
step size, not chi2 validity) can register "converged=True" over a garbage
state. Found in FPA2's uniform-mode rectified run: row 59 converged with
chi2~1.65e206 and a co2 bias of -357 ppm (population otherwise -35 to -60
ppm) -- one row that on its own inflated that pipeline's reported bias std
by ~3x. Rare (1 in ~73,000 row/pipeline/run combinations checked across all
8 real runs so far) but real, and silent -- every panel and every summary
statistic below now excludes rows flagged by `_chi2_outlier_mask()`
(robust, MAD-based, computed per pipeline's own converged population) and
counts them explicitly as a distinct failure mode, and `main()` prints a
median/robust-std summary table that isn't sensitive to a single such row
the way mean/std is. This is deliberately handled here in the analysis
layer rather than in `gert` itself -- `gert` is shared, foundational code;
this script owns interpreting its output.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_band_stress_test_plot.py --fpa 1
Output: plots/gd_band_stress_test_fpa<N>.png, plus a printed robust-stats table
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


def _chi2_outlier_mask(chi2: np.ndarray) -> np.ndarray:
    """True where chi2 is 'out of family' relative to the rest of a
    converged population -- a robust (MAD-based) threshold in log10 space,
    8 robust-sigma out. See module docstring for why this check exists.
    """
    chi2 = np.asarray(chi2, dtype=float)
    if len(chi2) == 0:
        return np.zeros(0, dtype=bool)
    log_chi2 = np.log10(np.maximum(chi2, 1e-300))
    med = np.median(log_chi2)
    mad = np.median(np.abs(log_chi2 - med))
    if mad < 1e-12:
        # Population too tight for MAD to be informative -- fall back to an
        # absolute ceiling well above any legitimate reduced-chi2 seen in
        # this study (good fits ~1e-4-1, even bad rectified fits ~1-5).
        return chi2 > 1e6
    return log_chi2 > med + 8.0 * 1.4826 * mad


def _robust_stats(x) -> tuple[float, float]:
    """(median, robust_std) -- robust_std = 1.4826*MAD, a normal-equivalent
    scale estimate that isn't dragged around by a single extreme outlier
    the way mean/std are. NaN-safe only in the sense that callers should
    pass finite values already (see _bias_series)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med, 1.4826 * mad


def _in_family_rows(out, rows, pipeline: str):
    """(ks, biases, chi2s, outlier_mask) for every converged row of one
    pipeline -- outlier_mask flags rows whose chi2 is out-of-family despite
    being reported converged=True (see _chi2_outlier_mask)."""
    order = PIPELINE_ORDER[pipeline]
    ks, biases, chi2s = [], [], []
    for k in rows:
        b = out[(pipeline, order, int(k))]["bias"]
        if _is_good(b):
            ks.append(int(k)); biases.append(b); chi2s.append(b["_chi2"])
    chi2s = np.array(chi2s, dtype=float)
    outlier = _chi2_outlier_mask(chi2s)
    return np.array(ks), biases, chi2s, outlier


def _bias_series(out, rows, x_km_of_row, pipeline: str, gas: str):
    ks, biases, _, outlier = _in_family_rows(out, rows, pipeline)
    xs, vals = [], []
    for k, b, is_out in zip(ks, biases, outlier):
        if is_out:
            continue
        v = b.get(gas, np.nan)
        if np.isfinite(v):
            xs.append(x_km_of_row[k]); vals.append(v)
    return np.array(xs), np.array(vals)


def _state_series(out, rows, x_km_of_row, pipeline: str, key: str):
    """Same as _bias_series but reads from b['_state'] (T_offset, albedo,
    dispersion coefficients -- see gd_band_stress_test.py's _retrieve()) --
    retrieved-minus-prior, not retrieved-minus-truth, since these elements
    have no along-slit truth to diff against. Returns empty arrays if this
    pipeline/order never had `key` (e.g. dispersion keys for undistorted,
    order=0) or if the results predate the 2026-07-25 _state capture."""
    ks, biases, _, outlier = _in_family_rows(out, rows, pipeline)
    xs, vals = [], []
    for k, b, is_out in zip(ks, biases, outlier):
        if is_out:
            continue
        v = b.get("_state", {}).get(key, np.nan)
        if np.isfinite(v):
            xs.append(x_km_of_row[k]); vals.append(v)
    return np.array(xs), np.array(vals)


# Nuisance state-vector elements to plot, beyond the gases/p_surface already
# in the main figure -- (key, ylabel, [pipelines expected to have it]).
# Dispersion keys only exist for order>0 (native/rectified); undistorted
# (order=0) never includes them (see gd_band_stress_test.py's
# pipeline_orders). albedo_slope_0 has an extremely tight prior (1-sigma
# 1e-10 in StateVector.gas_scaling's defaults) so it's expected to sit at
# ~0 regardless of row -- included for completeness, not because it's
# expected to show structure.
STATE_PANELS = [
    ("T_offset", "T_offset bias vs prior [K]", ("native", "rectified", "undistorted")),
    ("albedo_0", "albedo_0 bias vs prior", ("native", "rectified", "undistorted")),
    ("albedo_slope_0", "albedo_slope_0 bias vs prior\n(tight prior, ~0 expected)", ("native", "rectified", "undistorted")),
    ("disp_a0_0", "disp_a0_0 (shift) [cm-1]", ("native", "rectified")),
    ("disp_a1_0", "disp_a1_0 (stretch) [cm-1]", ("native", "rectified")),
    ("disp_a2_0", "disp_a2_0 (quadratic) [cm-1]", ("native", "rectified")),
]


def _plot_state_diagnostics(out, rows, x_km_of_row, FPA, out_path) -> bool:
    """Every other state-vector element (T_offset, albedo, dispersion
    coefficients) plotted vs along-slit position, retrieved-minus-prior.
    Returns False (and skips writing) if none of these results have a
    '_state' entry -- i.e. this .pkl predates the 2026-07-25 capture and
    needs a rerun of gd_band_stress_test.py to show anything here."""
    has_state = any(
        out[(pl, PIPELINE_ORDER[pl], int(rows[0]))]["bias"] is not None
        and "_state" in out[(pl, PIPELINE_ORDER[pl], int(rows[0]))]["bias"]
        for pl in ("native", "rectified", "undistorted")
    )
    if not has_state:
        print(f"  (no '_state' data in this .pkl -- rerun gd_band_stress_test.py "
              f"to get {out_path.name})")
        return False

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for ax, (key, ylabel, pipelines) in zip(axes.ravel(), STATE_PANELS):
        any_data = False
        for pipeline in pipelines:
            xs, vals = _state_series(out, rows, x_km_of_row, pipeline, key)
            if len(xs) == 0:
                continue
            any_data = True
            ax.plot(xs, vals, ".", ms=2, label=pipeline, color=PIPELINE_COLOR[pipeline], alpha=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(key)
        ax.set_xlabel("x [km]"); ax.set_ylabel(ylabel)
        if any_data:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")

    fig.suptitle(f"State-vector nuisance parameters (retrieved - prior) -- FPA{FPA}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return True


def _print_robust_summary(out, rows, gases, primary_gas: str) -> None:
    report_gases = list(dict.fromkeys([primary_gas] + [g for g in gases if g != primary_gas] + ["p_surface"]))
    print(f"\n{'pipeline':12s} {'gas':10s} {'n_good':>7s} {'n_outlier':>9s}   "
          f"{'median':>10s} {'robust_std':>10s}   {'mean':>10s} {'std':>10s}")
    for pipeline in ("native", "rectified", "undistorted"):
        ks, biases, chi2s, outlier = _in_family_rows(out, rows, pipeline)
        n_outlier = int(outlier.sum())
        for gas in report_gases:
            vals = np.array([b.get(gas, np.nan) for b, is_out in zip(biases, outlier) if not is_out])
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            med, rstd = _robust_stats(vals)
            print(f"{pipeline:12s} {gas:10s} {len(vals):7d} {n_outlier:9d}   "
                  f"{med:10.3f} {rstd:10.3f}   {vals.mean():10.3f} {vals.std():10.3f}")


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
    for pipeline, color, yoff in (("native", "tab:blue", 0.45), ("rectified", "tab:orange", 0.3),
                                  ("undistorted", "tab:green", 0.15)):
        order = PIPELINE_ORDER[pipeline]
        xs = [x_km_of_row[k] for k in rows if (b := out[(pipeline, order, int(k))]["bias"]) is not None and b.get("_diverged")]
        ax.plot(xs, [yoff] * len(xs), "x", color=color, ms=4, label=f"{pipeline} diverged (n={len(xs)})")
    for pipeline, color, yoff in (("native", "tab:brown", -0.15), ("rectified", "tab:pink", -0.3)):
        order = PIPELINE_ORDER[pipeline]
        xs = [x_km_of_row[k] for k in rows if (b := out[(pipeline, order, int(k))]["bias"]) is not None
             and not b.get("_diverged") and not b.get("_conv")]
        ax.plot(xs, [yoff] * len(xs), "+", color=color, ms=5, label=f"{pipeline} stalled (n={len(xs)})")
    for pipeline, color in (("native", "tab:blue"), ("rectified", "tab:orange"), ("undistorted", "tab:green")):
        ks, _, _, outlier = _in_family_rows(out, rows, pipeline)
        xs = [x_km_of_row[k] for k, is_out in zip(ks, outlier) if is_out]
        if xs:
            ax.plot(xs, [0.0] * len(xs), "*", color=color, ms=7,
                   label=f"{pipeline} chi2-outlier (n={len(xs)})")
    offdet_xs = [x_km_of_row[k] for k in rows if out[("rectified", 2, int(k))]["bias"] is None]
    ax.plot(offdet_xs, [-0.45] * len(offdet_xs), "o", color="gray", ms=3, label=f"rectified off-detector (n={len(offdet_xs)})")
    ax.set_title("Where each pipeline fails, vs along-slit position")
    ax.set_xlabel("x [km]"); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper center", ncol=2)

    # -- Panel: chi2 vs x_km, each pipeline at its own order --
    ax = axes[2, 0]
    for pipeline in ("native", "rectified", "undistorted"):
        ks, _, chi2s, outlier = _in_family_rows(out, rows, pipeline)
        xs = [x_km_of_row[k] for k, is_out in zip(ks, outlier) if not is_out]
        vals = chi2s[~outlier]
        ax.semilogy(xs, vals, ".", ms=2, label=f"{pipeline} (order={PIPELINE_ORDER[pipeline]})",
                   color=PIPELINE_COLOR[pipeline], alpha=0.6)
    ax.set_title("chi2_reduced vs along-slit position\n(chi2-outlier rows excluded, see legend counts above)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("chi2 (log)")
    ax.legend(fontsize=8)

    # -- Panel: h2o bias vs true p_surface -- the degeneracy, directly --
    ax = axes[2, 1]
    for pipeline in ("native", "rectified", "undistorted"):
        xs, biases = _bias_series(out, rows, x_km_of_row, pipeline, "h2o")
        ps = als.p_surface_hpa(xs)
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

    state_path = REPO_ROOT / "plots" / f"gd_band_stress_test_fpa{FPA}{tag_suffix}_state.png"
    _plot_state_diagnostics(out, rows, x_km_of_row, FPA, state_path)

    _print_robust_summary(out, rows, gases, primary_gas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
