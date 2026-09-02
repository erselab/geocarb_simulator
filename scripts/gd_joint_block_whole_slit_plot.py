#!/usr/bin/env python3
"""Stitch the 58 independent windows of the whole-slit joint-block sweep
(scripts/gd_joint_block_whole_slit_sweep.py) into slit-wide comparisons:
true/coarse/hires profile, retrieval bias, per-window fit quality, and the
adaptive window/G sizing itself, for EVERY state row saved in the sweep's
own snapshots (2026-08-31, was CO2-only) -- `co2_ppm`/`ch4_ppb`/`co_ppb`/
`h2o_surface_vmr`/`p_surface_hpa`/`albedo`, free or frozen. Every quantity
is reconstructed from the sweep's own saved StateSpec snapshots
(`w["coarse"]["params"][name]`/`w["hires"]["params"][name]`, already-
scaled physical values per window) -- no re-solving.

Reads `params[name]["values"]` directly rather than reconstructing via
`w["prior_co2_ppm_bins"] * w["x_hires"]` (fixed 2026-08-20): that legacy
top-level `prior_co2_ppm_bins` field is ALWAYS exact truth, kept only for
old back-compat plotters -- it does not reflect `--prior-fields`, so the
old reconstruction silently gave the wrong posterior for any non-"exact"
prior run (right by coincidence only when prior=truth, since `prior_co2_
ppm_bins` happens to equal the real prior in that one case). `gd_joint_
block_matrix.py`'s own `stitch()` was never affected -- it always read
`params[...]["values"]` this same way.

``--truth`` (2026-08-31) selects what "true" means for the comparison,
instead of always the raw continuous truth functions: `raw` (default,
today's exact behaviour, every existing call unaffected), `anchor`
(resolution-matched to a whole-slit ANCHOR grid,
`geocarb_gert.along_slit_scene.resolution_matched_fields`/
`resolution_matched_albedo_fn` sampled at `gd_build_resolution_matched_
truth.whole_slit_anchor_etas(fpa, --truth-anchor-density)`), or `bin`
(resolution-matched to the RETRIEVAL's own whole-slit BIN grid at a given
g_ratio, `whole_slit_bin_centers(fpa, --truth-g-ratio)`) -- the two
truth-image resolutions docs/PROJECT_STATUS.md Sec.10-12 use. Comparing a
resolution-matched-truth sweep's own posterior against the WRONG (raw)
truth silently overstates its error by exactly the representability gap
Sec.12 found and fixed -- this flag exists so that never happens by
accident.

``--rows`` (2026-08-31) selects which state rows to plot -- default ALL
rows present in the sweep's own saved snapshot (frozen rows included: a
frozen row's own profile/bias panel is a real, cheap check that it was
actually held at what the caller intended, e.g. Sec.11/12's frozen-
albedo-at-truth experiments).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_whole_slit_plot.py \\
        [results/gd_joint_block_whole_slit_fpa2_gratio1.pkl] [--truth raw|anchor|bin]
        [--truth-anchor-density 4] [--truth-g-ratio 1] [--rows co2_ppm,albedo]
Output: plots/joint_block/<input stem>.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_retrieve import FPA, _eta_of  # noqa: E402

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402

#: Display label per state row -- everything this script knows how to
#: plot. `als.STATE_FIELDS`'s own 5 atmosphere rows plus `albedo` (the one
#: `surface`-target row `state_spec_from_scene` ever adds).
ROW_LABELS = {
    "co2_ppm": "CO2 [ppm]", "ch4_ppb": "CH4 [ppb]", "co_ppb": "CO [ppb]",
    "h2o_surface_vmr": "H2O surface VMR", "p_surface_hpa": "p_surface [hPa]",
    "albedo": "albedo",
}


def _truth_fn_for(row_name: str, truth: str, match_x_km, band_label: str):
    """`fn(x_km) -> value` for one state row, honoring `--truth`."""
    if row_name == "albedo":
        if truth == "raw":
            return lambda x_km: als.albedo_for_label(x_km, band_label)  # noqa: E731
        return als.resolution_matched_albedo_fn(match_x_km, band_label)
    if truth == "raw":
        return als.STATE_FIELDS[row_name]
    return als.resolution_matched_fields(match_x_km)[row_name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=str, nargs="?",
                    default=str(REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}.pkl"),
                    help="path to a gd_joint_block_whole_slit_sweep.py (or _merge.py) output pickle")
    ap.add_argument("--truth", choices=["raw", "anchor", "bin"], default="raw",
                    help="'raw' (default): the raw continuous truth functions -- today's exact "
                         "behaviour. 'anchor': resolution-matched to a whole-slit ANCHOR grid "
                         "(--truth-anchor-density). 'bin': resolution-matched to a whole-slit "
                         "BIN grid (--truth-g-ratio) -- the zero-representability-gap reference "
                         "Sec.12's own ceiling test used. Must match whatever resolution the "
                         "sweep's OWN truth image was actually rendered at, or this plots a "
                         "real but misleading (representability-gap-inflated) bias.")
    ap.add_argument("--truth-anchor-density", type=int, default=4,
                    help="anchor_density for --truth anchor (default 4, matching Sec.10/11's "
                         "own convention)")
    ap.add_argument("--truth-g-ratio", type=float, default=1.0,
                    help="g_ratio for --truth bin (default 1.0, matching Sec.12's own ceiling "
                         "test -- should normally equal this sweep's OWN --g-ratio)")
    ap.add_argument("--rows", type=str, default=None,
                    help="comma-separated state rows to plot (default: every row present in "
                         f"the sweep's own saved snapshot). Known rows: {sorted(ROW_LABELS)}.")
    args = ap.parse_args()
    in_path = Path(args.input)

    with open(in_path, "rb") as f:
        d = pickle.load(f)
    results = d["results"]
    windows = sorted(results.values(), key=lambda r: r["row_lo"])
    fpa = d.get("fpa", FPA)
    band_label = GEOCARB_BANDS[fpa][0]

    row_names = ([r.strip() for r in args.rows.split(",")] if args.rows
                else list(windows[0]["hires"]["params"].keys()))

    if args.truth == "raw":
        match_x_km = None
        truth_desc = "raw continuous truth"
    else:
        from gd_build_resolution_matched_truth import whole_slit_anchor_etas, whole_slit_bin_centers
        # 2026-09-01 (bug fix, Sec.12.8): the tiling used to build the truth's
        # own anchor/bin grid MUST match the tiling THIS pkl's sweep actually
        # solved with, or the truth grid drifts from each window's own
        # row_lo/row_hi (concentrated at window boundaries) -- read straight
        # from the pkl's own saved metadata rather than re-deriving/assuming
        # defaults, since a sweep run with --overlap/--n-windows/--min-window
        # non-default silently tiles differently from build_window_tiles's
        # own bare defaults.
        tiling_kw = dict(min_window=d.get("min_window"), window_scale=d.get("window_scale", 1.0),
                         overlap=d.get("overlap", 0))
        if args.truth == "anchor":
            match_etas = whole_slit_anchor_etas(fpa, args.truth_anchor_density, **tiling_kw)
            truth_desc = f"resolution-matched truth (anchor_density={args.truth_anchor_density})"
        else:
            match_etas = whole_slit_bin_centers(fpa, args.truth_g_ratio, **tiling_kw)
            truth_desc = f"resolution-matched truth (g_ratio_bins={args.truth_g_ratio:g})"
        match_x_km = match_etas * als.SLIT_HALF_KM
    print(f"truth reference: {truth_desc}")
    print(f"rows: {row_names}")

    # --hires-only sweeps (--anchor-density / --state-interp variants) carry no
    # x_coarse/resid_coarse_rms at all, since the coarse solve is unaffected by
    # those flags and re-running it would just reproduce the baseline. Detect
    # that once here and drop the coarse curves rather than KeyError.
    has_coarse = all("x_coarse" in w for w in windows)
    if not has_coarse:
        print("no x_coarse in this pickle (hi-res-only sweep) -- plotting hi-res only")

    rows_all, width_all, G_all, resid_c_all, resid_h_all, row_mid_all = [], [], [], [], [], []
    per_row = {name: dict(true=[], coarse=[], hires=[], prior=[]) for name in row_names}
    truth_fns = {name: _truth_fn_for(name, args.truth, match_x_km, band_label) for name in row_names}

    for w in windows:
        row_lo, row_hi = w["row_lo"], w["row_hi"]
        rows_win = np.arange(row_lo, row_hi + 1)
        eta_win = _eta_of(fpa, np.full(len(rows_win), 512.0), rows_win.astype(float))
        x_km_win = eta_win * als.SLIT_HALF_KM

        for name in row_names:
            positions = np.asarray(w["hires"]["params"][name]["positions"])
            true_win = truth_fns[name](x_km_win)
            retrieved_coarse = (np.asarray(w["coarse"]["params"][name]["values"])
                                if has_coarse else None)
            retrieved_hires = np.asarray(w["hires"]["params"][name]["values"])
            prior_vals = np.asarray(w["hires"]["params"][name]["prior"])

            if len(positions) > 1:
                edges = 0.5 * (positions[:-1] + positions[1:])
                idx = np.searchsorted(edges, eta_win)
                coarse_win = retrieved_coarse[idx] if has_coarse else None
                hires_win = np.interp(eta_win, positions, retrieved_hires)
                prior_win = np.interp(eta_win, positions, prior_vals)
            else:
                coarse_win = (np.full(len(rows_win), retrieved_coarse[0])
                              if has_coarse else None)
                hires_win = np.full(len(rows_win), retrieved_hires[0])
                prior_win = np.full(len(rows_win), prior_vals[0])

            per_row[name]["true"].append(true_win)
            if has_coarse:
                per_row[name]["coarse"].append(coarse_win)
            per_row[name]["hires"].append(hires_win)
            per_row[name]["prior"].append(prior_win)

        rows_all.append(rows_win)
        width_all.append(w["width"])
        G_all.append(w["G"])
        if has_coarse:
            resid_c_all.append(w["resid_coarse_rms"])
        resid_h_all.append(w["resid_hires_rms"])
        row_mid_all.append(0.5 * (row_lo + row_hi))

    rows_all = np.concatenate(rows_all)
    for name in row_names:
        pr = per_row[name]
        pr["true"] = np.concatenate(pr["true"])
        pr["hires"] = np.concatenate(pr["hires"])
        pr["prior"] = np.concatenate(pr["prior"])
        pr["bias_hires"] = pr["hires"] - pr["true"]
        pr["bias_prior"] = pr["prior"] - pr["true"]
        if has_coarse:
            pr["coarse"] = np.concatenate(pr["coarse"])
            pr["bias_coarse"] = pr["coarse"] - pr["true"]

        bh, bp = pr["bias_hires"], pr["bias_prior"]
        print(f"\n[{name}] prior:  mean={bp.mean():+.4g} rms={np.sqrt(np.mean(bp**2)):.4g} "
             f"max|bias|={np.max(np.abs(bp)):.4g}")
        if has_coarse:
            bc = pr["bias_coarse"]
            print(f"[{name}] coarse: mean={bc.mean():+.4g} rms={np.sqrt(np.mean(bc**2)):.4g} "
                 f"max|bias|={np.max(np.abs(bc)):.4g}")
        print(f"[{name}] hires:  mean={bh.mean():+.4g} rms={np.sqrt(np.mean(bh**2)):.4g} "
             f"max|bias|={np.max(np.abs(bh)):.4g}  (worst at row {rows_all[np.argmax(np.abs(bh))]})")

    rc_fine = rows_crossed(fpa, rows_all.astype(float))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    n_rows = len(row_names)
    height_ratios = [2.2, 1.6] * n_rows + [1.0, 1.0]
    fig, axes = plt.subplots(2 * n_rows + 2, 1, figsize=(13, 3.6 * n_rows + 4), sharex=True,
                             gridspec_kw={"height_ratios": height_ratios})

    # Prior only gets its own trace/legend entry under an imperfect prior
    # (--prior-fields != "exact"). NOT a np.allclose(prior, true) check:
    # even under "exact" the prior trace is a PIECEWISE-LINEAR interpolation
    # of the state's own bin-center values (same as hires_win), which
    # differs from the continuous true_win by the already-documented
    # interpolation-representation-error floor (real, but a separate effect
    # from "prior is wrong" -- would almost always read as "differs" and
    # defeat the point of this check).
    prior_differs = d.get("prior_fields", "exact") != "exact"

    for i, name in enumerate(row_names):
        pr = per_row[name]
        label = ROW_LABELS.get(name, name)
        free = windows[0]["hires"]["params"][name].get("free", True)
        frozen_tag = "" if free else " [FROZEN]"

        ax = axes[2 * i]
        ax.plot(rows_all, pr["true"], color="black", lw=1.1, label="true", zorder=5)
        if prior_differs:
            ax.plot(rows_all, pr["prior"], color="0.5", lw=0.9, ls="--", label="prior", zorder=4)
        if has_coarse:
            ax.plot(rows_all, pr["coarse"], color="tab:orange", lw=0.9, alpha=0.85, label="coarse posterior")
        ax.plot(rows_all, pr["hires"], color="tab:blue", lw=0.9, alpha=0.85, label="hi-res posterior")
        ax.set_ylabel(label)
        title = f"true vs. retrieved {name}{frozen_tag}"
        if i == 0:
            title = (f"FPA{fpa} whole-slit joint block sweep, {len(windows)} independent "
                    f"windows ({truth_desc})\n{title}")
        ax.set_title(title, fontsize=11 if i else 12)
        ax.legend(fontsize=8, loc="upper right", markerscale=2)

        ax = axes[2 * i + 1]
        ax.axhline(0, color="black", lw=0.6)
        bh = pr["bias_hires"]
        if prior_differs:
            bp = pr["bias_prior"]
            ax.plot(rows_all, bp, color="0.5", lw=0.8, ls="--",
                   label=f"prior (rms={np.sqrt(np.mean(bp**2)):.3g}, max={np.max(np.abs(bp)):.3g})")
        if has_coarse:
            bc = pr["bias_coarse"]
            ax.plot(rows_all, bc, color="tab:orange", lw=0.8,
                   label=f"coarse (rms={np.sqrt(np.mean(bc**2)):.3g}, max={np.max(np.abs(bc)):.3g})")
        ax.plot(rows_all, bh, color="tab:blue", lw=0.8,
               label=f"hi-res (rms={np.sqrt(np.mean(bh**2)):.3g}, max={np.max(np.abs(bh)):.3g})")
        ax.set_ylabel("bias (-true)" if prior_differs else f"{name}\nbias (posterior-true)")
        ax.set_title(f"{name} bias across the whole slit", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[2 * n_rows]
    if has_coarse:
        ax.step(row_mid_all, resid_c_all, where="mid", color="tab:orange", lw=1.2, label="coarse")
    ax.step(row_mid_all, resid_h_all, where="mid", color="tab:blue", lw=1.2, label="hi-res")
    ax.set_yscale("log")
    ax.set_ylabel("per-window\nresid RMS")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("per-window residual RMS (fit quality)", fontsize=10.5)

    ax = axes[2 * n_rows + 1]
    ax2 = ax.twinx()
    ax.step(row_mid_all, width_all, where="mid", color="0.4", lw=1.2, label="window width [rows]")
    ax2.step(row_mid_all, G_all, where="mid", color="#A0631B", lw=1.2, label="G [bins]")
    ax.plot(rows_all, rc_fine * 4, color="0.75", lw=0.8, ls="--", label="rows_crossed x4 (scale ref.)")
    ax.set_ylabel("window width [rows]")
    ax2.set_ylabel("G [bins]", color="#A0631B")
    ax.set_xlabel("detector row")
    ax.set_title("adaptive window size and bin count (tied to local keystone)", fontsize=10.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    truth_suffix = ("" if args.truth == "raw" else
                    f"_truth-ad{args.truth_anchor_density}" if args.truth == "anchor" else
                    f"_truth-g{args.truth_g_ratio:g}bins")
    rows_suffix = "" if args.rows is None else "_" + "-".join(n.split("_")[0] for n in row_names)
    out_path = plots_dir / f"{in_path.stem}{truth_suffix}{rows_suffix}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
