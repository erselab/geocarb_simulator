#!/usr/bin/env python3
"""Compare the 58 independent windows of the whole-slit joint-block sweep
(scripts/gd_joint_block_whole_slit_sweep.py) against truth, for EVERY
state row saved in the sweep's own snapshots -- `co2_ppm`/`ch4_ppb`/
`co_ppb`/`h2o_surface_vmr`/`p_surface_hpa`/`albedo`, free or frozen.
Every quantity is reconstructed from the sweep's own saved StateSpec
snapshots (`w["hires"]["params"][name]`, already-scaled physical values
per window) -- no re-solving.

2026-09-02 (user): "the prior and posterior value should be compared to
the truth value AT THOSE POINTS. I don't want to see along-slit
interpolation in these results." Every retrieved quantity lives only at
its own state's discrete positions (bin centers); the truth reference is
now evaluated EXACTLY at those same positions, never interpolated onto
an intervening detector row. Each window is plotted as its own
disconnected '-o' segment (marker + line WITHIN a window, no line
bridging one window's last point to the next window's first) with grey
dashed vertical lines at window boundaries -- no whole-slit polyline,
which used to (a) require interpolating the state between bin centers
onto every detector row, and (b) for --truth bin specifically, silently
compare against a WHOLE-SLIT truth reference containing other windows'
own bin centers interleaved in eta-space with this window's (real, even
at overlap=0, since eta depends on column too -- keystone -- so row-
disjoint windows are not eta-disjoint) -- a scoring bug, not a real
retrieval error, found and fixed by removing interpolation entirely
rather than tracking down every place it could reappear.

Reads `params[name]["values"]` directly rather than reconstructing via
`w["prior_co2_ppm_bins"] * w["x_hires"]` (fixed 2026-08-20): that legacy
top-level `prior_co2_ppm_bins` field is ALWAYS exact truth, kept only for
old back-compat plotters -- it does not reflect `--prior-fields`, so the
old reconstruction silently gave the wrong posterior for any non-"exact"
prior run (right by coincidence only when prior=truth, since `prior_co2_
ppm_bins` happens to equal the real prior in that one case). `gd_joint_
block_matrix.py`'s own `stitch()` was never affected -- it always read
`params[...]["values"]` this same way.

``--truth`` selects what "true" means for the comparison: `raw`
(default) and `bin` are now IDENTICAL when evaluated only at bin centers
(the whole point of a `bin`-mode truth is that it equals the raw
continuous truth exactly there, by construction) -- `bin` is kept as an
explicit, self-documenting alias rather than removed. `anchor`
(resolution-matched to a whole-slit ANCHOR grid, genuinely different
from raw AT a bin center since bin centers generally fall between
anchor nodes) still needs the anchor-grid reconstruction -- but that one
IS a single, whole-slit-coherent function (no per-window locality issue:
`whole_slit_anchor_etas` truth images were rendered as one global scene,
unlike a `bin`-mode truth, which is built per-window).

``--rows`` selects which state rows to plot -- default ALL rows present
in the sweep's own saved snapshot (frozen rows included: a frozen row's
own profile/bias panel is a real, cheap check that it was actually held
at what the caller intended).

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_whole_slit_plot.py \\
        [results/gd_joint_block_whole_slit_fpa2_gratio1.pkl] [--truth raw|anchor|bin]
        [--truth-anchor-density 4] [--rows co2_ppm,albedo]
Output: figures/joint_block/<input stem>.png
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
    """`fn(x_km) -> value` for one state row, evaluated ONLY at the
    state's own discrete positions -- see module docstring for why `raw`
    and `bin` coincide there, and why `anchor` alone still needs the
    resolution-matched reconstruction."""
    if row_name == "albedo":
        if truth == "anchor":
            return als.resolution_matched_albedo_fn(match_x_km, band_label)
        return lambda x_km: als.albedo_for_label(x_km, band_label)  # noqa: E731
    if truth == "anchor":
        return als.resolution_matched_fields(match_x_km)[row_name]
    return als.STATE_FIELDS[row_name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=str, nargs="?",
                    default=str(REPO_ROOT / "results" / f"gd_joint_block_whole_slit_fpa{FPA}.pkl"),
                    help="path to a gd_joint_block_whole_slit_sweep.py (or _merge.py) output pickle")
    ap.add_argument("--truth", choices=["raw", "anchor", "bin"], default="raw",
                    help="'raw' (default) and 'bin': the raw continuous truth, evaluated exactly "
                         "at each bin center -- identical to each other there by construction. "
                         "'anchor': resolution-matched to a whole-slit ANCHOR grid "
                         "(--truth-anchor-density), genuinely different from raw at a bin center.")
    ap.add_argument("--truth-anchor-density", type=int, default=4,
                    help="anchor_density for --truth anchor (default 4, matching Sec.10/11's "
                         "own convention)")
    ap.add_argument("--rows", type=str, default=None,
                    help="comma-separated state rows to plot (default: every row present in "
                         f"the sweep's own saved snapshot). Known rows: {sorted(ROW_LABELS)}.")
    ap.add_argument("--truth-image", type=str, default=None,
                    help="path to a pickle holding the rendered whole-slit truth image this "
                         "sweep's y_true actually came from (e.g. scratch_work/"
                         "whole_slit_truth_ad4.pkl) -- accepts either a bare (1024,1024) array "
                         "or a dict with a 'representative'/'dense' key. Adds a continuum-SNR-"
                         "vs-row panel (real GeoCarb noise model, continuum = each row's own "
                         "brightest/least-absorbed column). Omitted by default since the sweep's "
                         "own output pickle doesn't carry the rendered image (only derived "
                         "state/residual quantities do).")
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

    if args.truth == "anchor":
        from gd_build_resolution_matched_truth import whole_slit_anchor_etas
        tiling_kw = dict(min_window=d.get("min_window"), window_scale=d.get("window_scale", 1.0),
                         overlap=d.get("overlap", 0))
        match_etas = whole_slit_anchor_etas(fpa, args.truth_anchor_density, **tiling_kw)
        match_x_km = match_etas * als.SLIT_HALF_KM
        truth_desc = f"resolution-matched truth (anchor_density={args.truth_anchor_density})"
    else:
        match_x_km = None
        truth_desc = "raw continuous truth (== bin-mode truth exactly at bin centers)"
    print(f"truth reference: {truth_desc}")
    print(f"rows: {row_names}")

    truth_fns = {name: _truth_fn_for(name, args.truth, match_x_km, band_label) for name in row_names}

    # η -> approximate detector row, for x-axis PLACEMENT only (never used
    # to reconstruct a value -- a nearest-row lookup, not an interpolation
    # of any physical quantity).
    row_lut = np.arange(1024.0)
    eta_lut = _eta_of(fpa, np.full(1024, 512.0), row_lut)
    order_lut = np.argsort(eta_lut)
    eta_lut_sorted, row_lut_sorted = eta_lut[order_lut], row_lut[order_lut]

    def eta_to_row(eta):
        idx = np.searchsorted(eta_lut_sorted, eta)
        idx = np.clip(idx, 0, len(eta_lut_sorted) - 1)
        return row_lut_sorted[idx]

    # --hires-only sweeps (--anchor-density / --state-interp variants) carry no
    # x_coarse/resid_coarse_rms at all. Detect that once here.
    has_coarse = all("x_coarse" in w for w in windows)
    if not has_coarse:
        print("no x_coarse in this pickle (hi-res-only sweep) -- plotting hi-res only")

    # Per window, per row: bin-center rows (x-axis) and true/prior/hires
    # VALUES AT THOSE SAME BIN CENTERS -- no interpolation anywhere.
    per_row = {name: [] for name in row_names}  # each entry: one window's own dict
    resid_c_all, resid_h_all, width_all, G_all, row_mid_all, boundary_rows = [], [], [], [], [], []

    for w in windows:
        row_lo, row_hi = w["row_lo"], w["row_hi"]
        for name in row_names:
            p = w["hires"]["params"][name]
            positions = np.asarray(p["positions"])
            x_km = positions * als.SLIT_HALF_KM
            true_vals = np.asarray(truth_fns[name](x_km), dtype=float)
            hires_vals = np.asarray(p["values"], dtype=float)
            prior_vals = np.asarray(p["prior"], dtype=float)
            coarse_vals = (np.asarray(w["coarse"]["params"][name]["values"], dtype=float)
                          if has_coarse else None)
            per_row[name].append(dict(
                rows=eta_to_row(positions), true=true_vals, hires=hires_vals,
                prior=prior_vals, coarse=coarse_vals))
        width_all.append(w["width"])
        G_all.append(w["G"])
        if has_coarse:
            resid_c_all.append(w["resid_coarse_rms"])
        resid_h_all.append(w["resid_hires_rms"])
        row_mid_all.append(0.5 * (row_lo + row_hi))
        boundary_rows.append(row_lo - 0.5)
    boundary_rows.append(windows[-1]["row_hi"] + 0.5)

    all_rows_seen = np.concatenate([wd["rows"] for name in row_names for wd in per_row[name]])
    rc_fine = rows_crossed(fpa, np.arange(all_rows_seen.min(), all_rows_seen.max() + 1, dtype=float))
    rc_rows = np.arange(all_rows_seen.min(), all_rows_seen.max() + 1)

    # Whole-run summary stats (concatenating every window's own bin-center
    # points -- a plain array op, not a reconstruction).
    prior_differs = d.get("prior_fields", "exact") != "exact"
    for name in row_names:
        wds = per_row[name]
        bh = np.concatenate([wd["hires"] - wd["true"] for wd in wds])
        bp = np.concatenate([wd["prior"] - wd["true"] for wd in wds])
        rows_cat = np.concatenate([wd["rows"] for wd in wds])
        print(f"\n[{name}] prior:  mean={bp.mean():+.4g} rms={np.sqrt(np.mean(bp**2)):.4g} "
             f"max|bias|={np.max(np.abs(bp)):.4g}")
        if has_coarse:
            bc = np.concatenate([wd["coarse"] - wd["true"] for wd in wds])
            print(f"[{name}] coarse: mean={bc.mean():+.4g} rms={np.sqrt(np.mean(bc**2)):.4g} "
                 f"max|bias|={np.max(np.abs(bc)):.4g}")
        print(f"[{name}] hires:  mean={bh.mean():+.4g} rms={np.sqrt(np.mean(bh**2)):.4g} "
             f"max|bias|={np.max(np.abs(bh)):.4g}  (worst near row {rows_cat[np.argmax(np.abs(bh))]})")

    # Continuum SNR vs. row (== eta, monotonically -- same x-axis every other
    # panel already uses): the sweep's own output pickle only carries derived
    # state/residual quantities, never the rendered detector image itself, so
    # this needs the actual truth image the run's y_true came from, passed in
    # explicitly via --truth-image. "Continuum" here is each row's own
    # brightest (least line-absorbed) column -- an honest, deterministic
    # proxy given these are noise-free synthetic truth images (real per-pixel
    # noise, if ever added to the image, would need a true continuum-region
    # mask instead of a raw max).
    snr_row = None
    if args.truth_image is not None:
        with open(args.truth_image, "rb") as f:
            truth_obj = pickle.load(f)
        if isinstance(truth_obj, dict):
            A_img = truth_obj.get("representative", truth_obj.get("dense"))
            if A_img is None:
                A_img = next(v for v in truth_obj.values() if isinstance(v, np.ndarray))
        else:
            A_img = truth_obj
        A_img = np.asarray(A_img, dtype=float)
        from geocarb_gert.radiometry import geocarb_noise_model
        noise_model = geocarb_noise_model(fpa)
        continuum = np.max(A_img, axis=1)
        sigma_continuum = np.sqrt(noise_model.N0 ** 2 + noise_model.N1 * np.abs(continuum))
        snr_row = continuum / np.maximum(sigma_continuum, 1e-30)
        snr_rows_axis = np.arange(A_img.shape[0])
        print(f"continuum SNR: min={snr_row.min():.4g} median={np.median(snr_row):.4g} "
             f"max={snr_row.max():.4g}  (from {args.truth_image})")

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8})

    n_rows = len(row_names)
    n_extra = 2 + (1 if snr_row is not None else 0)
    height_ratios = [2.2, 1.6] * n_rows + [1.0, 1.0] + ([1.0] if snr_row is not None else [])
    fig, axes = plt.subplots(2 * n_rows + n_extra, 1, figsize=(13, 3.6 * n_rows + 4 + (1.6 if snr_row is not None else 0)),
                             sharex=True, gridspec_kw={"height_ratios": height_ratios})

    def _boundaries(ax):
        for b in boundary_rows:
            ax.axvline(b, color="0.75", lw=0.6, ls="--", zorder=0)

    ms = 2.5
    for i, name in enumerate(row_names):
        wds = per_row[name]
        label = ROW_LABELS.get(name, name)
        free = windows[0]["hires"]["params"][name].get("free", True)
        frozen_tag = "" if free else " [FROZEN]"

        ax = axes[2 * i]
        _boundaries(ax)
        for j, wd in enumerate(wds):
            kw = dict(label=None if j else "true")
            ax.plot(wd["rows"], wd["true"], "-o", color="black", lw=1.1, ms=ms, zorder=5, **kw)
            if prior_differs:
                ax.plot(wd["rows"], wd["prior"], "-o", color="0.5", lw=0.9, ms=ms, ls="--", zorder=4,
                        label=None if j else "prior")
            if has_coarse:
                ax.plot(wd["rows"], wd["coarse"], "-o", color="tab:orange", lw=0.9, ms=ms, alpha=0.85,
                        label=None if j else "coarse posterior")
            ax.plot(wd["rows"], wd["hires"], "-o", color="tab:blue", lw=0.9, ms=ms, alpha=0.85,
                    label=None if j else "hi-res posterior")
        ax.set_ylabel(label)
        title = f"true vs. retrieved {name}{frozen_tag} -- per window, no along-slit interpolation"
        if i == 0:
            title = (f"FPA{fpa} whole-slit joint block sweep, {len(windows)} independent "
                    f"windows ({truth_desc})\n{title}")
        ax.set_title(title, fontsize=11 if i else 12)
        ax.legend(fontsize=8, loc="upper right", markerscale=2)

        ax = axes[2 * i + 1]
        _boundaries(ax)
        ax.axhline(0, color="black", lw=0.6)
        bh_all = np.concatenate([wd["hires"] - wd["true"] for wd in wds])
        bp_all = np.concatenate([wd["prior"] - wd["true"] for wd in wds])
        for j, wd in enumerate(wds):
            bh_j = wd["hires"] - wd["true"]
            if prior_differs:
                bp_j = wd["prior"] - wd["true"]
                ax.plot(wd["rows"], bp_j, "-o", color="0.5", lw=0.8, ms=ms, ls="--",
                        label=(None if j else f"prior (rms={np.sqrt(np.mean(bp_all**2)):.3g}, "
                                             f"max={np.max(np.abs(bp_all)):.3g})"))
            if has_coarse:
                bc_j = wd["coarse"] - wd["true"]
                bc_all = np.concatenate([wd2["coarse"] - wd2["true"] for wd2 in wds])
                ax.plot(wd["rows"], bc_j, "-o", color="tab:orange", lw=0.8, ms=ms,
                        label=(None if j else f"coarse (rms={np.sqrt(np.mean(bc_all**2)):.3g}, "
                                             f"max={np.max(np.abs(bc_all)):.3g})"))
            ax.plot(wd["rows"], bh_j, "-o", color="tab:blue", lw=0.8, ms=ms,
                    label=(None if j else f"hi-res (rms={np.sqrt(np.mean(bh_all**2)):.3g}, "
                                         f"max={np.max(np.abs(bh_all)):.3g})"))
        ax.set_ylabel("bias (-true)" if prior_differs else f"{name}\nbias (posterior-true)")
        ax.set_title(f"{name} bias, per window (dashed grey = window boundary)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[2 * n_rows]
    _boundaries(ax)
    if has_coarse:
        ax.step(row_mid_all, resid_c_all, where="mid", color="tab:orange", lw=1.2, label="coarse")
    ax.step(row_mid_all, resid_h_all, where="mid", color="tab:blue", lw=1.2, label="hi-res")
    ax.set_yscale("log")
    ax.set_ylabel("per-window\nresid RMS")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("per-window residual RMS (fit quality)", fontsize=10.5)

    ax = axes[2 * n_rows + 1]
    _boundaries(ax)
    ax2 = ax.twinx()
    ax.step(row_mid_all, width_all, where="mid", color="0.4", lw=1.2, label="window width [rows]")
    ax2.step(row_mid_all, G_all, where="mid", color="#A0631B", lw=1.2, label="G [bins]")
    ax.plot(rc_rows, rc_fine * 4, color="0.75", lw=0.8, ls="--", label="rows_crossed x4 (scale ref.)")
    ax.set_ylabel("window width [rows]")
    ax2.set_ylabel("G [bins]", color="#A0631B")
    if snr_row is None:
        ax.set_xlabel("detector row")
    ax.set_title("adaptive window size and bin count (tied to local keystone)", fontsize=10.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    if snr_row is not None:
        ax = axes[2 * n_rows + 2]
        _boundaries(ax)
        ax.step(snr_rows_axis, snr_row, where="mid", color="tab:green", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("continuum SNR")
        ax.set_xlabel("detector row")
        ax.set_title("continuum SNR vs. along-slit position (real GeoCarb noise model; "
                     "continuum = each row's own brightest column)", fontsize=10.5)

    fig.tight_layout()
    plots_dir = REPO_ROOT / "figures" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    truth_suffix = "" if args.truth != "anchor" else f"_truth-ad{args.truth_anchor_density}"
    rows_suffix = "" if args.rows is None else "_" + "-".join(n.split("_")[0] for n in row_names)
    out_path = plots_dir / f"{in_path.stem}{truth_suffix}{rows_suffix}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
