#!/usr/bin/env python3
"""Compare the hi-res forward-model variants against the baseline, on the
two quantities they turn out to affect independently: the post-fit RESIDUAL
and the retrieved CO2 BIAS.

The variants (`gd_joint_block_whole_slit_sweep.py`):

  baseline          anchors one per detector row, each carrying the exact
                    truth atmosphere at its own eta, pixels assigned
                    NEAREST-anchor.
  --anchor-density N  same construction, anchors at 1/N-row eta spacing.
                    Attacks the nearest-anchor STEP error directly: that
                    error is first order in anchor spacing h, ~(1/4)|f'|h,
                    so quartering h should quarter the residual.
  --state-interp    anchors unchanged in spacing, but each one's atmosphere
                    is built from state parameters linearly interpolated
                    from the G bin centres (every parameter through the
                    identical path -- `als.interp_state` over
                    `als.STATE_FIELDS` -- CO2 included, no special case),
                    then a fresh RT run. Changes WHAT the anchors carry, not
                    how finely they are sampled.

Measured result (FPA2 realistic, hi-res): the two are ORTHOGONAL.
`--anchor-density 4` cuts the residual 4.13x/4.16x (gratio1/gratio3),
matching the predicted 1/h scaling, but barely moves the bias.
`--state-interp` cuts the bias (33% at gratio3) but barely moves the
residual. Neither alone does both; combining them is untested.

This matters because the residual's large-scale structure was never CO2 or
keystone: it tracks the NUISANCE state's own nearest-anchor representation
error (p_surface +0.94 across the topographic depression at rows 329-527,
H2O +0.89 globally, versus +0.59 for CO2 and +0.38 for `rows_crossed`), and
those quantities are held at local truth rather than retrieved -- so the
error is pure forward-model representation, not knowledge. See
`gd_joint_block_residual_vs_scene.py`.

Run:  PYTHONPATH=. python3 scripts/gd_joint_block_variant_compare.py --gratio 1
      PYTHONPATH=. python3 scripts/gd_joint_block_variant_compare.py --gratio 3 \\
        --results-dir /path/to/pickles
Output: plots/joint_block/gd_joint_block_variant_compare_fpa<F>_gratio<G>.png
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

N_COLS = 1024
VARIANTS = [
    ("", "baseline (nearest, 1 anchor/row)", "0.25", "-"),
    ("_stateinterp", "--state-interp", "tab:green", "-"),
    ("_adens4", "--anchor-density 4", "tab:blue", "-"),
]
PLOT_STYLE = {
    "font.family": "serif", "font.size": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8,
}


def load(path: Path, eta: np.ndarray, true_row: np.ndarray):
    """Per-row residual RMS and per-row CO2 bias from one sweep pickle."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    rr = np.full(1024, np.nan)
    bias = np.full(1024, np.nan)
    for w in d["results"].values():
        if "resid_hires" not in w:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        blk = np.asarray(w["resid_hires"], dtype=float).reshape(hi - lo + 1, N_COLS)
        rr[lo:hi + 1] = np.sqrt(np.mean(blk ** 2, axis=1))
        bc = np.asarray(w["bin_centers"], dtype=float)
        ppm = w["prior_co2_ppm_bins"] * w["x_hires"]
        bias[lo:hi + 1] = np.interp(eta[lo:hi + 1], bc, ppm) - true_row[lo:hi + 1]
    return rr, bias


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpa", type=int, default=2)
    ap.add_argument("--gratio", type=str, default="1")
    ap.add_argument("--results-dir", type=str, default=None,
                    help="where the sweep pickles live (default: the repo's results/, "
                         "falling back to ~/geocarb_simulator_results for the baselines)")
    args = ap.parse_args()

    rows = np.arange(1024.0)
    _, s = xy_to_wavelength_slit(args.fpa, np.full(1024, 512.0), rows)
    eta = s / s_max(args.fpa)
    true_row = als.xco2_ppm(eta * als.SLIT_HALF_KM)
    psurf = als.p_surface_hpa(eta * als.SLIT_HALF_KM)
    row_bottom = int(np.argmin(psurf))

    stem = f"gd_joint_block_whole_slit_fpa{args.fpa}_gratio{args.gratio}"
    search = [Path(args.results_dir)] if args.results_dir else [
        REPO_ROOT / "results", Path.home() / "geocarb_simulator_results"]

    data = {}
    for suffix, label, color, ls in VARIANTS:
        found = None
        for d in search:
            p = d / f"{stem}{suffix}.pkl"
            if p.exists():
                found = p
                break
        if found is None:
            print(f"  MISSING: {stem}{suffix}.pkl (looked in {[str(x) for x in search]})")
            continue
        rr, bias = load(found, eta, true_row)
        data[label] = (rr, bias, color, ls)
        print(f"{label:34s} residRMS={np.sqrt(np.nanmean(rr**2)):.4e}  "
              f"biasRMS={np.sqrt(np.nanmean(bias**2)):.4f} ppm  "
              f"max|bias|={np.nanmax(np.abs(bias)):.4f}  [{found.name}]")
    if not data:
        print("nothing to plot")
        return 1

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.35, 1.0]})

    ax = axes[0]
    for label, (rr, _, color, ls) in data.items():
        ax.semilogy(rows, rr, lw=0.9, color=color, ls=ls, label=label)
    ax.set_ylabel("per-row residual RMS\n[W m$^{-2}$ sr$^{-1}$ $\\mu$m$^{-1}$]")
    ax.set_title(f"FPA{args.fpa} gratio{args.gratio}: anchor spacing sets the RESIDUAL, "
                 f"bin-grid interpolation sets the BIAS", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    ax.axhline(0, color="black", lw=0.6)
    for label, (_, bias, color, ls) in data.items():
        ax.plot(rows, bias, lw=0.8, color=color, ls=ls,
                label=f"{label} (rms={np.sqrt(np.nanmean(bias**2)):.4f} ppm)")
    ax.set_ylabel("CO2 bias\n(posterior - true) [ppm]")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)

    # Why the residual has the shape it does, on the same row axis.
    ax = axes[2]
    ax.plot(rows, psurf, lw=1.1, color="tab:purple")
    ax.axvline(row_bottom, color="crimson", lw=1.0, ls="--")
    ax.axvspan(329, 527, color="crimson", alpha=0.06)
    ax.annotate(f"depression bottom, row {row_bottom}\n(dp/drow = 0 -> residual null)",
                xy=(row_bottom, psurf[row_bottom]),
                xytext=(row_bottom + 110, psurf[row_bottom] + 70), fontsize=8.5,
                color="crimson", arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ax.set_ylabel("true p_surface\n[hPa]")
    ax.set_xlabel("detector row")
    ax.set_title("the nuisance quantity the residual actually tracks "
                 "(held at local truth, never retrieved)", fontsize=10.5)

    fig.tight_layout()
    out_dir = REPO_ROOT / "plots" / "joint_block"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gd_joint_block_variant_compare_fpa{args.fpa}_gratio{args.gratio}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
