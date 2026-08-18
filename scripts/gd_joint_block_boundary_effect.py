#!/usr/bin/env python3
"""Quantifies whether independent-window boundaries leave a residual
signature, rather than relying on eyeballing the full-detector image
(`gd_joint_block_residual_plot.py`) for it.

The confound that makes a naive check wrong
--------------------------------------------
Per-row residual RMS varies smoothly across the whole slit anyway (mostly
tracking the p_surface gradient -- see `gd_joint_block_residual_vs_scene.py`).
Comparing "rows near a boundary" against "rows far from a boundary" without
removing that ambient trend first would just re-measure the p_surface
gradient's own shape, not a boundary effect. Fix: normalize every row's
residual RMS by ITS OWN WINDOW's median first. That cancels the slow
between-window trend and isolates whether rows near an edge are
systematically high or low relative to their own window's typical row.

What "edge" means here, and what it would mean if real
--------------------------------------------------------
Each window is solved with its own independent regularization block --
`prior_form="exponential"` (all configs checked here use it) has NO edge
weakening in the PRIOR covariance itself (see `joint_state.py`'s own
`Sa_inv_block` docstring: unlike `tikhonov`, exponential's inverted
stationary covariance has compensating corner terms, not a weaker one).
So a real edge effect found here would point at something else: windows
share no information across their boundary at all, even though the true
field is continuous through it -- each window's own edge bins are pulled
only toward that window's own local prior, with nothing enforcing
consistency with the neighboring window's edge value.

Ruled out by construction, not left as an open confound: PSF-blur edge
handling. `predict_neighborhood` renders `PAD=4` extra rows on each side of
the requested window specifically so `gaussian_blur_rows`'s kernel (reach
`ceil(4*sigma)` ~ 3 rows at the default `spatial_psf_fwhm_px=1.5`) has
enough margin before the padding gets discarded -- the retained rows' own
blur should not depend on being near the window's edge. If an effect shows
up here anyway, it is not the render step.

Run:  PYTHONPATH=. python3 scripts/gd_joint_block_boundary_effect.py
      PYTHONPATH=. python3 scripts/gd_joint_block_boundary_effect.py \\
        analytic_jacobian_testing/results/co2p-58-g1_analytic.pkl \\
        analytic_jacobian_testing/results/co2p-29-g1_analytic.pkl --solve hires
Output: plots/joint_block/gd_joint_block_boundary_effect.png
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
DEFAULT_INPUTS = [
    REPO_ROOT / "analytic_jacobian_testing" / "results" / "co2p-58-g1_analytic.pkl",
    REPO_ROOT / "analytic_jacobian_testing" / "results" / "co2p-29-g1_analytic.pkl",
]
N_COLS = 1024
MAX_DIST = 8
PLOT_STYLE = {
    "font.family": "serif", "font.size": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8,
}
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def per_window_rows(path: Path, solve: str):
    """[(row_lo, row_hi, width, per_row_rms array of length width)], one
    per window that has a saved `resid_<solve>`."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = []
    for w in sorted(d["results"].values(), key=lambda r: r["row_lo"]):
        key = f"resid_{solve}"
        if key not in w:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        width = hi - lo + 1
        blk = np.asarray(w[key], dtype=float).reshape(width, N_COLS)
        rr = np.sqrt(np.mean(blk ** 2, axis=1))
        out.append((lo, hi, width, rr))
    return out, d


def edge_distance_dataset(windows):
    """Long-form arrays: dist[i] = rows from this row to its window's
    nearest edge (0 = the edge row itself); normalized[i] = that row's
    residual RMS divided by its OWN window's median residual RMS."""
    dists, normed, widths = [], [], []
    for lo, hi, width, rr in windows:
        med = np.median(rr)
        if not np.isfinite(med) or med <= 0:
            continue
        d = np.minimum(np.arange(width), width - 1 - np.arange(width))
        dists.append(d)
        normed.append(rr / med)
        widths.append(np.full(width, width))
    return (np.concatenate(dists), np.concatenate(normed), np.concatenate(widths))


def edge_interior_ratio(windows, edge_n=1, interior_margin=2):
    """One ratio per window wide enough to have a real interior:
    mean residual RMS of the `edge_n` outermost rows on each side, over the
    median residual RMS of everything at least `interior_margin` rows in."""
    ratios, widths = [], []
    for lo, hi, width, rr in windows:
        if width < 2 * (interior_margin + 1) + 1:
            continue
        edge = np.concatenate([rr[:edge_n], rr[-edge_n:]])
        interior = rr[interior_margin:width - interior_margin]
        interior_med = np.median(interior)
        if not np.isfinite(interior_med) or interior_med <= 0:
            continue
        ratios.append(np.mean(edge) / interior_med)
        widths.append(width)
    return np.asarray(ratios), np.asarray(widths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS,
                    help="one or more sweep pickles to compare (default: the "
                         "58- vs 29-window co2p analytic runs)")
    ap.add_argument("--solve", default="hires", choices=["coarse", "hires"])
    args = ap.parse_args()

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    ax_main, ax_ratio = axes[0]
    ax_n, ax_ratio_by_width = axes[1]

    print(f"=== boundary effect, solve={args.solve} ===")
    print(f"{'config':30s} {'n windows':>10s} {'edge/interior median':>21s} "
         f"{'IQR':>18s}  frac>1")

    summary_rows = []
    for i, path in enumerate(args.inputs):
        c = COLORS[i % len(COLORS)]
        windows, d = per_window_rows(path, args.solve)
        label = f"{path.stem} ({len(windows)}w, g_ratio={d.get('g_ratio')})"

        dist, normed, widths = edge_distance_dataset(windows)
        xs = np.arange(0, MAX_DIST + 1)
        med = np.full(xs.size, np.nan)
        q25 = np.full(xs.size, np.nan)
        q75 = np.full(xs.size, np.nan)
        n_at = np.full(xs.size, 0)
        for k, x in enumerate(xs):
            v = normed[dist == x]
            n_at[k] = v.size
            if v.size:
                med[k] = np.median(v)
                q25[k], q75[k] = np.percentile(v, [25, 75])

        ax_main.plot(xs, med, "-o", color=c, ms=3, lw=1.3, label=label)
        ax_main.fill_between(xs, q25, q75, color=c, alpha=0.15, lw=0)
        ax_n.semilogy(xs, n_at, "-o", color=c, ms=3, lw=1.0)

        ratios, rwidths = edge_interior_ratio(windows)
        jitter = (np.random.default_rng(i).uniform(-0.08, 0.08, ratios.size))
        ax_ratio.scatter(np.full(ratios.size, i) + jitter, ratios, s=14, color=c,
                         alpha=0.6, edgecolor="none")
        ax_ratio.plot([i - 0.25, i + 0.25], [np.median(ratios)] * 2, color=c, lw=2.2)

        ax_ratio_by_width.scatter(rwidths, ratios, s=14, color=c, alpha=0.6,
                                  edgecolor="none", label=label)

        q25r, q75r = np.percentile(ratios, [25, 75])
        frac_gt1 = float(np.mean(ratios > 1.0))
        print(f"{label:30s} {len(windows):10d} {np.median(ratios):21.3f} "
             f"[{q25r:.3f}, {q75r:.3f}]  {frac_gt1:.2f}")
        summary_rows.append(label)

    ax_main.axhline(1.0, color="black", lw=0.7, ls=":")
    ax_main.set_xlabel("distance from nearest window edge [rows]")
    ax_main.set_ylabel("per-row residual RMS /\nthat window's own median")
    ax_main.set_title("residual near a window edge, relative to its own window\n"
                      "(median + IQR band, pooled over all windows)", fontsize=10.5)
    ax_main.legend(fontsize=8, loc="best")
    ax_main.grid(alpha=0.25, lw=0.5)

    ax_n.set_xlabel("distance from nearest window edge [rows]")
    ax_n.set_ylabel("n rows\ncontributing")
    ax_n.set_title("sample size per distance bin -- only wide windows reach large distances",
                   fontsize=9)
    ax_n.grid(alpha=0.25, lw=0.5)

    ax_ratio.axhline(1.0, color="black", lw=0.7, ls=":")
    ax_ratio.set_xticks(range(len(args.inputs)))
    ax_ratio.set_xticklabels([p.stem for p in args.inputs], fontsize=7.5, rotation=12)
    ax_ratio.set_ylabel("edge / interior residual RMS\n(one point per window)")
    ax_ratio.set_title("per-window edge-vs-interior ratio\n(bar = median)", fontsize=10.5)
    ax_ratio.grid(alpha=0.25, lw=0.5)

    ax_ratio_by_width.axhline(1.0, color="black", lw=0.7, ls=":")
    ax_ratio_by_width.set_xlabel("window width [rows]")
    ax_ratio_by_width.set_ylabel("edge / interior ratio")
    ax_ratio_by_width.set_title("does the effect depend on window width?", fontsize=9)
    ax_ratio_by_width.legend(fontsize=7, loc="best")
    ax_ratio_by_width.grid(alpha=0.25, lw=0.5)

    fig.suptitle(f"Window-boundary residual effect, solve={args.solve} -- "
                f"exponential prior (no edge-weakening in the covariance itself)",
                fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = REPO_ROOT / "plots" / "joint_block"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gd_joint_block_boundary_effect.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
