#!/usr/bin/env python3
"""Every state-vector row along the slit -- retrieved vs. truth -- from a
sweep's saved snapshots.

Complements `gd_joint_block_whole_slit_plot.py`, which only ever plots CO2
and assumes the packed vector IS CO2. That assumption breaks the moment more
than one row is free (with `--free co2_ppm,p_surface_hpa` the packed vector
is 2G long), so this reads `spec.snapshot()` instead: per-row `values`,
`prior`, and `positions`, for every row including the FROZEN ones.

Plotting the frozen rows is not padding -- it is the check that they really
were held at truth, and it makes two runs with different free/frozen splits
directly comparable. A frozen row whose retrieved curve departs from truth
would mean the snapshot or the forward model is wrong.

For each row: absolute value against truth (top strip) and fractional
departure from truth (bottom strip), coarse and hi-res overlaid. Free rows
are labelled with their prior sigma and correlation length, since those set
how far the row was permitted to move.

Run:  PYTHONPATH=. python3 scripts/gd_joint_block_state_plot.py \\
        results/gd_joint_block_whole_slit_fpa2_gratio1_stateinterp_free-co2-p.pkl
Output: plots/joint_block/<stem>_state.png
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

SOLVE_STYLE = {"coarse": ("tab:orange", "-"), "hires": ("tab:blue", "-")}
PLOT_STYLE = {
    "font.family": "serif", "font.size": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8,
}


def stitch(results: dict, solve: str, eta_rows: np.ndarray):
    """{row_name: per-detector-row retrieved values}, stitched across windows.

    Each window's own bin centres are its row's `positions`; values are
    linearly interpolated onto the detector rows that window covers, which is
    the same reconstruction `gd_joint_block_whole_slit_plot` does for CO2.
    """
    out, meta = {}, {}
    for w in results.values():
        snap = w.get(solve)
        if snap is None:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        sl = slice(lo, hi + 1)
        for name, rec in snap["params"].items():
            arr = out.setdefault(name, np.full(len(eta_rows), np.nan))
            pos, val = np.asarray(rec["positions"]), np.asarray(rec["values"])
            arr[sl] = (np.interp(eta_rows[sl], pos, val) if pos.size > 1
                       else np.full(hi - lo + 1, val[0]))
            meta.setdefault(name, dict(free=rec["free"], sigma=rec["sigma"],
                                       corr_length=rec["corr_length"]))
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--fpa", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"MISSING: {in_path}")
        return 1
    with open(in_path, "rb") as f:
        d = pickle.load(f)
    results = d["results"]
    fpa = args.fpa if args.fpa is not None else int(d.get("fpa", 2))

    rows = np.arange(1024.0)
    _, s = xy_to_wavelength_slit(fpa, np.full(1024, 512.0), rows)
    eta_rows = s / s_max(fpa)
    x_km = eta_rows * als.SLIT_HALF_KM
    truth = {name: np.asarray(fn(x_km), dtype=float)
             for name, fn in als.STATE_FIELDS.items()}

    solves = [sv for sv in ("coarse", "hires")
              if any(sv in w for w in results.values())]
    if not solves:
        print(f"{in_path.name} has no snapshots -- was it produced before StateSpec?")
        return 1
    stitched = {sv: stitch(results, sv, eta_rows) for sv in solves}
    meta = stitched[solves[0]][1]
    names = [n for n in als.STATE_FIELDS if n in meta]

    print(f"{in_path.name}: {len(results)} windows, FPA{fpa}, solves={solves}, "
          f"free={[n for n in names if meta[n]['free']]}")

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(2 * len(names), 1, figsize=(13, 2.4 * 2 * len(names)),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.0] * len(names)})

    for i, name in enumerate(names):
        ax_v, ax_d = axes[2 * i], axes[2 * i + 1]
        m = meta[name]
        tag = (f"FREE (sigma={m['sigma']:.3g}, corr={m['corr_length']:.4f} eta "
               f"= {m['corr_length']*als.SLIT_HALF_KM:.0f} km)" if m["free"]
               else "frozen at local truth")

        ax_v.plot(rows, truth[name], color="black", lw=1.2, label="truth", zorder=5)
        for sv in solves:
            v = stitched[sv][0].get(name)
            if v is None:
                continue
            c, ls = SOLVE_STYLE[sv]
            ax_v.plot(rows, v, color=c, ls=ls, lw=0.9, alpha=0.85, label=sv)
            with np.errstate(invalid="ignore", divide="ignore"):
                ax_d.plot(rows, (v - truth[name]) / truth[name], color=c, ls=ls, lw=0.8)
        ax_v.set_ylabel(name)
        ax_v.set_title(f"{name} -- {tag}", fontsize=10)
        ax_v.legend(fontsize=8, loc="upper right")
        ax_d.axhline(0, color="black", lw=0.6)
        ax_d.set_ylabel("frac. dev.\nfrom truth")
        if not m["free"]:
            # A frozen row must sit exactly on truth; show the achieved scale
            # rather than an empty axis, so a violation is obvious.
            worst = max(np.nanmax(np.abs((stitched[sv][0][name] - truth[name])
                                         / truth[name])) for sv in solves)
            ax_d.text(0.99, 0.85, f"max |dev| = {worst:.2e}", ha="right",
                      transform=ax_d.transAxes, fontsize=8, color="0.35")
    axes[-1].set_xlabel("detector row")

    fig.suptitle(f"FPA{fpa} state vector along the slit -- {in_path.stem}", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_dir = REPO_ROOT / "plots" / "joint_block"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{in_path.stem}_state.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
