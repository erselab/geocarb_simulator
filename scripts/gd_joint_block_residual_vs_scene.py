#!/usr/bin/env python3
"""What actually drives the joint block's large-scale residual structure:
the along-slit SURFACE-PRESSURE gradient, not CO2 and not keystone.

Motivation. The full-detector residual image
(`gd_joint_block_residual_plot.py --solve hires`) shows two broad lobes
either side of detector row ~425, where the whole pattern flips sign and
the per-row residual RMS drops into a sharp null. Neither keystone nor
smile explains it: `rows_crossed` rises monotonically straight through
that row, and the lobes are not aligned with the eta bin edges. The CO2
field does not explain it either -- CO2 is nearly flat there (416.8 ->
416.0 ppm across rows 400-500; its broad plume peaks back at row 335).

What does explain it: `along_slit_scene.p_surface_hpa` carries a
topographic depression (`mountain`, x0=-250 km, width 140 km, amp -250
hPa) whose bottom lands at detector row 427. Surface pressure sets the
total absorbing air column, so it has far more radiance leverage than a
few ppm of CO2, and the joint block retrieves ONLY `co2_scale` per bin --
every other quantity, p_surface included, is pinned at each bin's own
local-truth value (the sweep's "local-truth nuisance idealization"). That
is exact AT bin centres and wrong BETWEEN them, so wherever p_surface has
a steep along-slit gradient the between-bin representation error is large,
and the solver has no free parameter whose spectral signature can absorb
it. It lands in the residual instead.

The null at row ~425 is then simply where dp_surface/drow passes through
zero at the bottom of the depression, and the sign flip is that gradient
reversing.

THE UNDERLYING ASYMMETRY (this is the real mechanism, not just "pressure").
`build_forward_hires` treats the retrieved quantity and the nuisance
quantities differently, and the difference is an order of accuracy:

  - `co2_scale` IS interpolated -- linearly, from the G-dim state at
    bin_centers onto the anchors (`np.interp`, state-space only). Its
    representation error is therefore SECOND order in anchor spacing h,
    ~(1/8)|f''|h^2.
  - p_surface, H2O, CO, CH4 are NOT interpolated. Each anchor gets its own
    exact `als.atmosphere_at(anchor_eta)`, and pixels are then assigned
    NEAREST-anchor via `nearest_bin_scene` -- a step function. Their
    representation error is FIRST order, ~(1/4)|f'|h.

So the one quantity actually being retrieved is represented an order of
accuracy BETTER than the quantities held fixed, and the residual is
dominated by nuisance-state stepping rather than by CO2. That is why
residual amplitude does not track CO2 representation error, and why it is
uncorrelated with keystone/smile.

Measured (FPA2, gratio1, hires), correlation of log10 per-row residual RMS
against log10 of each field's own fractional representation error, computed
in the NEAREST-ANCHOR form the model actually uses (an earlier version of
this note used linear interpolation through bin centres -- the wrong
functional form for the nuisance quantities, and it understated their error
by ~2 orders of magnitude):

    H2O           +0.887      <- highest overall; also the largest median error
    p_surface     +0.752      (+0.940 restricted to the depression, rows 329-527)
    CO2           +0.592
    CO            +0.445
    rows_crossed  +0.380      (keystone, for reference)
    CH4           +0.139

and against |dp_surface/drow| directly: +0.693 over the whole slit,
+0.921 restricted to the depression's own rows.

Row-by-row through the feature, p_surface's nearest-anchor error and the
residual move together: row 400, 0.766 hPa -> 3.45e-3; row 425, 0.080 hPa
-> 3.14e-4 (the null); row 500, 0.765 hPa -> 5.66e-3; row 850, 0.004 hPa
-> 3.05e-5.

PREDICTION worth testing: interpolating the anchor atmospheres instead of
nearest-assigning them turns these first-order errors into second-order
ones. For the depression (|dp/drow| ~ 2.5 hPa/row, h ~ 1 row) that is
~0.6 hPa -> ~0.011 hPa, roughly 70x, which should remove most of these
lobes. Not yet tested.

Run:  PYTHONPATH=. python3 scripts/gd_joint_block_residual_vs_scene.py \\
        [/path/to/gd_joint_block_whole_slit_fpa2_gratio1.pkl]
Output: plots/joint_block/<stem>_residual_vs_scene.png
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
from geocarb_gert.gd_polynomials import rows_crossed, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

N_COLS = 1024
PLOT_STYLE = {
    "font.family": "serif", "font.size": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8,
}


def per_row_resid_rms(results: dict, solve: str) -> np.ndarray:
    rr = np.full(1024, np.nan)
    for w in results.values():
        key = f"resid_{solve}"
        if key not in w:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        blk = np.asarray(w[key], dtype=float).reshape(hi - lo + 1, N_COLS)
        rr[lo:hi + 1] = np.sqrt(np.mean(blk ** 2, axis=1))
    return rr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    default="/Users/scrowell/geocarb_simulator_results/"
                            "gd_joint_block_whole_slit_fpa2_gratio1.pkl")
    ap.add_argument("--solve", default="hires", choices=["coarse", "hires"])
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"MISSING: {in_path}")
        return 1
    with open(in_path, "rb") as f:
        d = pickle.load(f)
    results, fpa = d["results"], int(d.get("fpa", 2))

    rows = np.arange(1024.0)
    _, s = xy_to_wavelength_slit(fpa, np.full(1024, 512.0), rows)
    eta = s / s_max(fpa)
    x_km = eta * als.SLIT_HALF_KM

    co2 = als.xco2_ppm(x_km)
    psurf = als.p_surface_hpa(x_km)
    dp = np.abs(np.gradient(psurf))
    rc = rows_crossed(fpa, rows)
    rr = per_row_resid_rms(results, args.solve)

    row_bottom = int(np.argmin(psurf))
    ok = np.isfinite(rr) & (dp > 1e-6) & (rr > 0)
    c_all = np.corrcoef(np.log10(rr[ok]), np.log10(dp[ok]))[0, 1]
    lo_m, hi_m = 329, 528
    c_mtn = np.corrcoef(np.log10(rr[lo_m:hi_m]), np.log10(dp[lo_m:hi_m]))[0, 1]
    print(f"depression bottom at row {row_bottom} (x={x_km[row_bottom]:.0f} km, "
          f"p={psurf[row_bottom]:.1f} hPa vs background {als.P_BG_HPA:.1f})")
    print(f"corr(log residRMS, log |dp/drow|): all slit {c_all:+.3f}, "
          f"rows {lo_m}-{hi_m-1} {c_mtn:+.3f}")

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.3]})

    def mark(ax):
        ax.axvline(row_bottom, color="crimson", lw=1.0, ls="--", zorder=1)
        ax.axvspan(lo_m, hi_m - 1, color="crimson", alpha=0.06, zorder=0)

    ax = axes[0]
    ax.semilogy(rows, rr, lw=0.9, color="tab:blue")
    mark(ax)
    ax.set_ylabel("per-row\nresidual RMS")
    ax.set_title(f"FPA{fpa} joint block ({args.solve}): the large residual lobes track the "
                 f"SURFACE-PRESSURE gradient, not CO2 and not keystone", fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    ax.plot(rows, co2, lw=1.1, color="black")
    mark(ax)
    ax.set_ylabel("true CO2 [ppm]")
    ax.text(0.5, 0.08, "CO2 is essentially FLAT across the lobes "
            "(416.8 -> 416.0 ppm, rows 400-500); its plume peaks back at row 335",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="0.35")

    ax = axes[2]
    ax.plot(rows, psurf, lw=1.1, color="tab:purple")
    mark(ax)
    ax.set_ylabel("true p_surface\n[hPa]")
    ax.annotate(f"depression bottom, row {row_bottom}",
                xy=(row_bottom, psurf[row_bottom]), xytext=(row_bottom + 90, psurf[row_bottom] + 60),
                fontsize=8.5, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))

    ax = axes[3]
    ax.semilogy(rows, np.maximum(dp, 1e-6), lw=1.1, color="tab:purple",
                label="|dp_surface/drow| [hPa/row]")
    ax.semilogy(rows, rr / np.nanmax(rr) * np.nanmax(dp), lw=0.9, color="tab:blue",
                alpha=0.8, label="per-row residual RMS (rescaled)")
    ax.semilogy(rows, np.maximum(rc, 1e-3), lw=0.9, color="0.55", ls=":",
                label="rows_crossed (keystone) -- monotonic, no null")
    mark(ax)
    ax.set_ylabel("gradient / residual\n(log, rescaled)")
    ax.set_xlabel("detector row")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"the null at row ~{row_bottom} is where dp_surface/drow passes through zero  "
                 f"(corr over shaded rows: {c_mtn:+.3f})", fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    out_dir = REPO_ROOT / "plots" / "joint_block"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{in_path.stem}_residual_vs_scene.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
