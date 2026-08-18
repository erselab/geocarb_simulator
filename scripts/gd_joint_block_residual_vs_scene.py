#!/usr/bin/env python3
"""Residual structure against BOTH the true scene and the retrieved state.

Originally this script only compared the residual to the *true* scene, and
concluded (correctly, for the CO2-only runs it was written against) that the
large-scale residual lobes track the along-slit SURFACE-PRESSURE gradient,
not CO2 and not keystone -- see "the original finding" below, which still
holds for any run where p_surface is frozen.

It now also plots the RETRIEVED CO2 and surface-pressure profiles alongside
the true ones, their departures from truth, and the per-row chi2 the solver
actually minimised, so the question "how do the retrieved patterns covary
with the residual / chi2 / RMS" can be read straight off one figure and off
the printed correlation table. That is the question that matters the moment
p_surface is free (`--free co2_ppm,p_surface_hpa`), because then the residual
is no longer a clean read-out of a *fixed* quantity's representation error:
the free rows have moved to absorb some of it, and how they moved is exactly
what the departure panels show.

Reads chi2 in the solver's own units. `gd_joint_block_whole_slit_sweep` uses
a uniform relative weighting, `Sy_inv = 1/y_scale**2` with `y_scale =
mean|y_true|` over the window, so per-row chi2-per-channel is exactly
`mean(resid**2)/y_scale**2` -- recovered here from the cached band image
(`scripts/gd_cache_band_image.py`), since no sweep pickle saves `y_true` or
even `y_scale`. Without that cache the chi2 panel is skipped with a warning
rather than being faked from raw radiance.

THE ORIGINAL FINDING (CO2-only runs, p_surface frozen)
------------------------------------------------------
The full-detector residual image shows two broad lobes either side of
detector row ~425, where the pattern flips sign and the per-row residual RMS
drops into a sharp null. Neither keystone nor smile explains it: `rows_
crossed` rises monotonically straight through that row. CO2 does not either
-- it is nearly flat there (416.8 -> 416.0 ppm across rows 400-500; its broad
plume peaks back at row 335).

What does: `along_slit_scene.p_surface_hpa` carries a topographic depression
(`mountain`, x0=-250 km, width 140 km, amp -250 hPa) whose bottom lands at
detector row 427. Surface pressure sets the total absorbing air column, so it
has far more radiance leverage than a few ppm of CO2, and with only
`co2_scale` retrieved the solver has no free parameter whose spectral
signature can absorb the p_surface error. It lands in the residual instead.
The null at row ~425 is simply where dp_surface/drow passes through zero at
the bottom of the depression; the sign flip is that gradient reversing.

The underlying asymmetry is an order of accuracy: the retrieved row IS
interpolated from the state onto the anchors (`np.interp`, second order,
~(1/8)|f''|h^2), while frozen rows are assigned NEAREST-anchor by
`nearest_bin_scene` (a step function, first order, ~(1/4)|f'|h). The one
quantity being retrieved is represented better than the quantities held
fixed, which is why residual amplitude does not track CO2 representation
error and is uncorrelated with keystone.

Measured (FPA2, gratio1, hires, CO2-only), correlation of log10 per-row
residual RMS against log10 of each field's own fractional nearest-anchor
representation error:

    H2O           +0.887      <- highest overall; also the largest median error
    p_surface     +0.752      (+0.940 restricted to the depression, rows 329-527)
    CO2           +0.592
    CO            +0.445
    rows_crossed  +0.380      (keystone, for reference)
    CH4           +0.139

and against |dp_surface/drow| directly: +0.693 over the whole slit, +0.921
restricted to the depression's own rows.

Run:  PYTHONPATH=. python3 scripts/gd_joint_block_residual_vs_scene.py \\
        results/gd_joint_block_whole_slit_fpa2_gratio1_stateinterp_free-co2-p.pkl
Output: plots/joint_block/<stem>_residual_vs_scene.png
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

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import rows_crossed, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402
from gd_joint_block_state_plot import stitch  # noqa: E402
from gd_joint_block_residual_plot import continuum_of, load_band_image  # noqa: E402

N_COLS = 1024
#: The depression's own rows, the sub-range the original finding quotes its
#: strongest correlation over. Kept as the shaded/annotated reference band.
MTN_ROWS = (329, 528)
SOLVE_STYLE = {"coarse": ("tab:orange", "-"), "hires": ("tab:blue", "-")}
#: Surface pressure above which `model_sampler.pressure_to_alt_std_atm`
#: returns NaN -- there is no standard atmosphere below sea level. See
#: `failed_windows` for why this matters to any run with p_surface free.
P_STD_CEILING_HPA = 1013.25
PLOT_STYLE = {
    "font.family": "serif", "font.size": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8,
}


def per_row_residual(results: dict, solve: str, A=None, cont=None):
    """Per-row residual RMS, plus chi2-per-channel and %-of-continuum RMS.

    `chi2` is the solver's own objective density: the sweep weights every
    channel identically with `Sy_inv = 1/y_scale**2`, `y_scale =
    mean|y_true|` over that window, so `mean(resid**2)/y_scale**2` is
    literally the data term it minimised, per channel. It needs `A` (the
    rendered band image) because `y_scale` is not in the pickle; without it
    both chi2 and the %-of-continuum RMS come back all-NaN.
    """
    rms = np.full(N_COLS, np.nan)
    chi2 = np.full(N_COLS, np.nan)
    pct = np.full(N_COLS, np.nan)
    for w in results.values():
        key = f"resid_{solve}"
        if key not in w:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        blk = np.asarray(w[key], dtype=float).reshape(hi - lo + 1, N_COLS)
        rms[lo:hi + 1] = np.sqrt(np.mean(blk ** 2, axis=1))
        if A is not None:
            # exactly the sweep's own y_scale: mean|y_true| over the window
            y_scale = float(np.mean(np.abs(A[lo:hi + 1, :])))
            chi2[lo:hi + 1] = np.mean(blk ** 2, axis=1) / y_scale ** 2
            pct[lo:hi + 1] = np.sqrt(np.mean((100.0 * blk / cont[lo:hi + 1, :]) ** 2, axis=1))
    return rms, chi2, pct


def legacy_state(results: dict, solve: str, eta_rows: np.ndarray):
    """Retrieved CO2 for sweeps written before `StateSpec` snapshots.

    Those pickles store only the packed vector (`x_<solve>`, pure co2_scale)
    and `prior_co2_ppm_bins`; every other quantity was frozen at local truth
    and has no stored value at all. Returns the same `{name: per-row array}`
    shape `stitch` does, with CO2 the only key -- so the caller's "is this
    row present?" logic is identical for both formats.
    """
    out = {}
    for w in results.values():
        if f"x_{solve}" not in w:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        arr = out.setdefault("co2_ppm", np.full(N_COLS, np.nan))
        bc = np.asarray(w["bin_centers"], dtype=float)
        vals = np.asarray(w["prior_co2_ppm_bins"], dtype=float) * np.asarray(w[f"x_{solve}"],
                                                                             dtype=float)
        arr[lo:hi + 1] = (np.interp(eta_rows[lo:hi + 1], bc, vals) if bc.size > 1
                          else vals[0])
    return out


def failed_windows(results: dict):
    """`[(row_lo, row_hi, message)]` for windows the sweep could not solve.

    `_worker` catches per-window exceptions and stores them as `error` rather
    than killing the sweep, so a pickle can be missing whole stretches of slit
    with nothing in the summary numbers to say so. Plotting them as a gap
    would read as "the residual is small here"; they are drawn hatched
    instead.

    The failure to expect with `--free ...,p_surface_hpa` is
    ``ValueError: cannot convert float NaN to integer``. Its cause:
    `pressure_to_alt_std_atm` returns NaN above sea level (1013.25 hPa) --
    there is no standard atmosphere below sea level -- and that NaN
    propagates through `atmosphere_from_params`' `z_km` into `q_levels`,
    surfacing much later as an int conversion. Before 2026-08-18 the truth
    field peaked 0.1 hPa under that ceiling, so a p_surface retrieved as
    `kind="scale"` had essentially no headroom: `gauss_newton_state`'s
    finite-difference probe alone (`step=1e-3`, so `p * 1.001` ~ +1.0 hPa)
    crossed it, and every FPA2 window spanning the eastern rise (rows
    583-844, 26% of the slit) died.

    Fixed at the source by `along_slit_scene.P_HEADROOM_HPA` = 10.0, so this
    should no longer fire on a sweep run against the current scene. Kept
    because the check is cheap, because older pickles still carry the
    failures, and because any future state row with a hard physical bound
    can fail the same silent way.
    """
    out = []
    for w in results.values():
        if "error" in w:
            out.append((int(w["row_lo"]), int(w["row_hi"]), str(w["error"])))
    return sorted(out)


def corr(a, b, log=False):
    """Pearson r over the rows where both are finite (and positive, if log)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if log:
        ok &= (a > 0) & (b > 0)
        a, b = np.log10(np.where(ok, a, 1.0)), np.log10(np.where(ok, b, 1.0))
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    default=str(REPO_ROOT / "results" /
                                "gd_joint_block_whole_slit_fpa2_gratio1_stateinterp"
                                "_free-co2-p.pkl"))
    ap.add_argument("--solve", default="both", choices=["coarse", "hires", "both"],
                    help="which solve(s) to draw (default: both, whichever are present)")
    ap.add_argument("--scatter-solve", default=None, choices=["coarse", "hires"],
                    help="solve used for the bottom covariation scatters "
                         "(default: hires if present, else coarse)")
    ap.add_argument("--x-axis", choices=["row", "eta", "km"], default="row",
                    help="along-slit coordinate (default: row, matching the "
                         "detector-space residual plots)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"MISSING: {in_path}")
        return 1
    with open(in_path, "rb") as f:
        d = pickle.load(f)
    results, fpa = d["results"], int(d.get("fpa", 2))
    free_rows = tuple(d.get("free", ("co2_ppm",)))

    rows = np.arange(float(N_COLS))
    _, s = xy_to_wavelength_slit(fpa, np.full(N_COLS, 512.0), rows)
    eta_rows = s / s_max(fpa)
    x_km = eta_rows * als.SLIT_HALF_KM
    truth = {name: np.asarray(fn(x_km), dtype=float) for name, fn in als.STATE_FIELDS.items()}

    psurf, co2_true = truth["p_surface_hpa"], truth["co2_ppm"]
    dp = np.abs(np.gradient(psurf))
    rc = rows_crossed(fpa, rows)
    row_bottom = int(np.argmin(psurf))
    lo_m, hi_m = MTN_ROWS

    solves = [sv for sv in ("coarse", "hires") if any(f"resid_{sv}" in w for w in results.values())]
    if args.solve != "both":
        solves = [sv for sv in solves if sv == args.solve]
    if not solves:
        print(f"{in_path.name} has no residuals for solve={args.solve}")
        return 1
    scat_solve = args.scatter_solve or ("hires" if "hires" in solves else solves[0])

    # chi2 and %-of-continuum both need the rendered radiance the sweep fitted
    # against, which no pickle carries. Skip those panels loudly rather than
    # substituting raw radiance and mislabelling it.
    A, cache_p = load_band_image(fpa, uniform=bool(d.get("uniform", False)))
    cont = continuum_of(A) if A is not None else None
    if A is None:
        print(f"WARNING: no cached band image at {cache_p} -- chi2 and %-of-continuum "
              f"panels will be blank. Generate it with:\n"
              f"  PYTHONPATH=. python3 scripts/gd_cache_band_image.py --fpa {fpa}"
              f"{' --uniform' if d.get('uniform') else ''}")

    resid, chi2, pct, state = {}, {}, {}, {}
    for sv in solves:
        resid[sv], chi2[sv], pct[sv] = per_row_residual(results, sv, A, cont)
        st, _meta = stitch(results, sv, eta_rows)
        state[sv] = st or legacy_state(results, sv, eta_rows)

    have_retrieved_p = any("p_surface_hpa" in state[sv] for sv in solves)

    # ---------------------------------------------------------------- report --
    print(f"{in_path.name}: {len(results)} windows, FPA{fpa}, g_ratio={d.get('g_ratio')}, "
          f"free={free_rows}, solves={solves}")
    print(f"depression bottom at row {row_bottom} (x={x_km[row_bottom]:.0f} km, "
          f"p={psurf[row_bottom]:.1f} hPa); truth peaks at {psurf.max():.2f} hPa, "
          f"{P_STD_CEILING_HPA - psurf.max():.2f} hPa of headroom under the ceiling")

    failed = failed_windows(results)
    if failed:
        n_rows_lost = sum(hi - lo + 1 for lo, hi, _ in failed)
        print(f"\n!! {len(failed)} of {len(results)} windows FAILED "
              f"({n_rows_lost} rows, {100.0 * n_rows_lost / N_COLS:.0f}% of the slit) -- "
              f"drawn hatched, excluded from every number below:")
        for lo, hi, msg in failed:
            print(f"     rows {lo:4d}-{hi:4d}  p_true max {psurf[lo:hi + 1].max():8.3f} hPa  {msg}")
        if any("NaN to integer" in m for _, _, m in failed):
            print(f"   see failed_windows() -- p_surface as kind='scale' has no headroom "
                  f"above the {P_STD_CEILING_HPA} hPa standard-atmosphere ceiling.")
    print()
    hdr = (f"{'solve':7s} {'residRMS':>10s} {'chi2/chan':>10s} {'dCO2 rms':>9s} "
           f"{'dP rms':>9s} | {'r(res,|dCO2|)':>13s} {'r(res,|dP|)':>11s} "
           f"{'r(res,|dp/dr|)':>14s} {'r(dCO2,dP)':>10s}")
    print(hdr)
    print("-" * len(hdr))
    dev = {}
    for sv in solves:
        dco2 = state[sv].get("co2_ppm", np.full(N_COLS, np.nan)) - co2_true
        dpr = (state[sv].get("p_surface_hpa", np.full(N_COLS, np.nan)) - psurf)
        dev[sv] = (dco2, dpr)
        print(f"{sv:7s} {np.nanmedian(resid[sv]):10.3e} "
              f"{np.nanmedian(chi2[sv]):10.3e} "
              f"{np.sqrt(np.nanmean(dco2 ** 2)):9.4f} "
              f"{np.sqrt(np.nanmean(dpr ** 2)):9.4f} | "
              f"{corr(resid[sv], np.abs(dco2), log=True):13.3f} "
              f"{corr(resid[sv], np.abs(dpr), log=True):11.3f} "
              f"{corr(resid[sv], dp, log=True):14.3f} "
              f"{corr(dco2, dpr):10.3f}")
    print()
    for sv in solves:
        c_all = corr(resid[sv], dp, log=True)
        c_mtn = corr(resid[sv][lo_m:hi_m], dp[lo_m:hi_m], log=True)
        print(f"  {sv}: corr(log residRMS, log |dp/drow|) -- all slit {c_all:+.3f}, "
              f"rows {lo_m}-{hi_m - 1} {c_mtn:+.3f}")

    # ------------------------------------------------------------------ plot --
    XAXIS = {"row": (rows, "detector row"),
             "eta": (eta_rows, r"along-slit $\eta$"),
             "km": (x_km, "along-slit distance [km]")}
    xv, xlabel = XAXIS[args.x_axis]
    x_bottom = xv[row_bottom]
    x_mtn = (xv[lo_m], xv[min(hi_m - 1, N_COLS - 1)])

    plt.rcParams.update(PLOT_STYLE)
    n_line = 8 if have_retrieved_p else 6
    fig = plt.figure(figsize=(13.5, 1.85 * n_line + 5.0))
    # a tall stack of shared-x line panels, then a spacer, then the scatters
    gs = fig.add_gridspec(n_line + 2, 3, height_ratios=[1.0] * n_line + [0.45, 2.0],
                          hspace=0.18, wspace=0.42, top=0.965, bottom=0.045,
                          left=0.075, right=0.965)
    line_axes = [fig.add_subplot(gs[i, :]) for i in range(n_line)]
    for ax in line_axes[:-1]:
        ax.tick_params(labelbottom=False)
    for ax in line_axes[1:]:
        ax.sharex(line_axes[0])

    def mark(ax, hatch_failed=True):
        ax.axvline(x_bottom, color="crimson", lw=1.0, ls="--", zorder=1)
        ax.axvspan(*x_mtn, color="crimson", alpha=0.06, zorder=0)
        if hatch_failed:
            for lo, hi, _ in failed:
                ax.axvspan(xv[lo], xv[min(hi, N_COLS - 1)], facecolor="0.85",
                           edgecolor="0.55", hatch="///", lw=0.0, alpha=0.55, zorder=0)
        ax.grid(alpha=0.25, lw=0.5)

    k = 0
    # 1 -- residual RMS
    ax = line_axes[k]; k += 1
    for sv in solves:
        c, ls = SOLVE_STYLE[sv]
        y = pct[sv] if A is not None else resid[sv]
        ax.semilogy(xv, y, lw=0.9, color=c, ls=ls, label=sv)
    ax.set_ylabel("per-row resid RMS\n" + ("[% of continuum]" if A is not None else "[radiance]"))
    ax.legend(fontsize=8, loc="upper right", ncol=len(solves))
    if failed:
        ax.text(0.01, 0.06, "hatched = windows the sweep FAILED to solve (no data, not a "
                            "small residual)", transform=ax.transAxes, fontsize=8,
                color="0.3")
    mark(ax)

    # 2 -- chi2 per channel, the solver's own objective density
    ax = line_axes[k]; k += 1
    for sv in solves:
        c, ls = SOLVE_STYLE[sv]
        ax.semilogy(xv, chi2[sv], lw=0.9, color=c, ls=ls, label=sv)
    ax.set_ylabel("chi2 per channel\n" + r"$\overline{r^2}/y_{scale}^2$")
    if A is None:
        ax.text(0.5, 0.5, "needs the cached band image -- see warning above",
                transform=ax.transAxes, ha="center", fontsize=9, color="crimson")
    mark(ax)

    # 3/4 -- CO2 retrieved vs truth, and its departure
    ax = line_axes[k]; k += 1
    ax.plot(xv, co2_true, lw=1.6, color="black", label="truth", zorder=2)
    for sv in solves:
        v = state[sv].get("co2_ppm")
        if v is None:
            continue
        c, ls = SOLVE_STYLE[sv]
        ax.plot(xv, v, lw=0.9, color=c, ls=ls, alpha=0.9, zorder=5,
                label=f"retrieved ({sv})")
    ax.set_ylabel("CO2 [ppm]")
    ax.legend(fontsize=8, loc="upper right", ncol=len(solves) + 1)
    mark(ax)

    ax = line_axes[k]; k += 1
    for sv in solves:
        c, ls = SOLVE_STYLE[sv]
        ax.plot(xv, dev[sv][0], lw=0.9, color=c, ls=ls)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("retr - true\nCO2 [ppm]")
    mark(ax)

    # 5/6 -- surface pressure, same pair (only when it was actually retrieved)
    if have_retrieved_p:
        ax = line_axes[k]; k += 1
        ax.plot(xv, psurf, lw=1.6, color="black", label="truth", zorder=2)
        for sv in solves:
            v = state[sv].get("p_surface_hpa")
            if v is None:
                continue
            c, ls = SOLVE_STYLE[sv]
            ax.plot(xv, v, lw=0.9, color=c, ls=ls, alpha=0.9, zorder=5,
                label=f"retrieved ({sv})")
        ax.annotate(f"depression bottom, row {row_bottom}",
                    xy=(x_bottom, psurf[row_bottom]),
                    xytext=(xv[min(row_bottom + 90, N_COLS - 1)], psurf[row_bottom] + 60),
                    fontsize=8.5, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
        # the hard ceiling that makes p_surface-as-a-scale fail: see failed_windows
        ax.axhline(P_STD_CEILING_HPA, color="crimson", lw=0.9, ls=":")
        ax.text(0.01, 0.94, f"{P_STD_CEILING_HPA} hPa -- standard-atmosphere ceiling; "
                            f"above it pressure_to_alt_std_atm returns NaN",
                transform=ax.transAxes, va="top", fontsize=8, color="crimson")
        ax.set_ylabel("p_surface [hPa]")
        ax.legend(fontsize=8, loc="lower right", ncol=len(solves) + 1)
        mark(ax)

        ax = line_axes[k]; k += 1
        for sv in solves:
            c, ls = SOLVE_STYLE[sv]
            ax.plot(xv, dev[sv][1], lw=0.9, color=c, ls=ls)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("retr - true\np_surf [hPa]")
        mark(ax)

    # 7 -- the two departures on one axis, in fractional units. This is the
    # degeneracy read-out: CO2 and p_surface are near-perfectly anti-correlated
    # in this band alone, so a free pressure row absorbs CO2 signal.
    ax = line_axes[k]; k += 1
    sv = scat_solve
    c2, pr = dev[sv]
    ax.plot(xv, c2 / co2_true, lw=1.0, color="tab:green", label=r"$\Delta$CO2 / CO2")
    if have_retrieved_p:
        ax.plot(xv, pr / psurf, lw=1.0, color="tab:purple", label=r"$\Delta p$ / p")
        r_deg = corr(c2 / co2_true, pr / psurf)
        ax.set_title(f"the degeneracy, {sv}: corr = {r_deg:+.3f}", fontsize=9.5, pad=3)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("fractional\ndeparture")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    mark(ax)

    # 8 -- the original mechanism panel, unchanged in content
    ax = line_axes[k]; k += 1
    ax.semilogy(xv, np.maximum(dp, 1e-6), lw=1.1, color="tab:purple",
                label="|dp_surface/drow| [hPa/row]")
    rr = resid[scat_solve]
    ax.semilogy(xv, rr / np.nanmax(rr) * np.nanmax(dp), lw=0.9, color="tab:blue", alpha=0.8,
                label=f"per-row residual RMS, {scat_solve} (rescaled)")
    ax.semilogy(xv, np.maximum(rc, 1e-3), lw=0.9, color="0.55", ls=":",
                label="rows_crossed (keystone) -- monotonic, no null")
    ax.set_ylabel("gradient / residual\n(log, rescaled)")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8, loc="lower right")
    c_mtn = corr(rr[lo_m:hi_m], dp[lo_m:hi_m], log=True)
    ax.set_title(f"null at row ~{row_bottom} is where dp_surface/drow passes through zero "
                 f"(corr over shaded rows, {scat_solve}: {c_mtn:+.3f})", fontsize=9.5, pad=3)
    mark(ax)

    # ------------------------------------------------- covariation scatters --
    dco2, dpr = dev[scat_solve]
    rr = pct[scat_solve] if A is not None else resid[scat_solve]
    rlabel = "resid RMS [% cont.]" if A is not None else "resid RMS [radiance]"

    ax = fig.add_subplot(gs[n_line + 1,0])
    ax.scatter(np.abs(dco2), rr, s=5, c=dp, cmap="viridis", norm="log", alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"|$\Delta$CO2| [ppm]"); ax.set_ylabel(rlabel)
    ax.set_title(f"r(log,log) = {corr(rr, np.abs(dco2), log=True):+.3f}", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.5)

    ax = fig.add_subplot(gs[n_line + 1,1])
    if have_retrieved_p:
        sc = ax.scatter(np.abs(dpr), rr, s=5, c=dp, cmap="viridis", norm="log", alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"|$\Delta p_{surf}$| [hPa]")
        ax.set_title(f"r(log,log) = {corr(rr, np.abs(dpr), log=True):+.3f}", fontsize=9.5)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("|dp/drow| [hPa/row]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "p_surface was frozen\n(nothing retrieved to plot)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(rlabel)
    ax.grid(alpha=0.25, lw=0.5)

    ax = fig.add_subplot(gs[n_line + 1,2])
    if have_retrieved_p:
        sc = ax.scatter(dco2 / co2_true, dpr / psurf, s=5, c=rr, cmap="magma_r",
                        norm="log", alpha=0.85)
        ax.set_xlabel(r"$\Delta$CO2 / CO2"); ax.set_ylabel(r"$\Delta p$ / p")
        ok = np.isfinite(dco2) & np.isfinite(dpr)
        slope = (np.polyfit(dco2[ok] / co2_true[ok], dpr[ok] / psurf[ok], 1)[0]
                 if ok.sum() > 2 else np.nan)
        ax.set_title(f"degeneracy: r = {corr(dco2, dpr):+.3f}, slope = {slope:+.2f}",
                     fontsize=9.5)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(rlabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "p_surface was frozen", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
    ax.grid(alpha=0.25, lw=0.5)

    free_tag = ", ".join(free_rows)
    fail_tag = (f"   [{len(failed)}/{len(results)} windows failed -- hatched]"
                if failed else "")
    fig.suptitle(f"FPA{fpa} joint block -- residual vs. true AND retrieved scene "
                 f"(free: {free_tag}){fail_tag}\n{in_path.stem}", fontsize=12.5, y=0.995)
    out_dir = REPO_ROOT / "plots" / "joint_block"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{in_path.stem}_residual_vs_scene.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
