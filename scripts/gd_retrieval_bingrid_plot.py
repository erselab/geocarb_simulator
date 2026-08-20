#!/usr/bin/env python3
"""Plots for the Phase 1 bingrid retrieval sweep (PROJECT_STATUS.md Sec.5:
prior=truth at the bin scale, state_interp="linear", CO2/CO2+p_surface,
g_ratio x anchor_density) -- gd_joint_block_matrix.py itself only ever
produces REPORT.md (a table), no figures, so this is the plotting step
that was missing.

Three kinds of figure, all reading gd_joint_block_matrix.py's own output
(results/config_matrix/<tag>/results/) rather than re-solving or
re-deriving anything:

  1. FPA residuals, per config: the FULL detector grid (reuses
     gd_joint_block_residual_plot.py's own full_detector_figure() --
     same convention as every other residual figure in this project) plus
     three ZOOMED crops -- the two CO2/CO hot spots and the p_surface
     "mountain" transition (row bands below), each its own local colour
     scale since the three regions' own amplitudes differ by orders of
     magnitude.
  2. Posterior error vs. slit position, per (free-set, retrieved
     quantity): retrieved-minus-truth across the whole slit, one line per
     (g_ratio, anchor_density), coloured/styled the same way
     forward_check_residual_by_row_fpa2.png's bingrid family already is
     (Blues, darker = finer g_ratio; linestyle = anchor_density) so this
     reads consistently with the forward-only plots from the same sweep
     design.
  3. RMS-across-the-sweep comparison: CO2/p_surface rms+max vs. g_ratio,
     one line per anchor_density -- the retrieval analogue of
     forward_check_g_ratio_sweep_fpa2.png. Both free-sets overlaid on the
     CO2 panel (co2 vs. co2p both retrieve co2_ppm, so this is a direct
     read on whether adding p_surface as a second free row costs CO2
     accuracy at fixed resolution); p_surface panel is co2p-only.

The 2D residual figures (1) need each config's own raw per-window `.pkl`
(the full 2D resid_hires block only lives there -- summary.pkl's own
`stitch()` already collapsed it to per-row RMS). Figures (2) and (3) only
need results/config_matrix/<tag>/results/summary.pkl, already written by
gd_joint_block_matrix.py's own postprocess run -- no raw pickles reloaded
for those.

Run:  PYTHONPATH=. python3 scripts/gd_retrieval_bingrid_plot.py \\
        --tag retrieval_bingrid_v1
Output: plots/config_matrix/<tag>/*.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gd_joint_block_matrix import parse_config, FREE_SETS  # noqa: E402
from gd_joint_block_residual_plot import full_detector_figure  # noqa: E402

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

N_COLS = 1024

#: row bands to zoom into on every config's own residual image -- the two
#: CO2/CO hot spots (same convention gd_realistic_prior_forward_check.py's
#: own HOTSPOT_ROW_BANDS uses) plus the p_surface "mountain" transition
#: (along_slit_scene.py's `mountain = _gauss(x_km, x0=-250.0, width=140.0,
#: ...)` term -- row range computed directly from the real FPA2 row<->eta
#: mapping at +/-3*width around x0, not eyeballed: rows 272-583 covers
#: x_km in [-670, 170]).
ZOOM_BANDS = [
    ("hotspot_west", (95, 125)),
    ("hotspot_east", (885, 925)),
    ("mountain", (272, 583)),
]
ZOOM_PAD = 15  # rows of context on each side of a band


def _stitch_raw_image(results: dict, solve: str):
    """Full (1024, N_COLS) raw residual image, stitched from a whole-slit
    sweep's own `results` dict -- the same loop full_detector_figure() uses
    internally, factored out here since that function only returns a
    rendered figure, not the array a zoom crop needs."""
    img = np.full((1024, N_COLS), np.nan)
    for w in results.values():
        key = f"resid_{solve}"
        if key not in w:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        nr = hi - lo + 1
        r = np.asarray(w[key], dtype=float)
        if r.size != nr * N_COLS or hi >= 1024:
            continue
        img[lo:hi + 1] = r.reshape(nr, N_COLS)
    return img


def _zoomed_detector_figure(results: dict, fpa: int, solve: str, band_name: str,
                            row_band: tuple[int, int], cid: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    img = _stitch_raw_image(results, solve)
    lo, hi = row_band
    lo_pad, hi_pad = max(0, lo - ZOOM_PAD), min(1023, hi + ZOOM_PAD)
    crop = img[lo_pad:hi_pad + 1, :]
    if not np.isfinite(crop).any():
        return None

    amax = float(np.nanmax(np.abs(crop))) or 1.0
    lin = float(np.nanmedian(np.abs(crop[np.isfinite(crop)]))) or amax * 1e-3

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(crop, aspect="auto", cmap="RdBu_r", origin="upper",
                   extent=[0, N_COLS, hi_pad + 1, lo_pad],
                   norm=SymLogNorm(linthresh=lin, vmin=-amax, vmax=amax, base=10))
    lo_hi = sorted((int(w["row_lo"]), int(w["row_hi"])) for w in results.values())
    for _, whi in lo_hi[:-1]:
        b = whi + 0.5
        if lo_pad <= b <= hi_pad:
            ax.axhline(b, color="0.3", lw=0.4, alpha=0.5, zorder=3)
    ax.set_xlabel("detector column")
    ax.set_ylabel("detector row")
    ax.set_title(f"{cid} ({solve}) -- {band_name}, rows {lo}-{hi} (+/-{ZOOM_PAD} shown)",
                fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="W/m2/sr/um")
    fig.tight_layout()
    return fig


def _eta_rows_x_km(fpa: int):
    rows = np.arange(float(N_COLS))
    _, s = xy_to_wavelength_slit(fpa, np.full(N_COLS, 512.0), rows)
    eta_rows = s / s_max(fpa)
    return eta_rows, eta_rows * als.SLIT_HALF_KM


def _posterior_error_plot(summary: dict, fpa: int, free_key: str, param: str, plot_dir: Path):
    """retrieved-minus-truth vs. detector row, one line per (g_ratio,
    anchor_density), for every config in `summary` matching `free_key` and
    that actually retrieved `param` -- same Blues-family colouring
    (darker=finer g_ratio) forward_check_residual_by_row_fpa2.png's own
    bingrid lines use, plus linestyle for anchor_density since that axis
    only takes 3 values here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, x_km = _eta_rows_x_km(fpa)
    truth = np.asarray(als.STATE_FIELDS[param](x_km), dtype=float)
    rows = np.arange(N_COLS)

    lines = []  # (g_ratio, adens, err)
    prior_fields_seen = set()
    for (cid, jacobian), sc in summary["scored"].items():
        if jacobian != "analytic":
            continue
        fk, nwin, gratio, adens, si, scene, _, prior_fields = parse_config(cid)
        if fk != free_key or "hires" not in sc:
            continue
        st = sc["hires"].get("_state", {})
        if param not in st:
            continue
        err = np.asarray(st[param], dtype=float) - truth
        lines.append((gratio, adens, err))
        prior_fields_seen.add(prior_fields)

    if not lines:
        print(f"_posterior_error_plot: no {free_key}/{param} configs found, skipping")
        return

    g_ratios = [g for g, _, _ in lines]
    g_lo, g_hi = min(g_ratios), max(g_ratios)
    ls_by_ad = {1: "-", 4: "--", 16: ":"}

    def shade(g):
        t = 0.0 if g_hi == g_lo else (np.log(g) - np.log(g_lo)) / (np.log(g_hi) - np.log(g_lo))
        return plt.cm.Blues(0.85 - 0.55 * (1.0 - t))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for gratio, adens, err in sorted(lines, key=lambda r: (-r[0], r[1])):
        ax.plot(rows, err, color=shade(gratio), ls=ls_by_ad.get(adens, "-"), lw=1.2,
               label=f"g_ratio={gratio:g}, ad{adens}")
    # Prior error (prior - truth), analytic and g_ratio/anchor_density-
    # independent -- als.PRIOR_FIELD_SETS[pf] is the same field function
    # regardless of bin placement, so one reference line per distinct
    # prior_fields value covers every (g_ratio, adens) line above. Skipped
    # entirely under "exact" (prior=truth -- would just draw a flat zero
    # line and add legend clutter for the common case).
    for pf in sorted(prior_fields_seen):
        if pf == "exact" or param not in als.PRIOR_FIELD_SETS[pf]:
            continue
        prior_err = np.asarray(als.PRIOR_FIELD_SETS[pf][param](x_km), dtype=float) - truth
        ax.plot(rows, prior_err, color="0.35", ls=(0, (1, 1)), lw=1.4, zorder=3,
               label=f"prior ({pf})")
    ax.axhline(0.0, color="0.5", lw=0.8, alpha=0.6)
    ax.set_xlabel("detector row")
    unit = "ppm" if param == "co2_ppm" else ("hPa" if param == "p_surface_hpa" else "")
    ax.set_ylabel(f"retrieved - truth ({unit})" if unit else "retrieved - truth")
    ax.set_title(f"FPA{fpa}: posterior error vs. slit position, free={FREE_SETS[free_key]}, "
                f"param={param} (solid=ad1, dashed=ad4, dotted=ad16; darker=finer g_ratio)",
                fontsize=10.5)
    ax.legend(fontsize=6.5, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = plot_dir / f"retrieval_posterior_error_{free_key}_{param}_fpa{fpa}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def _sweep_rms_plot(summary: dict, fpa: int, plot_dir: Path):
    """CO2/p_surface rms+max vs. g_ratio, one line per anchor_density,
    both free-sets overlaid on the CO2 panel (dashed=co2-only,
    solid=co2p) -- the retrieval analogue of
    forward_check_g_ratio_sweep_fpa2.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by = {}  # (free_key, param, metric) -> {adens: [(gratio, val), ...]}
    for (cid, jacobian), sc in summary["scored"].items():
        if jacobian != "analytic" or "hires" not in sc:
            continue
        fk, nwin, gratio, adens, si, scene, _, _ = parse_config(cid)
        r = sc["hires"]
        for param in ("co2_ppm", "p_surface_hpa"):
            for metric in ("rms", "max"):
                key = f"{param}_{metric}"
                if key not in r:
                    continue
                rows_by.setdefault((fk, param, metric), {}).setdefault(adens, []).append(
                    (gratio, r[key]))

    if not rows_by:
        print("_sweep_rms_plot: no scored configs found, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ad_colors = {1: "C0", 4: "C1", 16: "C2"}
    fk_ls = {"co2": "--", "co2p": "-"}
    panels = [("co2_ppm", axes[0], "CO2 (ppm)"), ("p_surface_hpa", axes[1], "p_surface (hPa)")]
    for param, ax, label in panels:
        for fk in ("co2", "co2p"):
            for metric, alpha, mk in (("rms", 1.0, "o"), ("max", 0.5, "s")):
                d = rows_by.get((fk, param, metric))
                if not d:
                    continue
                for adens, pts in sorted(d.items()):
                    pts.sort(key=lambda p: -p[0])
                    gs = [p[0] for p in pts]
                    vs = [p[1] for p in pts]
                    ax.plot(gs, vs, marker=mk, color=ad_colors.get(adens, "k"),
                           ls=fk_ls[fk], alpha=alpha, lw=1.4,
                           label=f"{fk} ad{adens} {metric}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("g_ratio (log scale -- more bins / finer to the right)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} retrieval error vs. g_ratio (solid=co2p, dashed=co2-only)",
                    fontsize=10.5)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"FPA{fpa}: retrieval RMS/max across the bingrid sweep "
                f"(prior=truth at bin scale, real gauss_newton_state solves)", fontsize=12)
    fig.tight_layout()
    out = plot_dir / f"retrieval_sweep_rms_fpa{fpa}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="retrieval_bingrid_v1")
    ap.add_argument("--fpa", type=int, default=2)
    ap.add_argument("--solve", default="hires", choices=["coarse", "hires"])
    ap.add_argument("--configs", type=str, default=None,
                    help="comma list of config ids to plot the 2D (full+zoom) figures for "
                         "(default: every config found under this tag's results/)")
    ap.add_argument("--skip-2d", action="store_true",
                    help="skip the per-config full-detector/zoom figures (30 configs x 4 "
                         "figures each is the slow, I/O-heavy part -- everything else only "
                         "reads summary.pkl)")
    args = ap.parse_args()

    tag_dir = REPO_ROOT / "results" / "config_matrix" / args.tag
    tag_results = tag_dir / "results"
    plot_dir = REPO_ROOT / "plots" / "config_matrix" / args.tag
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_path = tag_results / "summary.pkl"
    if not summary_path.exists():
        raise SystemExit(f"no {summary_path} -- run gd_joint_block_matrix.py --tag {args.tag} "
                         f"first (this script only plots, it doesn't score)")
    with open(summary_path, "rb") as f:
        summary = pickle.load(f)

    for free_key, param in (("co2", "co2_ppm"), ("co2p", "co2_ppm"), ("co2p", "p_surface_hpa")):
        _posterior_error_plot(summary, args.fpa, free_key, param, plot_dir)
    _sweep_rms_plot(summary, args.fpa, plot_dir)

    if args.skip_2d:
        return 0

    pkl_paths = sorted(tag_results.glob("*_analytic.pkl"))
    if args.configs:
        wanted = set(args.configs.split(","))
        pkl_paths = [p for p in pkl_paths if p.stem[:-len("_analytic")] in wanted]

    for p in pkl_paths:
        cid = p.stem[:-len("_analytic")]
        with open(p, "rb") as f:
            d = pickle.load(f)
        results = d["results"]
        fpa = int(d.get("fpa", args.fpa))
        fig, covered = full_detector_figure(results, fpa, args.solve)
        if fig is not None:
            out = plot_dir / f"retrieval_2d_full_{cid}_fpa{fpa}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"saved {out}  ({covered}/1024 rows covered)")
            import matplotlib.pyplot as plt
            plt.close(fig)
        for band_name, row_band in ZOOM_BANDS:
            zfig = _zoomed_detector_figure(results, fpa, args.solve, band_name, row_band, cid)
            if zfig is not None:
                out = plot_dir / f"retrieval_2d_zoom_{band_name}_{cid}_fpa{fpa}.png"
                zfig.savefig(out, dpi=150)
                print(f"saved {out}")
                import matplotlib.pyplot as plt
                plt.close(zfig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
