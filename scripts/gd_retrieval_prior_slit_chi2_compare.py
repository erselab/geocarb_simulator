#!/usr/bin/env python3
"""Prior error, posterior error, and per-window chi^2 vs. detector row, for
g_ratio in {0.5, 1, 3} (anchor_density=1), for the two Phase 2 "first try"
imperfect-prior experiments (docs/PROJECT_STATUS.md Sec.5/6):
    co2plus1pct -- free=co2_ppm,          prior=CO2+1%
    co2p        -- free=co2_ppm,p_surface_hpa, prior=CO2+1%/p_surface-1%

Shows THREE things per param panel: the prior's own error, the corrected
(post-Phase-D, real noise model) posterior error, and the old (pre-Phase-D,
flat-scalar Sy_inv) posterior error -- the last reproduced via
--flat-sy-inv (docs/PROJECT_STATUS.md Sec.6) since the true original run's
raw pkls were overwritten with no way to recover them; this reproduction
is verified bit-identical to the original REPORT.md numbers.

Reads per-config pickles directly --
    results/config_matrix/retrieval_prior_<tag>_v2/results/<cid>_analytic.pkl        (corrected)
    results/config_matrix/retrieval_prior_<tag>_flatsyinv/results/<cid>_analytic.pkl (old, reproduced)
-- "using the figures directly" (the raw per-window results the REPORT.md
summary/convergence-compare plot were themselves built from), not just the
REPORT.md table.

chi^2 is recomputed here (not stored in either pickle) for BOTH runs using
the SAME real noise-model sigma (geocarb_noise_model, Phase D) as the
common yardstick -- chi2 = mean((resid_hires / sigma)**2) per window. This
is a fair basis for the old-vs-new comparison specifically because the old
run's own SOLVE used a different (flat-scalar) weighting, but the sigma
used HERE, after the fact, to judge both fits is identical either way.

Both runs use noise=False (this project's production convention
throughout), so chi2 here is purely (representation error / assumed noise
floor)^2, with no random measurement noise to calibrate against. There is
therefore no statistical reason to expect chi2~=1 (that expectation only
holds when `sigma` is compared against actual noisy-data scatter); a small
chi2 just means representation error is small relative to the noise
floor, the expected outcome on a noiseless scene, not a goodness-of-fit
anomaly. Read this panel as "how big is representation/fit error relative
to the noise floor," not as a real chi2 test.

Run:  PYTHONPATH=. python3 scripts/gd_retrieval_prior_slit_chi2_compare.py
Output: plots/config_matrix/retrieval_prior_slit_chi2_<tag>.png (one per tag)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_test as gdt  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT  # noqa: E402

N_COLS = 1024
G_RATIOS = [0.5, 1, 3]
SHADE_NEW = {0.5: "#08306b", 1: "#2171b5", 3: "#6baed6"}  # blues, darker = finer
SHADE_OLD = {0.5: "#7f2704", 1: "#d94801", 3: "#fd8d3c"}  # oranges, darker = finer
VARIANTS = [("v2", SHADE_NEW, "corrected", "-"), ("flatsyinv", SHADE_OLD, "old (flat Sy_inv)", "--")]

TAGS = {
    "co2plus1pct": dict(prior_fields="co2_plus1pct", params=[("co2_ppm", "ppm")]),
    "co2p": dict(prior_fields="co2_plus1pct_psurf_minus1pct",
                params=[("co2_ppm", "ppm"), ("p_surface_hpa", "hPa")]),
}


def _eta_rows_x_km():
    rows = np.arange(float(N_COLS))
    _, s = xy_to_wavelength_slit(FPA, np.full(N_COLS, 512.0), rows)
    eta_rows = s / s_max(FPA)
    return eta_rows, eta_rows * als.SLIT_HALF_KM


def _stitch_param(windows, param, eta_rows):
    """(true, prior, posterior) arrays over all N_COLS rows, for one param."""
    _, x_km = _eta_rows_x_km()
    true_all = np.asarray(als.PRIOR_FIELD_SETS["exact"][param](x_km), dtype=float)
    prior_all = np.full(N_COLS, np.nan)
    post_all = np.full(N_COLS, np.nan)
    for w in windows:
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        sl = slice(lo, hi + 1)
        rec = w["hires"]["params"][param]
        pos, val, pri = np.asarray(rec["positions"]), np.asarray(rec["values"]), np.asarray(rec["prior"])
        if pos.size > 1:
            post_all[sl] = np.interp(eta_rows[sl], pos, val)
            prior_all[sl] = np.interp(eta_rows[sl], pos, pri)
        else:
            post_all[sl] = val[0]
            prior_all[sl] = pri[0]
    return true_all, prior_all, post_all


def _chi2_per_row(windows, band, eta_rows):
    chi2_all = np.full(N_COLS, np.nan)
    noise_model = geocarb_noise_model(FPA)
    for w in windows:
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        rows_win = np.arange(lo, hi + 1)
        y_true = band["A"][rows_win, :].ravel()
        sigma = noise_model.sigma([y_true], [None])
        resid = np.asarray(w["resid_hires"], dtype=float)
        chi2_pix = (resid / sigma) ** 2
        # one chi2 value per row (mean over that row's 1024 columns), so
        # this overlays on the same row axis as the state-error panels
        chi2_by_row = chi2_pix.reshape(hi - lo + 1, N_COLS).mean(axis=1)
        chi2_all[lo:hi + 1] = chi2_by_row
    return chi2_all


def _load_windows(tag, variant, g, prior_fields):
    gtag = f"{g:g}"
    cid_stem = f"{'co2p' if tag == 'co2p' else 'co2'}-58-g{gtag}-pf-{prior_fields}"
    pkl_path = (REPO_ROOT / "results" / "config_matrix" / f"retrieval_prior_{tag}_{variant}"
               / "results" / f"{cid_stem}_analytic.pkl")
    if not pkl_path.exists():
        print(f"missing {pkl_path}, skipping g_ratio={g} for {tag}/{variant}")
        return None
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return sorted(d["results"].values(), key=lambda r: r["row_lo"])


def _load_band():
    """Renders (or cache-hits) FPA2's realistic-scene truth image at the
    n_lookup_samples the imperfect-prior sweeps used (5600) -- same inputs
    as the original solve, so this is a cache HIT, not a re-render."""
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    return gdt._band_setup_cached(FPA, atm_center, absco, geo, solar, snr, 5600, None,
                                  False, False, 32, False, 0, False)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eta_rows, _ = _eta_rows_x_km()
    rows = np.arange(N_COLS)
    band = _load_band()
    plot_dir = REPO_ROOT / "plots" / "config_matrix"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for tag, spec in TAGS.items():
        params = spec["params"]
        n_rows_fig = len(params) + 1  # one panel per param's error, plus chi2
        fig, axes = plt.subplots(n_rows_fig, 1, figsize=(13, 3.6 * n_rows_fig), sharex=True)
        if n_rows_fig == 1:
            axes = [axes]

        data = {}  # (variant, g) -> dict(params=..., chi2=...)
        for variant, _, _, _ in VARIANTS:
            for g in G_RATIOS:
                windows = _load_windows(tag, variant, g, spec["prior_fields"])
                if windows is None:
                    continue
                data[(variant, g)] = dict(
                    params={p: _stitch_param(windows, p, eta_rows) for p, _ in params},
                    chi2=_chi2_per_row(windows, band, eta_rows),
                )

        for i, (param, unit) in enumerate(params):
            ax = axes[i]
            prior_drawn = False
            for variant, shade, vlabel, ls in VARIANTS:
                for g in G_RATIOS:
                    if (variant, g) not in data:
                        continue
                    true_all, prior_all, post_all = data[(variant, g)]["params"][param]
                    if not prior_drawn:
                        ax.plot(rows, prior_all - true_all, color="0.35", ls=(0, (1, 1)), lw=1.4,
                               label="prior", zorder=3)
                        prior_drawn = True
                    ax.plot(rows, post_all - true_all, color=shade[g], ls=ls, lw=1.3,
                           label=f"{vlabel}, g_ratio={g:g}")
            ax.axhline(0.0, color="0.5", lw=0.6)
            ax.set_ylabel(f"error ({unit})\n[{param}]")
            ax.set_title(f"{tag}: prior/posterior error vs. slit position, {param}", fontsize=10.5)
            ax.legend(fontsize=7, ncol=4, loc="upper right")
            ax.grid(alpha=0.3)

        ax = axes[-1]
        for variant, shade, vlabel, ls in VARIANTS:
            for g in G_RATIOS:
                if (variant, g) not in data:
                    continue
                ax.step(rows, data[(variant, g)]["chi2"], where="mid", color=shade[g], ls=ls,
                       lw=1.2, label=f"{vlabel}, g_ratio={g:g}")
        ax.set_yscale("log")
        ax.set_ylabel("(resid / noise floor)^2\n(mean per row)")
        ax.set_xlabel("detector row")
        ax.set_title(f"{tag}: fit error vs. assumed noise floor, per row -- both curves judged "
                    f"against the SAME real noise-model sigma (noiseless scene -- NOT a "
                    f"measurement-noise chi2 test, no chi2~=1 expectation applies)", fontsize=9)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(alpha=0.3, which="both")

        fig.suptitle(f"Imperfect-prior retrieval ({tag}): old (flat Sy_inv) vs. corrected "
                     f"(real noise model, Phase D), anchor_density=1", fontsize=12)
        fig.tight_layout()
        out = plot_dir / f"retrieval_prior_slit_chi2_{tag}.png"
        fig.savefig(out, dpi=150)
        print(f"saved {out}")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
