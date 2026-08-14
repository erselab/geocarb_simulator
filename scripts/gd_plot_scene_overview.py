#!/usr/bin/env python3
"""One-page overview of the three along-slit truth scenes (realistic
composition variation, uniform, barcode): true and prior gas mole
fraction(s) + surface pressure vs. along-slit position, one column per
scene, for any FPA count. Uses the no-noise .pkl for each scene (noise
doesn't change truth/prior).

Migrated 2026-08-11 from gd_band_stress_test_plot_scene_overview.py's
single-band-only version to gd_test.py's flat, N-band-generic pkl schema.
Two things the old single-band .pkl stored at the top level and the new
one doesn't: `xtrue_of_row`/`x_km_of_row` (recomputed here, deterministic
and cheap, same approach the old script already used for the prior) and
`A_raw`, the raw rendered detector image (not recomputed -- re-rendering
is slow and exactly what gd_plot_residual_spectra.py's own 2026-07-29
migration eliminated).

Instead of the raw image, the bottom row(s) show the post-fit *residual*
image (native pipeline, one row per band in fpas order): each converged
row's saved `_residuals` array stacked by along-slit position, at fixed
channel index (row x channel, RdBu_r, centered at 0, missing/non-converged
rows left blank). This is reconstructable entirely from what gd_test.py
already saves, and arguably more useful for this project's actual
question than the raw signal image was -- it shows directly where the
post-fit residual is large across the whole FPA, including barcode
convergence failures showing up as visible gaps.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_scene_overview.py --fpas 0
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_scene_overview.py --fpas 0,2
Output: plots/gd_joint_<fpas_tag>_scene_overview.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from geocarb_gert import along_slit_scene as als
from geocarb_gert.cross_band import fpas_tag
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit
from geocarb_gert.gd_render import s_max

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES = ("realistic", "uniform", "barcode")
SCENE_SUFFIX = {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}

GAS_UNITS = {"co2": 1e6, "ch4": 1e9, "co": 1e9}
GAS_LABEL = {"co2": "XCO2 [ppm]", "ch4": "XCH4 [ppb]", "co": "XCO [ppb]", "h2o": "h2o (profile-mean) [ppm]"}

_ATM_CENTER = als.atmosphere_at(0.0)
_P_SURFACE_PRIOR = float(als.p_surface_hpa(np.array([0.0]))[0])
_H2O_MEAN_PRIOR_PPM = float(np.mean(_ATM_CENTER.gases["h2o"])) * 1e6
_SPECIAL_KEYS = {"_state", "_state_full", "_chi2", "_conv", "_diverged", "_residuals", "_nus"}


def _gas_prior(gas: str) -> float:
    if gas == "h2o":
        return _H2O_MEAN_PRIOR_PPM
    return float(np.mean(_ATM_CENTER.gases[gas])) * GAS_UNITS[gas]


def _result_path(fpas, scene: str) -> Path:
    return REPO_ROOT / "results" / f"gd_joint_{fpas_tag(fpas)}{SCENE_SUFFIX[scene]}.pkl"


def _x_km_of_row(fpa: int, rows) -> np.ndarray:
    cols = np.full(len(rows), 512.0)
    _, s_of_row = xy_to_wavelength_slit(fpa, cols, np.asarray(rows, dtype=float))
    return (s_of_row / s_max(fpa)) * als.SLIT_HALF_KM


def _true_and_prior(scene: str, fpa_ref: int, rows: np.ndarray):
    """Deterministic recompute of xtrue_of_row (as gd_test.py's own
    _band_setup computes it) -- uniform/barcode force flat truth at the
    slit-centre prior; only realistic varies along the slit."""
    x_km = _x_km_of_row(fpa_ref, rows)
    xtrue_x_km = np.zeros(len(rows)) if scene in ("uniform", "barcode") else x_km
    h2o_ratio = als.h2o_surface_vmr(xtrue_x_km) / als.h2o_surface_vmr(0.0)
    xtrue = {
        "co2": als.xco2_ppm(xtrue_x_km), "ch4": als.xch4_ppb(xtrue_x_km),
        "co": als.xco_ppb(xtrue_x_km), "h2o": _H2O_MEAN_PRIOR_PPM * h2o_ratio,
        "p_surface": als.p_surface_hpa(xtrue_x_km),
    }
    return x_km, xtrue


def _infer_gases(out: dict) -> list:
    for (pl, o, r), v in out.items():
        if pl != "native" or v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        keys = [k for k in v if k not in _SPECIAL_KEYS and k != "p_surface"]
        if keys:
            return sorted(keys)
    return []


def _residual_image(out, fpas, band_idx: int, x_km_native: np.ndarray):
    """(n_rows_total, n_channels) array, native pipeline, one band -- NaN
    for rows that didn't converge or weren't tested, so gaps (e.g. barcode
    edge failures) show as blank stripes rather than silently compressing
    the row axis."""
    row_tuples = sorted(set(k[2] for k in out if k[0] == "native"))
    if not row_tuples:
        return None, None
    n_ch = None
    rows0 = [rt[0] for rt in row_tuples]
    img = np.full((len(rows0), 1), np.nan)
    for i, rt in enumerate(row_tuples):
        v = out.get(("native", 2, rt))
        if v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        resids = v.get("_residuals")
        if not resids or band_idx >= len(resids):
            continue
        r = np.asarray(resids[band_idx])
        if n_ch is None:
            n_ch = len(r)
            img = np.full((len(rows0), n_ch), np.nan)
        img[i, :len(r)] = r
    if n_ch is None:
        return None, None
    order_idx = np.argsort(x_km_native[rows0])
    return img[order_idx], x_km_native[rows0][order_idx]


def make_figure(fpas) -> "plt.Figure | None":
    data = {}
    for scene in SCENES:
        path = _result_path(fpas, scene)
        if not path.exists():
            print(f"  MISSING: {path.name}")
            return None
        with open(path, "rb") as f:
            data[scene] = pickle.load(f)

    gases = _infer_gases(data["realistic"]["out"])
    quantities = list(gases) + ["p_surface"]
    n_bands = len(fpas)

    sm = s_max(fpas[0])
    cols = np.full(1024, 512.0)
    _, s_native = xy_to_wavelength_slit(fpas[0], cols, np.arange(1024.0))
    x_km_native_full = (s_native / sm) * als.SLIT_HALF_KM

    nrows = len(quantities) + n_bands
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)

    for col, scene in enumerate(SCENES):
        out = data[scene]["out"]
        row_tuples = sorted(set(k[2] for k in out if k[0] == "native"))
        rows0 = np.array([rt[0] for rt in row_tuples])
        x_km, xtrue = _true_and_prior(scene, fpas[0], rows0)
        order_idx = np.argsort(x_km)

        for row, q in enumerate(quantities):
            ax = axes[row, col]
            true_vals = xtrue[q][order_idx]
            prior_val = _P_SURFACE_PRIOR if q == "p_surface" else _gas_prior(q)
            ax.plot(x_km[order_idx], true_vals, color="tab:red", lw=1.2, label="true")
            ax.axhline(prior_val, color="tab:gray", lw=1.2, ls="--", label="prior")
            if row == 0:
                ax.set_title(f"{scene}", fontsize=11)
            ylabel = "p_surface [hPa]" if q == "p_surface" else GAS_LABEL.get(q, q)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)

        for b in range(n_bands):
            ax_img = axes[len(quantities) + b, col]
            img, x_sorted = _residual_image(out, fpas, b, x_km_native_full)
            if img is None:
                ax_img.text(0.5, 0.5, "no data", ha="center", va="center",
                           transform=ax_img.transAxes, color="gray", fontsize=8)
            else:
                vmax = np.nanpercentile(np.abs(img), 99) or 1e-12
                im = ax_img.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r",
                                   vmin=-vmax, vmax=vmax,
                                   extent=[0, img.shape[1], x_sorted.min(), x_sorted.max()])
                fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
            ax_img.set_xlabel("spectral channel", fontsize=8)
            if col == 0:
                ax_img.set_ylabel(f"x [km]\n(FPA{fpas[b]} residual)", fontsize=8)
            ax_img.tick_params(labelsize=7)

    fig.suptitle(f"{'+'.join(f'FPA{f}' for f in fpas)} -- true/prior truth profiles "
                f"and native post-fit residual image, by scene", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpas", type=str, default="0,2",
                    help="comma-separated FPA indices, e.g. 0 for a single-band case or 0,2 for joint")
    args = ap.parse_args()
    fpas = [int(x) for x in args.fpas.split(",")]

    tag = fpas_tag(fpas)
    plots_dir = REPO_ROOT / "plots" / "band_stress_test" / tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'+'.join(f'FPA{f}' for f in fpas)} ...", flush=True)
    fig = make_figure(fpas)
    if fig is None:
        print("no figure produced (missing data)")
        return 1
    png_path = plots_dir / f"gd_joint_{tag}_scene_overview.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"saved {png_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
