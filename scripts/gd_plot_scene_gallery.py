#!/usr/bin/env python3
"""One figure per scene: true/prior geophysical-variable profiles (left
column) alongside a native post-fit residual image for each of the four
FPAs individually (right column) -- the "scene overview + detector image
per band" figure used at the top of each experiment section in the
GeoCarb geometric-distortion writeup.

Unlike gd_plot_scene_overview.py (which shows N bands from ONE joint
--fpas run, all three scenes as columns), this shows all 4 bands' own
single-band results for ONE scene, so it works as a band-independent
scene reference regardless of which FPA combination is discussed next.
Reuses gd_plot_scene_overview.py's truth/prior recompute and residual-
image helpers directly rather than duplicating them.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_plot_scene_gallery.py --scene realistic
Output: plots/gd_scene_gallery_<scene>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geocarb_gert import along_slit_scene as als
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit
from geocarb_gert.gd_render import s_max
from gd_plot_scene_overview import (
    SCENE_SUFFIX, GAS_LABEL, _P_SURFACE_PRIOR, _gas_prior,
    _true_and_prior, _residual_image,
)

QUANTITIES = ["co2", "ch4", "co", "h2o", "p_surface"]
FPAS = [0, 1, 2, 3]
FPA_LABEL = {0: "FPA0 — O₂ A-band", 1: "FPA1 — CO₂ weak",
             2: "FPA2 — CO₂ strong", 3: "FPA3 — CH₄/CO"}


def make_figure(scene: str):
    path = REPO_ROOT / "results" / f"gd_joint_fpa0{SCENE_SUFFIX[scene]}.pkl"
    if not path.exists():
        print(f"  MISSING: {path.name}")
        return None
    with open(path, "rb") as f:
        data0 = pickle.load(f)
    rows_dense = np.arange(0, 1024, 2)
    x_km, xtrue = _true_and_prior(scene, 0, rows_dense)
    order_idx = np.argsort(x_km)

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(5, 2, width_ratios=[1, 1.3], hspace=0.55, wspace=0.3)

    for row, q in enumerate(QUANTITIES):
        ax = fig.add_subplot(gs[row, 0])
        prior_val = _P_SURFACE_PRIOR if q == "p_surface" else _gas_prior(q)
        ax.plot(x_km[order_idx], xtrue[q][order_idx], color="tab:red", lw=1.3, label="true")
        ax.axhline(prior_val, color="tab:gray", lw=1.1, ls="--", label="prior")
        ax.set_ylabel("p_surface [hPa]" if q == "p_surface" else GAS_LABEL.get(q, q), fontsize=9)
        ax.tick_params(labelsize=8)
        if row == 0:
            ax.set_title("Truth / prior geophysical profiles", fontsize=11)
            ax.legend(fontsize=7)
        if row == len(QUANTITIES) - 1:
            ax.set_xlabel("along-slit position [km]", fontsize=9)

    sm0 = s_max(0)
    cols = np.full(1024, 512.0)
    _, s_native0 = xy_to_wavelength_slit(0, cols, np.arange(1024.0))
    x_km_native0 = (s_native0 / sm0) * als.SLIT_HALF_KM

    for i, fpa in enumerate(FPAS):
        ax_img = fig.add_subplot(gs[i, 1])
        bpath = REPO_ROOT / "results" / f"gd_joint_fpa{fpa}{SCENE_SUFFIX[scene]}.pkl"
        if not bpath.exists():
            ax_img.text(0.5, 0.5, "no data", ha="center", va="center",
                       transform=ax_img.transAxes, color="gray", fontsize=9)
            ax_img.set_title(FPA_LABEL[fpa], fontsize=10)
            continue
        with open(bpath, "rb") as f:
            bdata = pickle.load(f)
        sm = s_max(fpa)
        _, s_native = xy_to_wavelength_slit(fpa, cols, np.arange(1024.0))
        x_km_native = (s_native / sm) * als.SLIT_HALF_KM
        img, x_sorted = _residual_image(bdata["out"], [fpa], 0, x_km_native)
        if img is None:
            ax_img.text(0.5, 0.5, "no data", ha="center", va="center",
                       transform=ax_img.transAxes, color="gray", fontsize=9)
        else:
            vmax = np.nanpercentile(np.abs(img), 99) or 1e-12
            im = ax_img.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax,
                               extent=[0, img.shape[1], x_sorted.min(), x_sorted.max()])
            fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.03)
        ax_img.set_title(FPA_LABEL[fpa] + " — native residual", fontsize=10)
        ax_img.set_ylabel("x [km]", fontsize=8)
        ax_img.tick_params(labelsize=8)
        if i == len(FPAS) - 1:
            ax_img.set_xlabel("spectral channel", fontsize=9)

    fig.suptitle(f"Scene overview — {scene}", fontsize=14, y=0.995)
    return fig


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", choices=["realistic", "uniform", "barcode"], default="realistic")
    args = ap.parse_args()

    plots_dir = REPO_ROOT / "plots" / "scene_gallery"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure(args.scene)
    if fig is None:
        print("no figure produced (missing data)")
        return 1
    out_path = plots_dir / f"gd_scene_gallery_{args.scene}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
