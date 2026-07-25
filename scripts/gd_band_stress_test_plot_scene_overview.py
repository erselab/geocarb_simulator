#!/usr/bin/env python3
"""One-page-per-FPA overview of the three along-slit truth scenes
(realistic composition variation, uniform, barcode -- 2026-07-25): true
and prior gas mole fraction(s) + surface pressure vs. along-slit position,
and the corresponding rendered FPA image, one column per scene. Uses the
no-noise .pkl for each scene (noise doesn't change truth/prior, and would
only add visual grain to the FPA image panel without changing what this
figure is meant to show).

True values come from each case's saved xtrue_of_row/x_km_of_row. Prior
values are recomputed directly from geocarb_gert.along_slit_scene (the
slit-centre prior atmosphere, atmosphere_at(0.0)) rather than stored in the
.pkl -- deterministic and cheap, no rerun needed. Note that uniform/barcode
scenes have FLAT true profiles by design (xtrue_x_km pinned to 0 in
gd_band_stress_test.py) -- their true and prior lines coincide exactly;
only "realistic" shows real along-slit variation. That's the expected,
correct picture, not a bug.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_band_stress_test_plot_scene_overview.py
Output: plots/gd_band_stress_test_fpa<N>_scene_overview.png (4x)
        plots/gd_band_stress_test_all_scene_overview.pdf (4 pages, one per FPA)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

from geocarb_gert import along_slit_scene as als

REPO_ROOT = Path(__file__).resolve().parent.parent
FPAS = (0, 1, 2, 3)
SCENES = ("realistic", "uniform", "barcode")
SCENE_SUFFIX = {"realistic": "", "uniform": "_uniform", "barcode": "_barcode"}

GAS_UNITS = {"co2": 1e6, "ch4": 1e9, "co": 1e9}
GAS_LABEL = {"co2": "XCO2 [ppm]", "ch4": "XCH4 [ppb]", "co": "XCO [ppb]", "h2o": "h2o (profile-mean) [ppm]"}

# Slit-centre prior atmosphere -- same for every FPA/scene (the fixed
# forward-model prior gd_band_stress_test.py's _retrieve() always uses).
_ATM_CENTER = als.atmosphere_at(0.0)
_P_SURFACE_PRIOR = float(als.p_surface_hpa(np.array([0.0]))[0])
_H2O_MEAN_PRIOR_PPM = float(np.mean(_ATM_CENTER.gases["h2o"])) * 1e6


def _gas_prior(gas: str) -> float:
    if gas == "h2o":
        return _H2O_MEAN_PRIOR_PPM
    return float(np.mean(_ATM_CENTER.gases[gas])) * GAS_UNITS[gas]


def _result_path(fpa: int, scene: str) -> Path:
    return REPO_ROOT / "results" / f"gd_band_stress_test_fpa{fpa}{SCENE_SUFFIX[scene]}.pkl"


def make_fpa_figure(fpa: int):
    data = {}
    for scene in SCENES:
        path = _result_path(fpa, scene)
        if not path.exists():
            print(f"  MISSING: {path.name}")
            return None
        with open(path, "rb") as f:
            data[scene] = pickle.load(f)

    gases = data["realistic"]["gases"]   # same set for every scene of a given FPA
    quantities = list(gases) + ["p_surface"]   # e.g. ['co2','h2o','p_surface'] or ['ch4','co','h2o','p_surface']

    nrows = len(quantities) + 1   # + FPA image row
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)

    for col, scene in enumerate(SCENES):
        d = data[scene]
        x_km = d["x_km_of_row"]
        order_idx = np.argsort(x_km)
        xtrue = d["xtrue_of_row"]

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

        ax_img = axes[len(quantities), col]
        A = d["A_raw"]
        im = ax_img.imshow(A, aspect="auto", origin="lower", cmap="viridis")
        ax_img.set_xlabel("spectral channel", fontsize=8)
        if col == 0:
            ax_img.set_ylabel("slit row", fontsize=8)
        ax_img.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    for col in range(ncols):
        axes[len(quantities) - 1, col].set_xlabel("x [km]", fontsize=8)

    fig.suptitle(f"FPA{fpa} -- true/prior truth profiles and FPA image, by scene", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main() -> int:
    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    pdf_path = plots_dir / "gd_band_stress_test_all_scene_overview.pdf"
    n_saved = 0
    with PdfPages(pdf_path) as pdf:
        for fpa in FPAS:
            print(f"FPA{fpa} ...", flush=True)
            fig = make_fpa_figure(fpa)
            if fig is None:
                continue
            png_path = plots_dir / f"gd_band_stress_test_fpa{fpa}_scene_overview.png"
            fig.savefig(png_path, dpi=120)
            pdf.savefig(fig)
            plt.close(fig)
            n_saved += 1
            print(f"  saved {png_path.name}", flush=True)
    print(f"saved combined PDF ({n_saved} pages): {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
