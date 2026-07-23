#!/usr/bin/env python3
"""Plot raw-vs-rectified comparisons for the gd_rectify_retrieve.py verification test.

Loads results/gd_rectify_retrieve.pkl (produced by gd_rectify_retrieve.py) and
produces a 4-panel figure:

1. Raw detector image A (real-GD-curve render), with the tested rows marked.
2. Rectified spectrogram R (regular slit x wavenumber grid).
3. Row 512: rectified spectrum vs the *true* spectrum at that exact slit
   position (bypassing the raw-detector round trip entirely) -- isolates
   pure rectification/interpolation error from any retrieval or GD-curve
   effect (see KEYSTONE_SMILE_BIAS_PLAN.md Sec. 9l).
4. Retrieved CO2 bias vs row, order=0 vs order=2 dispersion.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_rectify_plot.py
Output: plots/gd_raw_vs_rectified_fpa2.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geosat_geometry as gg
from geocarb_gert import GEOCARB_BANDS, albedo_for, reference_atmosphere, sample_geometries
from geocarb_gert.focalplane import uniform_scene
from geocarb_gert.gd_polynomials import real_wavenumber_range
from geocarb_gert.gd_render import s_max, _diagonal_ils_convolve

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.rt_solver import SingleScatterSolver

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    with open(REPO_ROOT / "results" / "gd_rectify_retrieve.pkl", "rb") as f:
        d = pickle.load(f)
    A, Rimg = d["A_raw"], d["R_rectified"]
    s_grid, wn_grid = d["s_grid"], d["wn_grid"]
    out, test_k, FPA = d["out"], d["test_k"], d["FPA"]

    # -- regenerate the true hi-res spectrum, needed for panel 3 --
    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[FPA]
    wn_min, wn_max = real_wavenumber_range(FPA, margin_cm1=10.0)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    atm = reference_atmosphere()
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=list(mols), label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=300.0)
    fm_wide = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)
    albedo_desert = albedo_for(wide_inst, "desert")
    res0 = fm_wide.run(albedo=list(albedo_desert), albedo_slope=[0.0])
    wn_hires, S_desert = res0.wn_band_hires[0], res0.I_hires[0]
    radiance = uniform_scene(S_desert)

    k0 = 512
    sm = s_max(FPA)
    eta = np.full(len(wn_grid), s_grid[k0] / sm)
    S_row = np.asarray(radiance(eta), dtype=float)
    true_direct = _diagonal_ils_convolve(wn_hires, S_row, wn_grid, wide_win.ils)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    im = ax.imshow(A, aspect="auto", cmap="inferno", origin="lower")
    for k in test_k:
        ax.axhline(k, color="cyan", lw=0.5, alpha=0.6)
    ax.set_title(f"Raw detector image A (FPA{FPA}, real GD curves)")
    ax.set_xlabel("raw column"); ax.set_ylabel("raw row")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(Rimg, aspect="auto", cmap="inferno", origin="lower",
                   extent=[wn_grid[0], wn_grid[-1], s_grid[0], s_grid[-1]])
    for k in test_k:
        ax.axhline(s_grid[k], color="cyan", lw=0.5, alpha=0.6)
    ax.set_title("Rectified spectrogram R(slit, wavenumber)")
    ax.set_xlabel("wavenumber [cm$^{-1}$]"); ax.set_ylabel("slit angle [deg]")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    row = Rimg[k0, :]
    valid = ~np.isnan(row)
    ax.plot(wn_grid[valid], row[valid], label="rectified (raw render -> bilinear regrid)", lw=1.2)
    ax.plot(wn_grid[valid], true_direct[valid], label="true spectrum at same slit position\n(no raw-detector round trip)",
           lw=1.2, ls="--")
    ax.set_title(f"Row {k0}: rectification-interpolation error\n"
                f"(mean |resid| = {np.abs(row[valid]-true_direct[valid]).mean():.3f}, "
                f"signal mean = {true_direct[valid].mean():.2f})")
    ax.set_xlabel("wavenumber [cm$^{-1}$]"); ax.set_ylabel("radiance")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for order, marker in ((0, "o"), (2, "s")):
        ks, biases = [], []
        for k in test_k:
            bias = out[(order, k)]["bias"]
            if (bias is not None and not bias.get("_diverged") and bias.get("_conv")
                    and np.isfinite(bias.get("co2", np.nan))):
                ks.append(k); biases.append(bias["co2"])
        ax.plot(ks, biases, marker=marker, label=f"order={order}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Retrieved CO2 bias vs slit-grid row\n(rectify -> retrieve on shared nominal grid)")
    ax.set_xlabel("s_grid row index"); ax.set_ylabel("CO2 bias [ppm]")
    ax.legend(fontsize=8)

    fig.suptitle(f"gd_rectify_retrieve verification test -- FPA{FPA}, uniform desert scene", fontsize=13)
    fig.tight_layout()
    out_path = REPO_ROOT / "plots" / "gd_raw_vs_rectified_fpa2.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
