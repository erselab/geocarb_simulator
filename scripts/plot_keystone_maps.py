"""Keystone maps (2026-09-25): for each FPA, the row displacement of a scene point at each detector pixel
relative to where the same slit position lands in the centre column (col 512), from the real GD mapping
polynomials (geocarb_gert.gd_polynomials.xy_to_wavelength_slit, eta via eta_of_s). Signed: the slit projection grows with wavelength when the displacement is positive above the null row and
negative below it on the long-wavelength side (the antisymmetric pattern all four FPAs show). Shown in km along the slit (eta * SLIT_HALF_KM) and in detector rows.
Columns are oriented so wavelength increases left->right in every panel (FPA1/FPA3 are flipped).
Output: plots/keystone_maps_fpa0-3.png (km) and plots/keystone_maps_rows_fpa0-3.png (rows).
    PYTHONPATH=.:<gert> python scripts/plot_keystone_maps.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import N_PX, eta_of_s, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.multiband_geometry import band_tables  # noqa: E402

from geocarb_gert.gd_polynomials import xy_to_wavelength_slit as _xy  # noqa: E402


def wavelength_ascending_cols(f):
    """True if wavelength increases with detector column for FPA f (FPA0, FPA2); False for FPA1, FPA3, whose
    dispersion is reversed. Images are flipped along columns for the latter so every panel reads short
    wavelength (left) -> long wavelength (right) and the four FPAs are directly comparable (2026-09-25)."""
    lam, _ = _xy(f, np.array([0.0, 1023.0]), np.array([512.0, 512.0]))
    return bool(lam[1] > lam[0])


def oriented(img, f):
    return img if wavelength_ascending_cols(f) else img[:, ::-1]


def lam_range(f):
    lam, _ = _xy(f, np.array([0.0, 1023.0]), np.array([512.0, 512.0]))
    return min(lam), max(lam)


NAMES = {0: "FPA0 O2-A", 1: "FPA1 CO2 weak", 2: "FPA2 CO2 strong", 3: "FPA3 CH4/CO"}
cols, rows = np.meshgrid(np.arange(N_PX, dtype=float), np.arange(N_PX, dtype=float))   # [row, col]
shift = {}
for f in range(4):
    _, s = xy_to_wavelength_slit(f, cols.ravel(), rows.ravel())
    eta = eta_of_s(f, s).reshape(N_PX, N_PX)
    # Keystone as the ROW DISPLACEMENT of a scene point relative to where the same slit position lands in the
    # centre column: row - r512, with r512 the row whose centre-column eta equals this pixel's eta. Positive =
    # this column sees the scene point farther out along the slit (larger |row - null|) than the centre column
    # does, i.e. the slit projection GROWS with wavelength in the increasing-wavelength direction (user,
    # 2026-09-25). (The earlier eta shift AT FIXED ROW had the opposite sign.)
    bt = band_tables(f)
    r512 = np.interp(eta.ravel(), bt["eta_c"], np.arange(N_PX, dtype=float)).reshape(N_PX, N_PX)
    shift[f] = (rows - r512) * bt["spacing"]            # eta units, so the km/rows scaling below is unchanged
    print(f"FPA{f}: keystone shift range {shift[f].min()*als.SLIT_HALF_KM:+.2f} .. {shift[f].max()*als.SLIT_HALF_KM:+.2f} km, "
          f"{shift[f].min()/band_tables(f)['spacing']:+.2f} .. {shift[f].max()/band_tables(f)['spacing']:+.2f} rows", flush=True)

for unit, scale_fn, tag, lab in (("km", lambda f: als.SLIT_HALF_KM, "", "keystone: scene-point displacement vs centre column [km]"),
                                 ("rows", lambda f: 1.0 / abs(band_tables(f)["spacing"]), "_rows", "keystone: scene-point displacement vs centre column [rows]")):
    fig, axs = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for a, f in zip(axs.ravel(), range(4)):
        img = oriented(shift[f] * scale_fn(f), f)
        v = np.abs(img).max()
        im = a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal", interpolation="nearest")
        lo, hi = lam_range(f)
        a.set_title(f"{NAMES[f]}  ({lo:.3f}-{hi:.3f} um; max |shift| {v:.3g} {unit})", loc="left", fontsize=10)
        a.set_xlabel("detector column, wavelength increasing ->" + ("" if wavelength_ascending_cols(f) else " (flipped)"))
        a.set_ylabel("detector row")
        fig.colorbar(im, ax=a, shrink=0.8, label=lab)
    fig.suptitle(f"Keystone: displacement of a scene point relative to the centre column (signed rows; + above / - below the null row on the long-wavelength side = slit projection grows with wavelength) ({unit})", fontsize=11)
    out = REPO / f"plots/keystone_maps{tag}_fpa0-3.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.name)
