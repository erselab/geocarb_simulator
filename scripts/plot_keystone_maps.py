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
shift, EE = {}, {}
for f in range(4):
    _, s = xy_to_wavelength_slit(f, cols.ravel(), rows.ravel())
    eta = eta_of_s(f, s).reshape(N_PX, N_PX)
    # Keystone as the ROW DISPLACEMENT of a scene point relative to where the same slit position lands in the
    # centre column: row - r512, with r512 the row whose centre-column eta equals this pixel's eta. Positive =
    # this column sees the scene point farther out along the slit (larger |row - null|) than the centre column
    # does, i.e. the slit projection GROWS with wavelength in the increasing-wavelength direction (user,
    # 2026-09-25). (The earlier eta shift AT FIXED ROW had the opposite sign.)
    EE[f] = eta
    bt = band_tables(f)
    r512 = np.interp(eta.ravel(), bt["eta_c"], np.arange(N_PX, dtype=float)).reshape(N_PX, N_PX)
    shift[f] = (rows - r512) * bt["spacing"]            # eta units, so the km/rows scaling below is unchanged
    print(f"FPA{f}: keystone shift range {shift[f].min()*als.SLIT_HALF_KM:+.2f} .. {shift[f].max()*als.SLIT_HALF_KM:+.2f} km, "
          f"{shift[f].min()/band_tables(f)['spacing']:+.2f} .. {shift[f].max()/band_tables(f)['spacing']:+.2f} rows", flush=True)

COLORS = ["#111111", "#1b7f3b", "#7b3fb5", "#d98a00", "#c2185b"]
ROWV = np.arange(N_PX, dtype=float)


def sample_paths(f):
    """Scene-point paths across the columns (2026-09-25): start at the centre column on the null row (the row
    whose keystone stays ~0 across all columns) plus rows half-way and all the way toward each edge that has room
    (>=100 rows), where the displacement is largest. Path row at column c = the row whose eta equals the start
    eta, so path_row(c) - start_row is exactly the keystone map's value along the path."""
    null = int(np.argmin(np.abs(shift[f]).max(axis=1)))
    starts = [null]
    for edge in (15, 1008):
        if abs(edge - null) >= 100:
            starts += [int(round(null + 0.5 * (edge - null))), edge]
    starts = sorted(set(starts), key=lambda r: (r != null, r))
    paths = []
    for r0 in starts:
        e0 = EE[f][r0, 512]
        paths.append((r0, np.array([np.interp(e0, EE[f][:, c], ROWV) for c in range(N_PX)])))
    return null, paths


PATHS = {f: sample_paths(f) for f in range(4)}
XC = np.arange(N_PX)

for unit, scale_fn, tag, lab in (("km", lambda f: als.SLIT_HALF_KM, "", "keystone [km]"),
                                 ("rows", lambda f: 1.0 / abs(band_tables(f)["spacing"]), "_rows", "keystone [rows]")):
    fig = plt.figure(figsize=(12, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1.1, 3, 1.1])
    for k, f in enumerate(range(4)):
        ax = fig.add_subplot(gs[2 * (k // 2), k % 2])
        st = fig.add_subplot(gs[2 * (k // 2) + 1, k % 2])
        img = oriented(shift[f] * scale_fn(f), f)
        v = np.abs(img).max()
        im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal", interpolation="nearest")
        lo, hi = lam_range(f)
        asc = wavelength_ascending_cols(f)
        xs = XC if asc else XC[::-1]                     # display column (flipped for FPA1/FPA3)
        null, paths = PATHS[f]
        for j, (r0, pr) in enumerate(paths):
            col = COLORS[j % len(COLORS)]
            name = "null row" if r0 == null else f"row {r0}"
            ax.plot(xs, pr, color="white", lw=3.2, alpha=0.9)
            ax.plot(xs, pr, color=col, lw=1.6)
            ax.text(1015, r0, f"{r0}", color=col, fontsize=7, va="bottom", ha="right",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
            disp = (pr - r0) * (scale_fn(f) * abs(band_tables(f)["spacing"]))
            st.plot(xs, disp, color=col, lw=1.6, label=f"{name} (end-to-end {disp[np.argmax(xs)] - disp[np.argmin(xs)]:+.2g})")
        ax.set_title(f"{NAMES[f]}  ({lo:.3f}-{hi:.3f} um; max |keystone| {v:.3g} {unit})", loc="left", fontsize=10)
        if k % 2 == 0:
            ax.set_ylabel("detector row")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelbottom=False)
        fig.colorbar(im, ax=ax, shrink=0.8, label=lab)
        st.axhline(0, color="#c3c2b7", lw=1)
        st.set_xlim(0, N_PX - 1)
        st.set_xlabel("detector column, wavelength increasing ->" + ("" if asc else " (flipped)"))
        st.set_ylabel(f"path row - start row [{unit}]", fontsize=8)
        st.legend(frameon=False, fontsize=6.5, loc="best", ncol=2)
        st.grid(True, color="#e6e5e0", lw=0.8)
    fig.suptitle(f"Keystone: displacement of a scene point relative to the centre column (signed; + above / - below the null row on the "
                 f"long-wavelength side = slit projection grows with wavelength) [{unit}]\nLines: paths a fixed slit position traces across the "
                 f"columns from the centre column (labelled by start row); strips below show each path's displacement.", fontsize=10)
    out = REPO / f"plots/keystone_maps{tag}_fpa0-3.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.name)
